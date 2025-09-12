import numpy as np
import pandas as pd
import scipy.optimize as opt
from datetime import datetime
import matplotlib.pyplot as plt
import os
from dffc.holt_winters._holt_winters import HW
from concurrent.futures import ProcessPoolExecutor, as_completed
import numba
from numba import jit, prange
from tqdm import tqdm

# 添加均线窗口大小设置
MOVING_AVERAGE_WINDOW = 30

@jit(nopython=True, parallel=True, cache=True)
def sliding_average(arr, window):
    """
    CPU优化版本的滑动平均计算，使用Numba JIT编译和并行化
    """
    n = arr.shape[0]
    half = window // 2
    result = np.empty_like(arr, dtype=np.float64)
    
    if arr.ndim == 1:
        for i in prange(n):
            start = max(0, i - half)
            end = min(n, i + half + 1)
            result[i] = np.mean(arr[start:end])
    else:
        for i in prange(n):
            start = max(0, i - half)
            end = min(n, i + half + 1)
            result[i] = np.mean(arr[start:end], axis=0)
    return result

def holtwinters_rolling(arr, alpha, beta, gamma, season_length):
    hw = HW.run(arr, alpha=alpha, beta=beta, gamma=gamma, m=season_length, multiplicative=False)
    return hw.hw

@jit(nopython=True, cache=True)
def calc_scaling_factor(fluc_A, fluc_B):
    """
    计算最优缩放因子 a，使得 a * fluc_B 最接近 fluc_A，采用最小二乘法求解。
    使用Numba JIT编译加速
    """
    numerator = np.dot(fluc_A, fluc_B)
    denominator = np.dot(fluc_B, fluc_B)
    return numerator / denominator

@jit(nopython=True, cache=True)
def calc_RSS(fluc_A, fluc_B, scaling_factor):
    """
    根据缩放因子 scaling_factor，计算两个子序列之间的残差平方和（RSS）。
    使用Numba JIT编译加速
    """
    diff = fluc_A - scaling_factor * fluc_B
    return np.sum(diff * diff)

def _optimize_single_season_worker(args):
    """
    独立的工作函数，用于多进程优化单个季节长度的参数
    必须在模块级别定义以支持pickle序列化
    """
    season, original_data, holtwinters_begindate, holtwinters_enddate, fluc_sub = args
    
    # 确保数据类型为numpy数组，避免Numba类型问题
    original_data = np.asarray(original_data, dtype=np.float64)
    fluc_sub = np.asarray(fluc_sub, dtype=np.float64)
    
    initial_guess = [0.05, 0.01, 0.2]
    bounds = [(0.0, 1.0), (0.0, 1.0), (0.0, 1.0)]
    
    options = {
        'ftol': 1e-9,
        'gtol': 1e-6,
        'maxiter': 10000,
        'maxfun': 10000,
        'disp': False
    }

    def local_objective(params):
        alpha, beta, gamma = params
        try:
            smoothed = holtwinters_rolling(original_data, alpha, beta, gamma, season_length=season)
            holtwinters_fluc = original_data - smoothed
            holtwinters_fluc_sub = holtwinters_fluc[holtwinters_begindate:holtwinters_enddate]
            a = calc_scaling_factor(fluc_sub, holtwinters_fluc_sub)
            rss = calc_RSS(fluc_sub, holtwinters_fluc_sub, a)
            return rss
        except Exception as e:
            # 如果计算失败，返回一个很大的值
            print(f"Error in local_objective for season {season} with params {params}: {e}")
            return 1e10

    try:
        res = opt.minimize(local_objective,
                           initial_guess,
                           bounds=bounds,
                           method='L-BFGS-B',
                           options=options)
        return season, res.x, res.fun
    except Exception as e:
        print(f"Season {season} optimization failed: {e}")
        return season, initial_guess, np.inf

def optimize_holtwinters_parameters_parallel(original_data, holtwinters_begindate, holtwinters_enddate, n_jobs=-1):
    """
    并行优化版本：对不同季节长度使用多进程并行优化
    """
    # 预计算波动数据，避免重复计算
    mean_data = sliding_average(original_data, MOVING_AVERAGE_WINDOW)
    fluc_data = original_data - mean_data
    fluc_sub = fluc_data[holtwinters_begindate:holtwinters_enddate]
    
    # 准备参数
    seasons = list(range(7, 25))
    args_list = [(season, original_data, holtwinters_begindate, holtwinters_enddate, fluc_sub) 
                 for season in seasons]
    
    if n_jobs == -1:
        n_jobs = min(len(seasons), os.cpu_count())
    
    best_rss = np.inf
    best_params = None
    best_season = None
    
    if n_jobs == 1:
        # 单线程处理
        for args in args_list:
            season, params, rss = _optimize_single_season_worker(args)
            if rss < best_rss:
                best_rss = rss
                best_params = params
                best_season = season
    else:
        # 多进程处理
        with ProcessPoolExecutor(max_workers=n_jobs) as executor:
            # 提交所有任务
            futures = [executor.submit(_optimize_single_season_worker, args) for args in args_list]
            
            # 收集结果
            for future in as_completed(futures):
                try:
                    season, params, rss = future.result()
                    if rss < best_rss:
                        best_rss = rss
                        best_params = params
                        best_season = season
                except Exception as exc:
                    print(f'Future generated an exception: {exc}')

    return best_params, best_season, best_rss

def optimize_holtwinters_parameters(original_data, holtwinters_begindate, holtwinters_enddate):
    """
    单线程版本的参数优化，保持向后兼容性
    """
    return optimize_holtwinters_parameters_parallel(original_data, holtwinters_begindate, holtwinters_enddate, n_jobs=1)

def compute_optimize_result(end_day, original_data):
    """辅助函数，用于并行计算 - 使用并行优化版本"""
    best_params, best_season, best_rss = optimize_holtwinters_parameters_parallel(original_data, -800, end_day)
    return {
        "end_day": end_day,
        "alpha": best_params[0],
        "beta": best_params[1],
        "gamma": best_params[2],
        "season": best_season,
        "rss": best_rss
    }

def batch_optimize_holtwinters(original_data, end_days, n_jobs=-1):
    """
    批量优化多个时间点的Holt-Winters参数
    
    Args:
        original_data: 原始时间序列数据
        end_days: 结束时间点列表
        n_jobs: 并行处理的进程数，-1表示使用所有CPU核心
    
    Returns:
        list: 包含每个时间点优化结果的字典列表
    """
    if n_jobs == -1:
        n_jobs = min(len(end_days), os.cpu_count())
    
    results = []
    
    with ProcessPoolExecutor(max_workers=n_jobs) as executor:
        # 提交所有任务
        future_to_end_day = {
            executor.submit(compute_optimize_result, end_day, original_data): end_day 
            for end_day in end_days
        }
        
        # 收集结果
        for future in tqdm(as_completed(future_to_end_day), total=len(future_to_end_day), desc="Batch optimizing"):
            try:
                result = future.result()
                results.append(result)
                print(f"end_day={result['end_day']}, season={result['season']}, rss={result['rss']:.6f}")
            except Exception as exc:
                end_day = future_to_end_day[future]
                print(f'End day {end_day} generated an exception: {exc}')
    
    # 按end_day排序
    results.sort(key=lambda x: x['end_day'])
    return results


def process_single_hw_opt(price, output_base_dir, max_workers=-1):
    """
    处理单个基金的优化过程
    
    参数:
        fundcode: 基金代码
        output_base_dir: 输出根目录
        max_workers: 并行线程数
    """
    try:
        # fund_output_dir = os.path.join(output_base_dir, fundcode)
        # os.makedirs(fund_output_dir, exist_ok=True)
    
        # 处理数据
        original_data = price
        mean_data = sliding_average(original_data, MOVING_AVERAGE_WINDOW)
        
        # 并行优化
        end_days = list(range(-400, 0, 40))
        end_days.append(None)  # 包含完整数据的优化

        results = batch_optimize_holtwinters(original_data, end_days, n_jobs=max_workers)
        
        # 保存结果
        results_df = pd.DataFrame(results)
        results_df = results_df.sort_values('end_day')
        
        # 保存优化结果CSV
        # results_csv_path = os.path.join(fund_output_dir, f"holtwinters_results_{fundcode}_{MOVING_AVERAGE_WINDOW}.csv")
        # results_df.to_csv(results_csv_path, index=False)
        
        # 绘制并保存图形
        # plt.figure(figsize=(15, 10))
        
        # # 第一张图：原始数据、滑动平均和HoltWinter平滑结果
        # plt.subplot(2, 1, 1)
        # plt.plot(original_data, label='Original Data', marker='o', linestyle='-', markersize=1)
        # plt.plot(mean_data, label='Sliding Average', marker='x', linestyle='--', markersize=1)
        
        # # 使用最后一个优化结果
        last_result = results_df.iloc[-1]
        # holtwinter_smoothed = holtwinters_rolling(original_data, last_result['alpha'], last_result['beta'], 
        #                                          last_result['gamma'], int(last_result['season']))
        # plt.plot(holtwinter_smoothed, label='HoltWinter', linewidth=2)
        
        # plt.title(f'Data Comparison - Fund {fundcode}')
        # plt.xlabel('Index')
        # plt.ylabel('Value')
        # plt.legend()
        # plt.grid(True)
        
        # # 第二张图：参数变化
        # plt.subplot(2, 1, 2)
        # ax1 = plt.gca()
        # ax2 = ax1.twinx()
        
        # # 左y轴: alpha, beta, gamma
        # line1 = ax1.plot(results_df['end_day'], results_df['alpha'], 'b-', marker='o', label='Alpha', markersize=4)
        # line2 = ax1.plot(results_df['end_day'], results_df['beta'], 'g-', marker='s', label='Beta', markersize=4)
        # line3 = ax1.plot(results_df['end_day'], results_df['gamma'], 'r-', marker='^', label='Gamma', markersize=4)
        # ax1.set_xlabel('End Day')
        # ax1.set_ylabel('Alpha, Beta, Gamma', color='black')
        # ax1.tick_params(axis='y', labelcolor='black')
        # ax1.grid(True, alpha=0.3)
        
        # # 右y轴: season
        # line4 = ax2.plot(results_df['end_day'], results_df['season'], 'm-', marker='D', label='Season', markersize=4)
        # ax2.set_ylabel('Season Length', color='m')
        # ax2.tick_params(axis='y', labelcolor='m')
        
        # # 合并图例
        # lines = line1 + line2 + line3 + line4
        # labels = [l.get_label() for l in lines]
        # ax1.legend(lines, labels, loc='upper left')
        
        # plt.title(f'Parameters vs End Days - Fund {fundcode}')
        # plt.tight_layout()
        
        # # 保存图形
        # plot_path = os.path.join(fund_output_dir, f"holtwinters_plot_{fundcode}_{MOVING_AVERAGE_WINDOW}.png")
        # plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        # plt.close()  # 关闭图形以释放内存
        
        # 输出最终参数信息
        print(f"  最终参数: Alpha={last_result['alpha']:.6f}, Beta={last_result['beta']:.6f}, Gamma={last_result['gamma']:.6f}")
        print(f"  Season={last_result['season']}, RSS={last_result['rss']:.6f}")
        
        return {
            # 'fundcode': fundcode,
            'status': 'success',
            'final_params': {
                'alpha': last_result['alpha'],
                'beta': last_result['beta'],
                'gamma': last_result['gamma'],
                'season': last_result['season'],
                'rss': last_result['rss']
            }
        }
        
    except Exception as e:
        # print(f"  基金 {fundcode} 处理失败: {str(e)}")
        return {
            # 'fundcode': fundcode,
            'status': 'failed',
            'error': str(e)
        }



def process_hw_opt(prices, output_base_dir="./optimize_results", max_workers=-1):
    """
    批量处理基金列表
    
    参数:
        fund_codes: 基金代码列表
        output_base_dir: 输出根目录
        max_workers: 并行线程数
    """
    
    results = []
    for col in range(prices.shape[1]):
        results.append(process_single_hw_opt(prices[:, col], output_base_dir, max_workers))
    return results

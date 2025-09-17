import numpy as np
import pandas as pd
import scipy.optimize as opt
from datetime import datetime
import matplotlib.pyplot as plt
import os

from dffc.holt_winters._holt_winters import HW
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm

# 添加均线窗口大小设置
MOVING_AVERAGE_WINDOW = 30

def sliding_average(arr, window):
    """
    计算一个numpy数组或pandas Series的滑动平均，窗口大小自定义，
    两侧不足部分使用较小窗口平均补全，输出形状与原数据相同，保持原始索引。
    """
    if isinstance(arr, pd.Series):
        # 对于pandas Series，保持索引
        n = len(arr)
        half = window // 2
        result = pd.Series(index=arr.index, dtype=float)
        
        for i in range(n):
            start = max(0, i - half)
            end = min(n, i + half + 1)
            result.iloc[i] = arr.iloc[start:end].mean()
        return result
    else:
        # 对于numpy数组，保持原有逻辑
        arr = np.asarray(arr, dtype=float)
        n = arr.shape[0]
        half = window // 2
        result = np.empty_like(arr, dtype=float)
        
        if arr.ndim == 1:
            for i in range(n):
                start = max(0, i - half)
                end = min(n, i + half + 1)
                result[i] = arr[start:end].mean()
        else:
            for i in range(n):
                start = max(0, i - half)
                end = min(n, i + half + 1)
                result[i] = arr[start:end].mean(axis=0)
        return result

def holtwinters_rolling(arr, alpha, beta, gamma, season_length):
    """
    对numpy数组进行三参数Holt-Winters滚动平滑，
    第t个点的平滑结果只使用原数组的前t个数据。
    """
    hw = HW.run(arr, alpha, beta, gamma, season_length, multiplicative=False)
    return hw.hw

def calc_scaling_factor(fluc_A, fluc_B):
    """
    计算最优缩放因子 a，使得 a * fluc_B 最接近 fluc_A，采用最小二乘法求解。
    """
    a = np.dot(fluc_A, fluc_B) / np.dot(fluc_B, fluc_B)
    return a

def calc_RSS(fluc_A, fluc_B, scaling_factor):
    """
    根据缩放因子 scaling_factor，计算两个子序列之间的残差平方和（RSS）。
    """
    rss = np.sum((fluc_A - scaling_factor * fluc_B) ** 2)
    return rss

def optimize_single_season(season, original_data, fluc_data, holtwinters_begindate, holtwinters_enddate, options, bounds):
    """
    单个季节长度的优化函数，用于并行计算
    """
    # 改进的初始猜测 - 基于经验的更好起始点
    if season <= 7:
        # 短周期：较高的alpha(跟踪快)，较低的beta和gamma
        initial_guess = [0.3, 0.05, 0.1]
    elif season <= 15:
        # 中周期：中等的alpha，适中的beta和gamma
        initial_guess = [0.2, 0.1, 0.3]
    else:
        # 长周期：较低的alpha(平滑)，较高的beta和gamma
        initial_guess = [0.1, 0.15, 0.4]

    def local_objective(params):
        alpha, beta, gamma = params
        try:
            smoothed = holtwinters_rolling(original_data, alpha, beta, gamma, season_length=season)
            holtwinters_fluc = original_data - smoothed
            fluc_sub = fluc_data[holtwinters_begindate:holtwinters_enddate]
            holtwinters_fluc_sub = holtwinters_fluc[holtwinters_begindate:holtwinters_enddate]
            a = calc_scaling_factor(fluc_sub, holtwinters_fluc_sub)
            rss = calc_RSS(fluc_sub, holtwinters_fluc_sub, a)
            return rss
        except:
            # 如果计算失败，返回一个很大的值
            return 1e10

    res = opt.minimize(local_objective,
                       initial_guess,
                       bounds=bounds,
                       method='L-BFGS-B',
                       options=options)
    
    return {
        'season': season,
        'success': res.success,
        'fun': res.fun,
        'x': res.x
    }

def optimize_holtwinters_parameters(original_data, holtwinters_begindate, holtwinters_enddate, fundcode=""):
    """
    对给定数据区间进行参数优化，返回最优参数和最优季节长度。
    """
    # 在函数内计算波动数据，使用设定的均线窗口
    mean_data = sliding_average(original_data, MOVING_AVERAGE_WINDOW)
    fluc_data = original_data - mean_data

    options = {
        'ftol': 1e-9,
        'gtol': 1e-6,
        'maxiter': 10000,
        'maxfun': 10000,
        'disp': False  # 关闭详细输出
    }

    bounds = [(0.0, 1.0), (0.0, 1.0), (0.0, 1.0)]
    seasons = list(range(7, 25))
    
    # 并行优化所有季节长度
    
    with ProcessPoolExecutor(max_workers=min(len(seasons), 8)) as executor:
        futures = {
            executor.submit(optimize_single_season, season, original_data, fluc_data, 
                          holtwinters_begindate, holtwinters_enddate, options, bounds): season
            for season in seasons
        }
        
        results = []
        for future in tqdm(as_completed(futures), total=len(futures), desc=f"Optimizing seasons for fund {fundcode}"):
            result = future.result()
            if result['success']:
                results.append(result)
    
    # 找到最佳结果
    if not results:
        # 如果所有优化都失败了，回退到串行方式
        return _optimize_holtwinters_parameters_serial(original_data, holtwinters_begindate, holtwinters_enddate)
    
    best_result = min(results, key=lambda x: x['fun'])
    return best_result['x'], best_result['season'], best_result['fun']

def _optimize_holtwinters_parameters_serial(original_data, holtwinters_begindate, holtwinters_enddate):
    """
    串行版本的优化函数，作为并行版本失败时的备选
    """
    # 在函数内计算波动数据，使用设定的均线窗口
    mean_data = sliding_average(original_data, MOVING_AVERAGE_WINDOW)
    fluc_data = original_data - mean_data

    best_rss = np.inf
    best_params = None
    best_season = None

    options = {
        'ftol': 1e-9,
        'gtol': 1e-6,
        'maxiter': 10000,
        'maxfun': 10000,
        'disp': False  # 关闭详细输出
    }

    for season in range(7, 25):
        # 改进的初始猜测 - 基于经验的更好起始点  
        if season <= 7:
            initial_guess = [0.3, 0.05, 0.1]
        elif season <= 15:
            initial_guess = [0.2, 0.1, 0.3]
        else:
            initial_guess = [0.1, 0.15, 0.4]
            
        bounds = [(0.0, 1.0), (0.0, 1.0), (0.0, 1.0)]

        def local_objective(params):
            alpha, beta, gamma = params
            smoothed = holtwinters_rolling(original_data, alpha, beta, gamma, season_length=season)
            holtwinters_fluc = original_data - smoothed
            fluc_sub = fluc_data[holtwinters_begindate:holtwinters_enddate]
            holtwinters_fluc_sub = holtwinters_fluc[holtwinters_begindate:holtwinters_enddate]
            a = calc_scaling_factor(fluc_sub, holtwinters_fluc_sub)
            rss = calc_RSS(fluc_sub, holtwinters_fluc_sub, a)
            return rss

        res = opt.minimize(local_objective,
                           initial_guess,
                           bounds=bounds,
                           method='L-BFGS-B',
                           options=options)
        if res.fun < best_rss:
            best_rss = res.fun
            best_params = res.x
            best_season = season

    return best_params, best_season, best_rss

def compute_optimize_result(end_day, original_data, fundcode=""):
    """辅助函数，用于并行计算"""
    best_params, best_season, best_rss = optimize_holtwinters_parameters(original_data, -800, end_day, fundcode)
    return {
        "end_day": end_day,
        "alpha": best_params[0],
        "beta": best_params[1],
        "gamma": best_params[2],
        "season": best_season,
        "rss": best_rss
    }

def process_hw_opt(original_data, output_base_dir, max_workers=8):
    """
    处理多个基金的优化过程
    
    参数:
        original_data: pd.DataFrame，每列代表一个基金的价格数据，列名为基金代码
        output_base_dir: 输出根目录
        max_workers: 并行线程数
    """
    if not isinstance(original_data, pd.DataFrame):
        raise ValueError("original_data must be a pandas DataFrame")
    
    fundcode_list = original_data.columns.tolist()
    results_summary = []
    
    for i, fundcode in enumerate(fundcode_list):
        try:
            # 获取单个基金的数据
            fund_data = original_data[fundcode].dropna()
            
            # 创建基金特定的输出目录
            fund_output_dir = os.path.join(output_base_dir, str(fundcode))
            os.makedirs(fund_output_dir, exist_ok=True)
            
            # 处理数据
            mean_data = sliding_average(fund_data, MOVING_AVERAGE_WINDOW)
            
            # 优化参数
            result = compute_optimize_result(None, fund_data, fundcode)
            
            # 保存结果
            results_df = pd.DataFrame([result])
            
            # 保存优化结果CSV
            # results_csv_path = os.path.join(fund_output_dir, f"holtwinters_results_{MOVING_AVERAGE_WINDOW}.csv")
            # results_df.to_csv(results_csv_path, index=False)
            
            # 绘制并保存图形
            plt.figure(figsize=(15, 10))
            
            # 绘制原始数据、滑动平均和HoltWinter平滑结果
            plt.plot(fund_data.index, fund_data.values, label='Original Data', marker='o', linestyle='-', markersize=1)
            plt.plot(mean_data.index, mean_data.values, label='Sliding Average', marker='x', linestyle='--', markersize=1)
            
            # 使用优化结果
            last_result = results_df.iloc[-1]
            holtwinter_smoothed = holtwinters_rolling(fund_data.values, last_result['alpha'], last_result['beta'], 
                                                        last_result['gamma'], int(last_result['season']))
            plt.plot(fund_data.index, holtwinter_smoothed, label='HoltWinter', linewidth=2)
            
            plt.title(f'Data Comparison - Fund {fundcode}')
            plt.xlabel('Index')
            plt.ylabel('Value')
            plt.legend()
            plt.grid(True)
            
            # 保存图形
            # plot_path = os.path.join(fund_output_dir, f"holtwinters_plot_{MOVING_AVERAGE_WINDOW}.png")
            # plt.savefig(plot_path, dpi=150, bbox_inches='tight')
            
            # 关闭图形以释放内存
            plt.close()  
            
            # 记录成功结果
            fund_result = {
                'fundcode': fundcode,
                'status': 'success',
                'data_points': len(fund_data),
                'alpha': last_result['alpha'],
                'beta': last_result['beta'],
                'gamma': last_result['gamma'],
                'season': last_result['season'],
                'rss': last_result['rss']
                
            }
            results_summary.append(fund_result)

            print(f"  基金 {fundcode} 处理完成！参数: Alpha={last_result['alpha']:.6f}, Beta={last_result['beta']:.6f}, Gamma={last_result['gamma']:.6f}, Season={last_result['season']}")
            
        except Exception as e:
            print(f"  基金 {fundcode} 处理失败: {str(e)}")
            fund_result = {
                'fundcode': fundcode,
                'status': 'failed',
                'error': str(e)
            }
            results_summary.append(fund_result)
    
    # 保存汇总结果
    # summary_df = pd.DataFrame(results_summary)
    # summary_path = os.path.join(output_base_dir, "processing_summary.csv")
    # summary_df.to_csv(summary_path, index=False)
    
    # 输出汇总信息
    success_count = len([r for r in results_summary if r['status'] == 'success'])
    failed_count = len([r for r in results_summary if r['status'] == 'failed'])
    
    # print("\n" + "="*50)
    # print("批量处理完成!")
    # print(f"成功处理: {success_count} 个基金")
    # print(f"处理失败: {failed_count} 个基金")
    # print(f"汇总结果保存在: {summary_path}")
    
    if failed_count > 0:
        print("\n失败的基金:")
        for result in results_summary:
            if result['status'] == 'failed':
                print(f"  {result['fundcode']}: {result['error']}")
    
    return results_summary

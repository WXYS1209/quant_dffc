import numpy as np
import pandas as pd
import scipy.optimize as opt
from datetime import datetime
import matplotlib.pyplot as plt
import os

from dffc.holt_winters._holt_winters import HW
from concurrent.futures import ProcessPoolExecutor, as_completed

# 添加均线窗口大小设置
MOVING_AVERAGE_WINDOW = 30

def sliding_average(arr, window):
    """
    计算一个numpy数组的滑动平均，窗口大小自定义，
    两侧不足部分使用较小窗口平均补全，输出形状与原数据相同。
    """
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

def optimize_holtwinters_parameters(original_data, holtwinters_begindate, holtwinters_enddate):
    """
    对给定数据区间进行参数优化，返回最优参数和最优季节长度。
    """
    # 在函数内计算波动数据，使用设定的均线窗口
    mean_data = sliding_average(original_data, MOVING_AVERAGE_WINDOW)
    fluc_data = original_data - mean_data

    options = {
        'ftol': 1e-9,
        'gtol': 1e-6,
        'maxiter': 1000,
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
        for future in as_completed(futures):
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
        'maxiter': 1000,
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

def compute_optimize_result(end_day, original_data):
    """辅助函数，用于并行计算"""
    best_params, best_season, best_rss = optimize_holtwinters_parameters(original_data, -800, end_day)
    return {
        "end_day": end_day,
        "alpha": best_params[0],
        "beta": best_params[1],
        "gamma": best_params[2],
        "season": best_season,
        "rss": best_rss
    }

def process_single_fund(fundcode, original_data, output_base_dir, max_workers=10):
    """
    处理单个基金的优化过程
    
    参数:
        fundcode: 基金代码
        output_base_dir: 输出根目录
        max_workers: 并行线程数
    """
    try:
        
        # 创建基金特定的输出目录
        fund_output_dir = os.path.join(output_base_dir)
        os.makedirs(fund_output_dir, exist_ok=True)
        
        
        # 处理数据
        mean_data = sliding_average(original_data, MOVING_AVERAGE_WINDOW)
        
        # 并行优化
        end_days = list(range(-400, 0, 40))
        end_days.append(None)
        
        results = []
        print(f"  开始优化 {len(end_days)} 个时间点...")
        
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(compute_optimize_result, end_day, original_data): end_day for end_day in end_days}
            for i, future in enumerate(as_completed(futures)):
                result = future.result()
                results.append(result)
                print(f"  完成 {i+1}/{len(end_days)}: end_day={result['end_day']}, season={result['season']}, rss={result['rss']:.6f}")
        
        # 保存结果
        results_df = pd.DataFrame(results)
        results_df = results_df.sort_values('end_day')
        
        # 保存优化结果CSV
        results_csv_path = os.path.join(fund_output_dir, f"holtwinters_results_{fundcode}_{MOVING_AVERAGE_WINDOW}.csv")
        results_df.to_csv(results_csv_path, index=False)
        
        # 绘制并保存图形
        plt.figure(figsize=(15, 10))
        
        # 第一张图：原始数据、滑动平均和HoltWinter平滑结果
        plt.subplot(2, 1, 1)
        plt.plot(original_data, label='Original Data', marker='o', linestyle='-', markersize=1)
        plt.plot(mean_data, label='Sliding Average', marker='x', linestyle='--', markersize=1)
        
        # 使用最后一个优化结果
        last_result = results_df.iloc[-1]
        holtwinter_smoothed = holtwinters_rolling(original_data, last_result['alpha'], last_result['beta'], 
                                                 last_result['gamma'], int(last_result['season']))
        plt.plot(holtwinter_smoothed, label='HoltWinter', linewidth=2)
        
        plt.title(f'Data Comparison - Fund {fundcode}')
        plt.xlabel('Index')
        plt.ylabel('Value')
        plt.legend()
        plt.grid(True)
        
        # 第二张图：参数变化
        plt.subplot(2, 1, 2)
        ax1 = plt.gca()
        ax2 = ax1.twinx()
        
        # 左y轴: alpha, beta, gamma
        line1 = ax1.plot(results_df['end_day'], results_df['alpha'], 'b-', marker='o', label='Alpha', markersize=4)
        line2 = ax1.plot(results_df['end_day'], results_df['beta'], 'g-', marker='s', label='Beta', markersize=4)
        line3 = ax1.plot(results_df['end_day'], results_df['gamma'], 'r-', marker='^', label='Gamma', markersize=4)
        ax1.set_xlabel('End Day')
        ax1.set_ylabel('Alpha, Beta, Gamma', color='black')
        ax1.tick_params(axis='y', labelcolor='black')
        ax1.grid(True, alpha=0.3)
        
        # 右y轴: season
        line4 = ax2.plot(results_df['end_day'], results_df['season'], 'm-', marker='D', label='Season', markersize=4)
        ax2.set_ylabel('Season Length', color='m')
        ax2.tick_params(axis='y', labelcolor='m')
        
        # 合并图例
        lines = line1 + line2 + line3 + line4
        labels = [l.get_label() for l in lines]
        ax1.legend(lines, labels, loc='upper left')
        
        plt.title(f'Parameters vs End Days - Fund {fundcode}')
        plt.tight_layout()
        
        # 保存图形
        plot_path = os.path.join(fund_output_dir, f"holtwinters_plot_{fundcode}_{MOVING_AVERAGE_WINDOW}.png")
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close()  # 关闭图形以释放内存
        
        # 输出最终参数信息
        print(f"  基金 {fundcode} 处理完成！")
        print(f"  最终参数: Alpha={last_result['alpha']:.6f}, Beta={last_result['beta']:.6f}, Gamma={last_result['gamma']:.6f}")
        print(f"  Season={last_result['season']}, RSS={last_result['rss']:.6f}")
        print(f"  结果保存在: {fund_output_dir}")
        
        return {
            'fundcode': fundcode,
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
        print(f"  基金 {fundcode} 处理失败: {str(e)}")
        return {
            'fundcode': fundcode,
            'status': 'failed',
            'error': str(e)
        }

def process_fund_list(fund_codes, output_base_dir="./optimize_results", max_workers=10):
    """
    批量处理基金列表
    
    参数:
        fund_codes: 基金代码列表
        output_base_dir: 输出根目录
        max_workers: 并行线程数
    """
    # 创建输出目录
    os.makedirs(output_base_dir, exist_ok=True)
    
    # 处理结果汇总
    summary_results = []
    
    print(f"开始批量处理 {len(fund_codes)} 个基金...")
    print(f"输出目录: {output_base_dir}")
    print(f"并行线程数: {max_workers}")
    print("-" * 50)
    
    for i, fundcode in enumerate(fund_codes):
        print(f"\n[{i+1}/{len(fund_codes)}] 处理基金 {fundcode}")
        # TODO: 需要添加从fundcode获取original_data的逻辑
        # 这里假设有一个函数可以根据fundcode获取数据
        # original_data = load_fund_data(fundcode)  # 需要实现这个函数
        original_data = None  # 暂时设为None，需要用户提供数据获取逻辑
        result = process_single_fund(fundcode, original_data, output_base_dir, max_workers)
        summary_results.append(result)
    
    # 保存汇总结果
    summary_df = pd.DataFrame(summary_results)
    summary_path = os.path.join(output_base_dir, "processing_summary.csv")
    summary_df.to_csv(summary_path, index=False)
    
    # 输出汇总信息
    success_count = len([r for r in summary_results if r['status'] == 'success'])
    failed_count = len([r for r in summary_results if r['status'] == 'failed'])
    
    print("\n" + "="*50)
    print("批量处理完成!")
    print(f"成功处理: {success_count} 个基金")
    print(f"处理失败: {failed_count} 个基金")
    print(f"汇总结果保存在: {summary_path}")
    
    if failed_count > 0:
        print("\n失败的基金:")
        for result in summary_results:
            if result['status'] == 'failed':
                print(f"  {result['fundcode']}: {result['error']}")

if __name__ == "__main__":
    # 示例：定义要处理的基金代码列表
    fund_codes = [
        '008777',  # 华安沪深300ETF联接C
        '006221',  # 工银瑞信上证50ETF联接C  
        '011320',  # 国泰上证综合ETF联接C
        '004744',  # 易方达创业板ETF联接C
        '016786',  # 鹏华中证1000指数增强C
    ]
    
    # 可以从配置文件读取基金列表
    # 例如：
    import json
    with open('./fund_config_etf.json', 'r', encoding='utf-8') as f:
        config = json.load(f)
    fund_codes = [item['code'] for item in config]
    
    # 批量处理
    process_fund_list(
        fund_codes=fund_codes,
        output_base_dir="./optimize_results",
        max_workers=8  # 根据你的CPU核心数调整
    )

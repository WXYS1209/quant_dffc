from strategy import DualReallocationStrategy
import vectorbt as vbt
import dffc
from dffc import process_hw_opt
import pickle
import pandas as pd
import numpy as np

def main():
    codes = ['007467', '004253']
    names=['HL', 'GD']
    start_date = 0
    end_date = '2025-07-15'

    try:
        fund_data = dffc.FundData.load('hlvsgd_all.pkl')
    except FileNotFoundError:
        fund_data = dffc.FundData.download(
            codes,
            names=names,
            start=start_date,
            end=end_date
        )

        fund_data.save('hlvsgd_all.pkl')

    price_data = fund_data.get('cumulative_value').dropna()

    # 创建包含开始日期以及每隔一年日期的列表
    start_date = price_data.index[0]
    end_date = price_data.index[-1]
    
    # 生成每隔一年的日期列表
    date_points = []
    current_date = start_date
    
    while current_date <= end_date:
        date_points.append(current_date)
        # 添加一年（考虑闰年）
        try:
            current_date = current_date.replace(year=current_date.year + 1)
        except ValueError:
            # 处理2月29日的情况（闰年到非闰年）
            current_date = current_date.replace(year=current_date.year + 1, day=28)
    
    # 确保包含最后一个日期
    if date_points[-1] != end_date:
        date_points.append(end_date)
    
    print(f"生成的日期点: {[d.strftime('%Y-%m-%d') for d in date_points]}")
    
    # 步骤1: 用第一年的数据优化参数
    first_year_start = date_points[0]
    first_year_end = date_points[1] 
    first_year_data = price_data.loc[first_year_start:first_year_end]
    
    print(f"\n=== 步骤1: 用第一年数据({first_year_start.strftime('%Y-%m-%d')} 到 {first_year_end.strftime('%Y-%m-%d')})优化HW参数 ===")
    current_opt_res = process_hw_opt(
        first_year_data, 
        save=True, 
        output_base_dir="./hw_opt_results", 
        result_filename=f"hl_gd_train_{first_year_start.strftime('%Y%m%d')}_{first_year_end.strftime('%Y%m%d')}.pkl", 
        plot=False
    )
    print(f"第一年参数优化完成，将用于第二年回测")
    
    # 步骤2: 从第二年开始回测，收集所有年度数据用于最终统一回测
    backtest_start = date_points[1]  # 第二年开始
    backtest_end = date_points[-1]   # 最后一年结束
    
    print(f"\n=== 步骤2: 准备从第二年({backtest_start.strftime('%Y-%m-%d')})到最后({backtest_end.strftime('%Y-%m-%d')})的连续回测 ===")
    
    # 收集所有年度的参数和权重数据
    all_backtest_data = []
    all_opt_params = []
    
    for i in range(1, len(date_points)):  # 从第二年开始
        year_start = date_points[i-1]
        year_end = date_points[i]
        
        print(f"\n--- 第{i}年: {year_start.strftime('%Y-%m-%d')} 到 {year_end.strftime('%Y-%m-%d')} ---")
        
        if i == 1:
            # 第二年：使用第一年优化的参数
            year_opt_res = current_opt_res
            print(f"使用第一年({first_year_start.strftime('%Y-%m-%d')}-{first_year_end.strftime('%Y-%m-%d')})的优化参数")
        else:
            # 第三年及以后：使用前一年的数据优化参数
            prev_year_start = date_points[i-2]
            prev_year_end = date_points[i-1]
            prev_year_data = price_data.loc[prev_year_start:prev_year_end]
            
            print(f"使用前一年数据({prev_year_start.strftime('%Y-%m-%d')}-{prev_year_end.strftime('%Y-%m-%d')})优化参数")
            year_opt_res = process_hw_opt(
                prev_year_data, 
                save=True, 
                output_base_dir="./hw_opt_results", 
                result_filename=f"hl_gd_train_{prev_year_start.strftime('%Y%m%d')}_{prev_year_end.strftime('%Y%m%d')}.pkl", 
                plot=False
            )
            current_opt_res = year_opt_res
        
        # 收集当前年度的数据和参数
        year_data = price_data.loc[year_start:year_end]
        all_backtest_data.append(year_data)
        all_opt_params.append(year_opt_res)
        
        print(f"第{i}年数据和参数准备完成")
    
    # 步骤3: 创建完整的回测数据（第二年到最后）
    full_backtest_data = price_data.loc[backtest_start:backtest_end]
    
    print(f"\n=== 步骤3: 运行动态参数更新回测 ({backtest_start.strftime('%Y-%m-%d')} 到 {backtest_end.strftime('%Y-%m-%d')}) ===")
    
    # 实现动态参数更新的回测
    # 我们需要按年度分段回测，每段使用对应的参数
    
    cumulative_portfolio_value = [100000]  # 初始资金
    all_trades = []
    all_portfolio_values = []
    all_rebalance_dates = []
    
    current_cash = 100000
    current_positions = np.array([0.0, 0.0])  # 两个资产的持仓数量
    
    print(f"开始动态参数更新回测...")
    
    for i, (year_start, year_end, opt_params) in enumerate(zip(
        [backtest_start] + date_points[2:-1], 
        date_points[2:] + [backtest_end],
        all_opt_params
    )):
        print(f"\n--- 回测第{i+1}个时间段: {year_start.strftime('%Y-%m-%d')} 到 {year_end.strftime('%Y-%m-%d')} ---")
        
        # 获取当前时间段的数据
        segment_data = price_data.loc[year_start:year_end]
        
        if len(segment_data) == 0:
            print(f"时间段 {year_start} 到 {year_end} 没有数据，跳过")
            continue
            
        # 创建当前时间段的策略（使用对应的参数）
        segment_strategy = DualReallocationStrategy(
            prices=segment_data,
            adjust_factor=0.2,
            rebalance_freq='D',
            default_weights=[0.5, 0.5],
            up_weights=[0.2, 0.8],
            down_weights=[0.8, 0.2], 
            threshold=0.6,
            hw_params_list=opt_params
        )
        
        # 计算当前持仓价值作为初始资金
        if i > 0:  # 非第一个时间段
            # 使用前一段结束时的持仓价值
            start_prices = segment_data.iloc[0]
            position_value = (current_positions * start_prices).sum()
            initial_cash_for_segment = position_value
        else:
            initial_cash_for_segment = current_cash
            
        print(f"时间段初始资金: {initial_cash_for_segment:.2f}")
        print(f"使用参数: {[p['fundcode'] for p in opt_params]}")
        
        # 运行当前时间段的回测
        segment_portfolio, segment_rebalance_mask, segment_weights = segment_strategy.run_backtest(
            initial_cash=initial_cash_for_segment,
            fees=0.001,
            trade_delay=1
        )
        
        # 收集结果
        segment_values = segment_portfolio.value()
        all_portfolio_values.extend(segment_values.tolist())
        
        # 更新当前持仓（为下一个时间段准备）
        if len(segment_data) > 0:
            final_weights = segment_weights.iloc[-1].values
            final_value = segment_values.iloc[-1]
            final_prices = segment_data.iloc[-1]
            current_positions = (final_weights * final_value) / final_prices
            current_cash = final_value
            
        print(f"时间段结束资金: {current_cash:.2f}")
        print(f"时间段收益率: {(current_cash/initial_cash_for_segment - 1)*100:.2f}%")
    
    # 创建完整的回测数据用于分析
    full_backtest_data = price_data.loc[backtest_start:backtest_end]
    
    # 为了兼容分析函数，创建一个虚拟的portfolio对象
    # 这里我们使用最后一个策略的结构，但替换价值序列
    final_strategy = DualReallocationStrategy(
        prices=full_backtest_data,
        adjust_factor=0.2,
        rebalance_freq='D',
        default_weights=[0.5, 0.5],
        up_weights=[0.2, 0.8],
        down_weights=[0.8, 0.2], 
        threshold=0.6,
        hw_params_list=all_opt_params[-1]  # 使用最后一个参数集
    )
    
    # 运行完整回测以获取portfolio对象结构（用于分析）
    portfolio, rebalance_mask, actual_weights = final_strategy.run_backtest(
        initial_cash=100000,
        fees=0.001,
        trade_delay=1
    )
    
    # 计算动态更新的实际收益率
    dynamic_total_return = (current_cash / 100000) - 1
    trading_days = len(full_backtest_data)
    trading_years = trading_days / 252
    dynamic_annual_return = (current_cash / 100000) ** (1 / trading_years) - 1
    
    print(f"\n=== 动态参数更新回测完成 ===")
    print(f"初始资金: 100,000.00")
    print(f"最终资金: {current_cash:,.2f}")
    print(f"动态更新总收益率: {dynamic_total_return:.4f} ({dynamic_total_return*100:.2f}%)")
    print(f"动态更新年化收益率: {dynamic_annual_return:.4f} ({dynamic_annual_return*100:.2f}%)")
    
    # 分析最终结果（使用静态参数的portfolio进行结构分析）
    stats = final_strategy.analyze_results(portfolio)
    
    # 计算静态参数的指标（对比用）
    static_annual_return = portfolio.annualized_return()
    static_sharpe_ratio = portfolio.sharpe_ratio()
    static_max_drawdown = portfolio.max_drawdown()
    static_total_return = portfolio.total_return()
    
    # 汇总最终结果
    final_result = {
        'backtest_period': f"{backtest_start.strftime('%Y-%m-%d')} to {backtest_end.strftime('%Y-%m-%d')}",
        'start_date': backtest_start,
        'end_date': backtest_end,
        'dynamic_annual_return': dynamic_annual_return,
        'dynamic_total_return': dynamic_total_return,
        'static_annual_return': static_annual_return,
        'static_total_return': static_total_return,
        'static_sharpe_ratio': static_sharpe_ratio,
        'static_max_drawdown': static_max_drawdown,
        'final_cash': current_cash,
        'portfolio': portfolio,  # 静态参数的portfolio（用于结构分析）
        'rebalance_mask': rebalance_mask,
        'actual_weights': actual_weights,
        'final_strategy': final_strategy,
        'all_opt_params': all_opt_params  # 保存所有年度的参数
    }
    
    # 步骤4: 显示最终回测结果
    print(f"\n=== 最终回测结果对比 ({backtest_start.strftime('%Y-%m-%d')} 到 {backtest_end.strftime('%Y-%m-%d')}) ===")
    print(f"回测期间: {final_result['backtest_period']}")
    print(f"\n【动态参数更新结果】")
    print(f"年化收益率: {dynamic_annual_return:.4f} ({dynamic_annual_return*100:.2f}%)")
    print(f"总收益率: {dynamic_total_return:.4f} ({dynamic_total_return*100:.2f}%)")
    print(f"最终资金: {current_cash:,.2f}")
    
    print(f"\n【静态参数对比结果】(使用最后一年参数)")
    print(f"年化收益率: {static_annual_return:.4f} ({static_annual_return*100:.2f}%)")
    print(f"总收益率: {static_total_return:.4f} ({static_total_return*100:.2f}%)")
    print(f"夏普比率: {static_sharpe_ratio:.4f}")
    print(f"最大回撤: {static_max_drawdown:.4f} ({static_max_drawdown*100:.2f}%)")
    
    # 计算改进程度
    improvement = dynamic_annual_return - static_annual_return
    print(f"\n【动态参数优势】")
    print(f"年化收益率改进: {improvement:.4f} ({improvement*100:.2f} percentage points)")
    print(f"收益率改进比例: {(improvement/abs(static_annual_return))*100:.1f}%" if static_annual_return != 0 else "无法计算")
    
    # 显示参数更新历史
    print(f"\n=== 参数更新历史 ===")
    for i, opt_param in enumerate(all_opt_params):
        year_start = date_points[i+1]
        year_end = date_points[i+2] if i+2 < len(date_points) else date_points[-1]
        train_start = date_points[i] if i > 0 else date_points[0]
        train_end = date_points[i+1] if i > 0 else date_points[1]
        
        print(f"第{i+2}年({year_start.strftime('%Y-%m-%d')}-{year_end.strftime('%Y-%m-%d')}): "
              f"使用第{i+1}年({train_start.strftime('%Y-%m-%d')}-{train_end.strftime('%Y-%m-%d')})的训练参数")
    
    # 保存最终结果
    with open('./hw_opt_results/final_backtest_result.pkl', 'wb') as f:
        pickle.dump(final_result, f)
    
    print(f"\n最终回测结果已保存到 './hw_opt_results/final_backtest_result.pkl'")
    
    return final_result

if __name__ == "__main__":
    results = main()


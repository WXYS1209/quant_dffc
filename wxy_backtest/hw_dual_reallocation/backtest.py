from strategy import DualReallocationStrategy
import vectorbt as vbt
import dffc

if __name__ == '__main__':
    codes = ['007467', '004253']
    names=['HL', 'GD']
    start_date = '2022-07-01'
    end_date = '2025-07-01'

    fund_data = dffc.FundData.download(
        codes,
        names=names,
        start=start_date,
        end=end_date
    )

    price_data = fund_data.get('cumulative_value')

    strategy = DualReallocationStrategy(
        prices = price_data,
        adjust_factor=0.5,
        rebalance_freq='D',  # 日度再平衡
        default_weights = [0.5, 0.5],
        up_weights = [0.2, 0.8],
        down_weights= [0.8, 0.2], 
        threshold=0.6,
        optimization=True  # 先禁用优化避免多进程问题
    )

    portfolio, rebalance_mask, actual_weights = strategy.run_backtest(initial_cash=100000, fees = 0.001)

    # 分析结果
    stats = strategy.analyze_results(portfolio)

    # 绘制结果
    strategy.plot_results(portfolio, rebalance_mask)

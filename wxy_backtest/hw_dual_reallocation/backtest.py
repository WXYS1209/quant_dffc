from strategy import DualReallocationStrategy
import vectorbt as vbt
import sys
import os
project_root = os.path.join(os.path.dirname(__file__), '..', '..')
sys.path.append(os.path.abspath(project_root))
from dffc.fund_data import register_fund_data

register_fund_data()

codes = ['007467', '004253']
names=['HL', 'GD']
start_date = '2022-07-01'
end_date = '2025-07-31'

fund_data = vbt.FundData.download(
    codes,
    names=names,
    start=start_date,
    end=end_date
)

price_data = fund_data.get('cumulative_value')

strategy = DualReallocationStrategy(
    prices = price_data,
    adjust_factor=0.2,
    rebalance_freq='D',  # 日度再平衡
    default_weights = [0.5, 0.5],
    up_weights = [0.3, 0.7],
    down_weights= [0.7, 0.3], 
    threshold=1.2,
    optimization=True
)

portfolio, rebalance_mask, actual_weights = strategy.run_backtest(initial_cash=100000, fees = 0.001)

# 分析结果
stats = strategy.analyze_results(portfolio)

# 绘制结果
strategy.plot_results(portfolio, rebalance_mask)

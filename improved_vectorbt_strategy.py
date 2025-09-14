"""
改进的vectorbt双资产再平衡策略
基于MarketCalls教程的最佳实践
"""
import numpy as np
import pandas as pd
import vectorbt as vbt
import matplotlib.pyplot as plt
from dffc.holt_winters._holt_winters import HWDP
from dffc.holt_winters._optimization import process_hw_opt
from dffc.fund_data import register_fund_data

# 设置vectorbt配置
vbt.settings.array_wrapper['freq'] = 'days'
vbt.settings.returns['year_freq'] = '252 days'
vbt.settings.portfolio['seed'] = 42
vbt.settings.portfolio.stats['incl_unrealized'] = True

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

class DualReallocationStrategy:
    """
    基于vectorbt的双资产再平衡策略
    """
    
    def __init__(
        self, 
        prices, 
        default_weights=[0.5, 0.5],
        up_weights=[0.2, 0.8],
        down_weights=[0.8, 0.2], 
        threshold=0.6,
        adjust_factor=0.2,
        rebalance_freq='D',  # 'D', 'W', 'M', 'Q'
        optimization=True
    ):
        """
        初始化策略
        
        Args:
            prices: DataFrame, 价格数据
            default_weights: list, 默认权重
            up_weights: list, 上升趋势权重  
            down_weights: list, 下降趋势权重
            threshold: float, 磁滞回线阈值
            adjust_factor: float, 调整因子
            rebalance_freq: str, 再平衡频率 ('D', 'W', 'M', 'Q')
            optimization: bool, 是否优化HW参数
        """
        self.prices = prices
        self.default_weights = np.array(default_weights)
        self.up_weights = np.array(up_weights) 
        self.down_weights = np.array(down_weights)
        self.threshold = threshold
        self.adjust_factor = adjust_factor
        self.rebalance_freq = rebalance_freq
        self.optimization = optimization
        
        # 计算HW信号
        self.hw_signals = self._calculate_hw_signals()
        
        # 生成目标权重序列
        self.target_weights = self._generate_target_weights()
        
    def _calculate_hw_signals(self):
        """计算Holt-Winters信号"""
        hw_signals = pd.DataFrame(index=self.prices.index, columns=self.prices.columns)
        
        if self.optimization:
            print("优化Holt-Winters参数...")
            result = process_hw_opt(self.prices, ".", 8)
            hw_params = {}
            for fund_result in result:
                hw_params[fund_result['fundcode']] = {
                    'alpha': fund_result['alpha'],
                    'beta': fund_result['beta'], 
                    'gamma': fund_result['gamma'],
                    'm': fund_result['season']
                }
        else:
            # 使用默认参数
            hw_params = {col: {'alpha': 0.3, 'beta': 0.1, 'gamma': 0.1, 'm': 8} 
                        for col in self.prices.columns}
            
        for col in self.prices.columns:
            params = hw_params[col]
            hwdp_result = HWDP.run(
                self.prices[col], 
                alpha=params['alpha'],
                beta=params['beta'], 
                gamma=params['gamma'],
                m=params['m'],
                multiplicative=True
            )
            hw_signals[col] = hwdp_result.hwdp
            
        return hw_signals
    
    def _generate_target_weights(self):
        """生成目标权重序列"""
        print("生成目标权重序列...")
        
        # 计算HDP差值
        delta_hdp = self.hw_signals.iloc[:, 0] - self.hw_signals.iloc[:, 1]
        
        # 磁滞回线逻辑
        signals = pd.Series(index=self.prices.index, dtype=int)
        signals.iloc[0] = 0  # 初始信号
        memory_switch = True
        
        for i in range(1, len(delta_hdp)):
            prev_switch = memory_switch
            
            if prev_switch and delta_hdp.iloc[i] > self.threshold:
                signals.iloc[i] = 1  # 上升信号
                memory_switch = False
            elif not prev_switch and delta_hdp.iloc[i] < -self.threshold:
                signals.iloc[i] = -1  # 下降信号
                memory_switch = True
            else:
                signals.iloc[i] = signals.iloc[i-1]  # 保持前一个信号
        
        # 根据信号生成目标权重
        target_weights = pd.DataFrame(index=self.prices.index, columns=self.prices.columns)
        
        for i in range(len(signals)):
            if signals.iloc[i] == 1:
                target_weights.iloc[i] = self.up_weights
            elif signals.iloc[i] == -1:
                target_weights.iloc[i] = self.down_weights
            else:
                if i == 0:
                    target_weights.iloc[i] = self.default_weights
                else:
                    target_weights.iloc[i] = target_weights.iloc[i-1]
        
        return target_weights
    
    def _create_rebalance_schedule(self):
        """创建再平衡时间表"""
        if self.rebalance_freq == 'D':
            # 每日再平衡
            rb_mask = pd.Series(True, index=self.prices.index)
        elif self.rebalance_freq == 'W':
            # 每周再平衡  
            rb_mask = ~self.prices.index.to_period('W').duplicated()
        elif self.rebalance_freq == 'M':
            # 每月再平衡
            rb_mask = ~self.prices.index.to_period('M').duplicated()
        elif self.rebalance_freq == 'Q':
            # 每季度再平衡
            rb_mask = ~self.prices.index.to_period('Q').duplicated()
        else:
            raise ValueError("rebalance_freq must be one of 'D', 'W', 'M', 'Q'")
            
        return rb_mask
    
    def run_backtest(self, initial_cash=100000, fees=0.001):
        """
        运行回测 - 使用改进的vectorbt方法
        """
        print("准备回测数据...")
        
        # 创建MultiIndex结构（按MarketCalls教程）
        num_tests = 1
        _prices = self.prices.vbt.tile(num_tests, keys=pd.Index(np.arange(num_tests), name='symbol_group'))
        
        # 创建再平衡时间表
        rb_mask = self._create_rebalance_schedule()
        
        # 创建渐进调整的实际权重序列
        actual_weights = pd.DataFrame(index=self.prices.index, columns=self.prices.columns)
        actual_weights.iloc[0] = self.target_weights.iloc[0].copy()  # 初始权重
        
        # 渐进调整逻辑
        actual_rebalances = pd.Series(False, index=self.prices.index)
        actual_rebalances.iloc[0] = True  # 第一天初始化
        
        tolerance = 0.01  # 权重差异容忍度
        
        for i in range(1, len(self.target_weights)):
            if rb_mask[i]:  # 这是一个再平衡机会日
                target_weights = self.target_weights.iloc[i]
                current_weights = actual_weights.iloc[i-1]
                
                # 计算权重差异
                weight_diff = target_weights - current_weights
                max_diff = abs(weight_diff).max()
                
                if max_diff > tolerance:
                    # 需要调整权重
                    adjusted_weights = current_weights + weight_diff * self.adjust_factor
                    actual_weights.iloc[i] = adjusted_weights
                    actual_rebalances.iloc[i] = True
                    print(f"再平衡 {i}: {self.prices.index[i].date()}")
                    print(f"  目标权重: {target_weights.values}")
                    print(f"  当前权重: {current_weights.values}")
                    print(f"  调整权重: {adjusted_weights.values}")
                    print(f"  最大差异: {max_diff:.4f}")
                else:
                    # 权重差异很小，保持当前权重
                    actual_weights.iloc[i] = current_weights
            else:
                # 非再平衡日，保持前一天权重
                actual_weights.iloc[i] = actual_weights.iloc[i-1]
        
        rebalance_mask = actual_rebalances
        print(f"总再平衡次数: {rebalance_mask.sum()}")
        print(f"再平衡日期: {rebalance_mask.sum()}个")
        
        # 显示前几个再平衡日期
        if rebalance_mask.sum() > 0:
            rb_dates = self.prices.index[rebalance_mask]
            print(f"前5个再平衡日期: {rb_dates[:5].tolist()}")
        
        # 创建订单矩阵 - 使用实际权重而不是目标权重
        orders = np.full_like(_prices, np.nan)
        
        # 在再平衡日期设置实际权重
        for i, should_rebalance in enumerate(rebalance_mask):
            if should_rebalance:
                orders[i, :] = actual_weights.iloc[i].values
        
        print("运行vectorbt回测...")
        pd.DataFrame(orders).to_csv("./orders_debug.csv")
        # 使用vectorbt运行回测
        portfolio = vbt.Portfolio.from_orders(
            close=_prices,
            size=orders,
            size_type='TargetPercent',
            group_by='symbol_group',
            cash_sharing=True,
            call_seq='auto',  # 先卖后买
            fees=fees,
            init_cash=initial_cash,
            freq='1D',
            min_size=1,
            size_granularity=1
        )
        
        return portfolio, rebalance_mask, actual_weights
    
    def analyze_results(self, portfolio):
        """分析回测结果"""
        print("\\n=== 策略表现分析 ===")
        
        # 整体统计
        stats = portfolio.stats()
        print("整体统计:")
        print(stats)
        
        # 个股表现
        individual_returns = portfolio.total_return(group_by=False) * 100
        print("\\n个股收益率:")
        print(individual_returns)
        
        # 交易摘要
        print("\\n=== 交易摘要 ===")
        orders_count = portfolio.orders.count()
        if hasattr(orders_count, 'sum'):
            print(f"总交易次数: {orders_count.sum()}")
        else:
            print(f"总交易次数: {orders_count}")
            
        fees_paid = portfolio.orders.fees.sum()
        if hasattr(fees_paid, 'sum'):
            print(f"总交易费用: {fees_paid.sum():.2f}")
        else:
            print(f"总交易费用: {fees_paid:.2f}")
        
        return stats
    
    def plot_results(self, portfolio, rebalance_mask):
        """绘制结果"""
        fig, axes = plt.subplots(4, 1, figsize=(15, 16))
        
        # 1. 价格走势
        axes[0].plot(self.prices.index, self.prices.iloc[:, 0], 
                     label=self.prices.columns[0], linewidth=2)
        axes[0].plot(self.prices.index, self.prices.iloc[:, 1], 
                     label=self.prices.columns[1], linewidth=2)
        axes[0].set_title('资产价格走势', fontsize=14)
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # 2. HW信号差值
        delta_hdp = self.hw_signals.iloc[:, 0] - self.hw_signals.iloc[:, 1]
        axes[1].plot(delta_hdp.index, delta_hdp, label='HDP差值', linewidth=2)
        axes[1].axhline(y=self.threshold, color='r', linestyle='--', label=f'上阈值 ({self.threshold})')
        axes[1].axhline(y=-self.threshold, color='r', linestyle='--', label=f'下阈值 ({-self.threshold})')
        axes[1].axhline(y=0, color='k', linestyle='-', alpha=0.3)
        axes[1].set_title('Holt-Winters信号差值', fontsize=14)
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        # 3. 权重分配（类似MarketCalls教程的权重图）
        asset_values = portfolio.asset_value(group_by=False)
        total_value = portfolio.value()
        weights = asset_values.div(total_value, axis=0)
        
        # 绘制堆叠面积图显示权重变化
        axes[2].fill_between(weights.index, 0, weights.iloc[:, 0], 
                            label=self.prices.columns[0], alpha=0.7)
        axes[2].fill_between(weights.index, weights.iloc[:, 0], 1,
                            label=self.prices.columns[1], alpha=0.7)
        
        # 标记再平衡日期
        rb_dates = weights.index[rebalance_mask]
        for rb_date in rb_dates[::5]:  # 每5个标记一个，避免过密
            axes[2].axvline(x=rb_date, color='gray', linestyle=':', alpha=0.5)
            
        axes[2].set_title('实际权重分配变化', fontsize=14)
        axes[2].set_ylim(0, 1)
        axes[2].legend()
        axes[2].grid(True, alpha=0.3)
        
        # 4. 累计收益对比
        portfolio_value = portfolio.value()
        portfolio_returns = portfolio_value / portfolio_value.iloc[0]
        
        # 等权重基准
        benchmark_returns = self.prices.pct_change().mean(axis=1).fillna(0)
        benchmark_cumret = (1 + benchmark_returns).cumprod()
        
        axes[3].plot(portfolio_returns.index, portfolio_returns, 
                     label='再平衡策略', linewidth=3, color='blue')
        axes[3].plot(benchmark_cumret.index, benchmark_cumret, 
                     label='等权重基准', linewidth=2, color='orange')
        axes[3].set_title('累计收益对比', fontsize=14)
        axes[3].legend()
        axes[3].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()


def demo_improved_strategy():
    """演示改进的策略"""
    # 注册基金数据
    register_fund_data()
    
    print("下载数据...")
    fund_data = vbt.FundData.download(
        ['007467', '004253'],
        names=['HL', 'GD'],
        start='2022-09-01',
        end='2025-09-01'
    )
    prices = fund_data.get('cumulative_value').dropna()
    
    # 创建改进的策略
    strategy = DualReallocationStrategy(
        prices=prices,
        default_weights=[0.5, 0.5],
        up_weights=[0.2, 0.8],
        down_weights=[0.8, 0.2],
        threshold=1.2,
        adjust_factor=0.2,
        rebalance_freq='D',  # 月度再平衡
        optimization=True
    )
    
    # 运行回测
    portfolio, rebalance_mask, actual_weights = strategy.run_backtest(initial_cash=100000)
    
    # 分析结果
    stats = strategy.analyze_results(portfolio)
    
    # 绘制结果
    strategy.plot_results(portfolio, rebalance_mask)
    
    return strategy, portfolio, rebalance_mask, actual_weights


if __name__ == "__main__":
    strategy, portfolio, rebalance_mask, actual_weights = demo_improved_strategy()
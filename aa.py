"""
改进的vectorbt双资产再平衡策略
基    def __init__(
        self, 
        prices, 
        default_weights=[0.5, 0.5],
        up_weights=[0.2, 0.8],
        down_weights=[0.8, 0.2], 
        threshold=0.6,
        adjust_factor=0.2,
        rebalance_freq='M',  # 'D', 'W', 'M', 'Q'
        optimization=True,
        rolling_optimization=False,  # 是否使用滚动优化
        optimization_window=252,     # 优化窗口长度（交易日）
        reoptimize_freq='Q'          # 重新优化频率（'M', 'Q', 'Y'）
    ):s教程的最佳实践
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

class ImprovedVectorBTStrategy:
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
        rebalance_freq='M',  # 'D', 'W', 'M', 'Q'
        optimization=True,
        rolling_optimization=False,  # 是否使用滚动优化
        optimization_window=252,     # 优化窗口长度（交易日）
        reoptimize_freq='Q'          # 重新优化频率（'M', 'Q', 'Y'）
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
            rolling_optimization: bool, 是否使用滚动优化
            optimization_window: int, 优化窗口长度（天数）
            reoptimize_freq: str, 重新优化频率 ('M', 'Q', 'Y')
        """
        self.prices = prices
        self.default_weights = np.array(default_weights)
        self.up_weights = np.array(up_weights) 
        self.down_weights = np.array(down_weights)
        self.threshold = threshold
        self.adjust_factor = adjust_factor
        self.rebalance_freq = rebalance_freq
        self.optimization = optimization
        self.rolling_optimization = rolling_optimization
        self.optimization_window = optimization_window
        self.reoptimize_freq = reoptimize_freq
        
        # 计算HW信号
        if self.rolling_optimization:
            self.hw_signals = self._calculate_hw_signals_rolling()
        else:
            self.hw_signals = self._calculate_hw_signals()
        
        # 生成目标权重序列
        self.target_weights = self._generate_target_weights()
        
    def _get_reoptimization_dates(self):
        """获取重新优化的日期"""
        if self.reoptimize_freq == 'M':
            # 每月重新优化
            mask = ~self.prices.index.to_period('M').duplicated()
        elif self.reoptimize_freq == 'Q':
            # 每季度重新优化
            mask = ~self.prices.index.to_period('Q').duplicated()
        elif self.reoptimize_freq == 'Y':
            # 每年重新优化
            mask = ~self.prices.index.to_period('Y').duplicated()
        else:
            raise ValueError("reoptimize_freq must be one of 'M', 'Q', 'Y'")
        
        # 确保有足够的历史数据才开始优化
        reopt_dates = self.prices.index[mask]
        valid_dates = []
        for date in reopt_dates:
            days_from_start = (date - self.prices.index[0]).days
            if days_from_start >= self.optimization_window:
                valid_dates.append(date)
        
        return valid_dates
    
    def _optimize_hw_params_for_window(self, end_date):
        """为指定窗口优化HW参数"""
        # 获取优化窗口数据
        end_idx = self.prices.index.get_loc(end_date)
        start_idx = max(0, end_idx - self.optimization_window + 1)
        
        window_prices = self.prices.iloc[start_idx:end_idx+1]
        
        print(f"  优化窗口: {window_prices.index[0].date()} 至 {window_prices.index[-1].date()} ({len(window_prices)}天)")
        
        try:
            result = process_hw_opt(window_prices, ".", 8)
            hw_params = {}
            for fund_result in result:
                hw_params[fund_result['fundcode']] = {
                    'alpha': fund_result['alpha'],
                    'beta': fund_result['beta'], 
                    'gamma': fund_result['gamma'],
                    'm': fund_result['season']
                }
            return hw_params
        except Exception as e:
            print(f"  优化失败，使用默认参数: {e}")
            return {col: {'alpha': 0.3, 'beta': 0.1, 'gamma': 0.1, 'm': 8} 
                   for col in window_prices.columns}
    
    def _calculate_hw_signals_rolling(self):
        """使用滚动窗口计算HW信号"""
        print("使用滚动窗口优化HW参数...")
        print(f"优化窗口: {self.optimization_window}天")
        print(f"重新优化频率: {self.reoptimize_freq}")
        
        hw_signals = pd.DataFrame(index=self.prices.index, columns=self.prices.columns)
        
        # 获取重新优化日期
        reopt_dates = self._get_reoptimization_dates()
        print(f"重新优化日期: {len(reopt_dates)}个")
        
        # 当前使用的参数
        current_params = None
        current_params_date = None
        
        for i, date in enumerate(self.prices.index):
            # 检查是否需要重新优化参数
            if date in reopt_dates:
                print(f"\\n重新优化 {len([d for d in reopt_dates if d <= date])}/{len(reopt_dates)}: {date.date()}")
                current_params = self._optimize_hw_params_for_window(date)
                current_params_date = date
                
                # 显示优化结果
                for fund_code, params in current_params.items():
                    print(f"    {fund_code}: α={params['alpha']:.3f}, β={params['beta']:.3f}, γ={params['gamma']:.3f}, m={params['m']}")
            
            # 如果还没有参数（序列开始时），使用默认参数
            if current_params is None:
                if i >= self.optimization_window:
                    # 有足够数据时进行首次优化
                    print(f"\\n首次优化: {date.date()}")
                    current_params = self._optimize_hw_params_for_window(date)
                    current_params_date = date
                else:
                    # 使用默认参数
                    current_params = {col: {'alpha': 0.3, 'beta': 0.1, 'gamma': 0.1, 'm': 8} 
                                    for col in self.prices.columns}
            
            # 计算当天的HW信号
            for col in self.prices.columns:
                if i == 0:
                    hw_signals.loc[date, col] = 0  # 第一天信号为0
                else:
                    params = current_params[col]
                    # 使用截至当前日期的数据计算HWDP
                    price_series = self.prices.loc[:date, col]
                    
                    if len(price_series) >= params['m'] + 2:  # 确保有足够数据
                        try:
                            hwdp_result = HWDP.run(
                                price_series, 
                                alpha=params['alpha'],
                                beta=params['beta'], 
                                gamma=params['gamma'],
                                m=params['m'],
                                multiplicative=True
                            )
                            hw_signals.loc[date, col] = hwdp_result.hwdp.iloc[-1]
                        except:
                            hw_signals.loc[date, col] = hw_signals.iloc[i-1, hw_signals.columns.get_loc(col)]
                    else:
                        # 数据不够时保持前值
                        if i > 0:
                            hw_signals.loc[date, col] = hw_signals.iloc[i-1, hw_signals.columns.get_loc(col)]
                        else:
                            hw_signals.loc[date, col] = 0
        
        print("\\n滚动优化完成！")
        return hw_signals
        
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
        start='2022-07-01',
        end='2025-07-01'
    )
    prices = fund_data.get('cumulative_value').dropna()
    
    # 对比测试：静态优化 vs 滚动优化
    print("\\n=== 测试1: 传统静态优化 ===")
    strategy_static = ImprovedVectorBTStrategy(
        prices=prices,
        default_weights=[0.5, 0.5],
        up_weights=[0.2, 0.8],
        down_weights=[0.8, 0.2],
        threshold=0.6,
        adjust_factor=0.2,
        rebalance_freq='M',
        optimization=True,
        rolling_optimization=False  # 静态优化
    )
    portfolio_static, rebalance_mask_static, actual_weights_static = strategy_static.run_backtest(initial_cash=100000)
    stats_static = strategy_static.analyze_results(portfolio_static)
    
    print("\\n=== 测试2: 滚动窗口优化 ===")
    strategy_rolling = ImprovedVectorBTStrategy(
        prices=prices,
        default_weights=[0.5, 0.5],
        up_weights=[0.2, 0.8],
        down_weights=[0.8, 0.2], 
        threshold=0.6,
        adjust_factor=0.2,
        rebalance_freq='D',
        optimization=True,
        rolling_optimization=True,   # 启用滚动优化
        optimization_window=252,     # 1年窗口
        reoptimize_freq='Q'          # 每季度重新优化
    )
    portfolio_rolling, rebalance_mask_rolling, actual_weights_rolling = strategy_rolling.run_backtest(initial_cash=100000)
    stats_rolling = strategy_rolling.analyze_results(portfolio_rolling)
    
    # 比较结果
    print("\\n=== 策略对比 ===")
    print(f"静态优化总收益: {portfolio_static.total_return()*100:.2f}%")
    print(f"滚动优化总收益: {portfolio_rolling.total_return()*100:.2f}%")
    print(f"静态优化夏普比率: {portfolio_static.sharpe_ratio():.2f}")
    print(f"滚动优化夏普比率: {portfolio_rolling.sharpe_ratio():.2f}")
    
    # 绘制对比结果
    strategy_rolling.plot_results(portfolio_rolling, rebalance_mask_rolling)
    
    return strategy_rolling, portfolio_rolling, rebalance_mask_rolling, actual_weights_rolling


if __name__ == "__main__":
    strategy, portfolio, rebalance_mask, actual_weights = demo_improved_strategy()
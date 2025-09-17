"""
Improved vectorbt dual asset rebalancing strategy
Based on MarketCalls tutorial best practices

Features:
- Multi-asset rebalancing framework
- Configurable rebalancing frequency (D/W/M/Q/Y)
- Gradual weight adjustment with tolerance
- Simple trade execution delay support via weight matrix shifting
- Comprehensive performance analysis and visualization

Trade Delay Implementation:
- Simple and efficient: directly shifts weight matrix by N days
- T+0 (trade_delay=0): Immediate execution, suitable for stocks
- T+1 (trade_delay=1): Next-day execution, suitable for funds
- T+2+ (trade_delay=2+): Multi-day delay, suitable for special instruments

Usage:
    strategy = DualReallocationStrategy(prices=data, ...)
    portfolio, rebalances, weights = strategy.run_backtest(
        initial_cash=100000,
        fees=0.001,
        trade_delay=1  # T+1 for funds
    )
"""
import numpy as np
import pandas as pd
import vectorbt as vbt
import matplotlib.pyplot as plt
from abc import ABC, abstractmethod

# Set vectorbt configuration
vbt.settings.array_wrapper['freq'] = 'days'
vbt.settings.returns['year_freq'] = '252 days'
vbt.settings.portfolio['seed'] = 42
vbt.settings.portfolio.stats['incl_unrealized'] = True

# No Chinese font configuration needed
plt.rcParams['axes.unicode_minus'] = False


class Strategy(ABC):
    """
    vectorbt-based strategy base class
    
    Abstract base class for all strategies, defines basic interface and common functionality
    """
    
    def __init__(self, prices, **kwargs):
        """
        Initialize strategy base class
        
        Args:
            prices: DataFrame, price data
            **kwargs: other strategy-specific parameters
        """
        self.prices = prices
        self.validate_data()
    
    def validate_data(self):
        """Validate the validity of input data"""
        if self.prices is None or self.prices.empty:
            raise ValueError("Price data cannot be empty")
        
        if self.prices.isnull().any().any():
            print(f"Warning: Missing values found in price data, will perform forward fill")
            self.prices = self.prices.fillna(method='ffill')
    
    @abstractmethod
    def run_backtest(self, initial_cash=100000, fees=0.001):
        """
        Run backtest - must be implemented by subclasses
        
        Args:
            initial_cash: float, initial capital
            fees: float, trading fee rate
            
        Returns:
            portfolio: vectorbt Portfolio object
        """
        pass
    
    @abstractmethod
    def analyze_results(self, portfolio):
        """
        Analyze backtest results - must be implemented by subclasses
        
        Args:
            portfolio: vectorbt Portfolio object
            
        Returns:
            stats: statistical results
        """
        pass
    
    def plot_results(self, portfolio, **kwargs):
        """
        Plot results - provides default implementation, subclasses can override
        
        Args:
            portfolio: vectorbt Portfolio object
            **kwargs: plotting parameters
        """
        # Basic plotting: price trends and return curves
        fig, axes = plt.subplots(2, 1, figsize=(15, 10))
        
        # 1. Price trends
        for i, col in enumerate(self.prices.columns):
            axes[0].plot(self.prices.index, self.prices.iloc[:, i], 
                        label=col, linewidth=2)
        axes[0].set_title('Asset Price Trends', fontsize=14)
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # 2. Strategy returns
        portfolio_value = portfolio.value()
        portfolio_returns = portfolio_value / portfolio_value.iloc[0]
        
        axes[1].plot(portfolio_returns.index, portfolio_returns, 
                    label='Strategy Returns', linewidth=3, color='blue')
        axes[1].set_title('Cumulative Returns', fontsize=14)
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    def get_basic_stats(self, portfolio):
        """Get basic statistical information"""
        stats = {}
        
        # 总收益率
        total_return = portfolio.total_return() * 100
        stats['Total Return(%)'] = f"{total_return:.2f}"
        
        # 夏普比率
        if hasattr(portfolio, 'sharpe_ratio'):
            sharpe = portfolio.sharpe_ratio()
            stats['Sharpe Ratio'] = f"{sharpe:.2f}"
        
        # 最大回撤
        if hasattr(portfolio, 'max_drawdown'):
            max_dd = portfolio.max_drawdown() * 100
            stats['Max Drawdown(%)'] = f"{max_dd:.2f}"
        
        return stats


class ReallocationStrategy(Strategy):
    """
    General rebalancing strategy base class
    
    Multi-asset weight rebalancing framework that provides common rebalancing logic
    """
    
    def __init__(self, prices, rebalance_freq='W', adjust_factor=1.0, **kwargs):
        """
        Initialize rebalancing strategy
        
        Args:
            prices: DataFrame, price data (supports multi-asset)
            rebalance_freq: str, rebalancing frequency ('D', 'W', 'M', 'Q', 'Y')
            adjust_factor: float, weight adjustment factor (0-1), 1 means immediately adjust to target weight
            **kwargs: other strategy parameters
        """
        super().__init__(prices, **kwargs)
        self.rebalance_freq = rebalance_freq
        self.adjust_factor = adjust_factor
        
        # Weight-related attributes, set by subclasses
        self.target_weights = None
        
    def _create_rebalance_schedule(self):
        """Create rebalancing schedule"""
        if self.rebalance_freq == 'D':
            # Daily rebalancing
            return pd.Series(True, index=self.prices.index)
        elif self.rebalance_freq == 'W':
            # Weekly rebalancing
            week_starts = self.prices.groupby(pd.Grouper(freq='W')).first()
            return self.prices.index.isin(week_starts.index)
        elif self.rebalance_freq == 'M':
            # Monthly rebalancing
            month_starts = self.prices.groupby(pd.Grouper(freq='M')).first()
            return self.prices.index.isin(month_starts.index)
        elif self.rebalance_freq == 'Q':
            # Quarterly rebalancing
            quarter_starts = self.prices.groupby(pd.Grouper(freq='Q')).first()
            return self.prices.index.isin(quarter_starts.index)
        elif self.rebalance_freq == 'Y':
            # Annual rebalancing
            year_starts = self.prices.groupby(pd.Grouper(freq='Y')).first()
            return self.prices.index.isin(year_starts.index)
        else:
            raise ValueError(f"Unsupported rebalancing frequency: {self.rebalance_freq}")
    
    def _apply_gradual_adjustment(self, rb_mask, tolerance=0.01):
        """
        Apply gradual adjustment logic (supports multi-asset)
        
        Args:
            rb_mask: Series or array, rebalancing schedule
            tolerance: float, weight difference tolerance
            
        Returns:
            actual_weights: DataFrame, actual weight series
            actual_rebalances: Series, actual rebalancing schedule
        """
        if self.target_weights is None:
            raise ValueError("Target weight series not set, please call _generate_target_weights() first")
        
        # Ensure rb_mask is pandas Series
        if isinstance(rb_mask, np.ndarray):
            rb_mask = pd.Series(rb_mask, index=self.prices.index)
        
        actual_weights = pd.DataFrame(index=self.prices.index, columns=self.prices.columns)
        actual_weights.iloc[0] = self.target_weights.iloc[0].copy()
        
        actual_rebalances = pd.Series(False, index=self.prices.index)
        actual_rebalances.iloc[0] = True
        
        rebalance_count = 0
        
        for i in range(1, len(self.target_weights)):
            if rb_mask.iloc[i]:  # 再平衡机会日
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
                    rebalance_count += 1
                else:
                    # 权重差异很小，保持当前权重
                    actual_weights.iloc[i] = current_weights
            else:
                # 非再平衡日，保持前一天权重
                actual_weights.iloc[i] = actual_weights.iloc[i-1]
        
        print(f"Total rebalancing count: {actual_rebalances.sum()}")
        return actual_weights, actual_rebalances
    
    @abstractmethod
    def _generate_target_weights(self):
        """
        生成目标权重序列 - 子类必须实现
        
        Returns:
            target_weights: DataFrame, 目标权重序列（每行权重之和应为1）
        """
        pass
    
    def run_backtest(self, initial_cash=100000, fees=0.001, trade_delay=0):
        """
        运行再平衡策略回测（支持多资产）
        
        Args:
            initial_cash: float, 初始资金
            fees: float, 交易费用率
            trade_delay: int, 交易执行延迟天数 (0=T+0, 1=T+1, 2=T+2, etc.)
                        基金推荐使用 trade_delay=1 (T+1)
            
        Returns:
            portfolio: vectorbt Portfolio对象
            rebalance_mask: Series, 再平衡时间表
            actual_weights: DataFrame, 实际权重序列
        """
        print("Preparing backtest data...")
        
        # 确保目标权重已生成
        if self.target_weights is None:
            self.target_weights = self._generate_target_weights()
        
        # 创建MultiIndex结构
        num_tests = 1
        _prices = self.prices.vbt.tile(num_tests, keys=pd.Index(np.arange(num_tests), name='symbol_group'))
        
        # 创建再平衡时间表
        rb_mask = self._create_rebalance_schedule()
        
        # 应用渐进调整
        actual_weights, actual_rebalances = self._apply_gradual_adjustment(rb_mask)
        
        # 🔧 应用交易延迟：简单地将权重矩阵向后shift
        if trade_delay > 0:
            print(f"Applying T+{trade_delay} trade delay...")
            
            # 将权重向后shift，模拟交易延迟
            delayed_weights = actual_weights.shift(trade_delay).ffill()
            delayed_rebalances = actual_rebalances.shift(trade_delay).fillna(False)
            
            # 前面几天使用初始权重
            # for i in range(min(trade_delay, len(delayed_weights))):
            #     delayed_weights.iloc[i] = actual_weights.iloc[0]
            #     delayed_rebalances.iloc[i] = (i == 0)  # 只有第一天是初始买入
            
            actual_weights = delayed_weights
            actual_rebalances = delayed_rebalances
            
            print(f"Trade delay applied: T+{trade_delay}")
            print(f"Adjusted rebalancing count: {actual_rebalances.sum()}")
        
        # 创建订单矩阵
        orders = np.full_like(_prices, np.nan)
        
        # 在再平衡日期设置实际权重
        for i, should_rebalance in enumerate(actual_rebalances):
            if should_rebalance:
                orders[i, :] = actual_weights.iloc[i].values
        
        orders = pd.DataFrame(orders, index=_prices.index, columns=_prices.columns)

        print("Running vectorbt backtest...")
        
        # 使用vectorbt运行回测
        portfolio = vbt.Portfolio.from_orders(
            close=_prices,
            size=orders,
            size_type='TargetPercent',
            group_by='symbol_group',
            cash_sharing=True,
            call_seq='auto',
            fees=fees,
            init_cash=initial_cash,
            freq='1D',
            min_size=0.01,
            size_granularity=0.01
        )
        
        return portfolio, actual_rebalances, actual_weights
    
    def analyze_results(self, portfolio):
        """分析再平衡策略结果（支持多资产）"""
        print("\n=== Rebalancing Strategy Performance Analysis ===")
        
        # 基础统计
        basic_stats = self.get_basic_stats(portfolio)
        for key, value in basic_stats.items():
            print(f"{key}: {value}")
        
        # 详细统计
        stats = portfolio.stats()
        print("\nDetailed Statistics:")
        print(stats)
        
        # 交易摘要
        print("\n=== Trading Summary ===")
        orders_count = portfolio.orders.count()
        if hasattr(orders_count, 'sum'):
            print(f"Total trade count: {orders_count.sum()}")
        else:
            print(f"总交易次数: {orders_count}")
            
        fees_paid = portfolio.orders.fees.sum()
        if hasattr(fees_paid, 'sum'):
            print(f"总交易费用: {fees_paid.sum():.2f}")
        else:
            print(f"总交易费用: {fees_paid:.2f}")
        
        return stats
    
    def plot_results(self, portfolio, rebalance_mask, **kwargs):
        """绘制再平衡策略结果（支持多资产）"""
        num_assets = len(self.prices.columns)
        fig, axes = plt.subplots(3, 1, figsize=(15, 12))
        
        # 获取交易信息
        orders = portfolio.orders.records
        
        # 1. 价格走势 + 交易标记（支持多资产）
        colors = plt.cm.Set1(np.linspace(0, 1, num_assets))
        for i, col in enumerate(self.prices.columns):
            axes[0].plot(self.prices.index, self.prices.iloc[:, i], 
                        label=col, linewidth=2, color=colors[i])
        
        # 添加买入/卖出标记
        if hasattr(orders, 'side'):
            buy_orders = orders.side == 1
            sell_orders = orders.side == 0
            
            buy_times = orders.idx[buy_orders] if buy_orders.any() else []
            sell_times = orders.idx[sell_orders] if sell_orders.any() else []

            axes[0].scatter(self.prices.index[buy_times], orders.price[buy_orders], 
                            marker='^', color='green', s=20, alpha=0.7, 
                            label='买入')

            axes[0].scatter(self.prices.index[sell_times], orders.price[sell_orders], 
                            marker='v', color='red', s=20, alpha=0.7, 
                            label='卖出')
        
        axes[0].set_title(f'{num_assets}资产价格走势及交易标记', fontsize=14)
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # 2. 权重分配（支持多资产堆叠）
        asset_values = portfolio.asset_value(group_by=False)
        total_value = portfolio.value()
        weights = asset_values.div(total_value, axis=0)
        
        # 绘制堆叠面积图
        bottom = np.zeros(len(weights))
        for i, col in enumerate(self.prices.columns):
            axes[1].fill_between(weights.index, bottom, bottom + weights.iloc[:, i], 
                               label=col, alpha=0.7, color=colors[i])
            bottom += weights.iloc[:, i]
        
        # 标记再平衡日期和权重变化
        rb_dates = weights.index[rebalance_mask]
        if len(weights) > 1:
            weight_changes = weights.diff()
            
            for i, rb_date in enumerate(rb_dates[::max(1, len(rb_dates)//30)]):
                if i < 30 and rb_date in weight_changes.index:
                    # 计算最大权重变化
                    max_change = abs(weight_changes.loc[rb_date]).max()
                    
                    if max_change > 0.01:  # 权重变化超过1%才标记
                        # 用不同颜色表示变化强度
                        color = 'darkgreen' if max_change > 0.05 else 'orange'
                        axes[1].annotate('⟲', xy=(rb_date, 0.95), 
                                       ha='center', va='center',
                                       fontsize=12, color=color, weight='bold',
                                       alpha=0.8)
                
                # 标记再平衡日期的竖线
                if i % 3 == 0:  # 每3个标记一条线
                    axes[1].axvline(x=rb_date, color='gray', linestyle=':', alpha=0.3)
        
        axes[1].text(0.02, 0.98, '⟲: 再平衡（深绿>5%, 橙色1-5%）', 
                    transform=axes[1].transAxes, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        axes[1].set_title('权重分配变化（含交易标记）', fontsize=14)
        axes[1].set_ylim(0, 1)
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        # 3. 累计收益对比
        portfolio_value = portfolio.value()
        portfolio_returns = portfolio_value / portfolio_value.iloc[0]
        
        # 等权重基准
        benchmark_returns = self.prices.pct_change().mean(axis=1).fillna(0)
        benchmark_cumret = (1 + benchmark_returns).cumprod()
        
        axes[2].plot(portfolio_returns.index, portfolio_returns, 
                    label='再平衡策略', linewidth=3, color='blue')
        axes[2].plot(benchmark_cumret.index, benchmark_cumret, 
                    label='等权重基准', linewidth=2, color='orange')
        
        # 标记主要再平衡点
        rb_dates_major = rb_dates[::max(1, len(rb_dates)//10)]
        for rb_date in rb_dates_major[:10]:
            if rb_date in portfolio_returns.index:
                axes[2].axvline(x=rb_date, color='purple', linestyle='--', alpha=0.3, linewidth=0.8)
        
        # 添加交易费用说明
        total_fees = portfolio.orders.fees.sum()
        if hasattr(total_fees, 'sum'):
            total_fees = total_fees.sum()
        
        axes[2].text(0.02, 0.02, f'累计交易费用: {total_fees:.2f}', 
                    transform=axes[2].transAxes,
                    bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.7))
        
        axes[2].set_title('累计收益对比（紫线标记主要再平衡点）', fontsize=14)
        axes[2].legend()
        axes[2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
        # 打印交易统计摘要
        print("\n=== 交易标记说明 ===")
        print("价格图: ▲绿色三角 = 买入点, ▼红色三角 = 卖出点")
        print("权重图: ⟲ = 再平衡标记（深绿>5%变化, 橙色1-5%变化）")
        print("收益图: 紫色虚线 = 主要再平衡点")
        
        # 统计交易次数
        if hasattr(orders, 'side'):
            buy_count = (orders.side == 'Buy').sum() if hasattr(orders.side, 'sum') else 0
            sell_count = (orders.side == 'Sell').sum() if hasattr(orders.side, 'sum') else 0
            print(f"总买入次数: {buy_count}, 总卖出次数: {sell_count}")
        
        rebalance_count = rebalance_mask.sum()
        print(f"再平衡次数: {rebalance_count}")
        print(f"平均再平衡间隔: {len(self.prices) / max(1, rebalance_count):.1f} 天")
        print(f"资产数量: {num_assets}")



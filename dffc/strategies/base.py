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
from abc import ABC, abstractmethod

# 设置vectorbt配置
vbt.settings.array_wrapper['freq'] = 'days'
vbt.settings.returns['year_freq'] = '252 days'
vbt.settings.portfolio['seed'] = 42
vbt.settings.portfolio.stats['incl_unrealized'] = True

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


class Strategy(ABC):
    """
    基于vectorbt的策略基类
    
    所有策略的抽象基类，定义了策略的基本接口和通用功能
    """
    
    def __init__(self, prices, **kwargs):
        """
        初始化策略基类
        
        Args:
            prices: DataFrame, 价格数据
            **kwargs: 其他策略特定参数
        """
        self.prices = prices
        self.validate_data()
    
    def validate_data(self):
        """验证输入数据的有效性"""
        if self.prices is None or self.prices.empty:
            raise ValueError("价格数据不能为空")
        
        if self.prices.isnull().any().any():
            print(f"警告: 价格数据中存在缺失值，将进行前向填充")
            self.prices = self.prices.fillna(method='ffill')
    
    @abstractmethod
    def run_backtest(self, initial_cash=100000, fees=0.001):
        """
        运行回测 - 子类必须实现
        
        Args:
            initial_cash: float, 初始资金
            fees: float, 交易费用率
            
        Returns:
            portfolio: vectorbt Portfolio对象
        """
        pass
    
    @abstractmethod
    def analyze_results(self, portfolio):
        """
        分析回测结果 - 子类必须实现
        
        Args:
            portfolio: vectorbt Portfolio对象
            
        Returns:
            stats: 统计结果
        """
        pass
    
    def plot_results(self, portfolio, **kwargs):
        """
        绘制结果 - 提供默认实现，子类可重写
        
        Args:
            portfolio: vectorbt Portfolio对象
            **kwargs: 绘图参数
        """
        # 基础绘图：价格走势和收益曲线
        fig, axes = plt.subplots(2, 1, figsize=(15, 10))
        
        # 1. 价格走势
        for i, col in enumerate(self.prices.columns):
            axes[0].plot(self.prices.index, self.prices.iloc[:, i], 
                        label=col, linewidth=2)
        axes[0].set_title('资产价格走势', fontsize=14)
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # 2. 策略收益
        portfolio_value = portfolio.value()
        portfolio_returns = portfolio_value / portfolio_value.iloc[0]
        
        axes[1].plot(portfolio_returns.index, portfolio_returns, 
                    label='策略收益', linewidth=3, color='blue')
        axes[1].set_title('累计收益', fontsize=14)
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    def get_basic_stats(self, portfolio):
        """获取基础统计信息"""
        stats = {}
        
        # 总收益率
        total_return = portfolio.total_return() * 100
        stats['总收益率(%)'] = f"{total_return:.2f}"
        
        # 夏普比率
        if hasattr(portfolio, 'sharpe_ratio'):
            sharpe = portfolio.sharpe_ratio()
            stats['夏普比率'] = f"{sharpe:.2f}"
        
        # 最大回撤
        if hasattr(portfolio, 'max_drawdown'):
            max_dd = portfolio.max_drawdown() * 100
            stats['最大回撤(%)'] = f"{max_dd:.2f}"
        
        return stats


class ReallocationStrategy(Strategy):
    """
    通用再平衡策略基类
    
    支持多资产的权重再平衡框架，提供通用的再平衡逻辑
    """
    
    def __init__(self, prices, rebalance_freq='D', adjust_factor=0.2, **kwargs):
        """
        初始化再平衡策略
        
        Args:
            prices: DataFrame, 价格数据（支持多资产）
            rebalance_freq: str, 再平衡频率 ('D', 'W', 'M', 'Q', 'Y')
            adjust_factor: float, 权重调整因子 (0-1)，1表示立即调整到目标权重
            **kwargs: 其他策略参数
        """
        super().__init__(prices, **kwargs)
        self.rebalance_freq = rebalance_freq
        self.adjust_factor = adjust_factor
        
        # 权重相关属性，由子类设置
        self.target_weights = None
        
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
        elif self.rebalance_freq == 'Y':
            # 每年再平衡
            rb_mask = ~self.prices.index.to_period('Y').duplicated()
        else:
            raise ValueError("rebalance_freq must be one of 'D', 'W', 'M', 'Q', 'Y'")
            
        return rb_mask
    
    def _apply_gradual_adjustment(self, rb_mask, tolerance=0.01):
        """
        应用渐进调整逻辑（支持多资产）
        
        Args:
            rb_mask: Series or array, 再平衡时间表
            tolerance: float, 权重差异容忍度
            
        Returns:
            actual_weights: DataFrame, 实际权重序列
            actual_rebalances: Series, 实际再平衡时间表
        """
        if self.target_weights is None:
            raise ValueError("目标权重序列未设置，请先调用 _generate_target_weights()")
        
        # 确保 rb_mask 是 pandas Series
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
                    
                    # if rebalance_count <= 5:  # 只打印前5次再平衡信息
                    #     print(f"再平衡 {rebalance_count}: {self.prices.index[i].date()}")
                    #     print(f"  目标权重: {target_weights.values}")
                    #     print(f"  调整权重: {adjusted_weights.values}")
                    #     print(f"  最大差异: {max_diff:.4f}")
                else:
                    # 权重差异很小，保持当前权重
                    actual_weights.iloc[i] = current_weights
            else:
                # 非再平衡日，保持前一天权重
                actual_weights.iloc[i] = actual_weights.iloc[i-1]
        
        print(f"总再平衡次数: {actual_rebalances.sum()}")
        return actual_weights, actual_rebalances
    
    @abstractmethod
    def _generate_target_weights(self):
        """
        生成目标权重序列 - 子类必须实现
        
        Returns:
            target_weights: DataFrame, 目标权重序列（每行权重之和应为1）
        """
        pass
    
    def run_backtest(self, initial_cash=100000, fees=0.001):
        """
        运行再平衡策略回测（支持多资产）
        
        Args:
            initial_cash: float, 初始资金
            fees: float, 交易费用率
            
        Returns:
            portfolio: vectorbt Portfolio对象
            rebalance_mask: Series, 再平衡时间表
            actual_weights: DataFrame, 实际权重序列
        """
        print("准备回测数据...")
        
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
        
        # 创建订单矩阵
        orders = np.full_like(_prices, np.nan)
        
        # 在再平衡日期设置实际权重
        for i, should_rebalance in enumerate(actual_rebalances):
            if should_rebalance:
                orders[i, :] = actual_weights.iloc[i].values
        
        orders = pd.DataFrame(orders, index=_prices.index, columns=_prices.columns)

        print("运行vectorbt回测...")
        
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
        print("\n=== 再平衡策略表现分析 ===")
        
        # 基础统计
        basic_stats = self.get_basic_stats(portfolio)
        for key, value in basic_stats.items():
            print(f"{key}: {value}")
        
        # 详细统计
        stats = portfolio.stats()
        print("\n详细统计:")
        print(stats)
        
        # 交易摘要
        print("\n=== 交易摘要 ===")
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



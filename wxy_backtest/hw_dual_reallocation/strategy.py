import sys
import os
project_root = os.path.join(os.path.dirname(__file__), '..', '..')
sys.path.append(os.path.abspath(project_root))
from dffc.strategies import ReallocationStrategy
from dffc.holt_winters import HWDP, process_hw_opt
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

class DualReallocationStrategy(ReallocationStrategy):
    """
    基于Holt-Winters的双资产再平衡策略
    
    继承ReallocationStrategy，专注于HW信号生成和双资产权重分配逻辑
    """
    
    def __init__(
        self, 
        prices, 
        default_weights=[0.5, 0.5],
        up_weights=[0.2, 0.8],
        down_weights=[0.8, 0.2], 
        threshold=0.6,
        optimization=True,
        **kwargs
    ):
        """
        初始化双资产再平衡策略
        
        Args:
            prices: DataFrame, 价格数据（必须是2列）
            default_weights: list, 默认权重
            up_weights: list, 上升趋势权重  
            down_weights: list, 下降趋势权重
            threshold: float, 磁滞回线阈值
            optimization: bool, 是否优化HW参数
            **kwargs: 其他基类参数（如adjust_factor, rebalance_freq等）
        """
        # 验证输入数据
        if len(prices.columns) != 2:
            raise ValueError("DualReallocationStrategy只支持双资产，请提供2列价格数据")
        
        # 调用父类初始化
        super().__init__(prices, **kwargs)
        
        # 策略特定参数
        self.default_weights = np.array(default_weights)
        self.up_weights = np.array(up_weights) 
        self.down_weights = np.array(down_weights)
        self.threshold = threshold
        self.optimization = optimization
        
        # 验证权重
        self._validate_weights()
        
        # 计算HW信号
        self.hw_signals = self._calculate_hw_signals()
        
        # 生成目标权重序列
        self.target_weights = self._generate_target_weights()
    
    def _validate_weights(self):
        """验证权重配置"""
        weights_to_check = [self.default_weights, self.up_weights, self.down_weights]
        weight_names = ['default_weights', 'up_weights', 'down_weights']
        
        for weights, name in zip(weights_to_check, weight_names):
            if len(weights) != 2:
                raise ValueError(f"{name} 必须包含2个元素")
            if abs(weights.sum() - 1.0) > 1e-6:
                raise ValueError(f"{name} 权重之和必须为1，当前为 {weights.sum()}")
            if (weights < 0).any():
                raise ValueError(f"{name} 权重不能为负数")
    
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
        """生成目标权重序列（基于HW信号差值的磁滞回线逻辑）"""
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
    
    def plot_results(self, portfolio, rebalance_mask, **kwargs):
        """绘制双资产策略结果 - 扩展基类绘图，添加HW信号图"""
        fig, axes = plt.subplots(4, 1, figsize=(15, 16))
        
        # 获取交易信息
        orders = portfolio.orders.records
        
        # 1. 价格走势 + 交易标记
        axes[0].plot(self.prices.index, self.prices.iloc[:, 0], 
                     label=self.prices.columns[0], linewidth=2)
        axes[0].plot(self.prices.index, self.prices.iloc[:, 1], 
                     label=self.prices.columns[1], linewidth=2)
        
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
        
        axes[0].set_title('双资产价格走势及交易标记', fontsize=14)
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # 2. HW信号差值 - DualReallocationStrategy特有
        delta_hdp = self.hw_signals.iloc[:, 0] - self.hw_signals.iloc[:, 1]
        axes[1].plot(delta_hdp.index, delta_hdp, label='HDP差值', linewidth=2)
        axes[1].axhline(y=self.threshold, color='r', linestyle='--', label=f'上阈值 ({self.threshold})')
        axes[1].axhline(y=-self.threshold, color='r', linestyle='--', label=f'下阈值 ({-self.threshold})')
        axes[1].axhline(y=0, color='k', linestyle='-', alpha=0.3)
        axes[1].set_title('Holt-Winters信号差值与阈值', fontsize=14)
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        # 3. 权重分配 + 双资产特殊标记
        asset_values = portfolio.asset_value(group_by=False)
        total_value = portfolio.value()
        weights = asset_values.div(total_value, axis=0)
        
        # 如果权重有MultiIndex，简化列名
        if isinstance(weights.columns, pd.MultiIndex):
            weights.columns = weights.columns.get_level_values(-1)
        
        # 绘制堆叠面积图
        axes[2].fill_between(weights.index, 0, weights.iloc[:, 0], 
                            label=self.prices.columns[0], alpha=0.7)
        axes[2].fill_between(weights.index, weights.iloc[:, 0], 1,
                            label=self.prices.columns[1], alpha=0.7)
        
        # 标记权重变化方向（双资产特有逻辑）
        rb_dates = weights.index[rebalance_mask]
        if len(weights) > 1:
            weight_changes = weights.diff()
            
            for i, rb_date in enumerate(rb_dates[::3]):
                if i < 30 and rb_date in weight_changes.index:
                    change_asset1 = weight_changes.loc[rb_date, weights.columns[0]]
                    
                    if abs(change_asset1) > 0.01:  # 权重变化超过1%才标记
                        if change_asset1 > 0:
                            # 第一个资产权重增加
                            axes[2].annotate('↗', xy=(rb_date, 0.9), 
                                           xytext=(rb_date, 0.95),
                                           ha='center', va='center',
                                           fontsize=16, color='green', weight='bold',
                                           alpha=0.8)
                        else:
                            # 第一个资产权重减少
                            axes[2].annotate('↘', xy=(rb_date, 0.9), 
                                           xytext=(rb_date, 0.95),
                                           ha='center', va='center',
                                           fontsize=16, color='red', weight='bold',
                                           alpha=0.8)
                
                # 标记再平衡日期的竖线
                if i % 2 == 0:
                    axes[2].axvline(x=rb_date, color='gray', linestyle=':', alpha=0.5)
        
        axes[2].text(0.02, 0.98, f'↗: 增持{self.prices.columns[0]}  ↘: 减持{self.prices.columns[0]}', 
                    transform=axes[2].transAxes, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        axes[2].set_title('双资产权重分配变化（含交易标记）', fontsize=14)
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
                     label='HW双资产策略', linewidth=3, color='blue')
        axes[3].plot(benchmark_cumret.index, benchmark_cumret, 
                     label='等权重基准', linewidth=2, color='orange')
        
        # 标记主要再平衡点
        rb_dates_major = rb_dates[::10]
        for rb_date in rb_dates_major[:10]:
            if rb_date in portfolio_returns.index:
                axes[3].axvline(x=rb_date, color='purple', linestyle='--', alpha=0.3, linewidth=0.8)
        
        # 添加策略信息
        total_fees = portfolio.orders.fees.sum()
        if hasattr(total_fees, 'sum'):
            total_fees = total_fees.sum()
        
        info_text = f'阈值: ±{self.threshold}\n调整因子: {self.adjust_factor}\n累计费用: {total_fees:.2f}'
        axes[3].text(0.02, 0.02, info_text, transform=axes[3].transAxes,
                    bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.7))
        
        axes[3].set_title('累计收益对比（Holt-Winters双资产策略）', fontsize=14)
        axes[3].legend()
        axes[3].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
        # 打印双资产特有的统计信息
        print("\n=== 双资产HW策略统计 ===")
        print(f"阈值设置: ±{self.threshold}")
        print(f"权重配置: 默认{self.default_weights}, 上升{self.up_weights}, 下降{self.down_weights}")
        print(f"HW优化: {'是' if self.optimization else '否'}")
        
        # 计算信号统计
        delta_hdp = self.hw_signals.iloc[:, 0] - self.hw_signals.iloc[:, 1]
        上升信号数 = (delta_hdp > self.threshold).sum()
        下降信号数 = (delta_hdp < -self.threshold).sum()
        中性信号数 = len(delta_hdp) - 上升信号数 - 下降信号数
        
        print(f"信号分布: 上升{上升信号数}天 ({上升信号数/len(delta_hdp)*100:.1f}%), "
              f"下降{下降信号数}天 ({下降信号数/len(delta_hdp)*100:.1f}%), "
              f"中性{中性信号数}天 ({中性信号数/len(delta_hdp)*100:.1f}%)")
        
        rebalance_count = rebalance_mask.sum()
        print(f"再平衡次数: {rebalance_count}")
        print(f"平均再平衡间隔: {len(self.prices) / max(1, rebalance_count):.1f} 天")

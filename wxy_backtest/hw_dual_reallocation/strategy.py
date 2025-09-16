from dffc.strategies import ReallocationStrategy
from dffc.holt_winters import HWDP, process_hw_opt
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


class DualReallocationStrategy(ReallocationStrategy):
    """
    Dual Asset Rebalancing Strategy based on Holt-Winters
    
    Inherits from ReallocationStrategy, focuses on HW signal generation and dual asset weight allocation logic
    """
    
    def __init__(
        self, 
        prices, 
        default_weights=[0.5, 0.5],
        up_weights=[0.2, 0.8],
        down_weights=[0.8, 0.2], 
        threshold=0.6,
        optimization=True,
        hw_params_list=None,
        **kwargs
    ):
        """
        Initialize dual asset rebalancing strategy
        
        Args:
            prices: DataFrame, price data (must have 2 columns)
            default_weights: list, default weights
            up_weights: list, weights for uptrend  
            down_weights: list, weights for downtrend
            threshold: float, hysteresis threshold
            optimization: bool, whether to optimize HW parameters
            hw_params_list: list of dict, external HW parameters when optimization=False
                Format: [{"code": "fund_code", "params": {"alpha": 0.1, "beta": 0.1, "gamma": 0.1, "season_length": 8}}, ...]
            **kwargs: other base class parameters (like adjust_factor, rebalance_freq etc.)
        """
        # Validate input data
        if len(prices.columns) != 2:
            raise ValueError("DualReallocationStrategy only supports dual assets, please provide 2-column price data")
        
        # Call parent class initialization
        super().__init__(prices, **kwargs)
        
        # Strategy specific parameters
        self.default_weights = np.array(default_weights)
        self.up_weights = np.array(up_weights) 
        self.down_weights = np.array(down_weights)
        self.threshold = threshold
        self.optimization = optimization
        self.hw_params_list = hw_params_list
        
        # Validate parameters
        self._validate_weights()
        self._validate_hw_params()
        
        # Calculate HW signals
        self.hw_signals = self._calculate_hw_signals()
        
        # Generate target weight series
        self.target_weights = self._generate_target_weights()
    
    def _validate_weights(self):
        """Validate weight configuration"""
        weights_to_check = [self.default_weights, self.up_weights, self.down_weights]
        weight_names = ['default_weights', 'up_weights', 'down_weights']
        
        for weights, name in zip(weights_to_check, weight_names):
            if len(weights) != 2:
                raise ValueError(f"{name} must contain 2 elements")
            if abs(weights.sum() - 1.0) > 1e-6:
                raise ValueError(f"{name} weights must sum to 1, current sum is {weights.sum()}")
            if (weights < 0).any():
                raise ValueError(f"{name} weights cannot be negative")
    
    def _validate_hw_params(self):
        """Validate HW parameters configuration"""
        if not self.optimization and self.hw_params_list is not None:
            if not isinstance(self.hw_params_list, list):
                raise ValueError("hw_params_list must be a list of dictionaries")
            
            if len(self.hw_params_list) != len(self.prices.columns):
                raise ValueError(f"hw_params_list must contain {len(self.prices.columns)} elements to match price columns")
            
            for i, param_dict in enumerate(self.hw_params_list):
                if not isinstance(param_dict, dict):
                    raise ValueError(f"hw_params_list[{i}] must be a dictionary")
                
                if 'code' not in param_dict or 'params' not in param_dict:
                    raise ValueError(f"hw_params_list[{i}] must contain 'code' and 'params' keys")
                
                params = param_dict['params']
                required_keys = ['alpha', 'beta', 'gamma', 'season_length']
                for key in required_keys:
                    if key not in params:
                        raise ValueError(f"hw_params_list[{i}]['params'] must contain '{key}' key")
    
    def _calculate_hw_signals(self):
        """Calculate Holt-Winters signals"""
        hw_signals = pd.DataFrame(index=self.prices.index, columns=self.prices.columns)
        
        if self.optimization:
            print("Optimizing Holt-Winters parameters...")
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
            if self.hw_params_list is not None:
                # Use external parameters
                print("Using external Holt-Winters parameters...")
                hw_params = {}
                for i, param_dict in enumerate(self.hw_params_list):
                    col = self.prices.columns[i]
                    params = param_dict['params']
                    hw_params[col] = {
                        'alpha': params['alpha'],
                        'beta': params['beta'], 
                        'gamma': params['gamma'],
                        'm': params['season_length']  # Note: using season_length from external params
                    }
                    print(f"  {col} ({param_dict['code']}): alpha={params['alpha']:.4f}, beta={params['beta']:.4f}, gamma={params['gamma']:.4f}, season={params['season_length']}")
            else:
                # Use default parameters
                print("Using default Holt-Winters parameters...")
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
        """Generate target weight series (based on hysteresis logic of HW signal difference)"""
        print("Generating target weight series...")
        
        # Calculate HDP difference
        delta_hdp = self.hw_signals.iloc[:, 0] - self.hw_signals.iloc[:, 1]
        
        # Hysteresis logic
        signals = pd.Series(index=self.prices.index, dtype=int)
        signals.iloc[0] = 0  # Initial signal
        memory_switch = True
        
        for i in range(1, len(delta_hdp)):
            prev_switch = memory_switch
            
            if prev_switch and delta_hdp.iloc[i] > self.threshold:
                signals.iloc[i] = 1  # Up signal
                memory_switch = False
            elif not prev_switch and delta_hdp.iloc[i] < -self.threshold:
                signals.iloc[i] = -1  # Down signal
                memory_switch = True
            else:
                signals.iloc[i] = signals.iloc[i-1]  # Keep previous signal
        
        # Generate target weights based on signals
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
        """Plot dual asset strategy results - extend base class plotting, add HW signal charts"""
        fig, axes = plt.subplots(4, 1, figsize=(15, 16))
        
        # Get trading information
        orders = portfolio.orders.records
        
        # 1. Price trend + trading markers
        axes[0].plot(self.prices.index, self.prices.iloc[:, 0], 
                     label=self.prices.columns[0], linewidth=2)
        axes[0].plot(self.prices.index, self.prices.iloc[:, 1], 
                     label=self.prices.columns[1], linewidth=2)
        
        # Add buy/sell markers
        if hasattr(orders, 'side'):
            buy_orders = orders.side == 1
            sell_orders = orders.side == 0
            
            buy_times = orders.idx[buy_orders] if buy_orders.any() else []
            sell_times = orders.idx[sell_orders] if sell_orders.any() else []

            axes[0].scatter(self.prices.index[buy_times], orders.price[buy_orders], 
                            marker='^', color='green', s=20, alpha=0.7, 
                            label='Buy')

            axes[0].scatter(self.prices.index[sell_times], orders.price[sell_orders], 
                            marker='v', color='red', s=20, alpha=0.7, 
                            label='Sell')
        
        axes[0].set_title('Dual Asset Price Trends and Trading Markers', fontsize=14)
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # 2. HW signal difference - DualReallocationStrategy specific
        delta_hdp = self.hw_signals.iloc[:, 0] - self.hw_signals.iloc[:, 1]
        axes[1].plot(delta_hdp.index, delta_hdp, label='HDP Difference', linewidth=2)
        axes[1].axhline(y=self.threshold, color='r', linestyle='--', label=f'Upper Threshold ({self.threshold})')
        axes[1].axhline(y=-self.threshold, color='r', linestyle='--', label=f'Lower Threshold ({-self.threshold})')
        axes[1].axhline(y=0, color='k', linestyle='-', alpha=0.3)
        axes[1].set_title('Holt-Winters Signal Difference and Thresholds', fontsize=14)
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        # 3. Weight allocation + dual asset special markers
        asset_values = portfolio.asset_value(group_by=False)
        total_value = portfolio.value()
        weights = asset_values.div(total_value, axis=0)
        
        # If weights have MultiIndex, simplify column names
        if isinstance(weights.columns, pd.MultiIndex):
            weights.columns = weights.columns.get_level_values(-1)
        
        # Draw stacked area chart
        axes[2].fill_between(weights.index, 0, weights.iloc[:, 0], 
                            label=self.prices.columns[0], alpha=0.7)
        axes[2].fill_between(weights.index, weights.iloc[:, 0], 1,
                            label=self.prices.columns[1], alpha=0.7)
        
        # Mark weight change direction (dual asset specific logic)
        rb_dates = weights.index[rebalance_mask]
        if len(weights) > 1:
            weight_changes = weights.diff()
            
            for i, rb_date in enumerate(rb_dates[::3]):
                if i < 30 and rb_date in weight_changes.index:
                    change_asset1 = weight_changes.loc[rb_date, weights.columns[0]]
                    
                    if abs(change_asset1) > 0.01:  # Mark only when weight change exceeds 1%
                        if change_asset1 > 0:
                            # First asset weight increased
                            axes[2].annotate('↗', xy=(rb_date, 0.9), 
                                           xytext=(rb_date, 0.95),
                                           ha='center', va='center',
                                           fontsize=16, color='green', weight='bold',
                                           alpha=0.8)
                        else:
                            # First asset weight decreased
                            axes[2].annotate('↘', xy=(rb_date, 0.9), 
                                           xytext=(rb_date, 0.95),
                                           ha='center', va='center',
                                           fontsize=16, color='red', weight='bold',
                                           alpha=0.8)
                
                # Mark rebalancing date vertical lines
                if i % 2 == 0:
                    axes[2].axvline(x=rb_date, color='gray', linestyle=':', alpha=0.5)
        
        axes[2].text(0.02, 0.98, f'↗: Increase {self.prices.columns[0]}  ↘: Decrease {self.prices.columns[0]}', 
                    transform=axes[2].transAxes, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        axes[2].set_title('Dual Asset Weight Allocation Changes (with Trading Markers)', fontsize=14)
        axes[2].set_ylim(0, 1)
        axes[2].legend()
        axes[2].grid(True, alpha=0.3)
        
        # 4. Cumulative return comparison
        portfolio_value = portfolio.value()
        portfolio_returns = portfolio_value / portfolio_value.iloc[0]
        
        # Equal weight benchmark
        benchmark_returns = self.prices.pct_change().mean(axis=1).fillna(0)
        benchmark_cumret = (1 + benchmark_returns).cumprod()
        
        axes[3].plot(portfolio_returns.index, portfolio_returns, 
                     label='HW Dual Asset Strategy', linewidth=3, color='blue')
        axes[3].plot(benchmark_cumret.index, benchmark_cumret, 
                     label='Equal Weight Benchmark', linewidth=2, color='orange')
        
        # Mark major rebalancing points
        rb_dates_major = rb_dates[::10]
        for rb_date in rb_dates_major[:10]:
            if rb_date in portfolio_returns.index:
                axes[3].axvline(x=rb_date, color='purple', linestyle='--', alpha=0.3, linewidth=0.8)
        
        # Add strategy information
        total_fees = portfolio.orders.fees.sum()
        if hasattr(total_fees, 'sum'):
            total_fees = total_fees.sum()
        
        info_text = f'Threshold: ±{self.threshold}\nAdjust Factor: {self.adjust_factor}\nTotal Fees: {total_fees:.2f}'
        axes[3].text(0.02, 0.02, info_text, transform=axes[3].transAxes,
                    bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.7))
        
        axes[3].set_title('Cumulative Return Comparison (Holt-Winters Dual Asset Strategy)', fontsize=14)
        axes[3].legend()
        axes[3].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
        # Print dual asset specific statistics
        print("\n=== Dual Asset HW Strategy Statistics ===")
        print(f"Threshold setting: ±{self.threshold}")
        print(f"Weight configuration: Default{self.default_weights}, Up{self.up_weights}, Down{self.down_weights}")
        print(f"HW optimization: {'Yes' if self.optimization else 'No'}")
        
        # Calculate signal statistics
        delta_hdp = self.hw_signals.iloc[:, 0] - self.hw_signals.iloc[:, 1]
        up_signal_count = (delta_hdp > self.threshold).sum()
        down_signal_count = (delta_hdp < -self.threshold).sum()
        neutral_signal_count = len(delta_hdp) - up_signal_count - down_signal_count
        
        print(f"Signal distribution: Up {up_signal_count} days ({up_signal_count/len(delta_hdp)*100:.1f}%), "
              f"Down {down_signal_count} days ({down_signal_count/len(delta_hdp)*100:.1f}%), "
              f"Neutral {neutral_signal_count} days ({neutral_signal_count/len(delta_hdp)*100:.1f}%)")
        
        rebalance_count = rebalance_mask.sum()
        print(f"Rebalancing count: {rebalance_count}")
        print(f"Average rebalancing interval: {len(self.prices) / max(1, rebalance_count):.1f} days")

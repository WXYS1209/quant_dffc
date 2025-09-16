"""
策略模块 (Strategies)

提供量化交易策略的实现，包括基础策略类和具体策略实现

模块结构:
- base: 策略基类和通用再平衡策略框架
- hw_dual_reallocation: 基于Holt-Winters的双资产再平衡策略

主要类:
- Strategy: 抽象策略基类
- ReallocationStrategy: 通用再平衡策略基类
- DualReallocationStrategy: 双资产Holt-Winters策略
"""

from dffc.strategies.base import (
    Strategy,
    ReallocationStrategy
)

# 检查是否存在 hw_dual_reallocation 模块
try:
    from wxy_backtest.hw_dual_reallocation import *
except ImportError:
    pass  # 如果模块不存在，忽略导入错误

__all__ = [
    # 基础策略类
    "Strategy",
    "ReallocationStrategy"
]

# 策略模块元信息
__module_name__ = "strategies"
__module_description__ = "量化交易策略实现模块"
__supported_strategies__ = [
    "基础策略框架",
    "通用再平衡策略"
]
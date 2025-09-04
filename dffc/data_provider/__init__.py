"""
数据提供者模块

提供统一的数据提供者抽象层，支持多种资产类型和数据源
"""

from dffc.data_provider.base import DataProvider, DataProviderConfig, BS4DataProvider
from dffc.data_provider.eastmoney_provider import EastMoneyFundProvider, EastMoneyStockProvider

__all__ = [
    # Base classes
    'DataProviderConfig', 
    'DataProvider',
    'BS4DataProvider',
    
    # EastMoney providers
    'EastMoneyFundProvider',
    'EastMoneyStockProvider',
]

__pdoc__ = {k: False for k in __all__}
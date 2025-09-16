"""
数据提供者模块 (Data Provider)

提供统一的数据提供者抽象层，支持多种资产类型和数据源

模块结构:
- base: 数据提供者基类和配置
- eastmoney_provider: 东方财富数据提供者
- stock_net_value_crawler: 股票净值爬虫

主要功能:
- 基金数据获取
- 股票数据获取  
- 统一的数据接口
- 多数据源支持
"""

from dffc.data_provider.base import (
    DataProvider, 
    DataProviderConfig, 
    BS4DataProvider
)

from dffc.data_provider.eastmoney_provider import (
    EastMoneyFundProvider, 
    EastMoneyStockProvider
)

# 尝试导入爬虫模块
try:
    from dffc.data_provider.stock_net_value_crawler import *
except ImportError:
    pass  # 如果依赖不满足，忽略导入错误

__all__ = [
    # 基础类
    'DataProviderConfig', 
    'DataProvider',
    'BS4DataProvider',
    
    # 东方财富提供者
    'EastMoneyFundProvider',
    'EastMoneyStockProvider',
]

# 模块元信息
__module_name__ = "data_provider"
__module_description__ = "统一的金融数据提供者接口"
__supported_sources__ = [
    "东方财富 (EastMoney)",
    "自定义爬虫",
    "BS4网页解析"
]

# 数据提供者注册表
AVAILABLE_PROVIDERS = {
    'eastmoney_fund': EastMoneyFundProvider,
    'eastmoney_stock': EastMoneyStockProvider,
}

def get_provider(provider_name):
    """
    获取指定的数据提供者
    
    Args:
        provider_name: str, 提供者名称
        
    Returns:
        DataProvider: 数据提供者实例
    """
    if provider_name not in AVAILABLE_PROVIDERS:
        raise ValueError(f"未知的数据提供者: {provider_name}")
    return AVAILABLE_PROVIDERS[provider_name]

# 添加到导出列表
__all__.append('get_provider')
__all__.append('AVAILABLE_PROVIDERS')
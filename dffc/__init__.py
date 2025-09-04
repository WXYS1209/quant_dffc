"""
DFFC (Data Feeds for Finance in China) 模块

提供中国金融市场数据获取功能，包括基金、股票等资产类型
集成vectorbt框架，提供统一的数据接口
"""

# 版本信息
__version__ = "0.1.0"
__author__ = "DFFC Team"
__description__ = "Data Feeds for Finance in China - 中国金融数据提供者"

# 导入核心功能
from dffc.fund_data import *

# 导入数据提供者
from dffc.data_provider import *
from dffc._utils import *

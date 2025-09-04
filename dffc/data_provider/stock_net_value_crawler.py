# -*- coding: utf-8 -*-
"""
实时股票净值数据爬虫
支持多种数据源获取股票/基金的实时净值数据
作者: Rick Xie
日期: 2025/06/10
"""

import requests
import json
import time
import pandas as pd
from datetime import datetime
from typing import Dict, List, Optional
import logging
import threading
from concurrent.futures import ThreadPoolExecutor


class StockNetValueCrawler:
    """实时股票净值数据爬虫"""
    
    def __init__(self, log_level=logging.INFO):
        """
        初始化爬虫
        
        Args:
            log_level: 日志级别
        """
        self.setup_logging(log_level)
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        })
        
        # 数据存储
        self.data_cache = {}
        self.running = False
        self.update_interval = 3  # 默认3秒更新一次
        
    def setup_logging(self, level):
        """设置日志"""
        logging.basicConfig(
            level=level,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.StreamHandler(),
                logging.FileHandler('stock_crawler.log', encoding='utf-8')
            ]
        )
        self.logger = logging.getLogger(__name__)
        
    def fetch_sina_stock_data(self, code: str) -> Dict:
        """
        从新浪财经获取股票数据
        
        Args:
            code: 股票代码
            
        Returns:
            包含股票数据的字典
        """
        try:
            # 根据股票代码添加市场前缀
            if code.startswith('6'):
                full_code = f"sh{code}"
            elif code.startswith(('0', '3')):
                full_code = f"sz{code}"
            else:
                full_code = code
                
            url = f"http://hq.sinajs.cn/list={full_code}"
            response = self.session.get(url, timeout=5)
            response.encoding = 'gbk'
            
            if response.status_code == 200 and 'var hq_str_' in response.text:
                data_part = response.text.split('"')[1]
                if data_part:
                    data_list = data_part.split(',')
                    if len(data_list) >= 32:
                        current_price = float(data_list[3]) if data_list[3] else 0.0
                        yesterday_close = float(data_list[2]) if data_list[2] else 0.0
                        change = current_price - yesterday_close
                        change_percent = (change / yesterday_close * 100) if yesterday_close > 0 else 0.0
                        
                        return {
                            'code': code,
                            'name': data_list[0],
                            'current_price': current_price,
                            'yesterday_close': yesterday_close,
                            'today_open': float(data_list[1]) if data_list[1] else 0.0,
                            'today_high': float(data_list[4]) if data_list[4] else 0.0,
                            'today_low': float(data_list[5]) if data_list[5] else 0.0,
                            'volume': int(data_list[8]) if data_list[8] else 0,
                            'amount': float(data_list[9]) if data_list[9] else 0.0,
                            'change': change,
                            'change_percent': change_percent,
                            'timestamp': datetime.now(),
                            'source': 'sina'
                        }
        except Exception as e:
            self.logger.debug(f"新浪数据获取失败 {code}: {e}")
        return {}
        
    def fetch_eastmoney_fund_data(self, code: str) -> Dict:
        """
        从东方财富获取基金净值数据
        
        Args:
            code: 基金代码
            
        Returns:
            包含基金数据的字典
        """
        try:
            url = f"http://fundgz.1234567.com.cn/js/{code}.js"
            response = self.session.get(url, timeout=5)
            
            if response.status_code == 200:
                text = response.text.strip()
                  # 去掉JavaScript包装
                if text.startswith('jsonpgz(') and text.endswith(');'):
                    json_str = text[9:-2]
                elif text.startswith('jsonpgz(') and text.endswith(')'):
                    json_str = text[9:-1]
                else:
                    return {}
                
                if json_str and json_str != 'null':
                    # 修复可能的JSON格式问题
                    try:
                        data = json.loads(json_str)
                    except json.JSONDecodeError:
                        # 尝试修复常见的JSON格式问题
                        json_str = json_str.replace('{"', '{"').replace('"}', '"}')
                        if not json_str.startswith('{'):
                            json_str = '{' + json_str
                        if not json_str.endswith('}'):
                            json_str = json_str + '}'
                        try:
                            data = json.loads(json_str)
                        except json.JSONDecodeError as e:
                            self.logger.debug(f"JSON解析失败 {code}: {json_str[:100]}... 错误: {e}")
                            return {}
                    if data and isinstance(data, dict):
                        current_price = float(data.get('gsz', 0)) if data.get('gsz') else 0.0
                        yesterday_close = float(data.get('dwjz', 0)) if data.get('dwjz') else 0.0
                        change_percent = float(data.get('gszzl', 0)) if data.get('gszzl') else 0.0
                        change = current_price - yesterday_close if current_price > 0 and yesterday_close > 0 else 0.0
                        
                        return {
                            'code': code,
                            'name': data.get('name', ''),
                            'current_price': current_price,  # 估算净值
                            'yesterday_close': yesterday_close,  # 前日净值
                            'change': change,
                            'change_percent': change_percent,
                            'timestamp': datetime.now(),
                            'update_time': data.get('gztime', ''),
                            'source': 'eastmoney'
                        }
        except Exception as e:
            self.logger.debug(f"东方财富基金数据获取失败 {code}: {e}")
        return {}
        
    def fetch_tencent_stock_data(self, code: str) -> Dict:
        """
        从腾讯财经获取股票数据
        
        Args:
            code: 股票代码
            
        Returns:
            包含股票数据的字典
        """
        try:
            if code.startswith('6'):
                full_code = f"sh{code}"
            elif code.startswith(('0', '3')):
                full_code = f"sz{code}"
            else:
                full_code = code
                
            url = f"http://qt.gtimg.cn/q={full_code}"
            response = self.session.get(url, timeout=5)
            response.encoding = 'gbk'
            
            if response.status_code == 200 and '="' in response.text:
                data_str = response.text.split('="')[1].split('";')[0]
                data_list = data_str.split('~')
                
                if len(data_list) >= 47:
                    current_price = float(data_list[3]) if data_list[3] else 0.0
                    yesterday_close = float(data_list[4]) if data_list[4] else 0.0
                    change = current_price - yesterday_close
                    change_percent = (change / yesterday_close * 100) if yesterday_close > 0 else 0.0
                    
                    return {
                        'code': code,
                        'name': data_list[1],
                        'current_price': current_price,
                        'yesterday_close': yesterday_close,
                        'today_open': float(data_list[5]) if data_list[5] else 0.0,
                        'today_high': float(data_list[33]) if len(data_list) > 33 and data_list[33] else 0.0,
                        'today_low': float(data_list[34]) if len(data_list) > 34 and data_list[34] else 0.0,
                        'volume': int(data_list[6]) if data_list[6] else 0,
                        'amount': float(data_list[37]) if len(data_list) > 37 and data_list[37] else 0.0,
                        'change': change,
                        'change_percent': change_percent,
                        'timestamp': datetime.now(),
                        'source': 'tencent'
                    }        
        except Exception as e:
            self.logger.debug(f"腾讯数据获取失败 {code}: {e}")
        return {}
        
    def fetch_eastmoney_stock_data(self, code: str) -> Dict:
        """
        从东方财富获取股票数据（包括ETF）
        
        Args:
            code: 股票/ETF代码
            
        Returns:
            包含股票数据的字典
        """
        try:
            # 确定市场代码：沪市=1，深市=0
            if code.startswith('6') or code.startswith('51'):  # 沪市股票和ETF
                market = "1"
            elif code.startswith(('0', '3')):  # 深市股票
                market = "0"
            else:
                market = "1"  # 默认沪市
                
            url = f"http://push2.eastmoney.com/api/qt/stock/get?secid={market}.{code}&fields=f43,f44,f45,f46,f47,f48,f49,f50,f51,f52,f53,f54,f55,f56,f57,f58"
            response = self.session.get(url, timeout=5)
            
            if response.status_code == 200:
                data = response.json()
                if 'data' in data and data['data']:
                    stock_data = data['data']
                    
                    # 东方财富的价格字段需要除以100（分转元）
                    current_price = stock_data.get('f43', 0) / 100 if stock_data.get('f43') else 0.0
                    yesterday_close = stock_data.get('f44', 0) / 100 if stock_data.get('f44') else 0.0
                    change_percent = stock_data.get('f45', 0) / 100 if stock_data.get('f45') else 0.0
                    
                    return {
                        'code': code,
                        'name': stock_data.get('f58', ''),
                        'current_price': current_price,
                        'yesterday_close': yesterday_close,
                        'today_open': stock_data.get('f46', 0) / 100 if stock_data.get('f46') else 0.0,
                        'today_high': stock_data.get('f47', 0) / 100 if stock_data.get('f47') else 0.0,
                        'today_low': stock_data.get('f48', 0) / 100 if stock_data.get('f48') else 0.0,
                        'volume': stock_data.get('f49', 0),
                        'amount': stock_data.get('f50', 0),
                        'change': current_price - yesterday_close,
                        'change_percent': change_percent,
                        'timestamp': datetime.now(),
                        'source': 'eastmoney'
                    }
                    
        except Exception as e:
            self.logger.debug(f"东方财富股票数据获取失败 {code}: {e}")
            
        return {}
        
    def get_single_data(self, code: str, data_type: str = 'auto') -> Dict:
        """
        获取单个股票/基金的实时数据
        
        Args:
            code: 股票/基金代码
            data_type: 数据类型 ('auto', 'stock', 'fund')
            
        Returns:
            股票/基金数据字典
        """
        if data_type == 'auto':
            # 自动判断类型
            if len(code) == 6 and code.startswith(('00', '16', '51')):
                # 基金代码
                data = self.fetch_eastmoney_fund_data(code)
                if data:
                    return data
            
            # 股票代码，尝试多个数据源
            for fetch_func in [self.fetch_sina_stock_data, self.fetch_tencent_stock_data]:
                data = fetch_func(code)
                if data:
                    return data
                    
        elif data_type == 'fund':
            return self.fetch_eastmoney_fund_data(code)
        elif data_type == 'stock':
            # 优先使用新浪，备用腾讯
            data = self.fetch_sina_stock_data(code)
            if not data:
                data = self.fetch_tencent_stock_data(code)
            return data
        
        return {}
        
    def get_multiple_data(self, codes: List[str], data_type: str = 'auto', max_workers: int = 5) -> Dict[str, Dict]:
        """
        并发获取多个股票/基金的实时数据
        
        Args:
            codes: 股票/基金代码列表
            data_type: 数据类型
            max_workers: 最大并发数
            
        Returns:
            字典，键为代码，值为数据
        """
        results = {}
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_code = {
                executor.submit(self.get_single_data, code, data_type): code 
                for code in codes
            }
            
            for future in future_to_code:
                code = future_to_code[future]
                try:
                    data = future.result(timeout=10)
                    if data:
                        results[code] = data
                except Exception as e:
                    self.logger.debug(f"获取数据失败 {code}: {e}")
                    
        return results
        
    def start_monitoring(self, codes: List[str], callback=None, data_type: str = 'auto', interval: int = 3):
        """
        开始实时监控
        
        Args:
            codes: 要监控的代码列表
            callback: 数据更新回调函数 callback(data_dict)
            data_type: 数据类型
            interval: 更新间隔（秒）
        """
        self.running = True
        self.monitored_codes = codes
        self.update_interval = max(1, interval)
        
        def monitor_loop():
            while self.running:
                try:
                    # 获取最新数据
                    new_data = self.get_multiple_data(codes, data_type)
                    
                    # 更新缓存
                    for code, data in new_data.items():
                        if code not in self.data_cache:
                            self.data_cache[code] = []
                        self.data_cache[code].append(data)
                        
                        # 保持最近100条记录
                        if len(self.data_cache[code]) > 100:
                            self.data_cache[code] = self.data_cache[code][-100:]
                    
                    # 调用回调函数
                    if callback and new_data:
                        callback(new_data)
                        
                    time.sleep(self.update_interval)
                    
                except Exception as e:
                    self.logger.error(f"监控循环出错: {e}")
                    time.sleep(5)  # 出错后等待5秒再重试
                    
        # 在新线程中运行监控
        self.monitor_thread = threading.Thread(target=monitor_loop, daemon=True)
        self.monitor_thread.start()
        
        self.logger.info(f"开始实时监控 {len(codes)} 个标的，更新间隔 {self.update_interval} 秒")
        
    def stop_monitoring(self):
        """停止实时监控"""
        self.running = False
        if hasattr(self, 'monitor_thread'):
            self.monitor_thread.join(timeout=5)
        self.logger.info("停止实时监控")
        
    def get_latest_data(self, code: str) -> Dict:
        """获取指定代码的最新数据"""
        if code in self.data_cache and self.data_cache[code]:
            return self.data_cache[code][-1]
        return {}
        
    def get_historical_data(self, code: str, limit: int = 50) -> List[Dict]:
        """获取指定代码的历史数据"""
        if code in self.data_cache:
            return self.data_cache[code][-limit:]
        return []
        
    def export_to_dataframe(self, code: str) -> pd.DataFrame:
        """将缓存数据导出为DataFrame"""
        data_list = self.get_historical_data(code, limit=1000)
        if data_list:
            return pd.DataFrame(data_list)
        return pd.DataFrame()
        
    def export_to_csv(self, code: str, filename: str = None):
        """导出数据到CSV文件"""
        df = self.export_to_dataframe(code)
        if not df.empty:
            if filename is None:
                filename = f"{code}_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
            df.to_csv(filename, index=False, encoding='utf-8-sig')
            self.logger.info(f"数据已导出到: {filename}")
        else:
            self.logger.warning(f"没有找到 {code} 的数据")
            
    def print_data_summary(self, data: Dict):
        """打印数据摘要"""
        if not data:
            return
            
        print(f"\n{'='*60}")
        print(f"数据更新时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"{'='*60}")
        print(f"{'代码':<8} {'名称':<15} {'当前价':<10} {'涨跌额':<8} {'涨跌幅':<8} {'数据源'}")
        print(f"{'-'*60}")
        
        for code, stock_data in data.items():
            name = stock_data.get('name', 'N/A')[:15]
            current_price = stock_data.get('current_price', 0)
            change = stock_data.get('change', 0)
            change_percent = stock_data.get('change_percent', 0)
            source = stock_data.get('source', 'N/A')
            
            print(f"{code:<8} {name:<15} {current_price:<10.3f} {change:<+8.3f} {change_percent:<+7.2f}% {source}")


# 使用示例和测试函数
def demo_single_stock():
    """演示获取单个股票数据"""
    print("演示：获取单个股票数据")
    print("="*40)
    
    crawler = StockNetValueCrawler()
    
    # 获取平安银行数据
    stock_data = crawler.get_single_data('000001', 'stock')
    if stock_data:
        print(f"股票名称: {stock_data['name']}")
        print(f"股票代码: {stock_data['code']}")
        print(f"当前价格: {stock_data['current_price']:.3f} 元")
        print(f"涨跌幅度: {stock_data['change_percent']:+.2f}%")
        print(f"成交量: {stock_data.get('volume', 0):,} 手")
        print(f"数据源: {stock_data['source']}")
    else:
        print("获取股票数据失败")


def demo_single_fund():
    """演示获取单个基金数据"""
    print("\n演示：获取单个基金数据")
    print("="*40)
    
    crawler = StockNetValueCrawler()
    
    # 获取基金数据
    fund_data = crawler.get_single_data('008087', 'fund')
    if fund_data:
        print(f"基金名称: {fund_data['name']}")
        print(f"基金代码: {fund_data['code']}")
        print(f"当前净值: {fund_data['current_price']:.4f}")
        print(f"涨跌幅度: {fund_data['change_percent']:+.2f}%")
        print(f"更新时间: {fund_data.get('update_time', 'N/A')}")
        print(f"数据源: {fund_data['source']}")
    else:
        print("获取基金数据失败")


def demo_multiple_data():
    """演示批量获取数据"""
    print("\n演示：批量获取多个数据")
    print("="*40)
    
    crawler = StockNetValueCrawler()
    
    # 混合获取股票和基金数据
    codes = ['000001', '000002', '008087', '008299']
    batch_data = crawler.get_multiple_data(codes)
    
    if batch_data:
        crawler.print_data_summary(batch_data)
    else:
        print("批量获取数据失败")


def demo_realtime_monitoring():
    """演示实时监控"""
    print("\n演示：实时监控（运行15秒）")
    print("="*40)
    
    crawler = StockNetValueCrawler()
    
    # 要监控的代码
    codes = ['000001', '000002', '008087']
    
    # 定义回调函数
    def on_data_update(data):
        crawler.print_data_summary(data)
    
    # 开始监控
    crawler.start_monitoring(codes, callback=on_data_update, interval=5)
    
    try:
        print("监控中... (按 Ctrl+C 提前结束)")
        time.sleep(15)  # 运行15秒
    except KeyboardInterrupt:
        print("\n用户手动停止监控")
    finally:
        crawler.stop_monitoring()
        
        # 展示收集到的数据统计
        print("\n数据收集统计:")
        for code in codes:
            history = crawler.get_historical_data(code)
            if history:
                print(f"{code}: 收集了 {len(history)} 条数据")


def main():
    """主函数 - 运行所有演示"""
    print("实时股票净值数据爬虫")
    print("作者: Rick Xie")
    print("日期:", datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
    print("\n" + "="*60)
    
    try:
        # 运行各个演示
        demo_single_stock()
        demo_single_fund()
        demo_multiple_data()
        
        # 询问是否运行实时监控
        print("\n" + "="*60)
        user_input = input("是否运行实时监控演示？(y/n): ").lower().strip()
        if user_input in ['y', 'yes', '是', '1']:
            demo_realtime_monitoring()
        
    except Exception as e:
        print(f"程序运行出错: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "="*60)
    print("演示完成!")


if __name__ == "__main__":
    main()

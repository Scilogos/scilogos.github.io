#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Python量化回测系统 v2.0
编程理念：模块政治、思想防错、民众防错、政治防错、武力防错

v2.0大修(2026-06-18):
[P0] 1. train_and_predict异常返回值3->5元素
[P0] 2. 新增baostock_session上下文管理器
[P0] 3. fuse_predictions valid标记改为独立key
[P0] 4. fetch_single_stock_data添加mock降级路径
[P1] 5. 牛熊标签阈值改用分位数
[P1] 6. RSI计算修复为标准实现
[NEW] 7. 元叙事市场模拟骨架

v1.2新增内容：
1. 推荐持仓从4只增加到8只
2. baostock批量login优化（减少200+次握手）
3. 新增当天实时价格获取功能

v1.1修复内容：
1. 统一量纲体系：全链路收益率统一用小数，仅最终输出时转百分比
2. 股票池修复：移除紫光国微从交通行业、移除创业板股票（宁德时代、胜宏科技、智飞生物）
3. 风控模块强化：自动修正违规仓位、接入回撤检查、使用实际入场价计算止盈止损
4. 组合优化完善：调用风险平价优化、添加迭代约束调整
5. 数据泄漏修复：添加时间序列分割确保训练集不泄漏

v1.6更新内容：
1. 新增阶段0：市场牛熊环境预判（LightGBM + 贝叶斯融合，7板块+市场整体）
2. 删除分钟级预测模块（被牛熊预判替代）
3. 牛熊信号→风控：熊市收紧回撤红线至70%，牛市放宽至130%
4. 牛熊信号→组合：熊市整体减仓至70%+看空板块额外减半，牛市看多板块加仓15%
5. 控制台输出全面美化：统一分隔线（━/─）、模块标签精简、表格对齐
"""

# ==========================================
# 模块1：配置与常量定义模块 (Config Module)
# 设计意图：统一管理所有全局配置、依赖检测和降级策略
# 模块规模：约150行
# ==========================================

import os
import sys
import warnings

warnings.filterwarnings('ignore')

# --------------------------
# 依赖检测与降级策略 (政治防错)
# 设计意图：自动检测依赖库可用性，确保"运行就是一切"
# --------------------------
print("━" * 70)

DEPENDENCY_STATUS = {
    'baostock': False,
    'lightgbm': False,
    'arch': False,
    'sklearn': False,
    'pandas': True,
    'numpy': True,
    'scipy': True,
    'openpyxl': True,
    'matplotlib': False,
    'mootdx': False
}

try:
    import baostock as bs

    DEPENDENCY_STATUS['baostock'] = True
    print("  依赖检测")
    print("  ────────")
    print("  ✓ baostock")
except ImportError:
    print("  依赖检测")
    print("  ────────")
    print("  ✗ baostock（模拟数据）")

try:
    import lightgbm as lgb

    DEPENDENCY_STATUS['lightgbm'] = True
    print("  ✓ lightgbm")
except ImportError:
    print("  ✗ lightgbm（sklearn集成）")

try:
    from arch import arch_model

    DEPENDENCY_STATUS['arch'] = True
    print("  ✓ arch")
except ImportError:
    print("  ✗ arch（numpy实现）")

try:
    from sklearn.ensemble import GradientBoostingRegressor
    from sklearn.preprocessing import StandardScaler

    DEPENDENCY_STATUS['sklearn'] = True
    print("  ✓ sklearn")
except ImportError:
    print("  ✗ sklearn（基础统计）")

try:
    import matplotlib

    matplotlib.use('Agg')  # 使用非GUI后端
    import matplotlib.pyplot as plt

    DEPENDENCY_STATUS['matplotlib'] = True
    print("  ✓ matplotlib")
except ImportError:
    print("  ✗ matplotlib（TXT格式）")

try:
    from mootdx.quotes import Quotes

    DEPENDENCY_STATUS['mootdx'] = True
    print("  ✓ mootdx")
except ImportError:
    DEPENDENCY_STATUS['mootdx'] = False
    print("  ✗ mootdx（实时数据将降级使用baostock）")

# --------------------------
# 核心常量定义 (政治防错)
# 设计意图：所有全局配置集中管理，便于维护和修改
# --------------------------
import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import minimize
import datetime
import time
import random
from datetime import timedelta

# 设置随机种子确保可复现
np.random.seed(42)
random.seed(42)



# ==========================================
# 模块1.5：mootdx实时数据适配器 (MootdxAdapter)
# 设计意图：用mootdx替代baostock获取实时行情，解决baostock无实时数据的痛点
# 接口兼容：输出与原fetch_realtime_price()一致，下游代码零改动
# ==========================================

class MootdxAdapter:
    """
    mootdx实时数据适配器
    核心职责：将mootdx的API包装成量化脚本能直接调用的形式
    代码格式转换：baostock格式(sh.601857) ↔ mootdx格式(601857)
    生命周期：全局单例，阶段1前连接，全程复用，程序结束时关闭
    """

    def __init__(self):
        self.client = None
        self._connected = False
        self._price_cache = {}      # 批量实时价格缓存 {code_std: price}
        self._cache_time = None     # 缓存时间戳
        self._cache_ttl = 30        # 缓存有效期（秒）

    def connect(self):
        """连接mootdx服务器，bestip优先，失败降级"""
        if not DEPENDENCY_STATUS.get('mootdx', False):
            print("  ⚠ mootdx未安装，实时数据将降级使用baostock")
            return False

        try:
            self.client = Quotes.factory(market='std', bestip=True, timeout=15)
            self._connected = True
            print("  ✓ mootdx实时行情连接成功")
            return True
        except Exception as e:
            print(f"  ⚠ mootdx bestip连接失败: {e}")

        # bestip失败，尝试默认服务器
        try:
            self.client = Quotes.factory(market='std', bestip=False, timeout=15)
            self._connected = True
            print("  ✓ mootdx默认服务器连接成功")
            return True
        except Exception as e:
            print(f"  ✗ mootdx连接全部失败: {e}")
            self._connected = False
            return False

    def close(self):
        """关闭连接"""
        if self.client is not None:
            try:
                self.client.close()
            except Exception:
                pass
            self._connected = False
            self.client = None

    @staticmethod
    def _baostock_to_mootdx(code):
        """
        baostock代码 → mootdx纯数字代码
        'sh.601857' → '601857'
        'sz.000858' → '000858'
        '601857.SH' → '601857'
        '000858.SZ' → '000858'
        '601857'    → '601857' (已经是纯数字)
        """
        code = str(code).strip()
        # 去掉 sh./sz. 前缀
        if '.' in code and len(code.split('.')) == 2:
            parts = code.split('.')
            # sh.601857 或 601857.SH 格式
            if parts[0] in ('sh', 'sz'):
                return parts[1]
            elif parts[1] in ('SH', 'SZ'):
                return parts[0]
        return code

    @staticmethod
    def _mootdx_to_baostock(code):
        """
        mootdx纯数字代码 → baostock代码
        '601857' → 'sh.601857'  (6开头=沪)
        '000858' → 'sz.000858'  (0开头=深)
        '300059' → 'sz.300059'  (3开头=深)
        """
        code = str(code).strip()
        if code.startswith(('6', '9')):
            return f'sh.{code}'
        else:
            return f'sz.{code}'

    @staticmethod
    def _mootdx_to_code_std(code):
        """
        mootdx纯数字代码 → code_std格式
        '601857' → '601857.SH'
        '000858' → '000858.SZ'
        """
        code = str(code).strip()
        if code.startswith(('6', '9')):
            return f'{code}.SH'
        else:
            return f'{code}.SZ'

    def get_realtime_price(self, code):
        """
        获取单只股票实时价格
        输入：code - baostock格式(如'sh.601857')或code_std格式(如'601857.SH')或纯数字
        输出：实时价格(float)或None
        """
        if not self._connected or self.client is None:
            return None

        mootdx_code = self._baostock_to_mootdx(code)
        try:
            df = self.client.quotes(symbol=mootdx_code)
            if df is not None and not df.empty:
                price = float(df.iloc[0]['price'])
                if price > 0:
                    return price
        except Exception:
            pass
        return None

    def get_realtime_quotes_batch(self, code_list):
        """
        批量获取实时行情（一次网络请求拿多只股票）
        输入：code_list - 代码列表，支持混合格式 ['sh.601857', '000858.SZ', '600036']
        输出：{原始code: {'price': float, 'open': float, 'high': float, 'low': float,
                        'last_close': float, 'vol': int, 'amount': float,
                        'change_pct': float, 'bid1': float, 'ask1': float}}
        """
        if not self._connected or self.client is None:
            return {}

        # 转换代码格式，同时记住映射关系
        mootdx_codes = []
        code_map = {}  # mootdx_code → original_code
        for code in code_list:
            mc = self._baostock_to_mootdx(code)
            mootdx_codes.append(mc)
            code_map[mc] = code

        try:
            df = self.client.quotes(symbol=mootdx_codes)
            if df is None or df.empty:
                return {}

            result = {}
            for _, row in df.iterrows():
                mc = str(row['code'])
                orig_code = code_map.get(mc, mc)
                last_close = float(row['last_close']) if pd.notna(row['last_close']) else 0
                price = float(row['price']) if pd.notna(row['price']) else 0
                change_pct = ((price - last_close) / last_close * 100) if last_close > 0 else 0
                result[orig_code] = {
                    'price': price,
                    'open': float(row['open']) if pd.notna(row['open']) else 0,
                    'high': float(row['high']) if pd.notna(row['high']) else 0,
                    'low': float(row['low']) if pd.notna(row['low']) else 0,
                    'last_close': last_close,
                    'vol': int(row['vol']) if pd.notna(row['vol']) else 0,
                    'amount': float(row['amount']) if pd.notna(row['amount']) else 0,
                    'change_pct': round(change_pct, 2),
                    'bid1': float(row['bid1']) if pd.notna(row['bid1']) else 0,
                    'ask1': float(row['ask1']) if pd.notna(row['ask1']) else 0,
                }
            return result
        except Exception:
            return {}

    def get_realtime_prices_dict(self, code_list):
        """
        批量获取实时价格，返回 {code_std: price} 字典
        带缓存，30秒内重复调用直接返回缓存
        """
        now = time.time()
        if self._cache_time and (now - self._cache_time) < self._cache_ttl and self._price_cache:
            return self._price_cache

        quotes = self.get_realtime_quotes_batch(code_list)
        self._price_cache = {}
        for code, info in quotes.items():
            code_std = self._baostock_to_mootdx(code)
            code_std = self._mootdx_to_code_std(code_std)
            self._price_cache[code_std] = info['price']

        self._cache_time = now
        return self._price_cache

    def get_index_realtime(self, index_code='000001'):
        """
        获取指数实时数据
        输入：index_code - 纯数字代码（'000001'=上证, '399001'=深证, '399006'=创业板）
        输出：{'close': float, 'open': float, 'high': float, 'low': float, 'vol': int} 或 None
        """
        if not self._connected or self.client is None:
            return None

        try:
            df = self.client.index(symbol=index_code, frequency=9, offset=1)
            if df is not None and not df.empty:
                row = df.iloc[-1]
                return {
                    'close': float(row['close']),
                    'open': float(row['open']),
                    'high': float(row['high']),
                    'low': float(row['low']),
                    'vol': int(row['vol']) if pd.notna(row['vol']) else 0,
                }
        except Exception:
            pass
        return None

    def get_bars(self, code, frequency=9, offset=100):
        """
        获取K线数据（备用，baostock数据不足时补齐）
        frequency: 0=5分钟, 1=15分钟, 2=30分钟, 3=1小时, 4/9=日K, 5=周K, 6=月K
        输出：DataFrame(columns=open,close,high,low,vol,amount,datetime) 或 None
        """
        if not self._connected or self.client is None:
            return None

        mootdx_code = self._baostock_to_mootdx(code)
        try:
            df = self.client.bars(symbol=mootdx_code, frequency=frequency, offset=min(offset, 800))
            if df is not None and not df.empty:
                return df
        except Exception:
            pass
        return None

    def get_sector_realtime_summary(self, sector_stock_codes):
        """
        获取板块实时涨跌概况
        输入：sector_stock_codes - 该板块所有股票的code列表 ['sh.601857', ...]
        输出：{'up': int, 'down': int, 'flat': int, 'avg_change_pct': float,
               'leading': (code, change_pct), 'lagging': (code, change_pct)}
        """
        quotes = self.get_realtime_quotes_batch(sector_stock_codes)
        if not quotes:
            return None

        up, down, flat = 0, 0, 0
        changes = []
        for code, info in quotes.items():
            pct = info.get('change_pct', 0)
            changes.append((code, pct))
            if pct > 0.5:
                up += 1
            elif pct < -0.5:
                down += 1
            else:
                flat += 1

        changes.sort(key=lambda x: x[1], reverse=True)
        avg_pct = sum(c[1] for c in changes) / len(changes) if changes else 0

        return {
            'up': up, 'down': down, 'flat': flat,
            'avg_change_pct': round(avg_pct, 2),
            'leading': changes[0] if changes else (None, 0),
            'lagging': changes[-1] if changes else (None, 0),
            'total': len(changes),
        }


# 全局单例
MOOTDX_CLIENT = MootdxAdapter()

# 股票池：10个行业，各行业股票（已修复：移除创业板股票和重复的紫光国微）
STOCK_POOL = {
    '能源': [
        {'code': 'sh.601857', 'name': '中国石油', 'code_std': '601857.SH'},
        {'code': 'sh.600028', 'name': '中国石化', 'code_std': '600028.SH'},
        {'code': 'sh.601088', 'name': '中国神华', 'code_std': '601088.SH'},
        {'code': 'sh.600989', 'name': '宝丰能源', 'code_std': '600989.SH'},
        {'code': 'sz.000554', 'name': '泰山石油', 'code_std': '000554.SZ'},
        {'code': 'sh.600387', 'name': '海越能源', 'code_std': '600387.SH'},
        {'code': 'sz.000096', 'name': '广聚能源', 'code_std': '000096.SZ'},
        {'code': 'sh.600688', 'name': '上海石化', 'code_std': '600688.SH'},
        {'code': 'sh.601918', 'name': '新集能源', 'code_std': '601918.SH'},
        {'code': 'sz.000937', 'name': '冀中能源', 'code_std': '000937.SZ'}
    ],
    '金属': [
        {'code': 'sh.600547', 'name': '山东黄金', 'code_std': '600547.SH'},
        {'code': 'sz.002428', 'name': '云南锗业', 'code_std': '002428.SZ'},
        {'code': 'sh.601600', 'name': '中国铝业', 'code_std': '601600.SH'},
        {'code': 'sz.000962', 'name': '东方钽业', 'code_std': '000962.SZ'},
        {'code': 'sh.603993', 'name': '洛阳钼业', 'code_std': '603993.SH'},
        {'code': 'sh.600489', 'name': '中金黄金', 'code_std': '600489.SH'},
        {'code': 'sz.002155', 'name': '湖南黄金', 'code_std': '002155.SZ'},
        {'code': 'sh.601899', 'name': '紫金矿业', 'code_std': '601899.SH'},
        {'code': 'sz.000630', 'name': '铜陵有色', 'code_std': '000630.SZ'},
        {'code': 'sh.600259', 'name': '广晟有色', 'code_std': '600259.SH'},
        {'code': 'sz.000807', 'name': '云铝股份', 'code_std': '000807.SZ'}
    ],
    '金融': [
        {'code': 'sh.601398', 'name': '工商银行', 'code_std': '601398.SH'},
        {'code': 'sh.601318', 'name': '中国平安', 'code_std': '601318.SH'},
        {'code': 'sh.600036', 'name': '招商银行', 'code_std': '600036.SH'},
        {'code': 'sh.601988', 'name': '中国银行', 'code_std': '601988.SH'},
        {'code': 'sh.601689', 'name': '拓普集团', 'code_std': '601689.SH'},
        {'code': 'sh.601939', 'name': '建设银行', 'code_std': '601939.SH'},
        {'code': 'sh.600016', 'name': '民生银行', 'code_std': '600016.SH'},
        {'code': 'sz.000001', 'name': '平安银行', 'code_std': '000001.SZ'},
        {'code': 'sh.601628', 'name': '中国人寿', 'code_std': '601628.SH'},
        {'code': 'sh.601818', 'name': '光大银行', 'code_std': '601818.SH'},
        {'code': 'sz.002736', 'name': '国信证券', 'code_std': '002736.SZ'}
    ],
    '消费': [
        {'code': 'sz.000858', 'name': '五粮液', 'code_std': '000858.SZ'},
        {'code': 'sh.600519', 'name': '贵州茅台', 'code_std': '600519.SH'},
        {'code': 'sz.000333', 'name': '美的集团', 'code_std': '000333.SZ'},
        {'code': 'sz.002508', 'name': '老板电器', 'code_std': '002508.SZ'},
        {'code': 'sh.600887', 'name': '伊利股份', 'code_std': '600887.SH'},
        {'code': 'sz.002304', 'name': '洋河股份', 'code_std': '002304.SZ'},
        {'code': 'sh.600809', 'name': '山西汾酒', 'code_std': '600809.SH'},
        {'code': 'sz.000895', 'name': '双汇发展', 'code_std': '000895.SZ'},
        {'code': 'sz.002262', 'name': '恩华药业', 'code_std': '002262.SZ'},
        {'code': 'sh.603868', 'name': '飞科电器', 'code_std': '603868.SH'},
        {'code': 'sz.002419', 'name': '天虹股份', 'code_std': '002419.SZ'}
    ],
    # 【】半导体行业（从科技拆出，大幅扩充至15只+）
    '半导体': [
        # 原有半导体股票（7只）
        {'code': 'sz.002371', 'name': '北方华创', 'code_std': '002371.SZ'},
        {'code': 'sh.603986', 'name': '兆易创新', 'code_std': '603986.SH'},
        {'code': 'sh.600460', 'name': '士兰微', 'code_std': '600460.SH'},
        {'code': 'sh.603005', 'name': '晶方科技', 'code_std': '603005.SH'},
        {'code': 'sh.600584', 'name': '长电科技', 'code_std': '600584.SH'},
        {'code': 'sh.603501', 'name': '韦尔股份', 'code_std': '603501.SH'},
        {'code': 'sz.002185', 'name': '华天科技', 'code_std': '002185.SZ'},
        # 新增半导体股票（主板：8只）
        {'code': 'sh.600745', 'name': '闻泰科技', 'code_std': '600745.SH'},
        {'code': 'sh.600360', 'name': '华微电子', 'code_std': '600360.SH'},
        {'code': 'sh.600171', 'name': '上海贝岭', 'code_std': '600171.SH'},
        {'code': 'sh.600198', 'name': '大唐电信', 'code_std': '600198.SH'},
        {'code': 'sh.600877', 'name': '电科芯片', 'code_std': '600877.SH'},
        {'code': 'sh.603160', 'name': '汇顶科技', 'code_std': '603160.SH'},
        {'code': 'sh.603893', 'name': '瑞芯微', 'code_std': '603893.SH'},
        {'code': 'sh.603290', 'name': '斯达半导', 'code_std': '603290.SH'},
        {'code': 'sz.000021', 'name': '深科技', 'code_std': '000021.SZ'},
    ],
    # 【】人工智能行业（从科技拆出，扩充至12只+）
    '人工智能': [
        # 原有AI股票（8只）
        {'code': 'sz.002230', 'name': '科大讯飞', 'code_std': '002230.SZ'},
        {'code': 'sh.601360', 'name': '三六零', 'code_std': '601360.SH'},
        {'code': 'sh.603019', 'name': '中科曙光', 'code_std': '603019.SH'},
        {'code': 'sz.000977', 'name': '浪潮信息', 'code_std': '000977.SZ'},
        {'code': 'sh.601138', 'name': '工业富联', 'code_std': '601138.SH'},
        {'code': 'sz.000938', 'name': '紫光股份', 'code_std': '000938.SZ'},
        {'code': 'sz.002236', 'name': '大华股份', 'code_std': '002236.SZ'},
        {'code': 'sz.000997', 'name': '新大陆', 'code_std': '000997.SZ'},
        # 新增AI股票（主板：6只）
        {'code': 'sh.600845', 'name': '宝信软件', 'code_std': '600845.SH'},
        {'code': 'sh.600588', 'name': '用友网络', 'code_std': '600588.SH'},
        {'code': 'sz.002415', 'name': '海康威视', 'code_std': '002415.SZ'},
        {'code': 'sh.600446', 'name': '金证股份', 'code_std': '600446.SH'},
        {'code': 'sh.600633', 'name': '浙数文化', 'code_std': '600633.SH'},
        {'code': 'sh.603000', 'name': '人民网', 'code_std': '603000.SH'},
    ],
    # 【v1.4精简】通用科技行业（移除已归入半导体和AI的股票）
    '科技': [
        # 通信与算力赛道（5只）
        {'code': 'sh.600941', 'name': '中国移动', 'code_std': '600941.SH'},
        {'code': 'sh.601728', 'name': '中国电信', 'code_std': '601728.SH'},
        {'code': 'sh.600498', 'name': '烽火通信', 'code_std': '600498.SH'},
        # 消费电子与面板（3只）
        {'code': 'sz.002594', 'name': '比亚迪', 'code_std': '002594.SZ'},
        {'code': 'sz.000725', 'name': '京东方A', 'code_std': '000725.SZ'},
        {'code': 'sz.002475', 'name': '立讯精密', 'code_std': '002475.SZ'},
        # 其他科技（5只）
        {'code': 'sh.600271', 'name': '航天信息', 'code_std': '600271.SH'},
        {'code': 'sh.600776', 'name': '东方通信', 'code_std': '600776.SH'},
        {'code': 'sh.600100', 'name': '同方股份', 'code_std': '600100.SH'},
        {'code': 'sz.002384', 'name': '东山精密', 'code_std': '002384.SZ'},
        {'code': 'sz.002156', 'name': '通富微电', 'code_std': '002156.SZ'}
    ],
    '医药': [
        # 已删除创业板股票：智飞生物(300122)
        {'code': 'sh.600276', 'name': '恒瑞医药', 'code_std': '600276.SH'},
        {'code': 'sz.000661', 'name': '长春高新', 'code_std': '000661.SZ'},
        {'code': 'sz.002252', 'name': '上海莱士', 'code_std': '002252.SZ'},
        {'code': 'sh.600867', 'name': '通化东宝', 'code_std': '600867.SH'},
        {'code': 'sz.002007', 'name': '华兰生物', 'code_std': '002007.SZ'},
        {'code': 'sh.600196', 'name': '复星医药', 'code_std': '600196.SH'},
        {'code': 'sh.600332', 'name': '白云山', 'code_std': '600332.SH'},
        {'code': 'sz.002422', 'name': '科伦药业', 'code_std': '002422.SZ'},
        {'code': 'sh.600521', 'name': '华海药业', 'code_std': '600521.SH'},
        {'code': 'sz.002603', 'name': '以岭药业', 'code_std': '002603.SZ'}
    ],
    '制造': [
        {'code': 'sh.601766', 'name': '中国中车', 'code_std': '601766.SH'},
        {'code': 'sh.600031', 'name': '三一重工', 'code_std': '600031.SH'},
        {'code': 'sz.000157', 'name': '中联重科', 'code_std': '000157.SZ'},
        {'code': 'sh.600811', 'name': '东方集团', 'code_std': '600811.SH'},
        {'code': 'sz.000425', 'name': '徐工机械', 'code_std': '000425.SZ'},
        {'code': 'sh.601100', 'name': '恒立液压', 'code_std': '601100.SH'},
        {'code': 'sz.002097', 'name': '山河智能', 'code_std': '002097.SZ'},
        {'code': 'sh.600320', 'name': '振华重工', 'code_std': '600320.SH'},
        {'code': 'sz.002531', 'name': '天顺风能', 'code_std': '002531.SZ'},
        {'code': 'sh.600495', 'name': '晋西车轴', 'code_std': '600495.SH'},
        {'code': 'sz.002204', 'name': '大连重工', 'code_std': '002204.SZ'}
    ],
    '地产': [
        {'code': 'sh.600048', 'name': '保利发展', 'code_std': '600048.SH'},
        {'code': 'sz.000002', 'name': '万科A', 'code_std': '000002.SZ'},
        {'code': 'sh.600383', 'name': '金地集团', 'code_std': '600383.SH'},
        {'code': 'sh.601155', 'name': '新城控股', 'code_std': '601155.SH'},
        {'code': 'sz.000069', 'name': '华侨城A', 'code_std': '000069.SZ'},
        {'code': 'sh.600606', 'name': '绿地控股', 'code_std': '600606.SH'},
        {'code': 'sz.000961', 'name': '中南建设', 'code_std': '000961.SZ'},
        {'code': 'sh.600325', 'name': '华发股份', 'code_std': '600325.SH'},
        {'code': 'sh.600376', 'name': '首开股份', 'code_std': '600376.SH'},
        {'code': 'sh.600743', 'name': '华远地产', 'code_std': '600743.SH'},
        {'code': 'sz.000897', 'name': '津滨发展', 'code_std': '000897.SZ'}
    ],
    '交通': [
        # 已删除紫光国微(sz.002049)，该股票已在科技行业存在
        {'code': 'sh.601111', 'name': '中国国航', 'code_std': '601111.SH'},
        {'code': 'sh.600029', 'name': '南方航空', 'code_std': '600029.SH'},
        {'code': 'sh.601006', 'name': '大秦铁路', 'code_std': '601006.SH'},
        {'code': 'sh.600115', 'name': '东方航空', 'code_std': '600115.SH'},
        {'code': 'sh.601333', 'name': '广深铁路', 'code_std': '601333.SH'},
        {'code': 'sz.000089', 'name': '深圳机场', 'code_std': '000089.SZ'},
        {'code': 'sh.600018', 'name': '上港集团', 'code_std': '600018.SH'},
        {'code': 'sh.601866', 'name': '中远海发', 'code_std': '601866.SH'},
        {'code': 'sz.000905', 'name': '厦门港务', 'code_std': '000905.SZ'},
        {'code': 'sh.600798', 'name': '宁波海运', 'code_std': '600798.SH'}
    ],
    '化工': [
        {'code': 'sh.600309', 'name': '万华化学', 'code_std': '600309.SH'},
        {'code': 'sz.002493', 'name': '荣盛石化', 'code_std': '002493.SZ'},
        {'code': 'sh.601233', 'name': '桐昆股份', 'code_std': '601233.SH'},
        {'code': 'sh.600143', 'name': '金发科技', 'code_std': '600143.SH'},
        {'code': 'sz.000707', 'name': '双环科技', 'code_std': '000707.SZ'},
        {'code': 'sh.600409', 'name': '三友化工', 'code_std': '600409.SH'},
        {'code': 'sz.002092', 'name': '中泰化学', 'code_std': '002092.SZ'},
        {'code': 'sh.600299', 'name': '安迪苏', 'code_std': '600299.SH'},
        {'code': 'sz.002648', 'name': '卫星化学', 'code_std': '002648.SZ'},
        {'code': 'sh.600810', 'name': '神马股份', 'code_std': '600810.SH'},
        {'code': 'sz.000698', 'name': '沈阳化工', 'code_std': '000698.SZ'}
    ]
}

# 指数池
INDEX_LIST = [
    {'code': 'sh.000001', 'name': '上证指数', 'code_std': '000001.SH'},
    {'code': 'sz.399001', 'name': '深证成指', 'code_std': '399001.SZ'},
    {'code': 'sh.000300', 'name': '沪深300', 'code_std': '000300.SH'}
]

# 资金配置参数
INITIAL_CAPITAL = 100000  # 初始资金10万元
MAIN_CAPITAL = 80000  # 主仓8万元（50%长线+50%短线）
BOTTOM_CAPITAL = 20000  # 抄底仓2万元
MAX_POSITION_PER_STOCK = 0.20  # v1.7：单票上限降至20%，分散风险
MAX_POSITION_PER_INDUSTRY = 0.30  # 单一行业最大仓位30%
MAX_DRAWDOWN = 0.10  # 账户最大回撤10%
HOLDING_COUNT_RANGE = (0, 8)  # v1.7：弹性持仓，信号不达标可以0只

# 输出路径配置（自动处理Windows/Linux路径）
WINDOWS_OUTPUT_PATH = r'C:\Users\HUAWEI\Desktop\AA同花顺'
OUTPUT_PATH = WINDOWS_OUTPUT_PATH if os.name == 'nt' else os.path.join(os.getcwd(), 'output')

# 确保输出目录存在
os.makedirs(OUTPUT_PATH, exist_ok=True)

# 回测参数
TRAINING_DAYS = 250  # 训练数据：最近1年（250个交易日）
PREDICTION_HORIZONS = [1, 2, 3, 4, 5]  # 预测1-5日收益率
MIN_DATA_DAYS = 100  # 最小数据量要求

# 统计股票数量
total_stocks = sum(len(stocks) for stocks in STOCK_POOL.values())
total_industries = len(STOCK_POOL)

# ==========================================
# 牛熊预测模块（阶段0：市场环境预判）
# 来源：牛熊预测系统 v1.4，融合到主脚本
# 功能：在个股选股前判断市场牛熊环境，结果供风控和组合优化模块使用
# ==========================================


# ---------- 牛熊预测模块依赖 ----------
try:
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
except ImportError:
    pass
try:
    from scipy.stats import jarque_bera
except ImportError:
    pass
try:
    from statsmodels.tsa.arima.model import ARIMA
except ImportError:
    pass
try:
    from collections import defaultdict
except ImportError:
    pass
try:
    from typing import Dict, List, Tuple, Optional
except ImportError:
    pass
# matplotlib初始化（牛熊模块独立管理，不依赖主脚本try块内的局部变量）
try:
    import matplotlib

    matplotlib.use('Agg')
    import matplotlib.pyplot as plt_bb
    import matplotlib.font_manager as fm_bb
    from matplotlib.backends.backend_pdf import PdfPages as PdfPages_bb

    # 中文字体设置
    _bb_chinese_fonts = ['SimHei', 'Microsoft YaHei', 'STHeiti', 'WenQuanYi Micro Hei', 'Noto Sans CJK SC']
    _bb_available = {f.name for f in fm_bb.fontManager.ttflist}
    for _fn in _bb_chinese_fonts:
        if _fn in _bb_available:
            plt_bb.rcParams['font.sans-serif'] = [_fn, 'DejaVu Sans']
            break
    plt_bb.rcParams['axes.unicode_minus'] = False
except ImportError:
    plt_bb = None
    PdfPages_bb = None
    fm_bb = None

# ---------- 牛熊预测配置 ----------
VERBOSE_BB = False  # 牛熊预测详细输出开关


def _bb_print(*args, **kwargs):
    """牛熊预测可控输出"""
    if VERBOSE_BB:
        print(*args, **kwargs)


TRAIN_DAYS_BB = 750
PRED_HORIZONS_BB = [1, 2, 3, 4, 5]

INDEX_CONFIG_BB = {
    '市场整体': ['sh.000001', 'sz.399001', 'sh.000300'],
}

SECTOR_CONFIG_BB = {
    '能源': 'sh.000928',
    '金属': 'sh.000819',
    '金融': 'sh.000934',
    '消费': 'sh.000932',
    '科技': 'sz.399673',
    '医药': 'sh.000933',
    '制造': 'sh.000903',
}

ALL_SECTORS_BB = list(SECTOR_CONFIG_BB.keys())

# 牛熊预测结果全局变量（供风控和组合优化模块读取）
MARKET_ENV = {
    'market_regime': '震荡',  # '牛市' / '震荡' / '熊市'
    'bull_prob': 0.33,
    'bear_prob': 0.33,
    'neutral_prob': 0.34,
    'confidence': 0.0,  # 信号置信度（牛熊概率差值）
    'sector_signals': {},  # 各板块信号
    'available': False,  # 牛熊预测是否成功执行
    # 【v1.8】状态推演字段
    'regime_duration': 0,           # 当前状态预计持续天数
    'regime_transition_to': '震荡',  # 最可能转折方向
    'regime_transition_prob': 0,     # 转折概率
    'regime_narrative': '',          # 自然语言推演
    'regime_risk_hint': '',          # 风险提示
    'sector_regime_analysis': {},    # 各板块推演详情
}


class BB_DataFetcher:
    """
    数据获取模块
    功能：从baostock获取三大指数和10个行业板块的历史数据
    新增：双重备份机制（已移除，仅使用官方指数）
    【修复17】重命名fetch_single_index为fetch_single_data
    【修复2】备用行业指数使用真实OHLCV数据计算
    【v1.3.1】baostock连接稳定性：请求重试、限速、断连重登
    """

    # 请求间隔（秒），避免被服务器踢
    REQUEST_INTERVAL = 0.5
    # 单次请求最大重试次数
    MAX_RETRIES = 3
    # 重试间休眠基数（秒）
    RETRY_BASE_SLEEP = 2.0

    def __init__(self):
        self.login_success = False
        self._last_request_time = 0.0
        _bb_print("[DataFetcher] 初始化数据获取模块...")

    def _rate_limit(self):
        """限速：确保请求间隔不低于 REQUEST_INTERVAL"""
        elapsed = time.time() - self._last_request_time
        if elapsed < self.REQUEST_INTERVAL:
            time.sleep(self.REQUEST_INTERVAL - elapsed)
        self._last_request_time = time.time()

    def _reconnect(self) -> bool:
        """断连后重新登录baostock"""
        try:
            bs.logout()
        except Exception:
            pass
        time.sleep(1.0)
        return self.login()

    def login(self) -> bool:
        """登录baostock"""
        try:
            lg = bs.login()
            if lg.error_code == '0':
                self.login_success = True
                _bb_print("[DataFetcher] baostock登录成功")
                return True
            else:
                print(f"[DataFetcher] baostock登录失败: {lg.error_msg}")
                return False
        except Exception as e:
            print(f"[DataFetcher] 登录异常: {str(e)}")
            return False

    def logout(self):
        """登出baostock"""
        try:
            bs.logout()
            _bb_print("[DataFetcher] baostock已登出")
        except Exception as e:
            _bb_print(f"[DataFetcher] 登出异常: {str(e)}")

    def fetch_single_data(self, code: str, days: int = 1000) -> Optional[pd.DataFrame]:
        """
        【修复17】获取单个指数/股票数据
        【v1.3.1】加重试限速，连接断开自动重连
        Args:
            code: 指数或股票代码
            days: 获取天数
        Returns:
            包含OHLCV数据的DataFrame
        """
        for attempt in range(1, self.MAX_RETRIES + 1):
            try:
                self._rate_limit()

                end_date = datetime.datetime.now().strftime('%Y-%m-%d')
                start_date = (datetime.datetime.now() - datetime.timedelta(days=days * 2)).strftime('%Y-%m-%d')

                rs = bs.query_history_k_data_plus(
                    code,
                    "date,open,high,low,close,volume,amount,turn",
                    start_date=start_date,
                    end_date=end_date,
                    frequency="d",
                    adjustflag="3"
                )

                data_list = []
                while (rs.error_code == '0') & rs.next():
                    data_list.append(rs.get_row_data())

                if not data_list:
                    _bb_print(f"[DataFetcher] {code} 无数据返回")
                    return None

                df = pd.DataFrame(data_list, columns=rs.fields)

                # 类型转换
                for col in ['open', 'high', 'low', 'close', 'volume', 'amount', 'turn']:
                    df[col] = pd.to_numeric(df[col], errors='coerce')

                df['date'] = pd.to_datetime(df['date'])
                df = df.sort_values('date').reset_index(drop=True)
                df = df.tail(days).reset_index(drop=True)

                _bb_print(f"[DataFetcher] ✓ {code} 获取成功: {len(df)} 条数据")
                return df

            except Exception as e:
                err_msg = str(e)
                _bb_print(f"[DataFetcher] ✗ 获取 {code} 第{attempt}次异常: {err_msg}")

                # 连接被踢或网络错误 → 重连后重试
                if '10054' in err_msg or '远程主机' in err_msg or 'Connection' in err_msg:
                    if attempt < self.MAX_RETRIES:
                        sleep_time = self.RETRY_BASE_SLEEP * attempt + random.uniform(0.5, 1.5)
                        print(f"[DataFetcher] 连接断开，{sleep_time:.1f}秒后重连重试({attempt}/{self.MAX_RETRIES})...")
                        time.sleep(sleep_time)
                        self._reconnect()
                    else:
                        print(f"[DataFetcher] {code} 重试{self.MAX_RETRIES}次仍失败，跳过")
                else:
                    # 非网络错误，不重试
                    return None

        return None

    def fetch_all_data(self) -> Dict[str, pd.DataFrame]:
        """获取所有指数和板块数据
        【v1.3.1】增加进度提示和批次间隔
        """
        _bb_print("[DataFetcher] 开始获取所有数据...")
        result = {}

        if not self.login():
            return result

        # 获取三大指数用于计算市场整体
        _bb_print("\n[DataFetcher] 获取三大指数...")
        for name, codes in INDEX_CONFIG_BB.items():
            for code in codes:
                df = self.fetch_single_data(code)
                if df is not None:
                    result[code] = df

        # 板块之间额外等待，降低被踢概率
        time.sleep(1.0)

        # 获取10个行业板块
        print(f"[DataFetcher] 开始获取7个行业板块数据...")
        for idx, (sector_name, code) in enumerate(SECTOR_CONFIG_BB.items(), 1):
            print(f"  [{idx}/7] {sector_name}...", end="", flush=True)

            df = self.fetch_single_data(code)

            if df is not None and len(df) > 100:
                result[sector_name] = df
                print(f" ✓ ({len(df)}条)")
            else:
                print(f" ✗ 失败")

            # 每获取2个板块后额外休息，避免密集请求
            if idx % 2 == 0:
                time.sleep(1.0)

        self.logout()

        print(f"[DataFetcher] 数据获取完成，成功 {len(result)} 个数据集")
        return result


# ============================================================================
# 模块2：FeatureExtractor - 特征提取模块
# ============================================================================
class BB_FeatureExtractor:
    """
    特征提取模块
    功能：提取6大类共80个特征
    输入：OHLCV数据DataFrame
    输出：特征矩阵DataFrame

    【修复4】市场广度特征使用真实的跨板块数据
    【修复5】RSI改用Wilder指数平滑
    【修复6】KDJ改用递推指数平滑
    【修复13】均线排列得分斜率贡献归一化修正
    【修复14】Beta计算除零防护
    【修复19】return_cum_5d/10d改用对数收益率
    【修复20】volume_pct_change防护inf
    """

    def __init__(self):
        _bb_print("[FeatureExtractor] 初始化特征提取模块...")
        self.feature_names = []

    def calculate_price_volume_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """计算价量特征 (20个)"""
        features = pd.DataFrame(index=df.index)
        close = df['close']
        volume = df['volume']

        # 收益率特征
        for window in [1, 2, 3, 5, 10]:
            features[f'return_{window}d'] = close.pct_change(window)
            features[f'volatility_{window}d'] = close.pct_change().rolling(window).std()

        # 【修复20】成交量变化率 - 防护volume=0的情况
        for window in [1, 2, 3, 5, 10]:
            features[f'volume_chg_{window}d'] = volume.replace(0, np.nan).pct_change(window)
            features[f'volume_ma_ratio_{window}d'] = volume / volume.rolling(window).mean()

        # 高低点特征
        features['high_low_ratio'] = (df['high'] - df['low']) / df['close']
        features['body_size'] = abs(df['close'] - df['open']) / df['close']

        # 【修复19】累计收益率 - 改用对数收益率提高效率
        log_ret = np.log(close / close.shift(1))
        features['return_cum_5d'] = np.exp(log_ret.rolling(5).sum()) - 1
        features['return_cum_10d'] = np.exp(log_ret.rolling(10).sum()) - 1

        return features

    def calculate_volatility_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """计算波动率特征 (15个)"""
        features = pd.DataFrame(index=df.index)
        returns = df['close'].pct_change().dropna()

        # 历史波动率
        for window in [5, 10, 20, 30, 60]:
            features[f'hist_vol_{window}d'] = returns.rolling(window).std() * np.sqrt(252)

        # 波动率变化率
        features['vol_chg_5d'] = features['hist_vol_5d'].pct_change(5)
        features['vol_chg_10d'] = features['hist_vol_10d'].pct_change(10)

        # 波动率比率
        features['vol_ratio_5_20'] = features['hist_vol_5d'] / features['hist_vol_20d']
        features['vol_ratio_10_30'] = features['hist_vol_10d'] / features['hist_vol_30d']

        # 收益率分位数
        for q in [0.25, 0.5, 0.75]:
            features[f'return_quantile_{int(q * 100)}'] = returns.rolling(60).quantile(q)

        # 偏度和峰度
        features['return_skew'] = returns.rolling(30).skew()
        features['return_kurt'] = returns.rolling(30).kurt()

        return features

    def calculate_technical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """计算技术指标特征 (25个)"""
        features = pd.DataFrame(index=df.index)
        close = df['close']
        high = df['high']
        low = df['low']

        # MACD
        ema12 = close.ewm(span=12).mean()
        ema26 = close.ewm(span=26).mean()
        macd = ema12 - ema26
        signal = macd.ewm(span=9).mean()
        features['macd'] = macd
        features['macd_signal'] = signal
        features['macd_hist'] = macd - signal

        # 【修复5】RSI - 使用Wilder指数平滑(alpha=1/period)
        delta = close.diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)

        for period in [6, 12, 14]:
            avg_gain = gain.ewm(alpha=1 / period, adjust=False).mean()
            avg_loss = loss.ewm(alpha=1 / period, adjust=False).mean()
            rs = avg_gain / avg_loss.replace(0, np.nan)
            features[f'rsi_{period}'] = 100 - (100 / (1 + rs))

        # 【修复6】KDJ - 使用递推指数平滑
        low_min = low.rolling(9).min()
        high_max = high.rolling(9).max()
        rsv = (close - low_min) / (high_max - low_min).replace(0, np.nan) * 100

        # K: rsv的指数移动平均(alpha=1/3)
        features['kdj_k'] = rsv.ewm(alpha=1 / 3, adjust=False).mean()
        # D: K的指数移动平均(alpha=1/3)
        features['kdj_d'] = features['kdj_k'].ewm(alpha=1 / 3, adjust=False).mean()
        # J = 3*K - 2*D
        features['kdj_j'] = 3 * features['kdj_k'] - 2 * features['kdj_d']

        # 布林带
        for period in [20]:
            ma = close.rolling(period).mean()
            std = close.rolling(period).std()
            features[f'bb_upper'] = ma + 2 * std
            features[f'bb_middle'] = ma
            features[f'bb_lower'] = ma - 2 * std
            bb_width = features[f'bb_upper'] - features[f'bb_lower']
            features[f'bb_position'] = (close - features[f'bb_lower']) / bb_width.replace(0, np.nan)
            features[f'bb_width'] = bb_width / ma.replace(0, np.nan)

        # ADX趋势强度
        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        features['adx_tr'] = tr.rolling(14).mean()

        # 均线特征
        for ma in [5, 10, 20, 60]:
            features[f'ma_{ma}_ratio'] = close / close.rolling(ma).mean()

        return features

    def calculate_relative_strength_features(self, df: pd.DataFrame, benchmark: pd.DataFrame) -> pd.DataFrame:
        """计算相对强弱特征 (10个)"""
        features = pd.DataFrame(index=df.index)

        if benchmark is None:
            benchmark = df

        # 相对收益率
        for window in [1, 3, 5, 10, 20]:
            sector_return = df['close'].pct_change(window)
            bench_return = benchmark['close'].pct_change(window)
            features[f'relative_return_{window}d'] = sector_return - bench_return

        # 相对强弱排名特征
        for window in [5, 10, 20]:
            features[f'relative_strength_{window}d'] = features[f'relative_return_{window}d'].rolling(5).mean()

        # 【修复14】Beta系数 - 添加除零防护
        returns = df['close'].pct_change().dropna()
        bench_returns = benchmark['close'].pct_change().dropna()
        common_idx = returns.index.intersection(bench_returns.index)

        if len(common_idx) > 30:
            covariance = returns.loc[common_idx].rolling(30).cov(bench_returns.loc[common_idx])
            variance = bench_returns.loc[common_idx].rolling(30).var().replace(0, np.nan)
            # 【修复14】添加最小值防护，避免极端情况
            features['beta_30d'] = (covariance / variance).where(variance > 1e-10, np.nan)

        return features

    def calculate_market_breadth_features(self, all_data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """
        【修复4】计算真实的市场广度特征
        - 跨板块涨跌比率：统计当日有多少板块收涨vs收跌
        - 成交量动量离散度：各板块量比的标准差
        - 跨板块新高占比：当日创20日新高的板块数占比
        """
        # 使用沪深300作为基准，如果没有就用第一个数据
        if 'sh.000300' in all_data:
            df = all_data['sh.000300']
        else:
            df = list(all_data.values())[0]

        features = pd.DataFrame(index=df.index)

        # 【修复4】跨板块涨跌比率
        # 统计各板块当日涨跌情况
        sector_returns = []
        for sector_name in ALL_SECTORS_BB:
            if sector_name in all_data and len(all_data[sector_name]) > 0:
                sector_df = all_data[sector_name].copy()
                sector_df = sector_df.set_index('date').reindex(df['date'])
                if 'close' in sector_df.columns:
                    ret = sector_df['close'].pct_change()
                    sector_returns.append(ret)

        if sector_returns:
            sector_returns_df = pd.DataFrame(sector_returns).T

            # 跨板块涨跌比率：涨的板块数/跌的板块数
            advancing = (sector_returns_df > 0).sum(axis=1)
            declining = (sector_returns_df < 0).sum(axis=1)
            # 避免除零
            features['adv_decl_ratio'] = (advancing / declining.replace(0, 1)).rolling(5).mean()

            # 【修复4】成交量动量离散度：各板块量比的标准差
            sector_volume_ratios = []
            for sector_name in ALL_SECTORS_BB:
                if sector_name in all_data and len(all_data[sector_name]) > 0:
                    sector_df = all_data[sector_name].copy()
                    sector_df = sector_df.set_index('date').reindex(df['date'])
                    if 'volume' in sector_df.columns:
                        vol_ratio = sector_df['volume'] / sector_df['volume'].rolling(20).mean()
                        sector_volume_ratios.append(vol_ratio)

            if sector_volume_ratios:
                vol_ratios_df = pd.DataFrame(sector_volume_ratios).T
                features['volume_breadth'] = vol_ratios_df.std(axis=1).rolling(5).mean()

            # 【修复4】跨板块新高占比：创20日新高的板块数占比
            sector_new_highs = []
            for sector_name in ALL_SECTORS_BB:
                if sector_name in all_data and len(all_data[sector_name]) > 0:
                    sector_df = all_data[sector_name].copy()
                    sector_df = sector_df.set_index('date').reindex(df['date'])
                    if 'close' in sector_df.columns:
                        rolling_max = sector_df['close'].rolling(20).max().shift(1)
                        is_new_high = sector_df['close'] >= rolling_max
                        sector_new_highs.append(is_new_high)

            if sector_new_highs:
                new_highs_df = pd.DataFrame(sector_new_highs).T
                features['new_high_low_ratio'] = new_highs_df.mean(axis=1).rolling(5).mean()
        else:
            # 降级处理：如果没有板块数据
            features['adv_decl_ratio'] = np.where(df['close'].pct_change() > 0, 1.2, 0.8)
            features['adv_decl_ratio'] = features['adv_decl_ratio'].rolling(5).mean()
            features['volume_breadth'] = df['volume'] / df['volume'].rolling(20).mean()
            features['new_high_low_ratio'] = (df['close'] / df['close'].rolling(60).max()).rolling(5).mean()

        return features

    def calculate_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """计算时间特征 (5个)"""
        features = pd.DataFrame(index=df.index)
        dates = df['date']

        # 星期几
        features['day_of_week'] = dates.dt.dayofweek

        # 月初月末
        features['is_month_start'] = dates.dt.is_month_start.astype(int)
        features['is_month_end'] = dates.dt.is_month_end.astype(int)

        # 季度末
        features['is_quarter_end'] = dates.dt.is_quarter_end.astype(int)

        return features

    def extract_all_features(self, df: pd.DataFrame, benchmark: pd.DataFrame = None,
                             all_data: Dict = None) -> pd.DataFrame:
        """提取所有特征"""
        _bb_print(f"[FeatureExtractor] 开始提取特征, 输入数据形状: {df.shape}")

        # 合并所有特征
        all_features = pd.concat([
            self.calculate_price_volume_features(df),
            self.calculate_volatility_features(df),
            self.calculate_technical_features(df),
            self.calculate_relative_strength_features(df, benchmark),
            self.calculate_time_features(df)
        ], axis=1)

        # 添加市场广度特征
        if all_data is not None:
            breadth_feats = self.calculate_market_breadth_features(all_data)
            all_features = pd.concat([all_features, breadth_feats.reindex(all_features.index)], axis=1)

        # 处理无穷大和NaN
        all_features = all_features.replace([np.inf, -np.inf], np.nan)
        all_features = all_features.ffill().bfill().fillna(0)

        self.feature_names = all_features.columns.tolist()
        _bb_print(f"[FeatureExtractor] 特征提取完成, 共 {len(self.feature_names)} 个特征")

        return all_features


# ============================================================================
# 模块3：TraditionalModels - 传统金融模型模块
# ============================================================================
class BB_TraditionalModels:
    """
    传统金融模型模块
    功能：实现GARCH、EWMA、量价背离、均线排列等传统模型信号
    【修复8】GARCH置信度改用JB检验p值
    【修复16】添加valid标记
    【修复13】均线排列得分斜率归一化修正
    """

    def __init__(self):
        _bb_print("[TraditionalModels] 初始化传统金融模型模块...")

    def garch_volatility_forecast(self, df: pd.DataFrame, horizon: int = 2) -> Dict:
        """
        GARCH(1,1)波动率预测
        【修复8】使用Jarque-Bera检验p值计算置信度
        """
        try:
            returns = df['close'].pct_change().dropna() * 100
            train_data = returns.tail(60)

            if len(train_data) < 30:
                _bb_print("[TraditionalModels] GARCH数据不足，使用简单波动率")
                return {'garch_vol_forecast': 0.02, 'garch_vol_percentile': 0.5, 'garch_confidence': 0.3, 'valid': True}

            # 拟合GARCH(1,1)
            am = arch_model(train_data, vol='Garch', p=1, q=1, dist='normal')
            res = am.fit(disp='off', show_warning=False)

            # 预测
            forecast = res.forecast(horizon=horizon)
            vol_forecast = np.sqrt(forecast.variance.values[-1, :].mean()) / 100

            # 计算历史分位值
            hist_vol = returns.rolling(20).std().dropna() / 100
            vol_percentile = stats.percentileofscore(hist_vol.dropna(), vol_forecast) / 100

            # 【修复8】使用Jarque-Bera检验p值计算置信度
            # p值越大说明残差越接近正态，模型拟合越好，置信度越高
            try:
                resid = res.resid / res.conditional_volatility
                jb_stat, jb_pvalue = jarque_bera(resid.dropna())
                garch_confidence = min(0.95, 0.3 + 0.65 * jb_pvalue)
            except Exception:
                # 降级：如果JB检验失败，使用对数似然值的相对大小
                loglik = res.loglikelihood if hasattr(res, 'loglikelihood') else -1000
                garch_confidence = min(0.9, max(0.3, 0.5 + 0.1 * (loglik + 100)))

            result = {
                'garch_vol_forecast': float(vol_forecast),
                'garch_vol_percentile': float(vol_percentile),
                'garch_confidence': float(garch_confidence),
                'valid': True
            }

            _bb_print(
                f"[TraditionalModels] GARCH预测波动率: {vol_forecast:.4f}, 历史分位: {vol_percentile:.2%}, 置信度: {garch_confidence:.2f}")
            return result

        except Exception as e:
            _bb_print(f"[TraditionalModels] GARCH模型异常: {str(e)}")
            return {'garch_vol_forecast': 0.02, 'garch_vol_percentile': 0.5, 'garch_confidence': 0.3, 'valid': False}

    def ewma_trend(self, df: pd.DataFrame, half_life: int = 3) -> Dict:
        """
        指数加权移动平均(EWMA)
        半衰期3天，计算短期趋势强度和方向
        """
        try:
            close = df['close']
            alpha = 1 - np.exp(np.log(0.5) / half_life)

            ewma_fast = close.ewm(alpha=alpha).mean()
            ewma_slow = close.ewm(alpha=alpha / 2).mean()

            # 趋势方向
            trend_direction = 1 if ewma_fast.iloc[-1] > ewma_slow.iloc[-1] else -1

            # 趋势强度
            trend_strength = abs(ewma_fast.iloc[-1] - ewma_slow.iloc[-1]) / ewma_slow.iloc[-1]
            trend_strength = min(1.0, trend_strength * 50)  # 归一化

            # 动量
            momentum = (ewma_fast.pct_change(3).iloc[-1]) * 100

            result = {
                'ewma_trend_direction': trend_direction,
                'ewma_trend_strength': trend_strength,
                'ewma_momentum': momentum,
                'ewma_confidence': 0.8,
                'valid': True
            }

            _bb_print(
                f"[TraditionalModels] EWMA趋势: {'多头' if trend_direction > 0 else '空头'}, 强度: {trend_strength:.2f}")
            return result

        except Exception as e:
            _bb_print(f"[TraditionalModels] EWMA模型异常: {str(e)}")
            return {'ewma_trend_direction': 0, 'ewma_trend_strength': 0.5, 'ewma_momentum': 0, 'ewma_confidence': 0.5,
                    'valid': False}

    def volume_price_divergence(self, df: pd.DataFrame) -> Dict:
        """
        量价背离检测
        检测"价涨量缩"和"价跌量缩"短期反转信号
        输出信号强度(-1到1)
        """
        try:
            close = df['close']
            volume = df['volume']

            # 计算最近5天的价格和成交量变化
            price_change = close.pct_change(5).iloc[-1]
            volume_change = volume.pct_change(5).iloc[-1]

            # 量价背离检测
            divergence_score = 0

            # 价涨量缩：看跌信号
            if price_change > 0.01 and volume_change < -0.1:
                divergence_score = min(1.0, price_change * 10)

            # 价跌量增：看跌
            elif price_change < -0.01 and volume_change > 0.1:
                divergence_score = -min(1.0, abs(price_change) * 10)

            # 置信度
            confidence = 0.6 + 0.2 * (abs(divergence_score))

            result = {
                'vp_divergence_score': divergence_score,
                'vp_price_change': price_change,
                'vp_volume_change': volume_change,
                'vp_confidence': confidence,
                'valid': True
            }

            signal_text = '看涨' if divergence_score > 0 else '看跌' if divergence_score < 0 else '无信号'
            _bb_print(f"[TraditionalModels] 量价背离: {signal_text}, 强度: {divergence_score:.2f}")
            return result

        except Exception as e:
            _bb_print(f"[TraditionalModels] 量价背离检测异常: {str(e)}")
            return {'vp_divergence_score': 0, 'vp_price_change': 0, 'vp_volume_change': 0, 'vp_confidence': 0.5,
                    'valid': False}

    def ma_alignment_score(self, df: pd.DataFrame) -> Dict:
        """
        均线多空排列
        仅用5/10/20三条短期均线，计算多空得分(0-100)
        【修复13】修正斜率贡献归一化
        """
        try:
            close = df['close']
            ma5 = close.rolling(5).mean()
            ma10 = close.rolling(10).mean()
            ma20 = close.rolling(20).mean()

            score = 0

            # 多头排列检查
            if ma5.iloc[-1] > ma10.iloc[-1] > ma20.iloc[-1]:
                score += 50
            # 空头排列检查
            elif ma5.iloc[-1] < ma10.iloc[-1] < ma20.iloc[-1]:
                score -= 50

            # 【修复13】每条均线的斜率 - 修正归一化
            # 放大斜率敏感度，使min(1, abs(slope)*10)有意义
            for ma, weight in [(ma5, 30), (ma10, 15), (ma20, 5)]:
                slope = (ma.iloc[-1] - ma.iloc[-5]) / ma.iloc[-5] * 100
                # 【修复13】斜率贡献也归一化到合理范围
                slope_contrib = weight * np.sign(slope) * min(1, abs(slope) * 10)
                score += slope_contrib

            # 归一化到0-100
            normalized_score = 50 + score / 2
            normalized_score = max(0, min(100, normalized_score))

            result = {
                'ma_alignment_score': normalized_score,
                'ma5_above_ma10': 1 if ma5.iloc[-1] > ma10.iloc[-1] else 0,
                'ma10_above_ma20': 1 if ma10.iloc[-1] > ma20.iloc[-1] else 0,
                'ma_confidence': 0.75,
                'valid': True
            }

            _bb_print(f"[TraditionalModels] 均线得分: {normalized_score:.1f}/100")
            return result

        except Exception as e:
            _bb_print(f"[TraditionalModels] 均线计算异常: {str(e)}")
            return {'ma_alignment_score': 50, 'ma5_above_ma10': 0, 'ma10_above_ma20': 0, 'ma_confidence': 0.5,
                    'valid': False}

    def get_all_signals(self, df: pd.DataFrame) -> Dict:
        """获取所有传统模型信号"""
        _bb_print("[TraditionalModels] 计算所有传统模型信号...")

        signals = {}
        garch_result = self.garch_volatility_forecast(df)
        signals.update(garch_result)

        ewma_result = self.ewma_trend(df)
        signals.update(ewma_result)

        vp_result = self.volume_price_divergence(df)
        signals.update(vp_result)

        ma_result = self.ma_alignment_score(df)
        signals.update(ma_result)

        return signals


# ============================================================================
# 模块4：LGBMPredictor - LightGBM预测模块
# ============================================================================


# ============================================================================
# 模块3.5：BB_RealtimeAnalyzer - mootdx实时行情增强模块
# 设计意图：用mootdx实时数据为牛熊预测提供盘中信号补充
# 核心能力：实时涨跌家数、资金流向信号、盘中动量加速度
# ============================================================================

class BB_RealtimeAnalyzer:
    """
    mootdx实时行情分析器
    在baostock历史数据的基础上，叠加当日盘中实时信号：
    - 实时涨跌家数比（市场广度）
    - 板块实时涨跌分布
    - 盘中动量加速度（当前涨幅 vs 开盘涨幅）
    - 量能异常检测（量比>2 为放量信号）
    """

    def __init__(self):
        self._market_breadth = None     # 涨跌家数缓存
        self._sector_summary = {}       # 板块实时摘要缓存
        self._cache_time = None

    def get_market_breadth(self):
        """
        获取A股市场实时涨跌家数
        返回: {'up': int, 'down': int, 'flat': int, 'up_ratio': float,
                'limit_up': int, 'limit_down': int} 或 None
        """
        if not MOOTDX_CLIENT._connected:
            return None

        try:
            # 用上证+深证成分代表全市场
            # mootdx的板块成分股列表
            from mootdx.quotes import Quotes
            # 获取所有A股实时行情
            # 用行业板块指数的涨跌家数近似
            all_up, all_down, all_flat = 0, 0, 0
            all_limit_up, all_limit_down = 0, 0

            # 遍历各板块统计涨跌
            for sector_name, sector_code in SECTOR_CONFIG_BB.items():
                summary = self.get_sector_breadth(sector_name)
                if summary:
                    all_up += summary.get('up', 0)
                    all_down += summary.get('down', 0)
                    all_flat += summary.get('flat', 0)

            total = all_up + all_down + all_flat
            if total == 0:
                return None

            self._market_breadth = {
                'up': all_up,
                'down': all_down,
                'flat': all_flat,
                'up_ratio': all_up / total,
                'limit_up': all_limit_up,
                'limit_down': all_limit_down,
            }
            return self._market_breadth

        except Exception as e:
            _bb_print(f"[RealtimeAnalyzer] 市场广度获取失败: {e}")
            return None

    def get_sector_breadth(self, sector_name):
        """
        获取板块实时涨跌概况
        利用股票池中该板块的股票批量查询
        """
        if sector_name in self._sector_summary:
            return self._sector_summary[sector_name]

        # 从股票池拿该板块的所有代码
        sector_stocks = STOCK_POOL.get(sector_name, [])
        if not sector_stocks:
            return None

        codes = [s['code'] for s in sector_stocks]
        quotes = MOOTDX_CLIENT.get_realtime_quotes_batch(codes)

        if not quotes:
            return None

        up, down, flat = 0, 0, 0
        for code, info in quotes.items():
            pct = info.get('change_pct', 0)
            if pct > 0.5:
                up += 1
            elif pct < -0.5:
                down += 1
            else:
                flat += 1

        result = {'up': up, 'down': down, 'flat': flat, 'total': up + down + flat}
        self._sector_summary[sector_name] = result
        return result

    def get_intraday_momentum(self, code):
        """
        盘中动量加速度：当前涨幅 vs 开盘半小时涨幅
        返回: {'momentum': float, 'acceleration': float, 'signal': str}
        - momentum > 0: 盘中走强
        - momentum < 0: 盘中走弱
        - acceleration: 动量变化率
        """
        quotes = MOOTDX_CLIENT.get_realtime_quotes_batch([code])
        if not quotes or code not in quotes:
            return None

        info = quotes[code]
        current_pct = info.get('change_pct', 0)
        open_price = info.get('open', 0)
        last_close = info.get('last_close', 0)

        if last_close <= 0 or open_price <= 0:
            return None

        # 开盘涨幅（以开盘价vs昨收计算）
        open_pct = (open_price - last_close) / last_close * 100
        # 动量 = 当前涨幅 - 开盘涨幅
        momentum = current_pct - open_pct

        if momentum > 0.5:
            signal = '加速上涨'
        elif momentum < -0.5:
            signal = '加速下跌'
        elif current_pct > 0:
            signal = '高位横盘'
        elif current_pct < 0:
            signal = '低位横盘'
        else:
            signal = '平盘震荡'

        return {
            'current_pct': round(current_pct, 2),
            'open_pct': round(open_pct, 2),
            'momentum': round(momentum, 2),
            'signal': signal,
        }

    def get_realtime_enhanced_signals(self):
        """
        综合实时信号，返回供牛熊预测使用的增强信号字典
        """
        signals = {
            'realtime_available': False,
            'market_breadth': None,
            'sector_breadth': {},
            'intraday_momentum': {},
        }

        if not MOOTDX_CLIENT._connected:
            return signals

        # 1. 市场广度
        breadth = self.get_market_breadth()
        if breadth:
            signals['market_breadth'] = breadth
            signals['realtime_available'] = True

        # 2. 各板块涨跌分布
        for sector_name in ALL_SECTORS_BB:
            sb = self.get_sector_breadth(sector_name)
            if sb:
                signals['sector_breadth'][sector_name] = sb

        # 3. 主要指数盘中动量
        for idx_name, idx_code in [('上证', 'sh.000001'), ('深证', 'sz.399001')]:
            mom = self.get_intraday_momentum(idx_code)
            if mom:
                signals['intraday_momentum'][idx_name] = mom

        return signals


# 全局实例
BB_REALTIME = BB_RealtimeAnalyzer()

class BB_LGBMPredictor:
    """
    LightGBM预测模块
    功能：三分类牛熊预测，时间序列滚动验证
    【修复1】最后2行标签设为np.nan，训练前dropna
    【修复7】添加is_unbalance=True处理类别不平衡
    【修复10】对1日和2日分别训练模型
    【修复11】汇总所有板块特征重要性
    【修复12】收集所有板块性能指标
    【修复15】增加全局模型选项
    """

    def __init__(self):
        _bb_print("[LGBMPredictor] 初始化LightGBM预测模块...")
        self.model = None
        self.model_2d = None  # 【修复10】2日预测专用模型
        self.model_3d = None  # 3日预测专用模型
        self.subtype_model = None  # 走势细分模型
        self.feature_importance = None
        self.performance_metrics = {}
        self.all_cv_metrics = []  # 【修复12】收集所有板块的CV指标

        # 【修复7】LightGBM参数 - 添加is_unbalance=True
        self.params = {
            'objective': 'multiclass',
            'num_class': 3,
            'metric': 'multi_logloss',
            'learning_rate': 0.03,
            'max_depth': 4,
            'num_leaves': 12,
            'min_data_in_leaf': 20,
            'feature_fraction': 0.8,
            'bagging_fraction': 0.8,
            'bagging_freq': 5,
            'is_unbalance': True,  # 【修复7】处理类别不平衡
            'verbose': -1,
            'seed': 42
        }

        # 【修复10】2日预测模型参数
        self.params_2d = self.params.copy()

    def create_labels(self, df: pd.DataFrame, horizon: int = 1) -> pd.Series:
        """
        创建三分类标签 - 最后horizon行设为np.nan
        Args:
            df: 价格数据
            horizon: 预测天数
        """
        close = df['close']

        # 根据horizon计算未来收益率
        future_return = close.shift(-horizon) / close - 1

        labels = pd.Series(1, index=df.index)  # 默认震荡

        # 牛市条件
        # 【P1-1修复】阈值改用分位数替代固定0.008
        threshold_up = future_return.quantile(0.75)
        threshold_down = future_return.quantile(0.25)
        bull_condition = future_return > threshold_up
        labels[bull_condition] = 2

        # 熊市条件
        bear_condition = future_return < threshold_down
        labels[bear_condition] = 0

        # 最后horizon行标签设为np.nan（未来数据不存在）
        for i in range(horizon):
            labels.iloc[-(i + 1)] = np.nan

        return labels

    def create_subtype_labels(self, df: pd.DataFrame, horizon: int = 1) -> pd.Series:
        """
        创建走势细分标签（5分类）
        基于未来horizon日的开盘/收盘相对关系：
        0: 高开高走 (open>prev_close 且 close>open)
        1: 低开高走 (open<prev_close 且 close>open)
        2: 震荡 (close ≈ open)
        3: 高开低走 (open>prev_close 且 close<open)
        4: 低开低走 (open<prev_close 且 close<open)
        """
        close = df['close']
        open_ = df['open']

        # 未来第horizon日的开盘价和收盘价（近似：用当日收盘预测次日开盘）
        # 由于我们只有日K，无法直接知道"未来开盘"，用未来收盘日数据推算
        future_close = close.shift(-horizon)
        future_open = open_.shift(-horizon)
        prev_close = close  # 当日收盘 ≈ 次日开盘参考

        # 涨跌幅度
        future_return = (future_close - prev_close) / prev_close
        intraday_return = (future_close - future_open) / future_open.replace(0, np.nan)
        gap = (future_open - prev_close) / prev_close  # 跳空

        labels = pd.Series(2, index=df.index)  # 默认震荡

        # 高开高走：跳空高开 + 收阳
        labels[(gap > 0.003) & (intraday_return > 0.003)] = 0
        # 低开高走：跳空低开 + 收阳
        labels[(gap < -0.003) & (intraday_return > 0.003)] = 1
        # 高开低走：跳空高开 + 收阴
        labels[(gap > 0.003) & (intraday_return < -0.003)] = 3
        # 低开低走：跳空低开 + 收阴
        labels[(gap < -0.003) & (intraday_return < -0.003)] = 4
        # 其余为震荡（label=2）

        # 最后horizon行设为np.nan
        for i in range(horizon):
            labels.iloc[-(i + 1)] = np.nan

        return labels

    def calculate_autocorrelation_decay(self, returns: pd.Series) -> Dict[int, float]:
        """
        【修复10】计算收益率1-5日自相关系数，用于多日预测衰减
        """
        autocorr = {}
        for lag in range(1, 6):
            ac = returns.autocorr(lag=lag) if len(returns) > lag else 0
            # 自相关可能为负，转为正向衰减系数
            autocorr[lag] = max(0.1, ac if not np.isnan(ac) else 0.1)
        return autocorr

    def time_series_cv(self, X: pd.DataFrame, y: pd.Series, n_splits: int = 5) -> Dict:
        """时间序列滚动窗口交叉验证"""
        _bb_print("[LGBMPredictor] 开始时间序列交叉验证...")

        n_samples = len(X)
        train_size = int(n_samples * 0.6)
        test_size = int((n_samples - train_size) / n_splits)

        accuracies = []
        precisions = []
        recalls = []
        f1_scores = []

        for i in range(n_splits):
            train_end = train_size + i * test_size
            test_end = train_end + test_size

            if test_end >= n_samples:
                break

            X_train, X_test = X.iloc[:train_end], X.iloc[train_end:test_end]
            y_train, y_test = y.iloc[:train_end], y.iloc[train_end:test_end]

            # 【修复1】移除包含NaN的行（包括标签为NaN的最后几行）
            train_mask = ~(X_train.isna().any(axis=1)) & ~y_train.isna()
            test_mask = ~(X_test.isna().any(axis=1)) & ~y_test.isna()

            X_train, y_train = X_train[train_mask], y_train[train_mask]
            X_test, y_test = X_test[test_mask], y_test[test_mask]

            if len(X_train) < 100 or len(X_test) < 10:
                continue

            train_data = lgb.Dataset(X_train, label=y_train)
            model = lgb.train(self.params, train_data, num_boost_round=200, valid_sets=[train_data],
                              callbacks=[lgb.log_evaluation(0)])

            y_pred_proba = model.predict(X_test)
            y_pred = y_pred_proba.argmax(axis=1)

            accuracies.append(accuracy_score(y_test, y_pred))
            precisions.append(precision_score(y_test, y_pred, average='weighted', zero_division=0))
            recalls.append(recall_score(y_test, y_pred, average='weighted', zero_division=0))
            f1_scores.append(f1_score(y_test, y_pred, average='weighted', zero_division=0))

        metrics = {
            'cv_accuracy': np.mean(accuracies) if accuracies else 0.5,
            'cv_precision': np.mean(precisions) if precisions else 0.5,
            'cv_recall': np.mean(recalls) if recalls else 0.5,
            'cv_f1': np.mean(f1_scores) if f1_scores else 0.5
        }

        # 【修复12】收集所有板块的CV指标
        self.all_cv_metrics.append(metrics)

        _bb_print(f"[LGBMPredictor] CV结果 - 准确率: {metrics['cv_accuracy']:.3f}, F1: {metrics['cv_f1']:.3f}")
        return metrics

    def train_and_predict(self, features: pd.DataFrame, df: pd.DataFrame,
                          global_features: pd.DataFrame = None,
                          global_df: pd.DataFrame = None) -> Tuple[
        np.ndarray, np.ndarray, np.ndarray, np.ndarray, pd.DataFrame]:
        """
        训练模型并预测最新数据
        返回: (1日预测, 2日预测, 3日预测, 走势细分预测, 特征重要性)
        """
        try:
            _bb_print("[LGBMPredictor] 开始模型训练...")

            # 创建1日标签
            y = self.create_labels(df, horizon=1)

            # 对齐特征和标签
            common_idx = features.index.intersection(y.index)
            X = features.loc[common_idx]
            y = y.loc[common_idx]

            # 交叉验证
            metrics = self.time_series_cv(X, y)
            self.performance_metrics = metrics

            # 训练前移除NaN行
            mask = ~(X.isna().any(axis=1)) & ~y.isna()
            X_clean = X[mask]
            y_clean = y[mask]

            # 训练1日模型
            train_data = lgb.Dataset(X_clean, label=y_clean)
            self.model = lgb.train(self.params, train_data, num_boost_round=200, valid_sets=[train_data],
                                   callbacks=[lgb.log_evaluation(0)])

            # 训练2日模型
            y_2d = self.create_labels(df, horizon=2)
            y_2d = y_2d.loc[common_idx]
            mask_2d = ~(X.isna().any(axis=1)) & ~y_2d.isna()
            X_clean_2d = X[mask_2d]
            y_clean_2d = y_2d[mask_2d]

            prediction_2d = None
            if len(y_clean_2d) > 100:
                train_data_2d = lgb.Dataset(X_clean_2d, label=y_clean_2d)
                self.model_2d = lgb.train(self.params_2d, train_data_2d, num_boost_round=200,
                                          valid_sets=[train_data_2d], callbacks=[lgb.log_evaluation(0)])
                _bb_print("[LGBMPredictor] 2日预测模型训练完成")

            # 训练3日模型
            y_3d = self.create_labels(df, horizon=3)
            y_3d = y_3d.loc[common_idx]
            mask_3d = ~(X.isna().any(axis=1)) & ~y_3d.isna()
            X_clean_3d = X[mask_3d]
            y_clean_3d = y_3d[mask_3d]

            prediction_3d = None
            if len(y_clean_3d) > 100:
                train_data_3d = lgb.Dataset(X_clean_3d, label=y_clean_3d)
                self.model_3d = lgb.train(self.params_2d, train_data_3d, num_boost_round=200,
                                          valid_sets=[train_data_3d], callbacks=[lgb.log_evaluation(0)])
                _bb_print("[LGBMPredictor] 3日预测模型训练完成")

            # 训练走势细分模型（5分类）
            prediction_subtype = None
            try:
                y_sub = self.create_subtype_labels(df, horizon=1)
                y_sub = y_sub.loc[common_idx]
                mask_sub = ~(X.isna().any(axis=1)) & ~y_sub.isna()
                X_clean_sub = X[mask_sub]
                y_clean_sub = y_sub[mask_sub]

                if len(y_clean_sub) > 150:
                    subtype_params = self.params.copy()
                    subtype_params['num_class'] = 5
                    train_data_sub = lgb.Dataset(X_clean_sub, label=y_clean_sub)
                    self.subtype_model = lgb.train(subtype_params, train_data_sub, num_boost_round=200,
                                                   valid_sets=[train_data_sub], callbacks=[lgb.log_evaluation(0)])
                    _bb_print("[LGBMPredictor] 走势细分模型训练完成")
            except Exception as e:
                _bb_print(f"[LGBMPredictor] 走势细分模型训练失败: {str(e)}")

            # 特征重要性
            importance = pd.DataFrame({
                'feature': X.columns,
                'importance': self.model.feature_importance()
            }).sort_values('importance', ascending=False).reset_index(drop=True)

            # 汇总特征重要性
            if self.feature_importance is None:
                self.feature_importance = importance
            else:
                self.feature_importance = pd.concat([self.feature_importance, importance], ignore_index=True)

            # 预测最新数据
            latest_features = X.iloc[-1:].fillna(0)

            # 1日预测
            prediction_1d = self.model.predict(latest_features)[0]

            # 2日预测
            prediction_2d = self.model_2d.predict(latest_features)[0] if self.model_2d is not None else None

            # 3日预测
            prediction_3d = self.model_3d.predict(latest_features)[0] if self.model_3d is not None else None

            # 走势细分预测
            prediction_subtype = self.subtype_model.predict(latest_features)[
                0] if self.subtype_model is not None else None

            # 全局模型融合
            global_proba = None
            if global_features is not None and global_df is not None:
                try:
                    y_global = self.create_labels(global_df, horizon=1)
                    common_global = global_features.index.intersection(y_global.index)
                    X_global = global_features.loc[common_global]
                    y_global = y_global.loc[common_global]

                    mask_global = ~(X_global.isna().any(axis=1)) & ~y_global.isna()
                    X_global_clean = X_global[mask_global]
                    y_global_clean = y_global[mask_global]

                    if len(y_global_clean) > 200:
                        train_global = lgb.Dataset(X_global_clean, label=y_global_clean)
                        global_model = lgb.train(self.params, train_global, num_boost_round=200,
                                                 valid_sets=[train_global], callbacks=[lgb.log_evaluation(0)])
                        global_proba = global_model.predict(latest_features)[0]
                        _bb_print("[LGBMPredictor] 全局模型预测完成")
                except Exception as e:
                    _bb_print(f"[LGBMPredictor] 全局模型训练失败: {str(e)}")

            # 加权融合：全局模型权重0.3，板块模型权重0.7
            if global_proba is not None:
                prediction_1d = 0.7 * prediction_1d + 0.3 * global_proba
                prediction_1d = prediction_1d / prediction_1d.sum()
                _bb_print("[LGBMPredictor] 已融合全局模型预测")

            _bb_print(
                f"[LGBMPredictor] 预测完成 - 熊市概率: {prediction_1d[0]:.1%}, 震荡概率: {prediction_1d[1]:.1%}, 牛市概率: {prediction_1d[2]:.1%}")

            return prediction_1d, prediction_2d, prediction_3d, prediction_subtype, importance

        except Exception as e:
            _bb_print(f"[LGBMPredictor] 模型训练异常: {str(e)}")
            import traceback
            traceback.print_exc()
            # 降级：返回等概率
            return np.array([0.33, 0.34, 0.33]), None, None, None, pd.DataFrame({'feature': [], 'importance': []})


# ============================================================================
# 模块5：BayesianFusion - 贝叶斯融合模块
# ============================================================================
class BB_BayesianFusion:
    """
    贝叶斯融合模块
    【修复9】实现真正的贝叶斯后验更新
    【修复10】基于自相关衰减的多日预测
    【修复16】处理valid=False的信号
    """

    def __init__(self):
        _bb_print("[BayesianFusion] 初始化贝叶斯融合模块...")
        self.autocorr_decay = {}  # 【修复10】存储自相关系数

    def calculate_autocorrelation_decay(self, returns: pd.Series) -> Dict[int, float]:
        """
        【修复10】计算收益率自相关系数
        """
        autocorr = {}
        for lag in range(1, 6):
            if len(returns) > lag:
                ac = returns.autocorr(lag=lag)
                autocorr[lag] = max(0.15, ac if not np.isnan(ac) else 0.15)
            else:
                autocorr[lag] = 0.15
        self.autocorr_decay = autocorr
        return autocorr

    def bayesian_likelihood(self, signal_value: float, signal_name: str) -> Tuple[np.ndarray, float]:
        """
        【修复9】计算贝叶斯似然
        返回: (likelihood [bear, neutral, bull], confidence)
        """
        bear, neutral, bull = 0.33, 0.34, 0.33
        confidence = 0.5

        if signal_name == 'garch':
            # 【修复9】GARCH: 根据波动率分位数计算似然
            vol_percentile = signal_value
            if vol_percentile > 0.7:
                # 高波动：增加极端行情概率
                bear = 0.40
                bull = 0.40
                neutral = 0.20
                confidence = 0.8
            elif vol_percentile < 0.3:
                # 低波动：增加震荡概率
                neutral = 0.60
                bear = 0.20
                bull = 0.20
                confidence = 0.6
            else:
                confidence = 0.5

        elif signal_name == 'ewma':
            # 【修复9】EWMA: 根据趋势方向和强度计算似然
            direction = signal_value  # 1: 多头, -1: 空头
            strength = getattr(self, 'ewma_strength', 0.5)

            if direction > 0:
                bull = 0.33 + strength * 0.4
                bear = 0.33 - strength * 0.2
                neutral = 0.34 - strength * 0.2
            elif direction < 0:
                bear = 0.33 + strength * 0.4
                bull = 0.33 - strength * 0.2
                neutral = 0.34 - strength * 0.2
            else:
                neutral = 0.6
                bear = bull = 0.2

            confidence = 0.6 + 0.2 * strength

        elif signal_name == 'ma':
            # 【修复9】MA: 根据均线得分计算似然
            score = signal_value / 100.0  # 归一化到0-1
            bull = 0.25 + score * 0.45
            bear = 0.45 - score * 0.45
            neutral = 0.30
            confidence = 0.7

        elif signal_name == 'vp':
            # 【修复9】VP: 根据背离分数计算似然
            divergence = signal_value  # -1到1
            if divergence > 0:
                bull = 0.33 + divergence * 0.35
                bear = 0.33 - divergence * 0.2
                neutral = 0.34 - divergence * 0.15
            elif divergence < 0:
                bear = 0.33 - divergence * 0.35
                bull = 0.33 + divergence * 0.2
                neutral = 0.34 + divergence * 0.15
            else:
                neutral = 0.5
                bear = bull = 0.25

            confidence = 0.55 + 0.25 * abs(divergence)

        # 归一化
        total = bear + neutral + bull
        likelihood = np.array([bear, neutral, bull]) / total

        return likelihood, confidence

    def bayesian_update(self, prior: np.ndarray, likelihood: np.ndarray,
                        likelihood_confidence: float) -> np.ndarray:
        """
        【修复9】贝叶斯后验更新
        posterior ∝ prior × likelihood
        """
        # 加权似然（考虑置信度）
        weighted_likelihood = likelihood ** likelihood_confidence

        # 后验 = 先验 × 似然
        posterior = prior * weighted_likelihood

        # 归一化
        posterior = posterior / posterior.sum()

        return posterior

    def fuse_predictions(self, lgbm_proba: np.ndarray, traditional_signals: Dict) -> np.ndarray:
        """
        【修复9】贝叶斯加权融合 - 真正的贝叶斯后验更新
        【修复16】处理valid=False的信号
        """
        _bb_print("[BayesianFusion] 开始贝叶斯后验更新...")

        # 先验 = LightGBM输出的概率
        prior = lgbm_proba.copy()
        posterior = prior.copy()

        # 计算各传统模型的似然并进行贝叶斯更新
        signal_models = [
            ('garch', traditional_signals.get('garch_vol_percentile', 0.5),
             traditional_signals.get('garch_confidence', 0.5)),
            ('ewma', traditional_signals.get('ewma_trend_direction', 0),
             traditional_signals.get('ewma_confidence', 0.5)),
            ('ma', traditional_signals.get('ma_alignment_score', 50),
             traditional_signals.get('ma_confidence', 0.5)),
            ('vp', traditional_signals.get('vp_divergence_score', 0),
             traditional_signals.get('vp_confidence', 0.5)),
        ]

        # 【修复16】只使用valid=True的信号
        valid_signals_count = 0
        for signal_name, signal_value, signal_confidence in signal_models:
            # 检查信号是否有效
            # 【P0-3修复】每个信号单独检查valid
            signal_valid_key = f'{signal_name}_valid'
            if not traditional_signals.get(signal_valid_key, True):
                continue

            # 获取似然
            likelihood, model_conf = self.bayesian_likelihood(signal_value, signal_name)

            # 考虑模型自身的置信度
            combined_confidence = (signal_confidence + model_conf) / 2

            # 贝叶斯更新
            posterior = self.bayesian_update(posterior, likelihood, combined_confidence)
            valid_signals_count += 1

        # ===== 【v1.8】注入实时信号（mootdx盘中数据）=====
        # 市场广度信号
        rt_market_breadth = traditional_signals.get('realtime_breadth')
        if rt_market_breadth is not None:
            up_ratio = rt_market_breadth.get('up_ratio', 0.5)
            likelihood, model_conf = self.bayesian_likelihood(up_ratio, 'realtime_breadth')
            # 实时信号置信度适当降低（盘中可能变化）
            rt_conf = min(0.7, model_conf * 0.85)
            posterior = self.bayesian_update(posterior, likelihood, rt_conf)
            valid_signals_count += 1
            _bb_print(f"[BayesianFusion] 实时市场广度信号已注入: up_ratio={up_ratio:.2f}")

        # 盘中动量信号
        rt_momentum = traditional_signals.get('realtime_momentum')
        if rt_momentum is not None:
            momentum_val = rt_momentum.get('momentum', 0)
            likelihood, model_conf = self.bayesian_likelihood(momentum_val, 'realtime_momentum')
            rt_conf = min(0.65, model_conf * 0.8)
            posterior = self.bayesian_update(posterior, likelihood, rt_conf)
            valid_signals_count += 1
            _bb_print(f"[BayesianFusion] 实时动量信号已注入: momentum={momentum_val:.2f}")

        # 板块内涨跌分布信号
        rt_sector_breadth = traditional_signals.get('realtime_sector_breadth')
        if rt_sector_breadth is not None:
            up_ratio = rt_sector_breadth.get('up_ratio', 0.5)
            likelihood, model_conf = self.bayesian_likelihood(up_ratio, 'realtime_sector_breadth')
            rt_conf = min(0.65, model_conf * 0.8)
            posterior = self.bayesian_update(posterior, likelihood, rt_conf)
            valid_signals_count += 1

        # 【修复16】如果所有信号都无效，结果就等于LightGBM输出
        if valid_signals_count == 0:
            _bb_print("[BayesianFusion] 所有传统信号无效，使用纯LightGBM预测")
            posterior = lgbm_proba

        # 确保概率有效
        posterior = np.maximum(posterior, 0.01)
        posterior = posterior / posterior.sum()

        _bb_print(
            f"[BayesianFusion] 贝叶斯融合结果 - 熊市: {posterior[0]:.1%}, 震荡: {posterior[1]:.1%}, 牛市: {posterior[2]:.1%}")
        return posterior

    def generate_horizon_predictions(self, base_proba: np.ndarray,
                                     proba_2d: np.ndarray = None,
                                     proba_3d: np.ndarray = None,
                                     proba_subtype: np.ndarray = None,
                                     returns: pd.Series = None) -> Dict:
        """
        生成1-5日的预测概率 + 走势细分
        - 1日: 直接使用贝叶斯融合结果
        - 2日: 使用2日模型预测结果（如果有）
        - 3日: 使用3日模型预测结果（如果有）
        - 4-5日: 使用自相关衰减外推
        - 走势细分: 5分类概率
        """
        SUBTYPE_NAMES = ['高开高走', '低开高走', '震荡', '高开低走', '低开低走']
        horizon_predictions = {}

        # 1日预测 - 直接使用
        horizon_predictions[1] = {
            '熊市概率': round(base_proba[0], 4),
            '震荡概率': round(base_proba[1], 4),
            '牛市概率': round(base_proba[2], 4)
        }

        # 2日预测 - 使用2日模型结果
        if proba_2d is not None:
            horizon_predictions[2] = {
                '熊市概率': round(proba_2d[0], 4),
                '震荡概率': round(proba_2d[1], 4),
                '牛市概率': round(proba_2d[2], 4)
            }
        else:
            # 如果没有2日模型结果，使用衰减
            if returns is not None and len(self.autocorr_decay) > 0:
                decay_2d = self.autocorr_decay.get(2, 0.3)
            else:
                decay_2d = 0.3

            bear = base_proba[0] * (1 - decay_2d * 0.5)
            bull = base_proba[2] * (1 - decay_2d * 0.5)
            neutral = 1 - bear - bull
            total = bear + neutral + bull

            horizon_predictions[2] = {
                '熊市概率': round(bear / total, 4),
                '震荡概率': round(neutral / total, 4),
                '牛市概率': round(bull / total, 4)
            }

        # 3日预测 - 使用3日模型结果
        if proba_3d is not None:
            horizon_predictions[3] = {
                '熊市概率': round(proba_3d[0], 4),
                '震荡概率': round(proba_3d[1], 4),
                '牛市概率': round(proba_3d[2], 4)
            }
        else:
            # 如果没有3日模型结果，使用衰减
            if returns is not None and len(self.autocorr_decay) > 0:
                decay_3d = self.autocorr_decay.get(3, 0.2)
            else:
                decay_3d = 0.2

            bear = base_proba[0] * (1 - decay_3d * 0.5)
            bull = base_proba[2] * (1 - decay_3d * 0.5)
            neutral = 1 - bear - bull
            total = bear + neutral + bull

            horizon_predictions[3] = {
                '熊市概率': round(bear / total, 4),
                '震荡概率': round(neutral / total, 4),
                '牛市概率': round(bull / total, 4)
            }

        # 4-5日预测 - 使用自相关衰减外推
        if returns is not None and len(self.autocorr_decay) > 0:
            for horizon in [4, 5]:
                decay = self.autocorr_decay.get(horizon, 0.15)
                decay_factor = max(0.1, decay) if decay > 0 else 0.1

                bear = base_proba[0] * (1 - decay_factor * (horizon - 1) * 0.3)
                bull = base_proba[2] * (1 - decay_factor * (horizon - 1) * 0.3)
                neutral = 1 - bear - bull

                total = bear + neutral + bull
                horizon_predictions[horizon] = {
                    '熊市概率': round(bear / total, 4),
                    '震荡概率': round(neutral / total, 4),
                    '牛市概率': round(bull / total, 4)
                }
        else:
            # 降级：使用固定衰减
            for horizon in [4, 5]:
                decay_factor = 0.15 * (horizon - 1)
                bear = base_proba[0] * (1 - decay_factor)
                bull = base_proba[2] * (1 - decay_factor)
                neutral = 1 - bear - bull

                total = bear + neutral + bull
                horizon_predictions[horizon] = {
                    '熊市概率': round(bear / total, 4),
                    '震荡概率': round(neutral / total, 4),
                    '牛市概率': round(bull / total, 4)
                }

        # 走势细分预测
        if proba_subtype is not None:
            horizon_predictions['走势细分'] = {
                SUBTYPE_NAMES[i]: round(proba_subtype[i], 4) for i in range(len(SUBTYPE_NAMES))
            }

        return horizon_predictions


# ============================================================================
# 模块6：ResultExporter - 结果输出模块
# ============================================================================


# ============================================================================
# 模块5.5：BB_RegimeAnalyzer - 牛熊状态持续性 & 转折推演模块
# 设计意图：基于1-5日概率曲线推演"当前状态会持续多久、何时转折"
# 核心算法：马尔可夫转移概率 + 概率衰减外推 + 趋势加速度
# ============================================================================



# ============================================================================
# 模块8：元叙事市场模拟 (MetaNarrative Module) [v2.0新增]
# 从彩票博弈项目元叙事架构迁移：庄家->做市商，彩民Agent->交易Agent
# ============================================================================

class MarketEcosystem:
    """市场生态系统模拟"""
    MAKER_STRATEGIES = ['institutional', 'retail_driven', 'mixed', 'contrarian', 'momentum']
    
    def __init__(self, maker_strategy="institutional", n_agents=50, seed=42):
        self.maker_strategy = maker_strategy
        self.rng = np.random.RandomState(seed)
        types = TradingAgent.AGENT_TYPES
        self.agents = [TradingAgent(types[i % len(types)], seed=seed+i) for i in range(n_agents)]
        self.history = []
        self.round_count = 0
    
    def simulate_market_impact(self, market_state, agent_signals):
        self.round_count += 1
        buy_p = sum(1 for s in agent_signals if s > 0.3)
        sell_p = sum(1 for s in agent_signals if s < -0.3)
        total = len(agent_signals) + 1e-10
        net = (buy_p - sell_p) / total
        if self.maker_strategy == 'institutional':
            maker_signal = -net * 0.3
        elif self.maker_strategy == 'retail_driven':
            maker_signal = net * 0.1
        else:
            maker_signal = 0
        return {'buy_pressure': buy_p/total, 'sell_pressure': sell_p/total,
                'maker_signal': maker_signal, 'net_sentiment': net}
    
    def evolve_agents(self, performance_scores):
        threshold = sorted(performance_scores)[int(len(performance_scores) * 0.3)]
        survivors = [a for a, s in zip(self.agents, performance_scores) if s >= threshold]
        while len(survivors) < len(self.agents):
            parent = self.rng.choice(survivors)
            child = TradingAgent(parent.agent_type, seed=self.rng.randint(0, 100000))
            child.weights = parent.weights + self.rng.randn(len(parent.weights)) * 0.1
            child.weights = np.clip(child.weights, -5, 5)
            survivors.append(child)
        self.agents = survivors


class TradingAgent:
    """交易Agent - 从彩票MetaAgent迁移"""
    AGENT_TYPES = ['momentum', 'contrarian', 'mean_revert', 'breakout', 'value',
                   'growth', 'quant', 'sentiment', 'macro', 'adaptive']
    N_FEATURES = 20
    
    def __init__(self, agent_type, seed=42):
        self.agent_type = agent_type
        self.rng = np.random.RandomState(seed)
        self.weights = self._init_weights()
        self.temperature = 0.5 + self.rng.random() * 0.3
        self.confidence = 0.5 + self.rng.random() * 0.3
        self.fitness = 0.0
    
    def _init_weights(self):
        w = self.rng.randn(self.N_FEATURES) * 0.2
        if self.agent_type == 'momentum':
            w[:5] = [0.5, 0.8, 0.6, 0.3, 0.4]
        elif self.agent_type == 'contrarian':
            w[:5] = [-0.6, -0.8, -0.4, 0.2, 0.3]
        elif self.agent_type == 'mean_revert':
            w[:5] = [-0.3, -0.5, 0.7, 0.1, 0.2]
        elif self.agent_type == 'value':
            w[:5] = [0.1, 0.2, 0.1, 0.8, 0.5]
        elif self.agent_type == 'adaptive':
            w = self.rng.randn(self.N_FEATURES) * 0.1
        return np.clip(w, -3, 3)
    
    def predict_signal(self, features):
        if len(features) < self.N_FEATURES:
            features = np.pad(features, (0, self.N_FEATURES - len(features)))
        score = np.dot(features[:self.N_FEATURES], self.weights)
        signal = np.tanh(score / self.temperature)
        self._last_signal = signal
        return signal
    
    def update_fitness(self, actual_return):
        if hasattr(self, '_last_signal'):
            reward = self._last_signal * actual_return
            self.fitness = 0.95 * self.fitness + 0.05 * reward


class MetaSignalGenerator:
    """元叙事信号生成器 - 集成到贝叶斯融合流程"""
    
    def __init__(self, n_ecosystems=3, n_agents_per_eco=30, seed=42):
        self.ecosystems = []
        strategies = MarketEcosystem.MAKER_STRATEGIES[:n_ecosystems]
        for i, strategy in enumerate(strategies):
            eco = MarketEcosystem(strategy, n_agents=n_agents_per_eco, seed=seed+i*100)
            self.ecosystems.append(eco)
        self.is_trained = False
    
    def generate_signal(self, features, market_state=None):
        all_signals = []
        eco_signals = []
        for eco in self.ecosystems:
            agent_signals = [a.predict_signal(features) for a in eco.agents]
            impact = eco.simulate_market_impact(market_state or {}, agent_signals)
            mean_signal = np.mean(agent_signals)
            weighted_signal = mean_signal + impact.get('maker_signal', 0)
            all_signals.append(weighted_signal)
            eco_signals.append({
                'strategy': eco.maker_strategy,
                'signal': weighted_signal,
                'divergence': float(np.std(agent_signals)),
                'buy_pressure': impact.get('buy_pressure', 0.5)
            })
        combined = np.mean(all_signals)
        confidence = 1.0 / (1.0 + np.std(all_signals))
        return {'signal': combined, 'confidence': confidence, 'ecosystem_signals': eco_signals}

class BB_RegimeAnalyzer:
    """
    牛熊状态持续性分析器
    输入: generate_horizon_predictions() 的1-5日概率 + 实时信号
    输出: {current_state, duration_estimate, transition_point, confidence, narrative}
    """

    # 基于A股历史的经验参数：牛/熊市平均持续交易日
    EMPIRICAL_DURATION = {
        '牛市': {'mean': 45, 'std': 20},   # 约2个月
        '熊市': {'mean': 35, 'std': 15},    # 约1.5个月
        '震荡': {'mean': 25, 'std': 10},    # 约1个月
    }

    # 马尔可夫转移概率矩阵（基于A股历史统计）
    #        → 牛市  震荡  熊市
    TRANSITION_MATRIX = {
        '牛市': {'牛市': 0.75, '震荡': 0.20, '熊市': 0.05},
        '震荡': {'牛市': 0.20, '震荡': 0.55, '熊市': 0.25},
        '熊市': {'牛市': 0.05, '震荡': 0.20, '熊市': 0.75},
    }

    # 状态中文描述映射
    STATE_NARRATIVE = {
        '牛市': {
            'strong': '强势上涨，趋势稳固',
            'moderate': '上涨趋势延续，但动能有所减弱',
            'weakening': '上涨动能明显衰减，注意高位风险',
        },
        '熊市': {
            'strong': '下跌趋势强劲，建议规避',
            'moderate': '下跌延续，但恐慌情绪有所缓解',
            'weakening': '跌势放缓，可能接近底部区域',
        },
        '震荡': {
            'bull_biased': '偏强震荡，向上突破概率较大',
            'neutral': '多空拉锯，方向不明',
            'bear_biased': '偏弱震荡，向下破位风险较大',
        },
    }

    def analyze(self, horizon_predictions, sector_name='市场整体', realtime_signals=None):
        """
        核心分析方法：综合1-5日概率 + 实时信号 → 状态持续性 + 转折推演

        参数:
            horizon_predictions: {1: {'牛市概率':..., '熊市概率':..., '震荡概率':...}, 2:..., ...}
            sector_name: 板块名称
            realtime_signals: BB_RealtimeAnalyzer.get_realtime_enhanced_signals() 的输出

        返回:
            {
                'current_state': str,           # 当前状态 '牛市'/'熊市'/'震荡'
                'state_strength': str,          # 状态强度 'strong'/'moderate'/'weakening'
                'duration_days': int,           # 预计当前状态还能持续多少个交易日
                'duration_range': (int, int),   # 持续时间的置信区间 (low, high)
                'transition_to': str,           # 最可能的转折方向 '震荡'/'熊市'/'牛市'
                'transition_prob': float,       # 转折概率
                'transition_point': str,        # 转折时间描述 "约3-5个交易日后"
                'narrative': str,               # 自然语言描述
                'risk_hint': str,               # 风险提示
            }
        """
        # 1. 判断当前状态
        d1 = horizon_predictions.get(1, {})
        bull_p = d1.get('牛市概率', 0.33)
        bear_p = d1.get('熊市概率', 0.33)
        neutral_p = d1.get('震荡概率', 0.34)

        if bull_p > bear_p + 0.1 and bull_p > 0.4:
            current_state = '牛市'
        elif bear_p > bull_p + 0.1 and bear_p > 0.4:
            current_state = '熊市'
        else:
            current_state = '震荡'

        # 2. 分析状态强度（看1→5日概率衰减速率）
        state_strength = self._assess_strength(horizon_predictions, current_state)

        # 3. 计算预计持续时间
        duration_days, duration_range = self._estimate_duration(
            horizon_predictions, current_state, state_strength, realtime_signals
        )

        # 4. 推演转折方向
        transition_to, transition_prob = self._predict_transition(
            horizon_predictions, current_state, state_strength
        )

        # 5. 生成转折时间描述
        transition_point = self._format_transition_point(duration_days, duration_range, transition_to)

        # 6. 生成叙述
        narrative = self._generate_narrative(
            sector_name, current_state, state_strength,
            duration_days, transition_to, transition_prob, transition_point,
            horizon_predictions, realtime_signals
        )

        # 7. 风险提示
        risk_hint = self._generate_risk_hint(
            current_state, state_strength, transition_to, transition_prob
        )

        return {
            'current_state': current_state,
            'state_strength': state_strength,
            'duration_days': duration_days,
            'duration_range': duration_range,
            'transition_to': transition_to,
            'transition_prob': round(transition_prob, 3),
            'transition_point': transition_point,
            'narrative': narrative,
            'risk_hint': risk_hint,
        }

    def _assess_strength(self, horizon_predictions, current_state):
        """
        判断状态强度
        算法：看1日概率的绝对值 + 1→3日概率衰减速率
        - strong: 当前状态概率>60% 且衰减慢
        - moderate: 当前状态概率40-60% 或衰减中等
        - weakening: 当前状态概率接近阈值 或衰减快
        """
        d1 = horizon_predictions.get(1, {})
        d3 = horizon_predictions.get(3, {})

        state_key = f'{current_state}概率'
        p1 = d1.get(state_key, 0.33)
        p3 = d3.get(state_key, 0.33)

        # 衰减率
        decay_rate = (p1 - p3) / max(p1, 0.01) if p1 > 0 else 0

        if p1 > 0.6 and decay_rate < 0.15:
            return 'strong'
        elif p1 > 0.5 and decay_rate < 0.3:
            return 'moderate'
        elif p1 < 0.45 or decay_rate > 0.4:
            return 'weakening'
        else:
            return 'moderate'

    def _estimate_duration(self, horizon_predictions, current_state, state_strength, realtime_signals):
        """
        估算当前状态还能持续多少交易日
        方法：马尔可夫链外推 + 经验修正
        """
        empirical = self.EMPIRICAL_DURATION[current_state]
        base_duration = empirical['mean']
        base_std = empirical['std']

        # 1. 基于概率衰减的修正
        state_key = f'{current_state}概率'
        d1 = horizon_predictions.get(1, {})
        d3 = horizon_predictions.get(3, {})
        d5 = horizon_predictions.get(5, {})

        p1 = d1.get(state_key, 0.33)
        p3 = d3.get(state_key, 0.33)
        p5 = d5.get(state_key, 0.33)

        # 如果5日概率仍然>50%，状态稳固，加长持续时间
        if p5 > 0.5:
            prob_factor = 1.3
        elif p5 > 0.4:
            prob_factor = 1.0
        elif p5 > 0.3:
            prob_factor = 0.7
        else:
            prob_factor = 0.5

        # 2. 强度修正
        strength_factor = {'strong': 1.3, 'moderate': 1.0, 'weakening': 0.6}
        sf = strength_factor.get(state_strength, 1.0)

        # 3. 实时信号修正
        rt_factor = 1.0
        if realtime_signals and realtime_signals.get('realtime_available'):
            # 如果市场广度支持当前状态，加长
            breadth = realtime_signals.get('market_breadth')
            if breadth:
                up_ratio = breadth.get('up_ratio', 0.5)
                if current_state == '牛市' and up_ratio > 0.6:
                    rt_factor = 1.15
                elif current_state == '熊市' and up_ratio < 0.35:
                    rt_factor = 1.15
                elif current_state == '牛市' and up_ratio < 0.45:
                    rt_factor = 0.85  # 涨少跌多，牛市可能不稳固
                elif current_state == '熊市' and up_ratio > 0.5:
                    rt_factor = 0.85  # 跌少涨多，熊市可能要结束

        # 4. 马尔可夫外推：逐日计算状态保持概率
        markov_duration = self._markov_duration(current_state, state_key, p1)

        # 综合估算
        duration = int(base_duration * prob_factor * sf * rt_factor)
        duration = max(3, min(duration, 60))  # 3-60天范围

        # 置信区间
        low = max(2, duration - int(base_std * sf))
        high = min(90, duration + int(base_std * sf))

        # 如果马尔可夫外推明显更短，取两者加权
        if markov_duration < duration * 0.7:
            duration = int(duration * 0.6 + markov_duration * 0.4)
            low = max(2, duration - 5)
            high = min(90, duration + 10)

        return duration, (low, high)

    def _markov_duration(self, current_state, state_key, p1):
        """
        马尔可夫链外推：从当前状态出发，计算预期保持多少天
        每天有 TRANSITION_MATRIX 的概率转出当前状态
        """
        trans = self.TRANSITION_MATRIX[current_state]
        stay_prob = trans[current_state]

        # 如果1日概率本身就偏低，降低保持概率
        if p1 < 0.5:
            stay_prob *= 0.85
        elif p1 < 0.4:
            stay_prob *= 0.7

        # 期望保持天数 = 1/(1-stay_prob) 的几何分布
        if stay_prob >= 0.99:
            return 100
        expected = 1.0 / (1.0 - stay_prob)
        return int(expected)

    def _predict_transition(self, horizon_predictions, current_state, state_strength):
        """
        预测最可能的转折方向和概率
        方法：看5日概率中哪个"非当前状态"的概率升得最快
        """
        d1 = horizon_predictions.get(1, {})
        d5 = horizon_predictions.get(5, {})

        states = ['牛市', '震荡', '熊市']
        other_states = [s for s in states if s != current_state]

        best_transition = other_states[0]
        best_prob = 0

        for target in other_states:
            key = f'{target}概率'
            p1_target = d1.get(key, 0.33)
            p5_target = d5.get(key, 0.33)

            # 转折概率 = 目标状态5日概率 - 1日概率 的增幅
            delta = p5_target - p1_target
            # 加上马尔可夫先验
            prior = self.TRANSITION_MATRIX[current_state].get(target, 0.2)
            combined_prob = delta * 0.6 + prior * 0.4

            if combined_prob > best_prob:
                best_prob = combined_prob
                best_transition = target

        # 如果状态在减弱，转折概率更高
        if state_strength == 'weakening':
            best_prob = min(1.0, best_prob * 1.3)
        elif state_strength == 'strong':
            best_prob *= 0.7

        return best_transition, best_prob

    def _format_transition_point(self, duration_days, duration_range, transition_to):
        """格式化转折时间描述"""
        low, high = duration_range

        if duration_days <= 3:
            return f"极短期内({duration_days}个交易日内)可能转向{transition_to}"
        elif duration_days <= 7:
            return f"约{low}-{high}个交易日后可能转向{transition_to}"
        elif duration_days <= 15:
            return f"约1-2周后({low}-{high}个交易日)可能转向{transition_to}"
        elif duration_days <= 30:
            return f"约2-4周后可能转向{transition_to}，当前趋势仍有支撑"
        else:
            return f"约{duration_days//5}-{(duration_days+10)//5}周后可能转向{transition_to}，当前趋势较为稳固"

    def _generate_narrative(self, sector_name, current_state, state_strength,
                            duration_days, transition_to, transition_prob,
                            transition_point, horizon_predictions, realtime_signals):
        """生成自然语言分析叙述"""

        # 状态描述
        if current_state == '震荡':
            d1 = horizon_predictions.get(1, {})
            bull_p = d1.get('牛市概率', 0.33)
            bear_p = d1.get('熊市概率', 0.33)
            if bull_p > bear_p:
                state_desc = self.STATE_NARRATIVE['震荡']['bull_biased']
            elif bear_p > bull_p:
                state_desc = self.STATE_NARRATIVE['震荡']['bear_biased']
            else:
                state_desc = self.STATE_NARRATIVE['震荡']['neutral']
        else:
            state_desc = self.STATE_NARRATIVE[current_state].get(state_strength, '')

        narrative = f"{sector_name}当前处于{current_state}状态，{state_desc}。"

        # 持续时间
        if duration_days <= 5:
            narrative += f"预计该状态还能持续约{duration_days}个交易日。"
        elif duration_days <= 15:
            narrative += f"预计该状态还能持续约{duration_days}个交易日（约1-3周）。"
        else:
            narrative += f"预计该状态还能持续约{duration_days}个交易日（约{duration_days//5}-{(duration_days+5)//5}周）。"

        # 转折方向
        if transition_prob > 0.3:
            narrative += f" {transition_point}。"
        else:
            narrative += f" 短期内转向{transition_to}的可能性较低({transition_prob:.0%})。"

        # 实时信号补充
        if realtime_signals and realtime_signals.get('realtime_available'):
            breadth = realtime_signals.get('market_breadth')
            if breadth:
                up, down = breadth.get('up', 0), breadth.get('down', 0)
                total = up + down + breadth.get('flat', 0)
                if total > 0:
                    narrative += f" 盘中实时涨跌比{up}:{down}（上涨占比{up/total:.0%}）。"

            # 盘中动量
            momentum = realtime_signals.get('intraday_momentum', {})
            for idx_name, mom in momentum.items():
                if mom:
                    narrative += f" {idx_name}{mom.get('signal', '')}。"

        return narrative

    def _generate_risk_hint(self, current_state, state_strength, transition_to, transition_prob):
        """生成风险提示"""
        hints = []

        if current_state == '牛市' and state_strength == 'weakening':
            hints.append("上涨动能衰减，注意获利回吐风险")
        elif current_state == '牛市' and transition_to == '熊市' and transition_prob > 0.25:
            hints.append("牛转熊信号出现，建议逐步减仓锁定利润")

        if current_state == '熊市' and state_strength == 'strong':
            hints.append("下跌趋势强劲，切勿抄底")
        elif current_state == '熊市' and state_strength == 'weakening':
            hints.append("跌势放缓但尚未反转，观望为主")

        if current_state == '震荡' and transition_prob > 0.3:
            if transition_to == '熊市':
                hints.append("震荡偏弱，向下破位风险较大")
            elif transition_to == '牛市':
                hints.append("震荡偏强，可能酝酿向上突破")

        if not hints:
            if current_state == '牛市':
                hints.append("趋势尚可，但需关注量能配合")
            elif current_state == '熊市':
                hints.append("保持低仓位，等待明确信号")
            else:
                hints.append("方向不明，控制仓位")

        return '；'.join(hints)


# 全局实例
BB_REGIME_ANALYZER = BB_RegimeAnalyzer()

class BB_ResultExporter:
    """
    结果输出模块 v1.3
    功能：输出预测结果到CSV和PDF可视化报告
    新增：板块走势PDF + 市场整体PDF
    """

    def __init__(self):
        _bb_print("[ResultExporter] 初始化结果输出模块...")
        self.output_dir = OUTPUT_PATH
        self.ensure_output_dir()

    def ensure_output_dir(self):
        """确保输出目录存在"""
        try:
            if not os.path.exists(self.output_dir):
                os.makedirs(self.output_dir)
                _bb_print(f"[ResultExporter] 创建输出目录: {self.output_dir}")
            else:
                _bb_print(f"[ResultExporter] 输出目录已存在: {self.output_dir}")
        except Exception as e:
            _bb_print(f"[ResultExporter] 创建目录异常: {str(e)}，使用当前目录")
            self.output_dir = '.'

    def export_prediction_csv(self, all_predictions: Dict):
        """导出牛熊预测结果CSV（含走势细分）"""
        try:
            rows = []
            for sector_name, horizon_data in all_predictions.items():
                for horizon, probs in horizon_data.items():
                    if horizon == '走势细分':
                        # 走势细分单独一行
                        rows.append({
                            '板块名称': sector_name,
                            '预测天数': '走势细分',
                            '熊市概率': '-',
                            '震荡概率': '-',
                            '牛市概率': '-',
                            '牛熊差': '-',
                            '预测结论': f"高开高走{probs.get('高开高走', 0):.0%} 低开高走{probs.get('低开高走', 0):.0%} 震荡{probs.get('震荡', 0):.0%} 高开低走{probs.get('高开低走', 0):.0%} 低开低走{probs.get('低开低走', 0):.0%}"
                        })
                    else:
                        rows.append({
                            '板块名称': sector_name,
                            '预测天数': f'{horizon}日',
                            '熊市概率': f"{probs['熊市概率']:.2%}",
                            '震荡概率': f"{probs['震荡概率']:.2%}",
                            '牛市概率': f"{probs['牛市概率']:.2%}",
                            '牛熊差': f"{probs['牛市概率'] - probs['熊市概率']:.2%}",
                            '预测结论': '牛市' if probs['牛市概率'] > probs['熊市概率'] + 0.1 else
                            '熊市' if probs['熊市概率'] > probs['牛市概率'] + 0.1 else '震荡'
                        })

            df = pd.DataFrame(rows)
            filepath = os.path.join(self.output_dir, '牛熊预测_最新.csv')
            df.to_csv(filepath, index=False, encoding='utf-8-sig')
            print(f"[ResultExporter] 预测结果已保存: {filepath}")

        except Exception as e:
            _bb_print(f"[ResultExporter] 导出预测CSV异常: {str(e)}")

    def export_feature_importance(self, feature_importance: pd.DataFrame):
        """导出汇总后的特征重要性CSV"""
        try:
            if feature_importance is not None and len(feature_importance) > 0:
                aggregated_importance = feature_importance.groupby('feature')['importance'].mean().reset_index()
                aggregated_importance = aggregated_importance.sort_values('importance', ascending=False).reset_index(
                    drop=True)

                filepath = os.path.join(self.output_dir, '特征重要性.csv')
                aggregated_importance.to_csv(filepath, index=False, encoding='utf-8-sig')
                print(f"[ResultExporter] 特征重要性已保存: {filepath}")
        except Exception as e:
            _bb_print(f"[ResultExporter] 导出特征重要性异常: {str(e)}")

    def export_performance_report(self, performance: Dict, all_predictions: Dict,
                                  all_cv_metrics: List[Dict] = None):
        """导出模型性能报告 - v1.3不再生成TXT，只返回performance"""
        try:
            # 计算所有板块的CV指标均值
            if all_cv_metrics and len(all_cv_metrics) > 0:
                avg_accuracy = np.mean([m['cv_accuracy'] for m in all_cv_metrics])
                avg_precision = np.mean([m['cv_precision'] for m in all_cv_metrics])
                avg_recall = np.mean([m['cv_recall'] for m in all_cv_metrics])
                avg_f1 = np.mean([m['cv_f1'] for m in all_cv_metrics])

                performance = {
                    'cv_accuracy': avg_accuracy,
                    'cv_precision': avg_precision,
                    'cv_recall': avg_recall,
                    'cv_f1': avg_f1
                }
            return performance
        except Exception as e:
            _bb_print(f"[ResultExporter] 导出性能报告异常: {str(e)}")
            return performance

    def export_sector_pdf(self, all_predictions: Dict):
        """生成板块预测走势PDF"""
        if plt_bb is None or PdfPages_bb is None:
            _bb_print("[ResultExporter] matplotlib未安装，跳过板块PDF生成")
            return

        try:
            # 配色方案
            BULL_COLOR = '#E74C3C'  # 牛市红
            BEAR_COLOR = '#27AE60'  # 熊市绿
            NEUTRAL_COLOR = '#95A5A6'  # 震荡灰
            GRID_COLOR = '#E0E0E0'
            TEXT_COLOR = '#2C3E50'
            BG_COLOR = 'white'

            # 准备板块数据（排除市场整体）
            sectors_to_plot = [s for s in all_predictions.keys() if s != '市场整体']

            if not sectors_to_plot:
                _bb_print("[ResultExporter] 无板块数据可绘制")
                return

            filepath = os.path.join(self.output_dir, '牛熊预测_板块走势.pdf')

            # 计算子图布局
            n_sectors = len(sectors_to_plot)
            n_cols = 3
            n_rows = 3  # 3x3布局，7板块+市场整体=8个
            n_pages = (n_sectors + n_cols * n_rows - 1) // (n_cols * n_rows)

            _bb_print(f"[ResultExporter] 生成板块走势PDF: {filepath}")

            with PdfPages_bb(filepath) as pdf:
                for page in range(n_pages):
                    fig, axes = plt_bb.subplots(n_rows, n_cols, figsize=(14, 12))
                    fig.patch.set_facecolor(BG_COLOR)

                    start_idx = page * n_cols * n_rows
                    end_idx = min(start_idx + n_cols * n_rows, n_sectors)
                    page_sectors = sectors_to_plot[start_idx:end_idx]

                    for idx, sector in enumerate(page_sectors):
                        row = idx // n_cols
                        col = idx % n_cols
                        ax = axes[row, col]

                        horizon_data = all_predictions.get(sector, {})
                        if not horizon_data:
                            ax.set_visible(False)
                            continue

                        # 获取1-5日数据
                        horizons = [1, 2, 3, 4, 5]
                        bull_probs = [horizon_data.get(h, {}).get('牛市概率', 0) for h in horizons]
                        bear_probs = [horizon_data.get(h, {}).get('熊市概率', 0) for h in horizons]
                        neutral_probs = [horizon_data.get(h, {}).get('震荡概率', 0) for h in horizons]

                        # 确定1日结论
                        bull_1d = bull_probs[0]
                        bear_1d = bear_probs[0]
                        if bull_1d > bear_1d + 0.1:
                            conclusion = "牛"
                            bg_alpha = 0.02
                        elif bear_1d > bull_1d + 0.1:
                            conclusion = "熊"
                            bg_alpha = 0.02
                        else:
                            conclusion = "震"
                            bg_alpha = 0.0

                        # 设置背景色
                        ax.set_facecolor(BG_COLOR)
                        if conclusion == "牛":
                            ax.set_facecolor('#FFF5F5')
                        elif conclusion == "熊":
                            ax.set_facecolor('#F5FFF5')

                        # 绘制填充区域
                        ax.fill_between(horizons, 0, bull_probs, alpha=0.15, color=BULL_COLOR)
                        ax.fill_between(horizons, 0, bear_probs, alpha=0.15, color=BEAR_COLOR)

                        # 绘制三条线
                        ax.plot(horizons, bull_probs, color=BULL_COLOR, linewidth=2, marker='o', markersize=6,
                                label='牛市')
                        ax.plot(horizons, neutral_probs, color=NEUTRAL_COLOR, linewidth=2, marker='s', markersize=6,
                                label='震荡')
                        ax.plot(horizons, bear_probs, color=BEAR_COLOR, linewidth=2, marker='^', markersize=6,
                                label='熊市')

                        # 标注最大概率点
                        for i, (b, n, br) in enumerate(zip(bull_probs, neutral_probs, bear_probs)):
                            max_prob = max(b, n, br)
                            if max_prob == b:
                                ax.annotate('牛', (horizons[i], b), textcoords="offset points", xytext=(0, 8),
                                            ha='center', fontsize=8, color=BULL_COLOR, fontweight='bold')
                            elif max_prob == br:
                                ax.annotate('熊', (horizons[i], br), textcoords="offset points", xytext=(0, -12),
                                            ha='center', fontsize=8, color=BEAR_COLOR, fontweight='bold')
                            else:
                                ax.annotate('震', (horizons[i], n), textcoords="offset points", xytext=(0, 8),
                                            ha='center', fontsize=8, color=NEUTRAL_COLOR)

                        # 设置子图标题（含走势细分）
                        title_extra = ""
                        subtype_data = horizon_data.get('走势细分', {})
                        if subtype_data:
                            best_sub = max(subtype_data, key=subtype_data.get)
                            title_extra = f" | {best_sub}"
                        ax.set_title(f"{sector} → {conclusion}{title_extra}", fontsize=10, fontweight='bold',
                                     color=TEXT_COLOR)
                        ax.set_xlabel('预测天数', fontsize=8)
                        ax.set_ylabel('概率', fontsize=8)
                        ax.set_xticks(horizons)
                        ax.set_ylim(0, 1)
                        ax.grid(True, alpha=0.3, color=GRID_COLOR)
                        ax.tick_params(colors=TEXT_COLOR)

                        for spine in ax.spines.values():
                            spine.set_color(GRID_COLOR)

                    # 隐藏多余的子图
                    for idx in range(len(page_sectors), n_cols * n_rows):
                        row = idx // n_cols
                        col = idx % n_cols
                        axes[row, col].set_visible(False)

                    # 添加总标题
                    timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M')
                    fig.suptitle(f"A股牛熊预测 - 板块走势\n生成时间: {timestamp}",
                                 fontsize=14, fontweight='bold', color=TEXT_COLOR, y=0.98)

                    # 添加注释
                    fig.text(0.5, 0.02, "数据来源: baostock | 模型: LightGBM + 贝叶斯融合",
                             ha='center', fontsize=9, color='#666666')

                    plt_bb.tight_layout(rect=[0, 0.03, 1, 0.95])
                    pdf.savefig(fig, dpi=150, bbox_inches='tight')
                    plt_bb.close(fig)

            print(f"[ResultExporter] 板块走势PDF已保存: {filepath}")

        except Exception as e:
            print(f"[ResultExporter] 生成板块走势PDF异常: {str(e)}")
            import traceback
            traceback.print_exc()

    def export_market_pdf(self, all_predictions: Dict):
        """生成市场整体预测PDF"""
        if plt_bb is None or PdfPages_bb is None:
            _bb_print("[ResultExporter] matplotlib未安装，跳过市场PDF生成")
            return

        try:
            # 配色方案
            BULL_COLOR = '#E74C3C'
            BEAR_COLOR = '#27AE60'
            NEUTRAL_COLOR = '#95A5A6'
            GRID_COLOR = '#E0E0E0'
            TEXT_COLOR = '#2C3E50'
            BG_COLOR = 'white'

            filepath = os.path.join(self.output_dir, '牛熊预测_市场整体.pdf')
            _bb_print(f"[ResultExporter] 生成市场整体PDF: {filepath}")

            # 获取市场整体数据
            market_data = all_predictions.get('市场整体', {})
            if not market_data:
                _bb_print("[ResultExporter] 无市场整体数据")
                return

            with PdfPages_bb(filepath) as pdf:
                fig = plt_bb.figure(figsize=(12, 16))
                fig.patch.set_facecolor(BG_COLOR)

                # 上半部分：大面积折线图 (60%高度)
                ax1 = fig.add_axes([0.1, 0.45, 0.8, 0.45])

                horizons = [1, 2, 3, 4, 5]
                bull_probs = [market_data.get(h, {}).get('牛市概率', 0) for h in horizons]
                bear_probs = [market_data.get(h, {}).get('熊市概率', 0) for h in horizons]
                neutral_probs = [market_data.get(h, {}).get('震荡概率', 0) for h in horizons]

                # 填充区域
                ax1.fill_between(horizons, 0, bull_probs, alpha=0.2, color=BULL_COLOR, label='_nolegend_')
                ax1.fill_between(horizons, 0, bear_probs, alpha=0.2, color=BEAR_COLOR, label='_nolegend_')

                # 三条粗线
                ax1.plot(horizons, bull_probs, color=BULL_COLOR, linewidth=3, marker='o', markersize=10,
                         label='牛市概率')
                ax1.plot(horizons, neutral_probs, color=NEUTRAL_COLOR, linewidth=3, marker='s', markersize=10,
                         label='震荡概率')
                ax1.plot(horizons, bear_probs, color=BEAR_COLOR, linewidth=3, marker='^', markersize=10,
                         label='熊市概率')

                # 标注具体数值
                for i, (b, n, br) in enumerate(zip(bull_probs, neutral_probs, bear_probs)):
                    ax1.annotate(f'{b:.1%}', (horizons[i], b), textcoords="offset points", xytext=(0, 12),
                                 ha='center', fontsize=9, color=BULL_COLOR, fontweight='bold')
                    ax1.annotate(f'{br:.1%}', (horizons[i], br), textcoords="offset points", xytext=(0, -18),
                                 ha='center', fontsize=9, color=BEAR_COLOR, fontweight='bold')

                ax1.set_title('A股市场整体牛熊概率预测', fontsize=14, fontweight='bold', color=TEXT_COLOR, pad=10)
                ax1.set_xlabel('预测天数', fontsize=11)
                ax1.set_ylabel('概率', fontsize=11)
                ax1.set_xticks(horizons)
                ax1.set_ylim(0, 1.1)
                ax1.grid(True, alpha=0.4, color=GRID_COLOR)
                ax1.legend(loc='upper right', fontsize=10)
                ax1.tick_params(colors=TEXT_COLOR)

                for spine in ax1.spines.values():
                    spine.set_color(GRID_COLOR)

                # 下半部分左：水平条形图 (20%高度)
                ax2 = fig.add_axes([0.1, 0.15, 0.35, 0.2])

                # 收集各板块1日数据
                sectors_1d = []
                for sector in all_predictions.keys():
                    if sector != '市场整体':
                        data = all_predictions[sector].get(1, {})
                        if data:
                            sectors_1d.append({
                                'name': sector,
                                'bull': data.get('牛市概率', 0),
                                'bear': data.get('熊市概率', 0)
                            })

                # 按牛市概率排序
                sectors_1d.sort(key=lambda x: x['bull'], reverse=True)

                if sectors_1d:
                    y_pos = np.arange(len(sectors_1d))
                    bar_height = 0.35

                    ax2.barh(y_pos - bar_height / 2, [s['bull'] for s in sectors_1d], bar_height,
                             color=BULL_COLOR, alpha=0.8, label='牛市')
                    ax2.barh(y_pos + bar_height / 2, [s['bear'] for s in sectors_1d], bar_height,
                             color=BEAR_COLOR, alpha=0.8, label='熊市')

                    ax2.set_yticks(y_pos)
                    ax2.set_yticklabels([s['name'] for s in sectors_1d], fontsize=8)
                    ax2.set_xlabel('概率', fontsize=9)
                    ax2.set_title('各板块1日牛熊概率对比', fontsize=11, fontweight='bold', color=TEXT_COLOR)
                    ax2.set_xlim(0, 1)
                    ax2.grid(True, alpha=0.3, axis='x', color=GRID_COLOR)
                    ax2.legend(loc='lower right', fontsize=8)
                    ax2.tick_params(colors=TEXT_COLOR)

                    for spine in ax2.spines.values():
                        spine.set_color(GRID_COLOR)

                # 下半部分右：热力图 (20%高度)
                ax3 = fig.add_axes([0.55, 0.15, 0.35, 0.2])

                # 准备热力图数据
                heatmap_data = []
                heatmap_labels = []
                for sector in sectors_1d:
                    row = []
                    for h in horizons:
                        data = all_predictions[sector['name']].get(h, {})
                        diff = data.get('牛市概率', 0) - data.get('熊市概率', 0)
                        row.append(diff)
                    heatmap_data.append(row)
                    heatmap_labels.append(sector['name'])

                if heatmap_data:
                    heatmap_array = np.array(heatmap_data)

                    # 绘制热力图
                    im = ax3.imshow(heatmap_array, cmap='RdYlGn', aspect='auto', vmin=-0.5, vmax=0.5)

                    # 设置标签
                    ax3.set_xticks(np.arange(len(horizons)))
                    ax3.set_xticklabels([f'{h}日' for h in horizons], fontsize=9)
                    ax3.set_yticks(np.arange(len(heatmap_labels)))
                    ax3.set_yticklabels(heatmap_labels, fontsize=8)
                    ax3.set_title('板块×预测期 牛熊差热力图', fontsize=11, fontweight='bold', color=TEXT_COLOR)

                    # 添加数值标注
                    for i in range(len(heatmap_labels)):
                        for j in range(len(horizons)):
                            val = heatmap_array[i, j]
                            color = 'white' if abs(val) > 0.25 else TEXT_COLOR
                            ax3.text(j, i, f'{val:.0%}', ha='center', va='center',
                                     fontsize=7, color=color, fontweight='bold')

                    # 添加颜色条
                    cbar = plt_bb.colorbar(im, ax=ax3, orientation='vertical', fraction=0.05, pad=0.02)
                    cbar.set_label('牛熊差', fontsize=8)

                # 添加总标题
                timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M')
                fig.suptitle(f"A股牛熊预测 - 市场整体分析\n生成时间: {timestamp}",
                             fontsize=16, fontweight='bold', color=TEXT_COLOR, y=0.98)

                # 添加注释
                fig.text(0.5, 0.02, "数据来源: baostock | 模型: LightGBM + 贝叶斯融合",
                         ha='center', fontsize=9, color='#666666')

                plt_bb.tight_layout(rect=[0, 0.03, 1, 0.96])
                pdf.savefig(fig, dpi=150, bbox_inches='tight')
                plt_bb.close(fig)

            print(f"[ResultExporter] 市场整体PDF已保存: {filepath}")

        except Exception as e:
            print(f"[ResultExporter] 生成市场整体PDF异常: {str(e)}")
            import traceback
            traceback.print_exc()


# ============================================================================
# 主函数
# ============================================================================


def run_bull_bear_prediction():
    """
    牛熊预测入口函数 — 作为回测系统的阶段0
    在个股选股前先判断市场环境，结果写入全局变量MARKET_ENV
    输出：控制台精简摘要 + CSV + PDF
    """
    global MARKET_ENV

    print("\n" + "━" * 70)
    print("  阶段0 ▸ 市场牛熊环境预判")
    print("━" * 70)

    try:
        # 1. 数据获取
        data_fetcher = BB_DataFetcher()
        all_data = data_fetcher.fetch_all_data()

        if not all_data:
            print("  ⚠ 牛熊预测数据获取失败，跳过（后续模块不受影响）")
            return

        # 2. 初始化模块
        feature_extractor = BB_FeatureExtractor()
        traditional_models = BB_TraditionalModels()
        lgbm_predictor = BB_LGBMPredictor()
        bayesian_fusion = BB_BayesianFusion()
        result_exporter = BB_ResultExporter()

        # 3. 获取基准
        benchmark = all_data.get('sh.000300', list(all_data.values())[0])

        # 4. 全局训练数据
        _bb_print("\n[牛熊] 准备全局训练数据...")
        global_features_list = []
        global_df_list = []

        for sector in ALL_SECTORS_BB:
            if sector in all_data and len(all_data[sector]) > 100:
                df = all_data[sector]
                features = feature_extractor.extract_all_features(df, benchmark, all_data)
                global_features_list.append(features)
                global_df_list.append(df)

        global_features = pd.concat(global_features_list, ignore_index=True) if global_features_list else None
        global_df = pd.concat(global_df_list, ignore_index=True) if global_df_list else None

        if global_df is not None and 'close' in global_df.columns:
            global_returns = global_df['close'].pct_change().dropna()
            bayesian_fusion.calculate_autocorrelation_decay(global_returns)

        # 4.5 【v1.8】获取mootdx实时信号
        realtime_signals = BB_REALTIME.get_realtime_enhanced_signals()
        if realtime_signals.get('realtime_available'):
            breadth = realtime_signals.get('market_breadth', {})
            if breadth:
                total = breadth.get('up', 0) + breadth.get('down', 0) + breadth.get('flat', 0)
                print(f"  ✓ 实时行情已接入（样本{total}只 涨{breadth.get('up', '?')} 跌{breadth.get('down', '?')} "
                      f"上涨占比{breadth.get('up_ratio', 0):.0%}）")
        else:
            print("  ℹ 实时行情未接入（非交易时段或mootdx未连接）")

        # 5. 逐板块预测
        all_predictions = {}
        regime_analysis = {}  # 【v1.8】状态推演结果
        sectors_to_process = ALL_SECTORS_BB + ['市场整体']

        print(f"  预测 {len(sectors_to_process)} 个板块...", flush=True)

        for sector in sectors_to_process:
            try:
                if sector == '市场整体':
                    df = all_data.get('sh.000300', list(all_data.values())[0])
                else:
                    df = all_data.get(sector)

                if df is None or len(df) < 100:
                    continue

                features = feature_extractor.extract_all_features(df, benchmark, all_data)
                traditional_signals = traditional_models.get_all_signals(df)
                bayesian_fusion.ewma_strength = traditional_signals.get('ewma_trend_strength', 0.5)

                # 【v1.8】注入实时信号到传统信号字典
                if realtime_signals.get('realtime_available'):
                    # 市场广度
                    if realtime_signals.get('market_breadth'):
                        traditional_signals['realtime_breadth'] = realtime_signals['market_breadth']
                    # 板块涨跌分布
                    if sector in realtime_signals.get('sector_breadth', {}):
                        traditional_signals['realtime_sector_breadth'] = realtime_signals['sector_breadth'][sector]
                    # 盘中动量（取第一个可用的指数动量）
                    for _idx_name, _mom in realtime_signals.get('intraday_momentum', {}).items():
                        if _mom:
                            traditional_signals['realtime_momentum'] = _mom
                            break

                lgbm_proba, proba_2d, proba_3d, proba_subtype, feature_importance = lgbm_predictor.train_and_predict(
                    features, df, global_features, global_df
                )

                fused_proba = bayesian_fusion.fuse_predictions(lgbm_proba, traditional_signals)

                returns = df['close'].pct_change().dropna()
                all_predictions[sector] = bayesian_fusion.generate_horizon_predictions(
                    fused_proba, proba_2d, proba_3d, proba_subtype, returns
                )

                # 【v1.8】牛熊状态推演
                try:
                    regime_result = BB_REGIME_ANALYZER.analyze(
                        all_predictions[sector], sector, realtime_signals
                    )
                    regime_analysis[sector] = regime_result
                except Exception:
                    pass

            except Exception as e:
                _bb_print(f"✗ {sector} 预测异常: {str(e)}")
                continue

        if not all_predictions:
            print("  ⚠ 所有板块预测失败，跳过")
            return

        # 6. 写入全局变量
        market_data = all_predictions.get('市场整体', {}).get(1, {})
        bull_prob = market_data.get('牛市概率', 0.33)
        bear_prob = market_data.get('熊市概率', 0.33)
        neutral_prob = market_data.get('震荡概率', 0.34)
        confidence = abs(bull_prob - bear_prob)

        if bull_prob > bear_prob + 0.1:
            regime = '牛市'
        elif bear_prob > bull_prob + 0.1:
            regime = '熊市'
        else:
            regime = '震荡'

        # 收集板块信号
        sector_signals = {}
        for sector_name, horizon_data in all_predictions.items():
            if sector_name == '市场整体':
                continue
            d1 = horizon_data.get(1, {})
            sector_signals[sector_name] = {
                'regime': '牛市' if d1.get('牛市概率', 0) > d1.get('熊市概率', 0) + 0.1 else
                '熊市' if d1.get('熊市概率', 0) > d1.get('牛市概率', 0) + 0.1 else '震荡',
                'bull_prob': d1.get('牛市概率', 0),
                'bear_prob': d1.get('熊市概率', 0),
            }

        # 【v1.8】板块推演结果也写入
        sector_regime_analysis = {k: v for k, v in regime_analysis.items() if k != '市场整体'}
        market_regime_analysis = regime_analysis.get('市场整体', {})

        MARKET_ENV.update({
            'market_regime': regime,
            'bull_prob': bull_prob,
            'bear_prob': bear_prob,
            'neutral_prob': neutral_prob,
            'confidence': confidence,
            'sector_signals': sector_signals,
            'available': True,
            # 【v1.8】新增字段
            'regime_duration': market_regime_analysis.get('duration_days', 0),
            'regime_transition_to': market_regime_analysis.get('transition_to', '震荡'),
            'regime_transition_prob': market_regime_analysis.get('transition_prob', 0),
            'regime_narrative': market_regime_analysis.get('narrative', ''),
            'regime_risk_hint': market_regime_analysis.get('risk_hint', ''),
            'sector_regime_analysis': sector_regime_analysis,
        })

        # 7. 【v1.8】增强版控制台输出
        SUBTYPE_NAMES = ['高开高走', '低开高走', '震荡', '高开低走', '低开低走']
        emoji_map = {'高开高走': '🔥', '低开高走': '🔄', '震荡': '↔', '高开低走': '⚠', '低开低走': '💀'}
        regime_emoji = {'牛市': '📈', '震荡': '↔', '熊市': '📉'}
        strength_mark = {'strong': '█', 'moderate': '▓', 'weakening': '░'}

        print(
            f"\n  市场判定: {regime_emoji.get(regime, '')} {regime}（牛{bull_prob:.0%} 震{neutral_prob:.0%} 熊{bear_prob:.0%}）")
        print(f"  ────────")

        # 市场整体推演摘要
        market_regime = regime_analysis.get('市场整体')
        if market_regime:
            mr = market_regime
            sk = strength_mark.get(mr['state_strength'], '?')
            print(f"  ▸ 市场整体  {regime_emoji.get(mr['current_state'], '')}{mr['current_state']}"
                  f" {sk} │ 预计持续{mr['duration_days']}个交易日"
                  f" │ {mr['transition_point']}")
            if mr['risk_hint']:
                print(f"    ⚡ {mr['risk_hint']}")

        # 各板块推演
        print(f"  ────────")
        for sector_name in sorted(all_predictions.keys()):
            horizon_data = all_predictions[sector_name]
            d1 = horizon_data.get(1, {})
            bull = d1.get('牛市概率', 0)
            bear = d1.get('熊市概率', 0)
            if bull > bear + 0.1:
                trend = '📈牛'
            elif bear > bull + 0.1:
                trend = '📉熊'
            else:
                trend = '↔震'

            # 基础概率行
            line = f"  ▸ {sector_name:4s} 1日 {trend} (牛{bull:.0%} 震{d1.get('震荡概率', 0):.0%} 熊{bear:.0%})"

            # 走势细分
            subtype = horizon_data.get('走势细分', {})
            if subtype:
                best = max(subtype, key=subtype.get)
                line += f"  {emoji_map.get(best, '')}{best}"
            print(line)

            # 【v1.8】状态推演行
            ra = regime_analysis.get(sector_name)
            if ra:
                sk = strength_mark.get(ra['state_strength'], '?')
                dur = ra['duration_days']
                trans = ra['transition_to']
                tp = ra['transition_prob']
                print(f"    ↳ {sk}持续{dur}天 → 可能转{regime_emoji.get(trans, '')}{trans}({tp:.0%})")

        # 【v1.8】综合叙事
        print(f"\n  ── 状态推演 ──")
        if market_regime and market_regime['narrative']:
            print(f"  {market_regime['narrative']}")

        # 关键板块转折信号
        key_transitions = []
        for sn, ra in regime_analysis.items():
            if sn == '市场整体':
                continue
            if ra['transition_prob'] > 0.3 and ra['state_strength'] == 'weakening':
                key_transitions.append((sn, ra))
        if key_transitions:
            print(f"  ── 转折信号 ──")
            for sn, ra in sorted(key_transitions, key=lambda x: x[1]['transition_prob'], reverse=True)[:5]:
                print(f"  ⚡ {sn}: {ra['current_state']}→{ra['transition_to']}"
                      f" ({ra['transition_prob']:.0%}), {ra['risk_hint']}")

        # 8. 导出结果（静默）
        aggregated_importance = None
        if lgbm_predictor.feature_importance is not None:
            aggregated_importance = lgbm_predictor.feature_importance.groupby('feature')[
                'importance'].mean().reset_index()
            aggregated_importance = aggregated_importance.sort_values('importance', ascending=False).reset_index(
                drop=True)

        result_exporter.export_prediction_csv(all_predictions)
        result_exporter.export_feature_importance(aggregated_importance)
        result_exporter.export_sector_pdf(all_predictions)
        result_exporter.export_market_pdf(all_predictions)

        print(f"\n  ✓ 牛熊预测完成，结果已保存")

    except Exception as e:
        print(f"  ⚠ 牛熊预测异常: {str(e)[:80]}")
        import traceback
        _bb_print(traceback.format_exc())


print("━" * 70)


# ==========================================
# 模块2：数据源模块 (Data Fetcher Module)
# 设计意图：双模式数据获取，baostock不可用时自动切换模拟数据
# 模块规模：约180行
# ==========================================

def convert_stock_code(code, to_baostock=True):
    """
    股票代码格式转换函数
    设计意图：统一处理两种代码格式的互转
    输入：code - 股票代码
          to_baostock - True: 转为baostock格式(sh.601857)
                       False: 转为标准格式(601857.SH)
    输出：转换后的代码
    """
    if to_baostock:
        if '.' in code and code.split('.')[0] in ['sh', 'sz']:
            return code
        if code.endswith('.SH'):
            return 'sh.' + code[:-3]
        if code.endswith('.SZ'):
            return 'sz.' + code[:-3]
        return code
    else:
        if code.startswith('sh.'):
            return code[3:] + '.SH'
        if code.startswith('sz.'):
            return code[3:] + '.SZ'
        return code


def exponential_backoff_retry(func, max_retries=3, max_wait=3):
    """
    指数退避重试机制
    设计意图：网络请求失败时自动重试，提高鲁棒性
    输入：func - 要执行的函数
          max_retries - 最大重试次数
          max_wait - 最长等待时间(秒)
    输出：函数执行结果
    """
    for attempt in range(max_retries):
        try:
            result = func()
            return result, True
        except Exception as e:
            wait_time = min(2 ** attempt, max_wait)
            time.sleep(wait_time)
    return None, False


def generate_mock_stock_data(code, name, start_date, end_date, days=300):
    """
    模拟数据生成器
    设计意图：baostock不可用时提供高质量模拟数据
    输入：code - 股票代码
          name - 股票名称
          start_date/end_date - 日期范围
          days - 生成数据天数
    输出：模拟股票数据DataFrame
    """

    # 生成日期序列
    dates = pd.date_range(end=end_date, periods=days, freq='B')

    # 基于行业特性设置不同的波动率和漂移
    industry_vol = {
        '能源': 0.020, '金属': 0.025, '金融': 0.018, '消费': 0.022, '科技': 0.028,
        '医药': 0.024, '制造': 0.021, '地产': 0.023, '交通': 0.019, '化工': 0.022,
        '半导体': 0.032, '人工智能': 0.030  # 【】高波动科技行业
    }

    # 找到所属行业
    industry = '科技'  # 默认
    base_price = np.random.uniform(10, 100)

    for ind, stocks in STOCK_POOL.items():
        for s in stocks:
            if s['code'] == code or s['code_std'] == code:
                industry = ind
                break

    # 几何布朗运动生成价格
    mu = np.random.uniform(-0.0005, 0.0015)  # 日收益率均值
    sigma = industry_vol.get(industry, 0.022)  # 日波动率

    returns = np.random.normal(mu, sigma, days)
    prices = base_price * np.cumprod(1 + returns)

    # 生成OHLCV数据
    high = prices * (1 + np.random.uniform(0, 0.03, days))
    low = prices * (1 - np.random.uniform(0, 0.03, days))
    open_p = low + (high - low) * np.random.uniform(0.3, 0.7, days)
    close = prices
    volume = np.random.randint(1000000, 100000000, days)

    df = pd.DataFrame({
        'date': dates.strftime('%Y-%m-%d'),
        'code': code,
        'code_std': convert_stock_code(code, to_baostock=False),
        'name': name,
        'industry': industry,
        'open': open_p.round(2),
        'high': high.round(2),
        'low': low.round(2),
        'close': close.round(2),
        'volume': volume,
        'amount': volume * close * 100
    })

    return df


def fetch_single_stock_data(stock_info, start_date, end_date):
    """
    获取单只股票数据
    设计意图：统一数据获取接口，自动处理真实/模拟模式
    输入：stock_info - 股票信息字典
          start_date/end_date - 日期范围
    输出：股票数据DataFrame
    """
    code = stock_info['code']
    name = stock_info['name']
    code_std = stock_info['code_std']

    # 模式1: 使用baostock去掉内部login/logout，由外部批量管理
    if DEPENDENCY_STATUS['baostock']:
        def fetch_bs():
            # 假设已经在外部登录，不在此处login/logout
            rs = bs.query_history_k_data_plus(
                code,
                "date,code,open,high,low,close,volume,amount",
                start_date=start_date,
                end_date=end_date,
                frequency="d",
                adjustflag="3"
            )
            data_list = []
            while (rs.error_code == '0') & rs.next():
                data_list.append(rs.get_row_data())

            df = pd.DataFrame(data_list, columns=rs.fields)
            for col in ['open', 'high', 'low', 'close', 'volume', 'amount']:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            return df

        result, success = exponential_backoff_retry(fetch_bs)
        if success and result is not None and len(result) > 0:
            result['name'] = name
            result['code_std'] = code_std
            return result

        # 【P0-4修复】baostock失败时降级到mock数据，不再返回None
        _bb_print(f'[数据] ⚠ {name}({code}) 真实数据获取失败，使用模拟数据')
        mock_df = generate_mock_stock_data(code_std, name, start_date, end_date)
        return mock_df



# 【P0-2修复】baostock连接上下文管理器
from contextlib import contextmanager

@contextmanager
def baostock_session():
    """baostock连接上下文管理器，确保login/logout配对"""
    if DEPENDENCY_STATUS['baostock']:
        try:
            bs.login()
            yield
        finally:
            try:
                bs.logout()
            except:
                pass
    else:
        yield

def baostock_batch_login():
    """
    baostock批量登录
    设计意图：一次性登录，减少200+次重复握手开销
    输出：登录结果码
    """
    if not DEPENDENCY_STATUS['baostock']:
        return None
    try:
        result = bs.login()
        return result
    except Exception as e:
        print(f"[数据] ✗ baostock批量登录失败: {e}")
        return None


def baostock_batch_logout():
    """
    baostock批量登出
    设计意图：数据获取完成后统一登出
    输出：登出结果码
    """
    if not DEPENDENCY_STATUS['baostock']:
        return None
    try:
        result = bs.logout()
        return result
    except Exception as e:
        return None


def fetch_realtime_price(code, name):
    """
    获取当天实时价格
    设计意图：mootdx优先获取盘中实时价格，baostock降级兜底
    输入：code - 股票代码（如sh.601857）
          name - 股票名称
    输出：实时价格（float）或None
    """
    # 【v1.8】优先使用mootdx获取实时价格（毫秒级，无需baostock login）
    if MOOTDX_CLIENT._connected:
        price = MOOTDX_CLIENT.get_realtime_price(code)
        if price is not None and price > 0:
            return price

    # 降级：baostock分钟级数据获取（保留原逻辑作为兜底）
    if not DEPENDENCY_STATUS['baostock']:
        return None

    try:
        # 【v1.7修复】确认baostock连接存活，必要时重连
        try:
            # 用一次轻量查询测试连接是否存活
            test_rs = bs.query_history_k_data_plus(
                "sh.000001", "date",
                start_date=(datetime.datetime.now() - timedelta(days=3)).strftime('%Y-%m-%d'),
                end_date=datetime.datetime.now().strftime('%Y-%m-%d'),
                frequency="d", adjustflag="3"
            )
            if test_rs.error_code != '0':
                # 连接已断，重连
                try:
                    bs.logout()
                except:
                    pass
                bs.login()
                time.sleep(0.3)
        except Exception:
            # socket异常，重连
            try:
                bs.logout()
            except:
                pass
            bs.login()
            time.sleep(0.3)

        # 获取5天前的日期（分钟级数据只保留最近5个交易日）
        start_date = (datetime.datetime.now() - timedelta(days=5)).strftime('%Y-%m-%d')

        def fetch_minute():
            rs = bs.query_history_k_data_plus(
                code,
                "date,time,close,volume,amount",
                start_date=start_date,
                end_date=datetime.datetime.now().strftime('%Y-%m-%d'),
                frequency="5",  # 5分钟K线
                adjustflag="3"
            )
            if rs.error_code != '0':
                return pd.DataFrame()  # v1.7：查询失败返回空DataFrame
            data_list = []
            while rs.next():
                data_list.append(rs.get_row_data())
            return pd.DataFrame(data_list, columns=rs.fields) if data_list else pd.DataFrame()

        df_minute, success = exponential_backoff_retry(fetch_minute)

        if success and df_minute is not None and len(df_minute) > 0:
            # 取最新一条记录的close作为实时价格
            df_minute['close'] = pd.to_numeric(df_minute['close'], errors='coerce')
            df_minute = df_minute.dropna(subset=['close'])
            if len(df_minute) > 0:
                realtime_price = df_minute['close'].iloc[-1]
                return float(realtime_price)

        # 降级：使用日K线最新close

        def fetch_daily():
            rs = bs.query_history_k_data_plus(
                code,
                "date,close",
                start_date=(datetime.datetime.now() - timedelta(days=10)).strftime('%Y-%m-%d'),
                end_date=datetime.datetime.now().strftime('%Y-%m-%d'),
                frequency="d",
                adjustflag="3"
            )
            if rs.error_code != '0':
                return pd.DataFrame()  # v1.7：查询失败返回空DataFrame
            data_list = []
            while rs.next():
                data_list.append(rs.get_row_data())
            return pd.DataFrame(data_list, columns=rs.fields) if data_list else pd.DataFrame()

        df_daily, success = exponential_backoff_retry(fetch_daily)
        if success and df_daily is not None and len(df_daily) > 0:
            df_daily['close'] = pd.to_numeric(df_daily['close'], errors='coerce')
            df_daily = df_daily.dropna(subset=['close'])
            if len(df_daily) > 0:
                daily_price = df_daily['close'].iloc[-1]
                return float(daily_price)

        return None

    except Exception as e:
        # 【v1.7修复】socket异常时尝试重连，避免后续查询全部失败
        if 'socket' in str(e).lower() or '10038' in str(e) or 'WinError' in str(e):
            try:
                bs.logout()
            except:
                pass
            try:
                bs.login()
                time.sleep(0.3)
            except:
                pass
        return None



def preprocess_data(df):
    """
    数据预处理
    设计意图：异常值检测 + 缺失值处理
    输入：df - 原始数据
    输出：预处理后的数据
    """
    if df is None or len(df) == 0:
        return df

    # 复制数据避免修改原数据
    df = df.copy()

    # 确保数值列都是float类型
    numeric_cols = ['open', 'high', 'low', 'close', 'volume', 'amount']
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    # 1. 异常值检测（3倍标准差）
    for col in numeric_cols:
        if col in df.columns:
            mean = df[col].mean()
            std = df[col].std()
            if std > 0:
                outliers = (df[col] - mean).abs() > 3 * std
                df.loc[outliers, col] = np.nan
                if outliers.sum() > 0:
                    pass

    # 2. 缺失值处理
    df = df.ffill().bfill()
    df = df.fillna(df.mean(numeric_only=True))

    return df


def validate_data_quality(df, min_days=MIN_DATA_DAYS):
    """
    数据质量校验
    设计意图：确保数据质量符合回测要求
    输入：df - 数据
          min_days - 最小天数要求
    输出：是否通过校验
    """
    if df is None or len(df) < min_days:
        print(f"[数据] ✗ 数据不足{min_days}天，不纳入回测")
        return False

    # 检查关键列是否存在
    required_cols = ['date', 'open', 'high', 'low', 'close', 'volume']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        print(f"[数据] ✗ 缺少关键列: {missing_cols}")
        return False

    return True

    print("━" * 70)


# ==========================================
# 模块3：数据聚合模块 (Data Aggregator Module)
# 设计意图：实现三维联动数据聚合 - 个股+行业+指数
# 模块规模：约150行
# ==========================================

def aggregate_industry_data(all_stocks_data):
    """
    行业数据聚合
    设计意图：按行业计算成分股的聚合特征
    输入：all_stocks_data - 所有股票数据字典
    输出：各行业聚合数据DataFrame字典
    """

    industry_agg_data = {}

    for industry, stocks in STOCK_POOL.items():

        # 收集该行业所有股票的日数据
        industry_dfs = []
        for stock_info in stocks:
            code = stock_info['code']
            if code in all_stocks_data:
                df = all_stocks_data[code].copy()
                df['return'] = df['close'].pct_change()
                industry_dfs.append(df)

        if not industry_dfs:
            continue

        # 按日期对齐并聚合
        combined = pd.concat(industry_dfs)

        # 按日期计算行业聚合指标
        agg_by_date = combined.groupby('date').agg({
            'close': ['mean', 'std', 'min', 'max'],
            'return': ['mean', 'std', lambda x: (x > 0).sum() / len(x)],
            'volume': 'sum',
            'amount': 'sum'
        }).round(4)

        # 重命名列
        agg_by_date.columns = [
            'industry_close_mean', 'industry_close_std', 'industry_close_min', 'industry_close_max',
            'industry_return_mean', 'industry_return_std', 'industry_win_rate',
            'industry_volume_sum', 'industry_amount_sum'
        ]

        agg_by_date['industry'] = industry
        industry_agg_data[industry] = agg_by_date.reset_index()

    return industry_agg_data


def process_index_data(index_data_dict):
    """
    指数数据处理
    设计意图：统一处理三大指数数据，计算指数特征
    输入：index_data_dict - 指数数据字典
    输出：处理后的指数数据字典
    """

    processed_index = {}

    for code, df in index_data_dict.items():
        if df is None or len(df) == 0:
            continue

        df = df.copy()

        # 计算指数收益率和技术指标
        df['index_return'] = df['close'].pct_change()
        df['index_ma5'] = df['close'].rolling(5).mean()
        df['index_ma10'] = df['close'].rolling(10).mean()
        df['index_ma20'] = df['close'].rolling(20).mean()
        df['index_volatility'] = df['index_return'].rolling(20).std()

        # 计算相对强弱
        # 【P1-4修复】标准RSI计算
        delta = df['index_return'].diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        avg_gain = gain.rolling(window=14, min_periods=14).mean()
        avg_loss = loss.rolling(window=14, min_periods=14).mean()
        rs = avg_gain / (avg_loss + 1e-10)
        df['index_rsi'] = 100 - (100 / (1 + rs))

        processed_index[code] = df

    return processed_index


def merge_3d_data(stock_data, industry_data, index_data_dict, stock_info):
    """
    三维数据合并
    设计意图：将个股数据、行业聚合数据、指数数据合并为一张表
    输入：stock_data - 个股数据
          industry_data - 行业聚合数据
          index_data_dict - 指数数据字典
          stock_info - 股票信息
    输出：合并后的三维数据
    """
    industry = None
    for ind, stocks in STOCK_POOL.items():
        for s in stocks:
            if s['code'] == stock_info['code']:
                industry = ind
                break

    # 1. 个股数据预处理
    result = stock_data.copy()
    result['date'] = pd.to_datetime(result['date'])

    # 2. 合并行业数据
    if industry in industry_data:
        ind_data = industry_data[industry].copy()
        ind_data['date'] = pd.to_datetime(ind_data['date'])
        result = pd.merge(result, ind_data, on='date', how='left', suffixes=('', '_industry'))

    # 3. 合并三大指数数据
    for idx_code, idx_df in index_data_dict.items():
        if idx_df is not None and len(idx_df) > 0:
            idx_name = idx_df['name'].iloc[0] if 'name' in idx_df.columns else idx_code
            idx_merge = idx_df[['date', 'close', 'index_return', 'index_volatility']].copy()
            idx_merge.columns = ['date', f'{idx_name}_close', f'{idx_name}_return', f'{idx_name}_volatility']
            idx_merge['date'] = pd.to_datetime(idx_merge['date'])
            result = pd.merge(result, idx_merge, on='date', how='left')

    # 填充合并后的缺失值
    result = result.ffill().bfill()

    return result

    print("━" * 70)


# ==========================================
# 模块4：特征工程模块 (Feature Engineer Module)
# 设计意图：计算所有技术指标和模型特征
# 模块规模：约180行
# ==========================================


# ==========================================
# 模块7.5：多因子信号评分系统 (Signal Scoring Module)
# v1.7新增：从"预测收益率"转向"识别高概率交易信号"
# 设计理念：趋势确认 > 动量强度 > 入场质量 > ML辅助
# ==========================================

def calculate_trend_score(df):
    """
    趋势因子评分 (0-100)
    核心逻辑：只在上升趋势中做多
    """
    score = 50  # 基准分
    close = df['close'].astype(float)

    # 1. 均线多头排列 (最核心)
    if 'ma5' in df.columns and 'ma10' in df.columns and 'ma20' in df.columns and 'ma60' in df.columns:
        ma5 = df['ma5'].iloc[-1]
        ma10 = df['ma10'].iloc[-1]
        ma20 = df['ma20'].iloc[-1]
        ma60 = df['ma60'].iloc[-1]
        last_close = close.iloc[-1]

        # 价格在MA20和MA60之上
        if last_close > ma20:
            score += 10
        else:
            score -= 15  # 价格在MA20之下，大扣分

        if last_close > ma60:
            score += 8
        else:
            score -= 12

        # 均线排列：MA5 > MA10 > MA20 > MA60
        if ma5 > ma10 > ma20 > ma60:
            score += 15  # 完美多头排列
        elif ma5 > ma10 > ma20:
            score += 8  # 短中期多头
        elif ma5 < ma10 < ma20:
            score -= 15  # 空头排列

        # MA20斜率（趋势方向）
        if len(df['ma20'].dropna()) >= 5:
            ma20_slope = (df['ma20'].iloc[-1] - df['ma20'].iloc[-5]) / df['ma20'].iloc[-5]
            if ma20_slope > 0.01:  # 周涨幅>1%
                score += 5
            elif ma20_slope > 0.005:
                score += 3
            elif ma20_slope < -0.01:
                score -= 8

    # 2. MACD方向
    if 'macd' in df.columns and 'macd_signal' in df.columns and 'macd_hist' in df.columns:
        macd_hist = df['macd_hist'].iloc[-1]
        macd_hist_prev = df['macd_hist'].iloc[-2] if len(df) > 1 else 0

        if macd_hist > 0 and macd_hist > macd_hist_prev:
            score += 8  # MACD金叉且柱线扩大
        elif macd_hist > 0:
            score += 3  # MACD金叉但柱线缩小
        elif macd_hist < 0 and macd_hist < macd_hist_prev:
            score -= 10  # MACD死叉且柱线扩大
        elif macd_hist < 0:
            score -= 5

    return max(0, min(100, score))


def calculate_momentum_score(df):
    """
    动量因子评分 (0-100)
    核心逻辑：选有上涨动力的票，避开过度超买
    """
    score = 50
    close = df['close'].astype(float)

    # 1. RSI位置（30-65是甜区）
    if 'rsi' in df.columns:
        rsi = df['rsi'].iloc[-1]
        if 40 <= rsi <= 60:
            score += 12  # 中性偏强，最佳动量区间
        elif 30 <= rsi < 40:
            score += 8  # 超卖区，反弹潜力
        elif 60 < rsi <= 70:
            score += 3  # 偏强但注意风险
        elif rsi > 70:
            score -= 15  # 超买，追高风险大
        elif rsi < 30:
            score -= 5  # 极度超卖，可能继续跌

    # 2. 短期动量
    if 'return_5d' in df.columns:
        ret5 = df['return_5d'].iloc[-1]
        if 0.02 <= ret5 <= 0.08:
            score += 10  # 温和上涨
        elif 0 < ret5 < 0.02:
            score += 5  # 微涨
        elif ret5 > 0.08:
            score -= 5  # 涨太猛，回调风险
        elif -0.03 < ret5 < 0:
            score -= 3  # 微跌
        else:
            score -= 10  # 大跌

    # 3. 相对强度（vs指数）
    if 'excess_return_vs_industry' in df.columns:
        excess = df['excess_return_vs_industry'].iloc[-1]
        if excess > 0.01:
            score += 8
        elif excess > 0:
            score += 3
        elif excess < -0.02:
            score -= 8

    return max(0, min(100, score))


def calculate_entry_quality_score(df):
    """
    入场质量因子评分 (0-100)
    核心逻辑：好的入场点 = 上升趋势中的回调买入
    """
    score = 50
    close = df['close'].astype(float)

    # 1. 布林带位置
    if 'bb_position' in df.columns and 'ma20' in df.columns:
        bb_pos = df['bb_position'].iloc[-1]
        last_close = close.iloc[-1]
        ma20 = df['ma20'].iloc[-1]

        if last_close >= ma20:  # 在上升趋势中
            if 0.2 <= bb_pos <= 0.5:
                score += 15  # 上升趋势中回调到中轨附近，最佳入场
            elif 0.5 < bb_pos <= 0.75:
                score += 5  # 正常位置
            elif bb_pos > 0.85:
                score -= 10  # 上轨附近，追高风险
            elif bb_pos < 0.2:
                score += 5  # 接近下轨，可能有支撑
        else:  # 在MA20之下
            if bb_pos < 0.15:
                score += 3  # 极度超卖，可能反弹
            else:
                score -= 10  # 下降途中，不入场

    # 2. 近期回调幅度（在上升趋势中，回调是好的入场点）
    if 'return_3d' in df.columns and 'ma20' in df.columns:
        ret3 = df['return_3d'].iloc[-1]
        last_close = close.iloc[-1]
        ma20 = df['ma20'].iloc[-1]

        if last_close > ma20:  # 仍在上升趋势
            if -0.05 <= ret3 < -0.01:
                score += 12  # 上升趋势中回调2-5%，好的入场点
            elif -0.01 <= ret3 < 0.02:
                score += 5  # 温和变动
            elif ret3 > 0.05:
                score -= 5  # 连涨太多

    # 3. 量价配合
    if 'volume_ratio' in df.columns:
        vol_ratio = df['volume_ratio'].iloc[-1]
        if 'return_1d' in df.columns:
            ret1 = df['return_1d'].iloc[-1]
            # 上涨放量，回调缩量 = 健康趋势
            if ret1 > 0 and vol_ratio > 1.3:
                score += 8  # 放量上涨
            elif ret1 < 0 and vol_ratio < 0.8:
                score += 5  # 缩量回调，卖压不重
            elif ret1 < 0 and vol_ratio > 1.5:
                score -= 8  # 放量下跌，不好

    return max(0, min(100, score))


def calculate_composite_signal(df, ml_fused_result, market_regime='中性'):
    """
    多因子综合信号评分
    v1.7核心：替代纯预测收益率，转向信号质量评估

    权重分配：
    - 趋势因子 35%：最核心，不在下跌趋势中做多
    - 动量因子 25%：选择有上涨动力的票
    - 入场质量 20%：好的入场点降低风险
    - ML预测 20%：模型辅助，但权重降低

    返回：
    - composite_score: 综合评分 0-100
    - signal_level: 信号等级 (强多/偏多/中性/回避)
    - entry_type: 入场类型 (突破/回调/观望)
    - details: 各因子分项评分
    """
    trend_score = calculate_trend_score(df)
    momentum_score = calculate_momentum_score(df)
    entry_score = calculate_entry_quality_score(df)

    # ML预测转为0-100分
    ml_score = 50  # 默认中性
    if ml_fused_result and 3 in ml_fused_result:
        fused = ml_fused_result[3]
        ml_pred = fused.get('fused_mean', 0)
        ml_var = fused.get('fused_var', 0.01)
        # 将预测收益率映射为分数：0%→50, +3%→80, -3%→20
        # 方差越大，ML分数越趋向50（不确定时降权）
        confidence = 1.0 / (1.0 + ml_var * 100)  # 方差越大置信度越低
        ml_score = 50 + ml_pred * 1000 * confidence  # pred*1000将0.03映射到30
        ml_score = max(0, min(100, ml_score))

    # 加权综合评分
    composite_score = (
            trend_score * 0.35 +
            momentum_score * 0.25 +
            entry_score * 0.20 +
            ml_score * 0.20
    )

    # 牛熊环境调整
    if market_regime == '熊市':
        composite_score *= 0.7  # 熊市整体降分
    elif market_regime == '牛市':
        composite_score = min(100, composite_score * 1.1)  # 牛市小幅加分

    composite_score = max(0, min(100, composite_score))

    # 信号等级判定
    if composite_score >= 72:
        signal_level = '强多'
    elif composite_score >= 60:
        signal_level = '偏多'
    elif composite_score >= 48:
        signal_level = '中性'
    else:
        signal_level = '回避'

    # 入场类型判定
    rsi = df['rsi'].iloc[-1] if 'rsi' in df.columns else 50
    bb_pos = df['bb_position'].iloc[-1] if 'bb_position' in df.columns else 0.5
    ret3 = df['return_3d'].iloc[-1] if 'return_3d' in df.columns else 0

    if signal_level in ['强多', '偏多']:
        if ret3 < -0.01 and rsi < 55:
            entry_type = '回调买入'
        elif bb_pos > 0.7 and df['volume_ratio'].iloc[-1] > 1.2:
            entry_type = '突破追涨'
        else:
            entry_type = '顺势持有'
    else:
        entry_type = '观望回避'

    details = {
        'trend': round(trend_score, 1),
        'momentum': round(momentum_score, 1),
        'entry_quality': round(entry_score, 1),
        'ml_signal': round(ml_score, 1),
        'composite': round(composite_score, 1)
    }

    return composite_score, signal_level, entry_type, details


def calculate_technical_indicators(df):
    """
    计算技术指标
    设计意图：计算常用技术指标作为模型输入特征
    输入：df - 包含OHLCV的股票数据
    输出：添加技术指标后的DataFrame
    """

    df = df.copy()
    close = df['close']
    high = df['high']
    low = df['low']
    volume = df['volume']

    # 1. 移动平均线 MA
    for period in [5, 10, 20, 60]:
        df[f'ma{period}'] = close.rolling(period).mean()
        df[f'ma{period}_ratio'] = close / df[f'ma{period}'] - 1

    # 2. MACD
    ema12 = close.ewm(span=12).mean()
    ema26 = close.ewm(span=26).mean()
    df['macd'] = ema12 - ema26
    df['macd_signal'] = df['macd'].ewm(span=9).mean()
    df['macd_hist'] = df['macd'] - df['macd_signal']

    # 3. RSI 相对强弱指标【v1.5修复】改用Wilder/EMA平滑+除零保护
    delta = close.diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.ewm(alpha=1 / 14, min_periods=14, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / 14, min_periods=14, adjust=False).mean()
    rs = avg_gain / (avg_loss + 1e-10)
    df['rsi'] = 100 - (100 / (1 + rs))

    # 4. Bollinger Bands 布林带
    bb_mid = close.rolling(20).mean()
    bb_std = close.rolling(20).std()
    df['bb_upper'] = bb_mid + 2 * bb_std
    df['bb_lower'] = bb_mid - 2 * bb_std
    df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / bb_mid
    df['bb_position'] = (close - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])

    # 5. KDJ 随机指标
    low_min = low.rolling(9).min()
    high_max = high.rolling(9).max()
    rsv = (close - low_min) / (high_max - low_min) * 100
    df['kdj_k'] = rsv.rolling(3).mean()
    df['kdj_d'] = df['kdj_k'].rolling(3).mean()
    df['kdj_j'] = 3 * df['kdj_k'] - 2 * df['kdj_d']

    # 6. Volume 成交量指标
    df['volume_ma5'] = volume.rolling(5).mean()
    df['volume_ma10'] = volume.rolling(10).mean()
    df['volume_ratio'] = volume / df['volume_ma5']
    df['obv'] = (np.sign(close.diff()) * volume).cumsum()

    # 7. ATR 真实波幅
    tr1 = high - low
    tr2 = abs(high - close.shift())
    tr3 = abs(low - close.shift())
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    df['atr'] = tr.rolling(14).mean()
    df['atr_ratio'] = df['atr'] / close

    return df


def calculate_return_features(df):
    """
    计算收益率和波动率特征
    设计意图：生成收益率相关的统计特征
    输入：df - 股票数据
    输出：添加收益率特征的DataFrame
    v1.1修复：收益率统一用小数形式，不再*100
    """

    df = df.copy()
    close = df['close'].astype(float)

    # 【v1.1修复】不同周期收益率 - 保持小数形式（0.025表示2.5%）
    for period in [1, 3, 5, 10, 20]:
        df[f'return_{period}d'] = close.pct_change(period)  # 去掉*100

    # 滚动波动率（基于小数收益率）
    for window in [5, 10, 20, 60]:
        df[f'volatility_{window}d'] = df['return_1d'].rolling(window).std() * np.sqrt(252)

    # 收益率偏度和峰度
    df['return_skew_20d'] = df['return_1d'].rolling(20).skew()
    df['return_kurt_20d'] = df['return_1d'].rolling(20).kurt()

    # 【v1.1修复】夏普比率（滚动）- return_1d已是小数，不需要/100
    risk_free = 0.03 / 252  # 日化无风险收益率
    rolling_mean = df['return_1d'].rolling(20).mean()  # 不再/100
    rolling_std = df['return_1d'].rolling(20).std()  # 不再/100
    df['sharpe_20d'] = ((rolling_mean - risk_free) / (rolling_std + 1e-10)) * np.sqrt(252)

    # 最大回撤（滚动）
    def rolling_drawdown(x):
        if len(x) == 0:
            return 0
        peak = np.maximum.accumulate(x)
        dd = (x[-1] - peak[-1]) / peak[-1] if peak[-1] > 0 else 0
        return dd

    df['drawdown_20d'] = close.rolling(20).apply(rolling_drawdown, raw=True)

    return df


def calculate_relative_strength(df):
    """
    计算相对强弱特征
    设计意图：计算个股相对于行业和指数的强弱
    输入：df - 三维合并数据
    输出：添加相对强弱特征的DataFrame
    """

    df = df.copy()

    # 相对于行业的超额收益
    if 'industry_return_mean' in df.columns:
        df['excess_return_vs_industry'] = df['return_1d'] - df['industry_return_mean']

    # 相对于指数的超额收益
    for col in df.columns:
        if 'return' in col and '指数' in col:
            df[f'excess_return_vs_{col.split("_")[0]}'] = df['return_1d'] - df[col]

    # 行业内排名分位数
    if 'industry_return_mean' in df.columns:
        df['return_industry_rank'] = df['return_1d'].rolling(20).apply(
            lambda x: stats.percentileofscore(x, x.iloc[-1]) / 100
        )

    return df


def engineer_all_features(df):
    """
    完整特征工程入口
    设计意图：统一调用所有特征计算函数
    输入：df - 原始三维数据
    输出：包含所有特征的数据
    """

    # 计算所有特征
    df = calculate_technical_indicators(df)
    df = calculate_return_features(df)
    df = calculate_relative_strength(df)

    # 处理特征中的缺失值
    df = df.ffill().bfill()
    df = df.fillna(0)

    # 移除inf值
    df = df.replace([np.inf, -np.inf], 0)

    return df

    print("━" * 70)


# ==========================================
# 模块5：GARCH波动率模型模块 (GARCH Model Module)
# 设计意图：纯numpy实现GARCH(1,1)波动率预测，增强鲁棒性和完整性
# 模块规模：约180行
# ==========================================

def garch_log_likelihood(params, returns):
    """
    GARCH(1,1)对数似然函数（优化版）
    设计意图：用于极大似然估计参数，增强数值稳定性
    输入：params - [omega, alpha, beta]
          returns - 收益率序列（已标准化）
    输出：负对数似然值（用于最小化）
    """
    omega, alpha, beta = params

    # 严格参数约束（确保均值回复：alpha + beta < 1，且所有参数非负）
    if omega <= 1e-12 or alpha < 0 or beta < 0 or (alpha + beta) >= 0.999:
        return 1e15  # 惩罚项放大，避免无效参数

    T = len(returns)
    sigma2 = np.zeros(T)
    sigma2[0] = np.var(returns)  # 初始方差（用样本方差）

    # 向量化计算条件方差（替代循环，提升效率）
    for t in range(1, T):
        sigma2[t] = omega + alpha * (returns[t - 1] ** 2) + beta * sigma2[t - 1]

    # 数值稳定性保护（避免0或极小值）
    sigma2 = np.maximum(sigma2, 1e-10)

    # 对数似然计算（添加小常数避免log(0)）
    log_likelihood = -0.5 * np.sum(
        np.log(2 * np.pi * sigma2) + (returns ** 2) / sigma2
    )

    return -log_likelihood  # 返回负值用于最小化


def fit_garch_1_1(returns, max_iter=2000, tol=1e-6):
    """
    拟合GARCH(1,1)模型（优化版）
    设计意图：增强鲁棒性，添加数据清洗、参数校验、拟合优度评估
    输入：returns - 收益率序列（小数形式）
          max_iter - 最大迭代次数
          tol - 收敛容忍度
    输出：参数字典（含拟合优度、波动率等）
    v1.1修复：输入已经是小数形式，不再除以100
    """

    # ========== 第一步：数据预处理 ==========
    # 【v1.1修复】收益率已经是小数形式，直接使用
    returns = np.array(returns).flatten()
    returns = returns[~np.isnan(returns)]  # 剔除NaN
    returns = returns[np.abs(stats.zscore(returns)) < 3]  # 剔除3σ外异常值

    # 数据量校验
    if len(returns) < 60:
        print("[GARCH] 数据不足60观测值，用滚动波动率替代")
        rolling_vol = np.std(returns) * np.sqrt(252)
        return {
            'omega': 0.0, 'alpha': 0.1, 'beta': 0.85,
            'persistence': 0.95, 'volatility': rolling_vol,
            'conditional_volatility': np.full(len(returns), rolling_vol / np.sqrt(252)),
            'converged': False, 'aic': np.inf, 'bic': np.inf
        }

    # ========== 第二步：参数拟合 ==========
    # 初始参数优化（基于样本矩估计）
    init_omega = np.var(returns) * (1 - 0.9)  # 初始长期方差
    init_params = [init_omega, 0.1, 0.8]  # 更合理的初始值

    # 参数边界（更严格，确保数值稳定性）
    bounds = [
        (1e-10, 0.001),  # omega：非负且极小（避免过度拟合）
        (0.001, 0.4),  # alpha：0.001~0.4（符合金融数据特征）
        (0.5, 0.998)  # beta：0.5~0.998（保证均值回复）
    ]

    try:
        # 优化器（L-BFGS-B，添加收敛容忍度）
        result = minimize(
            garch_log_likelihood,
            init_params,
            args=(returns,),
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': max_iter, 'gtol': tol, 'disp': False}
        )

        # ========== 第三步：结果校验与后处理 ==========
        if not result.success:
            raise ValueError(f"优化失败：{result.message}")

        omega, alpha, beta = result.x
        persistence = alpha + beta

        # 计算条件方差
        T = len(returns)
        sigma2 = np.zeros(T)
        sigma2[0] = np.var(returns)
        for t in range(1, T):
            sigma2[t] = omega + alpha * (returns[t - 1] ** 2) + beta * sigma2[t - 1]
        conditional_vol = np.sqrt(sigma2)  # 日化条件波动率

        # 年化波动率（最终预测值）
        long_run_variance = omega / (1 - persistence) if (1 - persistence) > 1e-6 else np.var(returns)
        annual_vol = np.sqrt(long_run_variance * 252)

        # 模型评估（AIC/BIC）
        n_params = 3  # omega, alpha, beta
        log_likelihood = -result.fun
        aic = 2 * n_params - 2 * log_likelihood
        bic = n_params * np.log(T) - 2 * log_likelihood

        return {
            'omega': omega,
            'alpha': alpha,
            'beta': beta,
            'persistence': persistence,
            'volatility': annual_vol,
            'conditional_volatility': conditional_vol,
            'converged': result.success,
            'aic': aic,
            'bic': bic,
            'log_likelihood': log_likelihood
        }

    except Exception as e:
        # 降级方案：20日滚动波动率（年化）
        rolling_vol = returns[-20:].std() * np.sqrt(252) if len(returns) >= 20 else returns.std() * np.sqrt(252)
        return {
            'omega': 0.0, 'alpha': 0.1, 'beta': 0.85,
            'persistence': 0.95, 'volatility': rolling_vol,
            'conditional_volatility': np.full(len(returns), rolling_vol / np.sqrt(252)),
            'converged': False, 'aic': np.inf, 'bic': np.inf
        }


def predict_volatility_garch(returns, horizons=[1, 2, 3, 4, 5]):
    """
    使用GARCH预测多日波动率（完整实现版）
    设计意图：补全原截断代码，实现考虑均值回复的多期波动率预测
    输入：returns - 收益率序列（小数形式）
          horizons - 预测天数列表
    输出：各预测日的波动率字典（年化）
    """

    # 拟合GARCH模型
    garch_result = fit_garch_1_1(returns)

    # 提取核心参数
    omega = garch_result['omega']
    alpha = garch_result['alpha']
    beta = garch_result['beta']
    persistence = garch_result['persistence']
    long_run_var = omega / (1 - persistence) if (1 - persistence) > 1e-6 else np.var(returns)
    current_vol_var = garch_result['conditional_volatility'][-1] ** 2  # 最新条件方差（日化）

    # 多期波动率预测（考虑均值回复的GARCH期限结构）
    volatility_predictions = {}
    for h in horizons:
        # GARCH多期方差公式：Var(t+h) = 长期方差 + (persistence^h) * (当前方差 - 长期方差)
        future_var_h = long_run_var + (persistence ** h) * (current_vol_var - long_run_var)
        future_var_h = np.maximum(future_var_h, 1e-10)  # 数值保护
        future_vol_annual = np.sqrt(future_var_h * 252)  # 年化
        volatility_predictions[f'h_{h}d'] = future_vol_annual

    # 补充返回完整信息
    garch_result['volatility_predictions'] = volatility_predictions
    return garch_result

    # 模块加载验证
    print("━" * 70)


# ==========================================
# 模块6：卡尔曼滤波模块 (Kalman Filter Module)
# v1.1修复：收益率预测clip范围改为小数形式(-0.3, 0.3)
# ==========================================

class AdaptiveKalmanFilter:
    def __init__(self, dim_state=2, dim_obs=1):
        self.dim_state = dim_state
        self.dim_obs = dim_obs
        self.F = np.array([[1, 1], [0, 1]])
        self.H = np.array([[1, 0]])
        self.x = np.zeros((dim_state, 1))
        self.P = np.eye(dim_state) * 1000
        self.Q = np.eye(dim_state) * 0.01
        self.R = np.eye(dim_obs) * 0.1

    def predict(self):
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        return self.x.copy()

    def update(self, z):
        y = z - self.H @ self.x
        innovation = y @ y.T
        self.R = 0.95 * self.R + 0.05 * innovation
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        I = np.eye(self.dim_state)
        # 【v1.5修复】Joseph形式更新P，保证对称正定
        KH = K @ self.H
        IKH = I - KH
        self.P = IKH @ self.P @ IKH.T + K @ self.R @ K.T
        return self.x.copy()

    def filter_sequence(self, sequence):
        filtered_states = []
        predictions = []
        for i, obs in enumerate(sequence):
            pred = self.predict()
            predictions.append(pred[0, 0])
            if not np.isnan(obs):
                state = self.update(np.array([[obs]]))
            else:
                state = self.x.copy()
            filtered_states.append(state[0, 0])
        return np.array(filtered_states), np.array(predictions)


def kalman_predict_returns(returns, horizons=[1, 2, 3, 4, 5]):
    """
    使用卡尔曼滤波预测收益率
    v1.1修复：clip范围改为(-0.3, 0.3)小数形式，表示±30%
    """
    returns_clean = returns[~np.isnan(returns)]
    kf = AdaptiveKalmanFilter()
    filtered, predictions = kf.filter_sequence(returns_clean.values)
    return_predictions = {}
    current_state = kf.x.copy()
    for h in horizons:
        pred_state = current_state.copy()
        for _ in range(h):
            pred_state = kf.F @ pred_state
        # 【v1.1修复】收益率是小数形式，clip到±0.3（±30%）
        cumulative_return = pred_state[0, 0]
        cumulative_return = float(np.clip(cumulative_return, -0.3, 0.3))
        return_predictions[h] = cumulative_return
    return return_predictions, filtered


# ==========================================
# 模块7：机器学习预测模块 (ML Predictor Module)
# v1.1修复：标签保持小数，clip范围改为小数形式，修复数据泄漏
# ==========================================

def prepare_ml_dataset(df, target_horizons=[1, 2, 3, 4, 5]):
    """
    准备机器学习数据集
    v1.1修复：添加时间序列分割，避免数据泄漏
    v1.5修复：返回最新特征行用于预测（而非训练集末行）
    """
    df = df.copy()
    close_prices = df['close'].astype(float)

    # 生成预测标签（小数形式，不*100）
    for h in target_horizons:
        df[f'target_return_{h}d'] = close_prices.pct_change(h).shift(-h)

    exclude_cols = ['date', 'code', 'code_std', 'name', 'industry', 'open', 'high', 'low', 'close', 'volume', 'amount']
    exclude_cols += [f'target_return_{h}d' for h in target_horizons]
    feature_cols = [col for col in df.columns if col not in exclude_cols]

    for col in feature_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    df_recent = df.tail(TRAINING_DAYS).copy()

    # 【v1.5修复】提取最新特征行（不受标签缺失影响）
    latest_row = df_recent[feature_cols].iloc[-1:]
    latest_features = latest_row.values.astype(float)
    if np.any(np.isnan(latest_features)):
        # 最新行特征有NaN，用df_clean最后一行兜底
        latest_features = None

    # 【v1.1修复】先计算所有特征和标签，然后按时间分割
    df_clean = df_recent.dropna(subset=feature_cols + [f'target_return_{h}d' for h in target_horizons])

    # 【v1.1修复】时间序列分割：前80%训练，后20%验证
    n_samples = len(df_clean)
    train_size = int(n_samples * 0.8)

    if train_size < 50:
        # 数据不足，使用全部数据
        X = df_clean[feature_cols].values.astype(float)
        y_dict = {h: df_clean[f'target_return_{h}d'].values.astype(float) for h in target_horizons}
    else:
        # 前80%训练，后20%验证
        df_train = df_clean.iloc[:train_size]
        df_val = df_clean.iloc[train_size:]

        X = df_train[feature_cols].values.astype(float)
        y_dict = {h: df_train[f'target_return_{h}d'].values.astype(float) for h in target_horizons}

    return X, y_dict, feature_cols, df_clean, latest_features


def train_ml_model(X, y, model_type='auto'):
    """
    训练机器学习模型
    v1.7重构：强化正则化+早停+验证集，根治过拟合
    """
    X_mean = np.mean(X, axis=0)
    X_std = np.std(X, axis=0) + 1e-10
    X_scaled = (X - X_mean) / X_std

    class SimpleScaler:
        def __init__(self, mean, std):
            self.mean = mean
            self.std = std

        def transform(self, x):
            return (x - self.mean) / self.std

    scaler = SimpleScaler(X_mean, X_std)

    if DEPENDENCY_STATUS['lightgbm'] and model_type in ['auto', 'lightgbm']:
        # 【v1.7】时间序列分割：前80%训练，后20%验证（用于早停）
        n = len(X_scaled)
        split_idx = int(n * 0.8)
        if split_idx < 40:
            split_idx = n  # 数据太少不分割
        if split_idx < n:
            X_tr, X_val = X_scaled[:split_idx], X_scaled[split_idx:]
            y_tr, y_val = y[:split_idx], y[split_idx:]
        else:
            X_tr, X_val = X_scaled, X_scaled[-1:]
            y_tr, y_val = y, y[-1:]

        train_data = lgb.Dataset(X_tr, label=y_tr)
        val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)

        # 【v1.7】大幅强化正则化
        params = {
            'objective': 'regression',
            'metric': 'mse',
            'verbosity': -1,
            'learning_rate': 0.02,  # 降学习率 0.05→0.02
            'max_depth': 3,  # 限制深度 5→3
            'num_leaves': 8,  # 限制叶子 20→8
            'feature_fraction': 0.6,  # 特征采样 0.8→0.6
            'bagging_fraction': 0.7,  # 新增样本采样
            'bagging_freq': 1,
            'min_data_in_leaf': 20,  # 新增：每叶最少样本
            'lambda_l1': 0.5,  # 新增：L1正则
            'lambda_l2': 1.0,  # 新增：L2正则
            'min_gain_to_split': 0.01,  # 新增：最小分裂增益
        }
        # 【v1.7】早停：50轮不提升则停止
        callbacks = [lgb.early_stopping(stopping_rounds=50, verbose=False)]
        model = lgb.train(params, train_data, num_boost_round=500,
                          valid_sets=[val_data], callbacks=callbacks)
        return model, scaler, 'lightgbm'
    elif DEPENDENCY_STATUS['sklearn']:
        model = GradientBoostingRegressor(
            n_estimators=200, learning_rate=0.02, max_depth=3,
            min_samples_leaf=20, subsample=0.7,
            random_state=42, validation_fraction=0.2,
            n_iter_no_change=30, tol=1e-4
        )
        model.fit(X_scaled, y)
        return model, scaler, 'sklearn'
    else:

        class SimpleModel:
            def __init__(self, pred_value):
                self.pred_value = pred_value

            def predict(self, X):
                return np.full(X.shape[0], self.pred_value)

        pred_value = np.mean(y) if len(y) > 0 else 0
        model = SimpleModel(pred_value)
        return model, scaler, 'simple'


def predict_with_ml(X, y_dict, target_horizons=[1, 2, 3, 4, 5], X_latest_row=None):
    """
    使用机器学习模型进行多步预测
    v1.1修复：预测结果保持小数形式，clip范围改为(-0.3, 0.3)
    v1.5修复：使用最新特征行预测，而非训练集末行
    """
    predictions = {}
    prediction_vars = {}

    for h in target_horizons:
        y = y_dict[h]
        if len(y) < 50:
            # 数据不足，使用均值预测
            pred = float(np.clip(np.mean(y) if len(y) > 0 else 0, -0.3, 0.3))
            predictions[h] = pred
            prediction_vars[h] = float(np.var(y) * 3.0) if len(y) > 0 else 0.03  # v1.7：膨胀3倍
            continue

        # 训练模型（只用训练集）
        model, scaler, model_type = train_ml_model(X, y)

        # 【v1.5修复】优先使用最新特征行，否则降级到训练集末行
        if X_latest_row is not None:
            X_latest = X_latest_row.copy()
        else:
            X_latest = X[-1:].copy()
        X_latest_scaled = scaler.transform(X_latest)

        if model_type == 'lightgbm':
            pred = model.predict(X_latest_scaled)[0]
        else:
            pred = model.predict(X_latest_scaled)[0]

        # 计算预测方差
        X_scaled = scaler.transform(X)
        y_pred_all = model.predict(X_scaled)
        pred_var = np.var(y - y_pred_all)

        # 【v1.7】ML方差膨胀3倍：模型预测不确定度远大于训练残差
        pred_var *= 3.0

        # 【v1.1修复】clip到±0.3（±30%），保持小数形式
        pred = float(np.clip(pred, -0.3, 0.3))
        predictions[h] = pred
        prediction_vars[h] = float(pred_var)

        # 【v1.1修复】打印时转为百分比显示

    return predictions, prediction_vars


# ==========================================
# 模块8：贝叶斯多证据融合模块
# v1.1修复：clip范围改为小数形式，修复方差计算
# ==========================================

def bayesian_fusion(predictions_list, variances_list):
    """
    贝叶斯多证据融合
    v1.1修复：clip范围改为(-0.4, 0.4)小数形式
    """
    precisions = []
    weighted_preds = []

    for pred, var in zip(predictions_list, variances_list):
        precision = 1.0 / (var + 1e-10)
        precisions.append(precision)
        weighted_preds.append(pred * precision)

    total_precision = sum(precisions)
    fused_mean = sum(weighted_preds) / total_precision
    fused_var = 1.0 / total_precision
    weights = [p / total_precision for p in precisions]

    # 【v1.1修复】最终收益clip到±0.4（±40%），保持小数形式
    fused_mean = float(np.clip(fused_mean, -0.4, 0.4))

    # 【v1.1修复】打印时转为百分比显示

    return {
        'fused_mean': fused_mean,
        'fused_var': fused_var,
        'weights': weights,
        'individual_preds': predictions_list,
        'individual_vars': variances_list
    }


def fuse_all_predictions(ml_preds, ml_vars, kalman_preds, garch_vols, horizons=[1, 2, 3, 4, 5]):
    """
    融合所有模型的预测结果
    v1.1修复：卡尔曼方差基于GARCH日化方差计算
    """
    fused_results = {}

    for h in horizons:
        preds = []
        vars_ = []

        if h in ml_preds:
            preds.append(ml_preds[h])
            vars_.append(ml_vars[h])

        if h in kalman_preds:
            preds.append(kalman_preds[h])

            # 【v1.1修复】卡尔曼预测是h日收益率（小数），其方差基于GARCH日化方差
            # 【v1.5修复】GARCH波动率在volatility_predictions子字典中，非顶层key
            garch_annual_vol = garch_vols.get('volatility_predictions', {}).get(f'h_{h}d',
                                                                                garch_vols.get('volatility', 0.25))
            garch_daily_var = (garch_annual_vol / np.sqrt(252)) ** 2  # 日化方差
            kalman_var = h * garch_daily_var  # h日收益率的方差
            vars_.append(kalman_var)

        fused = bayesian_fusion(preds, vars_)
        fused_results[h] = fused

    return fused_results


# ==========================================
# 模块9：凯利公式与风险平价模块 (Portfolio Optimizer Module)
# 设计意图：计算最优仓位分配
# 模块规模：约150行
# v1.1修复：去掉不必要的/100，添加风险平价调用和迭代约束调整
# ==========================================

def improved_kelly_criterion(expected_return, volatility, win_rate=None, kelly_fraction=0.5):
    """
    改进凯利公式
    设计意图：计算单只股票的最优投资比例
    输入：expected_return - 期望收益率（小数）
          volatility - 波动率（小数）
          win_rate - 胜率
          kelly_fraction - 凯利分数（半凯利更稳健）
    输出：最优仓位比例（小数）
    """

    risk_free = 0.03 / 252  # 日化无风险利率

    if volatility <= 0:
        return 0

    # 计算凯利最优比例
    excess_return = expected_return - risk_free
    kelly_ratio = excess_return / (volatility ** 2)

    # 应用凯利分数（通常使用半凯利）
    kelly_ratio = kelly_ratio * kelly_fraction

    # 限制仓位范围
    kelly_ratio = max(0, min(kelly_ratio, MAX_POSITION_PER_STOCK))

    return kelly_ratio


def risk_parity_optimization(returns_matrix, target_volatility=0.15):
    """
    风险平价优化
    设计意图：计算使各资产风险贡献相等的权重
    输入：returns_matrix - 各资产收益率矩阵
          target_volatility - 目标组合波动率
    输出：最优权重
    """

    n_assets = returns_matrix.shape[1]

    # 计算协方差矩阵
    cov_matrix = np.cov(returns_matrix.T)

    def risk_contribution(weights):
        """计算各资产的风险贡献"""
        port_vol = np.sqrt(weights.T @ cov_matrix @ weights)
        marginal_contrib = cov_matrix @ weights / port_vol
        return weights * marginal_contrib

    def objective(weights):
        """目标函数：最小化风险贡献差异"""
        rc = risk_contribution(weights)
        return np.sum((rc - rc.mean()) ** 2)

    # 约束：权重和为1，非负
    constraints = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1}]
    bounds = [(0, 1) for _ in range(n_assets)]
    init_weights = np.ones(n_assets) / n_assets

    result = minimize(
        objective,
        init_weights,
        method='SLSQP',
        bounds=bounds,
        constraints=constraints
    )

    optimal_weights = result.x

    # 调整到目标波动率
    current_vol = np.sqrt(optimal_weights @ cov_matrix @ optimal_weights)
    if current_vol > 0:
        leverage = target_volatility / current_vol
        optimal_weights = optimal_weights * min(leverage, 1)

    return optimal_weights


def calculate_portfolio_weights(predictions_dict, all_returns_data):
    """
    计算最终投资组合权重
    v1.7重构：多因子信号门控 + 弹性仓位 + 趋势确认
    核心变化：
    1. 信号门控：综合评分<60的股票不入场
    2. 弹性持仓：不强凑8只，0-8只均可
    3. 仓位与信号强度挂钩：强多信号多配，偏多信号少配
    4. 排序标准：综合评分 > 纯夏普
    """

    portfolio_candidates = []

    # 第一步：计算每只股票的信号评分和凯利仓位
    for code, pred_info in predictions_dict.items():
        h = 3
        if h not in pred_info['fused']:
            continue

        fused = pred_info['fused'][h]
        expected_return = fused['fused_mean']
        volatility = np.sqrt(fused['fused_var'])

        # 获取v1.7信号评分
        signal_info = pred_info.get('signal', {})
        composite_score = signal_info.get('composite_score', 50)
        signal_level = signal_info.get('level', '中性')
        entry_type = signal_info.get('entry_type', '观望回避')
        signal_details = signal_info.get('details', {})

        # 凯利仓位计算
        kelly_weight = improved_kelly_criterion(expected_return, volatility, kelly_fraction=0.5)

        current_price = pred_info.get('current_price', 100)

        portfolio_candidates.append({
            'code': code,
            'name': pred_info['name'],
            'industry': pred_info['industry'],
            'expected_return': expected_return * 100,
            'volatility': volatility * 100,
            'kelly_weight': kelly_weight,
            'current_price': current_price,
            'sharpe': expected_return / (volatility + 1e-10),
            'composite_score': composite_score,
            'signal_level': signal_level,
            'entry_type': entry_type,
            'signal_details': signal_details
        })

    # 第二步：【v1.7核心】信号门控 - 只保留综合评分>=60的候选
    qualified = [s for s in portfolio_candidates if s['composite_score'] >= 60]
    rejected = [s for s in portfolio_candidates if s['composite_score'] < 60]

    print(f"  [信号门控] 全部{len(portfolio_candidates)}只 → 通过{len(qualified)}只 / 淘汰{len(rejected)}只")
    if rejected:
        rej_names = [f"{s['name']}({s['signal_level']})" for s in rejected[:5]]
        print(f"  淘汰示例: {', '.join(rej_names)}{'...' if len(rejected) > 5 else ''}")

    if len(qualified) == 0:
        print("  ⚠ 无合格信号，建议空仓观望！")
        return []

    # 第三步：按综合评分排序（不再用夏普排序）
    qualified.sort(key=lambda x: x['composite_score'], reverse=True)

    # 第四步：行业约束筛选
    industry_counts = {}
    final_selection = []

    for stock in qualified:
        industry = stock['industry']
        industry_counts[industry] = industry_counts.get(industry, 0) + 1

        if industry_counts[industry] <= 2:
            final_selection.append(stock)

        if len(final_selection) >= HOLDING_COUNT_RANGE[1]:
            break

    # 第五步：【v1.7】仓位与信号强度挂钩
    # 强多(>=72) → 基础权重×1.5，偏多(60-71) → 基础权重×1.0
    for s in final_selection:
        score = s['composite_score']
        if score >= 72:
            signal_multiplier = 1.5
        elif score >= 66:
            signal_multiplier = 1.2
        else:
            signal_multiplier = 0.8
        s['kelly_weight'] *= signal_multiplier

    # 第六步：风险平价优化
    if len(final_selection) >= 2:
        selected_codes = [s['code'] for s in final_selection]
        returns_list = []
        valid_codes = []

        for code in selected_codes:
            if code in all_returns_data:
                returns_list.append(all_returns_data[code])
                valid_codes.append(code)

        if len(returns_list) >= 2:
            min_len = min(len(r) for r in returns_list)
            aligned_returns = np.column_stack([r[-min_len:] for r in returns_list])
            rp_weights = risk_parity_optimization(aligned_returns)

            for i, stock in enumerate(final_selection):
                if stock['code'] in valid_codes:
                    idx = valid_codes.index(stock['code'])
                    if idx < len(rp_weights):
                        # 70%信号驱动 + 30%风险平价（v1.7：信号权重提高）
                        stock['kelly_weight'] = 0.7 * stock['kelly_weight'] + 0.3 * rp_weights[idx]

    # 第七步：约束调整和归一化
    max_iterations = 10
    for iteration in range(max_iterations):
        total_kelly = sum(s['kelly_weight'] for s in final_selection)
        for s in final_selection:
            if total_kelly > 0:
                s['final_weight'] = min(s['kelly_weight'] / total_kelly, MAX_POSITION_PER_STOCK)
            else:
                s['final_weight'] = 1.0 / len(final_selection)

        any_violation = False
        for s in final_selection:
            if s['final_weight'] > MAX_POSITION_PER_STOCK:
                s['final_weight'] = MAX_POSITION_PER_STOCK
                any_violation = True

        industry_weights = {}
        for s in final_selection:
            ind = s['industry']
            industry_weights[ind] = industry_weights.get(ind, 0) + s['final_weight']

        for ind, total_weight in industry_weights.items():
            if total_weight > MAX_POSITION_PER_INDUSTRY:
                reduction_ratio = MAX_POSITION_PER_INDUSTRY / total_weight
                for s in final_selection:
                    if s['industry'] == ind:
                        s['final_weight'] = s['final_weight'] * reduction_ratio
                any_violation = True

        if not any_violation:
            break

    # 再次归一化
    total_final = sum(s['final_weight'] for s in final_selection)
    if total_final > 0:
        for s in final_selection:
            s['final_weight'] = s['final_weight'] / total_final

    # 主仓/抄底仓分配
    for s in final_selection:
        s['main_capital_allocation'] = s['final_weight'] * MAIN_CAPITAL
        s['bottom_capital_allocation'] = s['final_weight'] * BOTTOM_CAPITAL * 0.5

    # 第八步：牛熊环境自适应
    if MARKET_ENV['available']:
        regime = MARKET_ENV['market_regime']
        sector_signals = MARKET_ENV.get('sector_signals', {})
        if regime == '熊市':
            for s in final_selection:
                s['final_weight'] *= 0.7
                industry = s.get('industry', '')
                sig = sector_signals.get(industry, {})
                if sig.get('regime') == '熊市':
                    s['final_weight'] *= 0.5
            total = sum(s['final_weight'] for s in final_selection)
            if total > 0:
                for s in final_selection:
                    s['final_weight'] /= total
            print(f"  📉 熊市减仓调整完成")
        elif regime == '牛市':
            for s in final_selection:
                industry = s.get('industry', '')
                sig = sector_signals.get(industry, {})
                if sig.get('regime') == '牛市':
                    s['final_weight'] *= 1.15
            total = sum(s['final_weight'] for s in final_selection)
            if total > 0:
                for s in final_selection:
                    s['final_weight'] /= total
            print(f"  📈 牛市加仓调整完成")

    # 输出组合摘要
    print(f"  [组合] ✓ 选择{len(final_selection)}只股票（信号门控后弹性持仓）")
    for i, s in enumerate(final_selection, 1):
        print(
            f"    {i}. {s['name']} | 评分{s['composite_score']:.0f} | {s['signal_level']} | {s['entry_type']} | 权重{s['final_weight']:.1%}")

    return final_selection

    print("━" * 70)


# ==========================================
# 模块10：风控模块 (Risk Manager Module)
# 设计意图：实现动态止盈止损和回撤控制
# 模块规模：约120行
# v1.1修复：自动修正违规仓位、接入回撤检查、使用实际入场价
# ==========================================

def calculate_dynamic_stop_loss(entry_price, volatility, lookback_days=20):
    """
    动态止盈止损计算
    设计意图：基于波动率计算自适应的止盈止损位
    输入：entry_price - 入场价格（实际价格，不再硬编码100）
          volatility - 年化波动率（小数形式）
          lookback_days - 回看天数
    输出：止盈止损价格
    """

    # 日化波动率（volatility已经是小数形式）
    daily_vol = volatility / np.sqrt(252)

    # 止损：基于波动率的ATR倍数
    stop_loss_multiplier = 2.0
    stop_loss_price = entry_price * (1 - stop_loss_multiplier * daily_vol * np.sqrt(lookback_days))

    # 止盈：盈亏比2:1
    take_profit_multiplier = 4.0
    take_profit_price = entry_price * (1 + take_profit_multiplier * daily_vol * np.sqrt(lookback_days))

    # 移动止损跟踪比例
    trailing_stop_pct = 1.5 * daily_vol * np.sqrt(5)

    return {
        'stop_loss_price': stop_loss_price,
        'take_profit_price': take_profit_price,
        'trailing_stop_pct': trailing_stop_pct,
        'volatility': volatility
    }


def calculate_max_drawdown(equity_curve):
    """
    计算最大回撤
    设计意图：监控账户最大回撤风险
    输入：equity_curve - 权益曲线
    输出：最大回撤和当前回撤
    """

    equity = np.array(equity_curve)
    peak = np.maximum.accumulate(equity)
    drawdown = (equity - peak) / peak

    max_dd = drawdown.min()
    current_dd = drawdown[-1]

    if abs(current_dd) > MAX_DRAWDOWN:
        print(f"  ⚠ 回撤超过红线 {MAX_DRAWDOWN:.0%}，建议减仓")

    return {
        'max_drawdown': max_dd,
        'current_drawdown': current_dd,
        'breach': abs(current_dd) > MAX_DRAWDOWN
    }


def risk_control_check(portfolio, account_equity):
    """
    综合风控检查（强化版）
    设计意图：执行所有风控规则检查，并自动修正违规仓位
    输入：portfolio - 投资组合
          account_equity - 账户权益
    输出：风控报告和修正后的投资组合
    v1.1修复：自动修正违规仓位，返回修正后的portfolio和违规记录
    """

    # 复制投资组合以避免修改原始数据
    portfolio = [s.copy() for s in portfolio]

    # 牛熊环境自适应：根据市场环境调整风控策略
    effective_max_drawdown = MAX_DRAWDOWN
    if MARKET_ENV['available']:
        regime = MARKET_ENV['market_regime']
        if regime == '熊市':
            # 熊市收紧回撤红线至70%
            effective_max_drawdown = MAX_DRAWDOWN * 0.7
            print(f"  📉 熊市环境，回撤红线收紧至 {effective_max_drawdown:.0%}")
        elif regime == '牛市':
            # 牛市放宽回撤红线至130%
            effective_max_drawdown = MAX_DRAWDOWN * 1.3

    risk_report = {
        'position_checks': [],
        'industry_checks': [],
        'drawdown_check': None,
        'adjustments': []
    }

    # 第一步：检查并修正单只股票仓位超限
    for stock in portfolio:
        if stock['final_weight'] > MAX_POSITION_PER_STOCK:
            old_weight = stock['final_weight']
            stock['final_weight'] = MAX_POSITION_PER_STOCK
            risk_report['position_checks'].append(
                f"⚠ {stock['name']}仓位{old_weight:.2%}超过上限，已截断至{MAX_POSITION_PER_STOCK:.0%}"
            )
            risk_report['adjustments'].append(
                f"{stock['name']}: {old_weight:.2%} → {MAX_POSITION_PER_STOCK:.2%}"
            )

    # 第二步：检查并修正行业仓位超限
    industry_weights = {}
    for stock in portfolio:
        ind = stock['industry']
        industry_weights[ind] = industry_weights.get(ind, 0) + stock['final_weight']

    for industry, total_weight in industry_weights.items():
        if total_weight > MAX_POSITION_PER_INDUSTRY:
            excess = total_weight - MAX_POSITION_PER_INDUSTRY
            # 找出该行业的股票
            industry_stocks = [s for s in portfolio if s['industry'] == industry]
            total_industry_weight = sum(s['final_weight'] for s in industry_stocks)

            if total_industry_weight > 0:
                for stock in industry_stocks:
                    # 按比例缩减
                    reduction = stock['final_weight'] * (excess / total_industry_weight)
                    stock['final_weight'] -= reduction
                    risk_report['adjustments'].append(
                        f"{stock['name']}({industry}): 缩减{-reduction:.2%}"
                    )

            risk_report['industry_checks'].append(
                f"⚠ {industry}行业仓位{total_weight:.2%}超过上限{MAX_POSITION_PER_INDUSTRY:.0%}，已按比例缩减"
            )

    # 第三步：归一化权重以确保总和为1
    total_weight = sum(s['final_weight'] for s in portfolio)
    if total_weight > 0 and abs(total_weight - 1.0) > 0.001:
        for stock in portfolio:
            stock['final_weight'] = stock['final_weight'] / total_weight

    # 第四步：检查回撤
    if isinstance(account_equity, list) and len(account_equity) > 0:
        drawdown_result = calculate_max_drawdown(account_equity)
        risk_report['drawdown_check'] = drawdown_result

    if risk_report['adjustments']:
        pass

    return portfolio, risk_report

    print("━" * 70)


# ==========================================
# 模块11：输出报告模块 (Output Generator Module)
# 设计意图：生成所有要求的输出文件，自动处理路径
# 模块规模：约180行
# v1.1修复：龙虎榜收益率显示*100，使用实际当前价格计算止盈止损
# ==========================================

def save_data_files(all_stocks_data, industry_agg_data, index_data):
    """
    保存数据文件
    设计意图：保存实时股票数据、行业聚合数据、指数数据
    输入：各类数据字典
    输出：保存的文件路径
    """

    # 1. 实时股票数据
    all_stocks_list = []
    for code, df in all_stocks_data.items():
        if df is not None and len(df) > 0:
            all_stocks_list.append(df)

    if all_stocks_list:
        stocks_df = pd.concat(all_stocks_list, ignore_index=True)
        stocks_path = os.path.join(OUTPUT_PATH, '实时股票数据.csv')
        stocks_df.to_csv(stocks_path, index=False, encoding='utf-8-sig')

    # 2. 行业板块聚合数据
    industry_list = []
    for industry, df in industry_agg_data.items():
        if df is not None and len(df) > 0:
            industry_list.append(df)

    if industry_list:
        industry_df = pd.concat(industry_list, ignore_index=True)
        industry_path = os.path.join(OUTPUT_PATH, '行业板块聚合数据.csv')
        industry_df.to_csv(industry_path, index=False, encoding='utf-8-sig')

    # 3. 三大指数数据
    index_list = []
    for code, df in index_data.items():
        if df is not None and len(df) > 0:
            index_list.append(df)

    if index_list:
        index_df = pd.concat(index_list, ignore_index=True)
        index_path = os.path.join(OUTPUT_PATH, '三大指数数据.csv')
        index_df.to_csv(index_path, index=False, encoding='utf-8-sig')


def generate_dragon_tiger_board(all_predictions, horizon):
    """
    生成龙虎榜报告
    设计意图：按预测期限生成Excel龙虎榜
    输入：all_predictions - 所有股票预测结果
          horizon - 预测天数
    输出：龙虎榜DataFrame
    v1.1修复：显示时将收益率转为百分比（*100）
    """
    board_data = []

    for code, pred_info in all_predictions.items():
        if horizon in pred_info['fused']:
            fused = pred_info['fused'][horizon]
            signal_info = pred_info.get('signal', {})
            board_data.append({
                '股票代码': pred_info['code_std'],
                '股票名称': pred_info['name'],
                '所属行业': pred_info['industry'],
                '综合评分': signal_info.get('composite_score', 0),
                '信号等级': signal_info.get('level', '中性'),
                '入场类型': signal_info.get('entry_type', '未知'),
                '期望收益率(%)': round(fused['fused_mean'] * 100, 4),
                '收益方差(%)': round(fused['fused_var'] * 10000, 6),
                '波动率': pred_info.get('volatility', {}).get(f'h_{horizon}d', 0)
            })

    df = pd.DataFrame(board_data)

    # v1.7：按综合评分排序（不再按收益率）
    df = df.sort_values('综合评分', ascending=False)

    # 波动率排名
    df['波动率排名'] = df['波动率'].rank(ascending=True, method='min')

    return df


def save_dragon_tiger_boards(all_predictions, horizons=[1, 2, 3, 4, 5]):
    """
    保存所有龙虎榜Excel文件
    设计意图：为每个预测期限生成独立的龙虎榜
    """

    with pd.ExcelWriter(os.path.join(OUTPUT_PATH, '龙虎榜报告.xlsx')) as writer:
        for h in horizons:
            df = generate_dragon_tiger_board(all_predictions, h)
            df.to_excel(writer, sheet_name=f'{h}日预测', index=False)


def generate_investment_advice(portfolio, all_predictions):
    """
    生成最终投资建议
    设计意图：生成Excel和TXT格式的投资建议
    输入：portfolio - 最优投资组合
          all_predictions - 所有预测结果
    v1.1修复：使用实际当前价格计算止盈止损
    """

    # 1. Excel格式建议
    advice_data = []
    for stock in portfolio:
        # 【v1.1修复】获取实际当前价格
        current_price = stock.get('current_price', 100)
        if current_price is None or current_price <= 0:
            current_price = 100  # 降级处理

        # 获取动态止盈止损（使用实际当前价格）
        stop_loss = calculate_dynamic_stop_loss(current_price, stock['volatility'] / 100)

        advice_data.append({
            '股票代码': stock['code'],
            '股票名称': stock['name'],
            '所属行业': stock['industry'],
            '综合评分': stock.get('composite_score', 0),
            '信号等级': stock.get('signal_level', '中性'),
            '入场类型': stock.get('entry_type', '未知'),
            '期望收益率(%)': round(stock['expected_return'], 4),
            '波动率(%)': round(stock['volatility'], 4),
            '最终投资比例(%)': round(stock['final_weight'] * 100, 2),
            '主仓分配(元)': round(stock['main_capital_allocation'], 2),
            '抄底仓分配(元)': round(stock['bottom_capital_allocation'], 2),
            '止损价(元)': round(stop_loss['stop_loss_price'], 2),
            '止盈价(元)': round(stop_loss['take_profit_price'], 2),
            '止损价(相对%)': round((stop_loss['stop_loss_price'] / current_price - 1) * 100, 2),
            '止盈价(相对%)': round((stop_loss['take_profit_price'] / current_price - 1) * 100, 2)
        })

    df_advice = pd.DataFrame(advice_data)

    # 保存Excel
    excel_path = os.path.join(OUTPUT_PATH, '最终投资建议.xlsx')
    df_advice.to_excel(excel_path, index=False)

    # 2. TXT格式建议
    txt_path = os.path.join(OUTPUT_PATH, '最终投资建议.txt')

    with open(txt_path, 'w', encoding='utf-8') as f:
        f.write("=" * 60 + "\n")
        f.write("Python量化回测系统 v1.7 - 最终投资建议报告\n")
        f.write("生成时间: " + datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S') + "\n")
        f.write("=" * 60 + "\n\n")

        f.write("【账户资金配置】\n")
        f.write(f"初始资金: {INITIAL_CAPITAL:,}元\n")
        f.write(f"主仓资金: {MAIN_CAPITAL:,}元 (50%长线 + 50%短线)\n")
        f.write(f"抄底仓资金: {BOTTOM_CAPITAL:,}元\n")
        f.write(f"单票最大仓位: {MAX_POSITION_PER_STOCK:.0%}\n")
        f.write(f"单行业最大仓位: {MAX_POSITION_PER_INDUSTRY:.0%}\n")
        f.write(f"最大回撤红线: {MAX_DRAWDOWN:.0%}\n\n")

        f.write("【推荐持仓组合】\n")
        for i, stock in enumerate(portfolio, 1):
            # 获取当前价格
            current_price = stock.get('current_price', 100)
            stop_loss = calculate_dynamic_stop_loss(current_price, stock['volatility'] / 100)

            f.write(f"\n{i}. {stock['name']}({stock['code']}) - {stock['industry']}\n")
            f.write(
                f"   综合评分: {stock.get('composite_score', 0):.0f} | 信号: {stock.get('signal_level', '中性')} | 入场: {stock.get('entry_type', '未知')}\n")
            signal_d = stock.get('signal_details', {})
            if signal_d:
                f.write(
                    f"   因子: 趋势{signal_d.get('trend', 0):.0f} 动量{signal_d.get('momentum', 0):.0f} 入场{signal_d.get('entry_quality', 0):.0f} ML{signal_d.get('ml_signal', 0):.0f}\n")
            f.write(f"   期望收益率: {stock['expected_return']:.2f}%\n")
            f.write(f"   投资比例: {stock['final_weight']:.1%}\n")
            f.write(f"   主仓分配: {stock['main_capital_allocation']:.0f}元\n")
            f.write(f"   抄底仓分配: {stock['bottom_capital_allocation']:.0f}元\n")
            f.write(f"   当前价格: {current_price:.2f}元\n")
            f.write(
                f"   止损价: {stop_loss['stop_loss_price']:.2f}元 ({stop_loss['stop_loss_price'] / current_price - 1:.2%})\n")
            f.write(
                f"   止盈价: {stop_loss['take_profit_price']:.2f}元 ({stop_loss['take_profit_price'] / current_price - 1:.2%})\n")

        f.write("\n" + "=" * 60 + "\n")
        f.write("【风险提示】\n")
        f.write("1. 本建议仅供参考，不构成投资建议\n")
        f.write("2. 股市有风险，投资需谨慎\n")
        f.write("3. 请严格执行止盈止损纪律\n")
        f.write("=" * 60 + "\n")

        print("━" * 70)


# ==========================================
# 模块12：错误处理与监视模块 (Error Handler Module)
# 设计意图：全局异常捕获，错误流程记录
# 模块规模：约80行
# ==========================================

class QuantErrorHandler:
    """
    量化系统错误处理器
    设计意图：武力防错 - 捕获并记录所有异常
    """

    def __init__(self):
        self.error_log = []

    def capture_exception(self, error, context=""):
        """
        捕获异常并记录详细信息
        设计意图：完整输出错误流程和相关数据
        """
        import traceback

        error_info = {
            'timestamp': datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'error_type': type(error).__name__,
            'error_message': str(error),
            'context': context,
            'traceback': traceback.format_exc()
        }

        self.error_log.append(error_info)

        print(f"  ⚠ 异常 │ {error_info['error_type']}")
        print(f"         │ 上下文 {context}")
        print(f"         │ 信息 {error_info['error_message']}")

        # 保存错误日志
        self.save_error_log()

        return error_info

    def save_error_log(self):
        """保存错误日志到文件【v1.5修复】改用追加模式，避免覆盖历史错误"""
        log_path = os.path.join(OUTPUT_PATH, 'error_log.txt')
        if len(self.error_log) == 0:
            return
        # 只追加最新一条错误，避免重复写入
        err = self.error_log[-1]
        with open(log_path, 'a', encoding='utf-8') as f:
            f.write(f"\n{'=' * 60}\n")
            f.write(f"错误 - {err['timestamp']}\n")
            f.write(f"类型: {err['error_type']}\n")
            f.write(f"信息: {err['error_message']}\n")
            f.write(f"上下文: {err['context']}\n")
            f.write(f"堆栈:\n{err['traceback']}\n")

    def get_error_summary(self):
        """获取错误摘要"""
        return {
            'total_errors': len(self.error_log),
            'error_types': [e['error_type'] for e in self.error_log]
        }


# 全局错误处理器实例
error_handler = QuantErrorHandler()


def safe_execute(func, *args, context="", **kwargs):
    """
    安全执行包装器
    设计意图：为函数调用提供异常保护
    """
    try:
        return func(*args, **kwargs), None
    except Exception as e:
        error_info = error_handler.capture_exception(e, context)
        return None, error_info

        print("━" * 70)


# ==========================================
# 模块13：主程序入口 (Main Entry)
# 设计意图：胶水代码，串联所有模块
# 模块规模：约150行
# v1.1修复：添加回撤检查，获取实际当前价格
# ==========================================

def main():
    """
    主程序入口
    设计意图：按顺序调用所有模块，完成完整回测流程
    民众防错：模块间仅通过胶水代码连接，无横向依赖
    """
    print("\n" + "━" * 70)
    print("  Python量化回测系统 v1.8")

    start_time = time.time()

    # 计算日期范围
    end_date = datetime.datetime.now().strftime('%Y-%m-%d')
    start_date = (datetime.datetime.now() - timedelta(days=400)).strftime('%Y-%m-%d')
    print(f"  回测日期范围: {start_date} 至 {end_date}")

    # ==================== mootdx实时行情连接 ====================
    print("  连接mootdx实时行情服务器...")
    MOOTDX_CLIENT.connect()

    # ==================== 阶段0: 牛熊环境预判 ====================
    run_bull_bear_prediction()

    # ==================== 阶段1: 数据获取 ====================
    print("\n" + "━" * 70)
    print("  阶段1 ▸ 获取股票和指数数据")
    print("━" * 70)

    # 批量登录baostock
    baostock_batch_login()

    all_stocks_data = {}
    all_predictions = {}

    # 获取所有股票数据
    for industry, stocks in STOCK_POOL.items():
        for stock_info in stocks:
            result, error = safe_execute(
                fetch_single_stock_data,
                stock_info,
                start_date,
                end_date,
                context=f"获取{stock_info['name']}数据"
            )

            if result is not None and validate_data_quality(result):
                df_processed = preprocess_data(result)
                all_stocks_data[stock_info['code']] = df_processed

    print(f"  成功获取{len(all_stocks_data)}只股票数据")

    # 获取指数数据
    index_data = {}
    for idx_info in INDEX_LIST:
        result, error = safe_execute(
            fetch_single_stock_data,
            idx_info,
            start_date,
            end_date,
            context=f"获取{idx_info['name']}数据"
        )
        if result is not None:
            index_data[idx_info['code']] = preprocess_data(result)

    # 批量登出baostock
    baostock_batch_logout()

    # ==================== 阶段2: 数据聚合 ====================
    print("\n" + "━" * 70)
    print("  阶段2 ▸ 三维数据聚合")
    print("━" * 70)

    # 行业数据聚合
    industry_agg_data = aggregate_industry_data(all_stocks_data)

    # 指数数据处理
    processed_index = process_index_data(index_data)

    # ==================== 阶段3: 单只股票预测 ====================
    print("\n" + "━" * 70)
    print("  阶段3 ▸ 个股预测计算")
    print("━" * 70)

    # 【v1.8】实时价格优先走mootdx，baostock仅作降级兜底
    # 如果mootdx连接失败，仍需baostock获取实时价格
    if not MOOTDX_CLIENT._connected:
        baostock_batch_login()
    else:
        print("  ✓ 实时价格由mootdx提供，无需重新登录baostock")

    all_returns_for_optimization = {}

    for industry, stocks in STOCK_POOL.items():
        for stock_info in stocks:
            code = stock_info['code']
            if code not in all_stocks_data:
                continue

            # 三维数据合并
            stock_data = all_stocks_data[code]
            merged_data = merge_3d_data(stock_data, industry_agg_data, processed_index, stock_info)

            # 特征工程
            featured_data = engineer_all_features(merged_data)

            # 准备ML数据（v1.1：添加时间序列分割）
            X, y_dict, feature_cols, df_clean, latest_features = prepare_ml_dataset(featured_data)

            # 1. ML预测【v1.5修复】传入最新特征行
            ml_preds, ml_vars = predict_with_ml(X, y_dict, X_latest_row=latest_features)

            # 2. GARCH波动率预测（v1.1：return_1d是小数）
            returns = featured_data['return_1d'].dropna()
            garch_vols = predict_volatility_garch(returns)

            # 3. 卡尔曼滤波预测（v1.1：returns是小数）
            kalman_preds, kalman_filtered = kalman_predict_returns(returns)

            # 4. 贝叶斯融合
            fused_results = fuse_all_predictions(ml_preds, ml_vars, kalman_preds, garch_vols)

            # 5. 【v1.7新增】多因子信号评分
            market_regime = MARKET_ENV.get('market_regime', '中性') if MARKET_ENV.get('available') else '中性'
            composite_score, signal_level, entry_type, signal_details = calculate_composite_signal(
                featured_data, fused_results, market_regime
            )

            # 获取当天实时价格，失败则降级使用最后收盘价
            realtime_price = fetch_realtime_price(code, stock_info['name'])
            if realtime_price is not None:
                current_price = realtime_price
            else:
                current_price = stock_data['close'].iloc[-1] if len(stock_data) > 0 else 100

            # 保存预测结果（v1.7新增signal字段）
            all_predictions[code] = {
                'code': code,
                'code_std': stock_info['code_std'],
                'name': stock_info['name'],
                'industry': industry,
                'current_price': current_price,
                'ml_predictions': ml_preds,
                'ml_variances': ml_vars,
                'kalman_predictions': kalman_preds,
                'volatility': garch_vols,
                'fused': fused_results,
                'signal': {
                    'composite_score': composite_score,
                    'level': signal_level,
                    'entry_type': entry_type,
                    'details': signal_details
                }
            }

            # 保存收益率用于组合优化
            all_returns_for_optimization[code] = featured_data['return_1d'].tail(TRAINING_DAYS).values

    # 【v1.8】如果阶段3重新登录了baostock，则登出
    if not MOOTDX_CLIENT._connected:
        baostock_batch_logout()

    # ==================== 阶段4: 组合优化 ====================
    print("\n" + "━" * 70)
    print("  阶段4 ▸ 投资组合优化")
    print("━" * 70)

    optimal_portfolio = calculate_portfolio_weights(all_predictions, all_returns_for_optimization)

    # 【v1.7】空仓判断：信号门控后无合格标的
    if len(optimal_portfolio) == 0:
        print("\n" + "━" * 70)
        print("  ⚠ 所有股票信号评分均不达标，建议空仓观望")
        print("━" * 70)
        # 仍然保存数据文件
        save_data_files(all_stocks_data, industry_agg_data, processed_index)
        print("\n  ✓ 系统运行完成（空仓状态）")
        MOOTDX_CLIENT.close()
        return

    # 【v1.1修复】风控检查返回修正后的组合
    optimal_portfolio, risk_report = risk_control_check(optimal_portfolio, [INITIAL_CAPITAL])

    # 回撤检查
    print("\n" + "━" * 70)
    print("  阶段5 ▸ 回撤风险检查")
    print("━" * 70)

    # 模拟权益曲线用于回撤检查
    simulated_equity = [INITIAL_CAPITAL]
    for stock in optimal_portfolio:
        # 模拟持仓期间的收益
        expected_ret = stock['expected_return'] / 100  # 转回小数
        position_value = stock['final_weight'] * INITIAL_CAPITAL
        pnl = position_value * expected_ret
        simulated_equity.append(simulated_equity[-1] + pnl)

    drawdown_result = calculate_max_drawdown(simulated_equity)

    if drawdown_result['breach']:
        print(f"  ⚠ 回撤超过红线 {MAX_DRAWDOWN:.0%}，建议减仓")

    # ==================== 阶段5: 输出报告 ====================
    print("\n" + "━" * 70)
    print("  阶段6 ▸ 生成输出报告")
    print("━" * 70)

    # 保存数据文件
    save_data_files(all_stocks_data, industry_agg_data, index_data)

    # 生成龙虎榜（v1.1：收益率显示*100）
    save_dragon_tiger_boards(all_predictions)

    # 生成投资建议（v1.1：使用实际当前价格）
    generate_investment_advice(optimal_portfolio, all_predictions)

    # ==================== 完成 ====================
    elapsed = time.time() - start_time

    print("\n" + "━" * 70)
    print(
        f"  执行完成  │  耗时 {elapsed:.1f}s  │  输出 {OUTPUT_PATH}  │  错误 {error_handler.get_error_summary()['total_errors']}个")
    print("━" * 70)

    print("\n  ─" + "─" * 38)
    print("  推荐持仓组合")
    print("  ─" + "─" * 38)
    for i, s in enumerate(optimal_portfolio, 1):
        print(f"{i}. {s['name']:8s}  │ 收益 +{s['expected_return']:>5.2f}%  │ 仓位 {s['final_weight']:>6.1%}")

    # 关闭mootdx连接
    MOOTDX_CLIENT.close()

    return optimal_portfolio, all_predictions


# ==========================================
# 【】涨停吃板概率预测模块
# ==========================================

def _get_limit_up_features(code, days=60):
    """
    获取涨停预测所需的技术面特征
    【v1.8】mootdx优先获取日K数据，baostock降级兜底，避免反复login/logout
    输入：code - 股票代码（如 sh.600498）
          days - 获取历史天数
    输出：特征字典
    """
    features = {
        'limit_up_freq_60d': 0.0,
        'limit_up_freq_20d': 0.0,
        'limit_up_freq_5d': 0.0,
        'consecutive_boards': 0,
        'today_change_pct': 0.0,
        'volume_ratio': 1.0,
        'rsi_14': 50.0,
        'macd_signal': 0,
        'recent_high_pct': 0.0,
    }

    df = None

    # ===== 【v1.8】优先用mootdx获取日K数据（无需login/logout）=====
    if MOOTDX_CLIENT._connected:
        try:
            df_mootdx = MOOTDX_CLIENT.get_bars(code, frequency=9, offset=days + 30)
            if df_mootdx is not None and len(df_mootdx) >= 20:
                # mootdx返回的列名映射到baostock格式
                df = pd.DataFrame()
                df['date'] = df_mootdx.index if hasattr(df_mootdx.index, 'strftime') else range(len(df_mootdx))
                if 'datetime' in df_mootdx.columns:
                    df['date'] = df_mootdx['datetime'].values
                for col in ['open', 'high', 'low', 'close', 'volume', 'amount']:
                    if col in df_mootdx.columns:
                        df[col] = pd.to_numeric(df_mootdx[col], errors='coerce')
                # mootdx日K没有pctChg，自己算
                if 'close' in df.columns and len(df) > 1:
                    df['pctChg'] = df['close'].pct_change() * 100
                    df['pctChg'] = df['pctChg'].fillna(0)
                df = df.dropna(subset=['close'])
        except Exception:
            df = None

    # ===== 降级：baostock获取（仅mootdx失败时）=====
    if df is None and DEPENDENCY_STATUS['baostock']:
        max_retries = 3
        for attempt in range(max_retries):
            try:
                # 【v1.8】不再每只股票都logout/login，复用现有连接
                try:
                    test_rs = bs.query_history_k_data_plus(
                        "sh.000001", "date",
                        start_date=(datetime.datetime.now() - timedelta(days=3)).strftime('%Y-%m-%d'),
                        end_date=datetime.datetime.now().strftime('%Y-%m-%d'),
                        frequency="d", adjustflag="3"
                    )
                    if test_rs.error_code != '0':
                        try:
                            bs.logout()
                        except Exception:
                            pass
                        bs.login()
                        time.sleep(0.3)
                except Exception:
                    try:
                        bs.logout()
                    except Exception:
                        pass
                    bs.login()
                    time.sleep(0.3)

                end_date = datetime.datetime.now().strftime('%Y-%m-%d')
                start_date = (datetime.datetime.now() - datetime.timedelta(days=days + 30)).strftime('%Y-%m-%d')

                bs_code = convert_stock_code(code, to_baostock=True)
                rs = bs.query_history_k_data_plus(
                    bs_code,
                    "date,code,open,high,low,close,volume,amount,pctChg",
                    start_date=start_date,
                    end_date=end_date,
                    frequency="d",
                    adjustflag="3"
                )

                data_list = []
                while (rs.error_code == '0') & rs.next():
                    data_list.append(rs.get_row_data())

                if data_list:
                    df = pd.DataFrame(data_list, columns=rs.fields)
                    for col in ['open', 'high', 'low', 'close', 'volume', 'pctChg']:
                        df[col] = pd.to_numeric(df[col], errors='coerce')
                    df = df.dropna()

                # 查询成功，跳出重试循环
                break

            except Exception as e:
                if attempt < max_retries - 1:
                    print(f"[涨停] ⚠ {code}第{attempt + 1}次查询失败({str(e)[:30]}), 重试中...")
                    import time
                    time.sleep(1)
                else:
                    print(f"[涨停] ✗ {code}查询{max_retries}次均失败: {str(e)[:50]}")
                    return features

    if df is None or len(df) < 20:
        return features

    # ===== 计算特征 =====
    try:
        # 涨停频率统计（ST股5%涨停，普通股10%）
        # 小盘股/中小板可能是10%涨停
        df['is_limit_up'] = (df['pctChg'] >= 9.5).astype(int)

        # 近60/20/5日涨停频率
        if len(df) >= 60:
            features['limit_up_freq_60d'] = df['is_limit_up'].iloc[-60:].mean()
        elif len(df) >= 20:
            features['limit_up_freq_60d'] = df['is_limit_up'].mean()
        if len(df) >= 20:
            features['limit_up_freq_20d'] = df['is_limit_up'].iloc[-20:].mean()
        if len(df) >= 5:
            features['limit_up_freq_5d'] = df['is_limit_up'].iloc[-5:].mean()

        # 连板天数（从最近一天往前数连续涨停天数）
        consecutive = 0
        for i in range(min(10, len(df))):
            if df['is_limit_up'].iloc[-(i + 1)] == 1:
                consecutive += 1
            else:
                break
        features['consecutive_boards'] = consecutive

        # 当日涨幅
        features['today_change_pct'] = float(df['pctChg'].iloc[-1])

        # 量比（今日成交量/5日均量）
        if len(df) >= 6:
            vol_ma5 = df['volume'].iloc[-6:-1].mean()  # 前5日均量（不含当日）
            features['volume_ratio'] = float(df['volume'].iloc[-1] / vol_ma5) if vol_ma5 > 0 else 1.0

        # RSI(14)计算
        if len(df) >= 15:
            delta = df['close'].diff()
            gain = delta.clip(lower=0)
            loss = (-delta).clip(lower=0)
            avg_gain = gain.rolling(window=14, min_periods=14).mean()
            avg_loss = loss.rolling(window=14, min_periods=14).mean()
            rs_val = avg_gain / (avg_loss + 1e-10)
            rsi_series = 100 - (100 / (1 + rs_val))
            features['rsi_14'] = float(rsi_series.iloc[-1]) if not np.isnan(rsi_series.iloc[-1]) else 50.0

        # MACD金叉判断
        if len(df) >= 27:
            ema12 = df['close'].ewm(span=12, adjust=False).mean()
            ema26 = df['close'].ewm(span=26, adjust=False).mean()
            dif = ema12 - ema26
            dea = dif.ewm(span=9, adjust=False).mean()
            macd_hist = (dif - dea) * 2
            # 前一日MACD<0，今日MACD>0 为金叉
            if len(macd_hist) >= 2 and not np.isnan(macd_hist.iloc[-1]) and not np.isnan(macd_hist.iloc[-2]):
                features['macd_signal'] = 1 if macd_hist.iloc[-1] > 0 and macd_hist.iloc[-2] <= 0 else 0

        # 近20日高点距离（越接近高点，突破涨停概率越大）
        if len(df) >= 20:
            recent_high = df['high'].iloc[-20:].max()
            current_price = df['close'].iloc[-1]
            features['recent_high_pct'] = float((recent_high - current_price) / (current_price + 1e-10))

    except Exception as e:
        print(f"[涨停] ⚠ {code}特征计算异常: {str(e)[:50]}")

    return features


def _calculate_limit_up_probability(features, industry_heat=0.5):
    """
    基于特征计算涨停概率【】
    设计意图：多因子融合计算涨停概率
    输入：features - 特征字典
          industry_heat - 板块热度 (0-1)
    输出：涨停概率 (0-1)
    """
    # 历史涨停频率因子（近因加权）
    hist_factor = (
            features['limit_up_freq_5d'] * 0.5 +
            features['limit_up_freq_20d'] * 0.3 +
            features['limit_up_freq_60d'] * 0.2
    )

    # 连板加成（1板→2板概率略低，2板→3板概率更低）
    board_bonus = 0.0
    if features['consecutive_boards'] == 1:
        board_bonus = 0.05  # 首板后次日涨停概率略增
    elif features['consecutive_boards'] == 2:
        board_bonus = 0.02  # 二板后三板概率较低
    elif features['consecutive_boards'] >= 3:
        board_bonus = 0.01  # 妖股概率极低

    # 当日涨幅因子（越接近10%越可能封板）
    change_pct = features['today_change_pct']
    change_factor = 0.0
    if 7 <= change_pct < 10:
        change_factor = (change_pct - 7) / 3 * 0.15  # 7%-10%区间
    elif change_pct >= 10:
        change_factor = 0.15  # 已涨停

    # 量价配合（温和放量最佳，缩量或爆量都不好）
    vol_ratio = features['volume_ratio']
    vol_factor = 0.0
    if 0.8 <= vol_ratio <= 2.0:
        vol_factor = 0.03
    elif 2.0 < vol_ratio <= 3.0:
        vol_factor = 0.01  # 爆量，筹码可能不稳

    # RSI超买区（70以上为超买，可能继续涨也可能回调）
    rsi = features['rsi_14']
    rsi_factor = 0.0
    if 70 <= rsi < 80:
        rsi_factor = 0.02
    elif rsi >= 80:
        rsi_factor = 0.01

    # MACD金叉加成
    macd_factor = 0.02 if features['macd_signal'] == 1 else 0.0

    # 近20日高点距离（已创新高继续涨的概率）
    high_pct = features['recent_high_pct']
    high_factor = 0.0
    if high_pct < 0.01:  # 距高点<1%
        high_factor = 0.03
    elif high_pct < 0.03:  # 距高点<3%
        high_factor = 0.02

    # 板块热度加成
    sector_factor = industry_heat * 0.05

    # 综合计算
    base_prob = 0.02  # 基准概率约2%（A股平均涨停概率约8%，保守估计）
    total_prob = base_prob + hist_factor * 0.3 + board_bonus + change_factor + vol_factor + rsi_factor + macd_factor + high_factor + sector_factor

    # 概率校准：限制在0.5%-15%范围内（避免极端值）
    calibrated_prob = np.clip(total_prob, 0.005, 0.15)

    return calibrated_prob


def run_limit_up_report(optimal_portfolio, all_predictions):
    """
    涨停吃板概率预测入口
    修复：独立管理baostock连接，不依赖分钟级报告模块的连接状态
    输入：optimal_portfolio - 推荐持仓
          all_predictions - 所有预测数据
    输出：涨停概率排名报告 + PDF可视化
    """
    print("\n" + "━" * 70)
    print("    涨停吃板概率预测")
    print("━" * 70)

    if not optimal_portfolio or len(optimal_portfolio) == 0:
        print("[涨停] ⚠ 无推荐持仓，跳过涨停预测")
        return {}

    results = []

    # 获取板块热度（简化：半导体和人工智能热度较高）
    sector_heat = {
        '半导体': 0.7,
        '人工智能': 0.65,
        '科技': 0.5,
        '能源': 0.3,
        '金属': 0.35,
        '金融': 0.3,
        '消费': 0.35,
        '医药': 0.4,
        '制造': 0.35,
        '地产': 0.25,
        '交通': 0.3,
        '化工': 0.35,
    }

    for stock in optimal_portfolio:
        code = stock['code']
        name = stock['name']

        # 获取股票所属行业
        industry = '科技'
        for ind, stocks in STOCK_POOL.items():
            for s in stocks:
                if s['code'] == code:
                    industry = ind
                    break

        # 获取技术面特征
        features = _get_limit_up_features(code, days=60)

        # 获取板块热度
        heat = sector_heat.get(industry, 0.5)

        # 计算涨停概率
        prob = _calculate_limit_up_probability(features, industry_heat=heat)

        # 判断连板状态
        consecutive = features['consecutive_boards']
        board_status = "首板" if consecutive == 0 else f"{consecutive}连板"

        # 关键信号
        signals = []
        if features['limit_up_freq_20d'] > 0.05:
            signals.append("近期涨停过")
        if features['macd_signal'] == 1:
            signals.append("MACD金叉")
        if features['rsi_14'] > 70:
            signals.append("RSI超买")
        if features['volume_ratio'] > 1.5:
            signals.append("量能放大")
        if features['today_change_pct'] > 5:
            signals.append(f"今日涨幅{features['today_change_pct']:.1f}%")

        result = {
            'name': name,
            'code': code,
            'industry': industry,
            'limit_up_prob': prob,
            'board_status': board_status,
            'signals': signals,
            'features': features,
        }
        results.append(result)

    # 按涨停概率排序
    results.sort(key=lambda x: x['limit_up_prob'], reverse=True)

    # 打印控制台报告
    print("\n  涨停吃板概率预测")
    print("  " + "─" * 86)
    print(f"{'序号':^4} │ {'股票名称':^10} │ {'代码':^12} │ {'涨停概率':^10} │ {'连板状态':^8} │ {'关键信号'}")
    print("  " + "─" * 86)

    for i, r in enumerate(results, 1):
        signals_str = ", ".join(r['signals']) if r['signals'] else "无明显信号"
        prob_pct = f"{r['limit_up_prob'] * 100:.2f}%"
        print(f"  {i:^4} │ {r['name']:^10} │ {r['code']:^12} │ {prob_pct:^10} │ {r['board_status']:^8} │ {signals_str}")

    print("  " + "─" * 86)
    print("  说明：涨停概率为基于历史统计和技术指标的次日涨停预估，仅供参考")
    print("  提示：一般市场下单票涨停概率约0.5%-15%，超过15%需谨慎")

    # 生成PDF可视化
    if DEPENDENCY_STATUS['matplotlib']:
        try:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
            from matplotlib.backends.backend_pdf import PdfPages

            plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans', 'Arial Unicode MS', 'sans-serif']
            plt.rcParams['axes.unicode_minus'] = False

            timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
            pdf_path = os.path.join(OUTPUT_PATH, f'涨停预测报告_{timestamp}.pdf')

            with PdfPages(pdf_path) as pdf:
                # 第1页：标题+排名
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))

                # 左侧：横向柱状图
                names = [r['name'] for r in results]
                probs = [r['limit_up_prob'] * 100 for r in results]
                colors = ['#FF6B6B' if p > 10 else '#FFD93D' if p > 5 else '#6BCB77' for p in probs]

                y_pos = range(len(names))
                ax1.barh(y_pos, probs, color=colors, alpha=0.8)
                ax1.set_yticks(y_pos)
                ax1.set_yticklabels(names)
                ax1.set_xlabel('涨停概率 (%)')
                ax1.set_title('次日涨停概率排名', fontsize=14, fontweight='bold')
                ax1.set_xlim(0, 20)

                # 添加数值标签
                for i, (v, n) in enumerate(zip(probs, names)):
                    ax1.text(v + 0.3, i, f'{v:.2f}%', va='center', fontsize=9)

                ax1.axvline(x=10, color='red', linestyle='--', alpha=0.5, label='高概率警戒线')
                ax1.axvline(x=5, color='orange', linestyle='--', alpha=0.5, label='中概率参考线')
                ax1.legend(loc='lower right', fontsize=8)
                ax1.grid(axis='x', alpha=0.3)

                # 右侧：关键因子条形图（每只股票前3个信号）
                ax2.axis('off')
                ax2.set_title('关键因子信号', fontsize=14, fontweight='bold')

                table_data = []
                for r in results:
                    sigs = r['signals'][:3] if len(r['signals']) > 3 else r['signals']
                    sigs_str = "\n".join(sigs) if sigs else "无"
                    table_data.append([r['name'], sigs_str, f"{r['limit_up_prob'] * 100:.2f}%"])

                table = ax2.table(
                    cellText=table_data,
                    colLabels=['股票', '关键信号', '概率'],
                    cellLoc='center',
                    loc='center',
                    colWidths=[0.3, 0.5, 0.2]
                )
                table.auto_set_font_size(False)
                table.set_fontsize(9)
                table.scale(1.2, 1.5)

                plt.tight_layout()

                # 添加风险提示
                fig.text(0.5, 0.02,
                         '【风险提示】涨停预测基于历史统计和技术分析，次日实际涨停受多重因素影响，本报告仅供参考',
                         ha='center', fontsize=9, style='italic', color='gray')

                pdf.savefig()
                plt.close()

                print(f"[涨停] ✓ PDF可视化报告已保存: {pdf_path}")

        except Exception as e:
            print(f"[涨停] ⚠ PDF生成失败: {str(e)[:100]}")

    # 涨停模块使用完毕后关闭baostock连接
    if DEPENDENCY_STATUS['baostock']:
        try:
            bs.logout()
        except Exception:
            pass

    # 返回结果字典
    return {
        'ranked_results': results,
        'timestamp': datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
    }


# ==========================================
# 【】重仓指数模块（HPI - Heavy Position Index）
# 设计意图：在推荐票中判断谁更适合重仓——高确定性、高胜率、模型共识强、尾部风险低
# 核心公式：HPI = 0.35×胜率得分 + 0.25×夏普得分 + 0.25×共识得分 + 0.15×下行安全得分
# ==========================================

def _calculate_model_consensus(pred_info, horizon=3):
    """
    计算三模型共识得分【】
    设计意图：ML/卡尔曼/GARCH三个模型方向是否一致，一致则预测更可信
    输入：pred_info - 单只股票的预测结果字典
          horizon - 预测天数
    输出：consensus_score (0-1), consensus_label (str)
    """
    ml_pred = pred_info.get('ml_predictions', {}).get(horizon, 0)
    kalman_pred = pred_info.get('kalman_predictions', {}).get(horizon, 0)

    # GARCH方向：条件波动率趋势（从volatility_predictions推算）
    garch_vols = pred_info.get('volatility', {})
    vol_preds = garch_vols.get('volatility_predictions', {})
    vol_h1 = vol_preds.get('h_1d', 0.25)
    vol_h5 = vol_preds.get('h_5d', 0.25)
    # 波动率下降+收益为正 → 看涨；波动率上升+收益为负 → 看跌
    garch_bullish = 1 if (vol_h5 <= vol_h1 and kalman_pred > 0) or (vol_h5 < vol_h1 * 0.95) else -1

    # 各模型方向：正收益=1, 负收益=-1
    ml_dir = 1 if ml_pred > 0 else -1
    kalman_dir = 1 if kalman_pred > 0 else -1

    # 加权投票：ML权重0.4（直接预测收益），卡尔曼0.35（自适应滤波），GARCH 0.25（间接信号）
    weighted_vote = ml_dir * 0.4 + kalman_dir * 0.35 + garch_bullish * 0.25

    # 归一化到[0,1]：加权投票范围[-1,1] → [0,1]
    consensus_score = (weighted_vote + 1) / 2

    if consensus_score >= 0.8:
        label = "强共识看涨"
    elif consensus_score >= 0.6:
        label = "弱共识看涨"
    elif consensus_score >= 0.4:
        label = "分歧"
    elif consensus_score >= 0.2:
        label = "弱共识看跌"
    else:
        label = "强共识看跌"

    return consensus_score, label


def _calculate_cvar(fused_mean, fused_std, confidence=0.95, df=5):
    """
    计算条件尾部期望损失（CVaR）【】
    设计意图：衡量"如果跌了，平均跌多少"——比VaR更保守的风险度量
    输入：fused_mean - 融合期望收益（小数）
          fused_std - 融合标准差（小数）
          confidence - 置信水平
          df - t分布自由度
    输出：cvar（小数，负值表示亏损）
    """
    from scipy import stats as sp_stats

    # VaR分位点
    alpha = 1 - confidence  # 0.05
    t_quantile = sp_stats.t.ppf(alpha, df=df)
    var = fused_mean + t_quantile * fused_std  # 这就是ci_lower

    # CVaR = E[X | X < VaR]，t分布的CVaR解析公式
    # CVaR = -fused_mean + fused_std * (f(t_quantile) / alpha) * ((df + t_quantile^2) / (df - 1))
    # 其中f是t分布PDF
    if df <= 1:
        return var  # 自由度太小，退化

    t_pdf = sp_stats.t.pdf(t_quantile, df=df)
    cvar = fused_mean - fused_std * (t_pdf / alpha) * ((df + t_quantile ** 2) / (df - 1))

    return cvar


def _calculate_win_probability(fused_mean, fused_std, df=5):
    """
    计算上涨概率（t分布）【】
    输入：fused_mean - 融合期望收益（小数）
          fused_std - 融合标准差（小数）
          df - t分布自由度
    输出：上涨概率 (0-1)
    """
    from scipy import stats as sp_stats

    if fused_std <= 0:
        return 0.5

    t_stat = fused_mean / fused_std
    win_prob = float(sp_stats.t.cdf(t_stat, df=df))

    # 硬上限90%
    return min(win_prob, 0.90)


def _calculate_hpi_for_stock(stock, pred_info, horizon=3):
    """
    计算单只股票的重仓指数【】
    输入：stock - 组合中的股票字典
          pred_info - 该股票的预测结果字典
          horizon - 预测天数
    输出：HPI详情字典
    """
    fused = pred_info.get('fused', {}).get(horizon, {})
    if not fused:
        return None

    fused_mean = fused.get('fused_mean', 0)
    fused_var = fused.get('fused_var', 0.01)
    fused_std = np.sqrt(fused_var)

    # 1. 胜率得分：上涨概率
    win_prob = _calculate_win_probability(fused_mean, fused_std, df=5)

    # 2. 夏普得分：风险调整后收益
    sharpe = fused_mean / (fused_std + 1e-10)

    # 3. 共识得分
    consensus_score, consensus_label = _calculate_model_consensus(pred_info, horizon)

    # 4. 下行安全得分：CVaR的负值越大越危险，归一化后越低越差
    cvar = _calculate_cvar(fused_mean, fused_std, confidence=0.95, df=5)
    # CVaR范围大约在[-0.15, 0]，越接近0越安全
    # 归一化：cvar=-0.15→0分，cvar=0→1分
    downside_safety = np.clip((cvar + 0.15) / 0.15, 0, 1)

    # 5. 各维度的原始值（用于展示）
    raw_metrics = {
        'win_prob': win_prob,
        'sharpe': sharpe,
        'consensus_score': consensus_score,
        'consensus_label': consensus_label,
        'cvar': cvar,
        'downside_safety': downside_safety,
        'fused_mean': fused_mean,
        'fused_std': fused_std,
    }

    return raw_metrics


def run_heavy_position_report(optimal_portfolio, all_predictions, horizon=3):
    """
    重仓指数报告入口【】
    对8只推荐票做横截面排名，输出谁更适合重仓
    输入：optimal_portfolio - 推荐持仓
          all_predictions - 所有预测数据
          horizon - 预测天数
    输出：HPI排名报告 + 控制台输出
    """
    print("\n" + "━" * 70)
    print("    重仓指数分析")
    print("━" * 70)

    if not optimal_portfolio or len(optimal_portfolio) == 0:
        print("[重仓] ⚠ 无推荐持仓，跳过")
        return {}

    # ========== 1. 计算每只票的HPI原始指标 ==========
    raw_results = []
    for stock in optimal_portfolio:
        code = stock['code']
        pred_info = all_predictions.get(code, {})

        metrics = _calculate_hpi_for_stock(stock, pred_info, horizon)
        if metrics is None:
            continue

        metrics['name'] = stock['name']
        metrics['code'] = code
        metrics['industry'] = stock['industry']
        raw_results.append(metrics)

    if len(raw_results) == 0:
        print("[重仓] ⚠ 无法计算HPI，跳过")
        return {}

    # ========== 2. 横截面min-max归一化 ==========
    def min_max_normalize(values):
        """将值归一化到[0,1]"""
        v_min = min(values)
        v_max = max(values)
        if v_max - v_min < 1e-10:
            return [0.5] * len(values)
        return [(v - v_min) / (v_max - v_min) for v in values]

    win_probs = [r['win_prob'] for r in raw_results]
    sharpes = [r['sharpe'] for r in raw_results]
    consensuses = [r['consensus_score'] for r in raw_results]
    safeties = [r['downside_safety'] for r in raw_results]

    win_norm = min_max_normalize(win_probs)
    sharpe_norm = min_max_normalize(sharpes)
    cons_norm = min_max_normalize(consensuses)
    safe_norm = min_max_normalize(safeties)

    # ========== 3. 加权计算HPI ==========
    for i, r in enumerate(raw_results):
        r['hpi_raw'] = (
                0.35 * win_norm[i] +
                0.25 * sharpe_norm[i] +
                0.25 * cons_norm[i] +
                0.15 * safe_norm[i]
        )
        r['hpi_rank'] = 0  # 稍后填充

    # 按HPI排序
    raw_results.sort(key=lambda x: x['hpi_raw'], reverse=True)

    # 填充排名
    for i, r in enumerate(raw_results):
        r['hpi_rank'] = i + 1

    # ========== 4. 重仓评级 ==========
    for r in raw_results:
        hpi = r['hpi_raw']
        if hpi >= 0.65:
            r['rating'] = '★★★'
            r['rating_desc'] = '高确定性'
            r['position_advice'] = '可重仓（尽量给满）'
        elif hpi >= 0.40:
            r['rating'] = '★★'
            r['rating_desc'] = '中等确定性'
            r['position_advice'] = '标准仓位'
        else:
            r['rating'] = '★'
            r['rating_desc'] = '偏投机'
            r['position_advice'] = '轻仓试探'

    # ========== 5. 控制台输出 ==========
    print("\n  ─" + "─" * 34)
    print("  重仓指数排名")
    print("  ─" + "─" * 34)
    print("  " + "─" * 106)
    print(f"{'排名':^4} │ {'评级':^6} | {'股票':^10} | {'代码':^12} | {'HPI':^6} | "
          f"{'胜率':^8} | {'夏普':^8} | {'共识':^12} | {'尾部风险':^10} | {'仓位建议'}")
    print("  " + "─" * 106)

    for r in raw_results:
        win_str = f"{r['win_prob'] * 100:.1f}%"
        sharpe_str = f"{r['sharpe']:.2f}"
        cons_str = r['consensus_label']
        cvar_str = f"{r['cvar'] * 100:.2f}%"
        hpi_str = f"{r['hpi_raw']:.3f}"

        print(f"  {r['hpi_rank']:^4} │ {r['rating']:^6} | {r['name']:^10} | {r['code']:^12} | {hpi_str:^6} | "
              f"{win_str:^8} | {sharpe_str:^8} | {cons_str:^12} | {cvar_str:^10} | {r['position_advice']}")

    print("  " + "─" * 106)

    # 说明
    print("\n  HPI说明")
    print("  HPI = 0.35×胜率 + 0.25×夏普 + 0.25×共识 + 0.15×下行安全")
    print("  ★★★ ≥ 0.65：高确定性，适合重仓  |  ★★ ≥ 0.40：中等确定性  |  ★ < 0.40：轻仓")
    print("  胜率：t分布(5)上涨概率  |  共识：ML/卡尔曼/GARCH方向一致性")
    print("  尾部风险：CVaR(95%)")

    # ========== 6. 重仓推荐摘要 ==========
    heavy_picks = [r for r in raw_results if r['rating'] == '★★★']
    if heavy_picks:
        print("\n  重仓推荐")
        for r in heavy_picks:
            print(f"  {r['rating']} {r['name']}({r['code']}) — "
                  f"胜率{r['win_prob'] * 100:.0f}% | 夏普{r['sharpe']:.2f} | "
                  f"{r['consensus_label']} | CVaR={r['cvar'] * 100:.2f}%")
    else:
        print("\n  当前无高确定性标的，建议标准仓位分散配置")

    return {
        'ranked_results': raw_results,
        'horizon': horizon,
        'timestamp': datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
    }


if __name__ == "__main__":
    try:
        optimal_portfolio, all_predictions = main()

        if optimal_portfolio and len(optimal_portfolio) > 0:
            # 涨停吃板概率预测
            limit_up_results = run_limit_up_report(optimal_portfolio, all_predictions)

            # 重仓指数分析
            hpi_results = run_heavy_position_report(optimal_portfolio, all_predictions)
        else:
            print("  ⚠ 无推荐组合，跳过涨停和重仓报告")

    except Exception as e:
        MOOTDX_CLIENT.close()
        error_handler.capture_exception(e, "主程序执行")
        print("\n  异常退出，请查看错误日志")
        sys.exit(1)

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
stock_data_manager.py — 五域归元·财略 数据获取与存储独立模块
============================================================
版本: v1.0
创建: 2026-06-18
设计意图:
  1. 从stocks3.py中剥离数据获取与存储，独立运行
  2. 分工: baostock拉历史全量, mootdx做增量更新
  3. 每只股票独立CSV, 支持断点续传+增量追加
  4. 大幅扩充股票池, 重点覆盖科技/半导体/小金属/石油/黄金

存储结构:
  stockdata/
  ├── metadata.json          # 每只股票最后更新日期+数据行数
  ├── stock_pool.json        # 当前使用的股票池快照
  ├── daily/                 # 日K线CSV(每只股票一个文件)
  │   ├── 601857.SH.csv
  │   ├── 000858.SZ.csv
  │   └── ...
  └── index/                 # 指数数据
      ├── 000001.SH.csv
      └── ...

CSV字段: date,open,high,low,close,volume,amount,turn(code_std,name列仅首行标记)

用法:
  # 首次全量下载(用baostock)
  python stock_data_manager.py --mode full

  # 每日增量更新(用mootdx, 失败降级baostock)
  python stock_data_manager.py --mode update

  # 仅更新指定板块
  python stock_data_manager.py --mode update --sectors 半导体,小金属

  # 作为模块导入
  from stock_data_manager import StockDataManager, STOCK_POOL, INDEX_LIST
  mgr = StockDataManager()
  mgr.full_download()
  mgr.incremental_update()
"""

import os
import sys
import json
import time
import random
import datetime
import argparse
import traceback
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from contextlib import contextmanager

import pandas as pd
import numpy as np

# ============================================================
# 依赖检测
# ============================================================
DEPS = {'baostock': False, 'mootdx': False}

try:
    import baostock as bs
    DEPS['baostock'] = True
except ImportError:
    pass

try:
    from mootdx.quotes import Quotes
    DEPS['mootdx'] = True
except ImportError:
    pass


# ============================================================
# 路径配置
# ============================================================
# 本地Windows路径
LOCAL_DATA_DIR = r'C:\Users\HUAWEI\Desktop\Adversarial Learning\stockdata'
# 云端/非Windows降级到当前目录
if os.name != 'nt':
    LOCAL_DATA_DIR = os.path.join(os.getcwd(), 'stockdata')

DAILY_DIR = os.path.join(LOCAL_DATA_DIR, 'daily')
INDEX_DIR = os.path.join(LOCAL_DATA_DIR, 'index')
METADATA_FILE = os.path.join(LOCAL_DATA_DIR, 'metadata.json')
POOL_SNAPSHOT_FILE = os.path.join(LOCAL_DATA_DIR, 'stock_pool.json')


# ============================================================
# 大幅扩充股票池 — 重点: 科技/半导体/小金属/石油/黄金
# ============================================================
STOCK_POOL = {
    # ======================== 能源·石油天然气 ========================
    '石油天然气': [
        {'code': 'sh.601857', 'name': '中国石油', 'code_std': '601857.SH'},
        {'code': 'sh.600028', 'name': '中国石化', 'code_std': '600028.SH'},
        {'code': 'sh.600938', 'name': '中国海油', 'code_std': '600938.SH'},
        {'code': 'sh.601088', 'name': '中国神华', 'code_std': '601088.SH'},
        {'code': 'sh.600989', 'name': '宝丰能源', 'code_std': '600989.SH'},
        {'code': 'sh.600803', 'name': '新奥股份', 'code_std': '600803.SH'},
        {'code': 'sh.601225', 'name': '陕西煤业', 'code_std': '601225.SH'},
        {'code': 'sh.600188', 'name': '兖矿能源', 'code_std': '600188.SH'},
        {'code': 'sh.600387', 'name': '海越能源', 'code_std': '600387.SH'},
        {'code': 'sz.000096', 'name': '广聚能源', 'code_std': '000096.SZ'},
        {'code': 'sh.600688', 'name': '上海石化', 'code_std': '600688.SH'},
        {'code': 'sz.000554', 'name': '泰山石油', 'code_std': '000554.SZ'},
        {'code': 'sh.601918', 'name': '新集能源', 'code_std': '601918.SH'},
        {'code': 'sz.000937', 'name': '冀中能源', 'code_std': '000937.SZ'},
    ],

    # ======================== 黄金 ========================
    '黄金': [
        {'code': 'sh.601899', 'name': '紫金矿业', 'code_std': '601899.SH'},
        {'code': 'sh.600547', 'name': '山东黄金', 'code_std': '600547.SH'},
        {'code': 'sh.600489', 'name': '中金黄金', 'code_std': '600489.SH'},
        {'code': 'sz.000975', 'name': '银泰黄金', 'code_std': '000975.SZ'},
        {'code': 'sh.600988', 'name': '赤峰黄金', 'code_std': '600988.SH'},
        {'code': 'sz.002155', 'name': '湖南黄金', 'code_std': '002155.SZ'},
        {'code': 'sh.601069', 'name': '西部黄金', 'code_std': '601069.SH'},
        {'code': 'sz.000506', 'name': '招金黄金', 'code_std': '000506.SZ'},
        {'code': 'sz.002237', 'name': '恒邦股份', 'code_std': '002237.SZ'},
        {'code': 'sz.000603', 'name': '盛达资源', 'code_std': '000603.SZ'},
        {'code': 'sh.600531', 'name': '豫光金铅', 'code_std': '600531.SH'},
        {'code': 'sh.601212', 'name': '白银有色', 'code_std': '601212.SH'},
    ],

    # ======================== 小金属·稀有金属 ========================
    '小金属': [
        # 稀土
        {'code': 'sh.600111', 'name': '北方稀土', 'code_std': '600111.SH'},
        {'code': 'sz.000831', 'name': '中国稀土', 'code_std': '000831.SZ'},
        {'code': 'sh.600392', 'name': '盛和资源', 'code_std': '600392.SH'},
        # 钨
        {'code': 'sz.000657', 'name': '中钨高新', 'code_std': '000657.SZ'},
        {'code': 'sh.600549', 'name': '厦门钨业', 'code_std': '600549.SH'},
        {'code': 'sz.002378', 'name': '章源钨业', 'code_std': '002378.SZ'},
        # 锑
        {'code': 'sh.601020', 'name': '华钰矿业', 'code_std': '601020.SH'},
        # 锗/镓
        {'code': 'sz.002428', 'name': '云南锗业', 'code_std': '002428.SZ'},
        # 锡
        {'code': 'sz.000960', 'name': '锡业股份', 'code_std': '000960.SZ'},
        # 钼
        {'code': 'sh.601958', 'name': '金钼股份', 'code_std': '601958.SH'},
        {'code': 'sh.603993', 'name': '洛阳钼业', 'code_std': '603993.SH'},
        # 钽/铌
        {'code': 'sz.000962', 'name': '东方钽业', 'code_std': '000962.SZ'},
        # 铟/铋
        {'code': 'sh.600301', 'name': '华锡有色', 'code_std': '600301.SH'},
        {'code': 'sh.600961', 'name': '株冶集团', 'code_std': '600961.SH'},
        # 锌/铅
        {'code': 'sh.600497', 'name': '驰宏锌锗', 'code_std': '600497.SH'},
        # 铜
        {'code': 'sh.600362', 'name': '江西铜业', 'code_std': '600362.SH'},
        {'code': 'sz.000630', 'name': '铜陵有色', 'code_std': '000630.SZ'},
        # 铝
        {'code': 'sh.601600', 'name': '中国铝业', 'code_std': '601600.SH'},
        {'code': 'sz.000807', 'name': '云铝股份', 'code_std': '000807.SZ'},
        # 钴/锂
        {'code': 'sz.000792', 'name': '盐湖股份', 'code_std': '000792.SZ'},
        {'code': 'sh.600259', 'name': '广晟有色', 'code_std': '600259.SH'},
    ],

    # ======================== 半导体 ========================
    '半导体': [
        # 设备
        {'code': 'sz.002371', 'name': '北方华创', 'code_std': '002371.SZ'},
        {'code': 'sh.688012', 'name': '中微公司', 'code_std': '688012.SH'},
        {'code': 'sh.688072', 'name': '拓荆科技', 'code_std': '688072.SH'},
        {'code': 'sz.300604', 'name': '长川科技', 'code_std': '300604.SZ'},
        # 晶圆代工
        {'code': 'sh.688981', 'name': '中芯国际', 'code_std': '688981.SH'},
        # 芯片设计
        {'code': 'sh.603986', 'name': '兆易创新', 'code_std': '603986.SH'},
        {'code': 'sh.688256', 'name': '寒武纪', 'code_std': '688256.SH'},
        {'code': 'sh.603893', 'name': '瑞芯微', 'code_std': '603893.SH'},
        {'code': 'sz.300458', 'name': '全志科技', 'code_std': '300458.SZ'},
        {'code': 'sh.688536', 'name': '思瑞浦', 'code_std': '688536.SH'},
        {'code': 'sh.688052', 'name': '纳芯微', 'code_std': '688052.SH'},
        # 封装测试
        {'code': 'sh.600584', 'name': '长电科技', 'code_std': '600584.SH'},
        {'code': 'sz.002185', 'name': '华天科技', 'code_std': '002185.SZ'},
        {'code': 'sz.002156', 'name': '通富微电', 'code_std': '002156.SZ'},
        # 功率半导体/分立器件
        {'code': 'sh.600460', 'name': '士兰微', 'code_std': '600460.SH'},
        {'code': 'sh.603290', 'name': '斯达半导', 'code_std': '603290.SH'},
        {'code': 'sh.600745', 'name': '闻泰科技', 'code_std': '600745.SH'},
        {'code': 'sh.600360', 'name': '华微电子', 'code_std': '600360.SH'},
        # EDA/IP
        {'code': 'sh.688519', 'name': '华大九天', 'code_std': '688519.SH'},
        {'code': 'sh.688521', 'name': '芯原股份', 'code_std': '688521.SH'},
        # 存储
        {'code': 'sh.688525', 'name': '佰维存储', 'code_std': '688525.SH'},
        # 材料
        {'code': 'sh.688019', 'name': '安集科技', 'code_std': '688019.SH'},
        {'code': 'sh.688126', 'name': '沪硅产业', 'code_std': '688126.SH'},
        {'code': 'sh.688268', 'name': '华特气体', 'code_std': '688268.SH'},
        # 其他
        {'code': 'sh.603005', 'name': '晶方科技', 'code_std': '603005.SH'},
        {'code': 'sh.603501', 'name': '韦尔股份', 'code_std': '603501.SH'},
        {'code': 'sh.600198', 'name': '大唐电信', 'code_std': '600198.SH'},
        {'code': 'sh.600171', 'name': '上海贝岭', 'code_std': '600171.SH'},
        {'code': 'sh.600877', 'name': '电科芯片', 'code_std': '600877.SH'},
        {'code': 'sh.603160', 'name': '汇顶科技', 'code_std': '603160.SH'},
        {'code': 'sz.000021', 'name': '深科技', 'code_std': '000021.SZ'},
    ],

    # ======================== 人工智能 ========================
    '人工智能': [
        {'code': 'sz.002230', 'name': '科大讯飞', 'code_std': '002230.SZ'},
        {'code': 'sh.601360', 'name': '三六零', 'code_std': '601360.SH'},
        {'code': 'sh.603019', 'name': '中科曙光', 'code_std': '603019.SH'},
        {'code': 'sz.000977', 'name': '浪潮信息', 'code_std': '000977.SZ'},
        {'code': 'sh.601138', 'name': '工业富联', 'code_std': '601138.SH'},
        {'code': 'sz.000938', 'name': '紫光股份', 'code_std': '000938.SZ'},
        {'code': 'sz.002415', 'name': '海康威视', 'code_std': '002415.SZ'},
        {'code': 'sz.002236', 'name': '大华股份', 'code_std': '002236.SZ'},
        {'code': 'sh.600845', 'name': '宝信软件', 'code_std': '600845.SH'},
        {'code': 'sh.600588', 'name': '用友网络', 'code_std': '600588.SH'},
        {'code': 'sh.600446', 'name': '金证股份', 'code_std': '600446.SH'},
        {'code': 'sz.300033', 'name': '同花顺', 'code_std': '300033.SZ'},
        {'code': 'sh.600633', 'name': '浙数文化', 'code_std': '600633.SH'},
        {'code': 'sh.603000', 'name': '人民网', 'code_std': '603000.SH'},
        {'code': 'sz.000997', 'name': '新大陆', 'code_std': '000997.SZ'},
    ],

    # ======================== 科技 ========================
    '科技': [
        {'code': 'sh.600941', 'name': '中国移动', 'code_std': '600941.SH'},
        {'code': 'sh.601728', 'name': '中国电信', 'code_std': '601728.SH'},
        {'code': 'sz.002594', 'name': '比亚迪', 'code_std': '002594.SZ'},
        {'code': 'sz.000725', 'name': '京东方A', 'code_std': '000725.SZ'},
        {'code': 'sz.002475', 'name': '立讯精密', 'code_std': '002475.SZ'},
        {'code': 'sh.600498', 'name': '烽火通信', 'code_std': '600498.SH'},
        {'code': 'sz.002384', 'name': '东山精密', 'code_std': '002384.SZ'},
        {'code': 'sh.600271', 'name': '航天信息', 'code_std': '600271.SH'},
        {'code': 'sh.600100', 'name': '同方股份', 'code_std': '600100.SH'},
        {'code': 'sh.600776', 'name': '东方通信', 'code_std': '600776.SH'},
    ],

    # ======================== 金融 ========================
    '金融': [
        {'code': 'sh.601398', 'name': '工商银行', 'code_std': '601398.SH'},
        {'code': 'sh.601318', 'name': '中国平安', 'code_std': '601318.SH'},
        {'code': 'sh.600036', 'name': '招商银行', 'code_std': '600036.SH'},
        {'code': 'sh.601988', 'name': '中国银行', 'code_std': '601988.SH'},
        {'code': 'sh.601939', 'name': '建设银行', 'code_std': '601939.SH'},
        {'code': 'sh.600016', 'name': '民生银行', 'code_std': '600016.SH'},
        {'code': 'sz.000001', 'name': '平安银行', 'code_std': '000001.SZ'},
        {'code': 'sh.601628', 'name': '中国人寿', 'code_std': '601628.SH'},
        {'code': 'sh.601818', 'name': '光大银行', 'code_std': '601818.SH'},
        {'code': 'sz.002736', 'name': '国信证券', 'code_std': '002736.SZ'},
        {'code': 'sh.601689', 'name': '拓普集团', 'code_std': '601689.SH'},
    ],

    # ======================== 消费 ========================
    '消费': [
        {'code': 'sh.600519', 'name': '贵州茅台', 'code_std': '600519.SH'},
        {'code': 'sz.000858', 'name': '五粮液', 'code_std': '000858.SZ'},
        {'code': 'sz.000333', 'name': '美的集团', 'code_std': '000333.SZ'},
        {'code': 'sh.600887', 'name': '伊利股份', 'code_std': '600887.SH'},
        {'code': 'sz.002304', 'name': '洋河股份', 'code_std': '002304.SZ'},
        {'code': 'sh.600809', 'name': '山西汾酒', 'code_std': '600809.SH'},
        {'code': 'sz.000895', 'name': '双汇发展', 'code_std': '000895.SZ'},
        {'code': 'sz.002508', 'name': '老板电器', 'code_std': '002508.SZ'},
        {'code': 'sh.603868', 'name': '飞科电器', 'code_std': '603868.SH'},
        {'code': 'sz.002419', 'name': '天虹股份', 'code_std': '002419.SZ'},
        {'code': 'sz.002262', 'name': '恩华药业', 'code_std': '002262.SZ'},
    ],

    # ======================== 医药 ========================
    '医药': [
        {'code': 'sh.600276', 'name': '恒瑞医药', 'code_std': '600276.SH'},
        {'code': 'sz.000661', 'name': '长春高新', 'code_std': '000661.SZ'},
        {'code': 'sz.002252', 'name': '上海莱士', 'code_std': '002252.SZ'},
        {'code': 'sh.600867', 'name': '通化东宝', 'code_std': '600867.SH'},
        {'code': 'sz.002007', 'name': '华兰生物', 'code_std': '002007.SZ'},
        {'code': 'sh.600196', 'name': '复星医药', 'code_std': '600196.SH'},
        {'code': 'sh.600332', 'name': '白云山', 'code_std': '600332.SH'},
        {'code': 'sz.002422', 'name': '科伦药业', 'code_std': '002422.SZ'},
        {'code': 'sh.600521', 'name': '华海药业', 'code_std': '600521.SH'},
        {'code': 'sz.002603', 'name': '以岭药业', 'code_std': '002603.SZ'},
    ],

    # ======================== 制造 ========================
    '制造': [
        {'code': 'sh.601766', 'name': '中国中车', 'code_std': '601766.SH'},
        {'code': 'sh.600031', 'name': '三一重工', 'code_std': '600031.SH'},
        {'code': 'sz.000157', 'name': '中联重科', 'code_std': '000157.SZ'},
        {'code': 'sz.000425', 'name': '徐工机械', 'code_std': '000425.SZ'},
        {'code': 'sh.601100', 'name': '恒立液压', 'code_std': '601100.SH'},
        {'code': 'sz.002097', 'name': '山河智能', 'code_std': '002097.SZ'},
        {'code': 'sh.600320', 'name': '振华重工', 'code_std': '600320.SH'},
        {'code': 'sz.002531', 'name': '天顺风能', 'code_std': '002531.SZ'},
        {'code': 'sh.600495', 'name': '晋西车轴', 'code_std': '600495.SH'},
        {'code': 'sz.002204', 'name': '大连重工', 'code_std': '002204.SZ'},
        {'code': 'sh.600811', 'name': '东方集团', 'code_std': '600811.SH'},
    ],

    # ======================== 地产 ========================
    '地产': [
        {'code': 'sh.600048', 'name': '保利发展', 'code_std': '600048.SH'},
        {'code': 'sz.000002', 'name': '万科A', 'code_std': '000002.SZ'},
        {'code': 'sh.600383', 'name': '金地集团', 'code_std': '600383.SH'},
        {'code': 'sh.601155', 'name': '新城控股', 'code_std': '601155.SH'},
        {'code': 'sz.000069', 'name': '华侨城A', 'code_std': '000069.SZ'},
        {'code': 'sh.600606', 'name': '绿地控股', 'code_std': '600606.SH'},
        {'code': 'sh.600325', 'name': '华发股份', 'code_std': '600325.SH'},
        {'code': 'sh.600376', 'name': '首开股份', 'code_std': '600376.SH'},
        {'code': 'sh.600743', 'name': '华远地产', 'code_std': '600743.SH'},
        {'code': 'sz.000897', 'name': '津滨发展', 'code_std': '000897.SZ'},
        {'code': 'sz.000961', 'name': '中南建设', 'code_std': '000961.SZ'},
    ],

    # ======================== 交通 ========================
    '交通': [
        {'code': 'sh.601111', 'name': '中国国航', 'code_std': '601111.SH'},
        {'code': 'sh.600029', 'name': '南方航空', 'code_std': '600029.SH'},
        {'code': 'sh.601006', 'name': '大秦铁路', 'code_std': '601006.SH'},
        {'code': 'sh.600115', 'name': '东方航空', 'code_std': '600115.SH'},
        {'code': 'sh.601333', 'name': '广深铁路', 'code_std': '601333.SH'},
        {'code': 'sz.000089', 'name': '深圳机场', 'code_std': '000089.SZ'},
        {'code': 'sh.600018', 'name': '上港集团', 'code_std': '600018.SH'},
        {'code': 'sh.601866', 'name': '中远海发', 'code_std': '601866.SH'},
        {'code': 'sh.600798', 'name': '宁波海运', 'code_std': '600798.SH'},
        {'code': 'sz.000905', 'name': '厦门港务', 'code_std': '000905.SZ'},
    ],

    # ======================== 化工 ========================
    '化工': [
        {'code': 'sh.600309', 'name': '万华化学', 'code_std': '600309.SH'},
        {'code': 'sz.002493', 'name': '荣盛石化', 'code_std': '002493.SZ'},
        {'code': 'sh.601233', 'name': '桐昆股份', 'code_std': '601233.SH'},
        {'code': 'sh.600143', 'name': '金发科技', 'code_std': '600143.SH'},
        {'code': 'sh.600409', 'name': '三友化工', 'code_std': '600409.SH'},
        {'code': 'sz.002648', 'name': '卫星化学', 'code_std': '002648.SZ'},
        {'code': 'sz.002092', 'name': '中泰化学', 'code_std': '002092.SZ'},
        {'code': 'sh.600810', 'name': '神马股份', 'code_std': '600810.SH'},
        {'code': 'sh.600299', 'name': '安迪苏', 'code_std': '600299.SH'},
        {'code': 'sz.000698', 'name': '沈阳化工', 'code_std': '000698.SZ'},
        {'code': 'sz.000707', 'name': '双环科技', 'code_std': '000707.SZ'},
    ],
}

# 指数池
INDEX_LIST = [
    {'code': 'sh.000001', 'name': '上证指数', 'code_std': '000001.SH'},
    {'code': 'sz.399001', 'name': '深证成指', 'code_std': '399001.SZ'},
    {'code': 'sh.000300', 'name': '沪深300', 'code_std': '000300.SH'},
    {'code': 'sz.399006', 'name': '创业板指', 'code_std': '399006.SZ'},
    {'code': 'sh.000016', 'name': '上证50', 'code_std': '000016.SH'},
    {'code': 'sz.399673', 'name': '创业板50', 'code_std': '399673.SZ'},
]

# 统计
def _count_stocks():
    total = sum(len(v) for v in STOCK_POOL.values())
    sectors = len(STOCK_POOL)
    return total, sectors

TOTAL_STOCKS, TOTAL_SECTORS = _count_stocks()


# ============================================================
# 代码格式转换工具函数
# ============================================================
def code_to_baostock(code_std: str) -> str:
    """code_std (601857.SH) → baostock (sh.601857)"""
    if '.' in code_std:
        num, mkt = code_std.split('.')
        return f"{'sh' if mkt == 'SH' else 'sz'}.{num}"
    if code_std.startswith(('6', '9')):
        return f'sh.{code_std}'
    return f'sz.{code_std}'

def code_to_mootdx(code_std: str) -> str:
    """code_std (601857.SH) → mootdx (601857)"""
    return code_std.split('.')[0] if '.' in code_std else code_std

def mootdx_to_code_std(mootdx_code: str) -> str:
    """mootdx (601857) → code_std (601857.SH)"""
    if mootdx_code.startswith(('6', '9')):
        return f'{mootdx_code}.SH'
    return f'{mootdx_code}.SZ'


# ============================================================
# StockDataManager — 核心类
# ============================================================
class StockDataManager:
    """
    数据获取与存储管理器
    策略: baostock拉历史全量, mootdx做增量日更
    """

    # baostock请求间隔与重试
    BS_REQUEST_INTERVAL = 0.5   # 秒
    BS_MAX_RETRIES = 3
    BS_RETRY_SLEEP = 2.0

    # mootdx参数
    MOOTDX_MAX_BARS = 800       # 单次最多取800根K线
    MOOTDX_TIMEOUT = 20         # 连接超时

    # 默认历史起始日
    DEFAULT_START_DATE = '2020-01-01'

    def __init__(self, data_dir: str = None):
        self.data_dir = data_dir or LOCAL_DATA_DIR
        self.daily_dir = os.path.join(self.data_dir, 'daily')
        self.index_dir = os.path.join(self.data_dir, 'index')
        self.metadata_file = os.path.join(self.data_dir, 'metadata.json')
        self.pool_file = os.path.join(self.data_dir, 'stock_pool.json')

        # mootdx客户端
        self._mootdx_client = None
        self._mootdx_connected = False

        # baostock登录状态
        self._bs_logged_in = False

        # 元数据
        self.metadata = self._load_metadata()

        # 统计
        self._stats = {'downloaded': 0, 'updated': 0, 'failed': 0, 'skipped': 0}

    # -------------------- 目录与元数据 --------------------

    def _ensure_dirs(self):
        """确保目录存在"""
        os.makedirs(self.daily_dir, exist_ok=True)
        os.makedirs(self.index_dir, exist_ok=True)

    def _load_metadata(self) -> dict:
        """加载元数据"""
        if os.path.exists(self.metadata_file):
            try:
                with open(self.metadata_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception:
                return {'stocks': {}, 'indices': {}, 'last_full_download': None}
        return {'stocks': {}, 'indices': {}, 'last_full_download': None}

    def _save_metadata(self):
        """保存元数据"""
        self._ensure_dirs()
        with open(self.metadata_file, 'w', encoding='utf-8') as f:
            json.dump(self.metadata, f, ensure_ascii=False, indent=2)

    def _save_pool_snapshot(self):
        """保存当前股票池快照"""
        self._ensure_dirs()
        pool_data = {
            'version': 'v1.0',
            'created': datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'total_stocks': TOTAL_STOCKS,
            'total_sectors': TOTAL_SECTORS,
            'sectors': {k: [{'code_std': s['code_std'], 'name': s['name']} for s in v]
                        for k, v in STOCK_POOL.items()},
            'indices': [{'code_std': i['code_std'], 'name': i['name']} for i in INDEX_LIST],
        }
        with open(self.pool_file, 'w', encoding='utf-8') as f:
            json.dump(pool_data, f, ensure_ascii=False, indent=2)

    # -------------------- baostock 操作 --------------------

    @contextmanager
    def baostock_session(self):
        """baostock上下文管理器，确保login/logout配对"""
        if DEPS['baostock']:
            try:
                lg = bs.login()
                if lg.error_code == '0':
                    self._bs_logged_in = True
                else:
                    print(f"  ✗ baostock登录失败: {lg.error_msg}")
                    self._bs_logged_in = False
                yield
            finally:
                if self._bs_logged_in:
                    try:
                        bs.logout()
                    except Exception:
                        pass
                    self._bs_logged_in = False
        else:
            yield

    def _bs_fetch_kline(self, code: str, start_date: str, end_date: str,
                        fields: str = "date,open,high,low,close,volume,amount,turn",
                        adjustflag: str = "3") -> Optional[pd.DataFrame]:
        """用baostock获取K线数据，带重试和限速"""
        for attempt in range(1, self.BS_MAX_RETRIES + 1):
            try:
                time.sleep(self.BS_REQUEST_INTERVAL)
                rs = bs.query_history_k_data_plus(
                    code, fields,
                    start_date=start_date,
                    end_date=end_date,
                    frequency="d",
                    adjustflag=adjustflag
                )
                data_list = []
                while (rs.error_code == '0') and rs.next():
                    data_list.append(rs.get_row_data())

                if not data_list:
                    return None

                df = pd.DataFrame(data_list, columns=rs.fields)
                # 类型转换
                numeric_cols = [c for c in ['open', 'high', 'low', 'close', 'volume', 'amount', 'turn'] if c in df.columns]
                for col in numeric_cols:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
                if 'date' in df.columns:
                    df['date'] = pd.to_datetime(df['date'])
                df = df.sort_values('date').reset_index(drop=True)
                return df

            except Exception as e:
                err_msg = str(e)
                if '10054' in err_msg or '远程主机' in err_msg or 'Connection' in err_msg:
                    if attempt < self.BS_MAX_RETRIES:
                        sleep_time = self.BS_RETRY_SLEEP * attempt + random.uniform(0.5, 1.5)
                        print(f"    连接断开, {sleep_time:.1f}s后重连重试({attempt}/{self.BS_MAX_RETRIES})...")
                        time.sleep(sleep_time)
                        try:
                            bs.logout()
                        except Exception:
                            pass
                        time.sleep(1.0)
                        lg = bs.login()
                        self._bs_logged_in = (lg.error_code == '0')
                    else:
                        print(f"    ✗ 重试{self.BS_MAX_RETRIES}次仍失败: {code}")
                else:
                    return None
        return None

    # -------------------- mootdx 操作 --------------------

    def _mootdx_connect(self) -> bool:
        """连接mootdx服务器"""
        if not DEPS['mootdx']:
            return False
        if self._mootdx_connected and self._mootdx_client is not None:
            return True
        try:
            self._mootdx_client = Quotes.factory(market='std', bestip=True, timeout=self.MOOTDX_TIMEOUT)
            self._mootdx_connected = True
            return True
        except Exception:
            pass
        try:
            self._mootdx_client = Quotes.factory(market='std', bestip=False, timeout=self.MOOTDX_TIMEOUT)
            self._mootdx_connected = True
            return True
        except Exception:
            self._mootdx_connected = False
            return False

    def _mootdx_close(self):
        """关闭mootdx连接"""
        if self._mootdx_client is not None:
            try:
                self._mootdx_client.close()
            except Exception:
                pass
            self._mootdx_connected = False
            self._mootdx_client = None

    def _mootdx_fetch_recent_bars(self, mootdx_code: str, start_date: str = None,
                                   end_date: str = None, offset: int = 800) -> Optional[pd.DataFrame]:
        """
        用mootdx获取近期K线
        mootdx的bars()不支持日期过滤，只能按offset取最近N根
        返回后我们在内存中过滤日期
        """
        if not self._mootdx_connected or self._mootdx_client is None:
            return None
        try:
            df = self._mootdx_client.bars(symbol=mootdx_code, frequency=9, offset=min(offset, self.MOOTDX_MAX_BARS))
            if df is None or df.empty:
                return None
            # mootdx返回的列名映射到标准格式
            result = pd.DataFrame()
            result['date'] = pd.to_datetime(df.get('datetime', df.get('date', df.index)))
            result['open'] = pd.to_numeric(df.get('open', 0), errors='coerce')
            result['high'] = pd.to_numeric(df.get('high', 0), errors='coerce')
            result['low'] = pd.to_numeric(df.get('low', 0), errors='coerce')
            result['close'] = pd.to_numeric(df.get('close', 0), errors='coerce')
            result['volume'] = pd.to_numeric(df.get('vol', df.get('volume', 0)), errors='coerce')
            result['amount'] = pd.to_numeric(df.get('amount', 0), errors='coerce')

            # 按日期过滤
            if start_date:
                start_dt = pd.to_datetime(start_date)
                result = result[result['date'] >= start_dt]
            if end_date:
                end_dt = pd.to_datetime(end_date)
                result = result[result['date'] <= end_dt]

            result = result.sort_values('date').reset_index(drop=True)
            return result if len(result) > 0 else None
        except Exception:
            return None

    # -------------------- 文件操作 --------------------

    def _csv_path(self, code_std: str, is_index: bool = False) -> str:
        """获取CSV文件路径"""
        d = self.index_dir if is_index else self.daily_dir
        return os.path.join(d, f'{code_std}.csv')

    def _read_existing_csv(self, code_std: str, is_index: bool = False) -> Optional[pd.DataFrame]:
        """读取已有CSV"""
        path = self._csv_path(code_std, is_index)
        if os.path.exists(path):
            try:
                df = pd.read_csv(path, parse_dates=['date'])
                return df
            except Exception:
                return None
        return None

    def _write_csv(self, df: pd.DataFrame, code_std: str, name: str,
                   is_index: bool = False):
        """写入CSV，确保列完整"""
        self._ensure_dirs()
        path = self._csv_path(code_std, is_index)
        # 确保包含code_std和name列
        if 'code_std' not in df.columns:
            df['code_std'] = code_std
        if 'name' not in df.columns:
            df['name'] = name
        # 标准列顺序
        standard_cols = ['date', 'code_std', 'name', 'open', 'high', 'low', 'close', 'volume', 'amount', 'turn']
        cols = [c for c in standard_cols if c in df.columns] + [c for c in df.columns if c not in standard_cols]
        df[cols].to_csv(path, index=False, encoding='utf-8-sig')

    def _merge_incremental(self, existing: pd.DataFrame, new: pd.DataFrame) -> pd.DataFrame:
        """
        合并增量数据到已有数据
        策略: 按日期去重, 新数据覆盖同日旧数据(修正/复权), 保留旧数据的code_std/name列
        """
        if existing is None or existing.empty:
            return new
        if new is None or new.empty:
            return existing

        # 确保date列类型一致
        existing['date'] = pd.to_datetime(existing['date'])
        new['date'] = pd.to_datetime(new['date'])

        # 保留元信息列
        meta_cols = ['code_std', 'name']
        for col in meta_cols:
            if col in existing.columns and col not in new.columns:
                new[col] = existing[col].iloc[0]

        # 合并: 先拼接, 按日期排序, 同日取新数据
        combined = pd.concat([existing, new], ignore_index=True)
        combined = combined.sort_values('date')
        combined = combined.drop_duplicates(subset=['date'], keep='last')
        combined = combined.reset_index(drop=True)
        return combined

    # -------------------- 核心: 全量下载 --------------------

    def full_download(self, sectors: List[str] = None, start_date: str = None):
        """
        全量下载(使用baostock)
        Args:
            sectors: 指定板块列表(为None则全部)
            start_date: 起始日期(默认2020-01-01)
        """
        if not DEPS['baostock']:
            print("✗ baostock未安装，无法全量下载！请: pip install baostock")
            return

        # 统一转成 YYYY-MM-DD 格式(baostock要求)
        if start_date:
            s = start_date.replace('-', '')
            if len(s) == 8:
                start_date = f"{s[:4]}-{s[4:6]}-{s[6:8]}"
        start_date = start_date or self.DEFAULT_START_DATE
        end_date = datetime.datetime.now().strftime('%Y-%m-%d')

        self._ensure_dirs()
        self._save_pool_snapshot()
        self._stats = {'downloaded': 0, 'updated': 0, 'failed': 0, 'skipped': 0}

        total_tasks = 0
        if sectors:
            pool = {k: v for k, v in STOCK_POOL.items() if k in sectors}
        else:
            pool = STOCK_POOL
        total_tasks = sum(len(v) for v in pool.values()) + len(INDEX_LIST)

        print(f"\n{'━' * 70}")
        print(f"  全量数据下载 (baostock)")
        print(f"  日期范围: {start_date} → {end_date}")
        print(f"  股票: {sum(len(v) for v in pool.values())} 只 | 指数: {len(INDEX_LIST)} 只")
        print(f"{'━' * 70}")

        with self.baostock_session():
            if not self._bs_logged_in:
                print("✗ baostock登录失败, 终止下载")
                return

            task_idx = 0
            # 下载指数
            for idx_info in INDEX_LIST:
                task_idx += 1
                code_std = idx_info['code_std']
                name = idx_info['name']
                bs_code = code_to_baostock(code_std)
                print(f"  [{task_idx}/{total_tasks}] {name}({code_std})...", end="", flush=True)

                df = self._bs_fetch_kline(bs_code, start_date, end_date)
                if df is not None and len(df) > 0:
                    self._write_csv(df, code_std, name, is_index=True)
                    self.metadata['indices'][code_std] = {
                        'last_date': df['date'].max().strftime('%Y-%m-%d'),
                        'rows': len(df),
                        'source': 'baostock',
                        'updated': datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                    }
                    self._stats['downloaded'] += 1
                    print(f" ✓ ({len(df)}行)")
                else:
                    self._stats['failed'] += 1
                    print(f" ✗ 无数据")

                # 指数之间稍等
                time.sleep(0.3)

            # 下载股票
            for sector_name, stocks in pool.items():
                print(f"\n  ▸ 板块: {sector_name} ({len(stocks)}只)")
                for stock_info in stocks:
                    task_idx += 1
                    code_std = stock_info['code_std']
                    name = stock_info['name']
                    bs_code = stock_info['code']  # 已经是baostock格式
                    print(f"    [{task_idx}/{total_tasks}] {name}({code_std})...", end="", flush=True)

                    df = self._bs_fetch_kline(bs_code, start_date, end_date)
                    if df is not None and len(df) > 0:
                        self._write_csv(df, code_std, name, is_index=False)
                        self.metadata['stocks'][code_std] = {
                            'last_date': df['date'].max().strftime('%Y-%m-%d'),
                            'rows': len(df),
                            'sector': sector_name,
                            'source': 'baostock',
                            'updated': datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                        }
                        self._stats['downloaded'] += 1
                        print(f" ✓ ({len(df)}行)")
                    else:
                        self._stats['failed'] += 1
                        print(f" ✗ 无数据")

                    # 板块间额外等待
                    if task_idx % 20 == 0:
                        time.sleep(2.0)

        # 保存元数据
        self.metadata['last_full_download'] = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        self._save_metadata()

        # 汇总
        print(f"\n{'━' * 70}")
        print(f"  全量下载完成!")
        print(f"  成功: {self._stats['downloaded']} | 失败: {self._stats['failed']}")
        print(f"  数据目录: {self.data_dir}")
        print(f"{'━' * 70}")

    # -------------------- 核心: 增量更新 --------------------

    def incremental_update(self, sectors: List[str] = None):
        """
        增量更新(优先mootdx, 失败降级baostock)
        逻辑: 读取已有CSV最后日期 → 从该日期+1开始获取新数据 → 合并写入
        """
        self._ensure_dirs()
        self._stats = {'downloaded': 0, 'updated': 0, 'failed': 0, 'skipped': 0}

        if sectors:
            pool = {k: v for k, v in STOCK_POOL.items() if k in sectors}
        else:
            pool = STOCK_POOL

        today = datetime.datetime.now().strftime('%Y-%m-%d')

        total_tasks = sum(len(v) for v in pool.values()) + len(INDEX_LIST)

        print(f"\n{'━' * 70}")
        print(f"  增量数据更新")
        print(f"  目标日期: {today}")
        print(f"  策略: mootdx优先 → baostock降级")
        print(f"{'━' * 70}")

        # 尝试连接mootdx
        mootdx_ok = self._mootdx_connect()
        if mootdx_ok:
            print("  ✓ mootdx连接成功, 优先使用mootdx增量更新")
        else:
            print("  ⚠ mootdx连接失败, 降级使用baostock")

        # 构建更新列表
        update_list = []
        # 指数
        for idx_info in INDEX_LIST:
            update_list.append((idx_info['code_std'], idx_info['name'], True))
        # 股票
        for sector_name, stocks in pool.items():
            for stock_info in stocks:
                update_list.append((stock_info['code_std'], stock_info['name'], False))

        need_bs = False  # 是否需要baostock
        bs_tasks = []    # 需要baostock降级的任务

        task_idx = 0
        for code_std, name, is_index in update_list:
            task_idx += 1
            print(f"  [{task_idx}/{total_tasks}] {name}({code_std})...", end="", flush=True)

            # 读取已有数据, 获取最后日期
            existing = self._read_existing_csv(code_std, is_index)
            if existing is not None and len(existing) > 0:
                existing['date'] = pd.to_datetime(existing['date'])
                last_date = existing['date'].max().strftime('%Y-%m-%d')

                # 如果已是今天的数据, 跳过
                if last_date >= today:
                    self._stats['skipped'] += 1
                    print(f" 已是最新({last_date})")
                    continue

                start_date = (pd.to_datetime(last_date) + datetime.timedelta(days=1)).strftime('%Y-%m-%d')
            else:
                # 无已有数据, 需要全量下载
                start_date = self.DEFAULT_START_DATE
                existing = None

            # 策略1: mootdx
            new_data = None
            if mootdx_ok:
                mootdx_code = code_to_mootdx(code_std)
                if is_index:
                    # 指数用index接口
                    try:
                        df = self._mootdx_client.index(symbol=mootdx_code, frequency=9, offset=100)
                        if df is not None and not df.empty:
                            new_data = pd.DataFrame()
                            new_data['date'] = pd.to_datetime(df['datetime'] if 'datetime' in df.columns else df.index)
                            new_data['open'] = pd.to_numeric(df['open'], errors='coerce')
                            new_data['high'] = pd.to_numeric(df['high'], errors='coerce')
                            new_data['low'] = pd.to_numeric(df['low'], errors='coerce')
                            new_data['close'] = pd.to_numeric(df['close'], errors='coerce')
                            new_data['volume'] = pd.to_numeric(df.get('vol', df.get('volume', 0)), errors='coerce')
                            new_data['amount'] = pd.to_numeric(df.get('amount', 0), errors='coerce')
                            if start_date:
                                new_data = new_data[new_data['date'] >= pd.to_datetime(start_date)]
                            new_data = new_data.sort_values('date').reset_index(drop=True)
                            if len(new_data) == 0:
                                new_data = None
                    except Exception:
                        new_data = None
                else:
                    new_data = self._mootdx_fetch_recent_bars(mootdx_code, start_date=start_date, offset=200)

            # 策略2: baostock降级
            if new_data is None:
                # 标记需要baostock
                bs_tasks.append((code_std, name, is_index, start_date, existing))
                need_bs = True
                print(f" →待baostock")
                continue

            # 合并并写入
            merged = self._merge_incremental(existing, new_data)
            self._write_csv(merged, code_std, name, is_index)

            meta_key = 'indices' if is_index else 'stocks'
            self.metadata[meta_key][code_std] = {
                'last_date': merged['date'].max().strftime('%Y-%m-%d'),
                'rows': len(merged),
                'sector': self.metadata[meta_key].get(code_std, {}).get('sector', ''),
                'source': 'mootdx',
                'updated': datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            }
            self._stats['updated'] += 1
            print(f" ✓ +{len(new_data)}行 → {len(merged)}行")

        # baostock降级批量处理
        if need_bs and DEPS['baostock']:
            print(f"\n  ▸ baostock降级处理 ({len(bs_tasks)}只)...")
            with self.baostock_session():
                if not self._bs_logged_in:
                    print("    ✗ baostock登录失败, 跳过降级任务")
                else:
                    for code_std, name, is_index, start_date, existing in bs_tasks:
                        bs_code = code_to_baostock(code_std)
                        end_date = today
                        print(f"    {name}({code_std})...", end="", flush=True)
                        df = self._bs_fetch_kline(bs_code, start_date, end_date)
                        if df is not None and len(df) > 0:
                            merged = self._merge_incremental(existing, df)
                            self._write_csv(merged, code_std, name, is_index)
                            meta_key = 'indices' if is_index else 'stocks'
                            self.metadata[meta_key][code_std] = {
                                'last_date': merged['date'].max().strftime('%Y-%m-%d'),
                                'rows': len(merged),
                                'sector': self.metadata[meta_key].get(code_std, {}).get('sector', ''),
                                'source': 'baostock',
                                'updated': datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                            }
                            self._stats['updated'] += 1
                            print(f" ✓ +{len(df)}行 → {len(merged)}行")
                        else:
                            self._stats['failed'] += 1
                            print(f" ✗ 无新数据")

        # 关闭mootdx
        self._mootdx_close()

        # 保存元数据
        self._save_metadata()

        # 汇总
        print(f"\n{'━' * 70}")
        print(f"  增量更新完成!")
        print(f"  更新: {self._stats['updated']} | 跳过(已是最新): {self._stats['skipped']} | 失败: {self._stats['failed']}")
        print(f"  数据目录: {self.data_dir}")
        print(f"{'━' * 70}")

    # -------------------- 数据读取接口(供stocks3.py调用) --------------------

    def load_stock_data(self, code_std: str) -> Optional[pd.DataFrame]:
        """加载单只股票数据"""
        return self._read_existing_csv(code_std, is_index=False)

    def load_index_data(self, code_std: str) -> Optional[pd.DataFrame]:
        """加载指数数据"""
        return self._read_existing_csv(code_std, is_index=True)

    def load_all_stocks(self, sectors: List[str] = None) -> Dict[str, pd.DataFrame]:
        """
        批量加载所有股票数据
        Returns: {code_std: DataFrame}
        """
        result = {}
        pool = {k: v for k, v in STOCK_POOL.items() if (sectors is None or k in sectors)}
        for sector, stocks in pool.items():
            for stock_info in stocks:
                df = self.load_stock_data(stock_info['code_std'])
                if df is not None and len(df) > 0:
                    # 确保baostock格式的code作为key(兼容旧代码)
                    result[stock_info['code']] = df
        return result

    def load_all_indices(self) -> Dict[str, pd.DataFrame]:
        """加载所有指数数据"""
        result = {}
        for idx_info in INDEX_LIST:
            df = self.load_index_data(idx_info['code_std'])
            if df is not None and len(df) > 0:
                result[idx_info['code']] = df
        return result

    # -------------------- 全市场扫描与下载 --------------------

    def full_market_download(self, start_date: str = None, min_list_days: int = 365):
        """
        全A股市场数据下载（~5400只）
        使用baostock query_stock_basic()获取全部上市股票，自动分类行业
        Args:
            start_date: 起始日期(默认2020-01-01)
            min_list_days: 上市至少多少天才纳入（过滤新股）
        """
        if not DEPS['baostock']:
            print("✗ baostock未安装，无法全量下载！请: pip install baostock")
            return

        # 统一转成 YYYY-MM-DD 格式
        if start_date:
            s = start_date.replace('-', '')
            if len(s) == 8:
                start_date = f"{s[:4]}-{s[4:6]}-{s[6:8]}"
        start_date = start_date or self.DEFAULT_START_DATE

        self._ensure_dirs()
        self._stats = {'downloaded': 0, 'updated': 0, 'failed': 0, 'skipped': 0}

        print(f"\n{'━' * 70}")
        print(f"  全A股市场数据下载 (baostock)")
        print(f"  日期范围: {start_date} → {datetime.datetime.now().strftime('%Y-%m-%d')}")
        print(f"  正在扫描全部上市股票...")
        print(f"{'━' * 70}")

        with self.baostock_session():
            if not self._bs_logged_in:
                print("✗ baostock登录失败, 终止下载")
                return

            # Step 1: 获取全部股票列表
            print("\n  ▸ Step 1: 扫描全部A股...")
            all_stocks = self._scan_all_stocks(min_list_days)
            if not all_stocks:
                print("✗ 未扫描到任何股票")
                return

            # 按行业分组
            sector_map = {}
            for info in all_stocks:
                sec = info.get('industry', '未分类')
                if sec not in sector_map:
                    sector_map[sec] = []
                sector_map[sec].append(info)

            total = len(all_stocks)
            print(f"  ✓ 扫描完成: {total}只 / {len(sector_map)}个行业")
            print(f"  跳过: ST/退市/停牌超1年/上市不足{min_list_days}天")

            # Step 2: 下载指数（6只）
            print(f"\n  ▸ Step 2: 下载指数数据 ({len(INDEX_LIST)}只)")
            for idx_info in INDEX_LIST:
                code_std = idx_info['code_std']
                name = idx_info['name']
                bs_code = code_to_baostock(code_std)
                print(f"  {name}({code_std})...", end="", flush=True)

                df = self._bs_fetch_kline(bs_code, start_date, datetime.datetime.now().strftime('%Y-%m-%d'))
                if df is not None and len(df) > 0:
                    self._write_csv(df, code_std, name, is_index=True)
                    self.metadata['indices'][code_std] = {
                        'last_date': df['date'].max().strftime('%Y-%m-%d'),
                        'rows': len(df),
                        'source': 'baostock',
                        'updated': datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                    }
                    self._stats['downloaded'] += 1
                    print(f" ✓ ({len(df)}行)")
                else:
                    self._stats['failed'] += 1
                    print(f" ✗")
                time.sleep(0.3)

            # Step 3: 分行业下载股票数据
            print(f"\n  ▸ Step 3: 下载全市场股票数据")
            task_idx = len(INDEX_LIST)
            skip_existing = 0

            for sector_name, stocks in sorted(sector_map.items()):
                print(f"\n  ▸ 行业: {sector_name} ({len(stocks)}只)")
                for stock_info in stocks:
                    task_idx += 1
                    code_std = stock_info['code_std']
                    name = stock_info['name']
                    bs_code = stock_info['code']

                    # 检查是否已有数据（断点续传）
                    csv_path = self._csv_path(code_std, is_index=False)
                    if os.path.exists(csv_path):
                        # 已有数据，跳过（增量更新用update模式）
                        skip_existing += 1
                        continue

                    if task_idx % 50 == 1:
                        pct = task_idx / (total + len(INDEX_LIST)) * 100
                        print(f"  [{task_idx}/{total+len(INDEX_LIST)}] ({pct:.1f}%)", end=" ", flush=True)

                    df = self._bs_fetch_kline(bs_code, start_date, datetime.datetime.now().strftime('%Y-%m-%d'))
                    if df is not None and len(df) > 0:
                        self._write_csv(df, code_std, name, is_index=False)
                        self.metadata['stocks'][code_std] = {
                            'last_date': df['date'].max().strftime('%Y-%m-%d'),
                            'rows': len(df),
                            'sector': sector_name,
                            'source': 'baostock',
                            'updated': datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                        }
                        self._stats['downloaded'] += 1
                    else:
                        self._stats['failed'] += 1

                    # 限速+防断连
                    if task_idx % 20 == 0:
                        time.sleep(2.0)
                        # 检查连接
                        try:
                            test_rs = bs.query_history_k_data_plus(
                                "sh.000001", "date",
                                start_date=start_date,
                                end_date=datetime.datetime.now().strftime('%Y-%m-%d'),
                                frequency="d"
                            )
                            if test_rs.error_code != '0':
                                print(f"\n  ⚠ 连接断开，重连中...")
                                try: bs.logout()
                                except: pass
                                time.sleep(1)
                                bs.login()
                        except:
                            try: bs.logout()
                            except: pass
                            time.sleep(1)
                            bs.login()

            # 保存
            self.metadata['last_full_download'] = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            self.metadata['full_market'] = True
            self._save_metadata()
            self._save_pool_snapshot()

            # 汇总
            print(f"\n{'━' * 70}")
            print(f"  全市场下载完成!")
            print(f"  成功: {self._stats['downloaded']} | 跳过(已有): {skip_existing} | 失败: {self._stats['failed']}")
            print(f"  数据目录: {self.data_dir}")
            print(f"{'━' * 70}")

    def _scan_all_stocks(self, min_list_days: int = 365) -> list:
        """
        扫描全部A股，过滤ST/退市/停牌超1年/上市不足N天的股票
        策略: query_all_stock()为主(更可靠) → query_stock_basic()补充上市日期
        Returns: list of {code, code_std, name, industry, list_date}
        """
        stocks = []
        today = datetime.datetime.now()
        cutoff = today - datetime.timedelta(days=min_list_days)

        # ====== Step 1: query_all_stock 获取当日全部在市股票 ======
        print("  ▸ 方法1: query_all_stock()...")
        stock_set = set()
        try:
            # 用最近一个工作日查询（避免周末/节假日无数据）
            query_day = today
            for offset in range(7):
                test_day = today - datetime.timedelta(days=offset)
                if test_day.weekday() < 5:  # 周一到周五
                    query_day = test_day
                    break
            day_str = query_day.strftime('%Y-%m-%d')

            rs = bs.query_all_stock(day=day_str)
            print(f"    query_all_stock(day={day_str}): error_code={rs.error_code}, msg={rs.error_msg}")

            row_count = 0
            while rs.next():
                row = rs.get_row_data()
                row_count += 1
                if not row or len(row) < 3:
                    continue

                code = row[0] if isinstance(row, list) else str(row).split(',')[0].strip()
                # 只取沪深A股 (sh.6xxxxx, sz.0xxxxx, sz.3xxxxx)
                if not (code.startswith('sh.6') or code.startswith('sz.0') or code.startswith('sz.3')):
                    continue

                name = row[2] if len(row) > 2 else ''
                if isinstance(name, str) and ('ST' in name or 'st' in name):
                    continue

                code_std = self._baostock_to_code_std(code)
                if not code_std:
                    continue

                stock_set.add(code)
                stocks.append({
                    'code': code,
                    'code_std': code_std,
                    'name': name,
                    'industry': '',
                    'list_date': '',
                })

            print(f"    query_all_stock返回 {row_count} 行, 筛选后 {len(stocks)} 只A股")

        except Exception as e:
            print(f"    ✗ query_all_stock失败: {e}")

        # ====== Step 1b: 如果query_all_stock返回空，尝试query_stock_basic ======
        if not stocks:
            print("  ▸ 方法1失败，尝试方法2: query_stock_basic()...")
            try:
                rs = bs.query_stock_basic()
                print(f"    query_stock_basic: error_code={rs.error_code}, msg={rs.error_msg}")
                if hasattr(rs, 'fields'):
                    print(f"    fields: {rs.fields}")

                row_count = 0
                while rs.next():
                    row = rs.get_row_data()
                    row_count += 1

                    if not row:
                        continue

                    # 处理row可能是字符串的情况
                    if isinstance(row, str):
                        row = row.split(',')

                    fields = rs.fields if hasattr(rs, 'fields') else ['code','code_name','ipoDate','outDate','type','status']
                    if len(row) < len(fields):
                        continue

                    row_dict = dict(zip(fields, row))

                    code = row_dict.get('code', '')
                    name = row_dict.get('code_name', '')
                    ipo_date = row_dict.get('ipoDate', '')
                    out_date = row_dict.get('outDate', '')
                    stock_type = str(row_dict.get('type', ''))
                    status = str(row_dict.get('status', ''))

                    # 只要股票(type=1)，不要指数/基金
                    if stock_type != '1':
                        continue

                    # 过滤已退市
                    if out_date and str(out_date) not in ('', '0'):
                        continue

                    # 只取沪深A股
                    if not (code.startswith('sh.6') or code.startswith('sz.0') or code.startswith('sz.3')):
                        continue

                    # 过滤ST
                    if 'ST' in name or 'st' in name:
                        continue

                    # 过滤上市不足N天
                    if ipo_date and str(ipo_date) not in ('', '0'):
                        try:
                            ipo_dt = datetime.datetime.strptime(str(ipo_date), '%Y-%m-%d')
                            if ipo_dt > cutoff:
                                continue
                        except:
                            pass

                    code_std = self._baostock_to_code_std(code)
                    if not code_std:
                        continue

                    stocks.append({
                        'code': code,
                        'code_std': code_std,
                        'name': name,
                        'industry': '',
                        'list_date': str(ipo_date),
                    })

                print(f"    query_stock_basic返回 {row_count} 行, 筛选后 {len(stocks)} 只A股")

            except Exception as e:
                print(f"    ✗ query_stock_basic也失败: {e}")

        if not stocks:
            print("  ✗ 两种方法均未扫描到股票")
            return []

        # ====== Step 2: 过滤上市不足N天的（补充检查） ======
        # query_all_stock不返回ipoDate，需额外查询
        # 但为效率，直接用上市日期检查：如果股票在cutoff日期之前不存在，说明上市不足N天
        # 简化处理：对没有list_date的股票，用query_stock_basic补充查询
        need_ipo_check = [s for s in stocks if not s.get('list_date')]
        if need_ipo_check:
            print(f"  ▸ 补充查询上市日期 ({len(need_ipo_check)}只)...")
            try:
                rs = bs.query_stock_basic()
                ipo_map = {}
                while rs.next():
                    row = rs.get_row_data()
                    if not row:
                        continue
                    if isinstance(row, str):
                        row = row.split(',')
                    fields = rs.fields if hasattr(rs, 'fields') else ['code','code_name','ipoDate','outDate','type','status']
                    if len(row) >= len(fields):
                        rd = dict(zip(fields, row))
                        ipo_map[rd.get('code', '')] = rd.get('ipoDate', '')

                for s in need_ipo_check:
                    s['list_date'] = ipo_map.get(s['code'], '')
                    # 检查上市天数
                    if s['list_date'] and str(s['list_date']) not in ('', '0'):
                        try:
                            ipo_dt = datetime.datetime.strptime(str(s['list_date']), '%Y-%m-%d')
                            if ipo_dt > cutoff:
                                stocks.remove(s)
                        except:
                            pass
                print(f"    补充完成，剩余 {len(stocks)} 只")
            except Exception as e:
                print(f"    ⚠ 上市日期查询失败: {e}，跳过此过滤")

        # ====== Step 3: 获取行业分类 ======
        print(f"  ▸ 获取行业分类 ({len(stocks)}只)...")
        # 优先用query_stock_industry批量查
        industry_fail_count = 0
        for idx, stock_info in enumerate(stocks):
            code = stock_info['code']
            try:
                rs_ind = bs.query_stock_industry(code=code)
                while rs_ind.next():
                    row = rs_ind.get_row_data()
                    if not row:
                        continue
                    if isinstance(row, str):
                        row = row.split(',')
                    fields = rs_ind.fields if hasattr(rs_ind, 'fields') else ['code','code_name','industry','industryClassification']
                    if len(row) >= len(fields):
                        rd = dict(zip(fields, row))
                        stock_info['industry'] = rd.get('industry', rd.get('code_name', ''))
                    break
            except:
                industry_fail_count += 1

            # 限速+进度
            if (idx + 1) % 200 == 0:
                print(f"    行业分类进度: {idx+1}/{len(stocks)}")
                time.sleep(1.0)
            elif (idx + 1) % 50 == 0:
                time.sleep(0.5)

        # 未分行业的归入"其他"
        no_industry = sum(1 for s in stocks if not s['industry'])
        for s in stocks:
            if not s['industry']:
                s['industry'] = '其他'

        print(f"  ✓ 扫描完成: {len(stocks)}只, 行业未知: {no_industry}只(归入'其他')")
        return stocks

    @staticmethod
    def _baostock_to_code_std(bs_code: str) -> str:
        """baostock代码(sh.601857) → code_std(601857.SH)"""
        if '.' in bs_code:
            num, mkt = bs_code.split('.')
            mkt_upper = mkt.upper()
            # baostock中6/9开头是sh，0/3开头是sz
            if mkt == 'sh':
                return f'{num}.SH'
            elif mkt == 'sz':
                return f'{num}.SZ'
        return ''

    # -------------------- 状态检查 --------------------

    def status(self):
        """打印数据状态"""
        print(f"\n{'━' * 70}")
        print(f"  数据目录: {self.data_dir}")
        print(f"  目录存在: {'✓' if os.path.exists(self.data_dir) else '✗'}")

        # 统计文件
        stock_count = 0
        index_count = 0
        if os.path.exists(self.daily_dir):
            stock_count = len([f for f in os.listdir(self.daily_dir) if f.endswith('.csv')])
        if os.path.exists(self.index_dir):
            index_count = len([f for f in os.listdir(self.index_dir) if f.endswith('.csv')])

        print(f"  股票数据: {stock_count} 只 | 指数数据: {index_count} 只")
        print(f"  股票池: {TOTAL_STOCKS} 只 / {TOTAL_SECTORS} 板块")

        # 上次更新
        if self.metadata.get('last_full_download'):
            print(f"  上次全量下载: {self.metadata['last_full_download']}")

        # 各板块覆盖
        for sector, stocks in STOCK_POOL.items():
            covered = sum(1 for s in stocks if os.path.exists(self._csv_path(s['code_std'])))
            bar = '█' * covered + '░' * (len(stocks) - covered)
            print(f"  {sector:8s} [{bar}] {covered}/{len(stocks)}")

        # 数据总量
        total_size = 0
        for d in [self.daily_dir, self.index_dir]:
            if os.path.exists(d):
                for f in os.listdir(d):
                    fp = os.path.join(d, f)
                    if os.path.isfile(fp):
                        total_size += os.path.getsize(fp)
        if total_size > 1024 * 1024:
            print(f"  数据总量: {total_size / 1024 / 1024:.1f} MB")
        elif total_size > 1024:
            print(f"  数据总量: {total_size / 1024:.1f} KB")
        else:
            print(f"  数据总量: {total_size} B")

        print(f"{'━' * 70}")


# ============================================================
# 命令行入口
# ============================================================
def main():
    parser = argparse.ArgumentParser(description='五域归元·财略 数据管理器')
    parser.add_argument('--mode', choices=['full', 'full-market', 'update', 'status'], default='status',
                        help='运行模式: full=精选池下载, full-market=全A股下载, update=增量更新, status=查看状态')
    parser.add_argument('--sectors', type=str, default=None,
                        help='指定板块(逗号分隔), 如: 半导体,小金属')
    parser.add_argument('--start-date', type=str, default=None,
                        help='全量下载起始日期(如2020-01-01或20200101)')
    parser.add_argument('--data-dir', type=str, default=None,
                        help='数据存储目录(默认自动检测)')

    args = parser.parse_args()

    mgr = StockDataManager(data_dir=args.data_dir)

    if args.mode == 'status':
        mgr.status()
    elif args.mode == 'full':
        sectors = args.sectors.split(',') if args.sectors else None
        mgr.full_download(sectors=sectors, start_date=args.start_date)
    elif args.mode == 'full-market':
        mgr.full_market_download(start_date=args.start_date)
    elif args.mode == 'update':
        sectors = args.sectors.split(',') if args.sectors else None
        mgr.incremental_update(sectors=sectors)


if __name__ == '__main__':
    main()

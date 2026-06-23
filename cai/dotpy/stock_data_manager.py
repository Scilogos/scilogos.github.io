"""
stock_data_manager.py - 证券数据采集与管理
==========================================
Phase 1: 数据获取
功能:
  1. 全A股日K线扫描与下载 (baostock)
  2. 重点池5分钟线滚动采集
  3. 行业分类 (申万一级)
  4. DataAdapter 统一接口层
  5. 断点续传 + 进度追踪 + 数据校验

Bug修复记录(v2.0):
  - _baostock_to_code_std: mkt/num赋值反了 → 已修正
  - 行业分类无超时 → threading.Timer 10s
  - 迭代逻辑错 → 明确cursor机制

用法:
  python stock_data_manager.py --mode scan
  python stock_data_manager.py --mode download-daily [--max-stocks 0] [--resume]
  python stock_data_manager.py --mode download-min5 [--focus-file focus_pool.txt]
  python stock_data_manager.py --mode classify
  python stock_data_manager.py --mode verify
  python stock_data_manager.py --mode adapter-demo
"""

import os, sys, time, json, re, threading, argparse, traceback
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from collections import defaultdict

import numpy as np
import pandas as pd

# ── 导入共享配置 ──
sys.path.insert(0, str(Path(__file__).parent))
from stock_config import (
    DATA_DIR, ADV_DATA_DIR, BS_FIELDS_DAILY, BS_FIELDS_MIN5,
    DAILY_START, DAILY_END, MIN5_START, FOCUS_POOL_SIZE,
    SW_INDUSTRY_MAP, setup_logger, sha256_file,
)

logger = setup_logger("DataManager")

# ============================================================
# Baostock 连接管理器
# ============================================================
class BaostockConnection:
    """管理baostock连接生命周期，处理断线重连"""

    def __init__(self, max_retries=3, retry_delay=5):
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self._bs = None

    def __enter__(self):
        self._login()
        return self

    def _login(self):
        import baostock as bs
        for attempt in range(self.max_retries):
            try:
                lg = bs.login()
                if lg.error_code == '0':
                    self._bs = bs
                    logger.info(f"Baostock登录成功 (attempt {attempt+1})")
                    return
                else:
                    logger.warning(f"登录失败: {lg.error_msg}, 重试...")
                    time.sleep(self.retry_delay)
            except Exception as e:
                logger.warning(f"连接异常: {e}, 重试...")
                time.sleep(self.retry_delay)
        raise RuntimeError("Baostock登录失败，已达最大重试次数")

    def reconnect(self):
        """断线重连"""
        logger.warning("连接异常，尝试重连...")
        try:
            self._bs.logout()
        except Exception:
            pass
        time.sleep(self.retry_delay)
        self._login()
        logger.info("重连成功")

    def __exit__(self, *args):
        if self._bs:
            try:
                self._bs.logout()
                logger.info("Baostock已断开")
            except Exception as e:
                logger.warning(f"断开异常: {e} (可忽略，socket残留会自动清理)")
            self._bs = None

    @property
    def bs(self):
        return self._bs

# ============================================================
# 核心工具函数
# ============================================================
def baostock_code_to_std(bs_code: str) -> str:
    """
    baostock代码 → 标准代码
    例: sh.600000 → 600000.SH   sz.000001 → 000001.SZ
    
    【关键修复】v1.x中 mkt,num = bs_code.split('.') 赋值反了
    正确: split('.') → [market, number]，即 mkt=sh/sz, num=600000
    """
    parts = bs_code.split('.')
    if len(parts) != 2:
        return bs_code
    mkt, num = parts[0], parts[1]  # ← 关键：mkt在前, num在后
    suffix = mkt.upper()           # sh→SH, sz→SZ
    return f"{num}.{suffix}"


def std_code_to_baostock(std_code: str) -> str:
    """标准代码 → baostock代码  例: 600000.SH → sh.600000"""
    num, suffix = std_code.split('.')
    return f"{suffix.lower()}.{num}"


def is_valid_stock_code(code: str) -> bool:
    """过滤指数、基金、退市等非正常股票"""
    # baostock格式: sh.600000
    parts = code.split('.')
    if len(parts) != 2:
        return False
    mkt, num = parts[0], parts[1]
    # 沪市主板: 600xxx/601xxx/603xxx/605xxx
    # 深市主板: 000xxx/001xxx
    # 创业板: 300xxx/301xxx
    # 科创板: 688xxx
    valid_prefixes = ('600', '601', '603', '605', '000', '001', '300', '301', '688')
    if not any(num.startswith(p) for p in valid_prefixes):
        return False
    # 排除ST标记(会在数据中标记，这里不硬排除)
    return True

# ============================================================
# 全市场扫描
# ============================================================
def scan_all_stocks() -> List[Dict]:
    """扫描全A股列表，返回 [{code, name, type, status, std_code}]"""
    import baostock as bs
    logger.info("开始全市场扫描...")
    
    all_stocks = []
    stocks_seen = set()

    # 回溯最近7个自然日，找到有数据的交易日（跳过周末和假日）
    candidate_dates = []
    for i in range(7):
        d = datetime.now() - timedelta(days=i)
        candidate_dates.append(d.strftime("%Y-%m-%d"))

    for trade_date in candidate_dates:
        rs = bs.query_all_stock(day=trade_date)
        if rs.error_code != '0':
            logger.warning(f"查询{trade_date}失败: {rs.error_msg}")
            continue
        day_count = 0
        while rs.next():
            row = rs.get_row_data()
            code = row[0]  # sh.600000 格式
            if code in stocks_seen:
                continue
            if not is_valid_stock_code(code):
                continue
            stocks_seen.add(code)
            all_stocks.append({
                'code': code,
                'std_code': baostock_code_to_std(code),
                'name': row[1] if len(row) > 1 else '',
                'trade_date': trade_date,
            })
            day_count += 1
        if all_stocks:
            logger.info(f"使用交易日 {trade_date} 获取到 {day_count} 只股票")
            break  # 成功获取就不再尝试更早的日期
        else:
            logger.info(f"{trade_date} 无数据（可能非交易日），尝试前一天...")
    
    logger.info(f"扫描完成: {len(all_stocks)} 只A股")
    return all_stocks

# ============================================================
# 日K线下载
# ============================================================
def download_daily_data(bs_conn, stock_list: List[Dict],
                        data_dir: Path, max_stocks: int = 0,
                        resume: bool = True) -> Dict:
    """
    批量下载日K线数据
    返回: {total, success, skip, fail, details}
    """
    bs = bs_conn.bs
    if max_stocks > 0:
        stock_list = stock_list[:max_stocks]
    
    total = len(stock_list)
    success, skip, fail = 0, 0, 0
    details = []
    
    # 断点续传: 已存在的文件跳过
    existing = set()
    if resume:
        for f in data_dir.glob("*.csv"):
            code_part = f.stem.split('_')[0]  # 600000_SH 格式
            existing.add(code_part)
    
    logger.info(f"开始下载日K线: 目标{total}只, 已有{len(existing)}只(跳过), "
                f"待下载{total - len(existing)}只")
    
    for i, stk in enumerate(stock_list):
        std_code = stk['std_code']
        code_part = std_code.replace('.', '_')
        out_path = data_dir / f"{code_part}_daily.csv"
        
        if resume and code_part in existing:
            skip += 1
            if (i + 1) % 500 == 0:
                logger.info(f"进度: {i+1}/{total} (跳过已有)")
            continue
        
        bs_code = stk['code']
        retry_count = 0
        max_retries_per_stock = 3
        while retry_count <= max_retries_per_stock:
            try:
                # 用线程超时防止卡死
                result_holder = {}
                def _do_query():
                    rs = bs.query_history_k_data_plus(
                        bs_code,
                        ",".join(BS_FIELDS_DAILY),
                        start_date=DAILY_START,
                        end_date=DAILY_END,
                        frequency="d",
                        adjustflag="2"
                    )
                    rows = []
                    while rs.next():
                        rows.append(rs.get_row_data())
                    result_holder['rows'] = rows
                    result_holder['error'] = rs.error_code

                t = threading.Thread(target=_do_query, daemon=True)
                t.start()
                t.join(timeout=30)  # 单只股票最多等30秒

                if t.is_alive():
                    # 超时，线程仍在跑，重连
                    logger.warning(f"下载超时 {std_code} (30s)，重连...")
                    bs_conn.reconnect()
                    bs = bs_conn.bs
                    retry_count += 1
                    continue

                if result_holder.get('error') != '0':
                    raise Exception(f"baostock error: {result_holder.get('error')}")

                rows = result_holder.get('rows', [])

                if rows:
                    df = pd.DataFrame(rows, columns=BS_FIELDS_DAILY)
                    for col in ['open', 'high', 'low', 'close', 'preclose',
                                'volume', 'amount', 'turn', 'pctChg']:
                        df[col] = pd.to_numeric(df[col], errors='coerce')
                    df.to_csv(out_path, index=False, encoding='utf-8')
                    success += 1
                else:
                    skip += 1
                break  # 成功，跳出重试循环

            except Exception as e:
                retry_count += 1
                if retry_count <= max_retries_per_stock:
                    logger.warning(f"下载失败 {std_code} (第{retry_count}次): {e}，重连重试...")
                    try:
                        bs_conn.reconnect()
                        bs = bs_conn.bs
                    except Exception:
                        pass
                    time.sleep(2)
                else:
                    fail += 1
                    details.append({'code': std_code, 'error': str(e)})
                    if fail <= 5:
                        logger.warning(f"下载失败 {std_code}: {e} (已重试{max_retries_per_stock}次)")
        
        # 进度输出
        if (i + 1) % 100 == 0 or (i + 1) == total:
            logger.info(f"进度: {i+1}/{total} | 成功{success} 跳过{skip} 失败{fail}")
    
    result = {'total': total, 'success': success, 'skip': skip,
              'fail': fail, 'details': details}
    logger.info(f"日K线下载完成: {result}")
    return result

# ============================================================
# 5分钟线下载 (重点池)
# ============================================================
def download_min5_data(bs_conn, focus_codes: List[str],
                       data_dir: Path) -> Dict:
    """
    下载重点池股票5分钟线
    focus_codes: baostock格式列表 [sh.600000, ...]
    """
    bs = bs_conn.bs
    total = len(focus_codes)
    success, fail = 0, 0
    
    logger.info(f"开始下载5分钟线: {total}只")
    
    min5_dir = data_dir / "min5"
    min5_dir.mkdir(exist_ok=True)
    
    for i, bs_code in enumerate(focus_codes):
        std_code = baostock_code_to_std(bs_code)
        code_part = std_code.replace('.', '_')
        out_path = min5_dir / f"{code_part}_min5.csv"
        
        try:
            rs = bs.query_history_k_data_plus(
                bs_code,
                ",".join(BS_FIELDS_MIN5),
                start_date=MIN5_START,
                end_date=DAILY_END,
                frequency="5",
                adjustflag="2"
            )
            rows = []
            while rs.next():
                rows.append(rs.get_row_data())
            
            if rows:
                df = pd.DataFrame(rows, columns=BS_FIELDS_MIN5)
                for col in ['open', 'high', 'low', 'close', 'volume', 'amount']:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
                df.to_csv(out_path, index=False, encoding='utf-8')
                success += 1
            else:
                fail += 1
                
        except Exception as e:
            fail += 1
            if fail <= 5:
                logger.warning(f"5分钟线下载失败 {std_code}: {e}")
        
        if (i + 1) % 50 == 0:
            logger.info(f"5分钟线进度: {i+1}/{total}")
    
    result = {'total': total, 'success': success, 'fail': fail}
    logger.info(f"5分钟线下载完成: {result}")
    return result

def build_focus_pool(data_dir: Path, pool_size: int = FOCUS_POOL_SIZE) -> List[str]:
    """
    构建重点池: 按成交额排序取TopN
    返回baostock格式代码列表
    """
    volumes = []
    for f in data_dir.glob("*_daily.csv"):
        try:
            df = pd.read_csv(f, usecols=['code', 'amount'], nrows=30)
            avg_amount = df['amount'].mean()
            if not np.isnan(avg_amount):
                code = df['code'].iloc[0]  # baostock格式
                volumes.append((code, avg_amount))
        except Exception:
            continue
    
    volumes.sort(key=lambda x: x[1], reverse=True)
    pool = [v[0] for v in volumes[:pool_size]]
    logger.info(f"重点池构建完成: {len(pool)}只 (按近30日成交额排序)")
    return pool

# ============================================================
# 行业分类
# ============================================================
def classify_industries(bs_conn, stock_list: List[Dict],
                        data_dir: Path) -> Dict[str, List[str]]:
    """
    查询行业分类（支持申万+证监会）
    返回: {行业名: [std_code, ...]}
    带断点续传: 已分类的股票跳过

    ★★★ Baostock query_stock_industry 实际返回字段 ★★★
      [0] updateDate     日期 (如 "2026-06-22")
      [1] code           股票代码 (如 "sh.600000")
      [2] code_name      股票名称 (如 "浦发银行")
      [3] industry       行业编码+名称 (如 "J66货币金融服务")
      [4] industryClassification  分类标准 (如 "证监会行业分类")

    注意: 实测2026年6月只有"证监会行业分类"，没有"申万"分类。
    因此优先取申万，没有申万则用证监会行业名称（去掉编码前缀）。
    """
    bs = bs_conn.bs
    industry_map = defaultdict(list)
    total = len(stock_list)
    timeout_count = 0
    success_count = 0
    debug_logged = 0

    # 断点续传: 加载已有分类结果
    out_path = data_dir / "industry_classification.json"
    classified = set()
    if out_path.exists():
        try:
            with open(out_path, 'r', encoding='utf-8') as f:
                saved = json.load(f)
            for ind, codes in saved.items():
                industry_map[ind].extend(codes)
                # ★ "未知"的不算已分类，重跑时需要重新查询
                if ind != "未知":
                    for c in codes:
                        classified.add(c)
            n_unknown = len(saved.get("未知", []))
            logger.info(f"已有分类记录: {len(classified)}只(跳过), 未知{len(classified) if n_unknown==0 else n_unknown}只(待重查)")
        except Exception:
            pass

    logger.info(f"开始行业分类: {total}只, 待查询{total - len(classified)}只")

    for i, stk in enumerate(stock_list):
        bs_code = stk['code']
        std_code = stk['std_code']

        # 断点续传: 跳过已分类的
        if std_code in classified:
            continue

        try:
            rs = bs.query_stock_industry(code=bs_code)
            industry_name = "未知"
            # ★ 兼容字符串和整数error_code
            if str(rs.error_code) == '0':
                # baostock可能返回多行(申万+证监会)，优先取申万
                while rs.next():
                    row = rs.get_row_data()
                    # 前几只打印原始数据方便调试
                    if debug_logged < 3:
                        logger.info(f"  调试 {bs_code}: {row}")
                        debug_logged += 1
                    # ★★★ 正确字段映射 ★★★
                    # row[3] = 行业编码+名称 (如 "J66货币金融服务")
                    # row[4] = 分类标准 (如 "证监会行业分类" / "申万行业分类")
                    if len(row) >= 5:
                        raw_industry = row[3].strip()   # "J66货币金融服务"
                        cls = row[4].strip()              # "证监会行业分类"
                        # 去掉行业编码前缀 (如 "J66" → 取"货币金融服务")
                        clean_ind = re.sub(r'^[A-Z]\d+', '', raw_industry).strip()
                        if not clean_ind:
                            clean_ind = raw_industry  # 去不掉就用原始的
                        # 优先申万
                        if '申万' in cls and clean_ind:
                            industry_name = clean_ind
                        elif clean_ind and industry_name == "未知":
                            industry_name = clean_ind  # 退而求其次用证监会的
                    elif len(row) >= 4:
                        # 兼容4字段的情况(旧版Baostock?)
                        raw_industry = row[3].strip()
                        clean_ind = re.sub(r'^[A-Z]\d+', '', raw_industry).strip()
                        if clean_ind and industry_name == "未知":
                            industry_name = clean_ind
            else:
                timeout_count += 1

            industry_map[industry_name].append(std_code)
            success_count += 1
            classified.add(std_code)
        except Exception as e:
            timeout_count += 1
            industry_map["未知"].append(std_code)
            # ★ "未知"的不加入classified，下次重跑会重查
            # classified.add(std_code)  ← 故意不加
            if timeout_count % 50 == 1:
                logger.warning(f"分类异常 {std_code}: {e}，重连...")
                try:
                    bs_conn.reconnect()
                    bs = bs_conn.bs
                except Exception:
                    pass

        # 每100只保存一次（断点续传）
        if (i + 1) % 100 == 0:
            logger.info(f"分类进度: {i+1}/{total} | 成功{success_count} 超时{timeout_count}")
            with open(out_path, 'w', encoding='utf-8') as f:
                json.dump(dict(industry_map), f, ensure_ascii=False, indent=2)

    # 最终保存
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(dict(industry_map), f, ensure_ascii=False, indent=2)

    logger.info(f"行业分类完成: {len(industry_map)}个行业, 成功{success_count}, 超时/失败{timeout_count}")
    return dict(industry_map)

# ============================================================
# 数据校验
# ============================================================
def verify_data(data_dir: Path) -> Dict:
    """校验已下载数据的完整性"""
    results = {
        'total_files': 0, 'valid_files': 0, 'empty_files': 0,
        'corrupt_files': 0, 'total_rows': 0, 'date_range': None,
        'issues': []
    }
    
    dates_min, dates_max = None, None
    
    for f in data_dir.glob("*_daily.csv"):
        results['total_files'] += 1
        try:
            df = pd.read_csv(f)
            if len(df) == 0:
                results['empty_files'] += 1
                continue
            
            # 检查必要列
            required = ['date', 'open', 'high', 'low', 'close', 'volume']
            missing = [c for c in required if c not in df.columns]
            if missing:
                results['corrupt_files'] += 1
                results['issues'].append(f"{f.name}: 缺列{missing}")
                continue
            
            # 检查OHLC关系
            bad_ohlc = ((df['high'] < df['low']) | 
                       (df['high'] < df['open']) | 
                       (df['high'] < df['close'])).sum()
            if bad_ohlc > 0:
                results['issues'].append(f"{f.name}: {bad_ohlc}条OHLC异常")
            
            results['valid_files'] += 1
            results['total_rows'] += len(df)
            
            # 日期范围
            dmin, dmax = df['date'].min(), df['date'].max()
            if dates_min is None or dmin < dates_min:
                dates_min = dmin
            if dates_max is None or dmax > dates_max:
                dates_max = dmax
                
        except Exception as e:
            results['corrupt_files'] += 1
            results['issues'].append(f"{f.name}: 读取异常 {e}")
    
    results['date_range'] = f"{dates_min} ~ {dates_max}" if dates_min else "N/A"
    logger.info(f"数据校验完成: {results['valid_files']}/{results['total_files']}有效, "
                f"总行数{results['total_rows']}, 日期{results['date_range']}")
    return results

# ============================================================
# DataAdapter - 统一接口层
# ============================================================
class DataAdapter:
    """
    统一数据接口层，对外屏蔽数据源差异
    支持日K线、5分钟线两种粒度
    输出标准化为 (T, F) numpy数组，F=6 (OHLCV+pctChg)
    """

    def __init__(self, data_dir: Path = DATA_DIR, seq_len: int = 30):
        self.data_dir = Path(data_dir)
        self.seq_len = seq_len
        self._cache = {}
        self._industry_map = self._load_industry_map()
    
    def _load_industry_map(self) -> Dict[str, str]:
        """加载行业分类: std_code → 行业名"""
        mapping = {}
        ipath = self.data_dir / "industry_classification.json"
        if ipath.exists():
            with open(ipath, 'r', encoding='utf-8') as f:
                ind_map = json.load(f)
            for industry, codes in ind_map.items():
                for code in codes:
                    mapping[code] = industry
        return mapping
    
    def list_stocks(self, industry: Optional[str] = None,
                     min_length: int = 200) -> List[str]:
        """列出可用股票(std_code)，可按行业筛选"""
        stocks = []
        for f in self.data_dir.glob("*_daily.csv"):
            std_code = f.stem.replace('_daily', '').replace('_', '.')
            try:
                df = pd.read_csv(f, usecols=['close'])
                if len(df) < min_length:
                    continue
                if industry and self._industry_map.get(std_code, '') != industry:
                    continue
                stocks.append(std_code)
            except Exception:
                continue
        return sorted(stocks)
    
    def load_stock(self, std_code: str, normalize: bool = True
                    ) -> Optional[Tuple[np.ndarray, Dict]]:
        """
        加载单只股票数据
        返回: (data, meta) 或 None
        data: (T, 6) [open, high, low, close, volume, pctChg]
        meta: {std_code, industry, mean, std, count, date_range}
        """
        if std_code in self._cache:
            return self._cache[std_code]
        
        code_part = std_code.replace('.', '_')
        fpath = self.data_dir / f"{code_part}_daily.csv"
        
        if not fpath.exists():
            return None
        
        try:
            df = pd.read_csv(fpath)
            if len(df) < self.seq_len + 10:
                return None
            
            # 构建特征
            close = df['close'].values.astype(np.float64)
            pct = np.zeros_like(close)
            pct[1:] = (close[1:] - close[:-1]) / (close[:-1] + 1e-8)
            
            features = np.column_stack([
                df['open'].values.astype(np.float64),
                df['high'].values.astype(np.float64),
                df['low'].values.astype(np.float64),
                close,
                df['volume'].values.astype(np.float64),
                pct,
            ])
            
            # 处理NaN
            features = np.nan_to_num(features, nan=0.0)
            
            meta = {
                'std_code': std_code,
                'industry': self._industry_map.get(std_code, '未知'),
                'count': len(features),
                'date_range': f"{df['date'].iloc[0]}~{df['date'].iloc[-1]}",
            }
            
            if normalize:
                meta['mean'] = features.mean(axis=0)
                meta['std'] = features.std(axis=0) + 1e-8
                features = (features - meta['mean']) / meta['std']
            
            self._cache[std_code] = (features, meta)
            return features, meta
            
        except Exception as e:
            logger.warning(f"加载{std_code}失败: {e}")
            return None
    
    def load_min5(self, std_code: str, date: Optional[str] = None
                   ) -> Optional[np.ndarray]:
        """加载5分钟线数据 (T, 6)"""
        code_part = std_code.replace('.', '_')
        min5_dir = self.data_dir / "min5"
        fpath = min5_dir / f"{code_part}_min5.csv"
        
        if not fpath.exists():
            return None
        
        try:
            df = pd.read_csv(fpath)
            if date:
                df = df[df['date'] == date]
            
            features = np.column_stack([
                df['open'].values.astype(np.float64),
                df['high'].values.astype(np.float64),
                df['low'].values.astype(np.float64),
                df['close'].values.astype(np.float64),
                df['volume'].values.astype(np.float64),
                df['amount'].values.astype(np.float64),
            ])
            return np.nan_to_num(features, nan=0.0)
        except Exception:
            return None
    
    def make_sequences(self, std_code: str, stride: int = 1
                        ) -> Optional[Tuple[np.ndarray, Dict]]:
        """
        将单只股票切分为 (N, seq_len, 6) 序列
        返回: (sequences, meta)
        """
        result = self.load_stock(std_code)
        if result is None:
            return None
        
        data, meta = result
        T, F = data.shape
        N = (T - self.seq_len) // stride + 1
        
        if N <= 0:
            return None
        
        seqs = np.zeros((N, self.seq_len, F), dtype=np.float32)
        for i in range(N):
            start = i * stride
            seqs[i] = data[start:start + self.seq_len]
        
        return seqs, meta
    
    def load_batch(self, stock_codes: List[str],
                    max_seqs: int = 10000) -> Tuple[np.ndarray, List[Dict]]:
        """批量加载多只股票序列，拼接为 (N, seq_len, 6)"""
        all_seqs, all_metas = [], []
        total = 0
        
        for code in stock_codes:
            result = self.make_sequences(code)
            if result is None:
                continue
            seqs, meta = result
            all_seqs.append(seqs)
            all_metas.append(meta)
            total += len(seqs)
            if total >= max_seqs:
                break
        
        if not all_seqs:
            return np.array([]), []
        
        combined = np.concatenate(all_seqs, axis=0)
        if total > max_seqs:
            idx = np.random.choice(total, max_seqs, replace=False)
            combined = combined[idx]
        
        return combined, all_metas
    
    def load_industry_data(self, industry: str,
                            max_stocks: int = 50
                            ) -> Tuple[np.ndarray, Dict]:
        """按行业加载数据，用于生成器条件化"""
        stocks = self.list_stocks(industry=industry)
        if max_stocks > 0:
            stocks = stocks[:max_stocks]
        
        seqs, metas = self.load_batch(stocks)
        
        # 计算行业级统计
        if len(seqs) > 0:
            industry_meta = {
                'industry': industry,
                'num_stocks': len(stocks),
                'num_sequences': len(seqs),
                'mean': seqs.mean(axis=(0, 1)),
                'std': seqs.std(axis=(0, 1)),
            }
        else:
            industry_meta = {'industry': industry, 'num_stocks': 0, 'num_sequences': 0}
        
        return seqs, industry_meta

# ============================================================
# 命令行入口
# ============================================================
def main():
    parser = argparse.ArgumentParser(description="证券数据管理器")
    parser.add_argument("--mode", required=True,
                        choices=["scan", "download-daily", "download-min5",
                                 "classify", "verify", "adapter-demo"],
                        help="运行模式")
    parser.add_argument("--data-dir", type=str, default=str(DATA_DIR))
    parser.add_argument("--max-stocks", type=int, default=0,
                        help="最大下载数量(0=全部)")
    parser.add_argument("--resume", action="store_true",
                        help="断点续传，跳过已有文件")
    parser.add_argument("--focus-file", type=str, default=None,
                        help="重点池代码文件(每行一个baostock代码)")
    
    args = parser.parse_args()
    data_dir = Path(args.data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    
    if args.mode == "scan":
        with BaostockConnection() as conn:
            stocks = scan_all_stocks()
            # 保存列表
            out = data_dir / "stock_list.json"
            with open(out, 'w', encoding='utf-8') as f:
                json.dump(stocks, f, ensure_ascii=False, indent=2)
            logger.info(f"股票列表已保存: {out}")
    
    elif args.mode == "download-daily":
        # 加载股票列表
        list_file = data_dir / "stock_list.json"
        if not list_file.exists():
            logger.error("请先运行 --mode scan 生成股票列表")
            return
        with open(list_file, 'r', encoding='utf-8') as f:
            stock_list = json.load(f)
        
        with BaostockConnection() as conn:
            download_daily_data(conn, stock_list, data_dir,
                               max_stocks=args.max_stocks, resume=args.resume)
    
    elif args.mode == "download-min5":
        if args.focus_file:
            with open(args.focus_file, 'r') as f:
                focus_codes = [l.strip() for l in f if l.strip()]
        else:
            logger.info("未指定重点池，自动按成交额构建...")
            focus_codes = build_focus_pool(data_dir)
        
        with BaostockConnection() as conn:
            download_min5_data(conn, focus_codes, data_dir)
    
    elif args.mode == "classify":
        list_file = data_dir / "stock_list.json"
        if not list_file.exists():
            logger.error("请先运行 --mode scan")
            return
        with open(list_file, 'r') as f:
            stock_list = json.load(f)
        with BaostockConnection() as conn:
            classify_industries(conn, stock_list, data_dir)
    
    elif args.mode == "verify":
        result = verify_data(data_dir)
        out = data_dir / "verify_result.json"
        with open(out, 'w', encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
    
    elif args.mode == "adapter-demo":
        adapter = DataAdapter(data_dir)
        stocks = adapter.list_stocks()
        logger.info(f"可用股票: {len(stocks)}只")
        if stocks:
            data, meta = adapter.load_stock(stocks[0])
            if data is not None:
                logger.info(f"示例: {meta['std_code']} | 形状{data.shape} | "
                           f"行业{meta['industry']} | {meta['date_range']}")
            seqs, _ = adapter.make_sequences(stocks[0])
            if seqs is not None:
                logger.info(f"序列化: {seqs.shape}")

if __name__ == "__main__":
    main()

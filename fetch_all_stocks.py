# -*- coding: utf-8 -*-
"""
A股全市场数据批量下载脚本
=========================
从baostock拉取沪深全市场5000+只股票的日线数据
首次运行耗时较长（1-2小时），之后每日增量更新很快

用法:
  python fetch_all_stocks.py              # 全量下载（首次）
  python fetch_all_stocks.py --update     # 增量更新（日常）
  python fetch_all_stocks.py --update --push  # 增量更新并git push
"""

import os
import sys
import json
import time
import logging
import subprocess
from datetime import datetime, timedelta
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

import baostock as bs
import pandas as pd

# ── 配置 ──
STORAGE_DIR = "stockdata"
HISTORY_YEARS = 1
GITHUB_REPO = "https://github.com/Scilogos/scilogos.github.io.git"
LOG_FILE = "stockdata_all_update.log"

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler(LOG_FILE, encoding='utf-8'),
        logging.StreamHandler(sys.stdout),
    ]
)
logger = logging.getLogger(__name__)


def login_baostock():
    """登录baostock"""
    lg = bs.login()
    if lg.error_code != '0':
        logger.error(f"baostock登录失败: {lg.error_msg}")
        sys.exit(1)
    logger.info("baostock登录成功")
    return lg


def get_all_stock_codes(date: str = None) -> list:
    """
    获取全市场股票代码列表
    返回: ['sh.600000', 'sz.000001', ...]
    """
    if date is None:
        date = datetime.now().strftime("%Y-%m-%d")
    
    rs = bs.query_all_stock(day=date)
    if rs.error_code != '0':
        logger.error(f"获取股票列表失败: {rs.error_msg}")
        return []
    
    all_stocks = []
    while rs.next():
        row = rs.get_row_data()
        code = row[0]  # 股票代码
        status = row[4] if len(row) > 4 else '1'  # 上市状态: 1=上市
        
        # 只保留A股（排除基金、债券、指数等）
        if code.startswith(('sh.6', 'sz.0', 'sh.688', 'sz.300', 'sh.9', 'sz.399')):
            # 排除指数本身（但保留指数成分股）
            if code.startswith(('sh.9', 'sz.399')):
                continue  # 跳过指数代码
            all_stocks.append(code)
    
    # 额外添加主要指数
    indices = ['sh.000001', 'sz.399001', 'sz.399006', 'sh.000300', 'sh.000016', 'sh.000905']
    all_stocks = indices + sorted(set(all_stocks) - set(indices))
    
    logger.info(f"全市场股票数: {len(all_stocks)}")
    return all_stocks


def fetch_one_stock(stock_code: str, start_date: str, end_date: str) -> dict:
    """
    获取单只股票数据
    返回: {'code': str, 'success': bool, 'rows': int, 'error': str}
    """
    result = {'code': stock_code, 'success': False, 'rows': 0, 'error': ''}
    
    try:
        fields = "date,code,open,high,low,close,volume,amount,turn,pctChg"
        rs = bs.query_history_k_data_plus(
            stock_code,
            fields,
            start_date=start_date,
            end_date=end_date,
            frequency="d",
            adjustflag="3"  # 不复权
        )
        
        if rs.error_code != '0':
            result['error'] = rs.error_msg
            return result
        
        rows = []
        while rs.next():
            rows.append(rs.get_row_data())
        
        if not rows:
            return result
        
        # 构建DataFrame
        df = pd.DataFrame(rows, columns=rs.fields)
        
        # 列名标准化
        df = df.rename(columns={'pctChg': 'pct_change', 'turn': 'turnover'})
        
        # 数值转换
        numeric_cols = ['open', 'high', 'low', 'close', 'volume', 'amount', 'turnover', 'pct_change']
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # 涨跌停判断
        if 'pct_change' in df.columns:
            df['is_limit_up'] = ((df['pct_change'] >= 9.8) & df['code'].str.startswith('sh.6')) | \
                                ((df['pct_change'] >= 9.8) & df['code'].str.startswith('sz.0')) | \
                                ((df['pct_change'] >= 19.5) & df['code'].str.startswith('sh.688')) | \
                                ((df['pct_change'] >= 19.5) & df['code'].str.startswith('sz.300'))
            df['is_limit_down'] = ((df['pct_change'] <= -9.8) & df['code'].str.startswith('sh.6')) | \
                                  ((df['pct_change'] <= -9.8) & df['code'].str.startswith('sz.0')) | \
                                  ((df['pct_change'] <= -19.5) & df['code'].str.startswith('sh.688')) | \
                                  ((df['pct_change'] <= -19.5) & df['code'].str.startswith('sz.300'))
            df['is_limit_up'] = df['is_limit_up'].astype(int)
            df['is_limit_down'] = df['is_limit_down'].astype(int)
        
        # 保存CSV
        csv_path = Path(STORAGE_DIR) / f"{stock_code}.csv"
        
        if csv_path.exists() and '--update' in sys.argv:
            # 增量模式：合并新旧数据
            existing = pd.read_csv(csv_path)
            df = pd.concat([existing, df]).drop_duplicates(subset=['date', 'code']).sort_values('date')
        
        df.to_csv(csv_path, index=False)
        result['success'] = True
        result['rows'] = len(df)
        
    except Exception as e:
        result['error'] = str(e)
    
    return result


def run_full_download():
    """全量下载全市场数据"""
    logger.info("=" * 60)
    logger.info("★ A股全市场数据全量下载")
    logger.info("=" * 60)
    
    Path(STORAGE_DIR).mkdir(exist_ok=True)
    login_baostock()
    
    # 获取所有股票代码
    stock_codes = get_all_stock_codes()
    if not stock_codes:
        logger.error("未获取到任何股票代码")
        bs.logout()
        return False
    
    # 计算日期范围
    end_date = datetime.now().strftime("%Y-%m-%d")
    start_date = (datetime.now() - timedelta(days=HISTORY_YEARS * 365)).strftime("%Y-%m-%d")
    
    logger.info(f"数据范围: {start_date} ~ {end_date}")
    logger.info(f"股票数量: {len(stock_codes)}")
    logger.info(f"预计耗时: {len(stock_codes) * 0.5 / 60:.0f} ~ {len(stock_codes) * 1.0 / 60:.0f} 分钟")
    logger.info("-" * 60)
    
    # 逐个下载（baostock不支持并发，只能串行）
    success_count = 0
    fail_count = 0
    no_data_count = 0
    start_time = time.time()
    
    for i, code in enumerate(stock_codes):
        result = fetch_one_stock(code, start_date, end_date)
        
        if result['success']:
            success_count += 1
        elif result['rows'] == 0:
            no_data_count += 1
        else:
            fail_count += 1
        
        # 进度报告
        if (i + 1) % 100 == 0 or (i + 1) == len(stock_codes):
            elapsed = time.time() - start_time
            speed = (i + 1) / elapsed if elapsed > 0 else 0
            remaining = (len(stock_codes) - i - 1) / speed if speed > 0 else 0
            logger.info(
                f"  进度: {i+1}/{len(stock_codes)} "
                f"({(i+1)/len(stock_codes)*100:.1f}%) | "
                f"成功:{success_count} 无数据:{no_data_count} 失败:{fail_count} | "
                f"速度:{speed:.1f}只/秒 | "
                f"剩余:{remaining/60:.1f}分钟"
            )
    
    elapsed_total = time.time() - start_time
    logger.info("=" * 60)
    logger.info(f"★ 全量下载完成!")
    logger.info(f"  总耗时: {elapsed_total/60:.1f}分钟")
    logger.info(f"  成功: {success_count} | 无数据: {no_data_count} | 失败: {fail_count}")
    logger.info(f"  数据目录: {Path(STORAGE_DIR).resolve()}")
    logger.info(f"  文件数量: {len(list(Path(STORAGE_DIR).glob('*.csv')))}")
    
    # 计算总大小
    total_size = sum(f.stat().st_size for f in Path(STORAGE_DIR).glob('*.csv'))
    logger.info(f"  数据大小: {total_size/1024/1024:.1f} MB")
    logger.info("=" * 60)
    
    bs.logout()
    return True


def run_incremental_update(push: bool = False):
    """增量更新（只拉最新交易日数据）"""
    logger.info("=" * 60)
    logger.info("★ A股全市场增量更新")
    logger.info("=" * 60)
    
    Path(STORAGE_DIR).mkdir(exist_ok=True)
    login_baostock()
    
    stock_codes = get_all_stock_codes()
    if not stock_codes:
        logger.error("未获取到股票代码")
        bs.logout()
        return False
    
    # 检测最后更新日期
    last_date = None
    for csv_file in Path(STORAGE_DIR).glob("*.csv"):
        try:
            df = pd.read_csv(csv_file, usecols=['date'], nrows=0)
            df_full = pd.read_csv(csv_file)
            if 'date' in df_full.columns and len(df_full) > 0:
                file_last = df_full['date'].max()
                if last_date is None or file_last > last_date:
                    last_date = file_last
        except:
            continue
    
    if last_date:
        start_date = (datetime.strptime(last_date, "%Y-%m-%d") + timedelta(days=1)).strftime("%Y-%m-%d")
    else:
        start_date = (datetime.now() - timedelta(days=30)).strftime("%Y-%m-%d")
    
    end_date = datetime.now().strftime("%Y-%m-%d")
    
    if start_date > end_date:
        logger.info("数据已是最新，无需更新")
        bs.logout()
        return True
    
    logger.info(f"增量范围: {start_date} ~ {end_date}")
    logger.info(f"股票数量: {len(stock_codes)}")
    
    success_count = 0
    fail_count = 0
    start_time = time.time()
    
    for i, code in enumerate(stock_codes):
        result = fetch_one_stock(code, start_date, end_date)
        if result['success']:
            success_count += 1
        else:
            fail_count += 1
        
        if (i + 1) % 200 == 0 or (i + 1) == len(stock_codes):
            elapsed = time.time() - start_time
            speed = (i + 1) / elapsed if elapsed > 0 else 0
            remaining = (len(stock_codes) - i - 1) / speed if speed > 0 else 0
            logger.info(f"  进度: {i+1}/{len(stock_codes)} | 成功:{success_count} 失败:{fail_count} | 剩余:{remaining/60:.1f}分钟")
    
    logger.info(f"增量更新完成: 成功{success_count}, 失败{fail_count}")
    
    # Git push
    if push:
        git_sync()
    
    bs.logout()
    return True


def git_sync():
    """Git同步到GitHub"""
    logger.info("开始Git同步...")
    try:
        subprocess.run(["git", "add", "stockdata/"], check=True, capture_output=True)
        
        # 检查是否有变更
        status = subprocess.run(["git", "status", "--porcelain", "stockdata/"], 
                              capture_output=True, text=True)
        if not status.stdout.strip():
            logger.info("没有数据变更，跳过commit")
            return True
        
        date_str = datetime.now().strftime("%Y-%m-%d")
        subprocess.run(["git", "commit", "-m", f"Auto update ALL stock data - {date_str}"], 
                      check=True, capture_output=True)
        
        # pull rebase then push
        try:
            subprocess.run(["git", "pull", "--rebase", "origin", "main"], 
                          check=True, capture_output=True, timeout=60)
        except:
            pass
        
        subprocess.run(["git", "push", "origin", "main"], 
                      check=True, capture_output=True, timeout=120)
        logger.info("Git同步完成")
        return True
        
    except subprocess.CalledProcessError as e:
        logger.error(f"Git操作失败: {e}")
        return False


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="A股全市场数据下载")
    parser.add_argument("--update", action="store_true", help="增量更新模式")
    parser.add_argument("--push", action="store_true", help="更新后git push")
    parser.add_argument("--years", type=int, default=1, help="历史年数(默认1)")
    
    args = parser.parse_args()
    HISTORY_YEARS = args.years
    
    if args.update:
        run_incremental_update(push=args.push)
    else:
        run_full_download()

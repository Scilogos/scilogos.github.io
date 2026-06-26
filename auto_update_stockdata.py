# -*- coding: utf-8 -*-
"""
A股数据每日自动更新脚本
- 定时获取当日增量数据（OHLCV + 涨跌停信息）
- 追加到本地CSV文件
- 自动推送到GitHub备份
"""

import os
import sys
import json
import time
import logging
import logging.handlers
import subprocess
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import baostock as bs


class StockDataUpdater:
    """A股数据更新器"""
    
    def __init__(self, config_path: str = "config.json"):
        """初始化更新器"""
        self.config_path = config_path
        self.config = self._load_config()
        self.storage_dir = Path(self.config["data"]["storage_dir"])
        self.logger = self._setup_logger()
        self.stocks = self._get_stock_list()
        self.today = datetime.now().strftime("%Y-%m-%d")
        
    def _load_config(self) -> dict:
        """加载配置文件"""
        config_file = Path(self.config_path)
        if not config_file.exists():
            raise FileNotFoundError(f"配置文件不存在: {self.config_path}")
        with open(config_file, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def _setup_logger(self) -> logging.Logger:
        """设置日志记录器"""
        log_config = self.config["log"]
        logger = logging.getLogger("StockDataUpdater")
        logger.setLevel(getattr(logging, log_config["level"]))
        
        # 避免重复添加handler
        if not logger.handlers:
            # 文件Handler（带轮转）
            file_handler = logging.handlers.RotatingFileHandler(
                log_config["file"],
                maxBytes=log_config["max_bytes"],
                backupCount=log_config["backup_count"],
                encoding='utf-8'
            )
            file_handler.setLevel(logging.DEBUG)
            
            # 控制台Handler
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setLevel(logging.INFO)
            
            # 格式化
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            file_handler.setFormatter(formatter)
            console_handler.setFormatter(formatter)
            
            logger.addHandler(file_handler)
            logger.addHandler(console_handler)
        
        return logger
    
    def _get_stock_list(self) -> list:
        """获取股票列表"""
        stocks = []
        
        # 添加指数
        stocks.extend(self.config["stocks"]["indices"])
        
        # 添加自定义股票
        stocks.extend(self.config["stocks"]["custom"])
        
        self.logger.info(f"股票列表共 {len(stocks)} 个: {stocks}")
        return stocks
    
    def _ensure_storage_dir(self):
        """确保存储目录存在"""
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        self.logger.info(f"存储目录: {self.storage_dir}")
    
    def _get_date_range(self) -> tuple:
        """获取需要获取数据的日期范围"""
        today = datetime.now()
        history_years = self.config["data"]["history_years"]
        
        # 计算历史数据起始日期
        start_date = (today - timedelta(days=history_years * 365)).strftime("%Y-%m-%d")
        end_date = today.strftime("%Y-%m-%d")
        
        # 检查是否已有数据，决定是全量还是增量
        sample_file = self.storage_dir / f"{self.stocks[0]}.csv"
        if sample_file.exists():
            existing_df = pd.read_csv(sample_file, nrows=1)
            if 'date' in existing_df.columns and len(existing_df) > 0:
                self.logger.info("检测到已有历史数据，执行增量更新")
                start_date = self._get_last_update_date() or start_date
                if start_date >= end_date:
                    self.logger.info("数据已是最新，无需更新")
                    return None, None
        
        self.logger.info(f"数据范围: {start_date} 至 {end_date}")
        return start_date, end_date
    
    def _get_last_update_date(self) -> str:
        """获取最后更新日期"""
        latest_file = None
        latest_date = None
        
        for stock in self.stocks:
            file_path = self.storage_dir / f"{stock}.csv"
            if file_path.exists():
                try:
                    df = pd.read_csv(file_path)
                    if 'date' in df.columns and len(df) > 0:
                        file_date = df['date'].max()
                        if latest_date is None or file_date > latest_date:
                            latest_date = file_date
                            latest_file = stock
                except Exception as e:
                    self.logger.warning(f"读取 {stock} 数据失败: {e}")
        
        if latest_date:
            # 下一天开始
            last_dt = datetime.strptime(latest_date, "%Y-%m-%d")
            next_dt = last_dt + timedelta(days=1)
            return next_dt.strftime("%Y-%m-%d")
        
        return None
    
    def _fetch_data_baostock(self, stock_code: str, start_date: str, end_date: str) -> pd.DataFrame:
        """使用baostock获取数据"""
        self.logger.info(f"Baostock获取数据: {stock_code} ({start_date} ~ {end_date})")
        
        retry_times = self.config["schedule"]["retry_times"]
        retry_delay = self.config["schedule"]["retry_delay"]
        
        for attempt in range(retry_times):
            try:
                # 格式转换: sh.600519 -> 600519
                bs_code = stock_code.replace("sh.", "").replace("sz.", "")
                
                rs = bs.query_history_k_data_plus(
                    bs_code,
                    "date,code,open,high,low,close,volume,amount,turnover,pctChg",
                    start_date=start_date,
                    end_date=end_date,
                    frequency="d",
                    adjustflag="3"  # 不复权
                )
                
                if rs is None:
                    raise Exception("Baostock返回None")
                
                data_list = []
                # 正确写法：直接用 while rs.next()
                while rs.next():
                    data_list.append(rs.get_row_data())
                
                if not data_list:
                    self.logger.warning(f"{stock_code} 无数据")
                    return pd.DataFrame()
                
                df = pd.DataFrame(data_list, columns=[
                    "date", "code", "open", "high", "low", "close",
                    "volume", "amount", "turnover", "pct_change"
                ])
                
                # 添加原始股票代码
                df["code"] = stock_code
                
                # 添加涨跌停标记
                df = self._add_limit_markers(df)
                
                self.logger.info(f"{stock_code} 获取 {len(df)} 条数据")
                return df
                
            except Exception as e:
                self.logger.warning(f"第{attempt+1}次尝试失败: {e}")
                if attempt < retry_times - 1:
                    time.sleep(retry_delay)
                else:
                    self.logger.error(f"{stock_code} 获取失败，已达最大重试次数")
                    return pd.DataFrame()
    
    def _fetch_data_mootdx(self, stock_code: str, start_date: str, end_date: str) -> pd.DataFrame:
        """使用mootdx作为备用数据源"""
        self.logger.info(f"Mootdx备用获取数据: {stock_code}")
        
        try:
            from mootdx import StockBar
            
            # 格式转换: sh.600519 -> 600519
            code = stock_code.replace("sh.", "").replace("sz.", "")
            market = "sh" if stock_code.startswith("sh") else "sz"
            
            client = StockBar()
            
            # 转换日期格式
            start = start_date.replace("-", "")
            end = end_date.replace("-", "")
            
            df = client.stock_bar(
                stock_code=code,
                market=market,
                start_date=start,
                end_date=end
            )
            
            if df is None or df.empty:
                self.logger.warning(f"{stock_code} mootdx无数据")
                return pd.DataFrame()
            
            # 重命名列以匹配
            df = df.rename(columns={
                "date": "date",
                "open": "open",
                "high": "high",
                "low": "low",
                "close": "close",
                "volume": "volume",
                "amount": "amount"
            })
            
            df["code"] = stock_code
            df["turnover"] = ""
            df["pct_change"] = ""
            df = self._add_limit_markers(df)
            
            return df
            
        except ImportError:
            self.logger.error("mootdx未安装，请运行: pip install 'mootdx[all]'")
            return pd.DataFrame()
        except Exception as e:
            self.logger.error(f"mootdx获取失败: {e}")
            return pd.DataFrame()
    
    def _add_limit_markers(self, df: pd.DataFrame) -> pd.DataFrame:
        """添加涨跌停标记"""
        if df.empty or "pct_change" not in df.columns:
            return df
        
        df["is_limit_up"] = 0
        df["is_limit_down"] = 0
        
        # 判断涨跌停（涨幅>=9.5%视为涨停）
        try:
            df["pct_change"] = pd.to_numeric(df["pct_change"], errors="coerce")
            df.loc[df["pct_change"] >= 9.5, "is_limit_up"] = 1
            df.loc[df["pct_change"] <= -9.5, "is_limit_down"] = 1
        except Exception as e:
            self.logger.warning(f"涨跌停标记计算失败: {e}")
        
        return df
    
    def _save_to_csv(self, df: pd.DataFrame, stock_code: str):
        """保存数据到CSV"""
        if df.empty:
            self.logger.warning(f"{stock_code} 无数据，跳过保存")
            return
        
        file_path = self.storage_dir / f"{stock_code}.csv"
        columns = self.config["data"]["csv_columns"]
        
        # 确保列顺序一致
        df = df[[c for c in columns if c in df.columns]]
        
        if file_path.exists():
            # 追加模式，去重
            existing_df = pd.read_csv(file_path)
            # 按日期去重，保留最新
            df = pd.concat([existing_df, df], ignore_index=True)
            df = df.drop_duplicates(subset=["date", "code"], keep="last")
            df = df.sort_values("date").reset_index(drop=True)
        
        df.to_csv(file_path, index=False, encoding='utf-8')
        self.logger.info(f"已保存 {stock_code} 到 {file_path} ({len(df)} 条)")
    
    def _git_commit_push(self):
        """Git提交并推送"""
        try:
            github_config = self.config["github"]
            
            # 检查是否是git仓库
            if not Path(".git").exists():
                self.logger.info("非Git仓库，初始化...")
                subprocess.run(["git", "init"], check=True, capture_output=True)
                subprocess.run([
                    "git", "remote", "add", "origin", github_config["repo_url"]
                ], check=True, capture_output=True)
            
            # 配置Git用户（如未配置）
            try:
                subprocess.run(["git", "config", "user.email", "auto@update.local"],
                              check=True, capture_output=True)
                subprocess.run(["git", "config", "user.name", "Auto Updater"],
                              check=True, capture_output=True)
            except:
                pass
            
            # 添加所有更改
            self.logger.info("Git添加文件...")
            subprocess.run(["git", "add", "."], check=True, capture_output=True)
            
            # 检查是否有更改
            result = subprocess.run(
                ["git", "status", "--porcelain"],
                capture_output=True, text=True
            )
            
            if not result.stdout.strip():
                self.logger.info("无更改需要提交")
                return
            
            # 提交
            commit_msg = github_config["commit_message"].format(date=self.today)
            self.logger.info(f"Git提交: {commit_msg}")
            subprocess.run(["git", "commit", "-m", commit_msg], check=True, capture_output=True)
            
            # 推送
            self.logger.info("Git推送...")
            subprocess.run([
                "git", "push", "-u", "origin", github_config["branch"],
                "--force"
            ], check=True, capture_output=True, timeout=120)
            
            self.logger.info("Git推送成功!")
            
        except subprocess.TimeoutExpired:
            self.logger.error("Git推送超时")
        except subprocess.CalledProcessError as e:
            self.logger.error(f"Git操作失败: {e}")
        except Exception as e:
            self.logger.error(f"Git同步失败: {e}")
    
    def run(self):
        """执行数据更新"""
        self.logger.info("=" * 50)
        self.logger.info(f"开始执行数据更新 - {self.today}")
        self.logger.info("=" * 50)
        
        start_time = time.time()
        
        # 确保目录存在
        self._ensure_storage_dir()
        
        # 登录baostock
        self.logger.info("连接Baostock...")
        lg = bs.login()
        if lg.error_code != '0':
            self.logger.error(f"Baostock登录失败: {lg.error_msg}")
            return False
        
        try:
            # 获取日期范围
            start_date, end_date = self._get_date_range()
            
            if start_date is None or end_date is None:
                self.logger.info("数据已是最新，退出")
                return True
            
            # 获取每只股票数据
            success_count = 0
            fail_count = 0
            
            for i, stock in enumerate(self.stocks):
                self.logger.info(f"进度: {i+1}/{len(self.stocks)} - {stock}")
                
                # 主数据源：baostock
                df = self._fetch_data_baostock(stock, start_date, end_date)
                
                # 备用数据源：mootdx
                if df.empty:
                    self.logger.warning(f"{stock} baostock无数据，尝试mootdx...")
                    df = self._fetch_data_mootdx(stock, start_date, end_date)
                
                if not df.empty:
                    self._save_to_csv(df, stock)
                    success_count += 1
                else:
                    fail_count += 1
                
                # 避免请求过快
                time.sleep(0.5)
            
            self.logger.info(f"数据获取完成: 成功 {success_count}, 失败 {fail_count}")
            
            # Git同步
            self.logger.info("开始Git同步...")
            self._git_commit_push()
            
            elapsed = time.time() - start_time
            self.logger.info(f"更新完成! 耗时: {elapsed:.1f}秒")
            
            return fail_count == 0
            
        except Exception as e:
            self.logger.error(f"更新过程出错: {e}", exc_info=True)
            return False
        
        finally:
            # 登出baostock
            bs.logout()
            self.logger.info("Baostock已断开连接")


def is_market_day() -> bool:
    """检查是否是交易日"""
    today = datetime.now()
    
    # 周末不是交易日
    if today.weekday() >= 5:
        return False
    
    # 简化的节假日检查（建议使用akshare的is_trade_date）
    # 这里只检查周末，节假日可通过配置文件控制
    return True


def run_scheduler():
    """定时运行的主函数"""
    import schedule
    
    config_path = Path(__file__).parent / "config.json"
    
    with open(config_path, 'r', encoding='utf-8') as f:
        config = json.load(f)
    
    update_time = config["schedule"]["update_time"]
    timezone = config["schedule"]["timezone"]
    
    logger = logging.getLogger("StockDataUpdater")
    logger.info(f"定时任务已启动，将在每天 {update_time} ({timezone}) 执行")
    
    def job():
        """定时执行的任务"""
        if not is_market_day():
            logger.info("非交易日，跳过更新")
            return
        
        updater = StockDataUpdater()
        updater.run()
    
    # 立即执行一次（可选）
    # job()
    
    # 设置定时任务
    schedule.every().day.at(update_time).do(job)
    
    # 持续运行
    while True:
        schedule.run_pending()
        time.sleep(60)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="A股数据自动更新脚本")
    parser.add_argument("--config", "-c", default="config.json", help="配置文件路径")
    parser.add_argument("--schedule", "-s", action="store_true", help="以定时模式运行")
    parser.add_argument("--once", action="store_true", help="立即执行一次更新")
    
    args = parser.parse_args()
    
    if args.schedule:
        run_scheduler()
    elif args.once or len(sys.argv) == 1:
        updater = StockDataUpdater(args.config)
        success = updater.run()
        sys.exit(0 if success else 1)
    else:
        parser.print_help()

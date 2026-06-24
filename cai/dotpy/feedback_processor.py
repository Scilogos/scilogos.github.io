"""
feedback_processor.py v2.0 - 实盘反馈处理器 (日更版)
=====================================================
核心升级: 一条龙夜间维护 → 第二天模型无缝衔接

v1.0: 解析反馈 + 信号评估 + 参数校准
v2.0: + 增量数据更新 + 基准线更新 + 检查点保存/恢复 + 早间恢复

【日常使用流程】
  晚上:
    python feedback_processor.py --mode daily
    → 增量更新数据 → 解析反馈 → 校准参数 → 更新基准线 → 保存检查点 → 生成早间指令

  第二天早上:
    python adversarial_env.py --mode resume
    → 读检查点 → 读校准参数 → 读新基准线 → 快速热身 → 正常输出

  也可以分步:
    python feedback_processor.py --mode update-data       # 只增量更新数据
    python feedback_processor.py --mode update-benchmark   # 只更新基准线
    python feedback_processor.py --mode checkpoint-save    # 只保存检查点
    python feedback_processor.py --mode parse              # 只解析反馈
    python feedback_processor.py --mode evaluate           # 只评估信号
    python feedback_processor.py --mode calibrate          # 只校准参数
    python feedback_processor.py --mode report             # 只生成报告
    python feedback_processor.py --mode full               # 解析+评估+校准+报告(不含数据更新)
    python feedback_processor.py --mode daily              # ★ 一条龙夜间维护
    python feedback_processor.py --mode demo               # 生成示例反馈文件

反馈文件格式 (feedback.txt):
──────────────────────────────
# 每行一条记录，| 分隔，# 开头为注释
# 格式: 日期 | 方向 | 代码 | 价格 | 数量 | 盈亏 | 备注
# 方向: 买入/卖出
# 盈亏: 卖出时填实际盈亏金额，买入时填0
# 备注: 自由文字，比如止损/止盈/追高/抄底
2026-06-23 | 买入 | 600519.SH | 1800.50 | 100 | 0 | 茅台，系统信号看多
2026-06-24 | 卖出 | 600519.SH | 1835.20 | 100 | 3470 | 茅台止盈，信号正确
──────────────────────────────
"""

import os, sys, argparse, json, time, re, shutil
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from collections import defaultdict
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from stock_config import (
    BASE_DIR, SCRIPT_DIR, DATA_DIR, ADV_MODEL_DIR, ADV_DATA_DIR, RESULTS_DIR,
    FEEDBACK_FILE, FeedbackConfig, setup_logger,
)

logger = setup_logger("FeedbackProcessor")


# ============================================================
# 交易记录数据结构
# ============================================================
@dataclass
class TradeRecord:
    """单条实盘交易记录"""
    date: str
    side: str
    code: str
    price: float
    volume: int
    pnl: float
    note: str
    line_no: int = 0

    @property
    def is_buy(self) -> bool:
        return self.side.strip() == '买入'

    @property
    def is_sell(self) -> bool:
        return self.side.strip() == '卖出'

    @property
    def is_win(self) -> Optional[bool]:
        if self.is_sell:
            return self.pnl > 0
        return None


# ============================================================
# 反馈解析器
# ============================================================
class FeedbackParser:
    """解析 feedback.txt，格式: 日期 | 方向 | 代码 | 价格 | 数量 | 盈亏 | 备注"""

    def __init__(self, feedback_path: Path = FEEDBACK_FILE):
        self.feedback_path = Path(feedback_path)

    def parse(self) -> List[TradeRecord]:
        if not self.feedback_path.exists():
            logger.warning(f"反馈文件不存在: {self.feedback_path}")
            return []

        records = []
        with open(self.feedback_path, 'r', encoding='utf-8') as f:
            for line_no, line in enumerate(f, 1):
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                parts = [p.strip() for p in line.split('|')]
                if len(parts) < 6:
                    logger.warning(f"第{line_no}行格式错误(少于6列): {line}")
                    continue
                try:
                    record = TradeRecord(
                        date=parts[0], side=parts[1], code=parts[2],
                        price=float(parts[3]), volume=int(float(parts[4])),
                        pnl=float(parts[5]),
                        note=parts[6] if len(parts) > 6 else '',
                        line_no=line_no,
                    )
                    records.append(record)
                except (ValueError, IndexError) as e:
                    logger.warning(f"第{line_no}行解析失败: {e} | 原文: {line}")
        logger.info(f"解析完成: {len(records)}条交易记录")
        return records

    def pair_trades(self, records: List[TradeRecord]) -> List[Dict]:
        by_code = defaultdict(list)
        for r in records:
            by_code[r.code].append(r)
        pairs = []
        for code, trades in by_code.items():
            trades.sort(key=lambda x: x.date)
            buy_stack = []
            for t in trades:
                if t.is_buy:
                    buy_stack.append(t)
                elif t.is_sell and buy_stack:
                    buy = buy_stack.pop(0)
                    pnl = t.pnl
                    cost = buy.price * buy.volume
                    pnl_pct = pnl / (cost + 1e-8) if cost > 0 else 0
                    try:
                        d_buy = datetime.strptime(buy.date, '%Y-%m-%d')
                        d_sell = datetime.strptime(t.date, '%Y-%m-%d')
                        hold_days = (d_sell - d_buy).days
                    except ValueError:
                        hold_days = 0
                    pairs.append({
                        'code': code, 'buy_date': buy.date, 'sell_date': t.date,
                        'buy_price': buy.price, 'sell_price': t.price,
                        'volume': buy.volume, 'pnl': pnl, 'pnl_pct': pnl_pct,
                        'hold_days': hold_days, 'signal_correct': pnl > 0,
                        'buy_note': buy.note, 'sell_note': t.note,
                    })
        logger.info(f"配对完成: {len(pairs)}对完整交易")
        return pairs


# ============================================================
# 信号评估器
# ============================================================
class SignalEvaluator:
    """评估系统信号的实战表现"""

    def __init__(self, results_dir: Path = RESULTS_DIR):
        self.results_dir = Path(results_dir)

    def evaluate(self, trade_pairs: List[Dict]) -> Dict:
        if not trade_pairs:
            return {'status': 'no_data', 'message': '无完整交易对'}

        n = len(trade_pairs)
        wins = [p for p in trade_pairs if p['signal_correct']]
        losses = [p for p in trade_pairs if not p['signal_correct']]
        win_rate = len(wins) / n if n > 0 else 0
        total_pnl = sum(p['pnl'] for p in trade_pairs)
        avg_pnl_pct = np.mean([p['pnl_pct'] for p in trade_pairs])
        avg_win = np.mean([p['pnl'] for p in wins]) if wins else 0
        avg_loss = abs(np.mean([p['pnl'] for p in losses])) if losses else 1
        profit_loss_ratio = avg_win / (avg_loss + 1e-8)
        hold_days = [p['hold_days'] for p in trade_pairs]

        by_code = defaultdict(list)
        for p in trade_pairs:
            by_code[p['code']].append(p)
        code_performance = {}
        for code, pairs in by_code.items():
            code_wins = sum(1 for p in pairs if p['signal_correct'])
            code_performance[code] = {
                'trades': len(pairs), 'wins': code_wins,
                'win_rate': code_wins / len(pairs),
                'total_pnl': sum(p['pnl'] for p in pairs),
            }

        note_categories = defaultdict(list)
        for p in trade_pairs:
            sell_note = p.get('sell_note', '')
            for keyword in ['止损', '止盈', '追高', '抄底', '震仓']:
                if keyword in sell_note:
                    note_categories[keyword].append(p)
                    break
            else:
                note_categories['其他'].append(p)
        by_note = {}
        for cat, pairs in note_categories.items():
            cat_wins = sum(1 for p in pairs if p['signal_correct'])
            by_note[cat] = {
                'count': len(pairs), 'win_rate': cat_wins / len(pairs) if pairs else 0,
                'avg_pnl_pct': np.mean([p['pnl_pct'] for p in pairs]),
            }

        return {
            'win_rate': win_rate, 'wins': len(wins), 'losses': len(losses),
            'total_pnl': total_pnl, 'avg_pnl_pct': avg_pnl_pct,
            'profit_loss_ratio': profit_loss_ratio,
            'avg_hold_days': np.mean(hold_days) if hold_days else 0,
            'by_code': code_performance, 'by_note': by_note,
        }


# ============================================================
# 模型校准器
# ============================================================
class ModelCalibrator:
    """根据实战表现校准模型参数"""

    def __init__(self, cfg: FeedbackConfig = None):
        self.cfg = cfg or FeedbackConfig()

    def calibrate(self, evaluation: Dict) -> Dict:
        win_rate = evaluation.get('win_rate', 0.5)
        pnl_ratio = evaluation.get('profit_loss_ratio', 1.0)
        avg_pnl = evaluation.get('avg_pnl_pct', 0)
        adjustments = {}

        # 庄家
        if win_rate < 0.4:
            adjustments['dealer'] = {
                'info_power_delta': 0.05, 'shake_intensity_delta': 0.01,
                'reason': f'胜率偏低({win_rate:.0%})，庄家信息优势不足，增强庄家'}
        elif win_rate > 0.65:
            adjustments['dealer'] = {
                'info_power_delta': -0.03, 'shake_intensity_delta': -0.005,
                'reason': f'胜率偏高({win_rate:.0%})，庄家过强，适度削弱'}
        else:
            adjustments['dealer'] = {
                'info_power_delta': 0, 'shake_intensity_delta': 0,
                'reason': f'胜率正常({win_rate:.0%})，庄家参数维持'}

        # 散户
        if pnl_ratio < 1.0:
            adjustments['retailer'] = {
                'herd_prob_delta': -0.05, 'value_prob_delta': 0.05,
                'reason': f'盈亏比偏低({pnl_ratio:.2f})，减少散户从众，增加价值型'}
        elif pnl_ratio > 2.0:
            adjustments['retailer'] = {
                'herd_prob_delta': 0.03, 'value_prob_delta': -0.02,
                'reason': f'盈亏比偏高({pnl_ratio:.2f})，适度增加散户从众压力'}
        else:
            adjustments['retailer'] = {
                'herd_prob_delta': 0, 'value_prob_delta': 0,
                'reason': f'盈亏比正常({pnl_ratio:.2f})，散户参数维持'}

        # 奖励
        if avg_pnl < -0.02:
            adjustments['reward'] = {
                'reward_shaping_weight_delta': 0.1,
                'reason': f'平均盈亏偏低({avg_pnl:.2%})，加强奖励塑形'}
        elif avg_pnl > 0.05:
            adjustments['reward'] = {
                'reward_shaping_weight_delta': -0.05,
                'reason': f'平均盈亏偏高({avg_pnl:.2%})，减弱奖励塑形防过拟合'}
        else:
            adjustments['reward'] = {
                'reward_shaping_weight_delta': 0,
                'reason': f'平均盈亏正常({avg_pnl:.2%})，奖励参数维持'}

        # 风控
        max_dd_delta = -0.02 if win_rate < 0.35 else (0.02 if win_rate > 0.7 else 0)
        sig_conf_delta = 0.05 if pnl_ratio < 0.8 else (-0.03 if pnl_ratio > 2.5 else 0)
        adjustments['risk'] = {
            'max_drawdown_delta': max_dd_delta,
            'signal_conf_thresh_delta': sig_conf_delta,
            'reason': '根据胜率和盈亏比调整风控阈值'}

        need_retrain = win_rate < 0.3 or pnl_ratio < 0.5
        retrain_reason = ''
        if need_retrain:
            retrain_reason = f'胜率{win_rate:.0%}或盈亏比{pnl_ratio:.2f}过低，建议增加训练轮次'

        return {
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'evaluation_summary': {
                'win_rate': win_rate, 'profit_loss_ratio': pnl_ratio,
                'avg_pnl_pct': avg_pnl,
            },
            'adjustments': adjustments,
            'need_retrain': need_retrain,
            'retrain_reason': retrain_reason,
        }

    def apply_calibrations(self, calibrations: Dict) -> Dict:
        adj = calibrations.get('adjustments', {})
        changes = []
        if 'dealer' in adj:
            d = adj['dealer']
            changes.append(f"庄家: info_power{d.get('info_power_delta',0):+.2f}, shake{d.get('shake_intensity_delta',0):+.2f}")
        if 'retailer' in adj:
            r = adj['retailer']
            changes.append(f"散户: herd{r.get('herd_prob_delta',0):+.2f}, value{r.get('value_prob_delta',0):+.2f}")
        if 'reward' in adj:
            rw = adj['reward']
            changes.append(f"奖励: shaping_weight{rw.get('reward_shaping_weight_delta',0):+.2f}")
        if 'risk' in adj:
            rk = adj['risk']
            changes.append(f"风控: 回撤{rk.get('max_drawdown_delta',0):+.2f}, 置信度{rk.get('signal_conf_thresh_delta',0):+.2f}")

        cal_file = Path(self.cfg.results_dir) / "calibration_params.json"
        with open(cal_file, 'w', encoding='utf-8') as f:
            json.dump(calibrations, f, ensure_ascii=False, indent=2)
        logger.info(f"校准参数已保存: {cal_file}")
        logger.info(f"变更: {'; '.join(changes)}")
        return {
            'applied': True, 'calibration_file': str(cal_file),
            'changes': changes, 'need_retrain': calibrations.get('need_retrain', False),
        }


# ============================================================
# 校准报告生成
# ============================================================
class CalibrationReporter:
    def generate(self, evaluation: Dict, calibrations: Dict, apply_result: Dict) -> str:
        lines = ["# 实盘反馈校准报告", f"生成时间: {time.strftime('%Y-%m-%d %H:%M:%S')}", ""]
        wr = evaluation.get('win_rate', 0)
        emoji = "🟢" if wr >= 0.5 else "🔴"
        lines += [
            "## 实战表现",
            f"- {emoji} **胜率**: {wr:.1%} ({evaluation.get('wins',0)}胜/{evaluation.get('losses',0)}负)",
            f"- **总盈亏**: {evaluation.get('total_pnl',0):,.0f} 元",
            f"- **平均盈亏率**: {evaluation.get('avg_pnl_pct',0):.2%}",
            f"- **盈亏比**: {evaluation.get('profit_loss_ratio',0):.2f}",
            f"- **平均持仓**: {evaluation.get('avg_hold_days',0):.1f} 天", "",
        ]
        by_code = evaluation.get('by_code', {})
        if by_code:
            lines += ["## 个股表现", "| 代码 | 交易数 | 胜率 | 总盈亏 |", "|------|--------|------|--------|"]
            for code, perf in sorted(by_code.items(), key=lambda x: x[1]['total_pnl'], reverse=True):
                e = "🟢" if perf['total_pnl'] >= 0 else "🔴"
                lines.append(f"| {code} | {perf['trades']} | {perf['win_rate']:.0%} | {e}{perf['total_pnl']:,.0f} |")
            lines.append("")

        adj = calibrations.get('adjustments', {})
        if adj:
            lines.append("## 校准建议")
            cn = {'dealer': '庄家', 'retailer': '散户', 'reward': '奖励', 'risk': '风控'}
            for category, params in adj.items():
                lines.append(f"### {cn.get(category, category)}")
                lines.append(f"- {params.get('reason', '')}")
                for k, v in params.items():
                    if k != 'reason' and 'delta' in k:
                        lines.append(f"  - {k}: {v:+}")
                lines.append("")
        else:
            lines += ["## 校准建议", "当前参数无需调整。", ""]

        if calibrations.get('need_retrain'):
            lines += ["## ⚠️ 重训建议", f"> {calibrations.get('retrain_reason', '')}", ""]

        if apply_result:
            lines += [
                "## 校准应用",
                f"- 状态: {'✓ 已应用' if apply_result.get('applied') else '✗ 未应用'}",
                f"- 参数文件: `{apply_result.get('calibration_file', '')}`",
            ]
            for c in apply_result.get('changes', []):
                lines.append(f"- {c}")
            lines.append("")
        return "\n".join(lines)


# ============================================================
# ★ v2.0: 增量数据更新器
# ============================================================
class IncrementalUpdater:
    """
    增量更新A股日K线数据

    工作原理:
      1. 扫描stockdata目录，找出每只股票最后一条数据的日期
      2. 从baostock拉取该日期到今天的新数据
      3. 追加写入CSV，不覆盖历史数据
    """

    def __init__(self, data_dir: Path = DATA_DIR):
        self.data_dir = Path(data_dir)
        self.log_file = self.data_dir / "incremental_update_log.json"

    def scan_latest_dates(self) -> Dict[str, str]:
        """扫描所有日K线CSV，获取每只股票的最后日期"""
        latest = {}
        csv_files = list(self.data_dir.glob("*_daily.csv"))
        logger.info(f"扫描{len(csv_files)}只股票的最新日期...")
        for i, csv_path in enumerate(csv_files):
            try:
                with open(csv_path, 'r', encoding='utf-8') as f:
                    lines = f.readlines()
                if len(lines) < 2:
                    continue
                last_line = lines[-1].strip()
                if not last_line:
                    last_line = lines[-2].strip()
                if last_line:
                    date = last_line.split(',')[0].strip()
                    parts = csv_path.stem.split('_')
                    if len(parts) >= 2:
                        code = parts[0] + '_' + parts[1]
                        latest[code] = date
            except Exception:
                continue
            if (i + 1) % 500 == 0:
                logger.info(f"  扫描进度: {i+1}/{len(csv_files)}")
        logger.info(f"扫描完成: {len(latest)}只股票有历史数据")
        return latest

    def update_incremental(self, stock_list_file: Path = None) -> Dict:
        try:
            import baostock as bs
        except ImportError:
            logger.error("baostock未安装，无法增量更新")
            return {'status': 'error', 'message': 'baostock未安装'}

        sl_file = stock_list_file or self.data_dir / "stock_list.json"
        if not sl_file.exists():
            logger.error(f"股票列表不存在: {sl_file}")
            return {'status': 'error', 'message': 'stock_list.json不存在'}

        with open(sl_file, 'r', encoding='utf-8') as f:
            stock_list = json.load(f)

        latest_dates = self.scan_latest_dates()

        today = datetime.now()
        if today.weekday() >= 5:
            today = today - timedelta(days=today.weekday() - 4)
        end_date = today.strftime('%Y-%m-%d')

        lg = bs.login()
        if lg.error_code != '0':
            logger.error(f"baostock登录失败: {lg.error_msg}")
            return {'status': 'error', 'message': f'baostock登录失败: {lg.error_msg}'}

        updated, skipped, failed = 0, 0, 0
        total = len(stock_list)
        logger.info(f"开始增量更新: {total}只股票, 目标截止日期={end_date}")

        for i, stk in enumerate(stock_list):
            std_code = stk.get('std_code', '')
            bs_code = stk.get('code', '')
            code_part = std_code.replace('.', '_')
            csv_path = self.data_dir / f"{code_part}_daily.csv"

            last_date = latest_dates.get(code_part, '')
            if last_date:
                try:
                    start_dt = datetime.strptime(last_date, '%Y-%m-%d') + timedelta(days=1)
                    start_date = start_dt.strftime('%Y-%m-%d')
                except ValueError:
                    start_date = '2020-01-01'
            else:
                start_date = '2020-01-01'

            if start_date > end_date:
                skipped += 1
                continue

            try:
                rs = bs.query_history_k_data_plus(
                    bs_code,
                    "date,code,open,high,low,close,preclose,volume,amount,turn,pctChg",
                    start_date=start_date, end_date=end_date,
                    frequency="d", adjustflag="2")
                new_rows = []
                while rs.next():
                    new_rows.append(rs.get_row_data())
                if new_rows:
                    new_df = pd.DataFrame(new_rows, columns=[
                        'date', 'code', 'open', 'high', 'low', 'close',
                        'preclose', 'volume', 'amount', 'turn', 'pctChg'])
                    for col in ['open', 'high', 'low', 'close', 'preclose',
                                'volume', 'amount', 'turn', 'pctChg']:
                        new_df[col] = pd.to_numeric(new_df[col], errors='coerce')
                    header = not csv_path.exists()
                    new_df.to_csv(csv_path, mode='a', header=header, index=False, encoding='utf-8')
                    updated += 1
                else:
                    skipped += 1
            except Exception as e:
                failed += 1
                if failed <= 5:
                    logger.warning(f"更新失败 {std_code}: {e}")
            if (i + 1) % 500 == 0:
                logger.info(f"  进度: {i+1}/{total} | 更新{updated} 跳过{skipped} 失败{failed}")

        bs.logout()
        result = {
            'status': 'ok', 'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'total': total, 'updated': updated, 'skipped': skipped,
            'failed': failed, 'end_date': end_date,
        }
        with open(self.log_file, 'w', encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        logger.info(f"增量更新完成: 更新{updated} 跳过{skipped} 失败{failed}")
        return result


# ============================================================
# ★ v2.0: 基准线更新器
# ============================================================
class BenchmarkUpdater:
    """
    更新价格基准线 (benchmark .npy)

    用途: adversarial_env.py 读取 price_benchmark 做实盘校准
    流程:
      1. 从stockdata读取最新收盘价序列 (上证指数/指定股票)
      2. 生成/更新 benchmark .npy 文件
    """

    def __init__(self, data_dir: Path = DATA_DIR, adv_data_dir: Path = ADV_DATA_DIR):
        self.data_dir = Path(data_dir)
        self.adv_data_dir = Path(adv_data_dir)

    def build_benchmark_from_index(self, index_code: str = "sh.000001",
                                   start_date: str = "2024-01-01") -> Optional[np.ndarray]:
        """从指数构建基准线 (上证指数)"""
        # 尝试从本地CSV读取
        csv_path = self.data_dir / f"{index_code.replace('.', '_')}_daily.csv"
        if csv_path.exists():
            logger.info(f"从本地CSV加载指数数据: {csv_path}")
            df = pd.read_csv(csv_path)
            close = pd.to_numeric(df['close'], errors='coerce').dropna().values
        else:
            # 在线拉取
            try:
                import baostock as bs
            except ImportError:
                logger.error("baostock未安装")
                return None
            lg = bs.login()
            if lg.error_code != '0':
                logger.error("baostock登录失败")
                return None
            rs = bs.query_history_k_data_plus(
                index_code, "date,close",
                start_date=start_date,
                end_date=datetime.now().strftime('%Y-%m-%d'),
                frequency="d")
            rows = []
            while rs.next():
                rows.append(rs.get_row_data())
            bs.logout()
            if not rows:
                logger.error("未获取到指数数据")
                return None
            df = pd.DataFrame(rows, columns=['date', 'close'])
            close = pd.to_numeric(df['close'], errors='coerce').dropna().values

        if len(close) < 10:
            logger.error(f"指数数据太短: {len(close)}条")
            return None

        # 归一化到10元起点 (与MarketEnv的initial_price对齐)
        benchmark = close / close[0] * 10.0
        logger.info(f"基准线构建完成: {len(benchmark)}条, 起点={benchmark[0]:.2f}, 终点={benchmark[-1]:.2f}")
        return benchmark

    def build_benchmark_from_stock(self, code: str = "sh.600519") -> Optional[np.ndarray]:
        """从单只股票构建基准线"""
        csv_name = code.replace('.', '_') + '_daily.csv'
        csv_path = self.data_dir / csv_name
        if not csv_path.exists():
            logger.error(f"股票数据不存在: {csv_path}")
            return None
        df = pd.read_csv(csv_path)
        close = pd.to_numeric(df['close'], errors='coerce').dropna().values
        if len(close) < 10:
            return None
        benchmark = close / close[0] * 10.0
        logger.info(f"股票基准线构建完成: {code}, {len(benchmark)}条")
        return benchmark

    def save_benchmark(self, benchmark: np.ndarray, name: str = "price_benchmark") -> Path:
        self.adv_data_dir.mkdir(parents=True, exist_ok=True)
        out_path = self.adv_data_dir / f"{name}.npy"
        np.save(str(out_path), benchmark)
        logger.info(f"基准线已保存: {out_path} | 形状={benchmark.shape}")
        return out_path

    def update_benchmark(self, name: str = "price_benchmark",
                         index_code: str = "sh.000001") -> Dict:
        """更新基准线 (增量或全量)"""
        benchmark_path = self.adv_data_dir / f"{name}.npy"
        if benchmark_path.exists():
            old_benchmark = np.load(str(benchmark_path))
            logger.info(f"现有基准线: {len(old_benchmark)}条")
            new_benchmark = self.build_benchmark_from_index(index_code)
            if new_benchmark is not None and len(new_benchmark) > len(old_benchmark):
                self.save_benchmark(new_benchmark, name)
                added = len(new_benchmark) - len(old_benchmark)
                logger.info(f"基准线已更新: {len(old_benchmark)}→{len(new_benchmark)}条 (+{added})")
                return {'status': 'updated', 'old_len': len(old_benchmark),
                        'new_len': len(new_benchmark), 'added': added}
            else:
                logger.info("基准线无需更新 (已是最新)")
                return {'status': 'uptodate', 'len': len(old_benchmark)}
        else:
            benchmark = self.build_benchmark_from_index(index_code)
            if benchmark is not None:
                self.save_benchmark(benchmark, name)
                return {'status': 'created', 'len': len(benchmark)}
            return {'status': 'error', 'message': '无法构建基准线'}


# ============================================================
# ★ v2.0: 检查点管理器
# ============================================================
class CheckpointManager:
    """
    训练检查点管理: 保存/恢复完整训练状态

    保存内容:
      - 对抗模型权重 (dealer + retailers + hotmoney)
      - 训练器状态 (generation, 参数, 统计)
      - 校准参数 (calibration_params.json)
      - 基准线路径 (benchmark来源)
      - 训练进度 (已跑episodes数)
    """

    def __init__(self, model_dir: Path = ADV_MODEL_DIR, results_dir: Path = RESULTS_DIR):
        self.model_dir = Path(model_dir)
        self.results_dir = Path(results_dir)

    def save_checkpoint(self, trainer_state: Dict = None,
                        benchmark_path: str = "",
                        benchmark_source: str = "",
                        episodes_done: int = 0) -> Dict:
        import torch
        checkpoint_dir = self.model_dir / "checkpoint"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # 1. 模型权重
        model_src = self.model_dir / "adversarial_model.pt"
        if model_src.exists():
            shutil.copy2(model_src, checkpoint_dir / "adversarial_model.pt")
            logger.info(f"  模型权重已备份")

        # 2. 组合模型
        combo_dir = self.model_dir / "combinations"
        if combo_dir.exists():
            combo_cp = checkpoint_dir / "combinations"
            if combo_cp.exists():
                shutil.rmtree(combo_cp)
            shutil.copytree(combo_dir, combo_cp)
            logger.info(f"  组合模型已备份")

        # 3. 训练器状态
        if trainer_state:
            with open(checkpoint_dir / "trainer_state.json", 'w', encoding='utf-8') as f:
                json.dump(trainer_state, f, ensure_ascii=False, indent=2)

        # 4. 校准参数
        cal_src = self.results_dir / "calibration_params.json"
        if cal_src.exists():
            shutil.copy2(cal_src, checkpoint_dir / "calibration_params.json")

        # 5. 元信息
        meta = {
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'episodes_done': episodes_done,
            'benchmark_path': str(benchmark_path),
            'benchmark_source': benchmark_source,
            'trainer_state_file': 'trainer_state.json',
            'model_file': 'adversarial_model.pt',
            'calibration_file': 'calibration_params.json',
        }
        with open(checkpoint_dir / "checkpoint_meta.json", 'w', encoding='utf-8') as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)

        logger.info(f"检查点已保存: {checkpoint_dir}")
        return {'status': 'saved', 'path': str(checkpoint_dir), 'meta': meta}

    def load_checkpoint(self) -> Optional[Dict]:
        """加载检查点元信息 (不含模型权重，由adversarial_env自行加载)"""
        meta_path = self.model_dir / "checkpoint" / "checkpoint_meta.json"
        if not meta_path.exists():
            logger.info("无检查点")
            return None
        with open(meta_path, 'r', encoding='utf-8') as f:
            meta = json.load(f)
        trainer_state_path = self.model_dir / "checkpoint" / "trainer_state.json"
        trainer_state = None
        if trainer_state_path.exists():
            with open(trainer_state_path, 'r', encoding='utf-8') as f:
                trainer_state = json.load(f)
        cal_path = self.model_dir / "checkpoint" / "calibration_params.json"
        calibration = None
        if cal_path.exists():
            with open(cal_path, 'r', encoding='utf-8') as f:
                calibration = json.load(f)
        return {'meta': meta, 'trainer_state': trainer_state, 'calibration': calibration}


# ============================================================
# ★ v2.0: 早间恢复计划器
# ============================================================
class MorningPlanner:
    """
    生成早间恢复指令

    读取检查点+校准参数+基准线信息，生成morning_resume.json
    adversarial_env.py --mode resume 会读取这个文件自动恢复
    """

    def __init__(self, results_dir: Path = RESULTS_DIR, adv_data_dir: Path = ADV_DATA_DIR):
        self.results_dir = Path(results_dir)
        self.adv_data_dir = Path(adv_data_dir)

    def generate_resume_plan(self, checkpoint_meta: Dict = None,
                             calibration: Dict = None,
                             benchmark_update: Dict = None) -> Dict:
        # 读取校准参数
        if calibration is None:
            cal_path = self.results_dir / "calibration_params.json"
            if cal_path.exists():
                with open(cal_path, 'r', encoding='utf-8') as f:
                    calibration = json.load(f)

        # 读取训练器状态
        trainer_state_path = self.results_dir / "trainer_state.json"
        trainer_state = None
        if trainer_state_path.exists():
            with open(trainer_state_path, 'r', encoding='utf-8') as f:
                trainer_state = json.load(f)

        # 检查基准线
        benchmark_path = None
        benchmark_source = ""
        for name in ["price_benchmark.npy", "generated_1000.npy"]:
            p = self.adv_data_dir / name
            if p.exists():
                benchmark_path = str(p)
                benchmark_source = "real" if "benchmark" in name else "fake"
                break

        # 热身episodes
        warmup_episodes = 50
        if calibration and calibration.get('adjustments'):
            has_change = any(
                any('delta' in k and v != 0 for k, v in params.items() if k != 'reason')
                for params in calibration['adjustments'].values()
            )
            if has_change:
                warmup_episodes = 100

        # 数据更新日志
        data_update_log = None
        update_log_path = self.adv_data_dir.parent / "stockdata" / "incremental_update_log.json"
        if update_log_path.exists():
            with open(update_log_path, 'r', encoding='utf-8') as f:
                data_update_log = json.load(f)

        plan = {
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'plan_type': 'morning_resume',
            'model_path': str(ADV_MODEL_DIR / "adversarial_model.pt"),
            'checkpoint_path': str(ADV_MODEL_DIR / "checkpoint"),
            'episodes_done': checkpoint_meta.get('episodes_done', 0) if checkpoint_meta else 0,
            'benchmark_path': benchmark_path or "",
            'benchmark_source': benchmark_source,
            'benchmark_update': benchmark_update or {},
            'calibration_file': str(self.results_dir / "calibration_params.json"),
            'calibration': calibration,
            'trainer_state': trainer_state,
            'warmup_episodes': warmup_episodes,
            'resume_training': True,
            'evolve': True,
            'data_update_log': data_update_log,
        }

        resume_path = self.results_dir / "morning_resume.json"
        with open(resume_path, 'w', encoding='utf-8') as f:
            json.dump(plan, f, ensure_ascii=False, indent=2, default=str)

        logger.info(f"早间恢复计划已保存: {resume_path}")
        logger.info(f"  热身episodes: {warmup_episodes}")
        logger.info(f"  基准线: {benchmark_source or '无'}")
        logger.info(f"  校准变更: {'有' if calibration and calibration.get('adjustments') else '无'}")
        return plan


# ============================================================
# ★ v2.0: 日更一条龙
# ============================================================
class DailyRoutine:
    """
    夜间一条龙维护

    依次执行:
      1. 增量数据更新 (baostock拉最新)
      2. 解析反馈 (feedback.txt)
      3. 评估信号表现
      4. 校准模型参数
      5. 更新价格基准线
      6. 保存检查点
      7. 生成早间恢复计划

    执行完后，第二天早上只需:
      python adversarial_env.py --mode resume
    """

    def __init__(self, data_dir: Path = DATA_DIR, adv_data_dir: Path = ADV_DATA_DIR,
                 model_dir: Path = ADV_MODEL_DIR, results_dir: Path = RESULTS_DIR,
                 feedback_file: Path = FEEDBACK_FILE):
        self.data_dir = Path(data_dir)
        self.adv_data_dir = Path(adv_data_dir)
        self.model_dir = Path(model_dir)
        self.results_dir = Path(results_dir)
        self.feedback_file = Path(feedback_file)

    def run(self, skip_data_update: bool = False, skip_feedback: bool = False) -> Dict:
        logger.info(f"\n{'★'*60}")
        logger.info(f"  夜间一条龙维护 启动")
        logger.info(f"  时间: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info(f"{'★'*60}")
        results = {}

        # Step 1: 增量数据更新
        if not skip_data_update:
            logger.info(f"\n{'─'*40}\nStep 1/7: 增量数据更新\n{'─'*40}")
            updater = IncrementalUpdater(self.data_dir)
            results['data_update'] = updater.update_incremental()
        else:
            results['data_update'] = {'status': 'skipped'}
            logger.info("Step 1/7: 跳过数据更新")

        # Step 2-4: 反馈解析+评估+校准
        if not skip_feedback and self.feedback_file.exists():
            logger.info(f"\n{'─'*40}\nStep 2/7: 解析反馈\n{'─'*40}")
            parser = FeedbackParser(self.feedback_file)
            records = parser.parse()
            pairs = parser.pair_trades(records)
            results['parse'] = {'records': len(records), 'pairs': len(pairs)}

            logger.info(f"\n{'─'*40}\nStep 3/7: 评估信号表现\n{'─'*40}")
            evaluator = SignalEvaluator(self.results_dir)
            evaluation = evaluator.evaluate(pairs)
            results['evaluation'] = evaluation

            logger.info(f"\n{'─'*40}\nStep 4/7: 校准模型参数\n{'─'*40}")
            calibrator = ModelCalibrator()
            calibrations = calibrator.calibrate(evaluation)
            apply_result = calibrator.apply_calibrations(calibrations)
            results['calibration'] = {'calibrations': calibrations, 'apply_result': apply_result}
        else:
            results['parse'] = {'status': 'skipped'}
            results['evaluation'] = {'status': 'skipped'}
            results['calibration'] = {'status': 'skipped'}
            logger.info("Step 2-4/7: 跳过反馈 (无反馈文件或指定跳过)")

        # Step 5: 更新基准线
        logger.info(f"\n{'─'*40}\nStep 5/7: 更新价格基准线\n{'─'*40}")
        bench_updater = BenchmarkUpdater(self.data_dir, self.adv_data_dir)
        bench_result = bench_updater.update_benchmark()
        results['benchmark_update'] = bench_result

        # Step 6: 保存检查点
        logger.info(f"\n{'─'*40}\nStep 6/7: 保存检查点\n{'─'*40}")
        ckpt_mgr = CheckpointManager(self.model_dir, self.results_dir)
        trainer_state = None
        ts_path = self.results_dir / "trainer_state.json"
        if ts_path.exists():
            with open(ts_path, 'r', encoding='utf-8') as f:
                trainer_state = json.load(f)
        benchmark_path = ""
        for name in ["price_benchmark.npy", "generated_1000.npy"]:
            p = self.adv_data_dir / name
            if p.exists():
                benchmark_path = str(p)
                break
        ckpt_result = ckpt_mgr.save_checkpoint(
            trainer_state=trainer_state,
            benchmark_path=benchmark_path,
            benchmark_source="real" if "benchmark" in benchmark_path else "fake",
            episodes_done=trainer_state.get('episodes_done', 0) if trainer_state else 0,
        )
        results['checkpoint'] = ckpt_result

        # Step 7: 早间恢复计划
        logger.info(f"\n{'─'*40}\nStep 7/7: 生成早间恢复计划\n{'─'*40}")
        planner = MorningPlanner(self.results_dir, self.adv_data_dir)
        cal_data = None
        cal_path = self.results_dir / "calibration_params.json"
        if cal_path.exists():
            with open(cal_path, 'r', encoding='utf-8') as f:
                cal_data = json.load(f)
        resume_plan = planner.generate_resume_plan(
            checkpoint_meta=ckpt_result.get('meta'),
            calibration=cal_data,
            benchmark_update=bench_result,
        )
        results['resume_plan'] = resume_plan

        # 完成
        logger.info(f"\n{'★'*60}")
        logger.info(f"  夜间维护完成!")
        logger.info(f"  数据更新: {results['data_update'].get('status', 'N/A')}")
        logger.info(f"  基准线更新: {bench_result.get('status', 'N/A')}")
        logger.info(f"  检查点: {ckpt_result.get('status', 'N/A')}")
        logger.info(f"  早间恢复计划: {self.results_dir / 'morning_resume.json'}")
        logger.info(f"")
        logger.info(f"  明天早上运行:")
        logger.info(f"    python adversarial_env.py --mode resume")
        logger.info(f"{'★'*60}")

        daily_log = self.results_dir / "daily_routine_log.json"
        with open(daily_log, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2, default=str)
        return results


# ============================================================
# 命令行入口
# ============================================================
def main():
    parser = argparse.ArgumentParser(
        description="实盘反馈处理器 v2.0 (日更版)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python feedback_processor.py --mode daily              # 夜间一条龙
  python feedback_processor.py --mode update-data        # 只增量更新数据
  python feedback_processor.py --mode update-benchmark   # 只更新基准线
  python feedback_processor.py --mode full               # 反馈全流程(不含数据更新)
  python feedback_processor.py --mode demo               # 生成示例反馈文件
""")
    parser.add_argument("--mode", required=True,
                        choices=["parse", "evaluate", "calibrate", "report", "full",
                                  "demo", "daily", "update-data", "update-benchmark",
                                  "checkpoint-save", "checkpoint-load"])
    parser.add_argument("--feedback-file", type=str, default=str(FEEDBACK_FILE))
    parser.add_argument("--results-dir", type=str, default=str(RESULTS_DIR))
    parser.add_argument("--data-dir", type=str, default=str(DATA_DIR))
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--skip-data-update", action="store_true")
    parser.add_argument("--skip-feedback", action="store_true")
    args = parser.parse_args()
    results_dir = Path(args.results_dir)
    data_dir = Path(args.data_dir)

    if args.mode == "demo":
        demo_content = """# 实盘反馈 - 格式: 日期 | 方向 | 代码 | 价格 | 数量 | 盈亏 | 备注
# 方向: 买入/卖出
# 盈亏: 卖出时填实际盈亏金额，买入时填0
# 备注: 自由文字(止损/止盈/追高/抄底等)
2026-06-23 | 买入 | 600519.SH | 1800.50 | 100 | 0 | 茅台，系统看多
2026-06-24 | 卖出 | 600519.SH | 1835.20 | 100 | 3470 | 茅台止盈
2026-06-25 | 买入 | 000001.SZ | 12.35 | 1000 | 0 | 平安银行，信号看多
2026-06-26 | 卖出 | 000001.SZ | 11.80 | 1000 | -550 | 止损出局
"""
        out = Path(args.feedback_file)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, 'w', encoding='utf-8') as f:
            f.write(demo_content)
        logger.info(f"示例反馈文件已生成: {out}")
        return

    if args.mode == "daily":
        routine = DailyRoutine(
            data_dir=data_dir, results_dir=results_dir,
            feedback_file=Path(args.feedback_file))
        routine.run(skip_data_update=args.skip_data_update, skip_feedback=args.skip_feedback)
        return

    if args.mode == "update-data":
        updater = IncrementalUpdater(data_dir)
        result = updater.update_incremental()
        print(json.dumps(result, ensure_ascii=False, indent=2, default=str))
        return

    if args.mode == "update-benchmark":
        bench = BenchmarkUpdater(data_dir)
        result = bench.update_benchmark()
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return

    if args.mode == "checkpoint-save":
        ckpt = CheckpointManager(ADV_MODEL_DIR, results_dir)
        ts_path = results_dir / "trainer_state.json"
        trainer_state = None
        if ts_path.exists():
            with open(ts_path, 'r', encoding='utf-8') as f:
                trainer_state = json.load(f)
        result = ckpt.save_checkpoint(trainer_state=trainer_state)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return

    if args.mode == "checkpoint-load":
        ckpt = CheckpointManager(ADV_MODEL_DIR, results_dir)
        result = ckpt.load_checkpoint()
        if result:
            print(json.dumps(result['meta'], ensure_ascii=False, indent=2))
        else:
            print("无检查点")
        return

    # ── v1.0原有模式 ──
    fp = FeedbackParser(Path(args.feedback_file))
    records = fp.parse()
    if not records and args.mode in ["parse", "evaluate", "calibrate", "report", "full"]:
        logger.error("无交易记录，请检查反馈文件")
        return

    if args.mode == "parse":
        pairs = fp.pair_trades(records)
        for p in pairs:
            print(f"  {p['code']}: {p['buy_date']}@{p['buy_price']:.2f} -> "
                  f"{p['sell_date']}@{p['sell_price']:.2f} | "
                  f"盈亏{p['pnl']:+,.0f} ({p['pnl_pct']:+.2%}) | "
                  f"{'OK' if p['signal_correct'] else 'X'} {p['sell_note']}")
        return

    pairs = fp.pair_trades(records)
    evaluator = SignalEvaluator(results_dir)
    evaluation = evaluator.evaluate(pairs)

    if args.mode == "evaluate":
        print(json.dumps(evaluation, ensure_ascii=False, indent=2))
        return

    calibrator = ModelCalibrator()
    calibrations = calibrator.calibrate(evaluation)
    if args.mode == "calibrate":
        print(json.dumps(calibrations, ensure_ascii=False, indent=2))
        return

    apply_result = calibrator.apply_calibrations(calibrations)
    if args.mode in ["report", "full"]:
        reporter = CalibrationReporter()
        report_md = reporter.generate(evaluation, calibrations, apply_result)
        out_path = Path(args.output) if args.output else results_dir / "calibration_report.md"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, 'w', encoding='utf-8') as f:
            f.write(report_md)
        json_path = out_path.with_suffix('.json')
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump({'evaluation': evaluation, 'calibrations': calibrations,
                       'apply_result': apply_result}, f, ensure_ascii=False, indent=2, default=str)
        logger.info(f"校准报告已保存: {out_path}")
        print(report_md)


if __name__ == "__main__":
    main()

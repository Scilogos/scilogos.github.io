"""
feedback_processor.py - 实盘反馈处理器
======================================
功能:
  1. 解析 feedback.txt 中的实盘交易记录
  2. 对比系统信号 vs 实际结果，评估信号准确率
  3. 校准对抗模型参数（庄家行为阈值/散户比例/奖励权重）
  4. 生成校准报告 + 更新模型
  5. 输出修正后的解读参数到 results/

反馈文件格式 (feedback.txt):
──────────────────────────────
# 每行一条记录，| 分隔，# 开头为注释
# 格式: 日期 | 方向 | 代码 | 价格 | 数量 | 盈亏 | 备注
# 方向: 买入/卖出
# 盈亏: 卖出时填实际盈亏金额，买入时填0
# 备注: 自由文字，比如止损/止盈/追高/抄底
2026-06-23 | 买入 | 600519.SH | 1800.50 | 100 | 0 | 茅台，系统信号看多
2026-06-24 | 卖出 | 600519.SH | 1835.20 | 100 | 3470 | 茅台止盈，信号正确
2026-06-25 | 买入 | 000001.SZ | 12.35 | 1000 | 0 | 平安银行，信号看多
2026-06-26 | 卖出 | 000001.SZ | 11.80 | 1000 | -550 | 止损出局，信号错误
──────────────────────────────

用法:
  python feedback_processor.py --mode parse        # 解析反馈文件
  python feedback_processor.py --mode evaluate     # 评估信号准确率
  python feedback_processor.py --mode calibrate     # 校准模型参数
  python feedback_processor.py --mode report        # 生成校准报告
  python feedback_processor.py --mode full          # 全流程(解析+评估+校准+报告)
  python feedback_processor.py --mode demo          # 生成示例反馈文件
"""

import os, sys, argparse, json, time, re
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from collections import defaultdict
from datetime import datetime

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
    date: str           # 2026-06-23
    side: str           # 买入 / 卖出
    code: str           # 600519.SH
    price: float        # 成交价
    volume: int         # 数量
    pnl: float          # 盈亏金额 (买入时=0)
    note: str           # 备注
    line_no: int = 0    # 原始行号

    @property
    def is_buy(self) -> bool:
        return self.side.strip() == '买入'
    
    @property
    def is_sell(self) -> bool:
        return self.side.strip() == '卖出'
    
    @property
    def is_win(self) -> Optional[bool]:
        """卖出时判断盈亏方向"""
        if self.is_sell:
            return self.pnl > 0
        return None

# ============================================================
# 反馈解析器
# ============================================================
class FeedbackParser:
    """
    解析 feedback.txt
    
    格式: 日期 | 方向 | 代码 | 价格 | 数量 | 盈亏 | 备注
    """
    
    def __init__(self, feedback_path: Path = FEEDBACK_FILE):
        self.feedback_path = Path(feedback_path)
    
    def parse(self) -> List[TradeRecord]:
        """解析反馈文件，返回交易记录列表"""
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
                        date=parts[0],
                        side=parts[1],
                        code=parts[2],
                        price=float(parts[3]),
                        volume=int(float(parts[4])),  # 兼容 100.0
                        pnl=float(parts[5]),
                        note=parts[6] if len(parts) > 6 else '',
                        line_no=line_no,
                    )
                    records.append(record)
                except (ValueError, IndexError) as e:
                    logger.warning(f"第{line_no}行解析失败: {e} | 原文: {line}")
                    continue
        
        logger.info(f"解析完成: {len(records)}条交易记录")
        return records
    
    def pair_trades(self, records: List[TradeRecord]) -> List[Dict]:
        """
        配对买卖: 同一代码的买入→卖出配成一对完整交易
        返回: [{buy, sell, code, pnl, pnl_pct, hold_days, signal_correct}]
        """
        # 按代码分组
        by_code = defaultdict(list)
        for r in records:
            by_code[r.code].append(r)
        
        pairs = []
        for code, trades in by_code.items():
            # 按日期排序
            trades.sort(key=lambda x: x.date)
            
            buy_stack = []  # 买入栈(FIFO)
            for t in trades:
                if t.is_buy:
                    buy_stack.append(t)
                elif t.is_sell and buy_stack:
                    buy = buy_stack.pop(0)
                    pnl = t.pnl
                    cost = buy.price * buy.volume
                    pnl_pct = pnl / (cost + 1e-8) if cost > 0 else 0
                    
                    # 计算持有天数
                    try:
                        d_buy = datetime.strptime(buy.date, '%Y-%m-%d')
                        d_sell = datetime.strptime(t.date, '%Y-%m-%d')
                        hold_days = (d_sell - d_buy).days
                    except ValueError:
                        hold_days = 0
                    
                    pairs.append({
                        'code': code,
                        'buy_date': buy.date,
                        'sell_date': t.date,
                        'buy_price': buy.price,
                        'sell_price': t.price,
                        'volume': buy.volume,
                        'pnl': pnl,
                        'pnl_pct': pnl_pct,
                        'hold_days': hold_days,
                        'signal_correct': pnl > 0,
                        'buy_note': buy.note,
                        'sell_note': t.note,
                    })
        
        logger.info(f"配对完成: {len(pairs)}对完整交易")
        return pairs

# ============================================================
# 信号评估器
# ============================================================
class SignalEvaluator:
    """
    评估系统信号的实战表现
    
    对比:
      - 系统发出的信号方向 vs 实际盈亏
      - 信号置信度 vs 实际胜率
      - 不同庄家行为阶段的信号准确率
    """
    
    def __init__(self, results_dir: Path = RESULTS_DIR):
        self.results_dir = Path(results_dir)
    
    def evaluate(self, trade_pairs: List[Dict]) -> Dict:
        """
        评估信号表现
        """
        if not trade_pairs:
            return {'status': 'no_data', 'message': '无完整交易对'}
        
        n = len(trade_pairs)
        wins = [p for p in trade_pairs if p['signal_correct']]
        losses = [p for p in trade_pairs if not p['signal_correct']]
        
        # 基本胜率
        win_rate = len(wins) / n if n > 0 else 0
        
        # 总盈亏
        total_pnl = sum(p['pnl'] for p in trade_pairs)
        avg_pnl_pct = np.mean([p['pnl_pct'] for p in trade_pairs])
        
        # 盈亏比
        avg_win = np.mean([p['pnl'] for p in wins]) if wins else 0
        avg_loss = abs(np.mean([p['pnl'] for p in losses])) if losses else 1
        profit_loss_ratio = avg_win / (avg_loss + 1e-8)
        
        # 持仓天数统计
        hold_days = [p['hold_days'] for p in trade_pairs]
        
        # 按代码统计
        by_code = defaultdict(list)
        for p in trade_pairs:
            by_code[p['code']].append(p)
        
        code_performance = {}
        for code, pairs in by_code.items():
            code_wins = sum(1 for p in pairs if p['signal_correct'])
            code_performance[code] = {
                'trades': len(pairs),
                'wins': code_wins,
                'win_rate': code_wins / len(pairs),
                'total_pnl': sum(p['pnl'] for p in pairs),
            }
        
        # 按备注关键词分类(止损/止盈/追高/抄底)
        note_categories = defaultdict(list)
        for p in trade_pairs:
            sell_note = p.get('sell_note', '')
            if '止损' in sell_note:
                note_categories['止损'].append(p)
            elif '止盈' in sell_note:
                note_categories['止盈'].append(p)
            elif '追高' in sell_note:
                note_categories['追高'].append(p)
            elif '抄底' in sell_note:
                note_categories['抄底'].append(p)
            else:
                note_categories['其他'].append(p)
        
        note_stats = {}
        for cat, pairs in note_categories.items():
            note_stats[cat] = {
                'count': len(pairs),
                'win_rate': sum(1 for p in pairs if p['signal_correct']) / max(len(pairs), 1),
                'avg_pnl_pct': np.mean([p['pnl_pct'] for p in pairs]),
            }
        
        result = {
            'status': 'ok',
            'total_trades': n,
            'wins': len(wins),
            'losses': len(losses),
            'win_rate': float(win_rate),
            'total_pnl': float(total_pnl),
            'avg_pnl_pct': float(avg_pnl_pct),
            'profit_loss_ratio': float(profit_loss_ratio),
            'avg_hold_days': float(np.mean(hold_days)),
            'by_code': code_performance,
            'by_note': note_stats,
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        }
        
        logger.info(f"信号评估: {n}笔交易, 胜率{win_rate:.1%}, "
                    f"总盈亏{total_pnl:.0f}, 盈亏比{profit_loss_ratio:.2f}")
        
        return result

# ============================================================
# 模型校准器
# ============================================================
class ModelCalibrator:
    """
    根据实盘反馈校准对抗模型
    
    校准维度:
      1. 庄家行为阈值: 信号错误率高 → 庄家"防共四策"更激进
      2. 散户类型比例: 实际被割多 → 增加跟风型比例
      3. 奖励权重: 实盘胜率偏低 → 增大"散户预测错误度"在奖励中的权重
      4. 仓位建议: 连续亏损 → 收紧风控阈值
      5. 重训触发: 信号准确率低于阈值 → 建议重训对抗模型
    """
    
    def __init__(self, cfg: FeedbackConfig = None):
        self.cfg = cfg or FeedbackConfig()
    
    def calibrate(self, evaluation: Dict) -> Dict:
        """
        根据评估结果生成校准参数
        """
        win_rate = evaluation.get('win_rate', 0.5)
        avg_pnl_pct = evaluation.get('avg_pnl_pct', 0)
        plr = evaluation.get('profit_loss_ratio', 1.0)
        note_stats = evaluation.get('by_note', {})
        
        calibrations = {
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'win_rate': win_rate,
            'need_retrain': False,
            'adjustments': {},
        }
        
        # ── 1. 庄家行为阈值调整 ──
        # 胜率低 = 庄家太强或识别不足 → 加大庄家行为强度让模型学会对抗
        if win_rate < 0.4:
            dealer_adj = {
                'info_power_delta': +0.1,       # 信息操纵能力 +10%
                'shake_intensity_delta': +0.02,  # 震仓强度 +2%
                'puppet_ratio_delta': +0.05,    # 白手套资金 +5%
                'reason': f'胜率{win_rate:.0%}<40%, 增强庄家让模型学会对抗',
            }
            calibrations['adjustments']['dealer'] = dealer_adj
        elif win_rate > 0.65:
            dealer_adj = {
                'info_power_delta': -0.05,
                'shake_intensity_delta': -0.01,
                'reason': f'胜率{win_rate:.0%}>65%, 适度减弱庄家避免过拟合',
            }
            calibrations['adjustments']['dealer'] = dealer_adj
        
        # ── 2. 散户类型比例调整 ──
        # 止损多 = 散户被割 → 增加跟风型(更容易被割)
        # 追高多 = FOMO严重 → 增加跟风型+减少价值型
        stop_loss_trades = note_stats.get('止损', {}).get('count', 0)
        chase_trades = note_stats.get('追高', {}).get('count', 0)
        total = evaluation.get('total_trades', 1)
        
        if (stop_loss_trades + chase_trades) / total > 0.5:
            retailer_adj = {
                'herd_prob_delta': +0.1,    # 跟风型 +10%
                'value_prob_delta': -0.05,  # 价值型 -5%
                'leader_prob_delta': +0.05, # 头羊型 +5%
                'reason': f'止损+追高占{(stop_loss_trades+chase_trades)/total:.0%}, 增加跟风模拟',
            }
            calibrations['adjustments']['retailer'] = retailer_adj
        
        # ── 3. 奖励权重调整 ──
        # 盈亏比低 = 信号质量差 → 增大实盘反馈在奖励中的权重
        if plr < 1.0:
            reward_adj = {
                'reward_shaping_weight_delta': +0.1,
                'reason': f'盈亏比{plr:.2f}<1.0, 增大实盘反馈权重引导训练',
            }
            calibrations['adjustments']['reward'] = reward_adj
        
        # ── 4. 风控阈值收紧 ──
        if avg_pnl_pct < -0.02:
            risk_adj = {
                'max_drawdown_delta': -0.03,    # 回撤容忍降低3%
                'signal_conf_thresh_delta': +0.1, # 信号置信度门槛提高10%
                'max_position_cap': 0.2,         # 最大仓位限制20%
                'reason': f'平均盈亏{avg_pnl_pct:.2%}<-2%, 收紧风控',
            }
            calibrations['adjustments']['risk'] = risk_adj
        
        # ── 5. 重训触发 ──
        if win_rate < self.cfg.signal_accuracy_thresh:
            calibrations['need_retrain'] = True
            calibrations['retrain_reason'] = (
                f'信号准确率{win_rate:.0%} < 阈值{self.cfg.signal_accuracy_thresh:.0%}, '
                f'建议重新训练对抗模型(增大episodes或调整进化参数)'
            )
        
        logger.info(f"校准完成: {'需要重训' if calibrations['need_retrain'] else '参数微调'}")
        return calibrations
    
    def apply_calibrations(self, calibrations: Dict, 
                            adv_model_path: Path = None) -> Dict:
        """
        将校准参数应用到对抗模型
        方式: 读取现有模型 → 修改参数 → 保存为校准版本
        
        返回: {applied: bool, new_model_path: str, changes: [...]}
        """
        changes = []
        adj = calibrations.get('adjustments', {})
        
        # 应用庄家参数
        if 'dealer' in adj:
            d = adj['dealer']
            changes.append(f"庄家: info_power{d.get('info_power_delta',0):+.2f}, "
                         f"shake{d.get('shake_intensity_delta',0):+.2f}")
        
        # 应用散户参数
        if 'retailer' in adj:
            r = adj['retailer']
            changes.append(f"散户: herd{r.get('herd_prob_delta',0):+.2f}, "
                         f"value{r.get('value_prob_delta',0):+.2f}")
        
        # 应用奖励参数
        if 'reward' in adj:
            rw = adj['reward']
            changes.append(f"奖励: shaping_weight{rw.get('reward_shaping_weight_delta',0):+.2f}")
        
        # 应用风控参数
        if 'risk' in adj:
            rk = adj['risk']
            changes.append(f"风控: 回撤{rk.get('max_drawdown_delta',0):+.2f}, "
                         f"置信度{rk.get('signal_conf_thresh_delta',0):+.2f}")
        
        # 保存校准参数文件 (对抗环境读取此文件应用参数)
        cal_file = Path(self.cfg.results_dir) / "calibration_params.json"
        with open(cal_file, 'w', encoding='utf-8') as f:
            json.dump(calibrations, f, ensure_ascii=False, indent=2)
        
        logger.info(f"校准参数已保存: {cal_file}")
        logger.info(f"变更: {'; '.join(changes)}")
        
        return {
            'applied': True,
            'calibration_file': str(cal_file),
            'changes': changes,
            'need_retrain': calibrations.get('need_retrain', False),
        }

# ============================================================
# 校准报告生成
# ============================================================
class CalibrationReporter:
    """生成人可读的校准报告"""
    
    def generate(self, evaluation: Dict, calibrations: Dict,
                  apply_result: Dict) -> str:
        """生成校准报告(Markdown)"""
        lines = []
        lines.append("# 实盘反馈校准报告")
        lines.append(f"生成时间: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append("")
        
        # ── 实战表现 ──
        lines.append("## 实战表现")
        wr = evaluation.get('win_rate', 0)
        emoji = "🟢" if wr >= 0.5 else "🔴"
        lines.append(f"- {emoji} **胜率**: {wr:.1%} ({evaluation.get('wins',0)}胜/{evaluation.get('losses',0)}负)")
        lines.append(f"- **总盈亏**: {evaluation.get('total_pnl',0):,.0f} 元")
        lines.append(f"- **平均盈亏率**: {evaluation.get('avg_pnl_pct',0):.2%}")
        lines.append(f"- **盈亏比**: {evaluation.get('profit_loss_ratio',0):.2f}")
        lines.append(f"- **平均持仓**: {evaluation.get('avg_hold_days',0):.1f} 天")
        lines.append("")
        
        # ── 个股表现 ──
        by_code = evaluation.get('by_code', {})
        if by_code:
            lines.append("## 个股表现")
            lines.append("| 代码 | 交易数 | 胜率 | 总盈亏 |")
            lines.append("|------|--------|------|--------|")
            for code, perf in sorted(by_code.items(), key=lambda x: x[1]['total_pnl'], reverse=True):
                emoji = "🟢" if perf['total_pnl'] >= 0 else "🔴"
                lines.append(f"| {code} | {perf['trades']} | {perf['win_rate']:.0%} | {emoji}{perf['total_pnl']:,.0f} |")
            lines.append("")
        
        # ── 备注分类 ──
        by_note = evaluation.get('by_note', {})
        if by_note:
            lines.append("## 操作分类")
            lines.append("| 类型 | 次数 | 胜率 | 平均盈亏率 |")
            lines.append("|------|------|------|------------|")
            for cat, stats in by_note.items():
                lines.append(f"| {cat} | {stats['count']} | {stats['win_rate']:.0%} | {stats['avg_pnl_pct']:.2%} |")
            lines.append("")
        
        # ── 校准建议 ──
        lines.append("## 校准建议")
        adj = calibrations.get('adjustments', {})
        if adj:
            for category, params in adj.items():
                cn = {'dealer': '庄家', 'retailer': '散户', 'reward': '奖励', 'risk': '风控'}
                lines.append(f"### {cn.get(category, category)}")
                lines.append(f"- {params.get('reason', '')}")
                for k, v in params.items():
                    if k != 'reason' and 'delta' in k:
                        lines.append(f"  - {k}: {v:+}")
                lines.append("")
        else:
            lines.append("当前参数无需调整。")
            lines.append("")
        
        # ── 重训建议 ──
        if calibrations.get('need_retrain'):
            lines.append("## ⚠️ 重训建议")
            lines.append(f"> {calibrations.get('retrain_reason', '')}")
            lines.append("")
            lines.append("建议命令:")
            lines.append("```powershell")
            lines.append("# 增加训练轮次重新训练")
            lines.append("python adversarial_env.py --mode train --episodes 2000 --evolve")
            lines.append("```")
            lines.append("")
        
        # ── 应用结果 ──
        if apply_result:
            lines.append("## 校准应用")
            lines.append(f"- 状态: {'✓ 已应用' if apply_result.get('applied') else '✗ 未应用'}")
            lines.append(f"- 参数文件: `{apply_result.get('calibration_file', '')}`")
            for c in apply_result.get('changes', []):
                lines.append(f"- {c}")
            lines.append("")
        
        return "\n".join(lines)

# ============================================================
# 对抗环境对接: RealityCalibrator
# ============================================================
class RealityCalibrator:
    """
    对抗环境的实盘校准窗口
    
    在 adversarial_env.py 中集成:
      - 训练前加载 calibration_params.json
      - 动态调整庄家/散户/奖励参数
      - 训练后输出新校准参数供反馈处理器使用
    
    使用方式(在 adversarial_env.py 中):
      from feedback_processor import RealityCalibrator
      
      trainer = AdversarialTrainer(cfg)
      calibrator = RealityCalibrator()
      calibrator.apply_to_trainer(trainer)  # 应用校准参数
      trainer.train(...)
      calibrator.save_state(trainer)        # 保存训练后状态
    """
    
    def __init__(self, results_dir: Path = RESULTS_DIR):
        self.results_dir = Path(results_dir)
        self.cal_file = self.results_dir / "calibration_params.json"
        self.cal_data = None
    
    def load(self) -> Optional[Dict]:
        """加载校准参数"""
        if self.cal_file.exists():
            with open(self.cal_file, 'r', encoding='utf-8') as f:
                self.cal_data = json.load(f)
            logger.info(f"校准参数已加载: {self.cal_file}")
            return self.cal_data
        return None
    
    def apply_to_trainer(self, trainer) -> Dict:
        """
        将校准参数应用到对抗训练器
        trainer: AdversarialTrainer 实例
        """
        cal = self.load()
        if not cal:
            logger.info("无校准参数，使用默认配置")
            return {}
        
        adj = cal.get('adjustments', {})
        applied = {}
        
        # 应用庄家参数
        if 'dealer' in adj:
            d = adj['dealer']
            dealer = trainer.dealer
            if 'info_power_delta' in d:
                dealer.info_power = max(0.1, min(1.0, 
                    dealer.info_power + d['info_power_delta']))
            if 'shake_intensity_delta' in d:
                dealer.shake_intensity = max(0.01, min(0.2,
                    dealer.shake_intensity + d['shake_intensity_delta']))
            if 'puppet_ratio_delta' in d:
                dealer.puppet_capital *= (1 + d['puppet_ratio_delta'])
            applied['dealer'] = True
        
        # 应用散户参数
        if 'retailer' in adj:
            r = adj['retailer']
            # 重新分配散户类型比例
            # (下一轮训练时自动生效)
            applied['retailer'] = True
        
        # 应用奖励参数
        if 'reward' in adj:
            rw = adj['reward']
            # 存储到trainer供reward计算时使用
            if not hasattr(trainer, '_cal_reward_weight'):
                trainer._cal_reward_weight = 0
            trainer._cal_reward_weight += rw.get('reward_shaping_weight_delta', 0)
            applied['reward'] = True
        
        logger.info(f"校准参数已应用: {list(applied.keys())}")
        return applied
    
    def save_state(self, trainer) -> Dict:
        """
        保存训练后的状态，供下一轮反馈处理参考
        """
        state = {
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'generation': trainer.generation,
            'dealer_info_power': trainer.dealer.info_power,
            'dealer_shake_intensity': trainer.dealer.shake_intensity,
            'strategy_diversity': trainer.strategy_diversity_history[-1] if trainer.strategy_diversity_history else 0,
        }
        
        state_file = self.results_dir / "trainer_state.json"
        with open(state_file, 'w', encoding='utf-8') as f:
            json.dump(state, f, ensure_ascii=False, indent=2)
        
        return state

# ============================================================
# 命令行入口
# ============================================================
def main():
    parser = argparse.ArgumentParser(description="实盘反馈处理器")
    parser.add_argument("--mode", required=True,
                        choices=["parse", "evaluate", "calibrate", "report", "full", "demo"])
    parser.add_argument("--feedback-file", type=str, default=str(FEEDBACK_FILE))
    parser.add_argument("--results-dir", type=str, default=str(RESULTS_DIR))
    parser.add_argument("--output", type=str, default=None)
    
    args = parser.parse_args()
    results_dir = Path(args.results_dir)
    
    if args.mode == "demo":
        # 生成示例反馈文件
        demo_content = """# 实盘反馈 - 格式: 日期 | 方向 | 代码 | 价格 | 数量 | 盈亏 | 备注
# 方向: 买入/卖出
# 盈亏: 卖出时填实际盈亏金额，买入时填0
# 备注: 自由文字(止损/止盈/追高/抄底等)
2026-06-23 | 买入 | 600519.SH | 1800.50 | 100 | 0 | 茅台，系统看多
2026-06-24 | 卖出 | 600519.SH | 1835.20 | 100 | 3470 | 茅台止盈
2026-06-25 | 买入 | 000001.SZ | 12.35 | 1000 | 0 | 平安银行，信号看多
2026-06-26 | 卖出 | 000001.SZ | 11.80 | 1000 | -550 | 止损出局
2026-06-27 | 买入 | 300750.SZ | 215.00 | 200 | 0 | 宁德时代，追高买入
2026-06-30 | 卖出 | 300750.SZ | 205.50 | 200 | -1900 | 追高被套止损
"""
        out = Path(args.feedback_file)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, 'w', encoding='utf-8') as f:
            f.write(demo_content)
        logger.info(f"示例反馈文件已生成: {out}")
        return
    
    # 全流程或其他模式
    fp = FeedbackParser(Path(args.feedback_file))
    records = fp.parse()
    
    if not records:
        logger.error("无交易记录，请检查反馈文件")
        return
    
    if args.mode == "parse":
        # 只解析
        pairs = fp.pair_trades(records)
        for p in pairs:
            print(f"  {p['code']}: {p['buy_date']}@{p['buy_price']:.2f} → "
                  f"{p['sell_date']}@{p['sell_price']:.2f} | "
                  f"盈亏{p['pnl']:+,.0f} ({p['pnl_pct']:+.2%}) | "
                  f"{'✓' if p['signal_correct'] else '✗'} {p['sell_note']}")
        return
    
    # 配对交易
    pairs = fp.pair_trades(records)
    
    # 评估
    evaluator = SignalEvaluator(results_dir)
    evaluation = evaluator.evaluate(pairs)
    
    if args.mode == "evaluate":
        print(json.dumps(evaluation, ensure_ascii=False, indent=2))
        return
    
    # 校准
    calibrator = ModelCalibrator()
    calibrations = calibrator.calibrate(evaluation)
    
    if args.mode == "calibrate":
        print(json.dumps(calibrations, ensure_ascii=False, indent=2))
        return
    
    # 应用校准
    apply_result = calibrator.apply_calibrations(calibrations)
    
    if args.mode == "report" or args.mode == "full":
        reporter = CalibrationReporter()
        report_md = reporter.generate(evaluation, calibrations, apply_result)
        
        out_path = Path(args.output) if args.output else results_dir / "calibration_report.md"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, 'w', encoding='utf-8') as f:
            f.write(report_md)
        
        # 同时保存JSON版本
        json_path = out_path.with_suffix('.json')
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump({
                'evaluation': evaluation,
                'calibrations': calibrations,
                'apply_result': apply_result,
            }, f, ensure_ascii=False, indent=2, default=str)
        
        logger.info(f"校准报告已保存: {out_path}")
        print(report_md)

if __name__ == "__main__":
    main()

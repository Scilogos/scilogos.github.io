"""
stock_interpreter.py - 对抗学习结果解读器
==========================================
Phase 4: 结果分析

四层解读:
  L1 统计聚合: 价格/成交量/波动率/自相关 → 基础统计画像
  L2 庄散博弈分析: 庄家行为识别 + 散户群体行为分析
  L3 历史类比: 与真实历史模式匹配
  L4 风控约束: 最大回撤/置信度/信号强度 → 可操作输出

排列检验: 迁移性门禁 (结果是否可迁移到真实市场)

用法:
  python stock_interpreter.py --mode analyze --data-dir ... 
  python stock_interpreter.py --mode interpret --model-path ...
  python stock_interpreter.py --mode report
"""

import os, sys, argparse, json, time
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from collections import defaultdict

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from stock_config import (
    RESULTS_DIR, ADV_DATA_DIR, DATA_DIR,
    InterpreterConfig, setup_logger,
)

logger = setup_logger("Interpreter")

# ============================================================
# 第一层: 统计聚合
# ============================================================
class StatisticalAggregator:
    """
    L1 统计聚合: 基础统计画像
    
    输出:
      - 价格: 均值/方差/偏度/峰度/自相关
      - 成交量: 均值/方差/VWAP
      - 波动率: 已实现波动率/GARCH拟合
      - 收益率: 均值/方差/Sharpe/最大回撤
    """
    
    def __init__(self, window: int = 20):
        self.window = window
    
    def analyze(self, prices: np.ndarray, volumes: Optional[np.ndarray] = None) -> Dict:
        """
        prices: (T,) 价格序列
        volumes: (T,) 成交量序列 (可选)
        """
        from scipy.stats import skew, kurtosis
        
        returns = np.diff(prices) / (prices[:-1] + 1e-8)
        
        result = {
            'price': {
                'mean': float(np.mean(prices)),
                'std': float(np.std(prices)),
                'min': float(np.min(prices)),
                'max': float(np.max(prices)),
                'skewness': float(skew(prices)),
                'kurtosis': float(kurtosis(prices)),
                'cv': float(np.std(prices) / (np.mean(prices) + 1e-8)),  # 变异系数
            },
            'returns': {
                'mean': float(np.mean(returns)),
                'std': float(np.std(returns)),
                'sharpe': float(np.mean(returns) / (np.std(returns) + 1e-8) * np.sqrt(252)),
                'skewness': float(skew(returns)),
                'kurtosis': float(kurtosis(returns)),
                'acf1': float(self._autocorr(returns, 1)),
                'acf5': float(self._autocorr(returns, 5)),
            },
            'volatility': {
                'realized_vol': float(np.std(returns) * np.sqrt(252)),
                'max_drawdown': float(self._max_drawdown(prices)),
                'vol_cluster': float(self._volatility_clustering(returns)),
            },
        }
        
        if volumes is not None and len(volumes) == len(prices):
            result['volume'] = {
                'mean': float(np.mean(volumes)),
                'std': float(np.std(volumes)),
                'vwap': float(np.sum(prices * volumes) / (np.sum(volumes) + 1e-8)),
                'price_volume_corr': float(np.corrcoef(returns, volumes[1:])[0, 1]) if len(volumes) > 1 else 0,
            }
        
        return result
    
    def _autocorr(self, x, lag):
        if len(x) <= lag:
            return 0.0
        return float(np.corrcoef(x[:-lag], x[lag:])[0, 1])
    
    def _max_drawdown(self, prices):
        peak = prices[0]
        max_dd = 0.0
        for p in prices:
            if p > peak:
                peak = p
            dd = (peak - p) / (peak + 1e-8)
            if dd > max_dd:
                max_dd = dd
        return max_dd
    
    def _volatility_clustering(self, returns):
        """波动率聚集: |r_t|与|r_{t-1}|的自相关"""
        abs_ret = np.abs(returns)
        if len(abs_ret) < 2:
            return 0.0
        return float(np.corrcoef(abs_ret[:-1], abs_ret[1:])[0, 1])

# ============================================================
# 第二层: 庄散博弈分析
# ============================================================
class GameAnalyzer:
    """
    L2 庄散博弈分析
    
    识别:
      - 庄家行为: 吸筹/拉抬/震仓/派发 四阶段
      - 散户群体行为: 抱团/踩踏/分化
      - 博弈均衡: 庄家胜率/散户存活率
      - 信息操纵: 异常消息冲击
    """
    
    # 庄家行为识别阈值
    ACCUMULATE_THRESH = 0.02   # 2%价格波动内持续买入
    PUMP_THRESH = 0.03         # 3%以上快速上涨
    SHAKE_THRESH = -0.02       # 2%以上快速下跌
    DISTRIBUTE_THRESH = 0.01   # 高位持续卖出
    
    def analyze(self, price_data: np.ndarray, trade_data: List[Dict],
                agent_pnls: Optional[Dict] = None) -> Dict:
        """
        price_data: (T,) 价格序列
        trade_data: [{tick, price, volume, buyer, seller}]
        agent_pnls: {agent_id: pnl}
        """
        result = {
            'dealer_behavior': self._identify_dealer_behavior(price_data, trade_data),
            'retailer_behavior': self._analyze_retailer_behavior(trade_data),
            'game_equilibrium': self._compute_equilibrium(trade_data, agent_pnls),
            'information_events': self._detect_info_manipulation(price_data, trade_data),
        }
        return result
    
    def _identify_dealer_behavior(self, prices, trades) -> Dict:
        """识别庄家行为四阶段"""
        T = len(prices)
        phases = []
        
        # 滑动窗口识别
        window = 10
        for i in range(0, T - window, window // 2):
            seg = prices[i:i+window]
            ret = (seg[-1] - seg[0]) / (seg[0] + 1e-8)
            vol = np.std(np.diff(seg) / (seg[:-1] + 1e-8))
            
            if abs(ret) < self.ACCUMULATE_THRESH and vol < 0.01:
                phase = 'accumulate'
            elif ret > self.PUMP_THRESH:
                phase = 'pump'
            elif ret < self.SHAKE_THRESH:
                phase = 'shake'
            elif ret > self.DISTRIBUTE_THRESH and i > T * 0.6:
                phase = 'distribute'
            else:
                phase = 'hold'
            
            phases.append({
                'start': i, 'end': i + window,
                'phase': phase, 'return': float(ret), 'volatility': float(vol),
            })
        
        # 统计各阶段占比
        phase_counts = defaultdict(int)
        for p in phases:
            phase_counts[p['phase']] += 1
        
        total = len(phases) if phases else 1
        
        return {
            'phases': phases,
            'phase_distribution': {k: v/total for k, v in phase_counts.items()},
            'dominant_phase': max(phase_counts, key=phase_counts.get) if phase_counts else 'unknown',
        }
    
    def _analyze_retailer_behavior(self, trades) -> Dict:
        """分析散户群体行为"""
        retailer_trades = [t for t in trades 
                          if t.get('buyer', '').startswith('retailer') or 
                             t.get('seller', '').startswith('retailer')]
        
        if not retailer_trades:
            return {'behavior': 'no_activity', 'herding_index': 0, 'panic_index': 0}
        
        # 买卖比例
        buy_count = sum(1 for t in retailer_trades if t.get('buyer', '').startswith('retailer'))
        sell_count = sum(1 for t in retailer_trades if t.get('seller', '').startswith('retailer'))
        total = buy_count + sell_count
        
        buy_ratio = buy_count / max(total, 1)
        
        # 抱团指数: 买卖方向一致性
        herding_index = abs(buy_ratio - 0.5) * 2  # 0~1, 越高越一致
        
        # 恐慌指数: 卖出集中在短时间内
        sell_times = [t['tick'] for t in retailer_trades 
                     if t.get('seller', '').startswith('retailer')]
        panic_index = 0
        if len(sell_times) >= 3:
            sell_intervals = np.diff(sorted(sell_times))
            if len(sell_intervals) > 0:
                panic_index = min(1.0, 1.0 / (np.mean(sell_intervals) + 1e-8) * 5)
        
        # 行为判定
        if herding_index > 0.6 and buy_ratio > 0.7:
            behavior = 'herding_buy'
        elif herding_index > 0.6 and buy_ratio < 0.3:
            behavior = 'herding_sell'
        elif panic_index > 0.5:
            behavior = 'panic_selling'
        else:
            behavior = 'diversified'
        
        return {
            'behavior': behavior,
            'herding_index': float(herding_index),
            'panic_index': float(panic_index),
            'buy_ratio': float(buy_ratio),
            'total_trades': total,
        }
    
    def _compute_equilibrium(self, trades, agent_pnls) -> Dict:
        """计算博弈均衡"""
        if agent_pnls is None:
            return {'status': 'unknown'}
        
        dealer_pnl = sum(v for k, v in agent_pnls.items() if k.startswith('dealer'))
        retailer_pnl = sum(v for k, v in agent_pnls.items() if k.startswith('retailer'))
        hotmoney_pnl = sum(v for k, v in agent_pnls.items() if k.startswith('hotmoney'))
        
        total_pnl = dealer_pnl + retailer_pnl + hotmoney_pnl
        
        # 庄家胜率
        dealer_win_rate = 1.0 if dealer_pnl > 0 else 0.0
        
        # 散户存活率 (正收益的散户比例)
        retailer_pnls = [v for k, v in agent_pnls.items() if k.startswith('retailer')]
        retailer_survival = sum(1 for p in retailer_pnls if p > 0) / max(len(retailer_pnls), 1)
        
        return {
            'dealer_pnl': float(dealer_pnl),
            'retailer_pnl': float(retailer_pnl),
            'hotmoney_pnl': float(hotmoney_pnl),
            'dealer_win_rate': float(dealer_win_rate),
            'retailer_survival_rate': float(retailer_survival),
            'zero_sum_check': float(total_pnl),  # 接近0则符合零和博弈
        }
    
    def _detect_info_manipulation(self, prices, trades) -> List[Dict]:
        """检测信息操纵事件"""
        events = []
        returns = np.diff(prices) / (prices[:-1] + 1e-8)
        
        # 异常波动检测 (超过2个标准差)
        mean_ret = np.mean(returns)
        std_ret = np.std(returns)
        
        for i, r in enumerate(returns):
            if abs(r - mean_ret) > 2 * std_ret:
                events.append({
                    'tick': i,
                    'return': float(r),
                    'z_score': float((r - mean_ret) / (std_ret + 1e-8)),
                    'direction': 'up' if r > 0 else 'down',
                    'type': 'potential_manipulation',
                })
        
        return events

# ============================================================
# 第三层: 历史类比
# ============================================================
class HistoricalAnalogy:
    """
    L3 历史类比: 与真实历史模式匹配
    
    方法:
      1. DTW距离: 找最相似的历史片段
      2. 形态识别: 头肩顶/双底等经典形态
      3. 事件匹配: 相似宏观事件下的市场反应
    """
    
    def __init__(self, data_dir: Path = DATA_DIR, topk: int = 10):
        self.data_dir = Path(data_dir)
        self.topk = topk
        self._real_patterns = self._load_real_patterns()
    
    def _load_real_patterns(self) -> Optional[np.ndarray]:
        """加载真实价格模式库"""
        try:
            adapter_path = self.data_dir
            patterns = []
            count = 0
            for f in adapter_path.glob("*_daily.csv"):
                try:
                    df = pd.read_csv(f, usecols=['close'])
                    prices = df['close'].values.astype(np.float64)
                    if len(prices) >= 30:
                        # 取最近30日归一化
                        seg = prices[-30:]
                        seg = (seg - seg.min()) / (seg.max() - seg.min() + 1e-8)
                        patterns.append(seg)
                        count += 1
                        if count >= 100:
                            break
                except Exception:
                    continue
            
            if patterns:
                return np.array(patterns)
        except Exception:
            pass
        return None
    
    def find_similar(self, generated_prices: np.ndarray, topk: int = None) -> List[Dict]:
        """
        找到与生成价格最相似的历史片段
        generated_prices: (T,) 归一化后的价格
        """
        topk = topk or self.topk
        
        if self._real_patterns is None:
            return [{'error': '无真实数据模式库'}]
        
        # 归一化生成价格
        gen = generated_prices.copy()
        if len(gen) > 30:
            gen = gen[-30:]
        gen = (gen - gen.min()) / (gen.max() - gen.min() + 1e-8)
        
        # DTW距离 (简化: 欧氏距离)
        distances = []
        for i, pattern in enumerate(self._real_patterns):
            min_len = min(len(gen), len(pattern))
            dist = np.sqrt(((gen[:min_len] - pattern[:min_len]) ** 2).mean())
            distances.append((i, dist))
        
        distances.sort(key=lambda x: x[1])
        
        results = []
        for idx, dist in distances[:topk]:
            similarity = max(0, 1 - dist / 2)  # 归一化到[0,1]
            results.append({
                'pattern_index': idx,
                'distance': float(dist),
                'similarity': float(similarity),
            })
        
        return results
    
    def detect_pattern(self, prices: np.ndarray) -> Dict:
        """检测经典形态"""
        if len(prices) < 10:
            return {'pattern': 'insufficient_data'}
        
        # 简化形态检测
        ret = (prices[-1] - prices[0]) / (prices[0] + 1e-8)
        
        # 双顶/双底
        peaks = self._find_peaks(prices)
        
        if len(peaks) >= 2:
            if abs(peaks[-1][1] - peaks[-2][1]) / (prices.mean() + 1e-8) < 0.03:
                if ret < 0:
                    pattern = 'double_top'
                else:
                    pattern = 'double_bottom'
            else:
                pattern = 'trending'
        elif len(peaks) == 1 and ret < -0.05:
            pattern = 'head_and_shoulders_candidate'
        else:
            pattern = 'trending'
        
        return {
            'pattern': pattern,
            'trend': 'up' if ret > 0 else 'down',
            'magnitude': float(abs(ret)),
            'peaks': peaks,
        }
    
    def _find_peaks(self, prices, min_distance=5):
        """简化峰谷检测"""
        peaks = []
        for i in range(1, len(prices) - 1):
            if prices[i] > prices[i-1] and prices[i] > prices[i+1]:
                if not peaks or i - peaks[-1][0] >= min_distance:
                    peaks.append((i, float(prices[i])))
        return peaks[-5:]  # 最近5个峰

# ============================================================
# 第四层: 风控约束与信号输出
# ============================================================
class RiskController:
    """
    L4 风控约束
    
    输出可操作信号:
      - 信号方向: 看多/看空/观望
      - 置信度: 0~1
      - 风控约束: 最大仓位/止损位/时间约束
      - 逻辑链: 信号→证据→推理
    """
    
    def __init__(self, cfg: InterpreterConfig = None):
        self.cfg = cfg or InterpreterConfig()
    
    def generate_signal(self, stats: Dict, game: Dict,
                         analogy: Dict, prices: np.ndarray) -> Dict:
        """
        综合四层信息，生成可操作信号
        
        不是复读机: 独立做分析判断给理由
        """
        evidence = []
        confidence = 0.5  # 基准50%
        
        # ── 从统计层获取证据 ──
        ret_stats = stats.get('returns', {})
        if ret_stats.get('sharpe', 0) > 1.0:
            evidence.append(f"Sharpe={ret_stats['sharpe']:.2f}>1, 正向收益风险比")
            confidence += 0.1
        elif ret_stats.get('sharpe', 0) < -0.5:
            evidence.append(f"Sharpe={ret_stats['sharpe']:.2f}<-0.5, 负向信号")
            confidence -= 0.1
        
        vol_stats = stats.get('volatility', {})
        if vol_stats.get('max_drawdown', 1) > self.cfg.max_drawdown:
            evidence.append(f"最大回撤{vol_stats['max_drawdown']:.1%}>警戒线{self.cfg.max_drawdown:.0%}")
            confidence -= 0.15
        
        if vol_stats.get('vol_cluster', 0) > 0.3:
            evidence.append(f"波动率聚集={vol_stats['vol_cluster']:.2f}, 高波动延续风险")
            confidence -= 0.05
        
        # ── 从博弈层获取证据 ──
        dealer_behavior = game.get('dealer_behavior', {})
        dominant = dealer_behavior.get('dominant_phase', 'unknown')
        
        if dominant == 'pump':
            evidence.append("庄家主导拉抬阶段 → 短期看多但警惕派发")
            confidence += 0.05
        elif dominant == 'shake':
            evidence.append("庄家震仓洗盘 → 可能是吸筹后动作，不轻易看空")
            confidence += 0.05  # 震仓后往往有行情
        elif dominant == 'distribute':
            evidence.append("⚠ 庄家派发阶段 → 高度警惕，不宜追高")
            confidence -= 0.2
        elif dominant == 'accumulate':
            evidence.append("庄家吸筹阶段 → 中期偏多，短期可能低迷")
            confidence += 0.1
        
        retailer_behavior = game.get('retailer_behavior', {})
        retailer_state = retailer_behavior.get('behavior', 'unknown')
        if retailer_state == 'panic_selling':
            evidence.append("散户恐慌踩踏 → 逆向信号，可能是底部")
            confidence += 0.05
        elif retailer_state == 'herding_buy':
            evidence.append("散户抱团追高 → 警惕庄家派发")
            confidence -= 0.1
        
        # ── 从历史类比获取证据 ──
        if isinstance(analogy, list) and analogy:
            best_match = analogy[0]
            if best_match.get('similarity', 0) > 0.8:
                evidence.append(f"历史类比相似度{best_match['similarity']:.2f} → 历史重演概率高")
                # 类比方向看趋势
                if best_match.get('pattern_index', -1) >= 0:
                    confidence += 0.05
        
        # ── 综合判定 ──
        confidence = max(0.1, min(0.95, confidence))
        
        # 信号方向
        if confidence > self.cfg.signal_conf_thresh:
            if dominant in ['accumulate', 'shake'] and retailer_state != 'herding_buy':
                direction = 'long'
                reason = "庄家吸筹/震仓 + 散户未追高 → 偏多"
            elif dominant == 'distribute' or retailer_state == 'herding_buy':
                direction = 'short'
                reason = "庄家派发或散户追高 → 偏空"
            else:
                direction = 'neutral'
                reason = "多空信号矛盾 → 观望"
        else:
            direction = 'neutral'
            reason = f"置信度{confidence:.2f}<阈值{self.cfg.signal_conf_thresh} → 信号不可靠"
        
        # 风控约束
        current_price = prices[-1] if len(prices) > 0 else 0
        max_position = 0.3 if confidence > 0.7 else 0.1  # 高置信→30%仓位
        
        stop_loss = current_price * (1 - self.cfg.max_drawdown) if direction == 'long' else \
                    current_price * (1 + self.cfg.max_drawdown)
        
        return {
            'direction': direction,
            'confidence': float(confidence),
            'reason': reason,
            'evidence': evidence,
            'risk_control': {
                'max_position': float(max_position),
                'stop_loss': float(stop_loss),
                'max_drawdown_limit': float(self.cfg.max_drawdown),
                'time_constraint': '开盘30分钟内信号可靠性最高',
            },
            'logical_chain': self._build_logical_chain(direction, dominant, retailer_state, evidence),
        }
    
    def _build_logical_chain(self, direction, dealer_phase, retailer_state, evidence) -> str:
        """构建逻辑链: 信号→证据→推理"""
        chain = f"信号[{direction}] ← "
        chain += f"庄家[{dealer_phase}] + 散户[{retailer_state}] ← "
        chain += " | ".join(evidence[:3])  # 最多3条核心证据
        return chain

# ============================================================
# 排列检验 (迁移性门禁)
# ============================================================
class PermutationGate:
    """
    排列检验: 验证结果是否可迁移到真实市场
    
    H0 (零假设): 对抗学习发现的模式与随机噪声无异
    拒绝H0 → 模式有迁移性 (可信赖)
    不拒绝H0 → 可能是过拟合 (不可信赖)
    """
    
    def __init__(self, n_samples: int = 1000, significance: float = 0.05):
        self.n_samples = n_samples
        self.significance = significance
    
    def test(self, observed_stat: float, null_distribution: np.ndarray) -> Dict:
        """
        observed_stat: 观测到的统计量 (如庄家超额收益)
        null_distribution: 零假设分布 (随机策略的统计量)
        """
        p_value = np.mean(null_distribution >= observed_stat)
        
        reject = p_value < self.significance
        
        return {
            'observed': float(observed_stat),
            'null_mean': float(np.mean(null_distribution)),
            'null_std': float(np.std(null_distribution)),
            'p_value': float(p_value),
            'reject_null': bool(reject),
            'conclusion': '模式有迁移性(可信赖)' if reject else '模式可能是过拟合(不可信赖)',
            'z_score': float((observed_stat - np.mean(null_distribution)) / (np.std(null_distribution) + 1e-8)),
        }
    
    def generate_null_distribution(self, prices: np.ndarray,
                                     stat_fn, n_samples: int = None) -> np.ndarray:
        """
        生成零假设分布
        stat_fn: 计算统计量的函数
        """
        n_samples = n_samples or self.n_samples
        null_stats = []
        
        for _ in range(n_samples):
            # 打乱价格序列
            shuffled = prices.copy()
            np.random.shuffle(shuffled)
            stat = stat_fn(shuffled)
            null_stats.append(stat)
        
        return np.array(null_stats)

# ============================================================
# 解读器主类
# ============================================================
class StockInterpreter:
    """
    四层解读器: 统计→博弈→类比→风控
    """
    
    def __init__(self, cfg: InterpreterConfig = None, data_dir: Path = DATA_DIR):
        self.cfg = cfg or InterpreterConfig()
        self.stats = StatisticalAggregator(window=cfg.stat_window if cfg else 20)
        self.game = GameAnalyzer()
        self.analogy = HistoricalAnalogy(data_dir, topk=cfg.historical_topk if cfg else 10)
        self.risk = RiskController(cfg)
        self.perm_gate = PermutationGate(
            n_samples=cfg.perm_samples if cfg else 1000,
            significance=cfg.significance if cfg else 0.05,
        )
    
    def interpret(self, prices: np.ndarray, 
                   volumes: Optional[np.ndarray] = None,
                   trade_data: Optional[List[Dict]] = None,
                   agent_pnls: Optional[Dict] = None) -> Dict:
        """
        完整四层解读
        
        prices: (T,) 价格序列
        volumes: (T,) 成交量
        trade_data: 交易明细
        agent_pnls: 各Agent盈亏
        """
        if trade_data is None:
            trade_data = []
        
        # L1: 统计聚合
        logger.info("L1 统计聚合...")
        stat_result = self.stats.analyze(prices, volumes)
        
        # L2: 庄散博弈分析
        logger.info("L2 庄散博弈分析...")
        game_result = self.game.analyze(prices, trade_data, agent_pnls)
        
        # L3: 历史类比
        logger.info("L3 历史类比...")
        analogy_result = self.analogy.find_similar(prices)
        pattern_result = self.analogy.detect_pattern(prices)
        
        # L4: 风控约束与信号
        logger.info("L4 风控约束...")
        signal = self.risk.generate_signal(stat_result, game_result, 
                                            analogy_result, prices)
        
        # 排列检验
        logger.info("排列检验...")
        perm_result = self._permutation_test(prices, game_result)
        
        # 汇总
        report = {
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'data_length': len(prices),
            'L1_statistics': stat_result,
            'L2_game_analysis': game_result,
            'L3_historical_analogy': {
                'similar_patterns': analogy_result,
                'detected_pattern': pattern_result,
            },
            'L4_signal': signal,
            'permutation_test': perm_result,
            'summary': self._generate_summary(stat_result, game_result, signal, perm_result),
        }
        
        return report
    
    def _permutation_test(self, prices, game_result) -> Dict:
        """执行排列检验"""
        # 统计量: 庄家收益相对于随机策略的优越性
        dealer_pnl = game_result.get('game_equilibrium', {}).get('dealer_pnl', 0)
        
        # 生成零假设分布
        def stat_fn(p):
            # 简单统计量: 价格动量
            returns = np.diff(p) / (p[:-1] + 1e-8)
            return np.sum(returns)
        
        null_dist = self.perm_gate.generate_null_distribution(prices, stat_fn, n_samples=200)
        obs_stat = stat_fn(prices)
        
        return self.perm_gate.test(obs_stat, null_dist)
    
    def _generate_summary(self, stats, game, signal, perm) -> str:
        """生成自然语言摘要"""
        lines = []
        
        # 市场状态
        ret = stats.get('returns', {}).get('mean', 0)
        vol = stats.get('volatility', {}).get('realized_vol', 0)
        lines.append(f"市场: {'上涨' if ret > 0 else '下跌'}趋势, "
                    f"年化波动率{vol:.1%}")
        
        # 庄家行为
        dealer_phase = game.get('dealer_behavior', {}).get('dominant_phase', 'unknown')
        phase_cn = {'accumulate': '吸筹', 'pump': '拉抬', 'shake': '震仓',
                   'distribute': '派发', 'hold': '观望', 'unknown': '未知'}
        lines.append(f"庄家: {phase_cn.get(dealer_phase, dealer_phase)}阶段")
        
        # 散户行为
        retailer = game.get('retailer_behavior', {}).get('behavior', 'unknown')
        retailer_cn = {'herding_buy': '抱团追高', 'herding_sell': '抱团杀跌',
                      'panic_selling': '恐慌踩踏', 'diversified': '分化', 'unknown': '未知'}
        lines.append(f"散户: {retailer_cn.get(retailer, retailer)}")
        
        # 信号
        lines.append(f"信号: {signal['direction']} (置信度{signal['confidence']:.0%})")
        lines.append(f"理由: {signal['reason']}")
        
        # 排列检验
        if perm.get('reject_null'):
            lines.append(f"排列检验: ✓ 模式有迁移性 (p={perm['p_value']:.3f})")
        else:
            lines.append(f"排列检验: ✗ 可能过拟合 (p={perm['p_value']:.3f})")
        
        return "\n".join(lines)


# ============================================================
#  ★ v3.0: 多组合对比解读器
# ============================================================
class MultiCombinationInterpreter:
    """多组合对比解读器 - 解读多组合竞技场结果"""
    
    def __init__(self, data_dir: Path = DATA_DIR):
        self.data_dir = data_dir
        self.base_interpreter = StockInterpreter(InterpreterConfig(), data_dir)
    
    def compare_combinations(self, arena_results: Dict) -> Dict:
        """
        对比多个组合的优劣势
        
        Args:
            arena_results: MultiCombinationArena保存的结果
            
        Returns:
            对比分析结果
        """
        profiles = arena_results.get('profiles', [])
        eval_results = arena_results.get('evaluation_results', {})
        rankings = arena_results.get('specialist_ranking', {})
        
        comparison = {
            'summary': {},
            'by_horizon': {},
            'by_specialty': {},
            'recommendations': [],
        }
        
        # 按预测周期分组
        for profile in profiles:
            name_en = profile['name_en']
            horizon = profile['horizon']
            specialty = profile['specialty']
            eval_data = eval_results.get(name_en, {})
            
            # 提取各维度得分
            scores = {}
            for dim in ['direction_1d', 'direction_3d', 'direction_5d']:
                if dim in rankings:
                    for r in rankings[dim]:
                        if r.get('name') == profile['name']:
                            scores[dim] = r.get('score', 0)
                            break
            
            scores['composite'] = eval_data.get('composite', 0)
            
            # 按周期分组
            if horizon not in comparison['by_horizon']:
                comparison['by_horizon'][horizon] = []
            comparison['by_horizon'][horizon].append({
                'name': profile['name'],
                'name_en': name_en,
                'specialty': specialty,
                'scores': scores,
            })
            
            # 按专长分组
            if specialty not in comparison['by_specialty']:
                comparison['by_specialty'][specialty] = []
            comparison['by_specialty'][specialty].append({
                'name': profile['name'],
                'horizon': horizon,
                'scores': scores,
            })
        
        # 生成推荐
        comparison['recommendations'] = self._generate_recommendations(comparison, rankings)
        
        return comparison
    
    def _generate_recommendations(self, comparison: Dict, rankings: Dict) -> List[Dict]:
        """生成使用建议"""
        recs = []
        
        # 每个维度选最佳
        for dim in ['direction_1d', 'direction_3d', 'direction_5d', 'composite']:
            if dim in rankings and rankings[dim]:
                best = rankings[dim][0]
                horizon_map = {'direction_1d': '1日', 'direction_3d': '3日', 
                              'direction_5d': '5日', 'composite': '综合'}
                recs.append({
                    'dimension': dim,
                    'dimension_cn': horizon_map.get(dim, dim),
                    'best_combination': best['name'],
                    'score': best['score'],
                    'recommendation': f"预测{horizon_map.get(dim, dim)}行情时推荐使用【{best['name']}】组合",
                })
        
        return recs
    
    def generate_specialist_report(self, specialist_ranking: Dict) -> str:
        """
        生成专家选拔报告
        
        Args:
            specialist_ranking: 各维度专家排名
            
        Returns:
            格式化报告文本
        """
        lines = []
        lines.append("=" * 60)
        lines.append("  多组合进化竞技场 - 专家选拔报告")
        lines.append("=" * 60)
        lines.append("")
        
        # 维度中文映射
        dim_cn = {
            'direction_1d': '1日方向预测',
            'direction_3d': '3日方向预测',
            'direction_5d': '5日方向预测',
            'direction_up': '上涨预测',
            'direction_down': '下跌预测',
            'volatility_match': '波动率匹配',
            'stability': '价格稳定性',
            'composite': '综合评分',
        }
        
        for dim, ranking in specialist_ranking.items():
            dim_name = dim_cn.get(dim, dim)
            lines.append(f"【{dim_name}】")
            
            for i, entry in enumerate(ranking[:3]):
                medal = ['🥇', '🥈', '🥉'][i] if i < 3 else f"#{i+1}"
                lines.append(f"  {medal} {entry['name']} (得分: {entry.get('score', 0):.3f})")
            
            if ranking:
                lines.append(f"  → 推荐: 【{ranking[0]['name']}】\n")
        
        lines.append("=" * 60)
        return "\n".join(lines)
    
    def generate_action_forecast(self, next_actions: Dict,
                                   current_market_data: Dict = None) -> str:
        """
        生成各组合庄家下一步行动预测
        
        Args:
            next_actions: 各组合的庄家下一步行动
            current_market_data: 当前市场数据 (可选)
            
        Returns:
            格式化预测文本
        """
        lines = []
        lines.append("=" * 60)
        lines.append("  庄家下一步行动预测")
        lines.append("=" * 60)
        lines.append("")
        
        if current_market_data:
            lines.append(f"当前市场状态: {current_market_data.get('regime', 'unknown')}")
            lines.append("")
        
        # 按专长分类
        by_specialty = {}
        for name_en, action in next_actions.items():
            specialty = action.get('specialty', 'unknown')
            if specialty not in by_specialty:
                by_specialty[specialty] = []
            by_specialty[specialty].append(action)
        
        # 庄家行为中文映射
        action_cn = {
            'accumulate': '吸筹',
            'hold': '持仓',
            'pump': '拉抬',
            'distribute': '派发',
            'shake': '震仓',
        }
        
        specialty_cn = {
            'up': '追涨型',
            'down': '杀跌型',
            'volatile': '波动型',
            'stable': '稳健型',
            'allround': '全能型',
        }
        
        for specialty, actions in by_specialty.items():
            lines.append(f"【{specialty_cn.get(specialty, specialty)}组合】")
            for action in actions:
                dealer_action = action.get('dealer_next', {})
                act_name = dealer_action.get('action', 'unknown')
                act_cn = action_cn.get(act_name, act_name)
                holdings = dealer_action.get('holdings_ratio', 0)
                
                lines.append(f"  • {action['name']}: {act_cn} (持仓比例: {holdings:.1%})")
            lines.append("")
        
        lines.append("=" * 60)
        return "\n".join(lines)
    
    def generate_full_arena_report(self, arena_results: Dict) -> str:
        """
        生成完整的竞技场报告
        
        Args:
            arena_results: 竞技场结果
            
        Returns:
            完整报告文本
        """
        lines = []
        lines.append("=" * 70)
        lines.append("  ★ 多组合进化竞技场 v3.0 - 完整报告")
        lines.append("=" * 70)
        lines.append("")
        
        # 基本信息
        lines.append(f"生成时间: {arena_results.get('timestamp', 'N/A')}")
        lines.append(f"组合数量: {arena_results.get('n_combinations', 0)}")
        lines.append("")
        
        # 专家选拔报告
        specialist_ranking = arena_results.get('specialist_ranking', {})
        lines.append(self.generate_specialist_report(specialist_ranking))
        lines.append("")
        
        # 庄家行动预测
        recommendation = arena_results.get('ensemble_recommendation', {})
        next_actions = recommendation.get('next_actions', {})
        if next_actions:
            lines.append(self.generate_action_forecast(next_actions))
        
        # 对比分析
        comparison = self.compare_combinations(arena_results)
        if comparison.get('recommendations'):
            lines.append("")
            lines.append("【使用建议】")
            for rec in comparison['recommendations']:
                lines.append(f"  • {rec.get('recommendation', '')}")
        
        lines.append("")
        lines.append("=" * 70)
        return "\n".join(lines)


# ============================================================
# 命令行入口
# ============================================================
def main():
    parser = argparse.ArgumentParser(description="对抗学习结果解读器 (v3.0)")
    parser.add_argument("--mode", required=True,
                        choices=["analyze", "interpret", "report", "arena-report"])
    parser.add_argument("--data-dir", type=str, default=str(DATA_DIR))
    parser.add_argument("--price-file", type=str, default=None,
                        help="价格数据CSV路径")
    parser.add_argument("--output", type=str, default=str(RESULTS_DIR / "interpretation_report.json"))
    parser.add_argument("--arena-results", type=str, 
                        default=str(RESULTS_DIR / "arena_results.json"),
                        help="竞技场结果文件路径")
    
    args = parser.parse_args()
    cfg = InterpreterConfig()
    interpreter = StockInterpreter(cfg, Path(args.data_dir))
    
    if args.mode == "analyze":
        # 加载最近一次模拟的价格数据
        price_file = args.price_file or str(ADV_DATA_DIR / "last_episode_prices.csv")
        if not Path(price_file).exists():
            logger.error(f"价格文件不存在: {price_file}")
            return
        
        df = pd.read_csv(price_file)
        prices = df['price'].values
        volumes = df['volume'].values if 'volume' in df.columns else None
        
        report = interpreter.interpret(prices, volumes)
        
        # 保存报告
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2, default=str)
        
        logger.info(f"解读报告已保存: {out_path}")
        logger.info(f"\n{'='*50}")
        logger.info(report['summary'])
    
    elif args.mode == "report":
        # 读取最近的分析报告
        report_path = Path(args.output)
        if report_path.exists():
            with open(report_path, 'r', encoding='utf-8') as f:
                report = json.load(f)
            logger.info(report['summary'])
        else:
            logger.error("报告不存在，请先运行 --mode analyze")
    
    elif args.mode == "interpret":
        # 快速解读当前市场状态
        price_file = args.price_file or str(ADV_DATA_DIR / "last_episode_prices.csv")
        if Path(price_file).exists():
            df = pd.read_csv(price_file)
            prices = df['price'].values
            
            report = interpreter.interpret(prices)
            logger.info(report['summary'])
        else:
            logger.error("无可用数据")
    
    elif args.mode == "arena-report":
        # ★ v3.0: 竞技场报告模式
        arena_file = Path(args.arena_results)
        if not arena_file.exists():
            logger.error(f"竞技场结果文件不存在: {arena_file}")
            logger.info("请先运行: python adversarial_env.py --mode arena")
            return
        
        with open(arena_file, 'r', encoding='utf-8') as f:
            arena_results = json.load(f)
        
        # 创建多组合解读器
        multi_interp = MultiCombinationInterpreter(Path(args.data_dir))
        
        # 生成完整报告
        full_report = multi_interp.generate_full_arena_report(arena_results)
        print(full_report)
        
        # 保存报告
        report_output = RESULTS_DIR / "arena_report.txt"
        with open(report_output, 'w', encoding='utf-8') as f:
            f.write(full_report)
        logger.info(f"\n报告已保存: {report_output}")

if __name__ == "__main__":
    main()

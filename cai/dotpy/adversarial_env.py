"""
adversarial_env.py - 庄散对抗学习环境 (v2.1 神经放电版)
======================================================
Phase 3: 庄散对抗（核心）

三方博弈:
  庄家(Dealer): 防共四策 → 思想防散/民众防散/政治防散/武力防散
  散户(Retailer): 五类型 → 跟风/价值/技术/散户头羊/被动
  游资(HotMoney): 打板快进快出

两层博弈:
  显式博弈: 买卖行为直接交互
  元博弈: 策略进化(拉马克+达尔文)

市场真实性机制 (v2.1新增):
  ★ 神经元放电机制: 63.21%阈值全或无行情
  ★ 涨跌停板制度: 主板±10%/创业板±20%
  ★ 市场冲击模型: 大单滑点+永久性冲击
  ★ 市场状态检测: 趋势/震荡/转折三态
  ★ T+1交易约束: 散户当日买入不可卖出
  ★ 隔夜跳空: 模拟开盘集合竞价缺口
  ★ 买卖盘压力累积: 为放电提供驱动力

防摆烂: 五方案保证散户不会退化到无策略

用法:
  python adversarial_env.py --mode train [--episodes 1000] [--evolve]
  python adversarial_env.py --mode evaluate [--model-path ...]
  python adversarial_env.py --mode demo
"""

import os, sys, argparse, json, time, copy, warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from collections import defaultdict
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
from enum import Enum, auto

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical

sys.path.insert(0, str(Path(__file__).parent))
from stock_config import (
    ADV_MODEL_DIR, ADV_DATA_DIR, RESULTS_DIR, SCRIPT_DIR,
    AdversarialConfig, setup_logger,
)

warnings.filterwarnings('ignore')
logger = setup_logger("Adversarial")


# ============================================================
#  市场状态枚举
# ============================================================
class MarketPhase(Enum):
    """市场阶段: 与神经放电一一对应"""
    RESTING      = auto()  # 静息: 无明显趋势, 买卖压力均衡
    ACCUMULATING = auto()  # 蓄积: 压力在暗中累积, 价格微动
    FIRING       = auto()  # 放电: 突破阈值, 全或无行情爆发
    REFRACTORY   = auto()  # 不应期: 放电后的整理消化, 价格回归


class TrendDirection(Enum):
    """趋势方向"""
    UP     = auto()
    DOWN   = auto()
    FLAT   = auto()


# ============================================================
#  ★ 神经元放电机制
# ============================================================
class NeuralFiringMechanism:
    """
    神经元放电机制: 模拟A股"全或无"行情
    
    核心思想:
      A股经常一涨涨不停、一跌跌不休，这和神经元动作电位极其相似:
        - 静息状态下, 买卖压力在暗中累积 (膜电位缓慢去极化)
        - 当压力超过阈值(63.21% = 1-1/e), 突然爆发方向性行情 (放电)
        - 放电后进入不应期, 市场消化整理 (复极化)
        - 不应期结束后回到静息, 压力重新累积
    
    数学模型:
      membrane_potential(t) = Σ(买入压力) - Σ(卖出压力)
      firing_threshold = 0.6321  (63.21% = 1 - 1/e)
      
      当 |membrane_potential| / normalization ≥ firing_threshold:
        → 触发放电
        → 价格突跳 = base_step × firing_amplitude
        → 进入不应期
    
    不应期模型:
      refractory_decay = exp(-t / τ)  τ=15 ticks
      不应期结束条件: refractory_decay < 0.1 或超时
    """
    
    FIRING_THRESHOLD = 0.6321       # 63.21% = 1 - 1/e
    REFRACTORY_TAU   = 15.0         # 不应期时间常数 (ticks)
    REFRACTORY_FLOOR = 0.1          # 不应期结束阈值
    BASE_AMPLITUDE   = 0.015        # 基础放电幅度 (1.5%价格跳变)
    MAX_AMPLITUDE    = 0.06          # 最大放电幅度 (6%)
    PRESSURE_DECAY   = 0.95          # 压力自然衰减 (每tick衰减5%)
    MIN_REFRACTORY   = 5             # 最短不应期 (ticks)
    MAX_REFRACTORY   = 30            # 最长不应期 (ticks)
    
    def __init__(self, initial_price: float = 10.0):
        self.initial_price = initial_price
        self.reset()
    
    def reset(self):
        """重置到静息状态"""
        self.phase = MarketPhase.RESTING
        self.trend = TrendDirection.FLAT
        self.membrane_potential = 0.0      # 膜电位 (买卖压力差)
        self.pressure_accumulator = 0.0    # 累积压力
        self.refractory_timer = 0          # 不应期计时器
        self.refractory_strength = 0.0    # 不应期残余强度
        self.firing_count = 0              # 放电次数
        self.last_firing_direction = None   # 上次放电方向
        self.consecutive_firings = 0        # 连续同向放电次数
        self.ticks_since_last_firing = 0    # 距上次放电的tick数
        
        # 历史记录 (供监察脚本读取)
        self.firing_events = []            # [{tick, direction, amplitude, pressure}]
        self.phase_history = []            # [(tick, phase)]
    
    def update(self, buy_pressure: float, sell_pressure: float,
               current_price: float, tick: int) -> Dict:
        """
        每个tick更新放电状态
        
        Args:
            buy_pressure: 买入压力 (归一化后的买盘总量)
            sell_pressure: 卖出压力 (归一化后的卖盘总量)
            current_price: 当前中间价
            tick: 当前tick编号
            
        Returns:
            {phase, price_impact, fired, direction, amplitude}
        """
        self.ticks_since_last_firing += 1
        
        # ── 根据当前阶段分别处理 ──
        if self.phase == MarketPhase.REFRACTORY:
            return self._update_refractory(buy_pressure, sell_pressure,
                                           current_price, tick)
        else:
            return self._update_active(buy_pressure, sell_pressure,
                                        current_price, tick)
    
    def _update_refractory(self, buy_pressure: float, sell_pressure: float,
                           current_price: float, tick: int) -> Dict:
        """不应期更新: 压力累积但不会触发放电"""
        self.refractory_timer += 1
        
        # 不应期衰减
        self.refractory_strength = np.exp(-self.refractory_timer / self.REFRACTORY_TAU)
        
        # 即使在不应期, 压力仍在累积 (只是被压制)
        net_pressure = buy_pressure - sell_pressure
        # 压力累积受不应期抑制
        suppressed = net_pressure * (1.0 - self.refractory_strength * 0.7)
        self.pressure_accumulator = (
            self.pressure_accumulator * self.PRESSURE_DECAY + suppressed
        )
        
        # 检查不应期是否结束
        if (self.refractory_strength < self.REFRACTORY_FLOOR and 
            self.refractory_timer >= self.MIN_REFRACTORY):
            self.phase = MarketPhase.RESTING
            self.phase_history.append((tick, MarketPhase.RESTING))
        elif self.refractory_timer >= self.MAX_REFRACTORY:
            # 强制结束不应期
            self.phase = MarketPhase.RESTING
            self.refractory_strength = 0.0
            self.phase_history.append((tick, MarketPhase.RESTING))
        
        # 不应期内价格微调 (回归均值倾向)
        price_impact = 0.0
        if self.trend == TrendDirection.UP:
            price_impact = -0.0003 * self.refractory_strength  # 小幅回落
        elif self.trend == TrendDirection.DOWN:
            price_impact = 0.0003 * self.refractory_strength  # 小幅反弹
        
        return {
            'phase': self.phase,
            'price_impact': price_impact,
            'fired': False,
            'direction': self.trend,
            'amplitude': 0.0,
        }
    
    def _update_active(self, buy_pressure: float, sell_pressure: float,
                       current_price: float, tick: int) -> Dict:
        """活跃期更新: 压力累积, 达到阈值则放电"""
        # 净压力
        net_pressure = buy_pressure - sell_pressure
        
        # 压力累积 (带衰减)
        self.pressure_accumulator = (
            self.pressure_accumulator * self.PRESSURE_DECAY + net_pressure
        )
        
        # 更新膜电位 (归一化: 用初始价格的1%作为归一化因子)
        normalization = current_price * 0.01
        self.membrane_potential = self.pressure_accumulator / (normalization + 1e-8)
        
        # ── 判断是否达到放电阈值 ──
        normalized_potential = abs(self.membrane_potential)
        
        if normalized_potential >= self.FIRING_THRESHOLD:
            return self._fire(current_price, tick)
        
        # ── 未达阈值: 判断蓄积还是静息 ──
        if normalized_potential > self.FIRING_THRESHOLD * 0.3:
            # 超过阈值30% → 进入蓄积期
            if self.phase != MarketPhase.ACCUMULATING:
                self.phase = MarketPhase.ACCUMULATING
                self.phase_history.append((tick, MarketPhase.ACCUMULATING))
        else:
            if self.phase != MarketPhase.RESTING:
                self.phase = MarketPhase.RESTING
                self.phase_history.append((tick, MarketPhase.RESTING))
        
        # 非放电期: 价格按压力方向微调
        price_impact = self.membrane_potential * 0.001  # 微弱趋势
        
        # 判断趋势方向
        if self.membrane_potential > 0.1:
            self.trend = TrendDirection.UP
        elif self.membrane_potential < -0.1:
            self.trend = TrendDirection.DOWN
        else:
            self.trend = TrendDirection.FLAT
        
        return {
            'phase': self.phase,
            'price_impact': price_impact,
            'fired': False,
            'direction': self.trend,
            'amplitude': 0.0,
        }
    
    def _fire(self, current_price: float, tick: int) -> Dict:
        """
        触发放电: 全或无行情爆发
        
        放电幅度:
          - 基础幅度 = BASE_AMPLITUDE
          - 连续同向放电: 幅度递增 (正反馈, 模拟涨不停/跌不休)
          - 反向放电: 幅度较小 (反转需要更多能量)
          - 上限 = MAX_AMPLITUDE
        """
        # 放电方向
        if self.membrane_potential > 0:
            direction = TrendDirection.UP
        else:
            direction = TrendDirection.DOWN
        
        # ── 放电幅度计算 ──
        amplitude = self.BASE_AMPLITUDE
        
        # 连续同向放电: 正反馈放大 (模拟趋势强化)
        if direction == self.last_firing_direction:
            self.consecutive_firings += 1
            # 连续放大: 1.0 → 1.3 → 1.6 → 1.9 → 2.2 ...
            boost = 1.0 + 0.3 * min(self.consecutive_firings, 5)
            amplitude *= boost
        else:
            self.consecutive_firings = 0
            # 反向放电: 幅度打折 (惯性阻力)
            if self.last_firing_direction is not None:
                amplitude *= 0.7
        
        # 超过阈值的程度越大, 放电越强
        overshoot = abs(self.membrane_potential) / self.FIRING_THRESHOLD - 1.0
        amplitude *= (1.0 + min(overshoot, 1.0))
        
        # 上限截断
        amplitude = min(amplitude, self.MAX_AMPLITUDE)
        
        # 价格影响
        if direction == TrendDirection.UP:
            price_impact = amplitude
        else:
            price_impact = -amplitude
        
        # ── 放电后状态更新 ──
        self.firing_count += 1
        self.last_firing_direction = direction
        self.trend = direction
        
        # 重置压力累积 (放电释放了累积压力)
        self.pressure_accumulator *= 0.2  # 保留20%残余压力
        self.membrane_potential *= 0.2
        
        # 进入不应期
        self.phase = MarketPhase.FIRING  # 先标记为放电
        self.refractory_timer = 0
        self.refractory_strength = 1.0
        # 放电后立即转为不应期
        self.phase = MarketPhase.REFRACTORY
        self.ticks_since_last_firing = 0
        
        # 记录事件
        event = {
            'tick': tick,
            'direction': direction.name,
            'amplitude': amplitude,
            'pressure_before': self.pressure_accumulator,
            'consecutive': self.consecutive_firings,
        }
        self.firing_events.append(event)
        self.phase_history.append((tick, MarketPhase.FIRING))
        
        logger.debug(f"  ⚡ 放电! 方向={direction.name} 幅度={amplitude:.3%} "
                    f"连续={self.consecutive_firings} tick={tick}")
        
        return {
            'phase': MarketPhase.FIRING,
            'price_impact': price_impact,
            'fired': True,
            'direction': direction,
            'amplitude': amplitude,
        }
    
    def get_state(self) -> Dict:
        """获取当前放电机制状态 (供监察/Agent观察)"""
        return {
            'phase': self.phase.name,
            'trend': self.trend.name,
            'membrane_potential': self.membrane_potential,
            'pressure_accumulator': self.pressure_accumulator,
            'normalized_potential': abs(self.membrane_potential) / self.FIRING_THRESHOLD,
            'refractory_strength': self.refractory_strength,
            'firing_count': self.firing_count,
            'consecutive_firings': self.consecutive_firings,
            'ticks_since_firing': self.ticks_since_last_firing,
            'is_firing_eligible': (
                self.phase != MarketPhase.REFRACTORY and 
                abs(self.membrane_potential) >= self.FIRING_THRESHOLD * 0.3
            ),
        }


# ============================================================
#  ★ 涨跌停板制度
# ============================================================
class PriceLimitManager:
    """
    涨跌停板制度: A股核心风控机制
    
    规则:
      - 主板: ±10%
      - 创业板/科创板(300xxx/688xxx): ±20%
      - ST股: ±5%
      - 北交所(8xxxxx/4xxxxx): ±30%
      - 新股上市首日: 不设限制 (简化处理)
    
    涨跌停对市场的影响:
      1. 瓶颈效应: 大量未成交挂单堆积在涨停/跌停价
      2. 负反馈: 涨停封不住 → 卖压增大 → 可能回落
      3. 连板效应: 连续涨停形成"连板" (与神经放电的连续放电对应)
      4. 隔夜压力: 涨停板上的未成交买单 → 次日跳空高开
    """
    
    MAINBOARD_LIMIT    = 0.10   # ±10%
    CHINEXT_LIMIT      = 0.20   # ±20% (创业板/科创板)
    ST_LIMIT           = 0.05   # ±5%
    BSE_LIMIT          = 0.30   # ±30% (北交所)
    IPO_FIRST_DAY_LIMIT = 0.44  # 首日±44% (简化)
    
    def __init__(self, stock_code: str = "", daily_open: float = 10.0):
        self.stock_code = stock_code
        self.daily_open = daily_open
        self.limit_pct = self._determine_limit()
        self.upper_limit = daily_open * (1 + self.limit_pct)
        self.lower_limit = daily_open * (1 - self.limit_pct)
        
        # 连板统计
        self.consecutive_limit_up = 0
        self.consecutive_limit_down = 0
        self.hit_limit_up_today = False
        self.hit_limit_down_today = False
        
        # 涨跌停封单量 (模拟封板强度)
        self.limit_up_volume = 0.0    # 涨停价上的买单量
        self.limit_down_volume = 0.0  # 跌停价上的卖单量
    
    def _determine_limit(self) -> float:
        """根据股票代码确定涨跌停幅度"""
        code = self.stock_code.replace('.SH', '').replace('.SZ', '').replace('.BJ', '')
        
        if code.startswith('688'):    # 科创板
            return self.CHINEXT_LIMIT
        elif code.startswith('300') or code.startswith('301'):  # 创业板
            return self.CHINEXT_LIMIT
        elif code.startswith(('4', '8')):  # 北交所
            return self.BSE_LIMIT
        elif code.startswith('ST') or 'ST' in self.stock_code.upper():
            return self.ST_LIMIT
        else:
            return self.MAINBOARD_LIMIT
    
    def reset_daily(self, new_open: float):
        """每日重置 (新的开盘价)"""
        self.daily_open = new_open
        self.upper_limit = new_open * (1 + self.limit_pct)
        self.lower_limit = new_open * (1 - self.limit_pct)
        
        if self.hit_limit_up_today:
            self.consecutive_limit_up += 1
        else:
            self.consecutive_limit_up = 0
        
        if self.hit_limit_down_today:
            self.consecutive_limit_down += 1
        else:
            self.consecutive_limit_down = 0
        
        self.hit_limit_up_today = False
        self.hit_limit_down_today = False
        self.limit_up_volume = 0.0
        self.limit_down_volume = 0.0
    
    def clamp_price(self, price: float) -> float:
        """将价格限制在涨跌停范围内"""
        return max(self.lower_limit, min(self.upper_limit, price))
    
    def is_at_limit_up(self, price: float) -> bool:
        """是否触及涨停"""
        return price >= self.upper_limit * 0.998  # 容差0.2%
    
    def is_at_limit_down(self, price: float) -> bool:
        """是否触及跌停"""
        return price <= self.lower_limit * 1.002
    
    def check_order(self, side: str, price: float) -> Tuple[bool, float]:
        """
        检查订单是否在涨跌停范围内
        
        Returns:
            (allowed, adjusted_price)
        """
        adjusted = self.clamp_price(price)
        
        if side == 'buy' and price > self.upper_limit:
            # 买入价超过涨停价 → 不允许 (涨停买不进)
            return False, adjusted
        elif side == 'sell' and price < self.lower_limit:
            # 卖出价低于跌停价 → 不允许 (跌停放不出)
            return False, adjusted
        
        return True, adjusted
    
    def record_trade(self, price: float, volume: float, side: str):
        """记录成交, 更新涨跌停状态"""
        if self.is_at_limit_up(price):
            self.hit_limit_up_today = True
            if side == 'buy':
                self.limit_up_volume += volume
        elif self.is_at_limit_down(price):
            self.hit_limit_down_today = True
            if side == 'sell':
                self.limit_down_volume += volume
    
    def get_state(self) -> Dict:
        return {
            'upper_limit': self.upper_limit,
            'lower_limit': self.lower_limit,
            'limit_pct': self.limit_pct,
            'consecutive_limit_up': self.consecutive_limit_up,
            'consecutive_limit_down': self.consecutive_limit_down,
            'at_limit_up': self.hit_limit_up_today,
            'at_limit_down': self.hit_limit_down_today,
            'limit_up_volume': self.limit_up_volume,
            'limit_down_volume': self.limit_down_volume,
        }


# ============================================================
#  ★ 市场冲击模型
# ============================================================
class MarketImpactModel:
    """
    市场冲击模型: 大单造成价格滑点
    
    基于 Almgren-Chriss 框架简化版:
      临时冲击 (Temporary Impact):
        slippage = σ × (volume/V)^(1/2) × sign
        
      永久冲击 (Permanent Impact):
        permanent_shift = η × (volume/V) × sign
        
    其中:
      V = 日均成交量 (从历史数据估算)
      σ = 日波动率
      η = 永久冲击系数
    
    效果:
      - 大单执行时产生滑点 (成交价劣于挂单价)
      - 大单永久性移动中间价 (价格发现)
      - 小单几乎无冲击 (流动性充足)
    """
    
    # 默认参数
    DEFAULT_DAILY_VOLUME = 1e6      # 默认日均成交量 (股)
    DEFAULT_DAILY_VOLATILITY = 0.02  # 默认日波动率 2%
    TEMP_IMPACT_COEFF = 0.5         # 临时冲击系数 (半衰)
    PERM_IMPACT_COEFF = 0.1          # 永久冲击系数
    
    def __init__(self, daily_volume: float = None, daily_volatility: float = None):
        self.daily_volume = daily_volume or self.DEFAULT_DAILY_VOLUME
        self.daily_volatility = daily_volatility or self.DEFAULT_DAILY_VOLATILITY
        self.tick_volume_scale = self.daily_volume / 240  # 每tick均量
    
    def compute_impact(self, order_volume: float, side: str,
                       current_price: float) -> Dict:
        """
        计算订单的市场冲击
        
        Returns:
            {slippage_pct, permanent_shift_pct, adjusted_price}
        """
        if order_volume <= 0:
            return {'slippage_pct': 0.0, 'permanent_shift_pct': 0.0}
        
        # 参与率: 订单量 / 该tick均量
        participation = order_volume / (self.tick_volume_scale + 1e-8)
        
        # 临时冲击 (滑点): 平方根模型
        temp_impact = (self.daily_volatility * 
                      self.TEMP_IMPACT_COEFF * 
                      np.sqrt(participation))
        
        # 永久冲击: 线性模型
        perm_impact = (self.daily_volatility *
                      self.PERM_IMPACT_COEFF *
                      participation)
        
        # 方向符号
        sign = 1.0 if side == 'buy' else -1.0
        
        return {
            'slippage_pct': temp_impact * sign,        # 正=买时向上滑, 负=卖时向下滑
            'permanent_shift_pct': perm_impact * sign,  # 同上
            'participation_rate': participation,
        }
    
    def apply_slippage(self, fill_price: float, order_volume: float,
                       side: str) -> float:
        """在成交价上应用滑点"""
        impact = self.compute_impact(order_volume, side, fill_price)
        # 买入时价格上滑, 卖出时价格下滑
        adjusted = fill_price * (1 + impact['slippage_pct'])
        return adjusted


# ============================================================
#  ★ 市场状态检测器
# ============================================================
class MarketRegimeDetector:
    """
    市场状态检测: 趋势/震荡/转折
    
    方法:
      1. ADX (平均方向指数) 简化版: 衡量趋势强度
      2. 波动率比率: 当前vol / 历史vol
      3. 成交量加权: 量价配合判断
      4. 与神经放电联动: 
         - FIRING → 趋势
         - RESTING → 震荡
         - ACCUMULATING → 转折前夜
    """
    
    WINDOW = 20  # 检测窗口
    
    def __init__(self):
        self.prices = []
        self.volumes = []
        self.regime = 'ranging'     # trending / ranging / transition
        self.trend_strength = 0.0   # 0~1, 趋势强度
        self.volatility_ratio = 1.0
    
    def update(self, price: float, volume: float = 0,
               firing_phase: str = 'RESTING') -> Dict:
        """更新市场状态"""
        self.prices.append(price)
        if volume > 0:
            self.volumes.append(volume)
        
        if len(self.prices) < self.WINDOW:
            self.regime = 'ranging'
            self.trend_strength = 0.0
            return self.get_state()
        
        window = self.prices[-self.WINDOW:]
        
        # ── 简化ADX ──
        returns = np.diff(window) / (np.array(window[:-1]) + 1e-8)
        pos_moves = np.sum(returns > 0)
        neg_moves = np.sum(returns < 0)
        total = len(returns)
        
        directional_index = abs(pos_moves - neg_moves) / max(total, 1)
        
        # ── 波动率比率 ──
        recent_vol = np.std(returns[-10:]) if len(returns) >= 10 else np.std(returns)
        hist_vol = np.std(returns) if len(returns) >= 5 else 0.02
        self.volatility_ratio = recent_vol / (hist_vol + 1e-8)
        
        # ── 综合判定 ──
        # 神经放电直接映射为趋势
        if firing_phase == 'FIRING':
            self.regime = 'trending'
            self.trend_strength = min(1.0, 0.7 + directional_index * 0.3)
        elif firing_phase == 'ACCUMULATING':
            self.regime = 'transition'
            self.trend_strength = 0.3 + directional_index * 0.3
        elif firing_phase == 'REFRACTORY':
            # 不应期可能反转也可能延续
            if self.volatility_ratio > 1.5:
                self.regime = 'transition'
            else:
                self.regime = 'ranging'
            self.trend_strength = 0.2
        else:  # RESTING
            if directional_index > 0.4:
                self.regime = 'trending'
                self.trend_strength = directional_index
            else:
                self.regime = 'ranging'
                self.trend_strength = directional_index * 0.5
        
        return self.get_state()
    
    def get_state(self) -> Dict:
        return {
            'regime': self.regime,
            'trend_strength': self.trend_strength,
            'volatility_ratio': self.volatility_ratio,
        }


# ============================================================
#  订单簿 (增强版)
# ============================================================
class OrderBook:
    """
    增强订单簿: 
      - 支撑价格发现和成交
      - 集成涨跌停板
      - 集成市场冲击
      - 集成神经放电
      - 买卖盘压力计算
    """
    
    def __init__(self, initial_price: float = 10.0,
                 stock_code: str = "",
                 impact_model: MarketImpactModel = None):
        self.initial_price = initial_price
        self.mid_price = initial_price
        self.bid_price = initial_price * 0.999
        self.ask_price = initial_price * 1.001
        self.spread = 0.002
        self.daily_open = initial_price   # 当日开盘价 (用于涨跌停)
        
        # 子模块
        self.price_limit = PriceLimitManager(stock_code, initial_price)
        self.impact_model = impact_model or MarketImpactModel()
        self.firing = NeuralFiringMechanism(initial_price)
        self.regime_detector = MarketRegimeDetector()
        
        # 买卖队列
        self.bids = []
        self.asks = []
        
        # 成交记录
        self.trades = []
        self.price_history = [initial_price]
        self.volume_history = [0]
        self.tick = 0
        
        # 压力追踪
        self.buy_pressure = 0.0
        self.sell_pressure = 0.0
    
    def reset(self, initial_price: float = 10.0):
        self.__init__(initial_price)
    
    def submit_order(self, side: str, volume: float, price: Optional[float],
                     agent_id: str, order_type: str = "limit") -> Dict:
        """
        提交订单 (增强版: 含涨跌停+冲击+放电)
        """
        if price is None:
            price = self.ask_price if side == 'buy' else self.bid_price
        
        # ── 涨跌停检查 ──
        allowed, adjusted_price = self.price_limit.check_order(side, price)
        if not allowed:
            # 超过涨跌停, 挂单但不成交 (排队的封单)
            if side == 'buy' and self.price_limit.is_at_limit_up(self.mid_price):
                # 涨停板上排队买入
                self.price_limit.limit_up_volume += volume
            elif side == 'sell' and self.price_limit.is_at_limit_down(self.mid_price):
                # 跌停板上排队卖出
                self.price_limit.limit_down_volume += volume
            return {'filled': 0, 'avg_price': adjusted_price, 'remaining': volume,
                    'blocked_by': 'price_limit'}
        
        price = adjusted_price
        
        # ── 撮合 ──
        filled = 0.0
        fill_price = 0.0
        
        if side == 'buy':
            remaining = volume
            new_asks = []
            for ask_p, ask_v, ask_id in self.asks:
                if remaining <= 0:
                    new_asks.append((ask_p, ask_v, ask_id))
                    continue
                if ask_p <= price:
                    fill_v = min(remaining, ask_v)
                    filled += fill_v
                    
                    # 应用市场冲击滑点
                    actual_price = self.impact_model.apply_slippage(
                        ask_p, fill_v, 'buy')
                    fill_price += fill_v * actual_price
                    remaining -= fill_v
                    
                    if ask_v > fill_v:
                        new_asks.append((ask_p, ask_v - fill_v, ask_id))
                    
                    self.trades.append({
                        'tick': self.tick, 'price': actual_price,
                        'volume': fill_v, 'buyer': agent_id, 'seller': ask_id,
                    })
                    self.price_limit.record_trade(actual_price, fill_v, 'buy')
                else:
                    new_asks.append((ask_p, ask_v, ask_id))
            self.asks = new_asks
            
            if remaining > 0 and order_type == "limit":
                self.bids.append((price, remaining, agent_id))
                self.bids.sort(key=lambda x: -x[0])
        
        else:  # sell
            remaining = volume
            new_bids = []
            for bid_p, bid_v, bid_id in self.bids:
                if remaining <= 0:
                    new_bids.append((bid_p, bid_v, bid_id))
                    continue
                if bid_p >= price:
                    fill_v = min(remaining, bid_v)
                    filled += fill_v
                    
                    actual_price = self.impact_model.apply_slippage(
                        bid_p, fill_v, 'sell')
                    fill_price += fill_v * actual_price
                    remaining -= fill_v
                    
                    if bid_v > fill_v:
                        new_bids.append((bid_p, bid_v - fill_v, bid_id))
                    
                    self.trades.append({
                        'tick': self.tick, 'price': actual_price,
                        'volume': fill_v, 'buyer': bid_id, 'seller': agent_id,
                    })
                    self.price_limit.record_trade(actual_price, fill_v, 'sell')
                else:
                    new_bids.append((bid_p, bid_v, bid_id))
            self.bids = new_bids
            
            if remaining > 0 and order_type == "limit":
                self.asks.append((price, remaining, agent_id))
                self.asks.sort(key=lambda x: x[0])
        
        avg_price = fill_price / max(filled, 1e-8)
        
        # ── 更新价格 ──
        if filled > 0:
            new_mid = avg_price
            
            # 永久冲击: 大单移动中间价
            impact = self.impact_model.compute_impact(filled, side, new_mid)
            new_mid *= (1 + impact['permanent_shift_pct'])
            
            # 涨跌停截断
            new_mid = self.price_limit.clamp_price(new_mid)
            
            self.mid_price = new_mid
            self.bid_price = self.mid_price * (1 - self.spread / 2)
            self.ask_price = self.mid_price * (1 + self.spread / 2)
            
            self.price_history.append(self.mid_price)
            self.volume_history.append(filled)
            
            # 更新买卖压力
            if side == 'buy':
                self.buy_pressure += filled
            else:
                self.sell_pressure += filled
        
        return {'filled': filled, 'avg_price': avg_price, 'remaining': volume - filled}
    
    def update_firing(self, tick: int):
        """
        每tick更新神经放电机制
        (在所有订单撮合完成后调用)
        """
        # 归一化买卖压力
        total_pressure = self.buy_pressure + self.sell_pressure + 1e-8
        norm_buy = self.buy_pressure / total_pressure
        norm_sell = self.sell_pressure / total_pressure
        
        # 更新放电机制
        result = self.firing.update(
            norm_buy, norm_sell, self.mid_price, tick
        )
        
        # 如果触发放电, 直接影响价格
        if result['fired']:
            new_price = self.mid_price * (1 + result['price_impact'])
            new_price = self.price_limit.clamp_price(new_price)
            self.mid_price = new_price
            self.bid_price = self.mid_price * (1 - self.spread / 2)
            self.ask_price = self.mid_price * (1 + self.spread / 2)
            self.price_history.append(self.mid_price)
            self.volume_history.append(0)  # 放电本身不产生额外成交量
        
        # 更新市场状态
        self.regime_detector.update(
            self.mid_price,
            self.volume_history[-1] if self.volume_history else 0,
            self.firing.phase.name
        )
        
        # 重置tick压力 (不累积到下一tick)
        self.buy_pressure *= 0.3   # 保留30%残余
        self.sell_pressure *= 0.3
    
    def get_state(self) -> Dict:
        """获取完整市场状态"""
        bid_depth = sum(v for _, v, _ in self.bids[:5])
        ask_depth = sum(v for _, v, _ in self.asks[:5])
        
        return {
            'mid_price': self.mid_price,
            'bid_price': self.bid_price,
            'ask_price': self.ask_price,
            'spread': self.ask_price - self.bid_price,
            'bid_depth': bid_depth,
            'ask_depth': ask_depth,
            'bid_ask_ratio': bid_depth / max(ask_depth, 1e-8),
            'tick': self.tick,
            'recent_trades': self.trades[-10:],
            # 新增
            'firing': self.firing.get_state(),
            'regime': self.regime_detector.get_state(),
            'price_limit': self.price_limit.get_state(),
            'buy_pressure': self.buy_pressure,
            'sell_pressure': self.sell_pressure,
        }


# ============================================================
#  市场环境 (增强版)
# ============================================================
class MarketEnv:
    """
    增强市场环境: 包装OrderBook + 管理时间推进 + 隔夜跳空
    
    一个episode = 1个交易日 = 240个tick (4h×60min)
    日内时间映射:
      9:30-11:30  = tick 0~119  (上午盘)
      11:30-13:00 = 午休 (不交易)
      13:00-15:00 = tick 120~239 (下午盘)
    
    ★ v2.2 价格基准线: 支持从假数据/实盘数据加载价格序列作为市场走势锚点
      - 有基准线时: Agent订单只影响偏离度, 大趋势跟随基准线
      - 无基准线时: 退回纯模拟模式(兼容旧逻辑)
    """
    
    # 隔夜跳空参数
    GAP_PROBABILITY = 0.4     # 40%概率出现隔夜跳空
    GAP_MEAN = 0.005          # 跳空均值 0.5%
    GAP_STD = 0.015           # 跳空标准差 1.5%
    
    # ★ v2.2 基准线参数
    BENCHMARK_BLEND = 0.85     # 基准线权重: 85%跟随基准 + 15%Agent订单影响
    
    def __init__(self, cfg: AdversarialConfig, stock_code: str = "",
                 price_benchmark: np.ndarray = None):
        """
        price_benchmark: (num_days, T, F) 或 (num_days, T) numpy数组
          - 若3D: 取第0列(close)作为基准价格序列
          - 若2D: 直接作为基准价格序列 (num_days个交易日, 每日T个tick的价格)
          - 若None: 纯模拟模式
        """
        self.cfg = cfg
        self.stock_code = stock_code
        
        # ★ v2.2 价格基准线
        self._setup_benchmark(price_benchmark)
        
        # 用基准线首个价格作为初始价(如果有)
        init_price = cfg.initial_price
        if self.benchmark_prices is not None and len(self.benchmark_prices) > 0:
            init_price = float(self.benchmark_prices[0][0])
            logger.info(f"  ★ 价格基准线已加载: {len(self.benchmark_prices)}个交易日, "
                       f"基准价={init_price:.2f}, 融合比={self.BENCHMARK_BLEND}")
        
        self.orderbook = OrderBook(init_price, stock_code)
        self.tick = 0
        self.episode_length = cfg.episode_length
        self._current_day_idx = 0  # ★ v2.2: 当前基准线日索引
        
        # 市场事件
        self.info_events = []
        self.shock_events = []
        
        # 日内统计
        self.intraday_high = cfg.initial_price
        self.intraday_low = cfg.initial_price
        self.day_count = 0
    
    def _setup_benchmark(self, price_benchmark):
        """★ v2.2: 设置价格基准线, 将生成数据/实盘数据转为每tick价格序列"""
        self.benchmark_prices = None  # list of arrays, 每个array是该日的240个tick价格
        
        if price_benchmark is None:
            return
        
        try:
            data = np.array(price_benchmark)
            if data.ndim == 3:
                # (num_days, T, F) → 取close列(第0列, 因为normalize后close在第0位)
                # normalize_ohlcv的输出: [close_pct, high_pct, low_pct, open_pct, volume_pct]
                self.benchmark_prices = []
                for day_idx in range(data.shape[0]):
                    day_prices = data[day_idx, :, 0]  # 取第0列(close_pct)
                    # 从百分比变化重建绝对价格序列
                    # 假设normalize后close_pct是收益率, 需要累积乘积
                    # 但更简单: 直接用pct的累积和 * base_price
                    abs_prices = np.cumprod(1 + day_prices) * self.cfg.initial_price
                    # 插值到240个tick (如果T≠240)
                    if len(abs_prices) != 240:
                        from scipy.interpolate import interp1d
                        x_old = np.linspace(0, 239, len(abs_prices))
                        f = interp1d(x_old, abs_prices, kind='linear')
                        abs_prices = f(np.arange(240))
                    self.benchmark_prices.append(abs_prices)
                    
            elif data.ndim == 2:
                # (num_days, T) → 直接作为价格
                self.benchmark_prices = []
                for day_idx in range(data.shape[0]):
                    day_prices = data[day_idx]
                    if len(day_prices) != 240:
                        from scipy.interpolate import interp1d
                        x_old = np.linspace(0, 239, len(day_prices))
                        f = interp1d(x_old, day_prices, kind='linear')
                        day_prices = f(np.arange(240))
                    self.benchmark_prices.append(day_prices)
            
            logger.info(f"  ★ 基准线已解析: {len(self.benchmark_prices)}个交易日")
        except Exception as e:
            logger.warning(f"  ★ 基准线解析失败, 退回纯模拟: {e}")
            self.benchmark_prices = None
    
    def reset(self):
        # ★ v2.2: 用基准线开盘价reset(如果有)
        init_price = self.cfg.initial_price
        if self.benchmark_prices is not None and self._current_day_idx < len(self.benchmark_prices):
            init_price = float(self.benchmark_prices[self._current_day_idx][0])
        
        self.orderbook.reset(init_price)
        self.tick = 0
        self.info_events = []
        self.shock_events = []
        self.intraday_high = init_price
        self.intraday_low = init_price
        self.day_count = 0
        return self.orderbook.get_state()
    
    def step(self, actions: Dict[str, Dict]) -> Dict:
        """
        执行一步: 各agent提交订单
        actions: {agent_id: {side, volume, price, order_type}}
        
        ★ v2.2: 如果有价格基准线, 每个tick的价格 = 基准线价格 * BLEND + Agent订单价格 * (1-BLEND)
        """
        self.tick += 1
        self.orderbook.tick = self.tick
        
        results = {}
        for agent_id, action in actions.items():
            result = self.orderbook.submit_order(
                side=action.get('side', 'buy'),
                volume=action.get('volume', 0),
                price=action.get('price', None),
                agent_id=agent_id,
                order_type=action.get('order_type', 'limit'),
            )
            results[agent_id] = result
        
        # ★ 更新神经放电机制
        self.orderbook.update_firing(self.tick)
        
        # ★ v2.2: 价格基准线融合
        self._apply_benchmark()
        
        # 更新日内高低
        self.intraday_high = max(self.intraday_high, self.orderbook.mid_price)
        self.intraday_low = min(self.intraday_low, self.orderbook.mid_price)
        
        state = self.orderbook.get_state()
        
        # 处理信息事件 (庄家操纵)
        for event in self.info_events:
            if event['tick'] == self.tick:
                self._apply_info_event(event)
        
        # ── 隔夜跳空 (模拟午休/收盘开盘) ──
        overnight_gap = 0.0
        if self.tick == 120:
            # 午休后开盘: 小幅跳空
            overnight_gap = self._generate_overnight_gap(magnitude=0.3)
        elif self.tick >= self.episode_length - 1:
            # 交易日结束 (下个episode开盘会处理)
            pass
        
        if abs(overnight_gap) > 0:
            self.orderbook.mid_price *= (1 + overnight_gap)
            self.orderbook.mid_price = self.orderbook.price_limit.clamp_price(
                self.orderbook.mid_price)
            self.orderbook.bid_price = self.orderbook.mid_price * (1 - self.orderbook.spread / 2)
            self.orderbook.ask_price = self.orderbook.mid_price * (1 + self.orderbook.spread / 2)
        
        # ★ v2.2: episode结束切到下一个基准线日
        if self.tick >= self.episode_length:
            self._current_day_idx += 1
            if self.benchmark_prices is not None:
                self._current_day_idx %= len(self.benchmark_prices)  # 循环使用
        
        return {
            'state': self.orderbook.get_state(),
            'results': results,
            'done': self.tick >= self.episode_length,
            'overnight_gap': overnight_gap,
        }
    
    def _apply_benchmark(self):
        """★ v2.2: 将当前tick价格向基准线融合"""
        if self.benchmark_prices is None:
            return
        
        day_idx = self._current_day_idx % len(self.benchmark_prices)
        tick_idx = min(self.tick, 239)
        
        if tick_idx >= len(self.benchmark_prices[day_idx]):
            return
        
        target_price = float(self.benchmark_prices[day_idx][tick_idx])
        current_price = self.orderbook.mid_price
        
        # 融合: 基准线 * BLEND + Agent市场 * (1-BLEND)
        blended = target_price * self.BENCHMARK_BLEND + current_price * (1 - self.BENCHMARK_BLEND)
        
        # 涨跌停限制
        blended = self.orderbook.price_limit.clamp_price(blended)
        
        self.orderbook.mid_price = blended
        self.orderbook.bid_price = blended * (1 - self.orderbook.spread / 2)
        self.orderbook.ask_price = blended * (1 + self.orderbook.spread / 2)
    
    def start_new_day(self):
        """
        新交易日开盘: 生成隔夜跳空 + 重置涨跌停
        """
        self.day_count += 1
        
        # 隔夜跳空
        gap = self._generate_overnight_gap(magnitude=1.0)
        new_open = self.orderbook.mid_price * (1 + gap)
        
        # 重置涨跌停
        self.orderbook.price_limit.reset_daily(new_open)
        self.orderbook.daily_open = new_open
        
        # 更新价格
        new_open = self.orderbook.price_limit.clamp_price(new_open)
        self.orderbook.mid_price = new_open
        self.orderbook.bid_price = new_open * (1 - self.orderbook.spread / 2)
        self.orderbook.ask_price = new_open * (1 + self.orderbook.spread / 2)
        
        self.intraday_high = new_open
        self.intraday_low = new_open
        
        return gap
    
    def _generate_overnight_gap(self, magnitude: float = 1.0) -> float:
        """
        生成隔夜跳空幅度
        magnitude: 1.0=正常隔夜, 0.3=午休
        """
        if np.random.random() > self.GAP_PROBABILITY * magnitude:
            return 0.0
        
        gap = np.random.normal(
            self.GAP_MEAN * magnitude,
            self.GAP_STD * magnitude
        )
        
        # 截断极端跳空
        max_gap = 0.05 * magnitude  # 最多5%跳空
        gap = max(-max_gap, min(max_gap, gap))
        
        return gap
    
    def _apply_info_event(self, event):
        """应用信息事件对市场的影响"""
        impact = event.get('impact', 0)
        self.orderbook.spread = min(0.05, self.orderbook.spread * (1 + abs(impact)))
        
        # 信息事件也影响放电压力
        if impact > 0:
            self.orderbook.buy_pressure += abs(impact) * 1000
        else:
            self.orderbook.sell_pressure += abs(impact) * 1000
    
    def add_info_event(self, event_type: str, tick: int, impact: float, source: str):
        self.info_events.append({
            'type': event_type, 'tick': tick,
            'impact': impact, 'source': source,
        })


# ============================================================
#  Agent 基类
# ============================================================
class MarketAgent(ABC):
    """市场Agent基类"""
    
    def __init__(self, agent_id: str, capital: float, cfg: AdversarialConfig):
        self.agent_id = agent_id
        self.capital = capital
        self.initial_capital = capital
        self.holdings = 0.0
        self.avg_cost = 0.0
        self.cfg = cfg
        self.history = []
        self.reward_history = []
        
        # ★ T+1约束: 记录当日买入的股票不能当日卖出
        self.t1_locked = 0.0  # 当日买入锁定的股数
    
    def reset_t1_lock(self):
        """每日开盘时重置T+1锁"""
        self.t1_locked = 0.0
    
    @abstractmethod
    def observe(self, state: Dict) -> np.ndarray:
        pass
    
    @abstractmethod
    def decide(self, state: Dict) -> Dict:
        pass
    
    def update_portfolio(self, fill_result: Dict, side: str):
        """更新持仓 (需要传入买卖方向)"""
        filled = fill_result['filled']
        avg_price = fill_result['avg_price']
        
        if filled <= 0:
            return
        
        if side == 'buy':
            cost = filled * avg_price
            if cost <= self.capital:
                old_holdings = self.holdings
                self.holdings += filled
                if self.holdings > 0:
                    self.avg_cost = (
                        (self.avg_cost * old_holdings + avg_price * filled) 
                        / self.holdings
                    )
                self.capital -= cost
                # ★ T+1锁定
                self.t1_locked += filled
        
        elif side == 'sell':
            # ★ T+1约束: 只能卖出非锁定部分
            sellable = self.holdings - self.t1_locked
            sell_vol = min(filled, max(0, sellable))
            if sell_vol > 0:
                self.capital += sell_vol * avg_price
                self.holdings -= sell_vol
                if self.holdings <= 0:
                    self.holdings = 0
                    self.avg_cost = 0
                    self.t1_locked = 0
    
    def get_sellable_volume(self) -> float:
        """获取可卖出数量 (扣除T+1锁定)"""
        return max(0, self.holdings - self.t1_locked)
    
    def get_pnl(self, current_price: float) -> float:
        return self.capital + self.holdings * current_price - self.initial_capital
    
    def get_return(self, current_price: float) -> float:
        total = self.capital + self.holdings * current_price
        return total / self.initial_capital - 1.0


# ============================================================
#  策略网络
# ============================================================
class PolicyNetwork(nn.Module):
    """通用策略网络: 观测 → 行动概率"""
    
    def __init__(self, obs_dim: int, action_dim: int, hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
        )
        self.action_head = nn.Linear(hidden, action_dim)
        self.value_head = nn.Linear(hidden, 1)
    
    def forward(self, x):
        h = self.net(x)
        action_logits = self.action_head(h)
        value = self.value_head(h)
        return action_logits, value
    
    def act(self, x, deterministic=False):
        logits, value = self.forward(x)
        dist = Categorical(logits=logits)
        if deterministic:
            action = logits.argmax(dim=-1)
        else:
            action = dist.sample()
        return action, dist.log_prob(action), value


# ============================================================
#  庄家 Agent - 防共四策
# ============================================================
class DealerAgent(MarketAgent):
    """
    庄家Agent: "阎锡山逻辑" - 防共四策
    
    四大行为模块:
      1. 思想防散 (信息操纵): 发布利好/利空消息引导散户预期
      2. 民众防散 (白手套基金): 通过关联资金拉抬/护盘
      3. 政治防散 (庄家联盟): 与其他庄家联合坐庄
      4. 武力防散 (暴力砸盘): 震仓洗盘，恐慌性抛售
    
    ★ v2.1增强:
      - 显式阶段转换: 吸筹→拉抬→震仓→派发 (带状态机)
      - 利用神经放电信息: 在蓄积期加仓, 放电前布局
      - 利用市场状态: 趋势期顺势, 震荡期逆势
      - 更精细的挂单策略: 不总是市价单
    """
    
    ACTION_HOLD = 0
    ACTION_ACCUMULATE = 1
    ACTION_PUMP = 2
    ACTION_SHAKE = 3
    ACTION_DISTRIBUTE = 4
    
    ACTION_NAMES = ['观望', '吸筹', '拉抬', '震仓', '派发']
    
    # 庄家阶段转换概率矩阵 (简化)
    # 从当前阶段到下一阶段的转换倾向
    PHASE_TRANSITIONS = {
        'accumulate': {'accumulate': 0.6, 'pump': 0.25, 'shake': 0.1, 'distribute': 0.05},
        'pump':       {'accumulate': 0.1, 'pump': 0.4, 'shake': 0.3, 'distribute': 0.2},
        'shake':      {'accumulate': 0.3, 'pump': 0.3, 'shake': 0.2, 'distribute': 0.2},
        'distribute': {'accumulate': 0.4, 'pump': 0.1, 'shake': 0.1, 'distribute': 0.4},
    }
    
    def __init__(self, agent_id: str, capital: float, cfg: AdversarialConfig):
        super().__init__(agent_id, capital, cfg)
        
        # 策略网络 (增强观测维度: 20→24)
        obs_dim = 24
        self.policy = PolicyNetwork(obs_dim, 5, hidden=128)
        
        # 防共四策参数
        self.info_power = 0.5
        self.alliance_threshold = 0.3
        self.shake_intensity = 0.05
        self.puppet_capital = capital * 0.2
        
        # ★ 显式状态机
        self.phase = 'accumulate'
        self.phase_ticks = 0          # 当前阶段持续tick数
        self.phase_history = []
        self.alliance_active = False
        self.info_events_sent = 0
        
        # ★ 利用神经放电的参数
        self.firing_awareness = 0.3    # 对放电的感知能力 (0~1)
    
    def observe(self, state: Dict) -> np.ndarray:
        """构建庄家观测向量 (增强: 加入放电/市场状态)"""
        firing = state.get('firing', {})
        regime = state.get('regime', {})
        
        features = [
            state['mid_price'] / self.cfg.initial_price - 1,
            state['spread'] / state['mid_price'],
            state['bid_ask_ratio'] - 1,
            state['bid_depth'] / max(state['ask_depth'], 1),
            self.holdings * state['mid_price'] / self.initial_capital,
            (state['mid_price'] - self.avg_cost) / (self.avg_cost + 1e-8),
            self.capital / self.initial_capital,
            len(state.get('recent_trades', [])) / 10,
            np.log1p(state['bid_depth']),
            np.log1p(state['ask_depth']),
            # ★ 放电相关特征
            firing.get('normalized_potential', 0),
            firing.get('consecutive_firings', 0),
            float(firing.get('phase', 'RESTING') == 'ACCUMULATING'),
            float(firing.get('phase', 'RESTING') == 'FIRING'),
            # ★ 市场状态特征
            float(regime.get('regime', 'ranging') == 'trending'),
            regime.get('trend_strength', 0),
            # ★ 自身阶段特征
            float(self.phase == 'accumulate'),
            float(self.phase == 'pump'),
            float(self.phase == 'shake'),
            float(self.phase == 'distribute'),
            # 买卖压力
            state.get('buy_pressure', 0) / 1e6,
            state.get('sell_pressure', 0) / 1e6,
            # 不应期强度
            firing.get('refractory_strength', 0),
            self.phase_ticks / 60.0,  # 当前阶段持续时间
        ]
        return np.array(features[:24], dtype=np.float32)
    
    def decide(self, state: Dict) -> Dict:
        """庄家决策: 结合RL策略 + 防共四策行为模块 + 放电感知"""
        obs = self.observe(state)
        obs_t = torch.FloatTensor(obs).unsqueeze(0)
        
        with torch.no_grad():
            action, log_prob, value = self.policy.act(obs_t)
        
        action = action.item()
        self.phase_history.append(action)
        self.phase_ticks += 1
        
        # ★ 阶段转换 (基于状态机而非纯RL)
        self._update_phase(state)
        
        # ★ 放电感知: 如果即将放电, 提前布局
        firing = state.get('firing', {})
        firing_override = self._check_firing_override(state, firing)
        if firing_override is not None:
            return firing_override
        
        price = state['mid_price']
        
        if action == self.ACTION_HOLD:
            return {'side': 'buy', 'volume': 0, 'price': None, 'order_type': 'limit'}
        
        elif action == self.ACTION_ACCUMULATE:
            volume = self.capital * 0.05 / price
            # ★ 蓄积期吸筹更隐蔽: 限价单挂在买二
            return {'side': 'buy', 'volume': volume,
                    'price': state['bid_price'] * 0.999, 'order_type': 'limit'}
        
        elif action == self.ACTION_PUMP:
            volume = self.puppet_capital * 0.1 / price
            return {'side': 'buy', 'volume': volume,
                    'price': None, 'order_type': 'market'}
        
        elif action == self.ACTION_SHAKE:
            volume = self.holdings * self.shake_intensity
            return {'side': 'sell', 'volume': volume,
                    'price': state['bid_price'] * 0.98, 'order_type': 'limit',
                    '_info_event': True, '_event_type': 'negative',
                    '_event_impact': -self.info_power * 0.5}
        
        elif action == self.ACTION_DISTRIBUTE:
            volume = self.holdings * 0.1
            return {'side': 'sell', 'volume': volume,
                    'price': state['ask_price'], 'order_type': 'limit',
                    '_info_event': True, '_event_type': 'positive',
                    '_event_impact': self.info_power * 0.3}
        
        return {'side': 'buy', 'volume': 0, 'price': None, 'order_type': 'limit'}
    
    def _update_phase(self, state: Dict):
        """★ 阶段状态机更新"""
        # 每隔20个tick评估是否转换阶段
        if self.phase_ticks % 20 != 0:
            return
        
        # 基于状态机概率 + 当前持仓
        transitions = self.PHASE_TRANSITIONS.get(self.phase, 
                      self.PHASE_TRANSITIONS['accumulate'])
        
        # 持仓调整: 仓位高→倾向派发/震仓, 仓位低→倾向吸筹
        holdings_ratio = self.holdings * state['mid_price'] / self.initial_capital
        if holdings_ratio > 0.5:
            transitions = {**transitions, 'distribute': transitions['distribute'] + 0.2,
                          'accumulate': transitions['accumulate'] - 0.1}
        elif holdings_ratio < 0.1:
            transitions = {**transitions, 'accumulate': transitions['accumulate'] + 0.2,
                          'distribute': transitions['distribute'] - 0.1}
        
        # 归一化
        total = sum(transitions.values())
        transitions = {k: v/total for k, v in transitions.items()}
        
        # 按概率选择
        phases = list(transitions.keys())
        probs = [transitions[p] for p in phases]
        new_phase = np.random.choice(phases, p=probs)
        
        if new_phase != self.phase:
            self.phase = new_phase
            self.phase_ticks = 0
    
    def _check_firing_override(self, state: Dict, firing: Dict) -> Optional[Dict]:
        """★ 检查神经放电状态, 适时覆盖RL决策"""
        if np.random.random() > self.firing_awareness:
            return None  # 大多数时候遵循RL
        
        phase = firing.get('phase', 'RESTING')
        norm_pot = firing.get('normalized_potential', 0)
        
        price = state['mid_price']
        
        # 蓄积期且压力高 → 提前吸筹 (放电前布局)
        if phase == 'ACCUMULATING' and norm_pot > 0.5:
            if firing.get('trend', 'FLAT') == 'UP' and self.capital > self.initial_capital * 0.3:
                volume = self.capital * 0.08 / price  # 加大吸筹力度
                logger.debug(f"  庄家放电感知: 蓄积期+看多→加大吸筹")
                return {'side': 'buy', 'volume': volume,
                        'price': state['ask_price'], 'order_type': 'limit'}
            elif firing.get('trend', 'FLAT') == 'DOWN':
                # 蓄积期看空 → 不吸筹, 甚至减仓
                if self.holdings > 0:
                    volume = self.holdings * 0.05
                    return {'side': 'sell', 'volume': volume,
                            'price': state['bid_price'], 'order_type': 'limit'}
        
        # 放电后不应期 → 减少交易 (让市场消化)
        if phase == 'REFRACTORY':
            return {'side': 'buy', 'volume': 0, 'price': None, 'order_type': 'limit'}
        
        return None
    
    def apply_strategy_override(self, state: Dict, action: Dict) -> Dict:
        """防共四策行为覆盖"""
        holdings_ratio = self.holdings * state['mid_price'] / self.initial_capital
        
        # 策略1: 思想防散 - 仓位过高时发布利好引导散户接盘
        if holdings_ratio > 0.5 and action.get('side') == 'sell':
            action['_info_event'] = True
            action['_event_type'] = 'positive'
            action['_event_impact'] = self.info_power * 0.4
            self.info_events_sent += 1
        
        # 策略2: 民众防散 - 价格下跌时白手套护盘
        if state['mid_price'] < self.avg_cost * 0.95:
            puppet_volume = self.puppet_capital * 0.2 / state['mid_price']
            action = {'side': 'buy', 'volume': puppet_volume,
                     'price': None, 'order_type': 'market', '_puppet': True}
        
        # 策略4: 武力防散 - 散户头羊出现时强制震仓
        if self._detect_retailer_leader(state):
            shake_vol = self.holdings * self.shake_intensity * 2
            action = {'side': 'sell', 'volume': shake_vol,
                     'price': state['bid_price'] * 0.97, 'order_type': 'limit',
                     '_info_event': True, '_event_type': 'negative',
                     '_event_impact': -self.info_power * 0.8}
        
        return action
    
    def _detect_retailer_leader(self, state: Dict) -> bool:
        """检测散户头羊行为"""
        recent = state.get('recent_trades', [])
        if len(recent) < 3:
            return False
        buy_count = sum(1 for t in recent if t.get('buyer', '').startswith('retailer'))
        return buy_count >= len(recent) * 0.6


# ============================================================
#  散户 Agent - 五类型 (增强T+1)
# ============================================================
class RetailerAgent(MarketAgent):
    """
    散户Agent: 五类型
    
    ★ v2.1增强:
      - T+1约束: 当日买入不可卖出
      - 放电感知: 感知市场蓄积/放电/不应期
      - 涨停追板: 跟风型在涨停板追入
      - 跌停恐慌: 跟风型在跌停时恐慌卖出
      - 更真实的行为模式
    """
    
    TYPE_PROBS = {'herd': 0.35, 'value': 0.15, 'technical': 0.20,
                  'leader': 0.10, 'passive': 0.20}
    
    def __init__(self, agent_id: str, capital: float, cfg: AdversarialConfig,
                 retailer_type: str = 'herd'):
        super().__init__(agent_id, capital, cfg)
        self.retailer_type = retailer_type
        
        obs_dim = 24
        self.policy = PolicyNetwork(obs_dim, 4, hidden=64)
        
        self.momentum_sensitivity = {
            'herd': 0.8, 'value': -0.3, 'technical': 0.5,
            'leader': 0.6, 'passive': 0.05,
        }[retailer_type]
        
        self.trade_frequency = {
            'herd': 0.6, 'value': 0.3, 'technical': 0.5,
            'leader': 0.7, 'passive': 0.05,
        }[retailer_type]
        
        self.monthly_salary = cfg.retailer_monthly_salary
        self.salary_timer = 0
        
        self.panicking = False
        self.fomo = False
        
        # ★ 放电感知强度 (跟风型最强)
        self.firing_sensitivity = {
            'herd': 0.9, 'value': 0.2, 'technical': 0.5,
            'leader': 0.7, 'passive': 0.05,
        }[retailer_type]
    
    def observe(self, state: Dict) -> np.ndarray:
        firing = state.get('firing', {})
        regime = state.get('regime', {})
        price_limit = state.get('price_limit', {})
        
        features = [
            state['mid_price'] / self.cfg.initial_price - 1,
            state['spread'] / state['mid_price'],
            state['bid_ask_ratio'] - 1,
            (state['mid_price'] - self.avg_cost) / (self.avg_cost + 1e-8),
            self.holdings * state['mid_price'] / max(self.initial_capital, 1),
            self.capital / max(self.initial_capital, 1),
            self.momentum_sensitivity,
            self.trade_frequency,
            float(self.panicking),
            float(self.fomo),
            # 价格动量
            (state['mid_price'] - self.cfg.initial_price) / (self.cfg.initial_price + 1e-8),
            # ★ 放电感知
            firing.get('normalized_potential', 0),
            float(firing.get('phase', 'RESTING') == 'ACCUMULATING'),
            float(firing.get('phase', 'RESTING') == 'FIRING'),
            float(firing.get('phase', 'RESTING') == 'REFRACTORY'),
            firing.get('consecutive_firings', 0),
            # ★ 市场状态
            float(regime.get('regime', 'ranging') == 'trending'),
            regime.get('trend_strength', 0),
            # ★ 涨跌停感知
            float(price_limit.get('at_limit_up', False)),
            float(price_limit.get('at_limit_down', False)),
            price_limit.get('consecutive_limit_up', 0),
            price_limit.get('consecutive_limit_down', 0),
            # 可卖量 (T+1)
            self.get_sellable_volume() / max(self.holdings, 1),
            self.t1_locked / max(self.holdings, 1),
        ]
        return np.array(features[:24], dtype=np.float32)
    
    def decide(self, state: Dict) -> Dict:
        # 月度工资注入
        self.salary_timer += 1
        if self.salary_timer >= 20:
            self.capital += self.monthly_salary
            self.salary_timer = 0
        
        # 被动型很少交易
        if self.retailer_type == 'passive':
            if np.random.random() > 0.02:
                return {'side': 'buy', 'volume': 0, 'price': None, 'order_type': 'limit'}
        
        # 抱团/踩踏检测
        self._detect_herd_state(state)
        
        # ★ 放电驱动的行为覆盖 (对跟风型尤其重要)
        firing_override = self._check_firing_override(state)
        if firing_override is not None:
            return firing_override
        
        # ★ 涨跌停行为 (对跟风型)
        limit_override = self._check_limit_override(state)
        if limit_override is not None:
            return limit_override
        
        obs = self.observe(state)
        obs_t = torch.FloatTensor(obs).unsqueeze(0)
        
        with torch.no_grad():
            action, _, _ = self.policy.act(obs_t)
        
        action = action.item()
        price = state['mid_price']
        
        if action == 0:
            return {'side': 'buy', 'volume': 0, 'price': None, 'order_type': 'limit'}
        
        elif action == 1:  # 买入
            volume = self.capital * self.trade_frequency * 0.1 / price
            if self.fomo:
                volume *= 2.0
            return {'side': 'buy', 'volume': volume,
                    'price': state['ask_price'], 'order_type': 'limit'}
        
        elif action == 2:  # 卖出
            # ★ T+1约束: 只能卖出非锁定部分
            sellable = self.get_sellable_volume()
            volume = sellable * self.trade_frequency * 0.1
            if self.panicking:
                volume = sellable  # 恐慌时全部卖出可卖部分
            if volume <= 0:
                return {'side': 'buy', 'volume': 0, 'price': None, 'order_type': 'limit'}
            return {'side': 'sell', 'volume': volume,
                    'price': state['bid_price'], 'order_type': 'limit'}
        
        elif action == 3:  # 跟单
            recent = state.get('recent_trades', [])
            if recent:
                last_side = 'buy' if recent[-1].get('buyer', '').startswith('retailer_leader') else 'sell'
                # ★ T+1约束
                if last_side == 'sell' and self.get_sellable_volume() <= 0:
                    return {'side': 'buy', 'volume': 0, 'price': None, 'order_type': 'limit'}
                volume = self.capital * 0.05 / price
                return {'side': last_side, 'volume': volume,
                        'price': None, 'order_type': 'market'}
            return {'side': 'buy', 'volume': 0, 'price': None, 'order_type': 'limit'}
        
        return {'side': 'buy', 'volume': 0, 'price': None, 'order_type': 'limit'}
    
    def _check_firing_override(self, state: Dict) -> Optional[Dict]:
        """★ 放电感知驱动的行为"""
        if np.random.random() > self.firing_sensitivity:
            return None
        
        firing = state.get('firing', {})
        phase = firing.get('phase', 'RESTING')
        direction = firing.get('trend', 'FLAT')
        price = state['mid_price']
        
        # 放电中 (全或无行情) → 跟风型强烈追涨/杀跌
        if phase == 'FIRING' and self.retailer_type == 'herd':
            if direction == 'UP':
                volume = self.capital * 0.3 / price  # 重仓追涨
                self.fomo = True
                return {'side': 'buy', 'volume': volume,
                        'price': None, 'order_type': 'market'}
            elif direction == 'DOWN':
                sellable = self.get_sellable_volume()
                if sellable > 0:
                    self.panicking = True
                    return {'side': 'sell', 'volume': sellable,
                            'price': None, 'order_type': 'market'}
        
        # 蓄积期 + 价值型 → 逆势布局
        if phase == 'ACCUMULATING' and self.retailer_type == 'value':
            if direction == 'DOWN' and self.capital > self.initial_capital * 0.5:
                volume = self.capital * 0.1 / price
                return {'side': 'buy', 'volume': volume,
                        'price': state['bid_price'], 'order_type': 'limit'}
        
        return None
    
    def _check_limit_override(self, state: Dict) -> Optional[Dict]:
        """★ 涨跌停行为覆盖"""
        price_limit = state.get('price_limit', {})
        price = state['mid_price']
        
        # 涨停板 → 跟风型追板
        if (price_limit.get('at_limit_up') and 
            self.retailer_type in ['herd', 'leader']):
            if np.random.random() < 0.3:
                volume = self.capital * 0.5 / price  # 半仓打板
                return {'side': 'buy', 'volume': volume,
                        'price': None, 'order_type': 'market'}
        
        # 跌停板 → 跟风型恐慌排队
        if (price_limit.get('at_limit_down') and 
            self.retailer_type in ['herd', 'leader']):
            sellable = self.get_sellable_volume()
            if sellable > 0 and np.random.random() < 0.5:
                return {'side': 'sell', 'volume': sellable,
                        'price': None, 'order_type': 'market'}
        
        return None
    
    def _detect_herd_state(self, state: Dict):
        """检测抱团/踩踏状态"""
        recent = state.get('recent_trades', [])
        
        if len(recent) >= 3:
            prices = [t['price'] for t in recent[-5:]]
            if len(prices) >= 3 and all(prices[i] < prices[i+1] for i in range(len(prices)-1)):
                self.fomo = True
                self.panicking = False
            elif len(prices) >= 3 and all(prices[i] > prices[i+1] for i in range(len(prices)-1)):
                self.panicking = True
                self.fomo = False
            else:
                self.fomo = False
                self.panicking = False


# ============================================================
#  游资 Agent - 打板快进快出
# ============================================================
class HotMoneyAgent(MarketAgent):
    """
    游资Agent: 打板策略
    
    ★ v2.1增强:
      - 涨停板策略: 涨停打板/撬板
      - 更严格止损
    """
    
    def __init__(self, agent_id: str, capital: float, cfg: AdversarialConfig):
        super().__init__(agent_id, capital, cfg)
        
        obs_dim = 24
        self.policy = PolicyNetwork(obs_dim, 3, hidden=64)
        
        self.momentum_thresh = cfg.hotmoney_momentum_thresh
        self.entry_price = 0.0
        self.max_hold_ticks = 30
        self.hold_ticks = 0
        self.stop_loss = 0.03
        self.take_profit = 0.08
    
    def observe(self, state: Dict) -> np.ndarray:
        firing = state.get('firing', {})
        regime = state.get('regime', {})
        price_limit = state.get('price_limit', {})
        
        features = [
            state['mid_price'] / self.cfg.initial_price - 1,
            state['spread'] / state['mid_price'],
            state['bid_ask_ratio'] - 1,
            (state['mid_price'] - self.avg_cost) / (self.avg_cost + 1e-8) if self.holdings > 0 else 0,
            float(self.holdings > 0),
            self.hold_ticks / self.max_hold_ticks,
            np.log1p(state['bid_depth']),
            np.log1p(state['ask_depth']),
            # ★ 放电感知 (游资对放电非常敏感)
            firing.get('normalized_potential', 0),
            float(firing.get('phase', 'RESTING') == 'FIRING'),
            float(firing.get('phase', 'RESTING') == 'ACCUMULATING'),
            firing.get('consecutive_firings', 0),
            # ★ 涨跌停
            float(price_limit.get('at_limit_up', False)),
            float(price_limit.get('at_limit_down', False)),
            price_limit.get('consecutive_limit_up', 0),
        ]
        while len(features) < 24:
            features.append(0.0)
        return np.array(features[:24], dtype=np.float32)
    
    def decide(self, state: Dict) -> Dict:
        price = state['mid_price']
        
        # 止损/止盈检查
        if self.holdings > 0:
            self.hold_ticks += 1
            pnl_pct = (price - self.entry_price) / (self.entry_price + 1e-8)
            
            if pnl_pct <= -self.stop_loss:
                sellable = self.get_sellable_volume()
                if sellable > 0:
                    return {'side': 'sell', 'volume': sellable,
                            'price': None, 'order_type': 'market'}
            elif pnl_pct >= self.take_profit:
                sellable = self.get_sellable_volume()
                if sellable > 0:
                    return {'side': 'sell', 'volume': sellable,
                            'price': None, 'order_type': 'market'}
            elif self.hold_ticks >= self.max_hold_ticks:
                sellable = self.get_sellable_volume()
                if sellable > 0:
                    return {'side': 'sell', 'volume': sellable,
                            'price': None, 'order_type': 'market'}
        
        obs = self.observe(state)
        obs_t = torch.FloatTensor(obs).unsqueeze(0)
        
        with torch.no_grad():
            action, _, _ = self.policy.act(obs_t)
        
        action = action.item()
        
        if action == 0:
            return {'side': 'buy', 'volume': 0, 'price': None, 'order_type': 'limit'}
        
        elif action == 1:  # 打板买入
            if self.holdings == 0:
                recent = state.get('recent_trades', [])
                firing = state.get('firing', {})
                
                # ★ 游资利用放电信号: 蓄积期准备, 放电时入场
                should_enter = False
                if firing.get('phase') == 'FIRING' and firing.get('trend') == 'UP':
                    should_enter = True
                elif len(recent) >= 2:
                    momentum = (recent[-1]['price'] - recent[0]['price']) / (recent[0]['price'] + 1e-8)
                    if momentum >= self.momentum_thresh:
                        should_enter = True
                
                if should_enter:
                    volume = self.capital * 0.5 / price
                    self.entry_price = price
                    self.hold_ticks = 0
                    return {'side': 'buy', 'volume': volume,
                            'price': None, 'order_type': 'market'}
            return {'side': 'buy', 'volume': 0, 'price': None, 'order_type': 'limit'}
        
        elif action == 2:  # 快速卖出
            sellable = self.get_sellable_volume()
            if sellable > 0:
                return {'side': 'sell', 'volume': sellable,
                        'price': None, 'order_type': 'market'}
            return {'side': 'buy', 'volume': 0, 'price': None, 'order_type': 'limit'}
        
        return {'side': 'buy', 'volume': 0, 'price': None, 'order_type': 'limit'}


# ============================================================
#  ★ 庄家联盟模块 (政治防散 - 集成到训练)
# ============================================================
class DealerAlliance:
    """
    政治防散: 庄家联盟
    多个庄家协调行动，共享部分信息
    
    ★ v2.1: 集成到训练循环
    """
    
    def __init__(self, dealers: List[DealerAgent], threshold: float = 0.3):
        self.dealers = dealers
        self.threshold = threshold
        self.active = False
        self.coordination_signal = None
    
    def check_alliance(self, market_state: Dict) -> bool:
        """检查是否触发联盟"""
        losing_count = sum(1 for d in self.dealers 
                          if d.get_pnl(market_state['mid_price']) < 0)
        
        if losing_count / len(self.dealers) >= self.threshold:
            self.active = True
            self.coordination_signal = 'defend'
        else:
            self.active = False
        
        return self.active
    
    def coordinate(self, market_state: Dict) -> Optional[Dict]:
        """联盟协调行动"""
        if not self.active:
            return None
        
        total_volume = sum(d.capital * 0.1 / market_state['mid_price'] 
                          for d in self.dealers)
        
        return {
            'action': 'buy',
            'volume': total_volume,
            'price': None,
            'order_type': 'market',
            '_alliance': True,
        }


# ============================================================
#  ★ 对抗训练环境 (增强版)
# ============================================================
class AdversarialTrainer:
    """
    对抗训练主控 (v2.2 基准线版)
    
    增强点:
      1. 神经放电机制: 63.21%阈值全或无行情
      2. 涨跌停板: ±10%/±20%
      3. 市场冲击: 大单滑点+永久性冲击
      4. 市场状态: 趋势/震荡/转折
      5. T+1约束: 散户当日买入不可卖出
      6. 隔夜跳空: 模拟开盘缺口
      7. 庄家联盟: 集成到训练循环
      8. 五方案防摆烂
      9. 增强监察: 记录放电/涨跌停等关键事件
     10. ★ v2.2: 价格基准线 — 先用假数据练兵, 再上实盘
    """
    
    def __init__(self, cfg: AdversarialConfig = None,
                 price_benchmark: np.ndarray = None,
                 benchmark_source: str = ""):
        """
        price_benchmark: 价格基准线数据 (假数据或实盘)
          - None: 纯模拟模式 (v2.1兼容)
          - numpy数组: 传入MarketEnv作为价格锚点
        
        benchmark_source: 标注来源, 如 "fake" / "real", 仅用于日志
        """
        self.cfg = cfg or AdversarialConfig()
        self.benchmark_source = benchmark_source
        self.env = MarketEnv(self.cfg, price_benchmark=price_benchmark)
        
        # 创建Agent群体
        self.dealer = DealerAgent("dealer",
            capital=self.cfg.initial_price * 100000 * self.cfg.dealer_capital_ratio,
            cfg=self.cfg)
        
        self.retailers = self._create_retailer_swarm()
        self.hotmoney_agents = self._create_hotmoney_swarm()
        
        # ★ 庄家联盟
        self.dealer_alliance = DealerAlliance([self.dealer])
        
        # 进化统计
        self.generation = 0
        self.best_dealer_reward = -float('inf')
        self.best_retailer_reward = -float('inf')
        self.strategy_diversity_history = []
        
        # ★ 监察统计
        self.firing_stats = {
            'total_firings': 0,
            'up_firings': 0,
            'down_firings': 0,
            'max_consecutive': 0,
        }
        self.limit_stats = {
            'limit_up_days': 0,
            'limit_down_days': 0,
        }
        
        # 训练记录
        self.episode_log = []
        
        # 实盘校准窗口
        self._cal_reward_weight = 0.0
        self._load_calibration()
    
    def _load_calibration(self):
        """加载 calibration_params.json"""
        cal_file = RESULTS_DIR / "calibration_params.json"
        if not cal_file.exists():
            return
        
        try:
            with open(cal_file, 'r', encoding='utf-8') as f:
                cal = json.load(f)
            
            adj = cal.get('adjustments', {})
            
            if 'dealer' in adj:
                d = adj['dealer']
                if 'info_power_delta' in d:
                    self.dealer.info_power = max(0.1, min(1.0,
                        self.dealer.info_power + d['info_power_delta']))
                if 'shake_intensity_delta' in d:
                    self.dealer.shake_intensity = max(0.01, min(0.2,
                        self.dealer.shake_intensity + d['shake_intensity_delta']))
                if 'puppet_ratio_delta' in d:
                    self.dealer.puppet_capital *= (1 + d['puppet_ratio_delta'])
                logger.info(f"  庄家校准: info_power={self.dealer.info_power:.2f}, "
                           f"shake={self.dealer.shake_intensity:.3f}")
            
            if 'reward' in adj:
                rw = adj['reward']
                self._cal_reward_weight = rw.get('reward_shaping_weight_delta', 0)
                logger.info(f"  奖励校准: shaping_weight +{self._cal_reward_weight:.2f}")
            
            logger.info(f"✓ 实盘校准参数已加载: {cal_file}")
            
        except Exception as e:
            logger.warning(f"加载校准参数失败: {e}")
    
    def _create_retailer_swarm(self, n: int = 20) -> List[RetailerAgent]:
        retailers = []
        total_capital = self.cfg.initial_price * 100000 * self.cfg.retailer_ratio
        
        types = list(RetailerAgent.TYPE_PROBS.keys())
        probs = list(RetailerAgent.TYPE_PROBS.values())
        
        for i in range(n):
            rtype = np.random.choice(types, p=probs)
            cap = total_capital / n
            agent = RetailerAgent(
                f"retailer_{rtype}_{i}", cap, self.cfg, rtype
            )
            retailers.append(agent)
        
        return retailers
    
    def _create_hotmoney_swarm(self, n: int = 5) -> List[HotMoneyAgent]:
        agents = []
        total_capital = self.cfg.initial_price * 100000 * self.cfg.hotmoney_ratio
        
        for i in range(n):
            cap = total_capital / n
            agents.append(HotMoneyAgent(f"hotmoney_{i}", cap, self.cfg))
        
        return agents
    
    def train(self, num_episodes: int = None, evolve: bool = True) -> Dict:
        """对抗训练主循环"""
        num_episodes = num_episodes or self.cfg.num_episodes
        logger.info(f"开始对抗训练: {num_episodes} episodes")
        logger.info(f"  庄家: 1 | 散户: {len(self.retailers)} | 游资: {len(self.hotmoney_agents)}")
        logger.info(f"  ★ 神经放电阈值: {NeuralFiringMechanism.FIRING_THRESHOLD:.4f} (63.21%)")
        
        all_rewards = {'dealer': [], 'retailer': [], 'hotmoney': []}
        
        for ep in range(num_episodes):
            state = self.env.reset()
            self._reset_agents()
            
            ep_rewards = {'dealer': 0, 'retailer': 0, 'hotmoney': 0}
            ep_firing_count = 0
            
            for tick in range(self.cfg.episode_length):
                # 收集所有Agent的行动
                actions = {}
                
                # ★ 庄家联盟检查
                alliance_action = None
                if self.dealer_alliance.check_alliance(state):
                    alliance_action = self.dealer_alliance.coordinate(state)
                
                if alliance_action:
                    actions[self.dealer.agent_id] = alliance_action
                else:
                    dealer_action = self.dealer.decide(state)
                    dealer_action = self.dealer.apply_strategy_override(state, dealer_action)
                    actions[self.dealer.agent_id] = dealer_action
                
                # 处理庄家信息操纵事件
                if actions[self.dealer.agent_id].get('_info_event'):
                    self.env.add_info_event(
                        actions[self.dealer.agent_id]['_event_type'],
                        tick + 1,
                        actions[self.dealer.agent_id]['_event_impact'],
                        self.dealer.agent_id
                    )
                
                # 散户
                for r in self.retailers:
                    actions[r.agent_id] = r.decide(state)
                
                # 游资
                for h in self.hotmoney_agents:
                    actions[h.agent_id] = h.decide(state)
                
                # 执行
                result = self.env.step(actions)
                state = result['state']
                
                # 更新各Agent持仓
                for agent_id, action in actions.items():
                    fill = result['results'].get(agent_id, {})
                    side = action.get('side', 'buy')
                    if agent_id == self.dealer.agent_id:
                        self.dealer.update_portfolio(fill, side)
                        self.dealer.history.append(action)
                    else:
                        for r in self.retailers:
                            if r.agent_id == agent_id:
                                r.update_portfolio(fill, side)
                                r.history.append(action)
                                break
                        for h in self.hotmoney_agents:
                            if h.agent_id == agent_id:
                                h.update_portfolio(fill, side)
                                h.history.append(action)
                                break
                
                # ★ 统计放电事件
                firing_state = state.get('firing', {})
                if firing_state.get('phase') == 'FIRING' and ep_firing_count < 100:
                    ep_firing_count += 1
                    direction = firing_state.get('trend', 'FLAT')
                    if direction == 'UP':
                        self.firing_stats['up_firings'] += 1
                    elif direction == 'DOWN':
                        self.firing_stats['down_firings'] += 1
                    self.firing_stats['total_firings'] += 1
                
                # 计算奖励
                rewards = self._compute_rewards(state)
                ep_rewards['dealer'] += rewards['dealer']
                ep_rewards['retailer'] += rewards['retailer']
                ep_rewards['hotmoney'] += rewards['hotmoney']
                
                # 防摆烂检查
                self._anti_degenerate_check(tick, ep)
                
                if result['done']:
                    break
            
            # 更新放电统计
            firing_state = state.get('firing', {})
            cons = firing_state.get('consecutive_firings', 0)
            self.firing_stats['max_consecutive'] = max(
                self.firing_stats['max_consecutive'], cons)
            
            # 涨跌停统计
            pl = state.get('price_limit', {})
            if pl.get('at_limit_up'):
                self.limit_stats['limit_up_days'] += 1
            if pl.get('at_limit_down'):
                self.limit_stats['limit_down_days'] += 1
            
            # 记录
            for key in all_rewards:
                all_rewards[key].append(ep_rewards[key])
            
            # 进化
            if evolve and (ep + 1) % self.cfg.evolution_interval == 0:
                self._evolve(all_rewards)
            
            # 日志
            if (ep + 1) % 50 == 0:
                d_avg = np.mean(all_rewards['dealer'][-50:])
                r_avg = np.mean(all_rewards['retailer'][-50:])
                h_avg = np.mean(all_rewards['hotmoney'][-50:])
                logger.info(
                    f"[Ep {ep+1}/{num_episodes}] "
                    f"庄家={d_avg:.2f} 散户={r_avg:.2f} 游资={h_avg:.2f} | "
                    f"放电={self.firing_stats['total_firings']} "
                    f"(↑{self.firing_stats['up_firings']} ↓{self.firing_stats['down_firings']})"
                )
        
        # 保存训练结果
        self._save_training_results(all_rewards)
        return all_rewards
    
    def _compute_rewards(self, state: Dict) -> Dict:
        """
        计算三方奖励
        
        庄家奖励核心: "散户预测错误度"
          - 散户在庄家拉抬时卖出 → 庄家得利
          - 散户在庄家派发时买入 → 庄家得利
        
        ★ v2.1增强:
          - 放电期间庄家奖励加权 (抓住行情更重要)
          - 市场状态影响奖励 (趋势期奖励更清晰)
        """
        price = state['mid_price']
        firing = state.get('firing', {})
        regime = state.get('regime', {})
        
        # 庄家奖励
        dealer_pnl = self.dealer.get_pnl(price)
        retailer_error = 0
        for r in self.retailers:
            r_pnl = r.get_pnl(price)
            if r_pnl < 0:
                retailer_error += abs(r_pnl) * 0.1
        
        dealer_reward = dealer_pnl + retailer_error
        
        # ★ 放电期间加权: 行情明确时, 庄家抓住行情的奖励更大
        if firing.get('phase') == 'FIRING':
            dealer_reward *= 1.3  # 放电期额外30%奖励
        
        # ★ 趋势期加权: 方向明确时奖励更清晰
        if regime.get('regime') == 'trending':
            dealer_reward *= 1.1
        
        # 实盘校准奖励
        if self._cal_reward_weight > 0:
            cal_signal = self._get_calibration_signal()
            if cal_signal is not None:
                dealer_reward += self._cal_reward_weight * cal_signal
        
        # 散户奖励 (PnL + 策略多样性)
        retailer_reward = np.mean([r.get_pnl(price) for r in self.retailers])
        
        # 游资奖励
        hotmoney_reward = np.mean([h.get_pnl(price) for h in self.hotmoney_agents])
        
        return {
            'dealer': dealer_reward,
            'retailer': retailer_reward,
            'hotmoney': hotmoney_reward,
        }
    
    def _get_calibration_signal(self) -> Optional[float]:
        try:
            cal_file = RESULTS_DIR / "calibration_params.json"
            if cal_file.exists():
                with open(cal_file, 'r', encoding='utf-8') as f:
                    cal = json.load(f)
                wr = cal.get('win_rate', 0.5)
                return (wr - 0.5) * 2.0
        except Exception:
            pass
        return None
    
    def _evolve(self, all_rewards: Dict):
        """双进化: 拉马克 + 达尔文"""
        self.generation += 1
        logger.info(f"\n{'='*30} 第{self.generation}代进化 {'='*30}")
        
        self._lamarck_evolution(all_rewards)
        self._darwin_evolution()
        
        diversity = self._compute_strategy_diversity()
        self.strategy_diversity_history.append(diversity)
        logger.info(f"  策略多样性: {diversity:.4f}")
        
        if diversity < self.cfg.min_strategy_diversity:
            logger.warning("  ⚠ 策略多样性不足，注入随机噪声")
            self._inject_diversity_noise()
    
    def _lamarck_evolution(self, all_rewards):
        """拉马克进化"""
        if all_rewards['dealer']:
            recent = all_rewards['dealer'][-self.cfg.evolution_interval:]
            if np.mean(recent) > self.best_dealer_reward:
                self.best_dealer_reward = np.mean(recent)
                best_params = copy.deepcopy(self.dealer.policy.state_dict())
                logger.info(f"  庄家策略更新: reward={self.best_dealer_reward:.2f}")
        
        retailer_pnls = [(i, r.get_pnl(self.env.orderbook.mid_price)) 
                         for i, r in enumerate(self.retailers)]
        retailer_pnls.sort(key=lambda x: x[1], reverse=True)
        
        top_k = max(1, len(self.retailers) // 5)
        for rank, (idx, pnl) in enumerate(retailer_pnls[top_k:]):
            if np.random.random() < self.cfg.lamarck_rate:
                donor_idx = retailer_pnls[np.random.randint(top_k)][0]
                donor_params = self.retailers[donor_idx].policy.state_dict()
                
                target_params = self.retailers[idx].policy.state_dict()
                for key in target_params:
                    target_params[key] = (
                        0.7 * target_params[key] + 0.3 * donor_params[key]
                    )
                self.retailers[idx].policy.load_state_dict(target_params)
    
    def _darwin_evolution(self):
        """达尔文进化"""
        for param in self.dealer.policy.parameters():
            if np.random.random() < self.cfg.darwin_rate:
                param.data += torch.randn_like(param) * 0.01
        
        for r in self.retailers:
            for param in r.policy.parameters():
                if np.random.random() < self.cfg.darwin_rate:
                    param.data += torch.randn_like(param) * 0.01
        
        for h in self.hotmoney_agents:
            for param in h.policy.parameters():
                if np.random.random() < self.cfg.darwin_rate:
                    param.data += torch.randn_like(param) * 0.01
    
    def _compute_strategy_diversity(self) -> float:
        all_params = []
        for r in self.retailers:
            flat = torch.cat([p.flatten() for p in r.policy.parameters()]).detach().numpy()
            all_params.append(flat)
        
        if len(all_params) < 2:
            return 1.0
        
        all_params = np.array(all_params)
        mean_vec = all_params.mean(axis=0)
        diversities = []
        for i in range(len(all_params)):
            cos_sim = np.dot(all_params[i], mean_vec) / (
                np.linalg.norm(all_params[i]) * np.linalg.norm(mean_vec) + 1e-8)
            diversities.append(1 - abs(cos_sim))
        
        return np.mean(diversities)
    
    def _inject_diversity_noise(self):
        for r in self.retailers:
            for param in r.policy.parameters():
                param.data += torch.randn_like(param) * 0.05
    
    def _anti_degenerate_check(self, tick: int, episode: int):
        """
        五方案防摆烂:
          1. 策略多样性奖励 (在reward中已体现)
          2. 最低行动阈值: 散户不能连续10步不交易
          3. 惩罚恒定策略: 连续相同动作会被惩罚
          4. 注入随机探索: 每步5%概率完全随机行动
          5. ★ 强制活跃检查: 检查实际成交量, 全零则注入市价单
        """
        for r in self.retailers:
            # 方案2: 最低行动阈值
            if tick > 0 and tick % 10 == 0:
                recent_volumes = [a.get('volume', 0) for a in r.history[-10:]]
                if sum(recent_volumes) == 0 and r.retailer_type != 'passive':
                    # 强制注入活力
                    r.capital = max(r.capital, r.initial_capital * 0.1)
            
            # 方案3: 惩罚恒定策略 (检测连续相同动作)
            if len(r.history) >= 5:
                last_5 = [str(a.get('side', '')) + str(round(a.get('volume', 0), 2)) 
                          for a in r.history[-5:]]
                if len(set(last_5)) == 1:
                    # 连续5步完全相同 → 注入变异
                    for param in r.policy.parameters():
                        param.data += torch.randn_like(param) * 0.01
            
            # 方案4: 随机探索
            if np.random.random() < 0.05:
                for param in r.policy.parameters():
                    param.data += torch.randn_like(param) * 0.003
        
        # ★ 方案5: 全局成交量检查 (每隔50个tick)
        if tick > 0 and tick % 50 == 0:
            total_volume = sum(r.history[-50:] and 
                              [a.get('volume', 0) for a in r.history[-50:]] 
                              if r.history else [0] 
                              for r in self.retailers)
            if isinstance(total_volume, list):
                total_volume = sum(total_volume)
            if total_volume < 1e-6:
                # 全部散户0成交量 → 强制注入
                logger.debug(f"  [Ep{episode} t{tick}] 全局成交量接近0, 注入活力")
                for r in self.retailers[:5]:
                    if r.retailer_type != 'passive':
                        r.capital = max(r.capital, r.initial_capital * 0.15)
    
    def _reset_agents(self):
        """重置所有Agent状态"""
        self.dealer.capital = self.dealer.initial_capital
        self.dealer.holdings = 0
        self.dealer.avg_cost = 0
        self.dealer.t1_locked = 0
        self.dealer.phase = 'accumulate'
        self.dealer.phase_ticks = 0
        
        for r in self.retailers:
            r.capital = r.initial_capital
            r.holdings = 0
            r.avg_cost = 0
            r.t1_locked = 0
            r.panicking = False
            r.fomo = False
            r.salary_timer = 0
        
        for h in self.hotmoney_agents:
            h.capital = h.initial_capital
            h.holdings = 0
            h.avg_cost = 0
            h.t1_locked = 0
            h.hold_ticks = 0
            h.entry_price = 0
    
    def _save_training_results(self, all_rewards: Dict):
        """保存训练结果"""
        results = {
            'config': {k: str(v) if isinstance(v, (list, Path)) else v 
                      for k, v in vars(self.cfg).items()},
            'generation': self.generation,
            'final_rewards': {k: float(np.mean(v[-100:])) for k, v in all_rewards.items()},
            'strategy_diversity': self.strategy_diversity_history,
            'firing_stats': self.firing_stats,
            'limit_stats': self.limit_stats,
        }
        
        # 保存模型
        save_dir = ADV_MODEL_DIR
        save_dir.mkdir(parents=True, exist_ok=True)
        
        torch.save({
            'dealer': self.dealer.policy.state_dict(),
            'retailers': [r.policy.state_dict() for r in self.retailers],
            'hotmoney': [h.policy.state_dict() for h in self.hotmoney_agents],
            'results': results,
        }, save_dir / "adversarial_model.pt")
        
        # 保存价格轨迹
        price_data = pd.DataFrame({
            'price': self.env.orderbook.price_history,
            'volume': self.env.orderbook.volume_history,
        })
        price_data.to_csv(ADV_DATA_DIR / "last_episode_prices.csv", index=False)
        
        with open(RESULTS_DIR / "adversarial_results.json", 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        logger.info(f"训练结果已保存: {save_dir}")
        
        # 保存训练器状态
        trainer_state = {
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'generation': self.generation,
            'dealer_info_power': self.dealer.info_power,
            'dealer_shake_intensity': self.dealer.shake_intensity,
            'dealer_puppet_capital': self.dealer.puppet_capital,
            'strategy_diversity': self.strategy_diversity_history[-1] if self.strategy_diversity_history else 0,
            'cal_reward_weight': self._cal_reward_weight,
            'firing_stats': self.firing_stats,
            'limit_stats': self.limit_stats,
        }
        state_file = RESULTS_DIR / "trainer_state.json"
        with open(state_file, 'w', encoding='utf-8') as f:
            json.dump(trainer_state, f, ensure_ascii=False, indent=2)
        logger.info(f"训练器状态已保存: {state_file}")


# ============================================================
#  命令行入口
# ============================================================
def main():
    parser = argparse.ArgumentParser(description="庄散对抗环境 (v2.2 基准线版)")
    parser.add_argument("--mode", required=True,
                        choices=["train", "evaluate", "demo"])
    parser.add_argument("--episodes", type=int, default=1000)
    parser.add_argument("--evolve", action="store_true")
    parser.add_argument("--model-path", type=str,
                        default=str(ADV_MODEL_DIR / "adversarial_model.pt"))
    # ★ v2.2: 价格基准线
    parser.add_argument("--price-data", type=str, default="",
                        help="价格基准线文件路径(.npy), 假数据或实盘数据; 不传则纯模拟")
    parser.add_argument("--benchmark-source", type=str, default="",
                        help="基准线来源标注: fake/real, 仅日志用")
    
    args = parser.parse_args()
    cfg = AdversarialConfig(num_episodes=args.episodes)
    
    # ★ v2.2: 加载价格基准线
    price_benchmark = None
    if args.price_data:
        data_path = Path(args.price_data)
        if not data_path.exists():
            # 尝试在adversarial data目录下找
            data_path = ADV_DATA_DIR / args.price_data
        if data_path.exists():
            price_benchmark = np.load(str(data_path))
            logger.info(f"★ 加载价格基准线: {data_path} | 形状={price_benchmark.shape} | 来源={args.benchmark_source or '未标注'}")
        else:
            logger.warning(f"★ 价格基准线文件不存在: {args.price_data}, 退回纯模拟模式")
    
    if args.mode == "train":
        trainer = AdversarialTrainer(cfg, price_benchmark=price_benchmark,
                                      benchmark_source=args.benchmark_source)
        trainer.train(num_episodes=args.episodes, evolve=args.evolve)
    
    elif args.mode == "evaluate":
        trainer = AdversarialTrainer(cfg)
        ckpt = torch.load(args.model_path, map_location='cpu', weights_only=False)
        trainer.dealer.policy.load_state_dict(ckpt['dealer'])
        for i, r in enumerate(trainer.retailers):
            if i < len(ckpt['retailers']):
                r.policy.load_state_dict(ckpt['retailers'][i])
        
        rewards = trainer.train(num_episodes=100, evolve=False)
        
        logger.info(f"评估结果:")
        logger.info(f"  庄家平均: {np.mean(rewards['dealer']):.2f}")
        logger.info(f"  散户平均: {np.mean(rewards['retailer']):.2f}")
        logger.info(f"  游资平均: {np.mean(rewards['hotmoney']):.2f}")
    
    elif args.mode == "demo":
        cfg.num_episodes = 50
        cfg.episode_length = 60
        trainer = AdversarialTrainer(cfg)
        trainer.train(num_episodes=50, evolve=True)
        
        prices = trainer.env.orderbook.price_history
        logger.info(f"\n价格轨迹 (最近50点):")
        for i, p in enumerate(prices[-50:]):
            bar = "█" * int(abs(p - cfg.initial_price) / cfg.initial_price * 1000)
            sign = "+" if p > cfg.initial_price else "-"
            logger.info(f"  {i:3d} | {p:.2f} {sign}{bar}")

if __name__ == "__main__":
    main()

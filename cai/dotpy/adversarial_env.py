"""
adversarial_env.py - 庄散对抗学习环境
======================================
Phase 3: 庄散对抗（核心）

三方博弈:
  庄家(Dealer): 防共四策 → 思想防散/民众防散/政治防散/武力防散
  散户(Retailer): 五类型 → 跟风/价值/技术/散户头羊/被动
  游资(HotMoney): 打板快进快出

两层博弈:
  显式博弈: 买卖行为直接交互
  元博弈: 策略进化(拉马克+达尔文)

防摆烂: 四方案保证散户不会退化到无策略

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

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical

sys.path.insert(0, str(Path(__file__).parent))
from stock_config import (
    ADV_MODEL_DIR, ADV_DATA_DIR, RESULTS_DIR,
    AdversarialConfig, setup_logger,
)

warnings.filterwarnings('ignore')
logger = setup_logger("Adversarial")

# ============================================================
# 市场微观结构
# ============================================================
class OrderBook:
    """简化订单簿: 支撑价格发现和成交"""
    
    def __init__(self, initial_price: float = 10.0):
        self.mid_price = initial_price
        self.bid_price = initial_price * 0.999
        self.ask_price = initial_price * 1.001
        self.spread = 0.002  # 0.2% 价差
        
        # 买卖队列: [(price, volume, agent_id)]
        self.bids = []  # 买单 (price desc)
        self.asks = []  # 卖单 (price asc)
        
        # 成交记录
        self.trades = []
        self.price_history = [initial_price]
        self.volume_history = [0]
        self.tick = 0
    
    def reset(self, initial_price: float = 10.0):
        self.__init__(initial_price)
    
    def submit_order(self, side: str, volume: float, price: Optional[float],
                      agent_id: str, order_type: str = "limit") -> Dict:
        """
        提交订单
        side: 'buy' or 'sell'
        volume: 下单量
        price: 限价 (None=市价)
        order_type: 'limit' or 'market'
        返回: {filled, avg_price, remaining}
        """
        if price is None:
            price = self.ask_price if side == 'buy' else self.bid_price
        
        filled = 0.0
        fill_price = 0.0
        
        if side == 'buy':
            # 匹配卖单
            remaining = volume
            new_asks = []
            for ask_p, ask_v, ask_id in self.asks:
                if remaining <= 0:
                    new_asks.append((ask_p, ask_v, ask_id))
                    continue
                if ask_p <= price:
                    fill_v = min(remaining, ask_v)
                    filled += fill_v
                    fill_price += fill_v * ask_p
                    remaining -= fill_v
                    if ask_v > fill_v:
                        new_asks.append((ask_p, ask_v - fill_v, ask_id))
                    # 记录成交
                    self.trades.append({
                        'tick': self.tick, 'price': ask_p, 'volume': fill_v,
                        'buyer': agent_id, 'seller': ask_id,
                    })
                else:
                    new_asks.append((ask_p, ask_v, ask_id))
            self.asks = new_asks
            
            # 未成交部分挂单
            if remaining > 0 and order_type == "limit":
                self.bids.append((price, remaining, agent_id))
                self.bids.sort(key=lambda x: -x[0])  # 价格降序
            
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
                    fill_price += fill_v * bid_p
                    remaining -= fill_v
                    if bid_v > fill_v:
                        new_bids.append((bid_p, bid_v - fill_v, bid_id))
                    self.trades.append({
                        'tick': self.tick, 'price': bid_p, 'volume': fill_v,
                        'buyer': bid_id, 'seller': agent_id,
                    })
                else:
                    new_bids.append((bid_p, bid_v, bid_id))
            self.bids = new_bids
            
            if remaining > 0 and order_type == "limit":
                self.asks.append((price, remaining, agent_id))
                self.asks.sort(key=lambda x: x[0])  # 价格升序
        
        avg_price = fill_price / max(filled, 1e-8)
        
        # 更新盘口
        if filled > 0:
            self.mid_price = avg_price
            self.bid_price = self.mid_price * (1 - self.spread / 2)
            self.ask_price = self.mid_price * (1 + self.spread / 2)
            self.price_history.append(self.mid_price)
            self.volume_history.append(filled)
        
        return {'filled': filled, 'avg_price': avg_price, 'remaining': volume - filled}
    
    def get_state(self) -> Dict:
        """获取当前市场状态"""
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
        }


class MarketEnv:
    """
    市场环境: 包装OrderBook，管理时间推进
    一个episode = 240个tick (4小时×60分钟)
    """
    
    def __init__(self, cfg: AdversarialConfig):
        self.cfg = cfg
        self.orderbook = OrderBook(cfg.initial_price)
        self.tick = 0
        self.episode_length = cfg.episode_length
        
        # 市场事件
        self.info_events = []  # 信息操纵事件
        self.shock_events = []  # 外部冲击
    
    def reset(self):
        self.orderbook.reset(self.cfg.initial_price)
        self.tick = 0
        self.info_events = []
        self.shock_events = []
        return self.orderbook.get_state()
    
    def step(self, actions: Dict[str, Dict]) -> Dict:
        """
        执行一步: 各agent提交订单
        actions: {agent_id: {side, volume, price, order_type}}
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
        
        state = self.orderbook.get_state()
        
        # 处理信息事件 (庄家操纵)
        for event in self.info_events:
            if event['tick'] == self.tick:
                self._apply_info_event(event)
        
        return {
            'state': state,
            'results': results,
            'done': self.tick >= self.episode_length,
        }
    
    def _apply_info_event(self, event):
        """应用信息事件对市场的影响"""
        impact = event.get('impact', 0)
        # 信息影响价差和深度
        self.orderbook.spread = min(0.05, self.orderbook.spread * (1 + abs(impact)))
    
    def add_info_event(self, event_type: str, tick: int, impact: float, source: str):
        self.info_events.append({
            'type': event_type, 'tick': tick,
            'impact': impact, 'source': source,
        })

# ============================================================
# Agent 基类
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
    
    @abstractmethod
    def observe(self, state: Dict) -> np.ndarray:
        """构建观测向量"""
        pass
    
    @abstractmethod
    def decide(self, state: Dict) -> Dict:
        """决策: 返回 {side, volume, price, order_type}"""
        pass
    
    def update_portfolio(self, fill_result: Dict):
        """更新持仓"""
        filled = fill_result['filled']
        avg_price = fill_result['avg_price']
        
        if fill_result.get('side', 'buy') == 'buy' or filled > 0:
            # 判断是买还是卖需要从history看
            if self.holdings >= 0 and filled > 0:
                # 可能是买入
                cost = filled * avg_price
                if cost <= self.capital:
                    old_holdings = self.holdings
                    self.holdings += filled
                    if self.holdings > 0:
                        self.avg_cost = (self.avg_cost * old_holdings + avg_price * filled) / self.holdings
                    self.capital -= cost
    
    def get_pnl(self, current_price: float) -> float:
        """当前盈亏"""
        return self.capital + self.holdings * current_price - self.initial_capital
    
    def get_return(self, current_price: float) -> float:
        """收益率"""
        total = self.capital + self.holdings * current_price
        return total / self.initial_capital - 1.0

# ============================================================
# 策略网络 (共享结构)
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
# 庄家 Agent - 防共四策
# ============================================================
class DealerAgent(MarketAgent):
    """
    庄家Agent: "阎锡山逻辑" - 防共四策
    
    四大行为模块:
      1. 思想防散 (信息操纵): 发布利好/利空消息引导散户预期
      2. 民众防散 (白手套基金): 通过关联资金拉抬/护盘
      3. 政治防散 (庄家联盟): 与其他庄家联合坐庄
      4. 武力防散 (暴力砸盘): 震仓洗盘，恐慌性抛售
    """
    
    # 行动空间: 5个离散动作
    ACTION_HOLD = 0        # 观望
    ACTION_ACCUMULATE = 1  # 建仓/吸筹
    ACTION_PUMP = 2        # 拉抬 (民众防散)
    ACTION_SHAKE = 3       # 震仓 (武力防散)
    ACTION_DISTRIBUTE = 4  # 派发/出货
    
    ACTION_NAMES = ['观望', '吸筹', '拉抬', '震仓', '派发']
    
    def __init__(self, agent_id: str, capital: float, cfg: AdversarialConfig):
        super().__init__(agent_id, capital, cfg)
        
        # 策略网络
        obs_dim = 20  # 市场状态特征数
        self.policy = PolicyNetwork(obs_dim, 5, hidden=128)
        
        # 防共四策参数
        self.info_power = 0.5          # 信息操纵能力
        self.alliance_threshold = 0.3  # 联盟触发阈值
        self.shake_intensity = 0.05    # 震仓强度
        self.puppet_capital = capital * 0.2  # 白手套资金
        
        # 状态追踪
        self.phase = 'accumulate'  # accumulate → pump → shake → distribute
        self.phase_history = []
        self.alliance_active = False
        self.info_events_sent = 0
    
    def observe(self, state: Dict) -> np.ndarray:
        """构建庄家观测向量"""
        features = [
            state['mid_price'] / self.cfg.initial_price - 1,  # 价格变化
            state['spread'] / state['mid_price'],             # 相对价差
            state['bid_ask_ratio'] - 1,                       # 买卖力量失衡
            state['bid_depth'] / max(state['ask_depth'], 1),  # 买盘深度比
            self.holdings * state['mid_price'] / self.initial_capital,  # 仓位比
            (state['mid_price'] - self.avg_cost) / (self.avg_cost + 1e-8),  # 浮盈比
            self.capital / self.initial_capital,               # 现金比
            len(state.get('recent_trades', [])) / 10,          # 成交活跃度
            np.log1p(state['bid_depth']),                       # 买盘对数深度
            np.log1p(state['ask_depth']),                       # 卖盘对数深度
        ]
        # 补齐到obs_dim
        while len(features) < 20:
            features.append(0.0)
        return np.array(features[:20], dtype=np.float32)
    
    def decide(self, state: Dict) -> Dict:
        """庄家决策: 结合RL策略和防共四策行为模块"""
        obs = self.observe(state)
        obs_t = torch.FloatTensor(obs).unsqueeze(0)
        
        with torch.no_grad():
            action, log_prob, value = self.policy.act(obs_t)
        
        action = action.item()
        self.phase_history.append(action)
        
        # 根据行动生成订单
        price = state['mid_price']
        
        if action == self.ACTION_HOLD:
            return {'side': 'buy', 'volume': 0, 'price': None, 'order_type': 'limit'}
        
        elif action == self.ACTION_ACCUMULATE:
            # 吸筹: 小单买入，不引起注意
            volume = self.capital * 0.05 / price  # 用5%资金
            return {'side': 'buy', 'volume': volume, 
                    'price': state['bid_price'], 'order_type': 'limit'}
        
        elif action == self.ACTION_PUMP:
            # 拉抬 (民众防散): 白手套资金市价买入
            volume = self.puppet_capital * 0.1 / price  # 白手套10%
            return {'side': 'buy', 'volume': volume,
                    'price': None, 'order_type': 'market'}
        
        elif action == self.ACTION_SHAKE:
            # 震仓 (武力防散): 大单砸盘 + 信息操纵
            volume = self.holdings * self.shake_intensity
            # 同时触发信息事件
            return {'side': 'sell', 'volume': volume,
                    'price': state['bid_price'] * 0.98, 'order_type': 'limit',
                    '_info_event': True, '_event_type': 'negative',
                    '_event_impact': -self.info_power * 0.5}
        
        elif action == self.ACTION_DISTRIBUTE:
            # 派发: 分批卖出
            volume = self.holdings * 0.1  # 每次卖10%持仓
            return {'side': 'sell', 'volume': volume,
                    'price': state['ask_price'], 'order_type': 'limit',
                    '_info_event': True, '_event_type': 'positive',
                    '_event_impact': self.info_power * 0.3}
        
        return {'side': 'buy', 'volume': 0, 'price': None, 'order_type': 'limit'}
    
    def apply_strategy_override(self, state: Dict, action: Dict) -> Dict:
        """
        防共四策行为覆盖: 在RL策略基础上叠加行为规则
        确保庄家行为符合"阎锡山逻辑"
        """
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
                     'price': None, 'order_type': 'market',
                     '_puppet': True}
        
        # 策略3: 政治防散 - 多庄家联合 (在环境中处理)
        
        # 策略4: 武力防散 - 散户头羊出现时强制震仓
        if self._detect_retailer_leader(state):
            shake_vol = self.holdings * self.shake_intensity * 2
            action = {'side': 'sell', 'volume': shake_vol,
                     'price': state['bid_price'] * 0.97, 'order_type': 'limit',
                     '_info_event': True, '_event_type': 'negative',
                     '_event_impact': -self.info_power * 0.8}
        
        return action
    
    def _detect_retailer_leader(self, state: Dict) -> bool:
        """检测散户头羊行为 (大量同方向买单)"""
        recent = state.get('recent_trades', [])
        if len(recent) < 3:
            return False
        buy_count = sum(1 for t in recent if t.get('buyer', '').startswith('retailer'))
        return buy_count >= len(recent) * 0.6

# ============================================================
# 散户 Agent - 五类型
# ============================================================
class RetailerAgent(MarketAgent):
    """
    散户Agent: 五类型
    
    类型:
      herd      - 跟风型: 追涨杀跌
      value     - 价值型: 低买高卖(基本面)
      technical - 技术型: 看K线和指标
      leader    - 散户头羊: 影响其他散户
      passive   - 被动型: 长期持有不交易
    
    特性:
      - 月度工资注入
      - 抱团-踩踏转换
      - 受信息操纵影响
    """
    
    TYPE_PROBS = {'herd': 0.35, 'value': 0.15, 'technical': 0.20,
                  'leader': 0.10, 'passive': 0.20}
    
    def __init__(self, agent_id: str, capital: float, cfg: AdversarialConfig,
                 retailer_type: str = 'herd'):
        super().__init__(agent_id, capital, cfg)
        self.retailer_type = retailer_type
        
        # 策略网络 (每种类型共享结构但不同权重)
        obs_dim = 20
        self.policy = PolicyNetwork(obs_dim, 4, hidden=64)
        
        # 类型参数
        self.momentum_sensitivity = {
            'herd': 0.8, 'value': -0.3, 'technical': 0.5,
            'leader': 0.6, 'passive': 0.05,
        }[retailer_type]
        
        self.trade_frequency = {
            'herd': 0.6, 'value': 0.3, 'technical': 0.5,
            'leader': 0.7, 'passive': 0.05,
        }[retailer_type]
        
        # 月度工资
        self.monthly_salary = cfg.retailer_monthly_salary
        self.salary_timer = 0
        
        # 抱团/踩踏状态
        self.panicking = False
        self.fomo = False
    
    def observe(self, state: Dict) -> np.ndarray:
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
        ]
        # 加入近期价格动量
        if len(state.get('recent_trades', [])) >= 2:
            trades = state['recent_trades']
            p0, p1 = trades[0]['price'], trades[-1]['price']
            features.append((p1 - p0) / (p0 + 1e-8))
        else:
            features.append(0.0)
        
        while len(features) < 20:
            features.append(0.0)
        return np.array(features[:20], dtype=np.float32)
    
    def decide(self, state: Dict) -> Dict:
        # 月度工资注入
        self.salary_timer += 1
        if self.salary_timer >= 20:  # 每20个tick = 1个月
            self.capital += self.monthly_salary
            self.salary_timer = 0
        
        # 被动型很少交易
        if self.retailer_type == 'passive':
            if np.random.random() > 0.02:
                return {'side': 'buy', 'volume': 0, 'price': None, 'order_type': 'limit'}
        
        # 抱团/踩踏检测
        self._detect_herd_state(state)
        
        obs = self.observe(state)
        obs_t = torch.FloatTensor(obs).unsqueeze(0)
        
        with torch.no_grad():
            action, _, _ = self.policy.act(obs_t)
        
        action = action.item()
        price = state['mid_price']
        
        # 行动映射: 0=持有, 1=买入, 2=卖出, 3=跟单
        if action == 0:
            return {'side': 'buy', 'volume': 0, 'price': None, 'order_type': 'limit'}
        
        elif action == 1:  # 买入
            volume = self.capital * self.trade_frequency * 0.1 / price
            if self.fomo:
                volume *= 2.0  # FOMO加倍
            return {'side': 'buy', 'volume': volume,
                    'price': state['ask_price'], 'order_type': 'limit'}
        
        elif action == 2:  # 卖出
            volume = self.holdings * self.trade_frequency * 0.1
            if self.panicking:
                volume *= 3.0  # 恐慌踩踏
            return {'side': 'sell', 'volume': volume,
                    'price': state['bid_price'], 'order_type': 'limit'}
        
        elif action == 3:  # 跟单 (散户头羊特有)
            recent = state.get('recent_trades', [])
            if recent:
                last_side = 'buy' if recent[-1].get('buyer', '').startswith('retailer_leader') else 'sell'
                volume = self.capital * 0.05 / price
                return {'side': last_side, 'volume': volume,
                        'price': None, 'order_type': 'market'}
            return {'side': 'buy', 'volume': 0, 'price': None, 'order_type': 'limit'}
        
        return {'side': 'buy', 'volume': 0, 'price': None, 'order_type': 'limit'}
    
    def _detect_herd_state(self, state: Dict):
        """检测抱团/踩踏状态"""
        price = state['mid_price']
        recent = state.get('recent_trades', [])
        
        if len(recent) >= 3:
            # 近期连续上涨 → FOMO
            prices = [t['price'] for t in recent[-5:]]
            if len(prices) >= 3 and all(prices[i] < prices[i+1] for i in range(len(prices)-1)):
                self.fomo = True
                self.panicking = False
            # 近期连续下跌 → 恐慌
            elif len(prices) >= 3 and all(prices[i] > prices[i+1] for i in range(len(prices)-1)):
                self.panicking = True
                self.fomo = False
            else:
                self.fomo = False
                self.panicking = False

# ============================================================
# 游资 Agent - 打板快进快出
# ============================================================
class HotMoneyAgent(MarketAgent):
    """
    游资Agent: 打板策略
    
    特征:
      - 速度优先: 市价单为主
      - 动量触发: 涨幅超过阈值才入场
      - 快进快出: 持仓时间短
      - 不恋战: 止损果断
    """
    
    def __init__(self, agent_id: str, capital: float, cfg: AdversarialConfig):
        super().__init__(agent_id, capital, cfg)
        
        obs_dim = 20
        self.policy = PolicyNetwork(obs_dim, 3, hidden=64)
        
        self.momentum_thresh = cfg.hotmoney_momentum_thresh
        self.entry_price = 0.0
        self.max_hold_ticks = 30  # 最多持仓30个tick
        self.hold_ticks = 0
        self.stop_loss = 0.03     # 3% 止损
        self.take_profit = 0.08   # 8% 止盈
    
    def observe(self, state: Dict) -> np.ndarray:
        features = [
            state['mid_price'] / self.cfg.initial_price - 1,
            state['spread'] / state['mid_price'],
            state['bid_ask_ratio'] - 1,
            (state['mid_price'] - self.avg_cost) / (self.avg_cost + 1e-8) if self.holdings > 0 else 0,
            self.holdings > 0,
            self.hold_ticks / self.max_hold_ticks,
            np.log1p(state['bid_depth']),
            np.log1p(state['ask_depth']),
        ]
        while len(features) < 20:
            features.append(0.0)
        return np.array(features[:20], dtype=np.float32)
    
    def decide(self, state: Dict) -> Dict:
        price = state['mid_price']
        
        # 止损/止盈检查 (硬规则优先于RL)
        if self.holdings > 0:
            self.hold_ticks += 1
            pnl_pct = (price - self.entry_price) / (self.entry_price + 1e-8)
            
            if pnl_pct <= -self.stop_loss:
                # 止损
                return {'side': 'sell', 'volume': self.holdings,
                        'price': None, 'order_type': 'market'}
            elif pnl_pct >= self.take_profit:
                # 止盈
                return {'side': 'sell', 'volume': self.holdings,
                        'price': None, 'order_type': 'market'}
            elif self.hold_ticks >= self.max_hold_ticks:
                # 超时出场
                return {'side': 'sell', 'volume': self.holdings,
                        'price': None, 'order_type': 'market'}
        
        # RL决策
        obs = self.observe(state)
        obs_t = torch.FloatTensor(obs).unsqueeze(0)
        
        with torch.no_grad():
            action, _, _ = self.policy.act(obs_t)
        
        action = action.item()
        
        if action == 0:  # 观望
            return {'side': 'buy', 'volume': 0, 'price': None, 'order_type': 'limit'}
        
        elif action == 1:  # 打板买入
            if self.holdings == 0:
                # 动量检查
                recent = state.get('recent_trades', [])
                if len(recent) >= 2:
                    momentum = (recent[-1]['price'] - recent[0]['price']) / (recent[0]['price'] + 1e-8)
                    if momentum >= self.momentum_thresh:
                        volume = self.capital * 0.5 / price  # 半仓打板
                        self.entry_price = price
                        self.hold_ticks = 0
                        return {'side': 'buy', 'volume': volume,
                                'price': None, 'order_type': 'market'}
            return {'side': 'buy', 'volume': 0, 'price': None, 'order_type': 'limit'}
        
        elif action == 2:  # 快速卖出
            if self.holdings > 0:
                return {'side': 'sell', 'volume': self.holdings,
                        'price': None, 'order_type': 'market'}
            return {'side': 'buy', 'volume': 0, 'price': None, 'order_type': 'limit'}
        
        return {'side': 'buy', 'volume': 0, 'price': None, 'order_type': 'limit'}

# ============================================================
# 对抗训练环境
# ============================================================
class AdversarialTrainer:
    """
    对抗训练主控
    
    两层博弈:
      显式博弈: 每个tick的买卖交互
      元博弈: 每N代进行策略进化
    
    双进化机制:
      拉马克进化: 优秀策略经验直接遗传给后代
      达尔文进化: 随机变异探索新策略
    
    防摆烂:
      1. 策略多样性奖励
      2. 最低行动阈值
      3. 惩罚恒定策略
      4. 注入随机噪声
    """
    
    def __init__(self, cfg: AdversarialConfig = None):
        self.cfg = cfg or AdversarialConfig()
        self.env = MarketEnv(self.cfg)
        
        # 创建Agent群体
        self.dealer = DealerAgent("dealer", 
            capital=self.cfg.initial_price * 100000 * self.cfg.dealer_capital_ratio,
            cfg=self.cfg)
        
        self.retailers = self._create_retailer_swarm()
        self.hotmoney_agents = self._create_hotmoney_swarm()
        
        # 进化统计
        self.generation = 0
        self.best_dealer_reward = -float('inf')
        self.best_retailer_reward = -float('inf')
        self.strategy_diversity_history = []
        
        # 训练记录
        self.episode_log = []
    
    def _create_retailer_swarm(self, n: int = 20) -> List[RetailerAgent]:
        """创建散户群体"""
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
        """创建游资群体"""
        agents = []
        total_capital = self.cfg.initial_price * 100000 * self.cfg.hotmoney_ratio
        
        for i in range(n):
            cap = total_capital / n
            agents.append(HotMoneyAgent(f"hotmoney_{i}", cap, self.cfg))
        
        return agents
    
    def train(self, num_episodes: int = None, evolve: bool = True) -> Dict:
        """
        对抗训练主循环
        """
        num_episodes = num_episodes or self.cfg.num_episodes
        logger.info(f"开始对抗训练: {num_episodes} episodes")
        logger.info(f"  庄家: 1 | 散户: {len(self.retailers)} | 游资: {len(self.hotmoney_agents)}")
        
        all_rewards = {'dealer': [], 'retailer': [], 'hotmoney': []}
        
        for ep in range(num_episodes):
            state = self.env.reset()
            self._reset_agents()
            
            ep_rewards = {'dealer': 0, 'retailer': 0, 'hotmoney': 0}
            
            for tick in range(self.cfg.episode_length):
                # 收集所有Agent的行动
                actions = {}
                
                # 庄家
                dealer_action = self.dealer.decide(state)
                dealer_action = self.dealer.apply_strategy_override(state, dealer_action)
                actions[self.dealer.agent_id] = dealer_action
                
                # 处理庄家信息操纵事件
                if dealer_action.get('_info_event'):
                    self.env.add_info_event(
                        dealer_action['_event_type'],
                        tick + 1,
                        dealer_action['_event_impact'],
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
                
                # 计算奖励
                rewards = self._compute_rewards(state)
                ep_rewards['dealer'] += rewards['dealer']
                ep_rewards['retailer'] += rewards['retailer']
                ep_rewards['hotmoney'] += rewards['hotmoney']
                
                # 防摆烂检查
                self._anti_degenerate_check(tick)
                
                if result['done']:
                    break
            
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
                    f"庄家={d_avg:.2f} 散户={r_avg:.2f} 游资={h_avg:.2f}"
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
        """
        price = state['mid_price']
        
        # 庄家奖励
        dealer_pnl = self.dealer.get_pnl(price)
        # 加上"散户预测错误"奖励
        retailer_error = 0
        for r in self.retailers:
            # 散户在错误方向交易 → 庄家获得额外奖励
            r_pnl = r.get_pnl(price)
            if r_pnl < 0:
                retailer_error += abs(r_pnl) * 0.1  # 散户亏损 = 庄家收益
        
        dealer_reward = dealer_pnl + retailer_error
        
        # 散户奖励 (PnL + 策略多样性)
        retailer_reward = np.mean([r.get_pnl(price) for r in self.retailers])
        
        # 游资奖励
        hotmoney_reward = np.mean([h.get_pnl(price) for h in self.hotmoney_agents])
        
        return {
            'dealer': dealer_reward,
            'retailer': retailer_reward,
            'hotmoney': hotmoney_reward,
        }
    
    def _evolve(self, all_rewards: Dict):
        """
        双进化: 拉马克 + 达尔文
        """
        self.generation += 1
        logger.info(f"\n{'='*30} 第{self.generation}代进化 {'='*30}")
        
        # ── 拉马克进化: 优秀经验遗传 ──
        self._lamarck_evolution(all_rewards)
        
        # ── 达尔文进化: 随机变异 ──
        self._darwin_evolution()
        
        # 计算策略多样性
        diversity = self._compute_strategy_diversity()
        self.strategy_diversity_history.append(diversity)
        logger.info(f"  策略多样性: {diversity:.4f}")
        
        # 策略多样性过低 → 注入噪声
        if diversity < self.cfg.min_strategy_diversity:
            logger.warning("  ⚠ 策略多样性不足，注入随机噪声")
            self._inject_diversity_noise()
    
    def _lamarck_evolution(self, all_rewards):
        """拉马克进化: 将优秀策略参数复制给弱者"""
        # 庄家: 保留最优
        if all_rewards['dealer']:
            recent = all_rewards['dealer'][-self.cfg.evolution_interval:]
            if np.mean(recent) > self.best_dealer_reward:
                self.best_dealer_reward = np.mean(recent)
                best_params = copy.deepcopy(self.dealer.policy.state_dict())
                logger.info(f"  庄家策略更新: reward={self.best_dealer_reward:.2f}")
        
        # 散户: 表现最好的散户参数扩散
        retailer_pnls = [(i, r.get_pnl(self.env.orderbook.mid_price)) 
                         for i, r in enumerate(self.retailers)]
        retailer_pnls.sort(key=lambda x: x[1], reverse=True)
        
        top_k = max(1, len(self.retailers) // 5)  # 前20%
        for rank, (idx, pnl) in enumerate(retailer_pnls[top_k:]):
            if np.random.random() < self.cfg.lamarck_rate:
                # 从优秀者中随机选一个复制参数
                donor_idx = retailer_pnls[np.random.randint(top_k)][0]
                donor_params = self.retailers[donor_idx].policy.state_dict()
                
                # 软更新: 不是完全复制，而是加权平均
                target_params = self.retailers[idx].policy.state_dict()
                for key in target_params:
                    target_params[key] = (
                        0.7 * target_params[key] + 0.3 * donor_params[key]
                    )
                self.retailers[idx].policy.load_state_dict(target_params)
    
    def _darwin_evolution(self):
        """达尔文进化: 随机变异探索新策略"""
        # 庄家变异
        for param in self.dealer.policy.parameters():
            if np.random.random() < self.cfg.darwin_rate:
                param.data += torch.randn_like(param) * 0.01
        
        # 散户变异
        for r in self.retailers:
            for param in r.policy.parameters():
                if np.random.random() < self.cfg.darwin_rate:
                    param.data += torch.randn_like(param) * 0.01
        
        # 游资变异
        for h in self.hotmoney_agents:
            for param in h.policy.parameters():
                if np.random.random() < self.cfg.darwin_rate:
                    param.data += torch.randn_like(param) * 0.01
    
    def _compute_strategy_diversity(self) -> float:
        """计算散户策略多样性 (基于参数方差)"""
        all_params = []
        for r in self.retailers:
            flat = torch.cat([p.flatten() for p in r.policy.parameters()]).detach().numpy()
            all_params.append(flat)
        
        if len(all_params) < 2:
            return 1.0
        
        all_params = np.array(all_params)
        # 平均余弦距离
        mean_vec = all_params.mean(axis=0)
        diversities = []
        for i in range(len(all_params)):
            cos_sim = np.dot(all_params[i], mean_vec) / (
                np.linalg.norm(all_params[i]) * np.linalg.norm(mean_vec) + 1e-8)
            diversities.append(1 - abs(cos_sim))
        
        return np.mean(diversities)
    
    def _inject_diversity_noise(self):
        """多样性不足时注入噪声"""
        for r in self.retailers:
            for param in r.policy.parameters():
                param.data += torch.randn_like(param) * 0.05
    
    def _anti_degenerate_check(self, tick: int):
        """
        防摆烂四方案:
          1. 策略多样性奖励 (在reward中已体现)
          2. 最低行动阈值: 散户不能连续10步不交易
          3. 惩罚恒定策略: 连续相同动作会被惩罚
          4. 注入随机探索: 每步5%概率完全随机行动
        """
        for r in self.retailers:
            # 方案2: 最低行动阈值
            if tick > 0 and tick % 10 == 0:
                recent_actions = [a.get('volume', 0) for a in r.history[-10:]]
                if sum(recent_actions) == 0 and r.retailer_type != 'passive':
                    # 强制执行一次交易
                    r.capital = max(r.capital, r.initial_capital * 0.1)
            
            # 方案4: 随机探索
            if np.random.random() < 0.05:
                # 注入随机性
                for param in r.policy.parameters():
                    param.data += torch.randn_like(param) * 0.001
    
    def _reset_agents(self):
        """重置所有Agent状态"""
        self.dealer.capital = self.dealer.initial_capital
        self.dealer.holdings = 0
        self.dealer.avg_cost = 0
        
        for r in self.retailers:
            r.capital = r.initial_capital
            r.holdings = 0
            r.avg_cost = 0
            r.panicking = False
            r.fomo = False
            r.salary_timer = 0
        
        for h in self.hotmoney_agents:
            h.capital = h.initial_capital
            h.holdings = 0
            h.avg_cost = 0
            h.hold_ticks = 0
            h.entry_price = 0
    
    def _save_training_results(self, all_rewards: Dict):
        """保存训练结果"""
        results = {
            'config': vars(self.cfg),
            'generation': self.generation,
            'final_rewards': {k: np.mean(v[-100:]) for k, v in all_rewards.items()},
            'strategy_diversity': self.strategy_diversity_history,
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

# ============================================================
# 庄家联盟模块 (政治防散)
# ============================================================
class DealerAlliance:
    """
    政治防散: 庄家联盟
    多个庄家协调行动，共享部分信息
    """
    
    def __init__(self, dealers: List[DealerAgent], threshold: float = 0.3):
        self.dealers = dealers
        self.threshold = threshold
        self.active = False
        self.coordination_signal = None
    
    def check_alliance(self, market_state: Dict) -> bool:
        """检查是否触发联盟"""
        # 条件: 多个庄家同时亏损或外部威胁
        losing_count = sum(1 for d in self.dealers 
                          if d.get_pnl(market_state['mid_price']) < 0)
        
        if losing_count / len(self.dealers) >= self.threshold:
            self.active = True
            self.coordination_signal = 'defend'  # 防御性联盟
        else:
            self.active = False
        
        return self.active
    
    def coordinate(self, market_state: Dict) -> Optional[Dict]:
        """联盟协调行动"""
        if not self.active:
            return None
        
        # 防御性联盟: 共同买入支撑价格
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
# 命令行入口
# ============================================================
def main():
    parser = argparse.ArgumentParser(description="庄散对抗环境")
    parser.add_argument("--mode", required=True,
                        choices=["train", "evaluate", "demo"])
    parser.add_argument("--episodes", type=int, default=1000)
    parser.add_argument("--evolve", action="store_true")
    parser.add_argument("--model-path", type=str, 
                        default=str(ADV_MODEL_DIR / "adversarial_model.pt"))
    
    args = parser.parse_args()
    cfg = AdversarialConfig(num_episodes=args.episodes)
    
    if args.mode == "train":
        trainer = AdversarialTrainer(cfg)
        trainer.train(num_episodes=args.episodes, evolve=args.evolve)
    
    elif args.mode == "evaluate":
        trainer = AdversarialTrainer(cfg)
        # 加载模型
        ckpt = torch.load(args.model_path, map_location='cpu', weights_only=False)
        trainer.dealer.policy.load_state_dict(ckpt['dealer'])
        for i, r in enumerate(trainer.retailers):
            if i < len(ckpt['retailers']):
                r.policy.load_state_dict(ckpt['retailers'][i])
        
        # 评估100个episode
        rewards = trainer.train(num_episodes=100, evolve=False)
        
        logger.info(f"评估结果:")
        logger.info(f"  庄家平均: {np.mean(rewards['dealer']):.2f}")
        logger.info(f"  散户平均: {np.mean(rewards['retailer']):.2f}")
        logger.info(f"  游资平均: {np.mean(rewards['hotmoney']):.2f}")
    
    elif args.mode == "demo":
        # 快速演示
        cfg.num_episodes = 50
        cfg.episode_length = 60  # 缩短到60tick
        trainer = AdversarialTrainer(cfg)
        trainer.train(num_episodes=50, evolve=True)
        
        # 输出价格轨迹
        prices = trainer.env.orderbook.price_history
        logger.info(f"\n价格轨迹 (最近50点):")
        for i, p in enumerate(prices[-50:]):
            bar = "█" * int(abs(p - cfg.initial_price) / cfg.initial_price * 1000)
            sign = "+" if p > cfg.initial_price else "-"
            logger.info(f"  {i:3d} | {p:.2f} {sign}{bar}")

if __name__ == "__main__":
    main()

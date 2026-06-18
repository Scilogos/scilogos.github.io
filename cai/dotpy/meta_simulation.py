#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
meta_simulation.py - 元叙事博弈模拟：训练多样化Agent群体学习庄家规律

核心思路：
  不再从"彩民怎么赢"出发，而是训练一批智能体去"理解庄家"。
  庄家操控的规律也是规律——如果能学到庄家选号函数的不变量，
  即使不知道实时投注分布，也能获得统计优势。

设计：
  - 6组训练环境，每组有不同的庄家策略变体
  - 每组100个多样化Agent（老辣+新手混合）
  - Agent输出1000维概率分布，而非简单选号
  - 训练2000轮后，用真实数据验证
  - 综合所有组的预测，统计高低频号码组合

用法:
  python meta_simulation.py --phase train --group 0          # 训练第0组
  python meta_simulation.py --phase train --group all        # 训练全部6组
  python meta_simulation.py --phase predict                  # 用真实数据预测
  python meta_simulation.py --phase validate                 # 验证预测结果
  python meta_simulation.py --phase full                     # 全流程
"""

import numpy as np
import pandas as pd
import json
import os
import sys
import argparse
import time
from collections import defaultdict
from copy import deepcopy

# ==================== 全局参数 ====================
N_NUMBERS = 1000
PAYOUT_ZHIXUAN = 520

# 市场结构
STRATEGIC_SHARE = 0.60   # 策略彩民占比（上调）
RANDOM_SHARE = 0.35      # 随机彩民占比
BANKER_SHARE = 0.05      # 保留

# 训练参数
N_GROUPS = 6
TRAIN_ROUNDS = 2000
AGENT_POP_SIZE = 100
EVOLVE_EVERY = 50
TOP_SURVIVE_RATE = 0.30
MUTATION_RATE = 0.20
MUTATION_STD = 0.4

# Agent类型池
AGENT_TYPES = [
    "pattern_learner",    # 模式学习器：学习庄家选号模式
    "freq_analyst",       # 频率分析师：统计号码出现频率
    "gap_hunter",         # 缺口猎人：寻找投注缺口
    "contrarian",         # 逆向者：反向操作
    "momentum",           # 动量追踪者：追热号
    "cold_catcher",       # 冷号捕捉者：买长期未出的号
    "entropy_hunter",     # 熵猎人：找信息熵异常的区间
    "meta_learner",       # 元学习者：学习其他agent的预测误差
    "novice_random",      # 新手：随机策略
    "novice_hot",         # 新手：追热门
]

# 庄家策略变体
BANKER_STRATEGIES = [
    "min_payout",         # 最小赔付（经典）
    "cold_preference",    # 偏好冷号
    "hot_avoidance",      # 回避热号
    "random_weighted",    # 按投注反比加权随机
    "anti_streak",        # 反连胜：避免连续开出同类型号
    "hybrid",             # 混合策略：轮换使用以上策略
]

# 真实数据分割
TRAIN_SPLIT = 6000       # 前6000期用于Agent学习历史
VALIDATE_SPLIT = 7633    # 后1633期用于验证


# ======================================================================
# 工具函数
# ======================================================================
def num_to_digits(n):
    return [n // 100, (n // 10) % 10, n % 10]

def digits_to_num(d):
    return d[0] * 100 + d[1] * 10 + d[2]

def softmax(x, temperature=1.0):
    e = np.exp((x - np.max(x)) / max(temperature, 0.01))
    return e / e.sum()

def kl_divergence(p, q, eps=1e-10):
    """KL(p||q)"""
    p = np.clip(p, eps, 1.0)
    q = np.clip(q, eps, 1.0)
    return np.sum(p * np.log(p / q))


# ======================================================================
# 特征提取器（增强版）
# ======================================================================
class EnhancedFeatureExtractor:
    """从历史数据中提取丰富的特征，供Agent和庄家使用"""

    def __init__(self, history_window=50):
        self.window = history_window
        self.history = []  # 最近N期开奖号码
        self.digit_history = [[], [], []]  # 百位/十位/个位分别的历史

    def update(self, drawn_number):
        self.history.append(drawn_number)
        d = num_to_digits(drawn_number)
        for i in range(3):
            self.digit_history[i].append(d[i])
        # 限制长度
        max_len = self.window * 3
        if len(self.history) > max_len:
            self.history = self.history[-max_len:]
            for i in range(3):
                self.digit_history[i] = self.digit_history[i][-max_len:]

    def get_number_features(self):
        """为所有1000个号码生成特征矩阵 (1000, n_features)
        
        特征：
          0: 频率得分 - 最近window期出现频率
          1: 冷度得分 - 距上次出现的归一化期数
          2: 热度得分 - 最近5期出现频率
          3: 近10期频率
          4: 吉利号得分 (8/6/9)
          5: 和值得分
          6: 跨度得分 (max-min of digits)
          7: 奇偶比
          8: 大小比
          9: 数字重复得分 (如112, 111)
          10: 邻号得分 (如123, 567)
          11: 偏置项
        """
        features = np.zeros((N_NUMBERS, 12))
        recent = self.history[-self.window:] if len(self.history) >= self.window else self.history
        recent5 = self.history[-5:] if len(self.history) >= 5 else self.history
        recent10 = self.history[-10:] if len(self.history) >= 10 else self.history

        for n in range(N_NUMBERS):
            d = num_to_digits(n)
            # 0: 频率
            features[n, 0] = recent.count(n) / max(len(recent), 1)
            # 1: 冷度
            last_seen = -1
            for i in range(len(self.history) - 1, -1, -1):
                if self.history[i] == n:
                    last_seen = len(self.history) - 1 - i
                    break
            features[n, 1] = last_seen / max(self.window, 1) if last_seen >= 0 else 1.0
            # 2: 热度
            features[n, 2] = recent5.count(n) / max(len(recent5), 1)
            # 3: 近10期
            features[n, 3] = recent10.count(n) / max(len(recent10), 1)
            # 4: 吉利号
            features[n, 4] = sum(1 for x in d if x in [8, 6, 9]) / 3
            # 5: 和值
            features[n, 5] = sum(d) / 27
            # 6: 跨度
            features[n, 6] = (max(d) - min(d)) / 9
            # 7: 奇偶比
            features[n, 7] = sum(1 for x in d if x % 2 == 1) / 3
            # 8: 大小比
            features[n, 8] = sum(1 for x in d if x >= 5) / 3
            # 9: 重复号
            features[n, 9] = 1.0 if len(set(d)) < 3 else 0.0
            # 10: 邻号
            is_seq = (d[1] == d[0]+1 and d[2] == d[1]+1) or (d[1] == d[0]-1 and d[2] == d[1]-1)
            features[n, 10] = 1.0 if is_seq else 0.0
            # 11: 偏置
            features[n, 11] = 1.0

        return features

    def get_digit_features(self):
        """为每位(百/十/个)生成特征矩阵 (3, 10, n_features)
        
        返回 shape (3, 10, 8):
          dim0=位置(0=百/1=十/2=个), dim1=数字(0-9), dim2=特征
          特征: 频率/冷度/热度/偏置/近5期频率/近10期频率/均值/标准差
        """
        result = np.zeros((3, 10, 8))
        for pos in range(3):
            hist = self.digit_history[pos]
            if not hist:
                continue
            recent = hist[-self.window:] if len(hist) >= self.window else hist
            for d in range(10):
                result[pos, d, 0] = recent.count(d) / max(len(recent), 1)  # 频率
                last_seen = -1
                for i in range(len(hist)-1, -1, -1):
                    if hist[i] == d:
                        last_seen = len(hist) - 1 - i
                        break
                result[pos, d, 1] = last_seen / max(self.window, 1) if last_seen >= 0 else 1.0  # 冷度
                recent5 = hist[-5:] if len(hist) >= 5 else hist
                result[pos, d, 2] = recent5.count(d) / max(len(recent5), 1)  # 热度
                result[pos, d, 3] = 1.0  # 偏置
                recent10 = hist[-10:] if len(hist) >= 10 else hist
                result[pos, d, 4] = recent10.count(d) / max(len(recent10), 1)
                result[pos, d, 5] = np.mean(hist[-30:]) / 9 if len(hist) >= 5 else 0.5  # 均值归一化
                result[pos, d, 6] = np.std(hist[-30:]) / 5 if len(hist) >= 5 else 0.3  # 标准差归一化
                result[pos, d, 7] = len(set(hist[-10:])) / 10 if len(hist) >= 5 else 0.5  # 多样性
        return result


# ======================================================================
# 庄家策略变体
# ======================================================================
class BankerStrategy:
    """多样化庄家策略"""

    def __init__(self, strategy_type="min_payout", noise=0.05, seed=42):
        self.strategy_type = strategy_type
        self.noise = noise
        self.rng = np.random.RandomState(seed)
        self.history = []
        self.round_count = 0

    def draw(self, total_dist, features):
        """根据策略选择开奖号码
        
        Args:
            total_dist: (1000,) 总投注分布
            features: (1000, 12) 号码特征矩阵
        """
        self.round_count += 1

        # 噪声模式：直接随机
        if self.rng.random() < self.noise:
            return self.rng.randint(0, N_NUMBERS)

        if self.strategy_type == "min_payout":
            # 最小赔付：选投注最少的号
            min_bet = total_dist.min()
            min_indices = np.where(total_dist <= min_bet * 1.01)[0]  # 允许1%容差
            drawn = self.rng.choice(min_indices)

        elif self.strategy_type == "cold_preference":
            # 冷号偏好：综合投注少+冷度
            cold_scores = features[:, 1]  # 冷度
            inv_dist = 1.0 / (total_dist + 1)
            scores = inv_dist * 0.6 + cold_scores * 0.4
            probs = softmax(scores, temperature=0.3)
            drawn = self.rng.choice(N_NUMBERS, p=probs)

        elif self.strategy_type == "hot_avoidance":
            # 热号回避：惩罚热号
            hot_penalty = features[:, 2] + features[:, 0]  # 热度+频率
            inv_dist = 1.0 / (total_dist + 1)
            scores = inv_dist * 0.7 - hot_penalty * 0.3
            probs = softmax(scores, temperature=0.3)
            drawn = self.rng.choice(N_NUMBERS, p=probs)

        elif self.strategy_type == "random_weighted":
            # 投注反比加权随机
            inv_dist = 1.0 / (total_dist + 1)
            probs = softmax(inv_dist, temperature=0.5)
            drawn = self.rng.choice(N_NUMBERS, p=probs)

        elif self.strategy_type == "anti_streak":
            # 反连胜：避免连续开出同类型号
            inv_dist = 1.0 / (total_dist + 1)
            base_scores = inv_dist
            if len(self.history) >= 3:
                last3 = self.history[-3:]
                last3_features = np.mean([features[n] for n in last3], axis=0)
                # 惩罚与最近3期特征相似的号码
                similarity = np.array([np.dot(features[i], last3_features) / 
                             (np.linalg.norm(features[i]) + 1e-10) for i in range(N_NUMBERS)])
                base_scores -= similarity * 0.3
            probs = softmax(base_scores, temperature=0.4)
            drawn = self.rng.choice(N_NUMBERS, p=probs)

        elif self.strategy_type == "hybrid":
            # 混合策略：每50轮切换
            phase = (self.round_count // 50) % 5
            strategies = ["min_payout", "cold_preference", "hot_avoidance", 
                         "random_weighted", "anti_streak"]
            temp_banker = BankerStrategy(strategies[phase], noise=0, seed=self.rng.randint(0, 100000))
            temp_banker.history = self.history
            temp_banker.round_count = self.round_count
            drawn = temp_banker.draw(total_dist, features)
        else:
            drawn = self.rng.randint(0, N_NUMBERS)

        self.history.append(drawn)
        return drawn


# ======================================================================
# 多样化Agent
# ======================================================================
class MetaAgent:
    """元叙事Agent：在模拟环境中训练，学习预测庄家行为"""

    N_FEATURES = 12  # 对应EnhancedFeatureExtractor

    def __init__(self, agent_type, seed=42, is_novice=False):
        self.rng = np.random.RandomState(seed)
        self.agent_type = agent_type
        self.is_novice = is_novice
        
        # 参数向量：决定如何将特征转化为预测分布
        if is_novice:
            # 新手：简单/少量特征
            self.weights = self.rng.randn(self.N_FEATURES) * 0.2
            self.temperature = 0.8  # 更高的温度=更不确定
            self.top_k = 50  # 关注更少的号码
            self.confidence = 0.3
        else:
            # 老手：丰富特征+低温度
            self.weights = self._init_weights_by_type()
            self.temperature = 0.3 + self.rng.random() * 0.4
            self.top_k = 20 + self.rng.randint(0, 30)
            self.confidence = 0.6 + self.rng.random() * 0.3

        self.bankroll = 10000.0 if not is_novice else 5000.0  # 增加资金防破产
        self.total_wagered = 0.0
        self.total_won = 0.0
        self.n_rounds = 0
        self.n_wins = 0
        self.prediction_history = []  # 每轮的预测分布
        self.accuracy_history = []    # 预测与实际的匹配度
        self.fitness = 0.0
        self._last_prediction = None

    def _init_weights_by_type(self):
        """根据Agent类型初始化不同的权重偏好"""
        w = self.rng.randn(self.N_FEATURES) * 0.3
        if self.agent_type == "pattern_learner":
            # 关注频率和近期模式
            w[0] = 1.5   # 频率
            w[3] = 1.2   # 近10期
            w[2] = 0.8   # 热度
        elif self.agent_type == "freq_analyst":
            # 纯频率分析
            w[0] = 3.0   # 频率
            w[1] = -1.0  # 冷度（负=回避冷号）
        elif self.agent_type == "gap_hunter":
            # 寻找投注缺口（需要配合投注分布）
            w[0] = -1.5  # 反频率
            w[1] = 2.0   # 冷度
        elif self.agent_type == "contrarian":
            # 逆向操作
            w[0] = -2.0  # 反频率
            w[2] = -2.0  # 反热度
            w[1] = 1.5   # 追冷
        elif self.agent_type == "momentum":
            # 追热
            w[2] = 3.0   # 热度
            w[0] = 1.0   # 频率
        elif self.agent_type == "cold_catcher":
            # 追冷
            w[1] = 3.0   # 冷度
        elif self.agent_type == "entropy_hunter":
            # 信息熵异常区
            w = self.rng.randn(self.N_FEATURES) * 0.5  # 更随机
        elif self.agent_type == "meta_learner":
            # 元学习：初始随机，通过训练调整
            w = self.rng.randn(self.N_FEATURES) * 0.1
        elif self.agent_type == "novice_random":
            w = self.rng.randn(self.N_FEATURES) * 0.1
        elif self.agent_type == "novice_hot":
            w[2] = 2.0   # 追热
        return w

    def predict_distribution(self, features, random_dist=None):
        """输出1000维概率分布预测
        
        Args:
            features: (1000, 12) 号码特征矩阵
            random_dist: (1000,) 随机彩民投注分布（可选）
            
        Returns:
            (1000,) 概率分布
        """
        scores = features @ self.weights
        
        # 某些类型需要投注分布信息
        if random_dist is not None:
            if self.agent_type == "gap_hunter":
                scores -= random_dist * 2.0
            elif self.agent_type == "contrarian":
                scores -= random_dist * 1.5
            elif self.agent_type == "pattern_learner":
                scores -= random_dist * 0.5
        
        dist = softmax(scores, temperature=self.temperature)
        self._last_prediction = dist.copy()
        return dist

    def place_bet(self, prediction_dist):
        """根据预测分布下注"""
        if self.bankroll <= 0:
            return {}
        
        # 选择top_k个最可能的号码
        top_indices = np.argsort(prediction_dist)[-self.top_k:]
        
        # 按概率分配投注额（Kelly-like）
        total_bet = min(self.bankroll * 0.05, 100)  # 每轮最多投5%资金
        bets = {}
        for idx in top_indices:
            proportion = prediction_dist[idx] / prediction_dist[top_indices].sum()
            bet_amount = max(1, int(total_bet * proportion))
            if self.bankroll >= bet_amount:
                bets[int(idx)] = bet_amount
                self.bankroll -= bet_amount
                self.total_wagered += bet_amount
        
        self.n_rounds += 1
        return bets

    def settle(self, drawn_number, bets):
        """结算"""
        if drawn_number in bets:
            win = bets[drawn_number] * PAYOUT_ZHIXUAN
            self.bankroll += win
            self.total_won += win
            self.n_wins += 1

    def update_prediction_accuracy(self, drawn_number):
        """更新预测准确度"""
        if self._last_prediction is not None:
            # 预测概率 vs 实际（one-hot）
            pred_prob = self._last_prediction[drawn_number]
            self.accuracy_history.append(pred_prob)
            # 更新适应度：使用对数概率的滑动平均
            if len(self.accuracy_history) > 50:
                recent = self.accuracy_history[-50:]
                self.fitness = np.mean(np.log(np.array(recent) + 1e-10))
            else:
                self.fitness = np.mean(np.log(np.array(self.accuracy_history) + 1e-10))

    def adapt_weights(self, drawn_number, features, lr=0.01):
        """根据结果微调权重（在线学习）"""
        if self._last_prediction is None:
            return
        # 简单的梯度信号：如果开的号预测概率低，增加其特征的权重
        reward = self._last_prediction[drawn_number] - 1.0/N_NUMBERS
        grad = reward * features[drawn_number]
        self.weights += lr * grad
        # 防止权重爆炸
        self.weights = np.clip(self.weights, -5, 5)

    @property
    def roi(self):
        return (self.total_won - self.total_wagered) / max(self.total_wagered, 1)

    @property
    def win_rate(self):
        return self.n_wins / max(self.n_rounds, 1)

    def get_state(self):
        """获取可序列化的状态"""
        return {
            "type": self.agent_type,
            "is_novice": self.is_novice,
            "weights": self.weights.tolist(),
            "temperature": self.temperature,
            "top_k": self.top_k,
            "confidence": self.confidence,
            "fitness": float(self.fitness),
            "roi": float(self.roi),
            "win_rate": float(self.win_rate),
            "bankroll": float(self.bankroll),
        }

    @classmethod
    def from_state(cls, state, seed=42):
        """从状态恢复Agent"""
        agent = cls(state["type"], seed=seed, is_novice=state["is_novice"])
        agent.weights = np.array(state["weights"])
        agent.temperature = state["temperature"]
        agent.top_k = state["top_k"]
        agent.confidence = state["confidence"]
        agent.fitness = state["fitness"]
        return agent


# ======================================================================
# 随机彩民市场（复用但有调整）
# ======================================================================
class RandomMarket:
    """模拟随机彩民的非均匀投注"""

    def __init__(self, seed=42):
        self.rng = np.random.RandomState(seed)
        self._weights = np.array([3.0, -0.5, 2.5, 1.0, 0.2, 2.0, 0.0, 0.1, 0.1, 0.0, 0.0, 0.5])

    def generate_distribution(self, features, total_bet):
        """生成随机彩民投注分布"""
        # 追热追冷混合
        lgbm_scores = features @ self._weights
        lgbm_probs = softmax(lgbm_scores, temperature=0.5)
        lgbm_bets = lgbm_probs * total_bet * 0.55

        hot_scores = features[:, 2] * 2 + features[:, 0]
        hot_probs = softmax(hot_scores, temperature=0.3)
        hot_bets = hot_probs * total_bet * 0.20

        cold_scores = features[:, 1]
        cold_probs = softmax(cold_scores, temperature=0.5)
        cold_bets = cold_probs * total_bet * 0.15

        uniform_bets = np.ones(N_NUMBERS) / N_NUMBERS * total_bet * 0.10

        return lgbm_bets + hot_bets + cold_bets + uniform_bets


# ======================================================================
# 训练环境
# ======================================================================
class TrainingEnvironment:
    """单组训练环境：包含庄家策略+随机市场+Agent种群"""

    def __init__(self, group_id, banker_strategy="min_payout", seed=42, 
                 data_path=None, rounds=TRAIN_ROUNDS):
        self.group_id = group_id
        self.banker_strategy_name = banker_strategy
        self.seed = seed
        self.n_rounds = rounds
        self.rng = np.random.RandomState(seed)

        # 特征提取器
        self.fe = EnhancedFeatureExtractor(history_window=50)
        self._init_history(data_path)

        # 庄家
        self.banker = BankerStrategy(banker_strategy, noise=0.08, seed=seed+1)

        # 随机市场
        self.market = RandomMarket(seed=seed+2)

        # Agent种群
        self.agents = self._create_population()

        # 统计
        self.round_log = []
        self.group_prediction_history = []  # 每轮的群体综合预测

    def _init_history(self, data_path):
        """用真实数据初始化特征"""
        if data_path and os.path.exists(data_path):
            df = pd.read_csv(data_path, dtype={"期号": str})
            n_init = min(150, len(df))
            for _, row in df.tail(n_init).iterrows():
                num = int(row["百位"]) * 100 + int(row["十位"]) * 10 + int(row["个位"])
                self.fe.update(num)

    def _create_population(self):
        """创建多样化的Agent种群"""
        agents = []
        # 70个老手 + 30个新手
        for i in range(70):
            atype = AGENT_TYPES[i % (len(AGENT_TYPES) - 2)]  # 排除novice类型
            agent = MetaAgent(atype, seed=self.seed*1000+i, is_novice=False)
            agents.append(agent)
        for i in range(30):
            atype = self.rng.choice(["novice_random", "novice_hot"])
            agent = MetaAgent(atype, seed=self.seed*1000+70+i, is_novice=True)
            agents.append(agent)
        return agents

    def run_training(self):
        """运行训练循环"""
        print(f"\n{'='*70}")
        print(f"  🎓 组{self.group_id} 开始训练 | 庄家策略: {self.banker_strategy_name}")
        print(f"  Agent数: {len(self.agents)} | 轮数: {self.n_rounds}")
        print(f"{'='*70}")

        start_time = time.time()

        for round_idx in range(self.n_rounds):
            features = self.fe.get_number_features()

            # 1. 随机彩民投注
            total_bet = 1000  # 每轮总投注基数
            random_dist = self.market.generate_distribution(features, total_bet * RANDOM_SHARE)

            # 2. 所有Agent预测+下注
            strategic_dist = np.zeros(N_NUMBERS)
            all_bets = []
            group_pred = np.zeros(N_NUMBERS)

            for agent in self.agents:
                if agent.bankroll <= 0:
                    all_bets.append({})
                    continue
                pred = agent.predict_distribution(features, random_dist)
                group_pred += pred * agent.confidence
                bets = agent.place_bet(pred)
                all_bets.append(bets)
                for num, amt in bets.items():
                    strategic_dist[num] += amt

            # 归一化群体预测
            if group_pred.sum() > 0:
                group_pred /= group_pred.sum()
            self.group_prediction_history.append(group_pred.copy())

            # 3. 总投注
            total_dist = random_dist + strategic_dist

            # 4. 庄家开奖
            drawn = self.banker.draw(total_dist, features)

            # 5. 结算
            for i, agent in enumerate(self.agents):
                if i < len(all_bets) and all_bets[i]:
                    agent.settle(drawn, all_bets[i])

            # 6. 更新预测准确度和在线学习
            for agent in self.agents:
                agent.update_prediction_accuracy(drawn)
                agent.adapt_weights(drawn, features, lr=0.005)

            # 7. 更新特征
            self.fe.update(drawn)

            # 8. 进化（淘汰+繁殖）
            if (round_idx + 1) % EVOLVE_EVERY == 0:
                self._evolve()

            # 9. 日志
            if (round_idx + 1) % 200 == 0:
                alive = sum(1 for a in self.agents if a.bankroll > 0)
                avg_fitness = np.mean([a.fitness for a in self.agents])
                avg_roi = np.mean([a.roi for a in self.agents if a.bankroll > 0])
                elapsed = time.time() - start_time
                print(f"  Round {round_idx+1}/{self.n_rounds} | "
                      f"存活={alive}/{len(self.agents)} | "
                      f"avg_fitness={avg_fitness:.4f} | avg_roi={avg_roi:.4f} | "
                      f"耗时={elapsed:.1f}s")

        elapsed = time.time() - start_time
        print(f"\n  ✅ 组{self.group_id} 训练完成 | 总耗时 {elapsed:.1f}s")
        return self.agents

    def _evolve(self):
        """进化：淘汰低适应度Agent，繁殖高适应度Agent（含多样性保护）"""
        # 按适应度排序
        ranked = sorted(self.agents, key=lambda a: a.fitness, reverse=True)
        n_survive = max(10, int(len(self.agents) * TOP_SURVIVE_RATE))
        survivors = ranked[:n_survive]

        new_agents = list(survivors)

        # === 多样性保护 ===
        # 1. 每种类型至少保留1个最优个体（生态位保护）
        type_best = {}
        for a in ranked:
            if a.agent_type not in type_best:
                type_best[a.agent_type] = a
        for a in type_best.values():
            if a not in new_agents:
                new_agents.append(a)

        # 2. 随机注入3个全新类型的Agent（移民机制）
        for _ in range(3):
            atype = self.rng.choice(AGENT_TYPES)
            newcomer = MetaAgent(atype, seed=self.rng.randint(0, 100000), is_novice=False)
            new_agents.append(newcomer)

        # 3. 填充剩余位置：交叉+变异，但强制保持类型多样性
        type_counts = defaultdict(int)
        for a in new_agents:
            type_counts[a.agent_type] += 1

        while len(new_agents) < AGENT_POP_SIZE:
            # 锦标赛选择
            p1 = self._tournament(survivors)
            p2 = self._tournament(survivors)

            # 交叉权重
            mask = self.rng.random(len(p1.weights)) < 0.5
            child_weights = np.where(mask, p1.weights, p2.weights)

            # 变异
            mut_mask = self.rng.random(len(child_weights)) < MUTATION_RATE
            child_weights[mut_mask] += self.rng.randn(mut_mask.sum()) * MUTATION_STD

            # 类型选择：偏向稀缺类型（多样性平衡）
            scarce_types = [t for t in AGENT_TYPES if type_counts.get(t, 0) < 8]
            if scarce_types and self.rng.random() < 0.4:
                child_type = self.rng.choice(scarce_types)
            else:
                child_type = self.rng.choice([p1.agent_type, p2.agent_type])

            child = MetaAgent(child_type, seed=self.rng.randint(0, 100000), 
                            is_novice=False)
            child.weights = child_weights
            child.temperature = (p1.temperature + p2.temperature) / 2 + self.rng.randn() * 0.05
            child.top_k = int((p1.top_k + p2.top_k) / 2 + self.rng.randint(-3, 4))
            child.top_k = max(5, min(100, child.top_k))
            child.bankroll = 10000.0

            new_agents.append(child)
            type_counts[child_type] += 1

        self.agents = new_agents[:AGENT_POP_SIZE]

        # 统计
        types_count = defaultdict(int)
        for a in self.agents:
            types_count[a.agent_type] += 1
        n_types = len(types_count)
        print(f"    🧬 进化完成 | {n_types}种类型 | 分布: {dict(types_count)}")

    def _tournament(self, pool, k=3):
        candidates = self.rng.choice(pool, size=min(k, len(pool)), replace=False)
        return max(candidates, key=lambda a: a.fitness)

    def save_trained(self, output_dir):
        """保存训练好的Agent状态"""
        os.makedirs(output_dir, exist_ok=True)
        states = [a.get_state() for a in self.agents]
        with open(os.path.join(output_dir, f"group_{self.group_id}_agents.json"), "w") as f:
            json.dump(states, f, ensure_ascii=False, indent=1)
        
        # 保存群体预测历史
        if self.group_prediction_history:
            pred_array = np.array(self.group_prediction_history[-500:])  # 只存最近500轮
            np.save(os.path.join(output_dir, f"group_{self.group_id}_pred_history.npy"), pred_array)
        
        # 保存摘要
        summary = {
            "group_id": self.group_id,
            "banker_strategy": self.banker_strategy_name,
            "n_agents": len(self.agents),
            "n_rounds": self.n_rounds,
            "alive": sum(1 for a in self.agents if a.bankroll > 0),
            "avg_fitness": float(np.mean([a.fitness for a in self.agents])),
            "avg_roi": float(np.mean([a.roi for a in self.agents])),
            "best_fitness": float(max(a.fitness for a in self.agents)),
            "best_roi": float(max(a.roi for a in self.agents)),
            "type_distribution": dict(defaultdict(int, 
                {t: sum(1 for a in self.agents if a.agent_type == t) for t in set(a.agent_type for a in self.agents)})),
        }
        with open(os.path.join(output_dir, f"group_{self.group_id}_summary.json"), "w") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        print(f"  💾 组{self.group_id} 已保存到 {output_dir}/")


# ======================================================================
# 真实数据预测阶段
# ======================================================================
class RealDataPredictor:
    """用训练好的Agent对真实数据进行预测"""

    def __init__(self, data_path, agent_dirs, seed=42):
        self.data_path = data_path
        self.agent_dirs = agent_dirs
        self.rng = np.random.RandomState(seed)

        # 加载真实数据
        df = pd.read_csv(data_path, dtype={"期号": str})
        self.real_data = df
        self.numbers = (df["百位"].astype(int) * 100 + 
                       df["十位"].astype(int) * 10 + 
                       df["个位"].astype(int)).values

        # 加载所有组的Agent
        self.all_groups = []
        for d in agent_dirs:
            group_agents = self._load_group(d)
            if group_agents:
                self.all_groups.append(group_agents)

        print(f"  📊 已加载 {len(self.all_groups)} 组Agent")

    def _load_group(self, agent_dir):
        """加载一组训练好的Agent"""
        json_path = os.path.join(agent_dir, 
            [f for f in os.listdir(agent_dir) if f.endswith("_agents.json")][0])
        with open(json_path) as f:
            states = json.load(f)
        return [MetaAgent.from_state(s, seed=self.rng.randint(0, 100000)) for s in states]

    def predict_and_validate(self, train_end=TRAIN_SPLIT, val_end=VALIDATE_SPLIT):
        """在真实数据上滚动预测并验证
        
        策略：每期用所有Agent预测，用多种聚合方法综合，
        然后与实际开奖号码对比。
        """
        print(f"\n{'='*70}")
        print(f"  🔮 真实数据预测与验证")
        print(f"  训练区间: 第1~{train_end}期（Agent学习历史模式）")
        print(f"  验证区间: 第{train_end+1}~{val_end}期")
        print(f"{'='*70}")

        fe = EnhancedFeatureExtractor(history_window=50)

        # 用前train_end期初始化特征
        for i in range(min(train_end, len(self.numbers))):
            fe.update(self.numbers[i])

        # 滚动预测
        predictions = []  # 每期的综合预测分布
        actuals = []      # 实际开奖号码
        hit_ranks = []    # 实际号码在预测中的排名
        # 按聚合方式分别记录
        agg_rank_details = {"mean": [], "vote": [], "confidence": []}

        val_start = train_end
        val_end = min(val_end, len(self.numbers))
        total_val = val_end - val_start

        # 给Agent初始资金做预测（不影响训练状态）
        for group_agents in self.all_groups:
            for agent in group_agents:
                agent.bankroll = 10000.0
                agent.prediction_history = []

        for i in range(val_start, val_end):
            features = fe.get_number_features()

            # 所有组的Agent预测
            group_predictions = []
            all_agent_preds = []  # 每个agent的预测+置信度
            for gid, group_agents in enumerate(self.all_groups):
                group_pred = np.zeros(N_NUMBERS)
                for agent in group_agents:
                    if agent.bankroll > 0:
                        pred = agent.predict_distribution(features)
                        group_pred += pred * agent.confidence
                        all_agent_preds.append((pred, agent.confidence, agent.fitness))
                if group_pred.sum() > 0:
                    group_pred /= group_pred.sum()
                group_predictions.append(group_pred)

            if not group_predictions:
                final_pred = np.ones(N_NUMBERS) / N_NUMBERS
            else:
                # === 多种聚合策略 ===
                # 1. 等权平均
                mean_pred = np.mean(group_predictions, axis=0)
                mean_pred /= mean_pred.sum()

                # 2. 投票法：每个agent的top-10投票
                vote_pred = np.zeros(N_NUMBERS)
                for pred, conf, fitness in all_agent_preds:
                    top10 = np.argsort(pred)[-10:]
                    weight = conf * max(fitness + 2, 0.1)  # 避免负权重
                    for idx in top10:
                        vote_pred[idx] += weight * pred[idx]
                if vote_pred.sum() > 0:
                    vote_pred /= vote_pred.sum()
                else:
                    vote_pred = np.ones(N_NUMBERS) / N_NUMBERS

                # 3. 置信度加权：按fitness+confidence加权
                conf_pred = np.zeros(N_NUMBERS)
                total_weight = 0
                for pred, conf, fitness in all_agent_preds:
                    w = conf * max(fitness + 2, 0.1)
                    conf_pred += pred * w
                    total_weight += w
                if total_weight > 0:
                    conf_pred /= conf_pred.sum()
                else:
                    conf_pred = np.ones(N_NUMBERS) / N_NUMBERS

                # 最终融合：三种聚合取平均
                final_pred = (mean_pred + vote_pred + conf_pred) / 3
                final_pred /= final_pred.sum()

            actual = self.numbers[i]
            predictions.append(final_pred)
            actuals.append(actual)

            # 计算排名（1=最可能）
            sorted_indices = np.argsort(final_pred)[::-1]
            rank = np.where(sorted_indices == actual)[0][0] + 1
            hit_ranks.append(rank)

            # 各聚合方式排名
            for name, pred in [("mean", mean_pred if group_predictions else np.ones(N_NUMBERS)/N_NUMBERS),
                               ("vote", vote_pred if group_predictions else np.ones(N_NUMBERS)/N_NUMBERS),
                               ("confidence", conf_pred if group_predictions else np.ones(N_NUMBERS)/N_NUMBERS)]:
                s = np.argsort(pred)[::-1]
                r = np.where(s == actual)[0][0] + 1
                agg_rank_details[name].append(r)

            # 更新特征
            fe.update(actual)

            # 在线微调Agent（更小的学习率，避免过拟合最近数据）
            for group_agents in self.all_groups:
                for agent in group_agents:
                    agent.adapt_weights(actual, features, lr=0.001)

            if (i - val_start + 1) % 200 == 0:
                avg_rank = np.mean(hit_ranks[-200:])
                top10_rate = sum(1 for r in hit_ranks[-200:] if r <= 10) / 200
                top50_rate = sum(1 for r in hit_ranks[-200:] if r <= 50) / 200
                # 各聚合方式
                agg_info = " | ".join(
                    f"{name}={np.mean(v[-200:]):.0f}" for name, v in agg_rank_details.items()
                )
                print(f"  期{i+1}/{val_end} | 平均排名={avg_rank:.1f} | "
                      f"Top10={top10_rate:.3f} Top50={top50_rate:.3f} | 聚合: {agg_info}")

        # 汇总统计
        self._validate(predictions, actuals, hit_ranks, agg_rank_details)
        return predictions, actuals, hit_ranks

    def _validate(self, predictions, actuals, hit_ranks, agg_rank_details=None):
        """验证预测效果"""
        print(f"\n{'='*70}")
        print(f"  📈 验证结果")
        print(f"{'='*70}")

        n = len(actuals)
        avg_rank = np.mean(hit_ranks)
        median_rank = np.median(hit_ranks)

        # Top-K命中率
        print(f"\n  === 综合预测 ===")
        for k in [1, 5, 10, 20, 50, 100, 200]:
            rate = sum(1 for r in hit_ranks if r <= k) / n
            expected = k / N_NUMBERS
            lift = rate / expected if expected > 0 else 0
            print(f"  Top-{k:>3d}: 命中率={rate:.4f} (期望={expected:.4f}) | Lift={lift:.2f}x")

        # 各聚合方式对比
        if agg_rank_details:
            print(f"\n  === 聚合方式对比 ===")
            print(f"  {'方式':<14} {'平均排名':>10} {'中位排名':>10} {'Top50率':>10} {'Top100率':>10}")
            print(f"  {'─'*56}")
            for name, ranks in agg_rank_details.items():
                avg = np.mean(ranks)
                med = np.median(ranks)
                top50 = sum(1 for r in ranks if r <= 50) / len(ranks)
                top100 = sum(1 for r in ranks if r <= 100) / len(ranks)
                print(f"  {name:<14} {avg:>10.1f} {med:>10.1f} {top50:>10.4f} {top100:>10.4f}")
            # 综合融合
            top50 = sum(1 for r in hit_ranks if r <= 50) / n
            top100 = sum(1 for r in hit_ranks if r <= 100) / n
            print(f"  {'融合':<14} {avg_rank:>10.1f} {median_rank:>10.1f} {top50:>10.4f} {top100:>10.4f}")

        # 预测概率 vs 实际频率
        pred_probs = np.array([p[a] for p, a in zip(predictions, actuals)])
        uniform_prob = 1.0 / N_NUMBERS
        avg_pred_prob = np.mean(pred_probs)
        print(f"\n  实际号码的平均预测概率: {avg_pred_prob:.6f}")
        print(f"  均匀分布期望概率: {uniform_prob:.6f}")
        print(f"  Lift: {avg_pred_prob / uniform_prob:.2f}x")

        # 高频号码分析
        avg_prediction = np.mean(predictions, axis=0)
        top20_pred = np.argsort(avg_prediction)[-20:]
        top20_actual_freq = np.zeros(20)
        for i, num in enumerate(top20_pred):
            top20_actual_freq[i] = actuals.count(int(num)) / n

        bottom20_pred = np.argsort(avg_prediction)[:20]
        bottom20_actual_freq = np.zeros(20)
        for i, num in enumerate(bottom20_pred):
            bottom20_actual_freq[i] = actuals.count(int(num)) / n

        print(f"\n  预测Top20号码的实际出现率: {np.mean(top20_actual_freq):.4f}")
        print(f"  预测Bottom20号码的实际出现率: {np.mean(bottom20_actual_freq):.4f}")
        print(f"  均匀期望: {20/N_NUMBERS:.4f}")

        # 排列检验：预测分布是否比随机好？
        print(f"\n  === 排列检验 ===")
        random_ranks = []
        n_perm = 50
        for pi in range(n_perm):
            perm_actuals = self.rng.choice(actuals, size=len(actuals), replace=False)
            perm_ranks = []
            for p, a in zip(predictions, perm_actuals):
                s = np.argsort(p)[::-1]
                r = np.where(s == a)[0][0] + 1
                perm_ranks.append(r)
            random_ranks.append(np.mean(perm_ranks))
        
        perm_mean = np.mean(random_ranks)
        perm_std = np.std(random_ranks)
        z_score = (avg_rank - perm_mean) / max(perm_std, 0.01)
        p_value = np.mean([r <= avg_rank for r in random_ranks])

        print(f"  实际平均排名: {avg_rank:.2f}")
        print(f"  随机重排平均排名: {perm_mean:.2f} ± {perm_std:.2f}")
        print(f"  z-score: {z_score:.3f}")
        print(f"  p-value: {p_value:.4f}")

        if p_value < 0.05:
            print(f"  ✅ 预测显著优于随机！(p < 0.05)")
        elif p_value < 0.10:
            print(f"  ⚠️ 预测边缘显著优于随机 (p < 0.10)")
        else:
            print(f"  ❌ 预测未能显著优于随机 (p >= 0.10)")

        # 保存验证结果
        results = {
            "n_periods": n,
            "avg_rank": float(avg_rank),
            "median_rank": float(median_rank),
            "top10_rate": float(sum(1 for r in hit_ranks if r <= 10) / n),
            "top50_rate": float(sum(1 for r in hit_ranks if r <= 50) / n),
            "top100_rate": float(sum(1 for r in hit_ranks if r <= 100) / n),
            "pred_prob_lift": float(avg_pred_prob / uniform_prob),
            "top20_actual_rate": float(np.mean(top20_actual_freq)),
            "bottom20_actual_rate": float(np.mean(bottom20_actual_freq)),
            "permutation_p": float(p_value),
            "z_score": float(z_score),
        }
        if agg_rank_details:
            results["agg_methods"] = {
                name: {"avg_rank": float(np.mean(ranks)), 
                       "top50": float(sum(1 for r in ranks if r <= 50)/len(ranks))}
                for name, ranks in agg_rank_details.items()
            }
        return results


# ======================================================================
# 主函数
# ======================================================================
def main():
    parser = argparse.ArgumentParser(description="元叙事博弈模拟")
    parser.add_argument("--phase", choices=["train", "predict", "validate", "full"], 
                        default="full", help="执行阶段")
    parser.add_argument("--group", type=int, default=-1, 
                        help="训练第N组(0-5)，-1=全部")
    parser.add_argument("--rounds", type=int, default=TRAIN_ROUNDS, 
                        help=f"训练轮数(默认{TRAIN_ROUNDS})")
    parser.add_argument("--data", default="data/pl3_history.csv",
                        help="真实数据路径")
    parser.add_argument("--output", default="meta_simulation_output",
                        help="输出目录")
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    if args.phase in ("train", "full"):
        # 训练阶段
        groups_to_train = range(N_GROUPS) if args.group < 0 else [args.group]
        
        for gid in groups_to_train:
            banker_strat = BANKER_STRATEGIES[gid % len(BANKER_STRATEGIES)]
            env = TrainingEnvironment(
                group_id=gid,
                banker_strategy=banker_strat,
                seed=42 + gid * 100,
                data_path=args.data,
                rounds=args.rounds,
            )
            env.run_training()
            env.save_trained(args.output)

    if args.phase in ("predict", "validate", "full"):
        # 预测+验证阶段
        agent_dirs = [args.output]  # 所有组在同一目录
        predictor = RealDataPredictor(args.data, [args.output])
        predictor.predict_and_validate()


if __name__ == "__main__":
    main()

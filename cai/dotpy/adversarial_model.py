#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
adversarial_model.py - 排列3对抗博弈模拟（K策略 + R策略 vs 庄家）
本地版：数据从 data/ 读取，模型保存到 adversarial model/

核心设计：
  - 随机彩民(50-60%)按LGBM模型/热度/冷度买 → 产生非均匀投注分布
  - K策略彩民(精英型)：少量精心训练的智能体，分析投注缺口
  - R策略彩民(进化型)：大量参数化智能体，达尔文式生灭
  - 庄家：开出总投注最少的号码（最小化赔付）

用法:
  python adversarial_model.py --strategy K          # 只跑K策略
  python adversarial_model.py --strategy R          # 只跑R策略
  python adversarial_model.py --strategy both       # K+R同时跑
  python adversarial_model.py --rounds 1000         # 模拟轮数
  python adversarial_model.py --banker-noise 0.1    # 庄家噪声(偶尔不选最少)
"""

import numpy as np
import pandas as pd
import json
import os
import argparse
import copy
from collections import defaultdict
from abc import ABC, abstractmethod

# ==================== 本地路径配置 ====================
BASE_DIR = r"C:\Users\HUAWEI\Desktop\Adversarial Learning"
DATA_DIR = os.path.join(BASE_DIR, "data")
MODEL_DIR = os.path.join(BASE_DIR, "adversarial model")

# ==================== 全局参数 ====================
PAYOUT_ZHIXUAN = 520       # 直选赔率(1040元/2元注)
N_NUMBERS = 1000            # 000~999
RANDOM_PLAYER_SHARE = 0.55  # 随机彩民占比
K_PLAYER_SHARE = 0.20       # K策略占比
R_PLAYER_SHARE = 0.25       # R策略占比

# 随机彩民内部构成
RANDOM_LGBM_RATIO = 0.60
RANDOM_HOT_RATIO = 0.20
RANDOM_COLD_RATIO = 0.10
RANDOM_UNIFORM_RATIO = 0.10

# R策略进化参数
R_POP_SIZE = 60
R_TOP_SURVIVE = 0.30       # 存活率
R_MUTATION_RATE = 0.15
R_MUTATION_STD = 0.3
R_EVOLVE_EVERY = 20        # 每N轮进化一次

# K策略参数
K_N_AGENTS = 5
K_BET_TOP_N = 5            # 每个K智能体投注前N个缺口

# 庄家参数
BANKER_NOISE_DEFAULT = 0.05  # 庄家有5%概率不选最少号


# ======================================================================
# 工具函数
# ======================================================================
def num_to_digits(n):
    """数字→三位数组 0->[0,0,0], 123->[1,2,3]"""
    return [n // 100, (n // 10) % 10, n % 10]


def digits_to_num(d):
    """三位数组→数字 [1,2,3]->123"""
    return d[0] * 100 + d[1] * 10 + d[2]


def softmax(x, temperature=1.0):
    """带温度的softmax"""
    e = np.exp((x - np.max(x)) / temperature)
    return e / e.sum()


# ======================================================================
# 历史特征提取（为每个号码生成特征向量）
# ======================================================================
class NumberFeatureExtractor:
    """从历史数据中提取每个号码的特征，供智能体使用"""

    def __init__(self, history_window=30):
        self.window = history_window
        self.history = []  # 最近N期开奖号码(0-999)

    def update(self, drawn_number):
        """更新历史记录"""
        self.history.append(drawn_number)
        if len(self.history) > self.window * 3:
            self.history = self.history[-self.window * 3:]

    def get_features(self):
        """为所有1000个号码生成特征矩阵 (1000, n_features)

        特征列:
          0: 频率得分 - 最近window期中该号出现次数/window
          1: 冷度得分 - 距上次出现的期数/window (从未出现=1.0)
          2: 热度得分 - 最近5期出现次数/5
          3: 吉利号得分 - 含8/6/9的数量/3
          4: 和值得分 - (百+十+个)/27
          5: 近期得分 - 最近10期出现次数/10
          6: 偏置项 - 1.0
        """
        features = np.zeros((N_NUMBERS, 7))

        recent = self.history[-self.window:] if len(self.history) >= self.window else self.history
        recent5 = self.history[-5:] if len(self.history) >= 5 else self.history
        recent10 = self.history[-10:] if len(self.history) >= 10 else self.history

        for n in range(N_NUMBERS):
            d = num_to_digits(n)

            # 0: 频率得分
            features[n, 0] = recent.count(n) / max(len(recent), 1)

            # 1: 冷度得分（距上次出现的期数）
            last_seen = -1
            for i in range(len(self.history) - 1, -1, -1):
                if self.history[i] == n:
                    last_seen = len(self.history) - 1 - i
                    break
            features[n, 1] = last_seen / max(self.window, 1) if last_seen >= 0 else 1.0

            # 2: 热度得分
            features[n, 2] = recent5.count(n) / max(len(recent5), 1)

            # 3: 吉利号得分（文化偏好：8>6>9）
            lucky = sum(1 for x in d if x in [8, 6, 9]) / 3
            features[n, 3] = lucky

            # 4: 和值得分
            features[n, 4] = sum(d) / 27

            # 5: 近期得分
            features[n, 5] = recent10.count(n) / max(len(recent10), 1)

            # 6: 偏置项
            features[n, 6] = 1.0

        return features


# ======================================================================
# 随机彩民投注分布模拟
# ======================================================================
class RandomPlayerMarket:
    """模拟随机彩民的投注分布（非均匀）"""

    def __init__(self, feature_extractor, seed=42):
        self.fe = feature_extractor
        self.rng = np.random.RandomState(seed)
        # 预热：模拟一个初始投注偏好
        self._lgbm_weights = np.array([3.0, -0.5, 2.5, 1.0, 0.2, 2.0, 0.5])
        self._lgbm_temp = 0.5

    def generate_distribution(self, total_bet_units):
        """生成随机彩民的投注分布

        返回: np.array shape=(1000,) 表示每个号码上的投注额
        """
        features = self.fe.get_features()

        # 1. LGBM追随者 (60%) - 跟模型预测的热门号
        lgbm_scores = features @ self._lgbm_weights
        lgbm_probs = softmax(lgbm_scores, temperature=self._lgbm_temp)
        lgbm_bets = lgbm_probs * total_bet_units * RANDOM_LGBM_RATIO

        # 2. 追热彩民 (20%) - 跟着最近的号买
        hot_scores = features[:, 2] * 2 + features[:, 0]  # 热度+频率
        hot_probs = softmax(hot_scores, temperature=0.3)
        hot_bets = hot_probs * total_bet_units * RANDOM_HOT_RATIO

        # 3. 追冷彩民 (10%) - 买长时间没出的号
        cold_scores = features[:, 1]  # 冷度
        cold_probs = softmax(cold_scores, temperature=0.5)
        cold_bets = cold_probs * total_bet_units * RANDOM_COLD_RATIO

        # 4. 纯随机 (10%)
        uniform_bets = np.ones(N_NUMBERS) / N_NUMBERS * total_bet_units * RANDOM_UNIFORM_RATIO

        total = lgbm_bets + hot_bets + cold_bets + uniform_bets
        return total


# ======================================================================
# K策略智能体（精英型）
# ======================================================================
class KAgent:
    """K策略彩民：精心设计的少量精英智能体"""

    TYPES = ["freq_gap", "cold_spot", "ml_inverse", "adaptive", "meta"]

    def __init__(self, agent_type, bet_per_round=1, top_n=K_BET_TOP_N):
        self.agent_type = agent_type
        self.bet_per_round = bet_per_round
        self.top_n = top_n
        self.bankroll = 1000.0
        self.total_wagered = 0.0
        self.total_won = 0.0
        self.n_rounds = 0
        self.n_wins = 0

        # adaptive类型的可调权重
        self._adaptive_weights = np.array([1.0, 1.0, 0.5, -0.5, 0.0, 0.8, 0.0])
        self._adaptive_lr = 0.05

        # meta类型记忆：最近庄家开的号码
        self._banker_history = []

    def predict_gaps(self, random_dist, features):
        """预测投注缺口，返回要买的号码列表

        Args:
            random_dist: np.array(1000,) 随机彩民的投注分布
            features: np.array(1000, 7) 号码特征矩阵

        Returns:
            list of int: 要投注的号码
        """
        if self.agent_type == "freq_gap":
            # 频率缺口分析：投注随机彩民最不投的号
            scores = -random_dist  # 负号：投注越少→分数越高
            # 加一点频率偏好（低频号更有可能是缺口）
            scores += features[:, 0] * (-0.5)

        elif self.agent_type == "cold_spot":
            # 冷号追踪：专门买冷号
            scores = features[:, 1]  # 冷度越高越好
            scores -= random_dist * 0.3  # 避免也买热号

        elif self.agent_type == "ml_inverse":
            # ML逆策略：LGBM预测热门的补集
            lgbm_weights = np.array([3.0, -0.5, 2.5, 1.0, 0.2, 2.0, 0.5])
            lgbm_scores = features @ lgbm_weights
            scores = -lgbm_scores  # LGBM越不看好→分数越高
            scores -= random_dist * 0.2

        elif self.agent_type == "adaptive":
            # 自适应策略：根据历史表现调整权重
            scores = features @ self._adaptive_weights
            scores -= random_dist * 0.5

        elif self.agent_type == "meta":
            # 元策略：分析庄家历史选择模式
            if len(self._banker_history) >= 5:
                recent_banker = self._banker_history[-5:]
                banker_features = np.mean([features[b] for b in recent_banker], axis=0)
                similarity = np.array([np.dot(features[i], banker_features) for i in range(N_NUMBERS)])
                scores = -similarity - random_dist * 0.3
            else:
                scores = -random_dist
        else:
            scores = -random_dist

        # 选top_n个分数最高的号码，但加入随机性避免所有K agent买同一组号
        # 给每个agent加不同的噪声扰动
        noise = np.random.RandomState(hash(self.agent_type) % 2**31).randn(N_NUMBERS) * 0.1
        scores += noise

        top_indices = np.argsort(scores)[-self.top_n:]
        return top_indices.tolist()

    def place_bet(self, gap_numbers):
        """下注"""
        if self.bankroll < self.bet_per_round * len(gap_numbers):
            # 资金不足，只买能买的
            can_buy = max(1, int(self.bankroll / self.bet_per_round))
            gap_numbers = gap_numbers[:can_buy]

        cost = self.bet_per_round * len(gap_numbers)
        self.bankroll -= cost
        self.total_wagered += cost
        self.n_rounds += 1

        return {n: self.bet_per_round for n in gap_numbers}

    def receive_payout(self, drawn_number, bets):
        """结算"""
        if drawn_number in bets:
            win = bets[drawn_number] * PAYOUT_ZHIXUAN
            self.bankroll += win
            self.total_won += win
            self.n_wins += 1

    def update_after_round(self, drawn_number, random_dist, features):
        """每轮后更新内部状态"""
        if self.agent_type == "adaptive":
            # 根据结果调整权重
            # 如果这轮庄家开的号恰好在我们的缺口里→正反馈
            # 否则→负反馈
            gap_scores = features @ self._adaptive_weights - random_dist * 0.5
            drawn_score = gap_scores[drawn_number]
            reward = 1.0 if drawn_number in self._get_last_bets() else -0.1
            # 简单的REINFORCE梯度
            grad = reward * features[drawn_number]
            self._adaptive_weights += self._adaptive_lr * grad
            # 防止权重爆炸
            self._adaptive_weights = np.clip(self._adaptive_weights, -5, 5)

        if self.agent_type == "meta":
            self._banker_history.append(drawn_number)

    def _get_last_bets(self):
        """获取最近一轮的投注号码（用于adaptive更新）"""
        return getattr(self, '_last_gap', [])

    @property
    def roi(self):
        if self.total_wagered == 0:
            return 0
        return (self.total_won - self.total_wagered) / self.total_wagered

    @property
    def win_rate(self):
        if self.n_rounds == 0:
            return 0
        return self.n_wins / self.n_rounds


# ======================================================================
# R策略智能体（进化型）
# ======================================================================
class RAgent:
    """R策略彩民：参数化智能体，进化筛选"""

    N_FEATURES = 7  # 对应NumberFeatureExtractor的7维特征

    def __init__(self, weights=None, bet_per_round=1, top_n=3, rng=None):
        self.rng = rng or np.random.RandomState()
        if weights is None:
            self.weights = self.rng.randn(self.N_FEATURES) * 0.5
        else:
            self.weights = np.array(weights, dtype=float)
        self.bet_per_round = bet_per_round
        self.top_n = top_n

        self.bankroll = 1000.0
        self.total_wagered = 0.0
        self.total_won = 0.0
        self.n_rounds = 0
        self.n_wins = 0
        self.fitness_history = []

    def predict_gaps(self, random_dist, features):
        """用参数向量预测缺口"""
        scores = features @ self.weights - random_dist * 0.5
        top_indices = np.argsort(scores)[-self.top_n:]
        return top_indices.tolist()

    def place_bet(self, gap_numbers):
        """下注"""
        if self.bankroll < self.bet_per_round * len(gap_numbers):
            can_buy = max(1, int(self.bankroll / self.bet_per_round))
            gap_numbers = gap_numbers[:can_buy]
        cost = self.bet_per_round * len(gap_numbers)
        self.bankroll -= cost
        self.total_wagered += cost
        self.n_rounds += 1
        return {n: self.bet_per_round for n in gap_numbers}

    def receive_payout(self, drawn_number, bets):
        """结算"""
        if drawn_number in bets:
            win = bets[drawn_number] * PAYOUT_ZHIXUAN
            self.bankroll += win
            self.total_won += win
            self.n_wins += 1

    @property
    def roi(self):
        if self.total_wagered == 0:
            return 0
        return (self.total_won - self.total_wagered) / self.total_wagered

    @property
    def win_rate(self):
        if self.n_rounds == 0:
            return 0
        return self.n_wins / self.n_rounds

    @property
    def fitness(self):
        """适应度 = ROI"""
        if self.n_rounds == 0:
            return 0
        return self.roi


class RPopulation:
    """R策略种群管理：选择、交叉、变异"""

    def __init__(self, pop_size=R_POP_SIZE, seed=42):
        self.pop_size = pop_size
        self.rng = np.random.RandomState(seed)
        self.agents = [RAgent(rng=np.random.RandomState(seed + i))
                       for i in range(pop_size)]
        self.generation = 0

    def evolve(self):
        """执行一轮进化：选择→交叉→变异"""
        self.generation += 1

        # 按适应度排序
        ranked = sorted(self.agents, key=lambda a: a.fitness, reverse=True)

        # 选择：top 30%存活
        n_survive = max(2, int(self.pop_size * R_TOP_SURVIVE))
        survivors = ranked[:n_survive]

        # 生成下一代
        new_agents = list(survivors)  # 存活者保留

        # 交叉+变异填充剩余位置
        while len(new_agents) < self.pop_size:
            # 锦标赛选择父母
            p1 = self._tournament_select(survivors)
            p2 = self._tournament_select(survivors)

            # 交叉
            child_weights = self._crossover(p1.weights, p2.weights)

            # 变异
            child_weights = self._mutate(child_weights)

            child = RAgent(
                weights=child_weights,
                bet_per_round=1,
                top_n=self.rng.randint(2, 6),
                rng=np.random.RandomState(self.rng.randint(0, 100000))
            )
            new_agents.append(child)

        self.agents = new_agents[:self.pop_size]

        # 统计
        best_roi = max(a.roi for a in self.agents)
        avg_roi = np.mean([a.roi for a in self.agents])
        print(f"  🧬 进化Gen{self.generation}: best_roi={best_roi:.4f}, avg_roi={avg_roi:.4f}, "
              f"存活={n_survive}, 新生={self.pop_size - n_survive}")

    def _tournament_select(self, pool, k=3):
        """锦标赛选择"""
        candidates = self.rng.choice(pool, size=min(k, len(pool)), replace=False)
        return max(candidates, key=lambda a: a.fitness)

    def _crossover(self, w1, w2):
        """均匀交叉"""
        mask = self.rng.random(len(w1)) < 0.5
        child = np.where(mask, w1, w2)
        return child

    def _mutate(self, weights):
        """高斯变异"""
        mask = self.rng.random(len(weights)) < R_MUTATION_RATE
        noise = self.rng.randn(len(weights)) * R_MUTATION_STD
        weights = weights.copy()
        weights[mask] += noise[mask]
        return weights

    def all_predict_and_bet(self, random_dist, features):
        """所有R智能体预测缺口并下注"""
        all_bets = {}
        for agent in self.agents:
            gaps = agent.predict_gaps(random_dist, features)
            bets = agent.place_bet(gaps)
            for num, amount in bets.items():
                all_bets[num] = all_bets.get(num, 0) + amount
        return all_bets, self.agents

    def all_settle(self, drawn_number):
        """所有R智能体结算"""
        for agent in self.agents:
            # 需要知道每个agent的bets...简化：只统计bankroll变化
            pass  # 结算在Game中统一处理

    def get_stats(self):
        """获取种群统计"""
        rois = [a.roi for a in self.agents]
        bankrolls = [a.bankroll for a in self.agents]
        win_rates = [a.win_rate for a in self.agents]
        return {
            "best_roi": max(rois),
            "avg_roi": np.mean(rois),
            "worst_roi": min(rois),
            "avg_bankroll": np.mean(bankrolls),
            "avg_win_rate": np.mean(win_rates),
            "alive": sum(1 for b in bankrolls if b > 0),
        }


# ======================================================================
# 庄家智能体
# ======================================================================
class BankerAgent:
    """庄家：选择总投注最少的号码开奖（或随机开奖作为对照）"""

    def __init__(self, noise=BANKER_NOISE_DEFAULT, seed=42, adversarial=True):
        self.noise = noise  # 概率不选最少号
        self.adversarial = adversarial  # True=对抗庄家, False=纯随机开奖
        self.rng = np.random.RandomState(seed)
        self.history = []  # 开奖历史

    def draw(self, total_dist):
        """根据总投注分布开奖

        Args:
            total_dist: np.array(1000,) 每个号码上的总投注额

        Returns:
            int: 开出的号码(0-999)
        """
        if not self.adversarial:
            # 非对抗模式：纯随机开奖（对照组）
            drawn = self.rng.randint(0, N_NUMBERS)
        elif self.rng.random() < self.noise:
            # 噪声模式：随机选一个(模拟真实随机性/监管约束)
            drawn = self.rng.randint(0, N_NUMBERS)
        else:
            # 理性模式：选投注最少的号码
            min_bet = total_dist.min()
            min_indices = np.where(total_dist == min_bet)[0]
            drawn = self.rng.choice(min_indices)

        self.history.append(drawn)
        return drawn


# ======================================================================
# 对抗博弈主环境
# ======================================================================
class AdversarialGame:
    """对抗博弈模拟主循环"""

    def __init__(self, strategy="both", rounds=500, banker_noise=BANKER_NOISE_DEFAULT,
                 total_bet_per_round=1000, seed=42, data_path=None, adversarial=True):
        self.strategy = strategy  # "K", "R", "both"
        self.n_rounds = rounds
        self.total_bet_per_round = total_bet_per_round
        self.rng = np.random.RandomState(seed)
        self.seed = seed
        self.adversarial = adversarial

        # 特征提取器
        self.fe = NumberFeatureExtractor(history_window=30)

        # 用真实历史数据初始化特征
        self._init_history(data_path)

        # 随机彩民市场
        self.market = RandomPlayerMarket(self.fe, seed=seed + 1)

        # 庄家
        self.banker = BankerAgent(noise=banker_noise, seed=seed + 2, adversarial=adversarial)

        # K策略智能体
        self.k_agents = []
        if strategy in ("K", "both"):
            for i, atype in enumerate(KAgent.TYPES):
                agent = KAgent(atype, bet_per_round=1, top_n=K_BET_TOP_N)
                self.k_agents.append(agent)

        # R策略种群
        self.r_pop = None
        if strategy in ("R", "both"):
            self.r_pop = RPopulation(pop_size=R_POP_SIZE, seed=seed + 3)

        # 全局统计
        self.round_log = []

    def _init_history(self, data_path):
        """用真实历史数据初始化特征提取器"""
        if data_path and os.path.exists(data_path):
            df = pd.read_csv(data_path, dtype={"期号": str})
            # 取最近100期作为初始历史
            n_init = min(100, len(df))
            for _, row in df.tail(n_init).iterrows():
                num = int(row["百位"]) * 100 + int(row["十位"]) * 10 + int(row["个位"])
                self.fe.update(num)
            print(f"  📊 已加载{n_init}期历史数据初始化特征")
        else:
            # 无数据时用随机初始历史
            for _ in range(30):
                self.fe.update(self.rng.randint(0, N_NUMBERS))
            print(f"  ⚠️ 无历史数据，用随机初始化")

    def run(self):
        """运行完整模拟"""
        print(f"\n{'='*70}")
        print(f"  🎰 排列3对抗博弈模拟")
        banker_mode = "对抗庄家" if self.adversarial else "纯随机开奖(对照)"
        print(f"  策略: {self.strategy} | 轮数: {self.n_rounds} | 庄家模式: {banker_mode}")
        if self.adversarial:
            print(f"  庄家噪声: {self.banker.noise:.0%}")
        print(f"{'='*70}")

        # 跟踪每轮每个K agent的投注
        k_agent_bets = {i: {} for i in range(len(self.k_agents))}

        for round_idx in range(self.n_rounds):
            features = self.fe.get_features()

            # 1. 随机彩民投注
            random_bet_total = self.total_bet_per_round * RANDOM_PLAYER_SHARE
            random_dist = self.market.generate_distribution(random_bet_total)

            # 2. 策略彩民投注
            k_total_dist = np.zeros(N_NUMBERS)
            r_total_dist = np.zeros(N_NUMBERS)

            # K策略
            if self.k_agents:
                for i, agent in enumerate(self.k_agents):
                    gaps = agent.predict_gaps(random_dist, features)
                    agent._last_gap = gaps  # 供adaptive更新用
                    bets = agent.place_bet(gaps)
                    k_agent_bets[i] = bets
                    for num, amt in bets.items():
                        k_total_dist[num] += amt

            # R策略
            r_agent_bets_list = []  # 每个agent的bets，按当前agents列表顺序
            if self.r_pop:
                for i, agent in enumerate(self.r_pop.agents):
                    gaps = agent.predict_gaps(random_dist, features)
                    bets = agent.place_bet(gaps)
                    r_agent_bets_list.append(bets)
                    for num, amt in bets.items():
                        r_total_dist[num] += amt

            # 3. 总投注分布
            total_dist = random_dist + k_total_dist + r_total_dist

            # 4. 庄家开奖
            drawn = self.banker.draw(total_dist)

            # 5. 结算
            # K智能体
            for i, agent in enumerate(self.k_agents):
                agent.receive_payout(drawn, k_agent_bets[i])
                agent.update_after_round(drawn, random_dist, features)

            # R智能体
            if self.r_pop:
                for i, agent in enumerate(self.r_pop.agents):
                    if i < len(r_agent_bets_list):
                        agent.receive_payout(drawn, r_agent_bets_list[i])

            # 6. 更新特征（庄家开的号作为新的"开奖"）
            self.fe.update(drawn)

            # 7. R策略进化
            if self.r_pop and (round_idx + 1) % R_EVOLVE_EVERY == 0:
                self.r_pop.evolve()

            # 8. 记录日志
            random_payout = random_dist[drawn] * PAYOUT_ZHIXUAN
            k_payout = k_total_dist[drawn] * PAYOUT_ZHIXUAN
            r_payout = r_total_dist[drawn] * PAYOUT_ZHIXUAN
            total_payout = random_payout + k_payout + r_payout
            total_wagered = total_dist.sum()

            log_entry = {
                "round": round_idx + 1,
                "drawn": drawn,
                "total_wagered": total_wagered,
                "total_payout": total_payout,
                "banker_profit": total_wagered - total_payout,
                "random_wagered": random_dist.sum(),
                "random_payout": random_payout,
                "k_wagered": k_total_dist.sum(),
                "k_payout": k_payout,
                "r_wagered": r_total_dist.sum(),
                "r_payout": r_payout,
            }
            self.round_log.append(log_entry)

            # 进度输出
            if (round_idx + 1) % 100 == 0:
                k_roi = np.mean([a.roi for a in self.k_agents]) if self.k_agents else 0
                r_stats = self.r_pop.get_stats() if self.r_pop else {"avg_roi": 0}
                print(f"  Round {round_idx+1}/{self.n_rounds}: "
                      f"庄家利润率={(total_wagered-total_payout)/total_wagered:.2%} | "
                      f"K_ROI={k_roi:.4f} | R_ROI={r_stats['avg_roi']:.4f}")

        print(f"\n{'='*70}")
        print(f"  🏁 模拟完成")
        print(f"{'='*70}")

    def generate_report(self):
        """生成详细报告"""
        if not self.round_log:
            print("⚠️ 无模拟数据")
            return

        df = pd.DataFrame(self.round_log)

        # ---- 全局统计 ----
        total_wagered = df["total_wagered"].sum()
        total_payout = df["total_payout"].sum()
        banker_total_profit = total_wagered - total_payout

        print(f"\n{'─'*70}")
        print(f"  📊 对抗博弈模拟报告")
        print(f"{'─'*70}")
        print(f"  总轮数: {len(df)}")
        print(f"  总投注额: {total_wagered:,.0f}")
        print(f"  总赔付额: {total_payout:,.0f}")
        print(f"  庄家总利润: {banker_total_profit:,.0f} ({banker_total_profit/total_wagered:.2%})")
        print()

        # ---- 各方ROI ----
        print(f"{'参与者':<16} {'总投注':>12} {'总赔付':>12} {'ROI':>10} {'利润率':>10}")
        print(f"{'─'*60}")

        # 随机彩民
        rw = df["random_wagered"].sum()
        rp = df["random_payout"].sum()
        r_roi = (rp - rw) / rw if rw > 0 else 0
        print(f"{'随机彩民':<16} {rw:>12,.0f} {rp:>12,.0f} {r_roi:>10.4f} {r_roi:>10.2%}")

        # K策略
        if self.k_agents:
            kw = df["k_wagered"].sum()
            kp = df["k_payout"].sum()
            k_roi = (kp - kw) / kw if kw > 0 else 0
            print(f"{'K策略(整体)':<16} {kw:>12,.0f} {kp:>12,.0f} {k_roi:>10.4f} {k_roi:>10.2%}")

            # 各K智能体详细
            print(f"\n  K策略各智能体详情:")
            print(f"  {'类型':<14} {'资金':>10} {'投注额':>10} {'中奖次数':>8} {'ROI':>10} {'胜率':>8}")
            print(f"  {'─'*60}")
            for agent in self.k_agents:
                print(f"  {agent.agent_type:<14} {agent.bankroll:>10.1f} "
                      f"{agent.total_wagered:>10.1f} {agent.n_wins:>8d} "
                      f"{agent.roi:>10.4f} {agent.win_rate:>8.4f}")

        # R策略
        if self.r_pop:
            rpw = df["r_wagered"].sum()
            rpp = df["r_payout"].sum()
            rp_roi = (rpp - rpw) / rpw if rpw > 0 else 0
            print(f"\n{'R策略(整体)':<16} {rpw:>12,.0f} {rpp:>12,.0f} {rp_roi:>10.4f} {rp_roi:>10.2%}")

            r_stats = self.r_pop.get_stats()
            print(f"\n  R策略种群统计 (进化{self.r_pop.generation}代):")
            print(f"  {'指标':<16} {'值':>12}")
            print(f"  {'─'*30}")
            for k, v in r_stats.items():
                if isinstance(v, float):
                    print(f"  {k:<16} {v:>12.4f}")
                else:
                    print(f"  {k:<16} {v:>12}")

        # ---- 核心判断 ----
        print(f"\n{'─'*70}")
        print(f"  🔍 核心判断")
        print(f"{'─'*70}")

        # 庄家利润率
        banker_margin = banker_total_profit / total_wagered
        print(f"  庄家利润率: {banker_margin:.2%}")

        if banker_margin > 0.40:
            print(f"  ⚠️ 庄家利润率>40% — 在对抗庄家模式下，彩民整体大幅亏损")
        elif banker_margin > 0.10:
            print(f"  📊 庄家利润率10-40% — 中等剥削，策略彩民略好于随机")
        else:
            print(f"  ✅ 庄家利润率<10% — 策略彩民有效压缩了庄家利润空间")

        # K vs R
        if self.k_agents and self.r_pop:
            k_roi_val = (df["k_payout"].sum() - df["k_wagered"].sum()) / df["k_wagered"].sum()
            r_roi_val = (df["r_payout"].sum() - df["r_wagered"].sum()) / df["r_wagered"].sum()
            if k_roi_val > r_roi_val + 0.05:
                print(f"  📊 K策略ROI({k_roi_val:.4f})显著优于R策略({r_roi_val:.4f})")
            elif r_roi_val > k_roi_val + 0.05:
                print(f"  📊 R策略ROI({r_roi_val:.4f})显著优于K策略({k_roi_val:.4f})")
            else:
                print(f"  📊 K策略({k_roi_val:.4f})与R策略({r_roi_val:.4f})表现接近")

        # 与对照组对比
        print(f"\n  对照组(P1)结论: LGBM单位命中率9.66%≈随机10%，传统ML无效")
        if self.k_agents:
            k_best = max(a.roi for a in self.k_agents)
            if k_best > -0.3:
                print(f"  K策略最佳ROI={k_best:.4f} — 对抗框架优于传统ML直推")
            else:
                print(f"  K策略最佳ROI={k_best:.4f} — 即使对抗框架仍无法盈利")

        return df

    def save_results(self, output_dir):
        """保存模拟结果"""
        os.makedirs(output_dir, exist_ok=True)

        df = pd.DataFrame(self.round_log)
        df.to_csv(os.path.join(output_dir, "adversarial_log.csv"), index=False, encoding="utf-8-sig")

        # 保存摘要
        summary = {
            "strategy": self.strategy,
            "rounds": self.n_rounds,
            "banker_noise": self.banker.noise,
            "total_wagered": float(df["total_wagered"].sum()),
            "total_payout": float(df["total_payout"].sum()),
            "banker_profit_rate": float((df["total_wagered"].sum() - df["total_payout"].sum()) / df["total_wagered"].sum()),
        }

        if self.k_agents:
            summary["k_agents"] = [{
                "type": a.agent_type,
                "roi": float(a.roi),
                "win_rate": float(a.win_rate),
                "bankroll": float(a.bankroll),
            } for a in self.k_agents]

        if self.r_pop:
            summary["r_population"] = self.r_pop.get_stats()
            summary["r_population"]["generations"] = self.r_pop.generation

        with open(os.path.join(output_dir, "adversarial_summary.json"), "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)

        print(f"\n💾 结果已保存到 {output_dir}/")
        print(f"   - adversarial_log.csv (逐轮明细)")
        print(f"   - adversarial_summary.json (摘要)")


# ======================================================================
# 主函数
# ======================================================================
def main():
    parser = argparse.ArgumentParser(description="排列3对抗博弈模拟")
    parser.add_argument("--strategy", choices=["K", "R", "both"], default="both",
                        help="策略选择 (默认both)")
    parser.add_argument("--rounds", type=int, default=500,
                        help="模拟轮数 (默认500)")
    parser.add_argument("--banker-noise", type=float, default=BANKER_NOISE_DEFAULT,
                        help=f"庄家噪声概率 (默认{BANKER_NOISE_DEFAULT})")
    parser.add_argument("--total-bet", type=int, default=1000,
                        help="每轮总投注额 (默认1000)")
    parser.add_argument("--seed", type=int, default=42,
                        help="随机种子 (默认42)")
    parser.add_argument("--data", default=os.path.join(DATA_DIR, "pl3_history.csv"),
                        help="真实历史数据路径")
    parser.add_argument("--output", default=MODEL_DIR,
                        help="输出目录 (默认adversarial model/)")
    parser.add_argument("--no-baseline", action="store_true",
                        help="跳过纯随机开奖对照组实验")
    parser.add_argument("--no-separate", action="store_true",
                        help="跳过K/R单独实验(仅both模式)")
    args = parser.parse_args()

    # ===== 1. 对抗庄家实验 =====
    game = AdversarialGame(
        strategy=args.strategy,
        rounds=args.rounds,
        banker_noise=args.banker_noise,
        total_bet_per_round=args.total_bet,
        seed=args.seed,
        data_path=args.data,
        adversarial=True,
    )
    game.run()
    df = game.generate_report()
    game.save_results(args.output)

    # ===== 2. 纯随机开奖对照组 =====
    if not args.no_baseline:
        print(f"\n\n{'='*70}")
        print(f"  🔄 纯随机开奖对照组（庄家不干预）")
        print(f"{'='*70}")
        game_random = AdversarialGame(
            strategy=args.strategy,
            rounds=args.rounds,
            banker_noise=1.0,  # 100%随机 = 纯随机开奖
            total_bet_per_round=args.total_bet,
            seed=args.seed + 1000,
            data_path=args.data,
            adversarial=False,
        )
        game_random.run()
        game_random.generate_report()
        game_random.save_results(os.path.join(args.output, "random_baseline"))

        # 对比总结
        if len(game.round_log) > 0 and len(game_random.round_log) > 0:
            adv_df = pd.DataFrame(game.round_log)
            rnd_df = pd.DataFrame(game_random.round_log)
            print(f"\n{'='*70}")
            print(f"  📊 对抗 vs 随机 对照总结")
            print(f"{'='*70}")
            adv_margin = (adv_df["total_wagered"].sum() - adv_df["total_payout"].sum()) / adv_df["total_wagered"].sum()
            rnd_margin = (rnd_df["total_wagered"].sum() - rnd_df["total_payout"].sum()) / rnd_df["total_wagered"].sum()
            print(f"  {'模式':<16} {'庄家利润率':>12} {'随机彩民ROI':>14} {'K策略ROI':>12} {'R策略ROI':>12}")
            print(f"  {'─'*66}")

            rnd_random_roi = (rnd_df["random_payout"].sum() - rnd_df["random_wagered"].sum()) / rnd_df["random_wagered"].sum()
            adv_random_roi = (adv_df["random_payout"].sum() - adv_df["random_wagered"].sum()) / adv_df["random_wagered"].sum()

            adv_k_roi = (adv_df["k_payout"].sum() - adv_df["k_wagered"].sum()) / max(adv_df["k_wagered"].sum(), 1)
            rnd_k_roi = (rnd_df["k_payout"].sum() - rnd_df["k_wagered"].sum()) / max(rnd_df["k_wagered"].sum(), 1)

            adv_r_roi = (adv_df["r_payout"].sum() - adv_df["r_wagered"].sum()) / max(adv_df["r_wagered"].sum(), 1)
            rnd_r_roi = (rnd_df["r_payout"].sum() - rnd_df["r_wagered"].sum()) / max(rnd_df["r_wagered"].sum(), 1)

            print(f"  {'对抗庄家':<16} {adv_margin:>12.2%} {adv_random_roi:>14.4f} {adv_k_roi:>12.4f} {adv_r_roi:>12.4f}")
            print(f"  {'纯随机开奖':<16} {rnd_margin:>12.2%} {rnd_random_roi:>14.4f} {rnd_k_roi:>12.4f} {rnd_r_roi:>12.4f}")
            print(f"  {'─'*66}")
            print(f"  💡 对抗庄家使庄家利润率增加: {(adv_margin - rnd_margin)*100:+.1f}个百分点")

    # ===== 3. K/R单独实验 =====
    if args.strategy == "both" and not args.no_separate:
        print(f"\n\n{'='*70}")
        print(f"  🔄 K策略单独实验")
        print(f"{'='*70}")
        game_k = AdversarialGame(
            strategy="K", rounds=args.rounds,
            banker_noise=args.banker_noise,
            total_bet_per_round=args.total_bet,
            seed=args.seed + 100,
            data_path=args.data,
            adversarial=True,
        )
        game_k.run()
        game_k.generate_report()
        game_k.save_results(os.path.join(args.output, "K_only"))

        print(f"\n\n{'='*70}")
        print(f"  🔄 R策略单独实验")
        print(f"{'='*70}")
        game_r = AdversarialGame(
            strategy="R", rounds=args.rounds,
            banker_noise=args.banker_noise,
            total_bet_per_round=args.total_bet,
            seed=args.seed + 200,
            data_path=args.data,
            adversarial=True,
        )
        game_r.run()
        game_r.generate_report()
        game_r.save_results(os.path.join(args.output, "R_only"))


if __name__ == "__main__":
    main()

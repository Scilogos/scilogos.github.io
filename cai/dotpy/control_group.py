#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
control_group.py - 排列3对照组预测模型
LightGBM vs 随机基线, 真实数据 vs 模拟数据对比

用法:
  python control_group.py                     # 完整对比
  python control_group.py --real-only         # 只跑真实数据
  python control_group.py --sim-only          # 只跑模拟数据
  python control_group.py --predict           # 预测下期号码
  python control_group.py --window 20         # 调整特征窗口
"""

import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, log_loss
import json
import os
import argparse
import warnings
warnings.filterwarnings("ignore")

# ==================== 配置 ====================
WINDOW = 10  # 特征窗口大小
LGB_PARAMS = {
    "n_estimators": 300,
    "learning_rate": 0.05,
    "num_leaves": 31,
    "max_depth": 6,
    "min_child_samples": 20,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "reg_alpha": 0.1,
    "reg_lambda": 0.1,
    "verbose": -1,
    "n_jobs": -1,
}


# ==================== 数据加载 ====================
def load_data(filepath):
    """加载CSV,自动映射中英文列名"""
    df = pd.read_csv(filepath, dtype={"期号": str})
    col_map = {"百位": "bai", "十位": "shi", "个位": "ge",
               "期号": "issue", "日期": "date", "销售额": "sales"}
    df.rename(columns={k: v for k, v in col_map.items() if k in df.columns}, inplace=True)
    df["bai"] = df["bai"].astype(int)
    df["shi"] = df["shi"].astype(int)
    df["ge"] = df["ge"].astype(int)
    return df


# ==================== 特征工程 ====================
def build_features(df, window=WINDOW):
    """从序列构建特征矩阵和标签"""
    features, labels_b, labels_s, labels_g = [], [], [], []

    for i in range(window, len(df)):
        feat = []
        w_df = df.iloc[i - window:i]

        # 1. 前N期各位原始值 (3*window维)
        for w in range(window):
            feat.extend([df.iloc[i - window + w]["bai"],
                         df.iloc[i - window + w]["shi"],
                         df.iloc[i - window + w]["ge"]])

        # 2. 各位频率 (30维)
        for col in ["bai", "shi", "ge"]:
            vc = w_df[col].value_counts()
            for d in range(10):
                feat.append(vc.get(d, 0))

        # 3. 各位统计 (12维)
        for col in ["bai", "shi", "ge"]:
            feat.extend([w_df[col].mean(), w_df[col].std(),
                         w_df[col].min(), w_df[col].max()])

        # 4. 和值/跨度特征 (4维)
        sums = w_df["bai"] + w_df["shi"] + w_df["ge"]
        spans = w_df[["bai", "shi", "ge"]].max(axis=1) - w_df[["bai", "shi", "ge"]].min(axis=1)
        feat.extend([sums.mean(), sums.std(), spans.mean(), spans.std()])

        # 5. 奇偶/大小比 (4维)
        for col in ["bai", "shi", "ge"]:
            odds = (w_df[col] % 2 == 1).sum() / window
            bigs = (w_df[col] >= 5).sum() / window
        feat.extend([odds, bigs])

        features.append(feat)
        labels_b.append(df.iloc[i]["bai"])
        labels_s.append(df.iloc[i]["shi"])
        labels_g.append(df.iloc[i]["ge"])

    return (pd.DataFrame(features),
            pd.Series(labels_b, name="bai"),
            pd.Series(labels_s, name="shi"),
            pd.Series(labels_g, name="ge"))


# ==================== 模型训练与评估 ====================
def train_and_evaluate(X_train, yb_t, ys_t, yg_t, X_test, yb_e, ys_e, yg_e):
    """训练LGBM+RF, 返回评估结果"""
    results = {}

    for model_name, make_model in [
        ("LGBM", lambda: lgb.LGBMClassifier(**LGB_PARAMS)),
        ("RF", lambda: RandomForestClassifier(n_estimators=200, max_depth=8, n_jobs=-1, random_state=42)),
    ]:
        accs, loglosses = [], []
        top10_hits, top50_hits, top100_hits = 0, 0, 0
        zx_hits, group_hits = 0, 0

        models = []
        probs = []

        for y_train, y_test in [(yb_t, yb_e), (ys_t, ys_e), (yg_t, yg_e)]:
            m = make_model()
            m.fit(X_train, y_train)
            models.append(m)
            prob = m.predict_proba(X_test)
            probs.append(prob)
            pred = np.argmax(prob, axis=1)
            accs.append(accuracy_score(y_test, pred))
            try:
                loglosses.append(log_loss(y_test, prob))
            except Exception:
                loglosses.append(2.3)  # ln(10)

        # 直选命中率
        pred_b = np.argmax(probs[0], axis=1)
        pred_s = np.argmax(probs[1], axis=1)
        pred_g = np.argmax(probs[2], axis=1)

        for i in range(len(yb_e)):
            if pred_b[i] == yb_e.iloc[i] and pred_s[i] == ys_e.iloc[i] and pred_g[i] == yg_e.iloc[i]:
                zx_hits += 1
            if set([pred_b[i], pred_s[i], pred_g[i]]) == set([yb_e.iloc[i], ys_e.iloc[i], yg_e.iloc[i]]):
                group_hits += 1

        # Top-K命中率 (联合概率)
        for i in range(min(len(yb_e), 200)):  # 采样200个避免太慢
            joint = np.zeros(1000)
            for b in range(10):
                for s in range(10):
                    for g in range(10):
                        joint[b * 100 + s * 10 + g] = probs[0][i, b] * probs[1][i, s] * probs[2][i, g]
            true_idx = yb_e.iloc[i] * 100 + ys_e.iloc[i] * 10 + yg_e.iloc[i]
            sorted_idx = np.argsort(joint)[::-1]
            rank = np.where(sorted_idx == true_idx)[0][0] + 1
            if rank <= 10:
                top10_hits += 1
            if rank <= 50:
                top50_hits += 1
            if rank <= 100:
                top100_hits += 1

        sample_n = min(len(yb_e), 200)
        results[model_name] = {
            "单位命中率": np.mean(accs),
            "百位命中": accs[0],
            "十位命中": accs[1],
            "个位命中": accs[2],
            "直选命中率": zx_hits / len(yb_e),
            "组选命中率": group_hits / len(yb_e),
            "Top10命中率": top10_hits / sample_n,
            "Top50命中率": top50_hits / sample_n,
            "Top100命中率": top100_hits / sample_n,
            "平均LogLoss": np.mean(loglosses),
        }

    return results, models


# ==================== 分割策略 ====================
def make_splits(df, train_months=3, test_days=7):
    """生成非等间隔训练/测试分割"""
    splits = []
    start = pd.Timestamp(df["date"].min())
    end = pd.Timestamp(df["date"].max())

    while start < end:
        # 训练期: 2-5个月随机
        t_months = np.random.randint(2, 6)
        train_end = start + pd.DateOffset(months=t_months)
        if train_end > end:
            break

        # 测试期: 7天
        test_start = train_end + pd.Timedelta(days=1)
        test_end = test_start + pd.Timedelta(days=test_days - 1)

        if test_start > end:
            break

        splits.append({
            "train_start": start.strftime("%Y-%m-%d"),
            "train_end": train_end.strftime("%Y-%m-%d"),
            "test_start": test_start.strftime("%Y-%m-%d"),
            "test_end": min(test_end, end).strftime("%Y-%m-%d"),
        })

        start = test_end + pd.Timedelta(days=1)

    return splits


# ==================== 运行实验 ====================
def run_experiment(data_path, label, window=WINDOW):
    """对单份数据运行完整实验"""
    print(f"\n{'='*60}")
    print(f"  {label}: {data_path}")
    print(f"{'='*60}")

    df = load_data(data_path)
    print(f"数据量: {len(df)} 期 ({df['date'].min()} ~ {df['date'].max()})")

    X, yb, ys, yg = build_features(df, window)
    print(f"特征: {X.shape[1]}维, 样本: {len(X)}")

    # 按时间分割: 80%训练 + 20%测试
    split_idx = int(len(X) * 0.8)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    yb_t, yb_e = yb.iloc[:split_idx], yb.iloc[split_idx:]
    ys_t, ys_e = ys.iloc[:split_idx], ys.iloc[split_idx:]
    yg_t, yg_e = yg.iloc[:split_idx], yg.iloc[split_idx:]

    print(f"训练集: {len(X_train)}, 测试集: {len(X_test)}")
    print("训练中...")

    results, models = train_and_evaluate(
        X_train, yb_t, ys_t, yg_t, X_test, yb_e, ys_e, yg_e
    )

    # 打印结果
    print(f"\n{'─'*60}")
    print(f"{'指标':<16} {'LGBM':>10} {'RF':>10} {'随机基线':>10}")
    print(f"{'─'*60}")

    baselines = {
        "单位命中率": 0.10,
        "百位命中": 0.10,
        "十位命中": 0.10,
        "个位命中": 0.10,
        "直选命中率": 0.001,
        "组选命中率": 0.006,
        "Top10命中率": 0.010,
        "Top50命中率": 0.050,
        "Top100命中率": 0.100,
        "平均LogLoss": 2.303,  # ln(10)
    }

    for metric in ["单位命中率", "直选命中率", "组选命中率",
                    "Top10命中率", "Top50命中率", "Top100命中率", "平均LogLoss"]:
        lgb_v = results["LGBM"][metric]
        rf_v = results["RF"][metric]
        base = baselines[metric]
        print(f"{metric:<16} {lgb_v:>10.4f} {rf_v:>10.4f} {base:>10.4f}")

    print(f"{'─'*60}")

    return results, models, df


# ==================== 预测下期 ====================
def predict_next(models, df, window=WINDOW):
    """用最近数据预测下一期"""
    X, _, _, _ = build_features(df.tail(window + 1), window)
    if len(X) == 0:
        print("数据不足，无法预测")
        return

    x = X.iloc[[-1]]
    probs = [m.predict_proba(x)[0] for m in models]
    names = ["百位", "十位", "个位"]

    print("\n🎯 下期预测:")
    for pos, prob, name in zip(range(3), probs, names):
        top3 = np.argsort(prob)[::-1][:3]
        print(f"  {name}: ", end="")
        for rank, d in enumerate(top3):
            print(f"{d}({prob[d]*100:.1f}%)", end="  ")
        print()

    # 联合Top-10
    joint = {}
    for b in range(10):
        for s in range(10):
            for g in range(10):
                joint[f"{b}{s}{g}"] = probs[0][b] * probs[1][s] * probs[2][g]
    top10 = sorted(joint.items(), key=lambda x: -x[1])[:10]
    print(f"\n  Top-10号码:")
    for i, (num, p) in enumerate(top10):
        print(f"    {i+1}. {num} ({p*100:.3f}%)")


# ==================== 主函数 ====================
def main():
    parser = argparse.ArgumentParser(description="排列3对照组预测模型")
    parser.add_argument("--real-only", action="store_true", help="只跑真实数据")
    parser.add_argument("--sim-only", action="store_true", help="只跑模拟数据")
    parser.add_argument("--predict", action="store_true", help="预测下期号码")
    parser.add_argument("--window", type=int, default=WINDOW, help="特征窗口大小")
    parser.add_argument("--real-data", default="pl3_history.csv")
    parser.add_argument("--sim-data", default="pl3_simulated.csv")
    args = parser.parse_args()

    if args.predict:
        df = load_data(args.real_data)
        X, yb, ys, yg = build_features(df, args.window)
        split_idx = int(len(X) * 0.8)
        models, _, _ = train_and_evaluate(
            X.iloc[:split_idx], yb.iloc[:split_idx], ys.iloc[:split_idx], yg.iloc[:split_idx],
            X.iloc[split_idx:], yb.iloc[split_idx:], ys.iloc[split_idx:], yg.iloc[split_idx:],
        )[1], None, None
        predict_next(models, df, args.window)
        return

    results_all = {}

    if not args.sim_only:
        r, m, df = run_experiment(args.real_data, "真实数据(排列3)", args.window)
        results_all["真实数据"] = r

    if not args.real_only:
        if os.path.exists(args.sim_data):
            r, m, df = run_experiment(args.sim_data, "模拟数据(纯随机)", args.window)
            results_all["模拟数据"] = r
        else:
            print(f"\n⚠️ 模拟数据文件不存在: {args.sim_data}")

    # 对比总结
    if len(results_all) == 2:
        print(f"\n{'='*60}")
        print(f"  📊 对比总结: 真实数据 vs 模拟数据")
        print(f"{'='*60}")
        print(f"{'指标':<16} {'真实LGBM':>10} {'模拟LGBM':>10} {'提升':>10}")
        print(f"{'─'*60}")
        for metric in ["单位命中率", "直选命中率", "组选命中率",
                        "Top10命中率", "Top100命中率"]:
            rv = results_all["真实数据"]["LGBM"][metric]
            sv = results_all["模拟数据"]["LGBM"][metric]
            diff = rv - sv
            print(f"{metric:<16} {rv:>10.4f} {sv:>10.4f} {diff:>+10.4f}")
        print(f"{'─'*60}")

        # 核心判断
        real_acc = results_all["真实数据"]["LGBM"]["单位命中率"]
        sim_acc = results_all["模拟数据"]["LGBM"]["单位命中率"]
        if real_acc > sim_acc + 0.02:
            print("⚠️ 真实数据命中率显著高于模拟数据 → 可能存在可利用的规律")
        elif real_acc < sim_acc + 0.02:
            print("✅ 真实数据命中率与模拟数据接近 → 模型未能发现超越随机的规律")
        else:
            print("📊 差异不显著 → 需要更多实验验证")

    print("\n🏁 实验完成")


if __name__ == "__main__":
    main()

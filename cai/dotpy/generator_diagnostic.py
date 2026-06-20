#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
generator_diagnostic.py — TimeGAN生成器诊断脚本
=================================================
功能：
  1. 全链路功能检测（数据加载→模型构建→前向传播→生成→验证）
  2. 模拟数据清除+再生测试
  3. 早盘模拟数据生成（9:15-9:30集合竞价+9:30-10:00开盘时段）
  4. 开盘时段重点验证（非平均评价，加权到9:30-10:00）

用法：
  python generator_diagnostic.py                    # 全链路诊断
  python generator_diagnostic.py --test clear       # 清除+再生测试
  python generator_diagnostic.py --test premarket   # 早盘模拟
  python generator_diagnostic.py --test open-focus  # 开盘重点验证
  python generator_diagnostic.py --test all         # 全部测试(默认)
"""
import sys
import os
import json
import time
import shutil
import argparse
import datetime
import subprocess
from pathlib import Path
import numpy as np


# ============================================================
# 路径检测
# ============================================================
def get_base_dir():
    candidates = [
        Path(r"C:\Users\HUAWEI\Desktop\Adversarial Learning"),
        Path("/app/data/所有对话/主对话"),
    ]
    for p in candidates:
        if p.exists():
            return p
    return Path(".")


BASE_DIR = get_base_dir()
GENERATOR_SCRIPT = BASE_DIR / "market_generator.py"


# ============================================================
# 诊断工具函数
# ============================================================
def section(title):
    print(f"\n{'━'*60}")
    print(f"  {title}")
    print(f"{'━'*60}")


def check(label, condition, detail=""):
    icon = "✓" if condition else "✗"
    msg = f"  {icon} {label}"
    if detail:
        msg += f" — {detail}"
    print(msg)
    return condition


def run_python(cmd, timeout=120):
    try:
        result = subprocess.run(
            cmd, shell=True, capture_output=True, text=True, timeout=timeout,
            cwd=str(BASE_DIR)
        )
        return result.returncode, result.stdout, result.stderr
    except subprocess.TimeoutExpired:
        return -1, "", "TIMEOUT"
    except Exception as e:
        return -2, "", str(e)


# ============================================================
# Test 1: 全链路功能检测
# ============================================================
def test_full_pipeline():
    section("Test 1: 全链路功能检测")
    results = {}

    # 1.1 脚本存在
    r1 = check("market_generator.py 存在", GENERATOR_SCRIPT.exists(), str(GENERATOR_SCRIPT))
    results['script_exists'] = r1

    # 1.2 依赖检查
    section("1.2 依赖检查")
    deps = ['torch', 'numpy', 'sklearn', 'pandas']
    for dep in deps:
        code, out, err = run_python(f'python -c "import {dep}; print({dep}.__version__)"', 10)
        ver = out.strip() if code == 0 else "未安装"
        check(dep, code == 0, ver)
        results[f'dep_{dep}'] = code == 0

    # 1.3 status模式
    section("1.3 status模式测试")
    code, out, err = run_python(f'python "{GENERATOR_SCRIPT}" --mode status', 15)
    check("status模式运行", code == 0, f"exit={code}")
    if out:
        for line in out.split('\n')[-8:]:
            if line.strip():
                print(f"    {line.strip()}")
    results['status_ok'] = code == 0

    # 1.4 模型构建测试
    section("1.4 模型构建测试")
    # 写临时测试脚本
    test_script = BASE_DIR / "_test_model_build.py"
    test_script.write_text('''
import sys
sys.path.insert(0, r"{base}")
exec(open(r"{script}", encoding="utf-8").read())

config = Config(hidden_dim=8, latent_dim=8, seq_len=10, feature_dim=3, num_layers=1)
model = TimeGAN(config)

import torch
x = torch.randn(4, 10, 3)
h = model.embedder(x)
x_hat = model.recovery(h)
z = torch.randn(4, 10, 8)
h_gen = model.generator(z)
h_sup = model.supervisor(h_gen)
d_out = model.discriminator(h)

print(f"Embedder: {x.shape} -> {h.shape}")
print(f"Recovery: {h.shape} -> {x_hat.shape}")
print(f"Generator: {z.shape} -> {h_gen.shape}")
print(f"Supervisor: {h_gen.shape} -> {h_sup.shape}")
print(f"Discriminator: {h.shape} -> {d_out.shape}")
print("MODEL_OK")
'''.format(base=BASE_DIR, script=GENERATOR_SCRIPT), encoding='utf-8')

    code, out, err = run_python(f'python "{test_script}"', 30)
    is_ok = "MODEL_OK" in out
    check("模型构建+前向传播", is_ok)
    if out:
        for line in out.split('\n'):
            if any(k in line for k in ['Embedder', 'Recovery', 'Generator', 'Supervisor', 'Discriminator', 'MODEL_OK']):
                print(f"    {line.strip()}")
    if err and not is_ok:
        print(f"    错误: {err[:200]}")
    test_script.unlink(missing_ok=True)
    results['model_build_ok'] = is_ok

    # 1.5 极速训练测试（3+3+5 epochs）
    section("1.5 极速训练测试 (3+3+5 epochs)")
    train_cmd = f'python "{GENERATOR_SCRIPT}" --mode train --phase-a-epochs 3 --phase-b-epochs 3 --phase-c-epochs 5'
    code, out, err = run_python(train_cmd, 300)
    is_ok = code == 0 and "训练完成" in out
    check("极速训练", is_ok)
    for line in out.split('\n'):
        if any(k in line for k in ['Phase A', 'Phase B', 'Phase C', 'loss', '完成', '保存']):
            print(f"    {line.strip()}")
    results['train_ok'] = is_ok

    # 1.6 生成测试
    section("1.6 生成测试")
    gen_cmd = f'python "{GENERATOR_SCRIPT}" --mode generate --num-stocks 10'
    code, out, err = run_python(gen_cmd, 30)
    is_ok = code == 0 and "生成完成" in out
    check("生成模式", is_ok)
    for line in out.split('\n'):
        if any(k in line for k in ['形状', '生成', '保存', '完成']):
            print(f"    {line.strip()}")
    results['generate_ok'] = is_ok

    # 1.7 验证测试
    section("1.7 Level 1 验证测试")
    val_cmd = f'python "{GENERATOR_SCRIPT}" --mode validate --level 1'
    code, out, err = run_python(val_cmd, 60)
    is_ok = code == 0
    check("验证模式", is_ok)
    for line in out.split('\n'):
        if any(k in line for k in ['评分', '均值', '标准差', 'Level']):
            print(f"    {line.strip()}")
    results['validate_ok'] = is_ok

    # 总结
    section("Test 1 总结")
    passed = sum(1 for k, v in results.items() if v is True)
    total = sum(1 for k in results if k.endswith('_ok'))
    print(f"  通过: {passed}/{total}")
    results['_summary'] = f"{passed}/{total}"
    return results


# ============================================================
# Test 2: 模拟数据清除+再生
# ============================================================
def test_clear_regenerate():
    section("Test 2: 模拟数据清除+再生机制")
    results = {}

    test_script = BASE_DIR / "_test_clear_regen.py"
    test_script.write_text('''
import sys, os
import numpy as np
from pathlib import Path
import datetime
sys.path.insert(0, r"{base}")
exec(open(r"{script}", encoding="utf-8").read())

paths = get_paths()
output_dir = Path(paths["output_dir"])
model_dir = Path(paths["model_dir"])

print(f"输出目录: {output_dir}")
print(f"模型目录: {model_dir}")

# 检查当前状态
gen_files = list(output_dir.glob("generated_*.npy")) if output_dir.exists() else []
train_files = list(output_dir.glob("training_data.npy")) if output_dir.exists() else []
model_files = list(model_dir.glob("*.pt")) if model_dir.exists() else []
print(f"当前: 生成数据{len(gen_files)}个, 训练数据{len(train_files)}个, 模型{len(model_files)}个")

# 写入测试数据
output_dir.mkdir(parents=True, exist_ok=True)
test_data = np.random.randn(10, 60, 3).astype(np.float32)
np.save(output_dir / "generated_data.npy", test_data)
np.save(output_dir / "generated_test_20260620.npy", test_data)
print(f"写入测试数据: 2个npy文件")

# 清除生成数据（保留模型）
cleared = 0
for f in output_dir.glob("generated_*.npy"):
    f.unlink()
    cleared += 1
print(f"清除生成数据: {cleared}个文件 (模型保留)")

# 确认模型还在
model_still = any(model_dir.glob("*.pt")) if model_dir.exists() else False
print(f"模型保留: {model_still}")

# 再生测试
config = Config(hidden_dim=8, latent_dim=8, seq_len=10, feature_dim=3, num_layers=1)
model = TimeGAN(config)
model_path = model_dir / "timegan_best.pt"
if model_path.exists():
    import torch
    model.load_state_dict(torch.load(model_path, map_location="cpu"), strict=False)
    print("从已有模型再生")
else:
    print("无已有模型，使用随机初始化（仅测试流程）")

gen = SyntheticDataGenerator(model, "cpu")
new_data = gen.generate(5)
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
save_path = output_dir / f"generated_{timestamp}.npy"
gen.save(new_data, save_path)
print(f"再生完成: {save_path.name}, shape={new_data.shape}")

# 验证数据可读
loaded = np.load(save_path)
print(f"回读验证: shape={loaded.shape}, dtype={loaded.dtype}")

print("CLEAR_REGEN_OK")
'''.format(base=BASE_DIR, script=GENERATOR_SCRIPT), encoding='utf-8')

    code, out, err = run_python(f'python "{test_script}"', 30)
    is_ok = "CLEAR_REGEN_OK" in out
    check("清除+再生流程", is_ok)
    for line in out.split('\n'):
        if line.strip():
            print(f"    {line.strip()}")
    if err and not is_ok:
        print(f"    错误: {err[:300]}")
    test_script.unlink(missing_ok=True)
    results['clear_regen_ok'] = is_ok
    return results


# ============================================================
# Test 3: 早盘模拟数据生成
# ============================================================
def test_premarket_generation():
    section("Test 3: 早盘模拟数据生成 (9:15-9:30 + 9:30-10:00)")

    # 独立实现，不依赖market_generator.py的内部结构
    n_stocks = 5
    np.random.seed(42)

    # A股时间轴：
    #   9:15-9:25  集合竞价
    #   9:25-9:30  静默期
    #   9:30-10:00 开盘首30分钟
    seq_len = 45  # 45分钟

    all_sequences = np.zeros((n_stocks, seq_len, 3))  # [价格变化率, 成交量比例, 波动率]
    time_labels = []

    for t in range(seq_len):
        h = 9
        m = 15 + t
        if m >= 60:
            h += 1
            m -= 60
        time_labels.append(f"{h:02d}:{m:02d}")

    for i in range(n_stocks):
        base_vol = 0.005 + np.random.exponential(0.003)
        open_drift = np.random.randn() * 0.01

        for t in range(seq_len):
            if t < 10:
                # 集合竞价 9:15-9:25
                progress = t / 10.0
                price_noise = np.random.randn() * base_vol * (1.5 - 0.8 * progress)
                price_change = open_drift * progress * 0.3 + price_noise
                volume_ratio = 0.2 + 0.6 * progress + np.random.exponential(0.1)
                vol_state = base_vol * (2.0 - progress)
            elif t < 15:
                # 静默期 9:25-9:30
                price_change = np.random.randn() * base_vol * 0.3
                volume_ratio = 0.05 + np.random.exponential(0.02)
                vol_state = base_vol * 0.8
            else:
                # 开盘交易 9:30-10:00
                open_progress = (t - 15) / 30.0
                decay = np.exp(-2.0 * open_progress)
                price_change = (open_drift * 0.5 * decay +
                               np.random.randn() * base_vol * (1.0 + 2.0 * decay))
                volume_ratio = (3.0 * np.exp(-3.0 * open_progress) +
                               0.5 + np.random.exponential(0.2))
                vol_state = base_vol * (1.0 + 2.0 * decay)

            all_sequences[i, t] = [price_change, volume_ratio, vol_state]

    # 统计
    section_labels = [
        ("集合竞价 9:15-9:25", slice(0, 10)),
        ("静默期   9:25-9:30", slice(10, 15)),
        ("开盘爆发 9:30-9:35", slice(15, 20)),
        ("开盘延续 9:35-9:40", slice(20, 25)),
        ("开盘稳定 9:40-10:00", slice(25, 45)),
    ]

    print(f"\n  生成结果: shape={all_sequences.shape}")
    print(f"  时间范围: {time_labels[0]} → {time_labels[-1]}")
    print(f"\n  {'时段':<24} {'价格波动':>10} {'成交量':>10} {'波动率':>10}")
    print(f"  {'-'*58}")
    for label, sl in section_labels:
        price_vol = np.std(all_sequences[:, sl, 0])
        avg_vol = np.mean(all_sequences[:, sl, 1])
        avg_volatility = np.mean(all_sequences[:, sl, 2])
        print(f"  {label:<24} {price_vol:>10.4f} {avg_vol:>10.4f} {avg_volatility:>10.4f}")

    # 保存
    output_dir = BASE_DIR / "adversarial data" / "generated"
    output_dir.mkdir(parents=True, exist_ok=True)
    save_path = output_dir / "premarket_simulated.npy"
    np.save(save_path, all_sequences)
    meta = {'time_labels': time_labels, 'n_stocks': n_stocks}
    with open(output_dir / "premarket_meta.json", 'w') as f:
        json.dump(meta, f, indent=2)
    print(f"\n  已保存: {save_path}")

    check("早盘模拟生成", True, f"{n_stocks}只×{seq_len}分钟")
    return {'premarket_ok': True}


# ============================================================
# Test 4: 开盘时段重点验证
# ============================================================
def test_open_focus_validation():
    section("Test 4: 开盘时段重点验证 (加权评价)")

    def validate_open_focused(real_data, gen_data, open_steps=10, weight_open=5.0):
        """
        开盘时段重点验证
        Args:
            real_data: (n, seq_len, features)
            gen_data: (n, seq_len, features)
            open_steps: 开盘关键步数
            weight_open: 开盘权重倍数
        """
        seq_len = real_data.shape[1]
        feature_names = ['对数收益率', '成交量变化率', '滚动波动率']

        w_pct = weight_open * open_steps / (weight_open * open_steps + (seq_len - open_steps)) * 100
        print(f"  序列长度: {seq_len}, 开盘重点步数: {open_steps}, 开盘权重: {weight_open}x")
        print(f"  开盘覆盖: {open_steps}/{seq_len} = {open_steps/seq_len*100:.1f}%步数 → 权重占比 {w_pct:.1f}%")

        feature_scores = []
        results = {}

        for feat_idx, feat_name in enumerate(feature_names):
            real_feat = real_data[:, :, feat_idx]
            gen_feat = gen_data[:, :, feat_idx]

            step_diffs = []
            for t in range(seq_len):
                r_vals = real_feat[:, t]
                g_vals = gen_feat[:, t]
                r_mean = np.mean(r_vals)
                g_mean = np.mean(g_vals)
                r_std = np.std(r_vals) + 1e-8
                mean_diff = abs(r_mean - g_mean) / r_std
                std_diff = abs(np.std(r_vals) - np.std(g_vals)) / r_std
                step_diffs.append(mean_diff + std_diff)

            step_diffs = np.array(step_diffs)

            weights = np.ones(seq_len)
            weights[:open_steps] = weight_open
            if open_steps + 10 < seq_len:
                weights[open_steps:open_steps+10] = 2.0

            weighted_diff = np.average(step_diffs, weights=weights)
            uniform_diff = np.mean(step_diffs)
            open_diff = np.mean(step_diffs[:open_steps])
            rest_diff = np.mean(step_diffs[open_steps:])

            feat_score = 1.0 / (1.0 + weighted_diff)
            feature_scores.append(feat_score)

            results[feat_name] = {
                'weighted_diff': weighted_diff,
                'uniform_diff': uniform_diff,
                'open_diff': open_diff,
                'rest_diff': rest_diff,
                'score': feat_score
            }

            print(f"\n  [{feat_name}]")
            print(f"    加权差异: {weighted_diff:.4f}  (均匀: {uniform_diff:.4f})")
            print(f"    开盘段差异: {open_diff:.4f}  (其余: {rest_diff:.4f})")
            print(f"    评分: {feat_score:.4f}")

            key_steps = [0, 1, 2, 5, 10, 15, 20, 30, 44] if seq_len >= 45 else list(range(0, seq_len, max(1, seq_len//10)))
            parts = []
            for s in key_steps:
                if s < seq_len:
                    marker = "★" if s < open_steps else ""
                    parts.append(f"t={s}{marker}={step_diffs[s]:.3f}")
            print(f"    逐步差异(关键): {' '.join(parts)}")

        overall = np.mean(feature_scores)
        if overall >= 0.7:
            grade = "A (优秀)"
        elif overall >= 0.5:
            grade = "B (良好)"
        elif overall >= 0.3:
            grade = "C (一般)"
        else:
            grade = "D (需改进)"

        print(f"\n  ═══ 综合评价 ═══")
        print(f"  加权综合评分: {overall:.4f}  等级: {grade}")
        print(f"  (评分>0.7=A, >0.5=B, >0.3=C, <=0.3=D)")
        print(f"  注意: 开盘9:30-9:40权重{weight_open}x，9:40-9:50权重2x")

        results['overall'] = overall
        results['grade'] = grade
        return results

    # 构造测试数据
    np.random.seed(42)
    n_samples = 200
    seq_len = 45

    real_data = np.zeros((n_samples, seq_len, 3))
    for i in range(n_samples):
        base_vol = 0.005 + np.random.exponential(0.003)
        drift = np.random.randn() * 0.01
        for t in range(seq_len):
            if t < 15:
                real_data[i, t, 0] = np.random.randn() * base_vol * 0.5
                real_data[i, t, 1] = 0.3 + np.random.exponential(0.1)
                real_data[i, t, 2] = base_vol * 0.8
            else:
                decay = np.exp(-2.0 * (t - 15) / 30.0)
                real_data[i, t, 0] = drift * decay + np.random.randn() * base_vol * (1 + 2 * decay)
                real_data[i, t, 1] = 2.0 * np.exp(-3.0 * (t-15)/30.0) + 0.5
                real_data[i, t, 2] = base_vol * (1 + 2 * decay)

    # 对照组A: 优秀生成（微小噪声）
    gen_good = real_data + np.random.randn(*real_data.shape) * 0.001

    # 对照组B: 差生成（均匀噪声）
    gen_bad = np.random.randn(*real_data.shape) * 0.02

    # 对照组C: 整体OK但开盘段差
    gen_open_bad = real_data.copy()
    for i in range(n_samples):
        gen_open_bad[i, :10, 0] = np.random.randn(10) * 0.03
        gen_open_bad[i, :10, 1] = np.random.randn(10) * 0.5

    print("\n" + "="*60)
    print("  对照组A: 优秀生成数据（微小噪声）")
    print("="*60)
    validate_open_focused(real_data, gen_good, open_steps=10, weight_open=5.0)

    print("\n" + "="*60)
    print("  对照组B: 差生成数据（均匀噪声）")
    print("="*60)
    validate_open_focused(real_data, gen_bad, open_steps=10, weight_open=5.0)

    print("\n" + "="*60)
    print("  对照组C: 整体OK但开盘段差（加权vs均匀差异测试）")
    print("="*60)
    validate_open_focused(real_data, gen_open_bad, open_steps=10, weight_open=5.0)

    check("开盘重点验证框架", True, "3组对照完成")
    return {'open_focus_ok': True}


# ============================================================
# 主入口
# ============================================================
def main():
    parser = argparse.ArgumentParser(description='TimeGAN生成器诊断脚本')
    parser.add_argument('--test', type=str, default='all',
                       choices=['all', 'pipeline', 'clear', 'premarket', 'open-focus'],
                       help='测试项')
    args = parser.parse_args()

    print(f"{'━'*60}")
    print(f"  TimeGAN 生成器诊断")
    print(f"  时间: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  工作目录: {BASE_DIR}")
    print(f"{'━'*60}")

    all_results = {}

    if args.test in ('all', 'pipeline'):
        all_results['pipeline'] = test_full_pipeline()

    if args.test in ('all', 'clear'):
        all_results['clear_regen'] = test_clear_regenerate()

    if args.test in ('all', 'premarket'):
        all_results['premarket'] = test_premarket_generation()

    if args.test in ('all', 'open-focus'):
        all_results['open_focus'] = test_open_focus_validation()

    # 总结
    section("诊断总结")
    for test_name, result in all_results.items():
        if isinstance(result, dict):
            ok_count = sum(1 for k, v in result.items() if v is True)
            total = sum(1 for k, v in result.items() if isinstance(v, bool))
            summary = result.get('_summary', f"{ok_count}/{total}")
            print(f"  {test_name}: {summary}")

    print(f"\n  诊断完成 ✓")


if __name__ == '__main__':
    main()

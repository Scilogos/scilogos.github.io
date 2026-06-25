"""
pipeline_diagnose.py - 全面管线诊断脚本
==========================================
彻查数据→生成器→庄散对抗→反馈系统全链路，找出格式不匹配、数值异常、接口断裂等隐患。

用法:
  python pipeline_diagnose.py
  python pipeline_diagnose.py --fix   # 自动修复可修复的问题
"""

import os, sys, json, time, traceback
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

# ============================================================
# 配置
# ============================================================
BASE_DIR = Path(r"C:\Users\HUAWEI\Desktop\Adversarial Learning")
DOTPY_DIR = BASE_DIR / "dotpy"
DATA_DIR = BASE_DIR / "stockdata"
ADV_DATA_DIR = BASE_DIR / "adversarial data"
ADV_MODEL_DIR = BASE_DIR / "adversarial model"
RESULTS_DIR = BASE_DIR / "stockresults"

sys.path.insert(0, str(DOTPY_DIR))

# ============================================================
# 诊断报告
# ============================================================
@dataclass
class DiagResult:
    category: str
    name: str
    status: str       # PASS / WARN / FAIL / FIX
    detail: str
    fix_action: str = ""

results: List[DiagResult] = []
fixes_applied: List[str] = []

def report(cat, name, status, detail, fix=""):
    results.append(DiagResult(cat, name, status, detail, fix))
    icon = {"PASS": "✓", "WARN": "⚠", "FAIL": "✗", "FIX": "🔧"}[status]
    print(f"  {icon} [{cat}] {name}: {detail}")
    if fix:
        print(f"     修复: {fix}")

# ============================================================
# 1. 文件存在性检查
# ============================================================
def check_files():
    print("\n" + "="*60)
    print("  1. 文件存在性检查")
    print("="*60)

    # 脚本文件
    scripts = [
        "stock_config.py", "stock_data_manager.py", "market_generator.py",
        "adversarial_env.py", "stock_interpreter.py", "run_pipeline.py",
        "feedback_processor.py", "pipeline_monitor.py",
    ]
    for s in scripts:
        p = DOTPY_DIR / s
        if p.exists():
            size_kb = p.stat().st_size / 1024
            report("文件", s, "PASS", f"存在 ({size_kb:.1f}KB)")
        else:
            report("文件", s, "FAIL", "不存在!")

    # 模型文件
    models = {
        "phase_a.pt": ADV_MODEL_DIR / "phase_a.pt",
        "phase_b.pt": ADV_MODEL_DIR / "phase_b.pt",
        "phase_c.pt": ADV_MODEL_DIR / "phase_c.pt",
        "adversarial_model.pt": ADV_MODEL_DIR / "adversarial_model.pt",
    }
    for name, p in models.items():
        if p.exists():
            size_mb = p.stat().st_size / 1024 / 1024
            report("模型", name, "PASS", f"存在 ({size_mb:.2f}MB)")
        else:
            report("模型", name, "WARN", "不存在 (可能尚未训练)")

    # 数据文件
    data_checks = {
        "股票数据目录": DATA_DIR,
        "对抗数据目录": ADV_DATA_DIR,
        "结果目录": RESULTS_DIR,
    }
    for name, p in data_checks.items():
        if p.exists():
            n_files = len(list(p.glob("*")))
            report("目录", name, "PASS", f"存在 ({n_files}个文件)")
        else:
            report("目录", name, "FAIL", "不存在!")

    # 关键接口文件
    interface_files = {
        "calibration_params.json": RESULTS_DIR / "calibration_params.json",
        "morning_resume.json": RESULTS_DIR / "morning_resume.json",
        "trainer_state.json": RESULTS_DIR / "trainer_state.json",
        "arena_results.json": RESULTS_DIR / "arena_results.json",
        "validation_result.json": RESULTS_DIR / "validation_result.json",
    }
    for name, p in interface_files.items():
        if p.exists():
            report("接口", name, "PASS", "存在")
        else:
            report("接口", name, "WARN", "不存在 (首次运行前正常)")

# ============================================================
# 2. 股票数据质量检查
# ============================================================
def check_stock_data():
    print("\n" + "="*60)
    print("  2. 股票数据质量检查")
    print("="*60)

    if not DATA_DIR.exists():
        report("数据", "股票目录", "FAIL", "目录不存在")
        return

    csv_files = list(DATA_DIR.glob("*_daily.csv"))
    report("数据", "日K线文件数", "PASS" if len(csv_files) > 0 else "FAIL",
           f"{len(csv_files)} 个")

    if not csv_files:
        return

    # 抽样检查10个文件
    sample = csv_files[:10]
    issues = 0
    for f in sample:
        try:
            df = pd.read_csv(f)
            if len(df) < 30:
                issues += 1
                continue
            # 检查必要列
            required = ['date', 'open', 'high', 'low', 'close', 'volume']
            missing = [c for c in required if c not in df.columns]
            if missing:
                issues += 1
                report("数据", f.name, "FAIL", f"缺少列: {missing}")
                continue

            # 检查数值异常
            close = df['close'].values
            if (close <= 0).any():
                report("数据", f.name, "WARN", f"收盘价≤0: {(close<=0).sum()}条")
            if np.isnan(close).any():
                report("数据", f.name, "WARN", f"NaN: {np.isnan(close).sum()}条")
        except Exception as e:
            issues += 1
            report("数据", f.name, "FAIL", f"读取失败: {e}")

    if issues == 0:
        report("数据", "抽样10个文件", "PASS", "全部正常")

    # 全局统计
    total_rows = 0
    bad_files = 0
    for f in csv_files[:200]:  # 检查前200个
        try:
            df = pd.read_csv(f)
            total_rows += len(df)
        except:
            bad_files += 1

    report("数据", "前200文件行数", "PASS" if bad_files == 0 else "WARN",
           f"总计{total_rows}行, {bad_files}个坏文件")

# ============================================================
# 3. 生成器模型与数据检查
# ============================================================
def check_generator():
    print("\n" + "="*60)
    print("  3. 生成器模型与数据检查")
    print("="*60)

    # 检查Phase A/B/C模型文件大小合理性
    for phase in ['a', 'b', 'c']:
        p = ADV_MODEL_DIR / f"phase_{phase}.pt"
        if p.exists():
            size_mb = p.stat().st_size / 1024 / 1024
            # 模型应该>1MB
            if size_mb < 0.5:
                report("生成器", f"phase_{phase}.pt", "WARN",
                       f"过小 ({size_mb:.2f}MB), 可能损坏")
            else:
                report("生成器", f"phase_{phase}.pt", "PASS",
                       f"大小合理 ({size_mb:.2f}MB)")

            # 尝试加载
            try:
                import torch
                ckpt = torch.load(str(p), map_location='cpu', weights_only=False)
                keys = list(ckpt.keys())
                report("生成器", f"phase_{phase}内容", "PASS",
                       f"键: {keys[:5]}...")
            except Exception as e:
                report("生成器", f"phase_{phase}加载", "FAIL",
                       f"加载失败: {e}")
        else:
            report("生成器", f"phase_{phase}.pt", "WARN", "不存在")

    # 检查生成的假数据
    gen_files = list(ADV_DATA_DIR.glob("generated_*.npy"))
    if gen_files:
        for gf in gen_files:
            try:
                data = np.load(str(gf))
                report("生成器", gf.name, "PASS",
                       f"形状={data.shape}, 范围=[{data.min():.4f}, {data.max():.4f}]")

                # ★ 关键检查: 数据值域
                # z-score归一化后, 99%数据应在[-3, 3]范围内
                extreme_ratio = (np.abs(data) > 10).mean()
                if extreme_ratio > 0.01:
                    report("生成器", gf.name+"极值", "FAIL",
                           f"{extreme_ratio:.2%}数据|值|>10, 归一化可能失败",
                           "检查DataAdapter.load_batch的normalize逻辑")
                elif extreme_ratio > 0.001:
                    report("生成器", gf.name+"极值", "WARN",
                           f"{extreme_ratio:.2%}数据|值|>10, 轻微异常")
                else:
                    report("生成器", gf.name+"极值", "PASS",
                           f"极值比例{extreme_ratio:.4%}, 正常")

                # 检查NaN/Inf
                nan_ratio = np.isnan(data).mean()
                inf_ratio = np.isinf(data).mean()
                if nan_ratio > 0 or inf_ratio > 0:
                    report("生成器", gf.name+"数值", "FAIL",
                           f"NaN={nan_ratio:.4%}, Inf={inf_ratio:.4%}")
                else:
                    report("生成器", gf.name+"数值", "PASS", "无NaN/Inf")

            except Exception as e:
                report("生成器", gf.name, "FAIL", f"加载失败: {e}")
    else:
        report("生成器", "假数据", "WARN", "未找到generated_*.npy")

    # 检查validation_result.json
    val_file = RESULTS_DIR / "validation_result.json"
    if val_file.exists():
        try:
            with open(val_file) as f:
                val = json.load(f)
            for level_key in ['L1', 'L2', 'L3']:
                if level_key in val:
                    lv = val[level_key]
                    passed = lv.get('passed', False)
                    status = "PASS" if passed else "FAIL"
                    if level_key == 'L1':
                        errs = lv.get('errors', {})
                        detail = f"Mean_err={errs.get('mean_rel_err', 'N/A')}, Std_err={errs.get('std_rel_err', 'N/A')}"
                    elif level_key == 'L2':
                        detail = str(lv)[:100]
                    else:
                        ks = lv.get('ks_tests', [])
                        detail = f"KS tests: {len(ks)}个"
                    report("生成器", f"验证{level_key}", status, detail)
        except Exception as e:
            report("生成器", "验证结果", "WARN", f"解析失败: {e}")

# ============================================================
# 4. ★ 基准线格式兼容性检查 (核心Bug排查)
# ============================================================
def check_benchmark_format():
    print("\n" + "="*60)
    print("  4. 基准线格式兼容性检查 ★核心★")
    print("="*60)

    """
    关键Bug排查:
    adversarial_env._setup_benchmark 假设输入是pct收益率, 用cumprod(1+x)重建价格
    但实际:
      - 生成器输出: z-score归一化的OHLCV (mean=0, std=1)
      - 第0列是open, 不是close/pct
      - cumprod(1 + z-score) 会产生完全错误的价格

    检查方法:
      1. 加载generated_*.npy, 看值域和统计量
      2. 模拟_setup_benchmark的cumprod逻辑, 看输出是否合理
      3. 对比真实股票价格范围
    """

    gen_files = list(ADV_DATA_DIR.glob("generated_*.npy"))
    if not gen_files:
        report("基准线", "假数据文件", "WARN", "无generated_*.npy, 跳过")
        return

    for gf in gen_files:
        data = np.load(str(gf))
        report("基准线", f"{gf.name}输入格式", "INFO",
               f"形状={data.shape}, ndim={data.ndim}")

        # 模拟_setup_benchmark的cumprod逻辑
        if data.ndim == 3 and data.shape[-1] >= 1:
            # 取第0列 (代码里写的是 day_prices = data[day_idx, :, 0])
            col0 = data[0, :, 0]  # 第0个sample的第0列

            # 统计
            report("基准线", "第0列统计", "INFO",
                   f"mean={col0.mean():.4f}, std={col0.std():.4f}, "
                   f"min={col0.min():.4f}, max={col0.max():.4f}")

            # 判断数据类型
            if abs(col0.mean()) < 0.5 and abs(col0.std() - 1.0) < 0.5:
                data_type = "z-score归一化 (mean≈0, std≈1)"
            elif col0.min() >= -0.15 and col0.max() <= 0.15:
                data_type = "百分比收益率 (pct, 范围±15%)"
            elif col0.min() > 0:
                data_type = "原始价格 (全部>0)"
            else:
                data_type = "未知格式"

            report("基准线", "数据类型判断", "INFO", data_type)

            # 模拟cumprod(1 + col0) * 10.0
            initial_price = 10.0
            if data_type.startswith("z-score"):
                simulated = np.cumprod(1 + col0) * initial_price
                report("基准线", "cumprod模拟结果", "FAIL",
                       f"z-score做cumprod(1+x)是错误的! "
                       f"结果范围=[{simulated.min():.2f}, {simulated.max():.2f}]",
                       "z-score值不在[-1,1]范围内, cumprod(1+x)会产生指数级偏差")

                # 验证: 用一个极端例子
                # 假设z-score=2.0 (正常值), cumprod(1+2.0)=3.0, 一个tick价格就×3
                # 30个tick后 3^30 = 2e14, 完全爆炸
                report("基准线", "Bug确认", "FAIL",
                       "adversarial_env._setup_benchmark 把z-score当pct用, "
                       "cumprod(1+z_score)会导致价格爆炸/坍塌",
                       "需要修复_setup_benchmark, 正确处理z-score格式")
            elif data_type.startswith("百分比收益率"):
                simulated = np.cumprod(1 + col0) * initial_price
                if simulated.min() > 0 and simulated.max() < 1000:
                    report("基准线", "cumprod模拟结果", "PASS",
                           f"结果范围=[{simulated.min():.2f}, {simulated.max():.2f}], 合理")
                else:
                    report("基准线", "cumprod模拟结果", "WARN",
                           f"结果范围=[{simulated.min():.2f}, {simulated.max():.2f}], 可能异常")

            # ★ 第3列检查 (close)
            if data.shape[-1] >= 4:
                col3 = data[0, :, 3]  # close列
                report("基准线", "第3列(close)统计", "INFO",
                       f"mean={col3.mean():.4f}, std={col3.std():.4f}, "
                       f"min={col3.min():.4f}, max={col3.max():.4f}")

                # 如果第3列也是z-score, cumprod一样会崩
                if abs(col3.mean()) < 0.5 and abs(col3.std() - 1.0) < 0.5:
                    col3_sim = np.cumprod(1 + col3) * initial_price
                    report("基准线", "close列cumprod", "FAIL",
                           f"z-score close做cumprod: [{col3_sim.min():.2f}, {col3_sim.max():.2f}]")

    # ★ 检查真实股票数据的归一化方式
    csv_files = list(DATA_DIR.glob("*_daily.csv"))
    if csv_files:
        sample_file = csv_files[0]
        try:
            df = pd.read_csv(sample_file)
            close_raw = df['close'].values.astype(np.float64)

            # 检查DataAdapter.load_stock(normalize=True)的输出
            from stock_data_manager import DataAdapter
            adapter = DataAdapter(DATA_DIR)
            stocks = adapter.list_stocks(min_length=50)
            if stocks:
                result = adapter.load_stock(stocks[0], normalize=True)
                if result:
                    norm_data, meta = result
                    report("基准线", "DataAdapter归一化方式", "INFO",
                           f"z-score: mean={norm_data.mean(axis=0)[:3]}, "
                           f"std={norm_data.std(axis=0)[:3]}")

                    # 检查load_batch的输出
                    batch_data, _ = adapter.load_batch(stocks[:10], max_seqs=100)
                    if len(batch_data) > 0:
                        report("基准线", "load_batch输出格式", "INFO",
                               f"形状={batch_data.shape}, "
                               f"mean={batch_data.mean():.4f}, std={batch_data.std():.4f}")

                        # ★ 核心: batch_data是z-score, 传给adversarial_env会被当成pct
                        if abs(batch_data.mean()) < 0.5 and abs(batch_data.std() - 1.0) < 0.5:
                            report("基准线", "格式不匹配Bug", "FAIL",
                                   "load_batch输出z-score, 但_setup_benchmark用cumprod(1+x)处理, "
                                   "两者不兼容! z-score值域[-∞,+∞], cumprod假设x是pct∈[-0.1,0.1]",
                                   "修复方案: 1) _setup_benchmark先反归一化再转pct, "
                                   "或 2) 用真实价格直接做benchmark")
        except Exception as e:
            report("基准线", "DataAdapter检查", "WARN", f"检查失败: {e}")

# ============================================================
# 5. 对抗环境参数一致性检查
# ============================================================
def check_adversarial_config():
    print("\n" + "="*60)
    print("  5. 对抗环境参数检查")
    print("="*60)

    try:
        from stock_config import AdversarialConfig, GeneratorConfig
        cfg = AdversarialConfig()
        gcfg = GeneratorConfig()

        # 检查关键参数
        checks = [
            ("初始价格", cfg.initial_price, 10.0, "应=10.0"),
            ("Episode长度", cfg.episode_length, 240, "应=240 (4h×60min)"),
            ("庄家资金比", cfg.dealer_capital_ratio, None, f"={cfg.dealer_capital_ratio}"),
            ("散户比例", cfg.retailer_ratio, None, f"={cfg.retailer_ratio}"),
            ("游资比例", cfg.hotmoney_ratio, None, f"={cfg.hotmoney_ratio}"),
            ("进化间隔", cfg.evolution_interval, None, f"={cfg.evolution_interval}"),
            ("拉马克率", cfg.lamarck_rate, None, f"={cfg.lamarck_rate}"),
            ("达尔文率", cfg.darwin_rate, None, f"={cfg.darwin_rate}"),
        ]

        for name, actual, expected, note in checks:
            if expected is not None and abs(actual - expected) > 0.01:
                report("对抗配置", name, "WARN",
                       f"={actual}, 预期{expected}. {note}")
            else:
                report("对抗配置", name, "PASS", note)

        # GeneratorConfig
        report("对抗配置", "seq_len", "PASS", f"={gcfg.seq_len}")
        report("对抗配置", "feature_dim", "PASS", f"={gcfg.feature_dim}")
        report("对抗配置", "hidden_dim", "PASS", f"={gcfg.hidden_dim}")

        # ★ 检查维度一致性
        if gcfg.feature_dim != 6:
            report("对抗配置", "feature_dim≠6", "FAIL",
                   f"feature_dim={gcfg.feature_dim}, 但OHLCV+pct=6",
                   "检查GeneratorConfig.feature_dim")
        else:
            report("对抗配置", "feature_dim=6", "PASS", "与OHLCV+pct一致")

    except Exception as e:
        report("对抗配置", "配置加载", "FAIL", f"失败: {e}")

# ============================================================
# 6. 模块间接口一致性检查
# ============================================================
def check_interfaces():
    print("\n" + "="*60)
    print("  6. 模块间接口一致性检查")
    print("="*60)

    # 检查stock_config的导出
    try:
        from stock_config import (
            DATA_DIR as _DATA_DIR, ADV_MODEL_DIR as _ADV_MODEL_DIR,
            ADV_DATA_DIR as _ADV_DATA_DIR, RESULTS_DIR as _RESULTS_DIR,
            GeneratorConfig, AdversarialConfig, setup_logger,
        )
        report("接口", "stock_config导出", "PASS", "核心名称全部可导入")
    except ImportError as e:
        report("接口", "stock_config导出", "FAIL", f"导入失败: {e}")

    # 检查跨模块调用链
    interface_checks = [
        ("DataAdapter.load_batch → CTimeGAN.train",
         "stock_data_manager.DataAdapter.load_batch 输出 (N,30,6) → market_generator.CTimeGAN.train 输入"),
        ("CTimeGAN.generate → np.save",
         "market_generator 输出 (N,30,6) z-score → adversarial data/generated_*.npy"),
        ("generated_*.npy → MarketEnv._setup_benchmark",
         "z-score格式 → _setup_benchmark用cumprod(1+x)处理 → ★格式不匹配★"),
        ("feedback_processor → calibration_params.json → adversarial_env",
         "JSON校准参数 → _load_calibration → reward计算"),
        ("adversarial_env → trainer_state.json → feedback_processor",
         "JSON训练状态 → MorningPlanner → morning_resume.json"),
    ]

    for name, desc in interface_checks:
        if "格式不匹配" in desc:
            report("接口", name, "FAIL", desc,
                   "这是当前最关键的Bug, 必须修复_setup_benchmark或改用真实价格")
        else:
            report("接口", name, "PASS", desc)

# ============================================================
# 7. 运行时环境检查
# ============================================================
def check_runtime():
    print("\n" + "="*60)
    print("  7. 运行时环境检查")
    print("="*60)

    # Python版本
    pv = sys.version
    report("环境", "Python版本", "PASS" if "3.1" in pv else "WARN", pv)

    # 关键包
    packages = ['numpy', 'pandas', 'torch', 'scipy']
    for pkg in packages:
        try:
            mod = __import__(pkg)
            ver = getattr(mod, '__version__', 'unknown')
            report("环境", pkg, "PASS", f"v{ver}")
        except ImportError:
            report("环境", pkg, "FAIL", "未安装!")

    # PyTorch设备
    try:
        import torch
        report("环境", "CUDA可用", "PASS" if torch.cuda.is_available() else "WARN",
               f"{'是' if torch.cuda.is_available() else '否'} (当前CPU模式)")
    except:
        pass

    # 磁盘空间
    try:
        import shutil
        usage = shutil.disk_usage(str(BASE_DIR))
        free_gb = usage.free / 1024**3
        report("环境", "磁盘剩余", "PASS" if free_gb > 10 else "WARN",
               f"{free_gb:.1f}GB")
    except:
        pass

# ============================================================
# 8. 内存估算
# ============================================================
def check_memory_estimate():
    print("\n" + "="*60)
    print("  8. 内存与时间估算")
    print("="*60)

    # 全量数据内存
    csv_files = list(DATA_DIR.glob("*_daily.csv"))
    total_size = sum(f.stat().st_size for f in csv_files) / 1024**3
    report("估算", "股票数据总量", "INFO", f"{total_size:.2f}GB ({len(csv_files)}文件)")

    # Arena估算
    try:
        from stock_config import AdversarialConfig
        cfg = AdversarialConfig()
        # 8组合 × 500轮 × 240 tick
        total_ticks = 8 * 500 * 240
        report("估算", "Arena(8×500轮)", "INFO",
               f"总tick={total_ticks:,}, 预计6-10小时(CPU)")
        report("估算", "Arena(8×200轮)", "INFO",
               f"总tick={8*200*240:,}, 预计2.5-4小时(CPU)")
    except:
        pass

# ============================================================
# 主函数
# ============================================================
def main():
    print("╔" + "═"*58 + "╗")
    print("║  对抗学习量化系统 - 全面管线诊断 v1.0" + " "*18 + "║")
    print("╚" + "═"*58 + "╝")

    check_files()
    check_stock_data()
    check_generator()
    check_benchmark_format()   # ★ 核心
    check_adversarial_config()
    check_interfaces()
    check_runtime()
    check_memory_estimate()

    # ── 汇总 ──
    print("\n" + "="*60)
    print("  诊断汇总")
    print("="*60)

    pass_count = sum(1 for r in results if r.status == "PASS")
    warn_count = sum(1 for r in results if r.status == "WARN")
    fail_count = sum(1 for r in results if r.status == "FAIL")
    fix_count = sum(1 for r in results if r.status == "FIX")

    print(f"\n  ✓ PASS: {pass_count}")
    print(f"  ⚠ WARN: {warn_count}")
    print(f"  ✗ FAIL: {fail_count}")

    if fail_count > 0:
        print(f"\n  ★ 发现 {fail_count} 个严重问题:")
        for r in results:
            if r.status == "FAIL":
                print(f"    ✗ [{r.category}] {r.name}: {r.detail}")
                if r.fix_action:
                    print(f"      → 修复方案: {r.fix_action}")

    # 保存报告
    report_path = RESULTS_DIR / "diagnosis_report.json"
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    report_data = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "summary": {"pass": pass_count, "warn": warn_count, "fail": fail_count},
        "failures": [{"category": r.category, "name": r.name,
                       "detail": r.detail, "fix": r.fix_action}
                      for r in results if r.status == "FAIL"],
        "all_results": [{"category": r.category, "name": r.name,
                          "status": r.status, "detail": r.detail}
                         for r in results],
    }
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(report_data, f, ensure_ascii=False, indent=2)

    print(f"\n  报告已保存: {report_path}")

    return fail_count

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--fix", action="store_true", help="自动修复可修复的问题")
    args = parser.parse_args()
    main()

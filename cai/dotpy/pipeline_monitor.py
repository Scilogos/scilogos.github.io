"""
pipeline_monitor.py - 对抗学习管线监察脚本
============================================
专门盯着各环节是否在"真心干活"，重点监察庄散对抗。

检查项:
  1. 数据完整性: 日K线文件数/质量/行业分类
  2. 生成器健康: 模型是否存在/训练历史/参数变化
  3. 庄散对抗健康（重点）:
     - 模型是否存在
     - Agent持仓是否在变（不是空转）
     - 奖励是否非零（有学习信号）
     - 庄家是否真的在执行防共四策
     - 散户是否有类型差异（不是全部一样）
     - 策略多样性是否在变化
     - 价格轨迹是否有波动（不是一条直线）
  4. 解读器: 报告是否存在/信号是否有方向
  5. 反馈闭环: 校准参数/trainer_state

用法:
  python pipeline_monitor.py                  # 全量检查
  python pipeline_monitor.py --focus adversarial  # 只查对抗训练
  python pipeline_monitor.py --verbose         # 详细输出
"""

import os, sys, json, time
from pathlib import Path
from collections import Counter

sys.path.insert(0, str(Path(__file__).parent))
from stock_config import (
    DATA_DIR, ADV_MODEL_DIR, ADV_DATA_DIR, RESULTS_DIR,
    setup_logger,
)

logger = setup_logger("Monitor")

# ============================================================
# 检查结果容器
# ============================================================
class CheckResult:
    def __init__(self, name, category):
        self.name = name
        self.category = category
        self.status = "UNKNOWN"  # PASS / WARN / FAIL / SKIP
        self.details = []
        self.metrics = {}
    
    def pass_(self, msg=""):
        self.status = "PASS"
        if msg:
            self.details.append(msg)
    
    def warn(self, msg):
        self.status = "WARN" if self.status != "FAIL" else "FAIL"
        self.details.append(f"⚠ {msg}")
    
    def fail(self, msg):
        self.status = "FAIL"
        self.details.append(f"✗ {msg}")
    
    def metric(self, key, value):
        self.metrics[key] = value
    
    def __str__(self):
        icon = {"PASS": "✅", "WARN": "⚠️", "FAIL": "❌", "SKIP": "⏭️", "UNKNOWN": "❓"}[self.status]
        line = f"  {icon} [{self.category}] {self.name}"
        if self.metrics:
            m = ", ".join(f"{k}={v}" for k, v in self.metrics.items())
            line += f" ({m})"
        for d in self.details:
            line += f"\n      {d}"
        return line


# ============================================================
# Phase 1: 数据检查
# ============================================================
def check_data(verbose=False):
    results = []
    
    # 1.1 日K线文件数
    r = CheckResult("日K线文件", "数据")
    csv_files = list(DATA_DIR.glob("*_daily.csv"))
    r.metric("文件数", len(csv_files))
    if len(csv_files) == 0:
        r.fail("无任何日K线文件，Phase 1未执行")
    elif len(csv_files) < 100:
        r.warn(f"仅{len(csv_files)}只，远少于全市场(~4400)")
    else:
        r.pass_(f"共{len(csv_files)}只")
    results.append(r)
    
    # 1.2 数据质量抽检
    r = CheckResult("数据质量抽检", "数据")
    if csv_files:
        import pandas as pd
        import numpy as np
        sample = csv_files[:min(20, len(csv_files))]
        issues = 0
        total_rows = 0
        for f in sample:
            try:
                df = pd.read_csv(f, usecols=['open', 'high', 'low', 'close', 'volume'])
                total_rows += len(df)
                # OHLC逻辑检查
                bad = ((df['high'] < df['low']) | (df['high'] < df['open']) | 
                       (df['high'] < df['close'])).sum()
                if bad > 0:
                    issues += 1
                # 全零检查
                if (df['close'] == 0).all():
                    issues += 1
            except Exception as e:
                issues += 1
                if verbose:
                    r.details.append(f"  读取失败: {f.name}: {e}")
        r.metric("抽检", len(sample))
        r.metric("问题文件", issues)
        r.metric("平均行数", int(total_rows / max(len(sample), 1)))
        if issues > len(sample) * 0.3:
            r.fail(f"{issues}/{len(sample)}文件有问题")
        elif issues > 0:
            r.warn(f"{issues}/{len(sample)}文件有小问题")
        else:
            r.pass_("抽检全部通过")
    else:
        r.skip("无文件可抽检")
    results.append(r)
    
    # 1.3 行业分类
    r = CheckResult("行业分类", "数据")
    ind_file = DATA_DIR / "industry_classification.json"
    if ind_file.exists():
        with open(ind_file, 'r', encoding='utf-8') as f:
            ind_map = json.load(f)
        industries = list(ind_map.keys())
        total_classified = sum(len(v) for v in ind_map.values())
        unknown_count = len(ind_map.get("未知", []))
        r.metric("行业数", len(industries))
        r.metric("已分类", total_classified)
        r.metric("未知", unknown_count)
        
        if len(industries) <= 1:
            r.fail(f"仅{industries[0] if industries else '无'}一个行业，分类失败")
        elif unknown_count > total_classified * 0.5:
            r.fail(f"超过50%分类为'未知'({unknown_count}/{total_classified})")
        elif unknown_count > total_classified * 0.2:
            r.warn(f"{unknown_count}/{total_classified}分类为'未知'")
        else:
            r.pass_(f"{len(industries)}个行业, {total_classified}只已分类")
            if verbose:
                for ind, codes in sorted(ind_map.items(), key=lambda x: -len(x[1]))[:5]:
                    r.details.append(f"  {ind}: {len(codes)}只")
    else:
        r.fail("industry_classification.json不存在")
    results.append(r)
    
    return results


# ============================================================
# Phase 2: 生成器检查
# ============================================================
def check_generator(verbose=False):
    results = []
    
    # 2.1 模型文件
    r = CheckResult("生成器模型", "生成器")
    model_files = {
        "Phase A": ADV_MODEL_DIR / "phase_a.pt",
        "Phase B": ADV_MODEL_DIR / "phase_b.pt",
        "Phase C": ADV_MODEL_DIR / "phase_c.pt",
    }
    existing = []
    for name, path in model_files.items():
        if path.exists():
            size = path.stat().st_size / 1024  # KB
            r.metric(name, f"{size:.0f}KB")
            existing.append(name)
        else:
            r.details.append(f"  {name}: 不存在")
    
    if len(existing) == 3:
        r.pass_("三阶段模型全部存在")
    elif len(existing) > 0:
        r.warn(f"仅{','.join(existing)}存在")
    else:
        r.fail("无任何模型文件，生成器未训练")
    results.append(r)
    
    # 2.2 训练历史
    r = CheckResult("训练历史", "生成器")
    hist_file = ADV_MODEL_DIR / "train_history.json"
    if hist_file.exists():
        with open(hist_file, 'r') as f:
            hist = json.load(f)
        
        pa = hist.get('phase_a', [])
        pb = hist.get('phase_b', [])
        pc = hist.get('phase_c', [])
        r.metric("A轮数", len(pa))
        r.metric("B轮数", len(pb))
        r.metric("C轮数", len(pc))
        
        # 检查loss是否在下降
        if len(pa) >= 2:
            a_start, a_end = pa[0], pa[-1]
            r.metric("A_loss", f"{a_start:.4f}→{a_end:.4f}")
            if a_end < a_start:
                r.pass_(f"Phase A loss下降 ({a_start:.4f}→{a_end:.4f})")
            else:
                r.warn(f"Phase A loss未下降 ({a_start:.4f}→{a_end:.4f})")
        else:
            r.warn("Phase A训练轮数不足")
        
        if len(pc) >= 2:
            d_losses = [e.get('d_loss', 0) for e in pc if isinstance(e, dict)]
            g_losses = [e.get('g_loss', 0) for e in pc if isinstance(e, dict)]
            if d_losses and g_losses:
                r.metric("D_loss", f"{d_losses[0]:.4f}→{d_losses[-1]:.4f}")
                r.metric("G_loss", f"{g_losses[0]:.4f}→{g_losses[-1]:.4f}")
                if d_losses[-1] < d_losses[0] or g_losses[-1] < g_losses[0]:
                    r.pass_("Phase C对抗loss有变化")
                else:
                    r.warn("Phase C loss无下降")
    else:
        r.fail("train_history.json不存在")
    results.append(r)
    
    # 2.3 生成数据
    r = CheckResult("生成数据", "生成器")
    gen_files = list(ADV_DATA_DIR.glob("generated_*.npy"))
    if gen_files:
        r.metric("文件数", len(gen_files))
        r.pass_(f"存在{len(gen_files)}个生成数据文件")
    else:
        r.warn("无生成数据文件(未执行--mode generate)")
    results.append(r)
    
    # 2.4 验证结果
    r = CheckResult("三级验证", "生成器")
    val_file = RESULTS_DIR / "validation_result.json"
    if val_file.exists():
        with open(val_file, 'r') as f:
            val = json.load(f)
        overall = val.get('overall', {})
        r.metric("通过", "是" if overall.get('passed') else "否")
        r.metric("验证级别", overall.get('levels_tested', '?'))
        
        for level in ['L1', 'L2', 'L3']:
            if level in val:
                passed = val[level].get('passed', False)
                icon = "✅" if passed else "❌"
                r.details.append(f"  {icon} {level}: {'通过' if passed else '未通过'}")
        
        if overall.get('passed'):
            r.pass_("三级验证全部通过")
        else:
            r.warn("存在未通过的验证级别")
    else:
        r.warn("validation_result.json不存在(未执行验证)")
    results.append(r)
    
    return results


# ============================================================
# Phase 3: 庄散对抗检查（重点）
# ============================================================
def check_adversarial(verbose=False):
    results = []
    
    # 3.1 对抗模型
    r = CheckResult("对抗模型", "对抗训练")
    model_path = ADV_MODEL_DIR / "adversarial_model.pt"
    if model_path.exists():
        size = model_path.stat().st_size / 1024
        r.metric("大小", f"{size:.0f}KB")
        r.pass_("模型文件存在")
    else:
        r.fail("adversarial_model.pt不存在，对抗训练未执行")
    results.append(r)
    
    # 3.2 训练结果
    r = CheckResult("训练结果", "对抗训练")
    results_file = RESULTS_DIR / "adversarial_results.json"
    if results_file.exists():
        with open(results_file, 'r') as f:
            adv_results = json.load(f)
        
        final_rewards = adv_results.get('final_rewards', {})
        d_reward = final_rewards.get('dealer', 0)
        r_reward = final_rewards.get('retailer', 0)
        h_reward = final_rewards.get('hotmoney', 0)
        gen = adv_results.get('generation', 0)
        diversity = adv_results.get('strategy_diversity', [])
        
        r.metric("庄家reward", f"{d_reward:.2f}")
        r.metric("散户reward", f"{r_reward:.2f}")
        r.metric("游资reward", f"{h_reward:.2f}")
        r.metric("进化代数", gen)
        
        # 【重点检查1】奖励是否全零（空转）
        if d_reward == 0 and r_reward == 0 and h_reward == 0:
            r.fail("⚠️ 三方reward全为0！训练完全空转，Agent持仓从未更新")
        elif abs(d_reward) < 0.01 and abs(r_reward) < 0.01:
            r.warn("reward接近0，训练可能未产生有效信号")
        else:
            r.pass_("reward非零，有学习信号")
        
        # 【重点检查2】庄家是否真的在赚钱（散户亏钱=庄家赚钱）
        if d_reward > 0 and r_reward < 0:
            r.details.append("  ✓ 庄家盈利/散户亏损 → 对抗机制有效")
        elif d_reward < 0 and r_reward > 0:
            r.details.append("  ⚠ 庄家亏损/散户盈利 → 庄家策略可能太弱")
        elif d_reward > 0 and r_reward > 0:
            r.details.append("  ⚠ 双方都盈利 → 可能不是零和博弈(有外部注入)")
        
        # 【重点检查3】策略多样性变化
        if diversity:
            r.metric("多样性", f"{diversity[0]:.4f}→{diversity[-1]:.4f}")
            if len(diversity) >= 2:
                if diversity[-1] < 0.01:
                    r.fail("策略多样性趋近0！所有散户策略趋同")
                elif diversity[-1] < diversity[0] * 0.5:
                    r.warn("策略多样性显著下降")
                else:
                    r.details.append("  ✓ 策略多样性保持")
    else:
        r.fail("adversarial_results.json不存在")
    results.append(r)
    
    # 3.3 价格轨迹检查
    r = CheckResult("价格轨迹", "对抗训练")
    price_file = ADV_DATA_DIR / "last_episode_prices.csv"
    if price_file.exists():
        import pandas as pd
        import numpy as np
        df = pd.read_csv(price_file)
        prices = df['price'].values
        
        if len(prices) < 10:
            r.fail(f"价格数据太短({len(prices)}条)")
        else:
            price_std = np.std(prices)
            price_range = (prices.max() - prices.min()) / (prices.mean() + 1e-8)
            returns = np.diff(prices) / (prices[:-1] + 1e-8)
            ret_std = np.std(returns)
            
            r.metric("数据点", len(prices))
            r.metric("价格波动", f"{price_std:.4f}")
            r.metric("极差比", f"{price_range:.2%}")
            r.metric("收益率std", f"{ret_std:.6f}")
            
            # 【重点检查4】价格是否是一条直线（没有交易）
            if price_std < 1e-6:
                r.fail("⚠️ 价格完全不动！OrderBook没有任何成交")
            elif ret_std < 1e-6:
                r.fail("⚠️ 收益率全为0，价格可能是线性变化")
            elif price_range < 0.001:
                r.warn("价格波动极小，市场可能不活跃")
            else:
                r.pass_(f"价格有波动(std={price_std:.4f})")
                
                # 检查是否有趋势
                if len(prices) >= 20:
                    first_half = np.mean(prices[:len(prices)//2])
                    second_half = np.mean(prices[len(prices)//2:])
                    if abs(second_half - first_half) / (first_half + 1e-8) > 0.02:
                        r.details.append("  ✓ 存在趋势变化(前半段vs后半段)")
                    else:
                        r.details.append("  整体无明显趋势")
                
                if verbose:
                    # 打印价格分布
                    percentiles = np.percentile(prices, [10, 25, 50, 75, 90])
                    r.details.append(f"  分位: P10={percentiles[0]:.2f} "
                                   f"P50={percentiles[2]:.2f} P90={percentiles[4]:.2f}")
    else:
        r.fail("last_episode_prices.csv不存在")
    results.append(r)
    
    # 3.4 Trainer状态
    r = CheckResult("训练器状态", "对抗训练")
    state_file = RESULTS_DIR / "trainer_state.json"
    if state_file.exists():
        with open(state_file, 'r', encoding='utf-8') as f:
            state = json.load(f)
        
        r.metric("进化代数", state.get('generation', 0))
        r.metric("信息操纵力", f"{state.get('dealer_info_power', 0):.2f}")
        r.metric("震仓强度", f"{state.get('dealer_shake_intensity', 0):.3f}")
        r.metric("白手套资金", f"{state.get('dealer_puppet_capital', 0):.0f}")
        r.metric("策略多样性", f"{state.get('strategy_diversity', 0):.4f}")
        r.metric("校准权重", f"{state.get('cal_reward_weight', 0):.2f}")
        
        # 【重点检查5】庄家参数是否合理
        info_power = state.get('dealer_info_power', 0)
        shake = state.get('dealer_shake_intensity', 0)
        if info_power == 0.5 and shake == 0.05:
            r.warn("庄家参数=初始值，可能未经历进化或校准")
        else:
            r.pass_("庄家参数已偏离初始值（有进化/校准痕迹）")
        
        r.details.append(f"  时间戳: {state.get('timestamp', 'N/A')}")
    else:
        r.warn("trainer_state.json不存在（训练未完成或未保存）")
    results.append(r)
    
    # 3.5 校准状态
    r = CheckResult("实盘校准", "对抗训练")
    cal_file = RESULTS_DIR / "calibration_params.json"
    if cal_file.exists():
        with open(cal_file, 'r', encoding='utf-8') as f:
            cal = json.load(f)
        
        r.metric("胜率", f"{cal.get('win_rate', 0):.1%}")
        r.metric("总交易", cal.get('total_trades', 0))
        r.metric("需重训", "是" if cal.get('need_retrain') else "否")
        
        if cal.get('need_retrain'):
            r.warn(f"校准建议重训: {cal.get('retrain_reason', '')}")
        else:
            r.pass_("校准参数存在")
    else:
        r.skip("未进行实盘校准（正常，首次运行无反馈）")
    results.append(r)
    
    # 3.6 代码完整性检查
    r = CheckResult("代码完整性", "对抗训练")
    try:
        import torch
        # 尝试加载模型检查Agent数量
        if model_path.exists():
            ckpt = torch.load(model_path, map_location='cpu', weights_only=False)
            n_retailers = len(ckpt.get('retailers', []))
            n_hotmoney = len(ckpt.get('hotmoney', []))
            has_dealer = 'dealer' in ckpt
            
            r.metric("庄家", "存在" if has_dealer else "缺失")
            r.metric("散户数", n_retailers)
            r.metric("游资数", n_hotmoney)
            
            if not has_dealer:
                r.fail("模型中无庄家！")
            elif n_retailers == 0:
                r.fail("模型中无散户！")
            elif n_hotmoney == 0:
                r.warn("模型中无游资")
            else:
                r.pass_(f"Agent配置: 1庄家+{n_retailers}散户+{n_hotmoney}游资")
            
            # 检查散户策略参数是否完全相同（没有多样性）
            if n_retailers >= 2:
                params_0 = ckpt['retailers'][0]
                all_same = True
                for i in range(1, min(n_retailers, 5)):
                    params_i = ckpt['retailers'][i]
                    for key in params_0:
                        if not torch.equal(params_0[key], params_i[key]):
                            all_same = False
                            break
                    if not all_same:
                        break
                
                if all_same:
                    r.fail("⚠️ 所有散户策略参数完全相同！没有多样性，进化未生效")
                else:
                    r.details.append("  ✓ 散户策略参数有差异")
    except Exception as e:
        r.warn(f"无法检查模型内部: {e}")
    results.append(r)
    
    return results


# ============================================================
# Phase 4: 解读器检查
# ============================================================
def check_interpreter(verbose=False):
    results = []
    
    r = CheckResult("解读报告", "解读器")
    report_file = RESULTS_DIR / "interpretation_report.json"
    if report_file.exists():
        with open(report_file, 'r', encoding='utf-8') as f:
            report = json.load(f)
        
        signal = report.get('L4_signal', {})
        direction = signal.get('direction', 'N/A')
        confidence = signal.get('confidence', 0)
        perm = report.get('permutation_test', {})
        
        r.metric("信号方向", direction)
        r.metric("置信度", f"{confidence:.0%}")
        r.metric("排列检验p", f"{perm.get('p_value', 0):.3f}")
        
        if direction == 'N/A' or confidence == 0:
            r.warn("信号缺失或置信度为0")
        elif perm.get('reject_null'):
            r.pass_(f"信号={direction}, 排列检验通过(p={perm.get('p_value', 0):.3f})")
        else:
            r.warn(f"信号={direction}, 但排列检验未通过(可能过拟合)")
        
        if verbose and signal.get('evidence'):
            for ev in signal['evidence'][:3]:
                r.details.append(f"  证据: {ev}")
    else:
        r.warn("解读报告不存在（未执行Phase 4）")
    results.append(r)
    
    return results


# ============================================================
# 主入口
# ============================================================
def main():
    import argparse
    parser = argparse.ArgumentParser(description="管线监察脚本")
    parser.add_argument("--focus", choices=["all", "data", "generator", "adversarial", "interpreter"],
                        default="all", help="检查重点")
    parser.add_argument("--verbose", action="store_true", help="详细输出")
    args = parser.parse_args()
    
    print("\n" + "=" * 60)
    print("  对抗学习管线监察报告")
    print(f"  时间: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  数据目录: {DATA_DIR}")
    print(f"  模型目录: {ADV_MODEL_DIR}")
    print(f"  结果目录: {RESULTS_DIR}")
    print("=" * 60)
    
    all_results = []
    
    if args.focus in ("all", "data"):
        print("\n📋 Phase 1: 数据检查")
        all_results.extend(check_data(args.verbose))
    
    if args.focus in ("all", "generator"):
        print("\n📋 Phase 2: 生成器检查")
        all_results.extend(check_generator(args.verbose))
    
    if args.focus in ("all", "adversarial"):
        print("\n📋 Phase 3: 庄散对抗检查（重点）")
        all_results.extend(check_adversarial(args.verbose))
    
    if args.focus in ("all", "interpreter"):
        print("\n📋 Phase 4: 解读器检查")
        all_results.extend(check_interpreter(args.verbose))
    
    # 输出结果
    print()
    for r in all_results:
        print(r)
    
    # 汇总
    n_pass = sum(1 for r in all_results if r.status == "PASS")
    n_warn = sum(1 for r in all_results if r.status == "WARN")
    n_fail = sum(1 for r in all_results if r.status == "FAIL")
    n_skip = sum(1 for r in all_results if r.status == "SKIP")
    
    print(f"\n{'=' * 60}")
    print(f"  汇总: ✅{n_pass} 通过  ⚠️{n_warn} 警告  ❌{n_fail} 失败  ⏭️{n_skip} 跳过")
    
    if n_fail > 0:
        print("  ⚠️ 存在失败项，请按上述提示修复")
        failed = [r for r in all_results if r.status == "FAIL"]
        print("\n  失败项:")
        for r in failed:
            print(f"    ❌ [{r.category}] {r.name}")
            for d in r.details:
                print(f"       {d}")
    elif n_warn > 0:
        print("  整体通过，但存在警告项")
    else:
        print("  ✅ 全部通过！管线健康")
    
    print()
    
    # 退出码: 有FAIL返回1
    sys.exit(1 if n_fail > 0 else 0)

if __name__ == "__main__":
    main()

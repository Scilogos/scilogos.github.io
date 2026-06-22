"""
run_pipeline.py - 对抗学习量化系统 总管线
==========================================
串联四阶段: 数据获取 → 生成器对抗 → 庄散对抗 → 结果分析
+ 实盘反馈: feedback.txt → 校准模型 → 重新训练

用法:
  python run_pipeline.py --phase all          # 全流程
  python run_pipeline.py --phase data         # 只跑数据
  python run_pipeline.py --phase generator    # 只跑生成器
  python run_pipeline.py --phase adversarial  # 只跑对抗
  python run_pipeline.py --phase interpret    # 只跑解读
  python run_pipeline.py --phase feedback     # 处理实盘反馈+校准
  python run_pipeline.py --phase 1-2          # 跑阶段1+2
  python run_pipeline.py --quick-test         # 快速验证(小数据量)
"""

import os, sys, argparse, time, json, subprocess
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent))
from stock_config import (
    DATA_DIR, ADV_MODEL_DIR, ADV_DATA_DIR, RESULTS_DIR,
    setup_logger, PYTHON_EXE,
)

logger = setup_logger("Pipeline")

SCRIPTS_DIR = Path(__file__).parent

def run_cmd(cmd: str, desc: str) -> bool:
    """执行命令并报告结果"""
    logger.info(f"\n{'='*50}")
    logger.info(f"▶ {desc}")
    logger.info(f"  命令: {cmd}")
    logger.info(f"{'='*50}")
    
    start = time.time()
    result = subprocess.run(cmd, shell=True, capture_output=False, text=True)
    elapsed = time.time() - start
    
    if result.returncode == 0:
        logger.info(f"✓ {desc} 完成 ({elapsed:.1f}s)")
        return True
    else:
        logger.error(f"✗ {desc} 失败 (exit code {result.returncode})")
        return False

# ============================================================
# Phase 1: 数据获取
# ============================================================
def phase_data(args):
    """数据获取阶段"""
    logger.info("\n" + "█" * 60)
    logger.info("  Phase 1: 数据获取")
    logger.info("█" * 60)
    
    steps = []
    
    # 1.1 扫描
    if not (DATA_DIR / "stock_list.json").exists():
        steps.append((
            f'cd {SCRIPTS_DIR} && {PYTHON_EXE} stock_data_manager.py --mode scan --data-dir "{DATA_DIR}"',
            "1.1 全市场扫描"
        ))
    else:
        logger.info("  跳过1.1 (stock_list.json已存在)")
    
    # 1.2 下载日K线
    csv_count = len(list(DATA_DIR.glob("*_daily.csv")))
    if csv_count < 4000:
        max_stocks = f"--max-stocks {args.max_stocks}" if args.max_stocks > 0 else ""
        steps.append((
            f'cd {SCRIPTS_DIR} && {PYTHON_EXE} stock_data_manager.py --mode download-daily '
            f'--data-dir "{DATA_DIR}" --resume {max_stocks}',
            "1.2 下载日K线"
        ))
    else:
        logger.info(f"  跳过1.2 (已有{csv_count}只日K线)")
    
    # 1.3 行业分类
    if not (DATA_DIR / "industry_classification.json").exists():
        steps.append((
            f'cd {SCRIPTS_DIR} && {PYTHON_EXE} stock_data_manager.py --mode classify --data-dir "{DATA_DIR}"',
            "1.3 行业分类"
        ))
    else:
        logger.info("  跳过1.3 (industry_classification.json已存在)")
    
    # 1.4 数据校验
    steps.append((
        f'cd {SCRIPTS_DIR} && {PYTHON_EXE} stock_data_manager.py --mode verify --data-dir "{DATA_DIR}"',
        "1.4 数据校验"
    ))
    
    results = []
    for cmd, desc in steps:
        ok = run_cmd(cmd, desc)
        results.append((desc, ok))
        if not ok:
            logger.warning(f"阶段中断: {desc} 失败")
            break
    
    return all(ok for _, ok in results)

# ============================================================
# Phase 2: 生成器对抗
# ============================================================
def phase_generator(args):
    """生成器训练阶段"""
    logger.info("\n" + "█" * 60)
    logger.info("  Phase 2: 生成器对抗 (C-TimeGAN)")
    logger.info("█" * 60)
    
    # 训练
    max_stocks = f"--max-stocks {args.max_stocks}" if args.max_stocks > 0 else ""
    train_cmd = (
        f'cd {SCRIPTS_DIR} && {PYTHON_EXE} market_generator.py --mode train '
        f'--data-dir "{DATA_DIR}" {max_stocks} '
        f'--phase-a-epochs {args.phase_a_epochs} '
        f'--phase-b-epochs {args.phase_b_epochs} '
        f'--phase-c-epochs {args.phase_c_epochs} '
        f'--batch-size {args.batch_size} '
        f'--device {args.device}'
    )
    
    ok = run_cmd(train_cmd, "2.1 C-TimeGAN 训练")
    if not ok:
        return False
    
    # 验证
    validate_cmd = (
        f'cd {SCRIPTS_DIR} && {PYTHON_EXE} market_generator.py --mode validate '
        f'--data-dir "{DATA_DIR}" --level {args.validate_level} '
        f'--device {args.device}'
    )
    
    ok = run_cmd(validate_cmd, "2.2 三级验证")
    
    # 生成
    gen_cmd = (
        f'cd {SCRIPTS_DIR} && {PYTHON_EXE} market_generator.py --mode generate '
        f'--num-samples 1000 --device {args.device}'
    )
    
    run_cmd(gen_cmd, "2.3 生成样本")
    
    return ok

# ============================================================
# Phase 3: 庄散对抗
# ============================================================
def phase_adversarial(args):
    """庄散对抗阶段"""
    logger.info("\n" + "█" * 60)
    logger.info("  Phase 3: 庄散对抗（核心）")
    logger.info("█" * 60)
    
    cmd = (
        f'cd {SCRIPTS_DIR} && {PYTHON_EXE} adversarial_env.py --mode train '
        f'--episodes {args.episodes} '
        f'{"--evolve" if args.evolve else ""}'
    )
    
    ok = run_cmd(cmd, "3.1 对抗训练")
    if not ok:
        return False
    
    # 评估
    eval_cmd = (
        f'cd {SCRIPTS_DIR} && {PYTHON_EXE} adversarial_env.py --mode evaluate '
        f'--episodes 100'
    )
    
    run_cmd(eval_cmd, "3.2 评估")
    return ok

# ============================================================
# Phase 4: 结果分析
# ============================================================
def phase_interpret(args):
    """结果解读阶段"""
    logger.info("\n" + "█" * 60)
    logger.info("  Phase 4: 结果分析")
    logger.info("█" * 60)
    
    cmd = (
        f'cd {SCRIPTS_DIR} && {PYTHON_EXE} stock_interpreter.py --mode analyze '
        f'--data-dir "{DATA_DIR}" '
        f'--output "{RESULTS_DIR}/interpretation_report.json"'
    )
    
    ok = run_cmd(cmd, "4.1 四层解读")
    
    if ok:
        logger.info("\n解读报告摘要:")
        report_path = RESULTS_DIR / "interpretation_report.json"
        if report_path.exists():
            with open(report_path, 'r', encoding='utf-8') as f:
                report = json.load(f)
            logger.info(report.get('summary', '无摘要'))
    
    return ok

# ============================================================
# Phase 5: 实盘反馈
# ============================================================
def phase_feedback(args):
    """实盘反馈处理 + 模型校准"""
    logger.info("\n" + "█" * 60)
    logger.info("  Phase 5: 实盘反馈校准")
    logger.info("█" * 60)
    
    feedback_file = SCRIPT_DIR / "feedback.txt"
    
    if not feedback_file.exists():
        logger.warning("feedback.txt 不存在，先生成示例文件...")
        demo_cmd = (
            f'cd {SCRIPTS_DIR} && {PYTHON_EXE} feedback_processor.py --mode demo '
            f'--feedback-file "{feedback_file}"'
        )
        run_cmd(demo_cmd, "5.0 生成示例反馈文件")
        logger.info("请编辑 feedback.txt 填入真实交易记录后重新运行")
        return False
    
    # 全流程: 解析+评估+校准+报告
    cmd = (
        f'cd {SCRIPTS_DIR} && {PYTHON_EXE} feedback_processor.py --mode full '
        f'--feedback-file "{feedback_file}" '
        f'--results-dir "{RESULTS_DIR}"'
    )
    
    ok = run_cmd(cmd, "5.1 实盘反馈全流程")
    
    if ok:
        # 检查是否需要重训
        cal_file = RESULTS_DIR / "calibration_params.json"
        if cal_file.exists():
            with open(cal_file, 'r', encoding='utf-8') as f:
                cal = json.load(f)
            if cal.get('need_retrain'):
                logger.warning("\n⚠️ 校准结果建议重训对抗模型！")
                logger.warning(f"原因: {cal.get('retrain_reason', '')}")
                logger.warning("运行: python run_pipeline.py --phase adversarial --evolve")
    
    return ok

# ============================================================
# 快速验证
# ============================================================
def quick_test(args):
    """快速验证: 小数据量跑完整个管线"""
    logger.info("\n" + "!" * 60)
    logger.info("  快速验证模式 (小数据量)")
    logger.info("!" * 60)
    
    # 覆盖参数
    args.max_stocks = 50
    args.phase_a_epochs = 5
    args.phase_b_epochs = 5
    args.phase_c_epochs = 10
    args.episodes = 20
    args.evolve = True
    args.validate_level = 1
    
    logger.info("参数: max_stocks=50, epochs=5/5/10, episodes=20")
    
    # 顺序执行
    phases = [
        ("Phase 1", phase_data),
        ("Phase 2", phase_generator),
        ("Phase 3", phase_adversarial),
        ("Phase 4", phase_interpret),
    ]
    
    for name, fn in phases:
        logger.info(f"\n{'▶'*30} {name} {'◀'*30}")
        ok = fn(args)
        if not ok:
            logger.error(f"{name} 失败，管线中断")
            return False
    
    logger.info("\n" + "✓" * 60)
    logger.info("  快速验证完成！全管线通过")
    logger.info("✓" * 60)
    return True

# ============================================================
# 主入口
# ============================================================
def main():
    parser = argparse.ArgumentParser(description="对抗学习量化系统 总管线")
    parser.add_argument("--phase", type=str, default="all",
                        help="执行阶段: all/data/generator/adversarial/interpret/1-2/1-3/quick-test")
    parser.add_argument("--quick-test", action="store_true",
                        help="快速验证模式")
    
    # Phase 1 参数
    parser.add_argument("--max-stocks", type=int, default=0)
    
    # Phase 2 参数
    parser.add_argument("--phase-a-epochs", type=int, default=100)
    parser.add_argument("--phase-b-epochs", type=int, default=100)
    parser.add_argument("--phase-c-epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--validate-level", type=int, default=3)
    parser.add_argument("--device", type=str, default="auto")
    
    # Phase 3 参数
    parser.add_argument("--episodes", type=int, default=1000)
    parser.add_argument("--evolve", action="store_true")
    
    args = parser.parse_args()
    
    start_time = time.time()
    
    if args.quick_test or args.phase == "quick-test":
        quick_test(args)
    elif args.phase == "all":
        for fn in [phase_data, phase_generator, phase_adversarial, phase_interpret, phase_feedback]:
            if not fn(args):
                break
    elif args.phase == "data":
        phase_data(args)
    elif args.phase == "generator":
        phase_generator(args)
    elif args.phase == "adversarial":
        phase_adversarial(args)
    elif args.phase == "interpret":
        phase_interpret(args)
    elif args.phase == "feedback":
        phase_feedback(args)
    elif "-" in args.phase:
        # 多阶段: "1-2" = phase 1+2
        phase_map = {"1": phase_data, "2": phase_generator,
                    "3": phase_adversarial, "4": phase_interpret,
                    "5": phase_feedback}
        phases = args.phase.split("-")
        for p in phases:
            if p in phase_map:
                if not phase_map[p](args):
                    break
    
    elapsed = time.time() - start_time
    logger.info(f"\n总耗时: {elapsed/60:.1f} 分钟")

if __name__ == "__main__":
    main()

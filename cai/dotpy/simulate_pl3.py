#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
排列3模拟数据生成器（本地版）
生成与真实pl3_history.csv格式完全一致的模拟开奖数据
输出到 data/ 目录, split_info.json 也存 data/

用法:
  python simulate_pl3.py                      # 默认参数
  python simulate_pl3.py --clean               # 清空重新生成
  python simulate_pl3.py --split               # 同时生成分割信息
  python simulate_pl3.py --start-date 2021-01-01 --end-date 2025-12-31
"""

import argparse
import csv
import json
import os
import random
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
from scipy.stats import chisquare


# ==================== 本地路径配置 ====================
BASE_DIR = r"C:\Users\HUAWEI\Desktop\Adversarial Learning"
DATA_DIR = os.path.join(BASE_DIR, "data")


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='排列3模拟数据生成器（本地版）')
    parser.add_argument('--start-date', default='2020-01-01', help='起始日期 (默认2020-01-01)')
    parser.add_argument('--end-date', default='2025-12-31', help='结束日期 (默认2025-12-31)')
    parser.add_argument('--seed', type=int, default=42, help='随机种子 (默认42)')
    parser.add_argument('--output', default=None, help='输出文件名 (默认data/pl3_simulated.csv)')
    parser.add_argument('--clean', action='store_true', help='清空已有模拟数据重新生成')
    parser.add_argument('--split', action='store_true', help='生成分割信息文件split_info.json')
    return parser.parse_args()


def generate_period_number(date):
    """根据日期生成期号: 年份(2位)+年内序号(3位)"""
    year = date.year % 100
    day_of_year = date.timetuple().tm_yday
    return f"{year:02d}{day_of_year:03d}"


def generate_lottery_number():
    """生成一个3位彩票号码,每位独立均匀随机0-9"""
    return [random.randint(0, 9) for _ in range(3)]


def get_group_type(digits):
    """判断组选类型: 豹子/组三/组六"""
    if digits[0] == digits[1] == digits[2]:
        return '豹子'
    elif digits[0] == digits[1] or digits[1] == digits[2] or digits[0] == digits[2]:
        return '组三'
    else:
        return '组六'


def generate_data(start_date, end_date, seed, output_file, clean_mode):
    """生成模拟数据并保存为CSV"""
    random.seed(seed)
    np.random.seed(seed)

    start = datetime.strptime(start_date, '%Y-%m-%d')
    end = datetime.strptime(end_date, '%Y-%m-%d')

    # 处理增量/清空模式
    existing_periods = set()
    if not clean_mode and os.path.exists(output_file):
        try:
            existing_df = pd.read_csv(output_file)
            existing_periods = set(existing_df['期号'].astype(str))
        except (pd.errors.EmptyDataError, FileNotFoundError):
            pass

    # 生成新数据
    new_rows = []
    current = start
    while current <= end:
        period = generate_period_number(current)
        if period not in existing_periods:
            digits = generate_lottery_number()
            sales = random.randint(10000000, 50000000)  # 模拟销售额
            new_rows.append([
                period,
                current.strftime('%Y-%m-%d'),
                digits[0],
                digits[1],
                digits[2],
                sales
            ])
        current += timedelta(days=1)

    # 确保目录存在
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    # 写入CSV
    file_exists = os.path.exists(output_file) and not clean_mode
    mode = 'a' if file_exists else 'w'
    with open(output_file, mode, newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(['期号', '日期', '百位', '十位', '个位', '销售额'])
        writer.writerows(new_rows)

    print(f"生成完成: {len(new_rows)} 条新记录写入 {output_file}")
    return output_file


def quality_check(file_path):
    """质量校验: 均匀性卡方检验和组选类型占比"""
    df = pd.read_csv(file_path)

    # 各位数字均匀性卡方检验
    print("\n=== 质量校验结果 ===")
    positions = ['百位', '十位', '个位']
    all_pass = True
    for pos in positions:
        counts = df[pos].value_counts().sort_index()
        expected = [len(df) / 10] * 10
        stat, p_value = chisquare(counts, f_exp=expected)
        result = "通过" if p_value > 0.05 else "不通过"
        if p_value <= 0.05:
            all_pass = False
        print(f"{pos}位均匀性卡方检验: p={p_value:.4f} ({result})")

    # 组选类型占比
    df['组选类型'] = df.apply(lambda row: get_group_type([row['百位'], row['十位'], row['个位']]), axis=1)
    type_counts = df['组选类型'].value_counts(normalize=True)
    expected_ratios = {'组六': 0.711, '组三': 0.279, '豹子': 0.010}

    print("\n组选类型占比:")
    for t in ['组六', '组三', '豹子']:
        actual = type_counts.get(t, 0) * 100
        expected = expected_ratios[t] * 100
        diff = abs(actual - expected)
        status = "通过" if diff < 3 else "不通过"
        if diff >= 3:
            all_pass = False
        print(f"  {t}: 实际{actual:.2f}% vs 理论{expected:.2f}% (偏差{diff:.2f}%) [{status}]")

    # 连号率
    def has_consecutive(digits):
        sorted_d = sorted(digits)
        return (sorted_d[1] - sorted_d[0] == 1) or (sorted_d[2] - sorted_d[1] == 1)

    df['连号'] = df.apply(lambda row: has_consecutive([row['百位'], row['十位'], row['个位']]), axis=1)
    consecutive_rate = df['连号'].mean() * 100
    print(f"\n连号率: {consecutive_rate:.2f}% (理论约4.9%)")

    # 平均重号
    def count_repeated(digits):
        return 3 - len(set(digits))

    df['重号数'] = df.apply(lambda row: count_repeated([row['百位'], row['十位'], row['个位']]), axis=1)
    avg_repeat = df['重号数'].mean()
    print(f"平均重号数: {avg_repeat:.3f} (理论约0.717)")

    if all_pass:
        print("\n✅ 所有校验通过!")
    else:
        print("\n⚠️  部分校验未通过,请检查数据质量")


def generate_split_info(file_path, seed):
    """生成非等间隔的分割信息，保存到 data/split_info.json"""
    random.seed(seed + 1000)  # 使用不同种子避免与数据生成冲突
    df = pd.read_csv(file_path)
    df['日期'] = pd.to_datetime(df['日期'])
    df = df.sort_values('日期')

    total_days = (df['日期'].max() - df['日期'].min()).days
    splits = []
    current_start = df['日期'].min()

    while True:
        # 随机2-4个月作为训练集
        train_months = random.randint(2, 4)
        train_days = train_months * 30
        train_end = current_start + timedelta(days=train_days)

        # 随机1周作为测试集
        test_days = 7
        test_start = train_end + timedelta(days=1)
        test_end = test_start + timedelta(days=test_days - 1)

        # 检查是否超出范围
        if test_end > df['日期'].max():
            if current_start <= df['日期'].max():
                splits.append({
                    'train_start': current_start.strftime('%Y-%m-%d'),
                    'train_end': df['日期'].max().strftime('%Y-%m-%d'),
                    'test_start': None,
                    'test_end': None
                })
            break

        splits.append({
            'train_start': current_start.strftime('%Y-%m-%d'),
            'train_end': train_end.strftime('%Y-%m-%d'),
            'test_start': test_start.strftime('%Y-%m-%d'),
            'test_end': test_end.strftime('%Y-%m-%d')
        })

        current_start = test_end + timedelta(days=1)

    # 保存到data目录
    split_path = os.path.join(DATA_DIR, 'split_info.json')
    with open(split_path, 'w', encoding='utf-8') as f:
        json.dump(splits, f, ensure_ascii=False, indent=2)
    print(f"\n分割信息已保存到 {split_path} ({len(splits)} 个分割块)")


def main():
    args = parse_args()

    # 默认输出路径
    if args.output is None:
        args.output = os.path.join(DATA_DIR, 'pl3_simulated.csv')

    print(f"排列3模拟数据生成器（本地版）")
    print(f"参数: start={args.start_date}, end={args.end_date}, seed={args.seed}")

    # 生成数据
    output_file = generate_data(
        args.start_date,
        args.end_date,
        args.seed,
        args.output,
        args.clean
    )

    # 质量校验
    quality_check(output_file)

    # 生成分割信息
    if args.split:
        generate_split_info(output_file, args.seed)


def batch_generate(seeds=None, start_date='2020-01-01', end_date='2025-12-31',
                   clean=True, run_quality=True):
    """批量生成多组不同种子的模拟数据

    Args:
        seeds: 种子列表，默认[42,123,456,789,1024,2048,3141,4096,5555,9999]
        start_date: 起始日期
        end_date: 结束日期
        clean: 是否清空旧数据重新生成
        run_quality: 是否运行质量校验

    Returns:
        dict: {seed: output_file_path}
    """
    if seeds is None:
        seeds = [42, 123, 456, 789, 1024, 2048, 3141, 4096, 5555, 9999]

    results = {}
    print(f"批量生成模式: {len(seeds)}组模拟数据")
    print(f"参数: start={start_date}, end={end_date}, clean={clean}")

    for i, seed in enumerate(seeds):
        output = os.path.join(DATA_DIR, f'pl3_sim_seed{seed}.csv')
        print(f"\n[{i+1}/{len(seeds)}] seed={seed}")

        # 生成数据
        output_file = generate_data(
            start_date, end_date, seed, output, clean
        )
        results[seed] = output_file

        # 质量校验
        if run_quality:
            quality_check(output_file)

    print(f"\n✅ 批量生成完成: {len(results)}组")
    for seed, path in results.items():
        print(f"  seed={seed} → {path}")

    return results


if __name__ == '__main__':
    main()

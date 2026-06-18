#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
fetch_history.py - 福彩3D & 体彩排列3 历史数据采集脚本
云端版：福彩3D全量 + 排列3优先体彩API/降级500走势图
本地版：同上，体彩API本地大概率可用可获全量数据

用法:
  python fetch_history.py            # 增量追加
  python fetch_history.py --refresh  # 全量刷新
"""

import requests
import pandas as pd
import time
import os
import argparse
import re

# ==================== 配置 ====================
FC3D_CSV = "fc3d_history.csv"
PL3_CSV = "pl3_history.csv"
UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
      "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36")
TIMEOUT = 15
MAX_RETRIES = 3
RETRY_DELAY = 2


# ==================== 工具函数 ====================
def safe_get(url, params=None, headers=None, retries=MAX_RETRIES):
    """带重试的GET请求"""
    for attempt in range(retries):
        try:
            resp = requests.get(url, params=params, headers=headers, timeout=TIMEOUT)
            resp.raise_for_status()
            return resp
        except Exception as e:
            if attempt < retries - 1:
                print(f"  ⚠️ 请求失败, 重试 {attempt+1}/{retries}: {e}")
                time.sleep(RETRY_DELAY)
            else:
                print(f"  ❌ 请求最终失败: {e}")
                return None


def load_existing_issues(csv_path):
    """读取已有CSV的期号集合"""
    if not os.path.exists(csv_path):
        return set()
    try:
        df = pd.read_csv(csv_path, dtype={"期号": str})
        return set(df["期号"].tolist())
    except Exception:
        return set()


def save_csv(df, csv_path, mode="a"):
    """保存CSV，自动去重"""
    if mode == "w" or not os.path.exists(csv_path):
        df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    else:
        existing = pd.read_csv(csv_path, dtype={"期号": str})
        combined = pd.concat([existing, df], ignore_index=True)
        combined.drop_duplicates(subset=["期号"], keep="last", inplace=True)
        combined.sort_values("期号", inplace=True)
        combined.to_csv(csv_path, index=False, encoding="utf-8-sig")


# ==================== 福彩3D（cwl.gov.cn ✅云端可用）====================
def fetch_fc3d_all():
    """全量获取福彩3D历史数据"""
    url = "https://www.cwl.gov.cn/cwl_admin/front/cwlkj/search/kjxx/findDrawNotice"
    params = {"name": "3d", "issueCount": 5000}
    headers = {"User-Agent": UA, "Referer": "https://www.cwl.gov.cn/"}

    print("📡 福彩3D: 请求官方API...")
    resp = safe_get(url, params=params, headers=headers)
    if not resp:
        return []

    try:
        records = resp.json().get("result", [])
    except Exception as e:
        print(f"  ❌ 解析JSON失败: {e}")
        return []

    results = []
    for item in records:
        try:
            code = item.get("code", "")
            date_raw = item.get("date", "")
            red = item.get("red", "")
            sales = item.get("sales", "")

            # "2026-06-17(三)" -> "2026-06-17"
            date = date_raw.split("(")[0] if "(" in date_raw else date_raw
            # "1,7,8" -> 百位,十位,个位
            nums = red.split(",")
            if len(nums) != 3:
                continue

            results.append({
                "期号": code,
                "日期": date,
                "百位": nums[0].strip(),
                "十位": nums[1].strip(),
                "个位": nums[2].strip(),
                "销售额": sales
            })
        except Exception:
            continue

    print(f"  ✅ 福彩3D: 获取 {len(results)} 条记录 "
          f"({results[-1]['期号']}~{results[0]['期号']})" if results else "  ❌ 无数据")
    return results


# ==================== 排列3 ====================
def fetch_pl3_sporttery():
    """用体彩官方API全量获取排列3（需翻页，云端可能567被封）"""
    base_url = "https://webapi.sporttery.cn/gateway/lottery/getHistoryPageListV1.qry"
    headers = {"User-Agent": UA, "Referer": "https://www.sporttery.cn/"}
    all_records = []
    page_no = 1

    while True:
        params = {
            "gameNo": "35",
            "provinceId": "0",
            "pageSize": "100",
            "isVerify": "1",
            "pageNo": str(page_no),
        }

        resp = safe_get(base_url, params=params, headers=headers)
        if not resp:
            return None  # None = API不可用，触发降级

        try:
            data = resp.json()
        except Exception:
            return None

        # 体彩API返回: {"value": {"list": [...], "pages": N, "total": N}, ...}
        value = data.get("value", {})
        if not value and "data" in data:
            value = data["data"].get("value", {})

        if not value:
            return None

        records = value.get("list", [])
        if not records:
            break

        # 翻页字段是pages不是totalPages
        total_pages = value.get("pages", value.get("totalPages", 0))
        if page_no == 1:
            print(f"  📊 排列3: 共 {total_pages} 页, {value.get('total', '?')} 条")

        for item in records:
            try:
                draw_result = item.get("lotteryDrawResult", "")
                nums = draw_result.split()
                if len(nums) != 3:
                    continue
                all_records.append({
                    "期号": item.get("lotteryDrawNum", ""),
                    "日期": item.get("lotteryDrawTime", "")[:10],
                    "百位": nums[0],
                    "十位": nums[1],
                    "个位": nums[2],
                    "销售额": ""
                })
            except Exception:
                continue

        print(f"  第 {page_no}/{total_pages} 页, 已获取 {len(all_records)} 条")

        if page_no >= total_pages:
            break
        page_no += 1
        time.sleep(0.5)

    return all_records


def fetch_pl3_500chart():
    """500彩票网走势图备用源（仅最近约30条，正则解析无需bs4）"""
    url = "https://datachart.500.com/pls/zoushi/jbzs.shtml"
    headers = {"User-Agent": UA, "Referer": "https://datachart.500.com/"}

    print("📡 排列3: 降级使用500走势图（仅最近约30条）...")
    resp = safe_get(url, headers=headers)
    if not resp:
        return []

    # gb2312编码
    try:
        html = resp.content.decode("gb2312", errors="ignore")
    except Exception:
        html = resp.text

    results = []
    trs = re.findall(r"<tr[^>]*>(.*?)</tr>", html, re.DOTALL)
    for tr in trs:
        tds = re.findall(r"<td[^>]*>(.*?)</td>", tr, re.DOTALL)
        if len(tds) < 5:
            continue
        clean = [re.sub(r"<[^>]+>", "", td).strip() for td in tds[:5]]
        # 验证: 期号5位数字 + 3个单数字
        if (len(clean[1]) == 5 and clean[1].isdigit()
                and clean[2].isdigit() and clean[3].isdigit() and clean[4].isdigit()):
            results.append({
                "期号": clean[1],
                "日期": clean[0],
                "百位": clean[2],
                "十位": clean[3],
                "个位": clean[4],
                "销售额": ""
            })

    print(f"  ⚠️ 500走势图: 获取 {len(results)} 条（非全量）")
    return results


def fetch_pl3_all():
    """排列3全量：优先体彩API → 降级500走势图"""
    print("📡 排列3: 尝试体彩官方API...")
    data = fetch_pl3_sporttery()

    if data is not None and len(data) > 0:
        print(f"  ✅ 体彩API成功: {len(data)} 条")
        return data

    # 降级
    data = fetch_pl3_500chart()
    return data


# ==================== 主逻辑 ====================
def main():
    parser = argparse.ArgumentParser(description="彩票历史数据采集")
    parser.add_argument("--refresh", action="store_true", help="全量刷新（默认增量追加）")
    args = parser.parse_args()

    # ===== 福彩3D =====
    print("=" * 50)
    fc3d_data = fetch_fc3d_all()
    if fc3d_data:
        df = pd.DataFrame(fc3d_data)
        if args.refresh:
            save_csv(df, FC3D_CSV, mode="w")
            print(f"  💾 福彩3D全量刷新: {len(df)} 条 → {FC3D_CSV}")
        else:
            existing = load_existing_issues(FC3D_CSV)
            new_data = [d for d in fc3d_data if d["期号"] not in existing]
            if new_data:
                save_csv(pd.DataFrame(new_data), FC3D_CSV, mode="a")
                print(f"  💾 福彩3D增量更新: 新增 {len(new_data)} 条")
            else:
                print("  ✅ 福彩3D数据已是最新")

    # ===== 排列3 =====
    print("=" * 50)
    pl3_data = fetch_pl3_all()
    if pl3_data:
        df = pd.DataFrame(pl3_data)
        if args.refresh:
            save_csv(df, PL3_CSV, mode="w")
            print(f"  💾 排列3全量刷新: {len(df)} 条 → {PL3_CSV}")
        else:
            existing = load_existing_issues(PL3_CSV)
            new_data = [d for d in pl3_data if d["期号"] not in existing]
            if new_data:
                save_csv(pd.DataFrame(new_data), PL3_CSV, mode="a")
                print(f"  💾 排列3增量更新: 新增 {len(new_data)} 条")
            else:
                print("  ✅ 排列3数据已是最新")

    print("=" * 50)
    print("🏁 采集完成")


if __name__ == "__main__":
    main()

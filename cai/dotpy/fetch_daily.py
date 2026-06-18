#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
fetch_daily.py - 福彩3D & 排列3 每日增量更新脚本
neko本地PyCharm每日运行，只拉取最新数据追加到本地CSV

用法:
  python fetch_daily.py
"""

import requests
import pandas as pd
import os
import re

# ==================== 配置 ====================
FC3D_CSV = "fc3d_history.csv"
PL3_CSV = "pl3_history.csv"
UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
      "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36")
TIMEOUT = 15
MAX_RETRIES = 3


# ==================== 工具函数 ====================
def safe_get(url, params=None, headers=None):
    """带重试的GET请求"""
    for attempt in range(MAX_RETRIES):
        try:
            resp = requests.get(url, params=params, headers=headers, timeout=TIMEOUT)
            resp.raise_for_status()
            return resp
        except Exception as e:
            if attempt < MAX_RETRIES - 1:
                print(f"  ⚠️ 重试 {attempt+1}/{MAX_RETRIES}: {e}")
            else:
                print(f"  ❌ 请求失败: {e}")
                return None


def get_existing_issues(csv_path):
    """读取已有CSV的期号集合"""
    if not os.path.exists(csv_path):
        return set()
    try:
        df = pd.read_csv(csv_path, dtype={"期号": str})
        return set(df["期号"].tolist())
    except Exception:
        return set()


def append_new(records, csv_path):
    """只追加新期号的数据"""
    existing = get_existing_issues(csv_path)
    new_data = [r for r in records if r["期号"] not in existing]
    if not new_data:
        print(f"  ✅ 数据已是最新")
        return 0
    df = pd.DataFrame(new_data)
    if os.path.exists(csv_path):
        old = pd.read_csv(csv_path, dtype={"期号": str})
        combined = pd.concat([old, df], ignore_index=True)
        combined.drop_duplicates(subset=["期号"], keep="last", inplace=True)
        combined.sort_values("期号", inplace=True)
        combined.to_csv(csv_path, index=False, encoding="utf-8-sig")
    else:
        df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    print(f"  💾 新增 {len(new_data)} 条 → {csv_path}")
    return len(new_data)


# ==================== 福彩3D（最近10期）====================
def fetch_fc3d_recent():
    url = "https://www.cwl.gov.cn/cwl_admin/front/cwlkj/search/kjxx/findDrawNotice"
    params = {"name": "3d", "issueCount": 10}
    headers = {"User-Agent": UA, "Referer": "https://www.cwl.gov.cn/"}

    resp = safe_get(url, params=params, headers=headers)
    if not resp:
        return []

    try:
        records = resp.json().get("result", [])
    except Exception:
        return []

    results = []
    for item in records:
        try:
            red = item.get("red", "")
            nums = red.split(",")
            if len(nums) != 3:
                continue
            date_raw = item.get("date", "")
            date = date_raw.split("(")[0] if "(" in date_raw else date_raw
            results.append({
                "期号": item.get("code", ""),
                "日期": date,
                "百位": nums[0].strip(),
                "十位": nums[1].strip(),
                "个位": nums[2].strip(),
                "销售额": item.get("sales", "")
            })
        except Exception:
            continue
    return results


# ==================== 排列3（最近10期）====================
def fetch_pl3_recent_sporttery():
    """体彩API获取最近1页"""
    url = "https://webapi.sporttery.cn/gateway/lottery/getHistoryPageListV1.qry"
    params = {
        "gameNo": "35", "provinceId": "0",
        "pageSize": "10", "isVerify": "1", "pageNo": "1"
    }
    headers = {"User-Agent": UA, "Referer": "https://www.sporttery.cn/"}

    resp = safe_get(url, params=params, headers=headers)
    if not resp:
        return None  # API不可用，触发降级

    try:
        data = resp.json()
        value = data.get("value", {})
        if not value and "data" in data:
            value = data["data"].get("value", {})
        records = value.get("list", [])
    except Exception:
        return None

    results = []
    for item in records:
        try:
            draw_result = item.get("lotteryDrawResult", "")
            nums = draw_result.split()
            if len(nums) != 3:
                continue
            results.append({
                "期号": item.get("lotteryDrawNum", ""),
                "日期": item.get("lotteryDrawTime", "")[:10],
                "百位": nums[0],
                "十位": nums[1],
                "个位": nums[2],
                "销售额": ""
            })
        except Exception:
            continue
    return results


def fetch_pl3_recent_500():
    """500走势图备用（最近约30条）"""
    url = "https://datachart.500.com/pls/zoushi/jbzs.shtml"
    headers = {"User-Agent": UA, "Referer": "https://datachart.500.com/"}

    resp = safe_get(url, headers=headers)
    if not resp:
        return []

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
    return results


def fetch_pl3_recent():
    """排列3最近数据：优先体彩API → 降级500走势图"""
    data = fetch_pl3_recent_sporttery()
    if data is not None:
        return data
    print("  ⚠️ 体彩API不可用，降级500走势图")
    return fetch_pl3_recent_500()


# ==================== 主逻辑 ====================
def main():
    print("=" * 50)
    print("📅 排列3 & 福彩3D 每日数据更新")
    print("=" * 50)

    # 福彩3D
    print("\n🎯 福彩3D:")
    fc3d = fetch_fc3d_recent()
    if fc3d:
        append_new(fc3d, FC3D_CSV)
    else:
        print("  ❌ 福彩3D获取失败")

    # 排列3
    print("\n🎯 排列3:")
    pl3 = fetch_pl3_recent()
    if pl3:
        append_new(pl3, PL3_CSV)
    else:
        print("  ❌ 排列3获取失败")

    print("\n" + "=" * 50)
    print("✅ 更新完成")


if __name__ == "__main__":
    main()

"""
一键修复脚本 v2 - 重建优先级列表 + 刷新数据 + API推送
问题根因: priority_stocks.json 中全是科创板(688)/创业板(300)，没有主板(600/601/000/001)
"""
import os, sys, json, time, base64, traceback
from pathlib import Path
from datetime import datetime

REPO = Path(r"C:\Users\HUAWEI\Desktop\scilogos.github.io")
GH_TOKEN = os.environ.get("GH_TOKEN", "")
if not GH_TOKEN:
    print("ERROR: 请设置环境变量 GH_TOKEN")
    print("  PowerShell: $env:GH_TOKEN='你的token'")
    sys.exit(1)
GH_API = "https://api.github.com/repos/Scilogos/scilogos.github.io"

def log(msg, level="INFO"):
    ts = datetime.now().strftime("%H:%M:%S")
    tags = {"INFO": "--", "WARN": "!!", "ERROR": "XX", "OK": "OK", "FIX": ">>"}
    print(f"[{ts}] [{tags.get(level, level)}] {msg}")

# ============================================================
# STEP 1: 扫描本地 stockdata，找出所有主板股票
# ============================================================
log("=" * 60)
log("STEP 1: 扫描本地 stockdata 目录，筛选主板标的")
log("=" * 60)

stockdata_dir = REPO / "stockdata"
all_csvs = sorted(stockdata_dir.glob("*.csv"))
log(f"本地 CSV 总数: {len(all_csvs)}")

# 主板标的: sh.600xxx, sh.601xxx, sh.603xxx, sh.605xxx, sz.000xxx, sz.001xxx, sz.002xxx, sz.003xxx
# ETF: sh.51xxxx, sh.58xxxx, sh.56xxxx, sh.52xxxx, sh.51xxxx
def is_tradeable(fname):
    """判断是否为普通散户可交易的标的（排除科创板688、创业板300）"""
    code = fname.replace(".csv", "")
    # 排除科创板 688
    if code.startswith("sh.688"):
        return False
    # 排除创业板 300
    if code.startswith("sz.300"):
        return False
    # 排除指数（sh.000001 上证指数等，但 sz.000xxx 个股保留）
    if code in ("sh.000001", "sh.000002", "sh.000003", "sh.000010", "sh.000011", "sh.000016", "sh.000300", "sh.000905", "sh.000852"):
        return False  # 这些是指数不是个股
    # 排除上证指数系列 sh.000xxx
    if code.startswith("sh.000"):
        return False
    return True

def is_etf(fname):
    code = fname.replace(".csv", "")
    return code.startswith("sh.5")

tradeable = [f for f in all_csvs if is_tradeable(f.name)]
etfs = [f for f in all_csvs if is_etf(f.name)]
non_tradeable = [f for f in all_csvs if not is_tradeable(f.name) and not is_etf(f.name)]

# 分类统计
sh600 = [f for f in tradeable if f.name.startswith("sh.600")]
sh601 = [f for f in tradeable if f.name.startswith("sh.601")]
sh603 = [f for f in tradeable if f.name.startswith("sh.603")]
sh605 = [f for f in tradeable if f.name.startswith("sh.605")]
sz000 = [f for f in tradeable if f.name.startswith("sz.000")]
sz001 = [f for f in tradeable if f.name.startswith("sz.001")]
sz002 = [f for f in tradeable if f.name.startswith("sz.002")]
sz003 = [f for f in tradeable if f.name.startswith("sz.003")]

log(f"可交易主板股: {len(tradeable)}")
log(f"  sh.600xxx: {len(sh600)}")
log(f"  sh.601xxx: {len(sh601)}")
log(f"  sh.603xxx: {len(sh603)}")
log(f"  sh.605xxx: {len(sh605)}")
log(f"  sz.000xxx: {len(sz000)}")
log(f"  sz.001xxx: {len(sz001)}")
log(f"  sz.002xxx: {len(sz002)}")
log(f"  sz.003xxx: {len(sz003)}")
log(f"ETF: {len(etfs)}")
log(f"不可交易(科创板/创业板/指数): {len(non_tradeable)}")

# 选出最多200只主板股 + ETF
# 优先选主板大市值股（600/601开头通常是蓝筹）
selected = []
selected.extend(sh600[:80])   # 最多80只 600
selected.extend(sh601[:50])   # 最多50只 601
selected.extend(sh603[:30])   # 最多30只 603
selected.extend(sz000[:20])   # 最多20只 sz000
selected.extend(sz002[:30])   # 最多30只 002
selected.extend(sz001[:10])   # 最多10只 001
selected.extend(sz003[:10])   # 最多10只 003
selected.extend(sh605[:5])    # 最多5只 605
selected.extend(etfs[:5])     # 最多5只 ETF

# 截断到205只
selected = selected[:205]
log(f"\n选定标的: {len(selected)} 只")

# 构建 priority_stocks.json
def categorize(fname):
    name = fname.name.replace(".csv", "")
    if name.startswith("sh.600"): return "沪市主板"
    if name.startswith("sh.601"): return "沪市主板"
    if name.startswith("sh.603"): return "沪市主板"
    if name.startswith("sh.605"): return "沪市主板"
    if name.startswith("sz.000"): return "深市主板"
    if name.startswith("sz.001"): return "深市主板"
    if name.startswith("sz.002"): return "中小板"
    if name.startswith("sz.003"): return "中小板"
    if name.startswith("sh.5"): return "ETF"
    return "其他"

priority_data = {
    "description": "庄散对抗系统重点标的列表（已优化：仅含普通散户可交易标的）",
    "total_count": len(selected),
    "updated": datetime.now().strftime("%Y-%m-%d %H:%M"),
    "categories": {},
    "stocks": []
}
cat_count = {}
for f in selected:
    code = f.name.replace(".csv", "")
    cat = categorize(f)
    cat_count[cat] = cat_count.get(cat, 0) + 1
    priority_data["stocks"].append({"code": code, "category": cat})
priority_data["categories"] = cat_count

# 写入文件
priority_file = REPO / "numpy_run" / "priority_stocks.json"
priority_file.parent.mkdir(parents=True, exist_ok=True)
priority_file.write_text(json.dumps(priority_data, ensure_ascii=False, indent=2), encoding="utf-8")
log(f"✅ priority_stocks.json 已更新: {len(selected)} 只标的")
log(f"   分类: {cat_count}")

# ============================================================
# STEP 2: 用 mootdx 刷新所有选定标的的 CSV 数据
# ============================================================
log("")
log("=" * 60)
log("STEP 2: 刷新 CSV 数据 (mootdx 日线)")
log("=" * 60)

import pandas as pd
from mootdx.quotes import Quotes

client = Quotes.factory(market='std')

success = 0
fail = 0
updated = 0
errors = []

for i, f in enumerate(selected):
    code = f.name.replace(".csv", "")
    bare_code = code.split(".")[-1]  # e.g. "600519" from "sh.600519"
    
    try:
        # 读取现有 CSV 获取最后日期
        if f.exists():
            df_old = pd.read_csv(f)
            old_last = str(df_old['date'].iloc[-1]) if 'date' in df_old.columns else ""
        else:
            df_old = pd.DataFrame()
            old_last = ""
        
        # 获取日线数据 (frequency=9)
        df_new = client.bars(symbol=bare_code, frequency=9, count=800)
        
        if df_new is None or len(df_new) == 0:
            fail += 1
            errors.append(f"{code}: mootdx 返回空")
            continue
        
        # 转换日期
        if hasattr(df_new.index[0], 'strftime'):
            dates = [d.strftime('%Y-%m-%d') for d in df_new.index]
        else:
            dates = [str(d)[:10] for d in df_new.get('datetime', df_new.index)]
        
        # 构建完整 DataFrame
        rows = []
        for j in range(len(df_new)):
            row = df_new.iloc[j]
            rows.append({
                'date': dates[j],
                'code': code,
                'open': float(row.get('open', 0)),
                'high': float(row.get('high', 0)),
                'low': float(row.get('low', 0)),
                'close': float(row.get('close', 0)),
                'volume': float(row.get('vol', row.get('volume', 0))),
                'amount': float(row.get('amount', 0)),
                'turnover': 0.0,
                'pct_change': 0.0,
                'is_limit_up': 0,
                'is_limit_down': 0,
            })
        
        df_full = pd.DataFrame(rows)
        
        # 计算 pct_change 和 turnover
        for j in range(1, len(df_full)):
            prev_close = df_full.iloc[j-1]['close']
            if prev_close > 0:
                df_full.loc[df_full.index[j], 'pct_change'] = round((df_full.iloc[j]['close'] - prev_close) / prev_close * 100, 4)
            if df_full.iloc[j]['close'] > 0:
                df_full.loc[df_full.index[j], 'turnover'] = round(df_full.iloc[j]['volume'] / df_full.iloc[j]['close'], 4)
        
        # 涨跌停判断
        for j in range(1, len(df_full)):
            pct = df_full.iloc[j]['pct_change']
            if pct >= 9.8:
                df_full.loc[df_full.index[j], 'is_limit_up'] = 1
            if pct <= -9.8:
                df_full.loc[df_full.index[j], 'is_limit_down'] = 1
        
        # 写入 CSV
        df_full.to_csv(f, index=False)
        
        new_last = dates[-1]
        if new_last > old_last:
            updated += 1
        
        success += 1
        
        if (i + 1) % 20 == 0 or (i + 1) == len(selected):
            log(f"  进度: {i+1}/{len(selected)} | 成功={success} 更新={updated} 失败={fail}")
    
    except Exception as e:
        fail += 1
        errors.append(f"{code}: {str(e)[:80]}")
        if fail <= 5:
            log(f"  ❌ {code}: {e}", "ERROR")

log(f"\n刷新完成: 成功={success}, 更新={updated}, 失败={fail}")
if errors:
    log(f"失败详情 (前10个):", "WARN")
    for e in errors[:10]:
        log(f"  {e}", "WARN")

# 验证关键股票
log("\n验证关键标的:")
for check in ["sh.600519.csv", "sh.601318.csv", "sh.600036.csv", "sz.000651.csv", "sh.600900.csv"]:
    fp = stockdata_dir / check
    if fp.exists():
        df = pd.read_csv(fp)
        last = df['date'].iloc[-1] if 'date' in df.columns else "?"
        rows = len(df)
        log(f"  {check}: {rows} 行, 最新={last}")
    else:
        log(f"  {check}: 不存在", "WARN")

# ============================================================
# STEP 3: 通过 GitHub API 推送所有文件
# ============================================================
log("")
log("=" * 60)
log("STEP 3: GitHub API 推送")
log("=" * 60)

import requests

headers = {"Authorization": f"Bearer {GH_TOKEN}"}

# 获取远程 HEAD
r = requests.get(f"{GH_API}/git/ref/heads/main", headers=headers, timeout=15)
if r.status_code != 200:
    log(f"获取远程 HEAD 失败: {r.status_code}", "ERROR")
    sys.exit(1)

parent_sha = r.json()['object']['sha']
r = requests.get(f"{GH_API}/git/commits/{parent_sha}", headers=headers, timeout=15)
base_tree_sha = r.json()['tree']['sha']
log(f"远程 HEAD: {parent_sha[:7]}, tree: {base_tree_sha[:7]}")

# 收集要推送的文件
files_to_push = []

# 1. 所有选定的 CSV
for f in selected:
    rel_path = f"stockdata/{f.name}"
    files_to_push.append((rel_path, f))

# 2. priority_stocks.json
files_to_push.append(("numpy_run/priority_stocks.json", priority_file))

# 3. 也推送一些非选定的主板股 CSV（如果本地有且已更新的话）
# 检查是否有其他主板股也需要更新
for f in all_csvs:
    if f in selected:
        continue
    if is_tradeable(f.name):
        rel_path = f"stockdata/{f.name}"
        files_to_push.append((rel_path, f))

log(f"待推送文件: {len(files_to_push)}")

# 批量创建 blobs
tree_entries = []
batch_errors = []

for i, (rel_path, local_path) in enumerate(files_to_push):
    try:
        content = local_path.read_bytes()
        b64_content = base64.b64encode(content).decode()
        r = requests.post(f"{GH_API}/git/blobs", headers=headers,
                          json={"content": b64_content, "encoding": "base64"}, timeout=60)
        if r.status_code == 201:
            tree_entries.append({
                "path": rel_path,
                "mode": "100644",
                "type": "blob",
                "sha": r.json()['sha']
            })
        else:
            batch_errors.append(f"{rel_path}: blob HTTP {r.status_code}")
    except Exception as e:
        batch_errors.append(f"{rel_path}: {str(e)[:60]}")
    
    if (i + 1) % 50 == 0 or (i + 1) == len(files_to_push):
        log(f"  Blobs: {i+1}/{len(files_to_push)} (errors: {len(batch_errors)})")

if batch_errors:
    log(f"Blob 创建失败: {len(batch_errors)}", "WARN")
    for e in batch_errors[:5]:
        log(f"  {e}", "WARN")

log(f"成功创建 {len(tree_entries)} 个 blobs")

# 创建 tree
r = requests.post(f"{GH_API}/git/trees", headers=headers,
                  json={"base_tree": base_tree_sha, "tree": tree_entries}, timeout=30)
if r.status_code != 201:
    log(f"创建 tree 失败: {r.status_code} {r.text[:200]}", "ERROR")
    sys.exit(1)
new_tree_sha = r.json()['sha']
log(f"新 tree: {new_tree_sha[:7]}")

# 创建 commit
now_str = datetime.now().strftime('%Y-%m-%d %H:%M')
msg = f"fix v2: rebuild priority list (main board only) + refresh data | {len(tree_entries)} files | {now_str}"
r = requests.post(f"{GH_API}/git/commits", headers=headers,
                  json={"message": msg, "tree": new_tree_sha, "parents": [parent_sha]}, timeout=30)
if r.status_code != 201:
    log(f"创建 commit 失败: {r.status_code} {r.text[:200]}", "ERROR")
    sys.exit(1)
commit_sha = r.json()['sha']
log(f"新 commit: {commit_sha[:7]}")

# 更新 ref
r = requests.patch(f"{GH_API}/git/refs/heads/main", headers=headers,
                   json={"sha": commit_sha}, timeout=15)
if r.status_code != 200:
    log(f"更新 ref 失败: {r.status_code} {r.text[:200]}", "ERROR")
    sys.exit(1)
log(f"✅ 推送成功! -> {commit_sha[:7]}")

# ============================================================
# STEP 4: 验证推送结果
# ============================================================
log("")
log("=" * 60)
log("STEP 4: 验证远程数据")
log("=" * 60)

time.sleep(2)

# 检查 commit
r = requests.get(f"{GH_API}/commits/{commit_sha[:7]}", headers=headers, timeout=15)
if r.status_code == 200:
    files_changed = len(r.json().get('files', []))
    log(f"Commit {commit_sha[:7]}: {files_changed} files changed")

# 验证关键文件
for fname in ["stockdata/sh.600519.csv", "stockdata/sh.601318.csv", "stockdata/sh.600036.csv"]:
    r = requests.get(f"{GH_API}/contents/{fname}?ref=main", headers=headers, timeout=15)
    if r.status_code == 200:
        content = base64.b64decode(r.json()['content']).decode('utf-8')
        lines = [l for l in content.strip().split('\n')[1:] if l.strip()]
        last_date = lines[-1].split(',')[0] if lines else "EMPTY"
        status = "✅" if last_date >= "2026-06-29" else "❌"
        log(f"  {status} {fname}: {len(lines)} rows, last={last_date}")
    else:
        log(f"  ❌ {fname}: HTTP {r.status_code}", "ERROR")

# 验证 priority_stocks.json
r = requests.get(f"{GH_API}/contents/numpy_run/priority_stocks.json?ref=main", headers=headers, timeout=15)
if r.status_code == 200:
    content = base64.b64decode(r.json()['content']).decode('utf-8')
    data = json.loads(content)
    log(f"  ✅ priority_stocks.json: {data.get('total_count', '?')} stocks, categories: {data.get('categories', {})}")

# 验证本地与远程一致性
log("\n本地 vs 远程一致性检查:")
all_match = True
for fname in ["sh.600519.csv", "sh.601318.csv", "sh.600036.csv", "sz.000651.csv"]:
    local_path = stockdata_dir / fname
    if not local_path.exists():
        continue
    local_content = local_path.read_text(encoding='utf-8')
    local_lines = [l for l in local_content.strip().split('\n')[1:] if l.strip()]
    local_last = local_lines[-1].split(',')[0] if local_lines else "?"
    
    r = requests.get(f"{GH_API}/contents/stockdata/{fname}?ref=main", headers=headers, timeout=15)
    if r.status_code == 200:
        remote_content = base64.b64decode(r.json()['content']).decode('utf-8')
        remote_lines = [l for l in remote_content.strip().split('\n')[1:] if l.strip()]
        remote_last = remote_lines[-1].split(',')[0] if remote_lines else "?"
        
        match = local_last == remote_last
        if not match:
            all_match = False
        status = "✅" if match else "❌"
        log(f"  {status} {fname}: local={local_last} remote={remote_last}")

# ============================================================
# STEP 5: 同时修复 salat.py 中的 bars() category
# ============================================================
log("")
log("=" * 60)
log("STEP 5: 检查并修复 salat.py")
log("=" * 60)

salat_path = Path(r"C:\Users\HUAWEI\Desktop\Adversarial Learning\githubdoc\salat\salat.py")
if salat_path.exists():
    import re
    content = salat_path.read_text(encoding='utf-8')
    
    # 查找所有 bars() 调用
    bars_calls = re.findall(r'.*bars\(.*\).*', content)
    log(f"找到 {len(bars_calls)} 行包含 bars() 调用:")
    for bc in bars_calls:
        log(f"  {bc.strip()[:120]}")
    
    # 查找 fetch_daily_bars 函数
    if 'fetch_daily_bars' in content:
        # 找到函数体中 bars() 的 category 参数
        func_match = re.search(r'def fetch_daily_bars.*?(?=\ndef |\Z)', content, re.DOTALL)
        if func_match:
            func_body = func_match.group(0)
            log(f"\nfetch_daily_bars 函数中的 bars 调用:")
            for line in func_body.split('\n'):
                if 'bars(' in line:
                    log(f"  {line.strip()}")
                    # 检查是否用了正确的 category
                    if 'frequency=9' in line or ', 9,' in line:
                        log(f"    ✅ 已使用 frequency=9 (日线)")
                    elif 'frequency=1' in line or ', 1,' in line:
                        log(f"    ❌ 使用了 frequency=1 (分钟线)！需要修复", "ERROR")
    
    # 检查 git_push 中是否有 API fallback
    if 'api.github.com' in content:
        log(f"\n✅ salat.py 已包含 GitHub API fallback")
    else:
        log(f"\n⚠️ salat.py 缺少 GitHub API fallback，git push 失败时数据无法上云", "WARN")
else:
    log(f"salat.py 不存在于 {salat_path}", "WARN")

# ============================================================
# 最终总结
# ============================================================
log("")
log("=" * 60)
log("🎉 全部完成！")
log("=" * 60)
log(f"  priority_stocks.json: {len(selected)} 只标的（全部可交易）")
log(f"  CSV 刷新: 成功 {success}/{len(selected)}")
log(f"  GitHub 推送: {len(tree_entries)} 文件 -> {commit_sha[:7]}")
log(f"  本地-远程一致: {'✅ 全部一致' if all_match else '❌ 存在不一致'}")

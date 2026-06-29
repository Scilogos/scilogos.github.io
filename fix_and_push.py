#!/usr/bin/env python3
"""
一键修复脚本：刷新205只重点标的CSV数据 + GitHub API推送
用法：
  在PowerShell中：
  $env:GH_TOKEN="ghp_你的token"; python fix_and_push.py
"""
import os, sys, json, time, base64, requests, traceback
from datetime import datetime, timedelta
from pathlib import Path

# ============ 配置 ============
REPO_PATH = r"C:\Users\HUAWEI\Desktop\scilogos.github.io"
PRIORITY_FILE = os.path.join(REPO_PATH, "numpy_run", "priority_stocks.json")
STOCKDATA_DIR = os.path.join(REPO_PATH, "stockdata")
TOKEN = os.environ.get("GH_TOKEN", "")
API_BASE = "https://api.github.com/repos/Scilogos/scilogos.github.io"
HDR = {"Authorization": f"Bearer {TOKEN}", "Accept": "application/vnd.github.v3+json"}

if not TOKEN:
    print("ERROR: 请设置环境变量 GH_TOKEN")
    print("PowerShell: $env:GH_TOKEN='ghp_xxx'; python fix_and_push.py")
    sys.exit(1)

def log(msg):
    ts = datetime.now().strftime("%H:%M:%S")
    print(f"[{ts}] {msg}")

def get_market_prefix(code):
    if code.startswith(('6', '9', '5')):
        return 'sh'
    else:
        return 'sz'

# ============ STEP 1: 从mootdx拉取最新数据 ============
def refresh_csv_data(codes):
    from mootdx.quotes import Quotes
    client = Quotes.factory(market='std')
    success, fail, total = 0, 0, len(codes)
    
    for i, code in enumerate(codes):
        try:
            prefix = get_market_prefix(code)
            full_code = f"{prefix}.{code}"
            csv_path = os.path.join(STOCKDATA_DIR, f"{full_code}.csv")
            
            if not os.path.exists(csv_path):
                log(f"  SKIP {full_code}: file not found")
                fail += 1
                continue
            
            with open(csv_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            
            if not lines:
                fail += 1
                continue
            
            # 获取最后日期
            last_date = ""
            for line in reversed(lines):
                line = line.strip()
                if line and line[0] == '2':
                    last_date = line.split(",")[0]
                    break
            
            # 从mootdx拉取数据
            bars = client.bars(symbol=code, frequency=9, count=5)
            if bars is None or bars.empty:
                fail += 1
                continue
            
            new_rows = []
            for _, row in bars.iterrows():
                dt = str(row.get('date', ''))[:10]
                if dt > last_date:
                    o = round(float(row.get('open', 0)), 2)
                    h = round(float(row.get('high', 0)), 2)
                    l = round(float(row.get('low', 0)), 2)
                    c = round(float(row.get('close', 0)), 2)
                    v = int(float(row.get('vol', 0)))
                    a = round(float(row.get('amount', 0)), 2)
                    new_rows.append(f"{dt},{full_code},{o},{h},{l},{c},{v},{a},0,0,0,0")
            
            if new_rows:
                with open(csv_path, 'a', encoding='utf-8') as f:
                    for row in new_rows:
                        f.write(row + "\n")
                success += 1
            else:
                success += 1
                
            if (i+1) % 20 == 0:
                log(f"  进度: {i+1}/{total}")
        except Exception as e:
            fail += 1
            if fail <= 3:
                log(f"  ERROR {code}: {e}")
    
    log(f"CSV刷新完成: 成功{success} 失败{fail} 共{total}")
    return success, fail

# ============ STEP 2: GitHub API推送 ============
def push_via_api():
    log("获取远程最新commit...")
    r = requests.get(f"{API_BASE}/commits?sha=main&per_page=1", headers=HDR, timeout=15)
    if r.status_code != 200:
        log(f"ERROR: {r.status_code}"); return False
    remote = r.json()[0]
    parent_sha, tree_sha = remote["sha"], remote["commit"]["tree"]["sha"]
    log(f"远程HEAD: {parent_sha[:7]}")
    
    log("收集文件...")
    file_blobs = []
    
    # Priority CSVs
    if os.path.exists(PRIORITY_FILE):
        with open(PRIORITY_FILE, 'r', encoding='utf-8') as f:
            pdata = json.load(f)
        cnt = 0
        for s in pdata.get("stocks", []):
            code = s.get("code", "")
            if not code: continue
            fp = os.path.join(STOCKDATA_DIR, f"{code}.csv")
            if not os.path.exists(fp): continue
            try:
                content = base64.b64encode(open(fp, 'rb').read()).decode()
                rb = requests.post(f"{API_BASE}/git/blobs",
                    headers={**HDR, "Content-Type": "application/octet-stream"},
                    json={"content": content, "encoding": "base64"}, timeout=60)
                if rb.status_code == 201:
                    file_blobs.append({"path": f"stockdata/{code}.csv", "mode": "100644",
                                       "type": "blob", "sha": rb.json()["sha"]})
                    cnt += 1
                time.sleep(0.05)
            except: pass
        log(f"  CSV: {cnt} files")
    
    # priority_stocks.json
    if os.path.exists(PRIORITY_FILE):
        content = base64.b64encode(open(PRIORITY_FILE, 'rb').read()).decode()
        rb = requests.post(f"{API_BASE}/git/blobs",
            headers={**HDR, "Content-Type": "application/octet-stream"},
            json={"content": content, "encoding": "base64"}, timeout=60)
        if rb.status_code == 201:
            file_blobs.append({"path": "numpy_run/priority_stocks.json", "mode": "100644",
                               "type": "blob", "sha": rb.json()["sha"]})
    
    # realtime_data
    rt_dir = os.path.join(REPO_PATH, "realtime_data")
    if os.path.exists(rt_dir):
        rt_cnt = 0
        now = time.time()
        for root, dirs, files in os.walk(rt_dir):
            for fn in files:
                fp = os.path.join(root, fn)
                if now - os.path.getmtime(fp) < 172800:
                    rel = os.path.relpath(fp, REPO_PATH).replace("\\", "/")
                    if rel in [b["path"] for b in file_blobs]: continue
                    try:
                        content = open(fp, 'rb').read()
                        if len(content) > 5*1024*1024: continue
                        b64 = base64.b64encode(content).decode()
                        rb = requests.post(f"{API_BASE}/git/blobs",
                            headers={**HDR, "Content-Type": "application/octet-stream"},
                            json={"content": b64, "encoding": "base64"}, timeout=60)
                        if rb.status_code == 201:
                            file_blobs.append({"path": rel, "mode": "100644",
                                               "type": "blob", "sha": rb.json()["sha"]})
                            rt_cnt += 1
                    except: pass
        log(f"  Realtime: {rt_cnt} files")
    
    log(f"总计: {len(file_blobs)} files")
    if not file_blobs:
        log("无文件可上传"); return False
    
    # Create tree
    log("创建tree...")
    r = requests.post(f"{API_BASE}/git/trees", headers=HDR,
                      json={"base_tree": tree_sha, "tree": file_blobs}, timeout=60)
    if r.status_code != 201:
        log(f"tree失败: {r.status_code} {r.text[:200]}"); return False
    new_tree = r.json()["sha"]
    
    # Create commit
    ts = datetime.now().strftime("%Y-%m-%d %H:%M")
    msg = f"fix + data sync {ts} | {len(file_blobs)} files"
    log(f"创建commit: {msg}")
    r = requests.post(f"{API_BASE}/git/commits", headers=HDR,
                      json={"message": msg, "tree": new_tree, "parents": [parent_sha]}, timeout=30)
    if r.status_code != 201:
        log(f"commit失败: {r.status_code}"); return False
    new_commit = r.json()["sha"]
    
    # Update ref
    log("更新main分支...")
    r = requests.patch(f"{API_BASE}/git/refs/heads/main", headers=HDR,
                       json={"sha": new_commit}, timeout=15)
    if r.status_code == 200:
        log(f"✅ 推送成功! {new_commit[:7]}")
        return True
    else:
        log(f"ref更新失败: {r.status_code} {r.text[:200]}")
        return False

def verify():
    log("验证...")
    r = requests.get(f"{API_BASE}/commits?per_page=3", headers=HDR, timeout=15)
    if r.status_code == 200:
        for c in r.json():
            log(f"  {c['sha'][:7]} {c['commit']['author']['date'][:16]} {c['commit']['message'].split(chr(10))[0][:60]}")
    r = requests.get(f"{API_BASE}/contents/stockdata/sh.600519.csv",
                     headers={"Accept": "application/vnd.github.v3.raw"}, timeout=15)
    if r.status_code == 200:
        lines = r.text.strip().split("\n")
        last = lines[-1] if lines else ""
        date = last.split(",")[0]
        log(f"  sh.600519.csv: {date}")
        if date >= "2026-06-29": log("  ✅ 数据已更新!")
        else: log(f"  ⚠️ 仍是旧数据 {date}")

def main():
    log("="*60)
    log("🔧 一键修复：刷新CSV + API推送")
    log("="*60)
    
    if not os.path.exists(PRIORITY_FILE):
        log(f"ERROR: {PRIORITY_FILE} not found"); sys.exit(1)
    
    with open(PRIORITY_FILE, 'r', encoding='utf-8') as f:
        pdata = json.load(f)
    
    codes = [s.get("code","").replace("sh.","").replace("sz.","") for s in pdata.get("stocks",[])]
    codes = [c for c in codes if c]
    log(f"加载 {len(codes)} 只标的")
    
    log("\nSTEP 1: 刷新CSV数据")
    try: refresh_csv_data(codes)
    except Exception as e:
        log(f"CSV刷新异常: {e}"); traceback.print_exc()
    
    log("\nSTEP 2: API推送")
    push_via_api()
    
    log("\nSTEP 3: 验证")
    verify()
    log("\n✅ 全部完成!")

if __name__ == "__main__":
    main()

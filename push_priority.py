#!/usr/bin/env python3
"""push_priority.py - 只推送205只priority标的CSV + priority_stocks.json (共206文件)"""
import json, hashlib, base64, time, os, sys
from pathlib import Path

REPO = "Scilogos/scilogos.github.io"
TOKEN = os.environ.get("GH_TOKEN", "")
API = "https://api.github.com"
H = {"Authorization": f"token {TOKEN}", "Accept": "application/vnd.github+json"}

if not TOKEN:
    print("ERROR: set GH_TOKEN first")
    sys.exit(1)

import requests

def _r(method, url, **kw):
    for attempt in range(3):
        try:
            r = getattr(requests, method)(url, headers=H, timeout=60, **kw)
            if r.status_code < 300:
                return r.json() if r.text else {}
            if r.status_code == 502:
                print(f"  502, wait 10s...")
                time.sleep(10)
                continue
            print(f"  {r.status_code}: {r.text[:200]}")
            return None
        except Exception as e:
            if attempt < 2:
                print(f"  net err, retry...")
                time.sleep(5)
            else:
                print(f"  FAIL: {str(e)[:100]}")
                return None
    return None

def head():
    d = _r('get', f"{API}/repos/{REPO}/git/ref/heads/main")
    sha = d['object']['sha']
    d2 = _r('get', f"{API}/repos/{REPO}/git/commits/{sha}")
    return sha, d2['tree']['sha']

def blob(path, data):
    sha = hashlib.sha1(data).hexdigest()
    for attempt in range(3):
        r = _r('post', f"{API}/repos/{REPO}/git/blobs", json={
            "content": base64.b64encode(data).decode(),
            "encoding": "base64"
        })
        if r and 'sha' in r:
            return r['sha'], sha
        if attempt < 2:
            time.sleep(3)
    return None, sha

def tree(blobs_list, base_sha):
    # Try in batches if too many
    for batch_size in [200, 100, 50]:
        items = []
        for path, sha in blobs_list:
            items.append({"path": path, "mode": "100644", "type": "blob", "sha": sha})
        
        if len(items) <= batch_size:
            r = _r('post', f"{API}/repos/{REPO}/git/trees", json={
                "base_tree": base_sha, "tree": items
            })
            if r and 'sha' in r:
                return r['sha']
            print(f"  tree({len(items)} files) failed, trying smaller batch...")
            continue
        
        # Batch approach: create subtree for each batch, then combine
        # Actually, let's just try sending all at once with a smaller set
        break
    
    # Fallback: try with just the items
    r = _r('post', f"{API}/repos/{REPO}/git/trees", json={
        "base_tree": base_sha, "tree": items
    })
    if r and 'sha' in r:
        return r['sha']
    return None

# ====== MAIN ======
t0 = time.time()
print(f"[{time.strftime('%H:%M:%S')}] push_priority.py - 精简推送 (仅206文件)")
print(f"[{time.strftime('%H:%M:%S')}] " + "="*50)

repo = Path(r"C:\Users\HUAWEI\Desktop\scilogos.github.io")
pf = repo / "numpy_run" / "priority_stocks.json"
sd = repo / "stockdata"

# Load priority list
pj = json.loads(pf.read_bytes())
codes = [s['code'] for s in pj['stocks']]
print(f"[{time.strftime('%H:%M:%S')}] priority标的: {len(codes)}只")

# Build file list
files = []
# 1. priority_stocks.json
fd = pf.read_bytes()
files.append(("numpy_run/priority_stocks.json", fd))

# 2. priority CSVs
csv_ok = 0
for code in codes:
    fp = sd / f"{code}.csv"
    if fp.exists():
        files.append((f"stockdata/{code}.csv", fp.read_bytes()))
        csv_ok += 1
    else:
        print(f"  SKIP {code}: not found")
print(f"[{time.strftime('%H:%M:%S')}] 待推送: {len(files)} 文件 ({csv_ok} CSV + 1 JSON)")

# Check if files already match remote (skip unchanged)
print(f"[{time.strftime('%H:%M:%S')}] 获取远程HEAD...")
commit_sha, base_tree = head()
print(f"[{time.strftime('%H:%M:%S')}] HEAD: {commit_sha[:7]}, tree: {base_tree[:7]}")

# Create blobs
print(f"[{time.strftime('%H:%M:%S')}] 创建blobs...")
tree_items = []
fail = 0
t1 = time.time()
for i, (path, data) in enumerate(files):
    bsha, lsha = blob(path, data)
    if bsha:
        tree_items.append((path, bsha))
    else:
        fail += 1
        print(f"  FAIL: {path}")
    if (i+1) % 20 == 0:
        elapsed = time.time() - t1
        print(f"[{time.strftime('%H:%M:%S')}] {i+1}/{len(files)} ({fail} err, {elapsed:.0f}s)")

print(f"[{time.strftime('%H:%M:%S')}] Blobs完成: {len(tree_items)} ok, {fail} fail, {time.time()-t1:.0f}s")

# Create tree
print(f"[{time.strftime('%H:%M:%S')}] 创建tree ({len(tree_items)} items)...")
t2 = time.time()
tree_sha = tree(tree_items, base_tree)
if not tree_sha:
    print(f"[{time.strftime('%H:%M:%S')}] FATAL: tree创建失败")
    sys.exit(1)
print(f"[{time.strftime('%H:%M:%S')}] tree: {tree_sha[:7]} ({time.time()-t2:.0f}s)")

# Create commit
msg = f"priority fix: 205 主板标的 + data sync {time.strftime('%Y-%m-%d %H:%M')}"
r = _r('post', f"{API}/repos/{REPO}/git/commits", json={
    "message": msg, "tree": tree_sha, "parents": [commit_sha]
})
if not r:
    print(f"[{time.strftime('%H:%M:%S')}] FATAL: commit创建失败")
    sys.exit(1)
new_sha = r['sha']
print(f"[{time.strftime('%H:%M:%S')}] commit: {new_sha[:7]}")

# Update ref
r = _r('patch', f"{API}/repos/{REPO}/git/refs/heads/main", json={"sha": new_sha})
if r:
    print(f"[{time.strftime('%H:%M:%S')}] ✅ 推送成功!")
else:
    print(f"[{time.strftime('%H:%M:%S')}] ❌ 分支更新失败")
    sys.exit(1)

# Verify
print(f"[{time.strftime('%H:%M:%S')}] 验证...")
for code in ["sh.600519", "sz.000001", "sh.601318", "sz.002230", "sh.600036"]:
    r = _r('get', f"{API}/repos/{REPO}/contents/stockdata/{code}.csv")
    if r and 'content' in r:
        content = base64.b64decode(r['content']).decode(errors='ignore')
        lines = content.strip().split('\n')
        last = lines[-1].split(',')[0] if lines else '?'
        print(f"  {code}: {len(lines)}行, 最新={last}")

total = time.time() - t0
print(f"[{time.strftime('%H:%M:%S')}] ✅ 全部完成! 耗时{total:.0f}秒")

# -*- coding: utf-8 -*-
"""
one_click_arena.py - 本地一键运行脚本
在Windows本地一键启动对抗学习Arena训练

【使用方法】
1. 双击运行: python one_click_arena.py
2. 或在命令行: python one_click_arena.py

【功能】
- 自动检测并安装依赖
- 自动生成模拟数据（如需要）
- 自动从检查点恢复训练
- 实时进度显示
- 训练完成后自动打开结果

【断点续传】
如果训练中途断开，再次运行会自动从检查点恢复
"""

import os
import sys
import json
import time
import subprocess
import platform
import shutil
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, List

# ============================================================================
# 配置
# ============================================================================

# 项目路径（Windows本地）
if os.name == "nt":
    BASE_DIR = Path(r"C:\Users\HUAWEI\Desktop\Adversarial Learning")
else:
    BASE_DIR = Path.home() / "AdversarialLearning"

# 子目录
SCRIPT_DIR = BASE_DIR / "dotpy"
DATA_DIR = BASE_DIR / "stockdata"
ADV_DATA_DIR = BASE_DIR / "adversarial_data"
ADV_MODEL_DIR = BASE_DIR / "adversarial_model"
RESULTS_DIR = BASE_DIR / "stockresults"
LOG_DIR = RESULTS_DIR / "logs"

# 检查点文件
CHECKPOINT_FILE = BASE_DIR / "checkpoint.json"

# GitHub源
GITHUB_BASE = "https://raw.githubusercontent.com/Scilogs/scilogos.github.io/main/cai/dotpy"

# 训练参数
ARENA_EPISODES = 500
BATCH_SIZE = 64

# ============================================================================
# 工具函数
# ============================================================================

def print_banner():
    """打印横幅"""
    banner = """
╔══════════════════════════════════════════════════════════════╗
║     🏛️  对抗学习量化系统 - Arena 本地训练                   ║
║     庄家 vs 散户: 8×8 组合对战                              ║
╚══════════════════════════════════════════════════════════════╝
    """
    print(banner)

def ensure_directories():
    """确保目录结构存在"""
    dirs = [BASE_DIR, SCRIPT_DIR, DATA_DIR, ADV_DATA_DIR, ADV_MODEL_DIR, RESULTS_DIR, LOG_DIR]
    for d in dirs:
        d.mkdir(parents=True, exist_ok=True)
    print(f"📁 目录检查完成: {BASE_DIR}")

def check_dependencies() -> bool:
    """检查并安装依赖"""
    print("\n" + "="*50)
    print("📦 检查依赖...")
    print("="*50)
    
    required = ["torch", "numpy", "scipy", "pandas"]
    missing = []
    
    for lib in required:
        try:
            __import__(lib)
            print(f"  ✅ {lib}")
        except ImportError:
            print(f"  ❌ {lib} - 缺失")
            missing.append(lib)
    
    if missing:
        print(f"\n⚠️ 缺少依赖: {missing}")
        choice = input("是否自动安装? (Y/n): ").strip().lower()
        if choice != 'n':
            cmd = f"pip install {' '.join(missing)}"
            print(f"执行: {cmd}")
            os.system(cmd)
            return check_dependencies()
        return False
    
    # 检查PyTorch CUDA
    try:
        import torch
        print(f"\nPyTorch版本: {torch.__version__}")
        if torch.cuda.is_available():
            print(f"✅ GPU可用: {torch.cuda.get_device_name(0)}")
        else:
            print("ℹ️ 使用CPU训练（较慢但可运行）")
    except:
        pass
    
    return True

def download_scripts() -> bool:
    """下载GitHub脚本"""
    print("\n" + "="*50)
    print("📥 检查脚本文件...")
    print("="*50)
    
    scripts = [
        "stock_config.py",
        "adversarial_env.py", 
        "market_generator.py",
        "stock_data_manager.py",
        "stock_interpreter.py"
    ]
    
    all_exist = True
    for s in scripts:
        path = SCRIPT_DIR / s
        if path.exists():
            size = path.stat().st_size / 1024
            print(f"  ✅ {s} ({size:.1f} KB)")
        else:
            print(f"  ❌ {s} - 缺失")
            all_exist = False
    
    if not all_exist:
        print("\n⚠️ 部分脚本缺失")
        choice = input("是否从GitHub下载? (Y/n): ").strip().lower()
        if choice != 'n':
            import urllib.request
            for s in scripts:
                url = f"{GITHUB_BASE}/{s}"
                dest = SCRIPT_DIR / s
                try:
                    print(f"  下载: {s}...")
                    urllib.request.urlretrieve(url, dest)
                    print(f"    ✅ 完成")
                except Exception as e:
                    print(f"    ❌ 失败: {e}")
            return download_scripts()  # 重新检查
        return False
    
    return True

def generate_fake_data() -> bool:
    """生成模拟数据"""
    data_file = ADV_DATA_DIR / "generated_2000.npy"
    
    if data_file.exists():
        print(f"\n📊 数据文件已存在: {data_file}")
        return True
    
    print("\n" + "="*50)
    print("🔧 生成模拟股价数据...")
    print("="*50)
    
    try:
        import numpy as np
        
        def generate_fake_prices(n_samples=2000, seq_len=30, n_features=6, seed=42):
            np.random.seed(seed)
            data = np.zeros((n_samples, seq_len, n_features))
            
            for i in range(n_samples):
                if (i + 1) % 500 == 0:
                    print(f"  进度: {i+1}/{n_samples}")
                
                p0 = np.random.uniform(5, 50)
                mu = np.random.uniform(-0.001, 0.002)
                sigma = np.random.uniform(0.01, 0.04)
                
                returns = np.random.normal(mu, sigma, seq_len)
                returns = np.clip(returns, -0.10, 0.10)
                close = p0 * np.cumprod(1 + returns)
                
                high_spread = np.abs(np.random.normal(0, sigma*0.5, seq_len))
                low_spread = np.abs(np.random.normal(0, sigma*0.5, seq_len))
                open_offset = np.random.normal(0, sigma*0.3, seq_len)
                
                open_price = close * (1 + open_offset)
                high_price = np.maximum(open_price, close) * (1 + high_spread)
                low_price = np.minimum(open_price, close) * (1 - low_spread)
                volume = np.random.lognormal(mean=15, sigma=1, size=seq_len).astype(float)
                
                pct = np.zeros(seq_len)
                pct[0] = returns[0]
                pct[1:] = np.diff(close) / close[:-1]
                
                data[i, :, 0] = open_price
                data[i, :, 1] = high_price
                data[i, :, 2] = low_price
                data[i, :, 3] = close
                data[i, :, 4] = volume
                data[i, :, 5] = pct
            
            return data
        
        print("生成2000条模拟股价序列...")
        fake_data = generate_fake_prices(2000, 30, 6)
        np.save(data_file, fake_data)
        
        print(f"\n✅ 数据已保存: {data_file}")
        print(f"   形状: {fake_data.shape}")
        return True
        
    except Exception as e:
        print(f"❌ 数据生成失败: {e}")
        return False

def load_checkpoint() -> Optional[Dict]:
    """加载检查点"""
    if not CHECKPOINT_FILE.exists():
        return None
    
    try:
        with open(CHECKPOINT_FILE, 'r', encoding='utf-8') as f:
            return json.load(f)
    except:
        return None

def save_checkpoint(episode: int, stats: Dict):
    """保存检查点"""
    checkpoint = {
        "episode": episode,
        "timestamp": datetime.now().isoformat(),
        "stats": stats,
    }
    with open(CHECKPOINT_FILE, 'w', encoding='utf-8') as f:
        json.dump(checkpoint, f, indent=2, ensure_ascii=False)

def run_arena(resume_episode: int = 0) -> bool:
    """运行Arena训练"""
    print("\n" + "="*50)
    if resume_episode > 0:
        print(f"🔄 从第 {resume_episode} 轮恢复训练...")
    else:
        print("🏛️ 开始Arena训练...")
    print("="*50)
    
    data_file = ADV_DATA_DIR / "generated_2000.npy"
    if not data_file.exists():
        print("❌ 数据文件不存在!")
        return False
    
    # 检查adversarial_env.py
    arena_script = SCRIPT_DIR / "adversarial_env.py"
    if not arena_script.exists():
        print(f"❌ 找不到adversarial_env.py: {arena_script}")
        return False
    
    # 构建命令
    cmd = [
        sys.executable,
        str(arena_script),
        "--mode", "arena",
        "--price-data", str(data_file),
        "--benchmark-source", "fake",
        "--arena-episodes", str(ARENA_EPISODES),
    ]
    
    if resume_episode > 0:
        cmd.extend(["--resume-from", str(resume_episode)])
    
    print(f"\n执行命令: {' '.join(cmd)}\n")
    print("-" * 50)
    
    # 创建日志文件
    log_file = LOG_DIR / f"arena_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    
    try:
        # 实时显示输出
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            cwd=str(SCRIPT_DIR)
        )
        
        start_time = time.time()
        last_checkpoint_time = start_time
        
        for line in process.stdout:
            print(line, end='')  # 实时输出
            
            # 写入日志
            with open(log_file, 'a', encoding='utf-8') as f:
                f.write(line)
            
            # 定期保存检查点（每60秒）
            if time.time() - last_checkpoint_time > 60:
                # 尝试解析进度
                save_checkpoint(0, {"status": "running", "log": str(log_file)})
                last_checkpoint_time = time.time()
        
        process.wait()
        elapsed = time.time() - start_time
        
        print("-" * 50)
        print(f"\n⏱️ 训练完成! 总耗时: {elapsed/60:.1f} 分钟")
        print(f"📝 日志文件: {log_file}")
        
        return process.returncode == 0
        
    except KeyboardInterrupt:
        print("\n\n⚠️ 用户中断训练")
        print("💡 重新运行将从检查点恢复")
        return False
    except Exception as e:
        print(f"\n❌ 运行出错: {e}")
        import traceback
        traceback.print_exc()
        return False

def show_results():
    """显示训练结果"""
    print("\n" + "="*50)
    print("📊 训练结果")
    print("="*50)
    
    # 查找结果文件
    result_files = list(RESULTS_DIR.glob("*.json"))
    
    if result_files:
        print(f"\n找到 {len(result_files)} 个结果文件:")
        for f in result_files:
            print(f"  📄 {f.name}")
            print(f"     大小: {f.stat().st_size / 1024:.1f} KB")
        
        # 显示最新的结果
        latest = max(result_files, key=lambda p: p.stat().st_mtime)
        print(f"\n📄 最新结果: {latest.name}")
        
        try:
            with open(latest, 'r', encoding='utf-8') as f:
                results = json.load(f)
            print("\n" + json.dumps(results, indent=2, ensure_ascii=False)[:2000])
        except Exception as e:
            print(f"读取失败: {e}")
    else:
        print("\n⚠️ 未找到结果文件")

def open_results_folder():
    """打开结果文件夹"""
    print(f"\n📂 结果目录: {RESULTS_DIR}")
    choice = input("是否打开结果文件夹? (y/N): ").strip().lower()
    if choice == 'y':
        if os.name == "nt":
            os.system(f'explorer "{RESULTS_DIR}"')
        elif sys.platform == "darwin":
            os.system(f'open "{RESULTS_DIR}"')
        else:
            os.system(f'xdg-open "{RESULTS_DIR}"')

def update_cross_platform_config():
    """更新跨平台配置"""
    print("\n📝 更新跨平台配置...")
    
    config_file = SCRIPT_DIR / "stock_config.py"
    
    # 创建跨平台配置内容
    cross_config = '''# -*- coding: utf-8 -*-
"""
stock_config.py - 跨平台配置版本
自动检测Windows/Colab/Linux环境
"""

import os
import sys
import json
import logging
import numpy as np
from dataclasses import dataclass, field
from pathlib import Path
from datetime import datetime

# 【核心】跨平台路径自动检测
if os.path.exists("/content"):
    # Google Colab
    BASE_DIR = Path("/content/AdversarialLearning")
elif os.name == "nt":
    # Windows本地
    BASE_DIR = Path(r"C:\\Users\\HUAWEI\\Desktop\\Adversarial Learning")
else:
    # 其他Linux
    BASE_DIR = Path.home() / "AdversarialLearning"

# 目录定义
SCRIPT_DIR = BASE_DIR / "dotpy"
DATA_DIR = BASE_DIR / "stockdata"
ADV_DATA_DIR = BASE_DIR / "adversarial_data"
ADV_MODEL_DIR = BASE_DIR / "adversarial_model"
RESULTS_DIR = BASE_DIR / "stockresults"
FEEDBACK_FILE = SCRIPT_DIR / "feedback.txt"
PYMANAGER_FILE = SCRIPT_DIR / "pymanager.txt"

# 自动创建目录
for d in [DATA_DIR, ADV_DATA_DIR, ADV_MODEL_DIR, RESULTS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# 检查点
CHECKPOINT_FILE = BASE_DIR / "checkpoint.json"

def save_checkpoint(episode, stats):
    with open(CHECKPOINT_FILE, 'w', encoding='utf-8') as f:
        json.dump({"episode": episode, "timestamp": datetime.now().isoformat(), "stats": stats}, f)

def load_checkpoint():
    if CHECKPOINT_FILE.exists():
        with open(CHECKPOINT_FILE, 'r', encoding='utf-8') as f:
            return json.load(f)
    return None

print(f"[配置] 跨平台模式启动, 路径: {BASE_DIR}")
'''
    
    try:
        with open(config_file, 'w', encoding='utf-8') as f:
            f.write(cross_config)
        print(f"✅ 跨平台配置已更新: {config_file}")
    except Exception as e:
        print(f"⚠️ 配置更新失败: {e}")

# ============================================================================
# 主程序
# ============================================================================

def main():
    """主程序"""
    os.system('chcp 65001 >nul 2>&1')  # Windows中文编码
    os.system('cls' if os.name == 'nt' else 'clear')
    
    print_banner()
    
    # 系统信息
    print(f"🖥️  系统: {platform.system()} {platform.release()}")
    print(f"🐍  Python: {platform.python_version()}")
    print(f"📂  工作目录: {BASE_DIR}")
    print(f"⏰  开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # 确保目录存在
    ensure_directories()
    
    # 更新跨平台配置
    update_cross_platform_config()
    
    # 检查依赖
    if not check_dependencies():
        print("\n❌ 依赖检查未通过，请手动安装后重试")
        input("按Enter退出...")
        sys.exit(1)
    
    # 下载脚本
    if not download_scripts():
        print("\n⚠️ 脚本文件不完整，训练可能无法运行")
        proceed = input("是否继续? (y/N): ").strip().lower()
        if proceed != 'y':
            sys.exit(0)
    
    # 生成数据
    if not generate_fake_data():
        print("\n❌ 数据生成失败")
        input("按Enter退出...")
        sys.exit(1)
    
    # 检查是否有检查点
    checkpoint = load_checkpoint()
    resume_episode = 0
    if checkpoint:
        print("\n" + "="*50)
        print("🔄 检测到之前的训练进度!")
        print(f"   最后回合: {checkpoint.get('episode', '?')}")
        print(f"   时间: {checkpoint.get('timestamp', '?')}")
        print("="*50)
        
        choice = input("\n1) 从检查点恢复训练\n2) 重新开始\n选择 (1/2): ").strip()
        if choice == '1':
            resume_episode = checkpoint.get('episode', 0)
    
    # 运行训练
    print()
    success = run_arena(resume_episode)
    
    # 显示结果
    if success:
        show_results()
    else:
        print("\n⚠️ 训练未完成，但可以稍后恢复")
        print(f"   检查点: {CHECKPOINT_FILE}")
    
    # 打开结果目录
    open_results_folder()
    
    print("\n" + "="*50)
    print("✅ 程序结束")
    print("="*50)
    
    input("\n按Enter退出...")

if __name__ == "__main__":
    main()

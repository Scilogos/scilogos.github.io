# -*- coding: utf-8 -*-
"""
stock_config_cloud.py - 跨平台配置版本
为Google Colab和本地Windows双环境设计的对抗学习量化系统配置

【使用方法】
1. Colab: 从GitHub下载此文件，覆盖原stock_config.py
2. Windows本地: 直接使用此文件，或替换原文件

【环境检测逻辑】
- Linux/Colab: 检测到/content目录 → 使用 /content/AdversarialLearning/
- Windows: os.name == 'nt' → 使用原路径 C:\Users\HUAWEI\Desktop\Adversarial Learning
- 其他Linux: 使用用户家目录 ~/AdversarialLearning
"""

import os
import sys
import json
import logging
import hashlib
import numpy as np
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Tuple
from pathlib import Path
from datetime import datetime

# ============================================================================
# 【核心】跨平台路径自动检测
# ============================================================================

def get_base_dir() -> Path:
    """
    自动检测运行环境，返回项目根目录
    
    检测顺序:
    1. Google Colab环境: /content/AdversarialLearning/
    2. Windows环境: C:\Users\HUAWEI\Desktop\Adversarial Learning
    3. 其他Linux: ~/AdversarialLearning/
    """
    # 检测Google Colab
    if os.path.exists("/content"):
        base = Path("/content/AdversarialLearning")
        print(f"[跨平台配置] 检测到Colab环境，使用路径: {base}")
        return base
    
    # 检测Windows
    if os.name == "nt" or os.name == "ce":
        base = Path(r"C:\Users\HUAWEI\Desktop\Adversarial Learning")
        print(f"[跨平台配置] 检测到Windows环境，使用路径: {base}")
        return base
    
    # 其他Linux/Unix系统
    base = Path.home() / "AdversarialLearning"
    print(f"[跨平台配置] 检测到Linux/Unix环境，使用路径: {base}")
    return base

# 项目根目录
BASE_DIR = get_base_dir()

# ============================================================================
# 【核心】目录结构定义（所有路径基于BASE_DIR自动生成）
# ============================================================================

# 脚本目录
SCRIPT_DIR = BASE_DIR / "dotpy"

# 数据目录
DATA_DIR = BASE_DIR / "stockdata"           # 原始股票数据
ADV_DATA_DIR = BASE_DIR / "adversarial_data"  # 对抗训练数据
ADV_MODEL_DIR = BASE_DIR / "adversarial_model"  # 模型保存
RESULTS_DIR = BASE_DIR / "stockresults"     # 结果输出

# 反馈文件
FEEDBACK_FILE = SCRIPT_DIR / "feedback.txt"
PYMANAGER_FILE = SCRIPT_DIR / "pymanager.txt"

# ============================================================================
# 【核心】自动创建所有必要目录
# ============================================================================

def ensure_directories():
    """确保所有必要的目录都存在，不存在则自动创建"""
    dirs_to_create = [
        BASE_DIR,
        SCRIPT_DIR,
        DATA_DIR,
        ADV_DATA_DIR,
        ADV_MODEL_DIR,
        RESULTS_DIR,
    ]
    
    created = []
    for d in dirs_to_create:
        if isinstance(d, Path):
            d = str(d)
        if not os.path.exists(d):
            os.makedirs(d, exist_ok=True)
            created.append(d)
    
    if created:
        print(f"[跨平台配置] 已创建目录: {created}")
    else:
        print(f"[跨平台配置] 所有目录已存在，无需创建")

# 启动时自动调用
ensure_directories()

# ============================================================================
# 【配置】Python环境与执行设置
# ============================================================================

# Python解释器
if sys.platform == "win32":
    PYTHON_EXE = "python"
else:
    PYTHON_EXE = "python3"

# 并行计算设置
NUM_WORKERS = 4  # 数据加载线程数
DEVICE = "cuda" if os.path.exists("/content") else "cpu"  # Colab优先用GPU

# ============================================================================
# 【配置】日志设置
# ============================================================================

def setup_logging(log_level: str = "INFO") -> logging.Logger:
    """
    配置日志系统
    
    Args:
        log_level: DEBUG, INFO, WARNING, ERROR
    
    Returns:
        配置好的logger对象
    """
    log_dir = RESULTS_DIR / "logs"
    log_dir.mkdir(exist_ok=True)
    
    log_file = log_dir / f"arena_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s [%(levelname)s] %(message)s',
        handlers=[
            logging.FileHandler(log_file, encoding='utf-8'),
            logging.StreamHandler()
        ]
    )
    
    logger = logging.getLogger("AdversarialArena")
    return logger

# ============================================================================
# 【配置】数据参数
# ============================================================================

@dataclass
class DataConfig:
    """数据相关配置"""
    # 序列长度（交易日数）
    SEQ_LEN: int = 30
    
    # 特征维度: [open, high, low, close, volume, pct_change]
    N_FEATURES: int = 6
    
    # OHLC列索引
    OPEN_IDX: int = 0
    HIGH_IDX: int = 1
    LOW_IDX: int = 2
    CLOSE_IDX: int = 3
    VOLUME_IDX: int = 4
    PCT_IDX: int = 5
    
    # 数据归一化范围
    PRICE_MIN: float = 0.0
    PRICE_MAX: float = 100.0
    
    # 涨跌停限制（A股10%）
    LIMIT_UP: float = 0.10
    LIMIT_DOWN: float = -0.10

DATA_CONFIG = DataConfig()

# ============================================================================
# 【配置】模型超参数
# ============================================================================

@dataclass
class ModelConfig:
    """模型超参数配置"""
    # 生成器 (Market Generator)
    G_HIDDEN: int = 256
    G_LAYERS: int = 3
    G_LR: float = 0.0001
    G_BETA1: float = 0.5
    
    # 判别器 (Market Interpreter)
    D_HIDDEN: int = 256
    D_LAYERS: int = 3
    D_LR: float = 0.0001
    D_BETA1: float = 0.5
    
    # 训练参数
    BATCH_SIZE: int = 64
    EPOCHS: int = 100
    LATENT_DIM: int = 100
    
    # 强化学习参数
    RL_GAMMA: float = 0.99
    RL_LR: float = 0.001

MODEL_CONFIG = ModelConfig()

# ============================================================================
# 【配置】庄家-散户对抗设置
# ============================================================================

@dataclass
class ArenaConfig:
    """对抗竞技场配置"""
    # 庄家策略池
    ZHUANG_STRATEGIES: List[str] = field(default_factory=lambda: [
        "pump_and_dump",      # 拉高出货
        "wash_and_sale",      # 洗盘出货
        "high_floating",      # 高位横盘
        "short_selling",      # 做空打压
        "range_manipulation", # 区间震荡
        "news_drive",         # 消息驱动
        "volume_manipulation",# 量价配合
        "sector_rotation",    # 板块轮动
    ])
    
    # 散户策略池
    SAN_STRATEGIES: List[str] = field(default_factory=lambda: [
        "momentum_following",    # 趋势跟踪
        "mean_reversion",        # 均值回归
        "breakout_trading",      # 突破交易
        "value_investing",       # 价值投资
        "contrarian",            # 逆向投资
        "sector_rotation",       # 板块轮动
        "stop_loss",             # 止损纪律
        "position_management",   # 仓位管理
    ])
    
    # 对抗参数
    N_EPISODES: int = 500           # Arena总回合数
    EPISODE_STEPS: int = 30         # 每回合步数（交易日）
    INITIAL_CAPITAL: float = 100000.0  # 初始资金
    
    # 评估指标
    SHARPE_RISK_FREE: float = 0.03  # 无风险利率（年化3%）

ARENA_CONFIG = ArenaConfig()

# ============================================================================
# 【工具函数】路径兼容性
# ============================================================================

def get_data_path(filename: str) -> Path:
    """获取数据文件路径，兼容Windows路径分隔符"""
    return DATA_DIR / filename

def get_model_path(filename: str) -> Path:
    """获取模型文件路径"""
    return ADV_MODEL_DIR / filename

def get_result_path(filename: str) -> Path:
    """获取结果文件路径"""
    return RESULTS_DIR / filename

# ============================================================================
# 【工具函数】检查点管理
# ============================================================================

CHECKPOINT_FILE = BASE_DIR / "checkpoint.json"

def save_checkpoint(episode: int, stats: Dict, models: Dict = None):
    """保存训练检查点"""
    checkpoint = {
        "episode": episode,
        "timestamp": datetime.now().isoformat(),
        "stats": stats,
        "base_dir": str(BASE_DIR),
    }
    with open(CHECKPOINT_FILE, 'w', encoding='utf-8') as f:
        json.dump(checkpoint, f, indent=2, ensure_ascii=False)
    print(f"[检查点] 已保存: episode={episode}")

def load_checkpoint() -> Optional[Dict]:
    """加载检查点，返回None表示无检查点"""
    if not CHECKPOINT_FILE.exists():
        return None
    with open(CHECKPOINT_FILE, 'r', encoding='utf-8') as f:
        return json.load(f)

# ============================================================================
# 【工具函数】文件哈希验证
# ============================================================================

def get_file_hash(filepath: str) -> str:
    """计算文件MD5哈希"""
    if not os.path.exists(filepath):
        return ""
    hash_md5 = hashlib.md5()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()

# ============================================================================
# 【启动信息】
# ============================================================================

def print_startup_info():
    """打印启动信息"""
    print("\n" + "="*60)
    print("对抗学习量化系统 - 跨平台配置")
    print("="*60)
    print(f"项目根目录: {BASE_DIR}")
    print(f"脚本目录:   {SCRIPT_DIR}")
    print(f"数据目录:   {DATA_DIR}")
    print(f"对抗数据:   {ADV_DATA_DIR}")
    print(f"模型目录:   {ADV_MODEL_DIR}")
    print(f"结果目录:   {RESULTS_DIR}")
    print(f"检查点:     {CHECKPOINT_FILE}")
    print("="*60 + "\n")

# 如果直接运行此文件，打印启动信息
if __name__ == "__main__":
    print_startup_info()

"""
stock_config_cloud.py - 云端自适应配置
========================================
自动检测运行环境(Colab/Windows/Linux)，调整路径。
所有脚本import此文件而非stock_config.py。
"""

import os, sys, json, logging, hashlib, platform
import numpy as np
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Tuple
from pathlib import Path
from datetime import datetime


# ============================================================
# 环境检测与路径配置
# ============================================================
def _detect_environment() -> str:
    """检测运行环境"""
    if 'COLAB_GPU' in os.environ or os.path.exists('/content') or 'google.colab' in str(sys.modules):
        return 'colab'
    elif platform.system() == 'Windows':
        return 'windows'
    elif platform.system() == 'Linux':
        return 'linux'
    else:
        return 'unknown'

ENV = _detect_environment()

if ENV == 'colab':
    # Google Colab 环境
    BASE_DIR = Path('/content/AdversarialLearning')
    # 如果有Google Drive挂载，使用Drive持久化
    GDRIVE_DIR = Path('/content/drive/MyDrive/AdversarialLearning')
elif ENV == 'windows':
    BASE_DIR = Path(r"C:\Users\HUAWEI\Desktop\Adversarial Learning")
    GDRIVE_DIR = None
elif ENV == 'linux':
    # 通用Linux (如沙箱)
    BASE_DIR = Path('/app/data/所有对话/主对话/cloud_arena/workspace')
    GDRIVE_DIR = None
else:
    BASE_DIR = Path('./AdversarialLearning')
    GDRIVE_DIR = None

SCRIPT_DIR = BASE_DIR / "dotpy"
DATA_DIR = BASE_DIR / "stockdata"
ADV_DATA_DIR = BASE_DIR / "adversarial_data"
ADV_MODEL_DIR = BASE_DIR / "adversarial_model"
RESULTS_DIR = BASE_DIR / "stockresults"
FEEDBACK_FILE = SCRIPT_DIR / "feedback.txt"
PYMANAGER_FILE = SCRIPT_DIR / "pymanager.txt"
CHECKPOINT_DIR = ADV_MODEL_DIR / "checkpoint"

# 创建所有目录
for d in [SCRIPT_DIR, DATA_DIR, ADV_DATA_DIR, ADV_MODEL_DIR, 
          RESULTS_DIR, CHECKPOINT_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# Python解释器
if ENV == 'windows':
    PYTHON_EXE = r"C:\Users\HUAWEI\AppData\Local\Programs\Python\Python312\python.exe"
else:
    PYTHON_EXE = sys.executable

print(f"[stock_config] 环境: {ENV} | BASE: {BASE_DIR}")


# ============================================================
# Baostock 数据源配置
# ============================================================
BS_FIELDS_DAILY = [
    "date", "code", "open", "high", "low", "close",
    "preclose", "volume", "amount", "adjustflag",
    "turn", "tradestatus", "pctChg", "isST"
]
DAILY_START = "2020-01-01"  # 云端版缩短到2020年，减少下载时间
DAILY_END = "2026-06-25"
FOCUS_POOL_SIZE = 200  # 云端版聚焦200只代表股

# ============================================================
# 生成器 C-TimeGAN 配置 (云端缩减版)
# ============================================================
@dataclass
class GeneratorConfig:
    hidden_dim: int = 64
    num_layers: int = 3
    seq_len: int = 30
    feature_dim: int = 6
    condition_levels: int = 3
    # 云端缩减epoch数(训练更快)
    phase_a_epochs: int = 80
    phase_b_epochs: int = 80
    phase_c_epochs: int = 150
    batch_size: int = 64
    learning_rate: float = 1e-3
    # 验证
    dtw_threshold: float = 2.0
    pearson_threshold: float = 0.85
    ks_pvalue: float = 0.05
    w930_940: float = 5.0
    w940_950: float = 2.0
    w_other: float = 1.0
    max_similar: int = 3

# ============================================================
# 对抗环境配置
# ============================================================
@dataclass
class AdversarialConfig:
    num_episodes: int = 500     # 云端缩减到500(本地1000)
    episode_length: int = 240   # 4h × 60min ticks
    initial_price: float = 10.0
    # 庄家
    dealer_capital_ratio: float = 0.30
    dealer_info_manip_prob: float = 0.05
    # 散户
    retailer_ratio: float = 0.55
    retailer_monthly_salary: float = 10000.0
    retailer_types: List[str] = field(default_factory=lambda: [
        "herd", "value", "technical", "leader", "passive"
    ])
    # 游资
    hotmoney_ratio: float = 0.15
    hotmoney_momentum_thresh: float = 0.03
    # 进化
    lamarck_rate: float = 0.1
    darwin_rate: float = 0.05
    evolution_interval: int = 50
    # 防摆烂
    anti_degenerate_thresh: float = 0.1
    min_strategy_diversity: float = 0.3

# ============================================================
# 解读器配置
# ============================================================
@dataclass
class InterpreterConfig:
    stat_window: int = 20
    game_depth: int = 5
    historical_topk: int = 10
    max_drawdown: float = 0.15
    perm_samples: int = 1000
    significance: float = 0.05
    signal_conf_thresh: float = 0.6

# ============================================================
# 实盘反馈配置
# ============================================================
@dataclass
class FeedbackConfig:
    feedback_file: str = str(FEEDBACK_FILE)
    results_dir: str = str(RESULTS_DIR)
    model_dir: str = str(ADV_MODEL_DIR)
    calibration_window: int = 20
    signal_accuracy_thresh: float = 0.55
    max_position_adjust: float = 0.1
    reward_shaping_weight: float = 0.3

# ============================================================
# 数据工具
# ============================================================
def normalize_ohlcv(arr: np.ndarray) -> np.ndarray:
    mn, mx = arr.min(axis=0, keepdims=True), arr.max(axis=0, keepdims=True)
    rng = mx - mn
    rng[rng == 0] = 1.0
    return (arr - mn) / rng

def denormalize_ohlcv(arr: np.ndarray, mn: np.ndarray, mx: np.ndarray) -> np.ndarray:
    rng = mx - mn
    rng[rng == 0] = 1.0
    return arr * rng + mn

def pct_change(close: np.ndarray) -> np.ndarray:
    out = np.zeros_like(close)
    out[1:] = (close[1:] - close[:-1]) / (close[:-1] + 1e-8)
    return out

def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()

def save_config(cfg, path: str):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(asdict(cfg), f, ensure_ascii=False, indent=2)

def load_config(cfg_cls, path: str):
    with open(path, "r", encoding="utf-8") as f:
        return cfg_cls(**json.load(f))

# ============================================================
# 日志
# ============================================================
def setup_logger(name: str, log_file: Optional[str] = None, level=logging.INFO):
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
    logger.setLevel(level)
    fmt = logging.Formatter("[%(asctime)s][%(name)s][%(levelname)s] %(message)s",
                            datefmt="%H:%M:%S")
    sh = logging.StreamHandler()
    sh.setFormatter(fmt)
    logger.addHandler(sh)
    if log_file:
        fh = logging.FileHandler(log_file, encoding="utf-8")
        fh.setFormatter(fmt)
        logger.addHandler(fh)
    return logger

"""
stock_config.py - 对抗学习量化系统 共享配置与工具
=================================================
项目：证券市场股票及基金对抗学习量化脚本
版本：v2.0 (从头重建)
管线：数据获取 → 生成器对抗 → 庄散对抗 → 结果分析
"""

import os, json, logging, hashlib, numpy as np
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Tuple
from pathlib import Path
from datetime import datetime

# ============================================================
# 路径配置 (本地Windows环境)
# ============================================================
BASE_DIR = Path(r"C:\Users\HUAWEI\Desktop\Adversarial Learning")
SCRIPT_DIR = BASE_DIR / "dotpy"            # 脚本所在目录
DATA_DIR = BASE_DIR / "stockdata"           # 日K线数据
ADV_DATA_DIR = BASE_DIR / "adversarial data" # 对抗训练数据
ADV_MODEL_DIR = BASE_DIR / "adversarial model" # 对抗模型
RESULTS_DIR = BASE_DIR / "stockresults"     # 结果输出
FEEDBACK_FILE = SCRIPT_DIR / "feedback.txt" # 实盘反馈文件
PYMANAGER_FILE = SCRIPT_DIR / "pymanager.txt" # 命令手册

for d in [DATA_DIR, ADV_DATA_DIR, ADV_MODEL_DIR, RESULTS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ============================================================
# Python解释器
# ============================================================
PYTHON_EXE = r"C:\Users\HUAWEI\AppData\Local\Programs\Python\Python312\python.exe"

# ============================================================
# Baostock 数据源配置
# ============================================================
BS_FIELDS_DAILY = [
    "date", "code", "open", "high", "low", "close",
    "preclose", "volume", "amount", "adjustflag",
    "turn", "tradestatus", "pctChg", "isST"
]
BS_FIELDS_MIN5 = [
    "date", "time", "code", "open", "high", "low", "close",
    "volume", "amount", "adjustflag"
]
DAILY_START = "2018-01-01"
DAILY_END = "2026-06-20"
MIN5_START = "2024-01-01"
FOCUS_POOL_SIZE = 300

# ============================================================
# 行业映射 (申万一级)
# ============================================================
SW_INDUSTRY_MAP = {
    "801010": "农林牧渔", "801020": "采掘", "801030": "化工",
    "801040": "钢铁", "801050": "有色金属", "801080": "电子",
    "801110": "家用电器", "801120": "食品饮料", "801130": "纺织服装",
    "801140": "轻工制造", "801150": "医药生物", "801160": "公用事业",
    "801170": "交通运输", "801180": "房地产", "801200": "商业贸易",
    "801210": "休闲服务", "801230": "综合", "801710": "建筑材料",
    "801720": "建筑装饰", "801730": "电气设备", "801740": "国防军工",
    "801750": "计算机", "801760": "传媒", "801770": "通信",
    "801780": "银行", "801790": "非银金融", "801880": "汽车",
    "801890": "机械设备",
}

# ============================================================
# 生成器 C-TimeGAN 配置
# ============================================================
@dataclass
class GeneratorConfig:
    hidden_dim: int = 64
    num_layers: int = 3
    seq_len: int = 30
    feature_dim: int = 6         # OHLCV + pctChg
    condition_levels: int = 3    # 宏观→板块→个股
    phase_a_epochs: int = 100
    phase_b_epochs: int = 100
    phase_c_epochs: int = 200
    batch_size: int = 64
    learning_rate: float = 1e-3
    # 验证
    dtw_threshold: float = 2.0
    pearson_threshold: float = 0.85
    ks_pvalue: float = 0.05
    # 开盘权重
    w930_940: float = 5.0
    w940_950: float = 2.0
    w_other: float = 1.0
    # 查重
    max_similar: int = 3

# ============================================================
# 对抗环境配置
# ============================================================
@dataclass
class AdversarialConfig:
    num_episodes: int = 1000
    episode_length: int = 240    # 4h × 60min ticks
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
    # 校准参数
    calibration_window: int = 20       # 最近N笔交易用于校准
    signal_accuracy_thresh: float = 0.55  # 信号准确率低于此值触发重训
    max_position_adjust: float = 0.1   # 校准后仓位最大调整幅度
    reward_shaping_weight: float = 0.3  # 实盘反馈对reward的权重

# ============================================================
# DeepSeek API
# ============================================================
DS_API_KEY = "sk-a50618b3cc7c40e2a633eab13a59969e"
DS_API_URL = "https://api.deepseek.com/chat/completions"
DS_MODEL = "deepseek-chat"
DS_MAX_TOKENS = 4096

# ============================================================
# 日志
# ============================================================
def setup_logger(name: str, log_file: Optional[str] = None, level=logging.INFO):
    logger = logging.getLogger(name)
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

# ============================================================
# 数据工具
# ============================================================
def normalize_ohlcv(arr: np.ndarray) -> np.ndarray:
    """Min-max归一化到[0,1]"""
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

# ============================================================
# 保存/加载配置
# ============================================================
def save_config(cfg, path: str):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(asdict(cfg), f, ensure_ascii=False, indent=2)

def load_config(cfg_cls, path: str):
    with open(path, "r", encoding="utf-8") as f:
        return cfg_cls(**json.load(f))

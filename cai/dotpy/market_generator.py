"""
A股市场数据生成器 - TimeGAN架构
项目: A股对抗学习研究 (Adversarial Learning for A-Stock Market)
版本: v1.0
创建日期: 2025
设计意图: 实现Yoon et al.(2019) TimeGAN，用于生成"以假乱真"的A股价格序列

TimeGAN架构说明:
- Phase A: 自编码器预训练 (Embedder + Recovery)
- Phase B: 监督训练 (Supervisor学习真实step-wise转移)
- Phase C: 联合对抗训练 (所有组件，4个损失联合优化)
"""

import os
import sys
import json
import argparse
import warnings
from pathlib import Path
from datetime import datetime
from typing import Optional, Tuple, List, Dict, Any
from dataclasses import dataclass, asdict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

warnings.filterwarnings('ignore')

# ============================================================================
# 路径配置
# ============================================================================

def get_paths():
    """获取跨平台路径配置"""
    if sys.platform == 'win32':
        base_dir = Path(os.path.expanduser("~")) / "Desktop" / "Adversarial Learning"
        data_dir = base_dir / "stockdata"
        model_dir = base_dir / "adversarial model" / "generator"
        output_dir = base_dir / "adversarial data"
    else:
        base_dir = Path("/app/data/所有对话/主对话")
        data_dir = base_dir / "stockdata" / "daily"
        model_dir = base_dir / "adversarial model" / "generator"
        output_dir = base_dir / "adversarial data"
    
    model_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    return data_dir, model_dir, output_dir

DATA_DIR, MODEL_DIR, OUTPUT_DIR = get_paths()

# ============================================================================
# 配置类
# ============================================================================

@dataclass
class Config:
    """TimeGAN模型配置"""
    # 模型结构
    hidden_dim: int = 24
    num_layers: int = 2
    latent_dim: int = 24
    seq_len: int = 60
    feature_dim: int = 3  # 对数收益率, 成交量变化率, 滚动波动率
    
    # 训练参数
    batch_size: int = 64
    lr: float = 0.001
    phase_a_epochs: int = 100
    phase_b_epochs: int = 100
    phase_c_epochs: int = 200
    
    # 验证参数
    acf_lags: int = 20
    
    # 路径
    data_dir: str = str(DATA_DIR)
    model_dir: str = str(MODEL_DIR)
    output_dir: str = str(OUTPUT_DIR)
    
    # 设备
    device: str = "cpu"
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

# ============================================================================
# 数据处理
# ============================================================================

class StockDataset(Dataset):
    """股票序列数据集"""
    
    def __init__(self, data: np.ndarray):
        """
        Args:
            data: (n_samples, seq_len, feature_dim) 的标准化数据
        """
        self.data = torch.FloatTensor(data)
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> torch.Tensor:
        return self.data[idx]


def load_and_preprocess(data_dir: Path, seq_len: int = 60, step: int = 20) -> np.ndarray:
    """
    加载CSV数据并进行预处理
    
    处理流程:
    1. 读取CSV文件
    2. 计算对数收益率: log(close_t / close_{t-1})
    3. 计算成交量变化率: log(volume_t / volume_{t-1})
    4. 计算滚动波动率 (20日std)
    5. 分段处理 (seq_len长度，滑动窗口step)
    6. 标准化 (每段减均值除标准差)
    
    Returns:
        normalized_data: (n_samples, seq_len, feature_dim)
    """
    import glob
    
    # 查找CSV文件
    csv_files = list(data_dir.glob("**/*.csv"))
    
    if not csv_files:
        print(f"[警告] 在 {data_dir} 中未找到CSV文件，生成模拟数据用于测试")
        return generate_synthetic_data(n_samples=1000, seq_len=seq_len)
    
    print(f"[INFO] 找到 {len(csv_files)} 个CSV文件")
    
    all_sequences = []
    
    for csv_file in csv_files[:50]:  # 限制股票数量加速
        try:
            df = load_single_stock(csv_file, seq_len, step)
            if df is not None and len(df) > 0:
                all_sequences.append(df)
        except Exception as e:
            print(f"[警告] 处理 {csv_file.name} 失败: {e}")
            continue
    
    if not all_sequences:
        print(f"[警告] 未能成功处理任何股票数据，生成模拟数据")
        return generate_synthetic_data(n_samples=1000, seq_len=seq_len)
    
    # 合并所有序列
    data = np.vstack(all_sequences)
    print(f"[INFO] 预处理完成，共 {len(data)} 个序列样本")
    
    return data


def load_single_stock(csv_path: Path, seq_len: int, step: int) -> Optional[np.ndarray]:
    """加载并处理单个股票数据"""
    try:
        import pandas as pd
        
        # 读取CSV
        df = pd.read_csv(csv_path)
        
        # 检查必要字段
        required_cols = ['close', 'volume']
        if not all(col in df.columns for col in required_cols):
            return None
        
        # 按日期排序
        if 'date' in df.columns:
            df = df.sort_values('date')
        
        close_prices = df['close'].values
        volumes = df['volume'].values
        
        # 计算特征
        features = compute_features(close_prices, volumes)
        
        if features is None or len(features) < seq_len:
            return None
        
        # 分段处理
        sequences = []
        for i in range(0, len(features) - seq_len + 1, step):
            seq = features[i:i + seq_len]
            
            # 标准化 (每段独立)
            mean = np.nanmean(seq, axis=0, keepdims=True)
            std = np.nanstd(seq, axis=0, keepdims=True)
            std = np.where(std < 1e-8, 1.0, std)  # 防止除零
            
            seq_normalized = (seq - mean) / std
            sequences.append(seq_normalized)
        
        return np.array(sequences) if sequences else None
        
    except Exception as e:
        print(f"[错误] 加载 {csv_path.name} 失败: {e}")
        return None


def compute_features(close_prices: np.ndarray, volumes: np.ndarray) -> Optional[np.ndarray]:
    """
    计算时间序列特征
    
    Returns:
        features: (n_timesteps, 3) - [对数收益率, 成交量变化率, 滚动波动率]
    """
    n = len(close_prices)
    if n < 21:  # 需要至少21天计算20日波动率
        return None
    
    features = np.zeros((n, 3))
    
    # 1. 对数收益率
    close_shifted = np.roll(close_prices, 1)
    close_shifted[0] = close_prices[0]
    log_returns = np.log(close_prices / close_shifted)
    log_returns[0] = 0
    features[:, 0] = log_returns
    
    # 2. 成交量变化率
    vol_shifted = np.roll(volumes, 1)
    vol_shifted[0] = volumes[0]
    vol_change = np.log((volumes + 1) / (vol_shifted + 1))
    vol_change[0] = 0
    features[:, 1] = vol_change
    
    # 3. 滚动波动率 (20日std)
    window = 20
    returns_for_vol = log_returns[1:]  # 避免重复计算
    for i in range(1, n):
        start_idx = max(1, i - window + 1)
        end_idx = i + 1
        features[i, 2] = np.std(log_returns[start_idx:end_idx])
    
    # 处理NaN和Inf
    features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
    
    return features


def generate_synthetic_data(n_samples: int = 1000, seq_len: int = 60) -> np.ndarray:
    """
    生成模拟股票数据用于测试
    模拟真实A股市场的特征: 波动率聚集、自相关、均值回复
    """
    print(f"[INFO] 生成 {n_samples} 个模拟序列用于测试")
    
    data = []
    
    for _ in range(n_samples):
        # 初始化
        sequence = np.zeros((seq_len, 3))
        
        # 生成随机游走 + 波动率聚集效应
        volatility = 0.02  # 基础波动率
        
        for t in range(seq_len):
            # 更新波动率 (GARCH-like)
            volatility = 0.01 + 0.9 * volatility + 0.05 * np.random.randn() ** 2
            volatility = max(0.005, min(0.1, volatility))
            
            # 生成对数收益率
            log_return = np.random.randn() * volatility
            sequence[t, 0] = log_return
            
            # 成交量变化率 (与收益率相关)
            volume_change = log_return * 2 + np.random.randn() * 0.3
            sequence[t, 1] = volume_change
            
            # 波动率
            sequence[t, 2] = volatility
        
        # 标准化
        mean = np.mean(sequence, axis=0, keepdims=True)
        std = np.std(sequence, axis=0, keepdims=True)
        std = np.where(std < 1e-8, 1.0, std)
        sequence = (sequence - mean) / std
        
        data.append(sequence)
    
    return np.array(data)


# ============================================================================
# TimeGAN 模型组件
# ============================================================================

class Embedder(nn.Module):
    """
    Embedder: 将真实价格序列编码到低维潜空间
    使用GRU进行时序编码
    """
    
    def __init__(self, feature_dim: int, hidden_dim: int, num_layers: int):
        super().__init__()
        self.gru = nn.GRU(
            input_size=feature_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, feature_dim) 真实序列
        Returns:
            h: (batch, seq_len, hidden_dim) 潜表示
        """
        output, h = self.gru(x)
        return output


class Recovery(nn.Module):
    """
    Recovery: 从潜空间解码回原始空间
    使用GRU进行时序解码
    """
    
    def __init__(self, hidden_dim: int, feature_dim: int, num_layers: int):
        super().__init__()
        self.gru = nn.GRU(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True
        )
        self.fc = nn.Linear(hidden_dim, feature_dim)
    
    def forward(self, h: torch.Tensor) -> torch.Tensor:
        """
        Args:
            h: (batch, seq_len, hidden_dim) 潜表示
        Returns:
            x_recon: (batch, seq_len, feature_dim) 重建序列
        """
        output, _ = self.gru(h)
        x_recon = self.fc(output)
        return x_recon


class Generator(nn.Module):
    """
    Generator: 在潜空间生成序列
    输入: 随机噪声
    输出: 潜空间序列
    """
    
    def __init__(self, latent_dim: int, hidden_dim: int, num_layers: int, seq_len: int):
        super().__init__()
        self.seq_len = seq_len
        self.hidden_dim = hidden_dim
        
        # 将随机噪声映射到初始隐藏状态
        self.fc_init = nn.Linear(latent_dim, hidden_dim * num_layers)
        
        self.gru = nn.GRU(
            input_size=latent_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True
        )
    
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z: (batch, seq_len, latent_dim) 随机噪声
        Returns:
            h: (batch, seq_len, hidden_dim) 生成的潜序列
        """
        batch_size = z.size(0)
        
        # 初始化GRU隐藏状态
        h0 = self.fc_init(z[:, 0, :])  # 使用第一时刻的噪声
        h0 = h0.view(batch_size, -1, self.hidden_dim)
        h0 = h0.permute(1, 0, 2).contiguous()  # (num_layers, batch, hidden)
        
        output, _ = self.gru(z, h0)
        return output


class Supervisor(nn.Module):
    """
    Supervisor: 学习潜空间中的step-wise转移
    用于指导生成器产生更真实的序列
    """
    
    def __init__(self, hidden_dim: int, num_layers: int):
        super().__init__()
        self.gru = nn.GRU(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True
        )
    
    def forward(self, h: torch.Tensor) -> torch.Tensor:
        """
        Args:
            h: (batch, seq_len, hidden_dim) 潜表示
        Returns:
            h_supervised: (batch, seq_len, hidden_dim) 监督后的潜表示
        """
        output, _ = self.gru(h)
        return output


class Discriminator(nn.Module):
    """
    Discriminator: 区分真实vs生成潜表示
    使用MLP进行分类
    """
    
    def __init__(self, hidden_dim: int, num_layers: int = 2):
        super().__init__()
        
        layers = []
        input_dim = hidden_dim
        
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.LeakyReLU(0.2))
            input_dim = hidden_dim
        
        layers.append(nn.Linear(input_dim, 1))
        self.mlp = nn.Sequential(*layers)
    
    def forward(self, h: torch.Tensor) -> torch.Tensor:
        """
        Args:
            h: (batch, seq_len, hidden_dim) 潜表示
        Returns:
            logits: (batch, seq_len, 1) 判别结果
        """
        # 对时间维度取平均
        h_mean = torch.mean(h, dim=1)  # (batch, hidden_dim)
        logits = self.mlp(h_mean)  # (batch, 1)
        return logits


class TimeGAN(nn.Module):
    """
    TimeGAN完整模型
    整合所有组件用于联合训练
    """
    
    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        
        # 五个核心组件
        self.embedder = Embedder(
            config.feature_dim, config.hidden_dim, config.num_layers
        )
        self.recovery = Recovery(
            config.hidden_dim, config.feature_dim, config.num_layers
        )
        self.generator = Generator(
            config.latent_dim, config.hidden_dim, config.num_layers, config.seq_len
        )
        self.supervisor = Supervisor(config.hidden_dim, config.num_layers)
        self.discriminator = Discriminator(config.hidden_dim, config.num_layers)
    
    def _create_noise(self, batch_size: int) -> torch.Tensor:
        """创建随机噪声"""
        z = torch.randn(batch_size, self.config.seq_len, self.config.latent_dim)
        return z.to(self.config.device)
    
    def train_autoencoder(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        自编码器前向传播 (Phase A)
        """
        # 编码
        h = self.embedder(x)
        # 解码
        x_recon = self.recovery(h)
        return h, x_recon
    
    def train_supervisor(self, x: torch.Tensor) -> torch.Tensor:
        """
        Supervisor前向传播 (Phase B)
        """
        h = self.embedder(x)
        h_supervised = self.supervisor(h)
        return h_supervised
    
    def train_adversarial(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        联合对抗训练前向传播 (Phase C)
        """
        # 真实数据流程
        h_real = self.embedder(x)
        h_real_supervised = self.supervisor(h_real)
        
        # 生成数据流程
        z = self._create_noise(x.size(0))
        h_gen = self.generator(z)
        h_gen_supervised = self.supervisor(h_gen)
        
        # 判别
        d_real = self.discriminator(h_real_supervised)
        d_fake = self.discriminator(h_gen_supervised.detach())
        
        return {
            'h_real': h_real,
            'h_real_supervised': h_real_supervised,
            'h_gen': h_gen,
            'h_gen_supervised': h_gen_supervised,
            'd_real': d_real,
            'd_fake': d_fake
        }
    
    def generate(self, n_samples: int) -> torch.Tensor:
        """
        生成虚拟数据
        """
        self.eval()
        with torch.no_grad():
            z = self._create_noise(n_samples)
            h_gen = self.generator(z)
            h_gen_supervised = self.supervisor(h_gen)
            x_gen = self.recovery(h_gen_supervised)
        return x_gen


# ============================================================================
# 损失函数
# ============================================================================

class TimeGANLoss:
    """TimeGAN的四个损失函数"""
    
    def __init__(self, device: str = "cpu"):
        self.bce = nn.BCEWithLogitsLoss()
        self.mse = nn.MSELoss()
        self.device = device
    
    def reconstruction_loss(self, x: torch.Tensor, x_recon: torch.Tensor) -> torch.Tensor:
        """重建损失: 自编码器重建真实数据的准确性"""
        return self.mse(x_recon, x)
    
    def supervised_loss(self, h: torch.Tensor, h_supervised: torch.Tensor) -> torch.Tensor:
        """监督损失: Supervisor学习真实step-wise转移"""
        return self.mse(h_supervised[:, 1:, :], h[:, 1:, :])
    
    def generator_loss(self, d_fake: torch.Tensor) -> torch.Tensor:
        """生成器损失: 欺骗判别器"""
        real_labels = torch.ones_like(d_fake).to(self.device)
        return self.bce(d_fake, real_labels)
    
    def discriminator_loss(self, d_real: torch.Tensor, d_fake: torch.Tensor) -> torch.Tensor:
        """判别器损失: 区分真实与生成"""
        # 真实数据应被判别为真
        real_labels = torch.ones_like(d_real).to(self.device)
        loss_real = self.bce(d_real, real_labels)
        
        # 生成数据应被判别为假
        fake_labels = torch.zeros_like(d_fake).to(self.device)
        loss_fake = self.bce(d_fake, fake_labels)
        
        return loss_real + loss_fake


# ============================================================================
# 训练器
# ============================================================================

class TimeGANTrainer:
    """TimeGAN训练器"""
    
    def __init__(self, config: Config):
        self.config = config
        self.device = torch.device(config.device)
        
        # 模型
        self.model = TimeGAN(config).to(self.device)
        self.loss_fn = TimeGANLoss(self.device)
        
        # 优化器
        self.setup_optimizers()
        
        # 训练状态
        self.phase_a_loss = []
        self.phase_b_loss = []
        self.phase_c_generator_loss = []
        self.phase_c_discriminator_loss = []
        
        # 检查点路径
        self.checkpoint_path = Path(config.model_dir) / "timegan_checkpoint.pt"
        self.best_model_path = Path(config.model_dir) / "timegan_best.pt"
    
    def setup_optimizers(self):
        """设置优化器"""
        lr = self.config.lr
        
        # Phase A: 自编码器
        self.opt_ae = torch.optim.Adam(
            list(self.model.embedder.parameters()) + 
            list(self.model.recovery.parameters()),
            lr=lr
        )
        
        # Phase B: Supervisor
        self.opt_supervisor = torch.optim.Adam(
            list(self.model.embedder.parameters()) +
            list(self.model.supervisor.parameters()),
            lr=lr
        )
        
        # Phase C: 联合优化
        self.opt_embedder = torch.optim.Adam(
            self.model.embedder.parameters(), lr=lr
        )
        self.opt_recovery = torch.optim.Adam(
            self.model.recovery.parameters(), lr=lr
        )
        self.opt_generator = torch.optim.Adam(
            list(self.model.generator.parameters()) +
            list(self.model.supervisor.parameters()),
            lr=lr
        )
        self.opt_discriminator = torch.optim.Adam(
            self.model.discriminator.parameters(), lr=lr
        )
    
    def save_checkpoint(self, phase: str, epoch: int):
        """保存检查点"""
        checkpoint = {
            'phase': phase,
            'epoch': epoch,
            'model_state': self.model.state_dict(),
            'opt_ae_state': self.opt_ae.state_dict(),
            'opt_supervisor_state': self.opt_supervisor.state_dict(),
            'opt_embedder_state': self.opt_embedder.state_dict(),
            'opt_recovery_state': self.opt_recovery.state_dict(),
            'opt_generator_state': self.opt_generator.state_dict(),
            'opt_discriminator_state': self.opt_discriminator.state_dict(),
            'phase_a_loss': self.phase_a_loss,
            'phase_b_loss': self.phase_b_loss,
            'phase_c_generator_loss': self.phase_c_generator_loss,
            'phase_c_discriminator_loss': self.phase_c_discriminator_loss,
            'config': self.config.to_dict()
        }
        torch.save(checkpoint, self.checkpoint_path)
        print(f"[保存] 检查点已保存到 {self.checkpoint_path}")
    
    def load_checkpoint(self) -> Tuple[Optional[str], int]:
        """加载检查点"""
        if not self.checkpoint_path.exists():
            return None, 0
        
        checkpoint = torch.load(self.checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state'])
        self.opt_ae.load_state_dict(checkpoint['opt_ae_state'])
        self.opt_supervisor.load_state_dict(checkpoint['opt_supervisor_state'])
        self.opt_embedder.load_state_dict(checkpoint['opt_embedder_state'])
        self.opt_recovery.load_state_dict(checkpoint['opt_recovery_state'])
        self.opt_generator.load_state_dict(checkpoint['opt_generator_state'])
        self.opt_discriminator.load_state_dict(checkpoint['opt_discriminator_state'])
        self.phase_a_loss = checkpoint.get('phase_a_loss', [])
        self.phase_b_loss = checkpoint.get('phase_b_loss', [])
        self.phase_c_generator_loss = checkpoint.get('phase_c_generator_loss', [])
        self.phase_c_discriminator_loss = checkpoint.get('phase_c_discriminator_loss', [])
        
        phase = checkpoint.get('phase', 'phase_a')
        epoch = checkpoint.get('epoch', 0)
        print(f"[加载] 从检查点恢复: {phase}, epoch {epoch}")
        
        return phase, epoch
    
    def train_phase_a(self, dataloader: DataLoader, epochs: int, resume: bool = False):
        """
        Phase A: 自编码器预训练
        目标: 学习重建真实数据的表示
        """
        start_epoch = 0
        if resume:
            phase, start_epoch = self.load_checkpoint()
        
        print(f"\n{'='*60}")
        print(f"[Phase A] 自编码器预训练 (Embedder + Recovery)")
        print(f"{'='*60}")
        
        for epoch in range(start_epoch, epochs):
            epoch_loss = 0.0
            n_batches = 0
            
            for batch in dataloader:
                batch = batch.to(self.device)
                
                # 前向传播
                h, x_recon = self.model.train_autoencoder(batch)
                
                # 重建损失
                loss = self.loss_fn.reconstruction_loss(batch, x_recon)
                
                # 反向传播
                self.opt_ae.zero_grad()
                loss.backward()
                self.opt_ae.step()
                
                epoch_loss += loss.item()
                n_batches += 1
            
            avg_loss = epoch_loss / n_batches
            self.phase_a_loss.append(avg_loss)
            
            if (epoch + 1) % 10 == 0:
                print(f"  [Epoch {epoch+1}/{epochs}] 重建损失: {avg_loss:.6f}")
            
            # 保存检查点
            if (epoch + 1) % 50 == 0:
                self.save_checkpoint('phase_a', epoch + 1)
        
        # 保存Phase A最佳模型
        self.save_checkpoint('phase_a', epochs)
        print(f"[Phase A] 完成，最终损失: {self.phase_a_loss[-1]:.6f}")
    
    def train_phase_b(self, dataloader: DataLoader, epochs: int, resume: bool = False):
        """
        Phase B: 监督训练
        目标: Supervisor学习真实step-wise转移
        """
        start_epoch = 0
        if resume:
            _, start_epoch = self.load_checkpoint()
        
        print(f"\n{'='*60}")
        print(f"[Phase B] 监督训练 (Supervisor)")
        print(f"{'='*60}")
        
        for epoch in range(start_epoch, epochs):
            epoch_loss = 0.0
            n_batches = 0
            
            for batch in dataloader:
                batch = batch.to(self.device)
                
                # 前向传播
                h = self.model.embedder(batch)
                h_supervised = self.model.supervisor(h)
                
                # 监督损失
                loss = self.loss_fn.supervised_loss(h, h_supervised)
                
                # 反向传播
                self.opt_supervisor.zero_grad()
                loss.backward()
                self.opt_supervisor.step()
                
                epoch_loss += loss.item()
                n_batches += 1
            
            avg_loss = epoch_loss / n_batches
            self.phase_b_loss.append(avg_loss)
            
            if (epoch + 1) % 10 == 0:
                print(f"  [Epoch {epoch+1}/{epochs}] 监督损失: {avg_loss:.6f}")
            
            if (epoch + 1) % 50 == 0:
                self.save_checkpoint('phase_b', epoch + 1)
        
        self.save_checkpoint('phase_b', epochs)
        print(f"[Phase B] 完成，最终损失: {self.phase_b_loss[-1]:.6f}")
    
    def train_phase_c(self, dataloader: DataLoader, epochs: int, resume: bool = False):
        """
        Phase C: 联合对抗训练
        目标: 同时优化所有组件，平衡重建质量和生成质量
        """
        start_epoch = 0
        if resume:
            _, start_epoch = self.load_checkpoint()
        
        print(f"\n{'='*60}")
        print(f"[Phase C] 联合对抗训练 (全部组件)")
        print(f"{'='*60}")
        
        best_g_loss = float('inf')
        
        for epoch in range(start_epoch, epochs):
            epoch_g_loss = 0.0
            epoch_d_loss = 0.0
            n_batches = 0
            
            for batch in dataloader:
                batch = batch.to(self.device)
                batch_size = batch.size(0)
                
                # ===== 训练判别器 =====
                for _ in range(1):  # 判别器训练次数
                    results = self.model.train_adversarial(batch)
                    
                    d_loss = self.loss_fn.discriminator_loss(
                        results['d_real'], results['d_fake']
                    )
                    
                    self.opt_discriminator.zero_grad()
                    d_loss.backward()
                    self.opt_discriminator.step()
                
                # ===== 训练生成器 =====
                # 自编码器重建
                h, x_recon = self.model.train_autoencoder(batch)
                loss_recon = self.loss_fn.reconstruction_loss(batch, x_recon)
                
                # Supervisor监督
                h_supervised = self.model.supervisor(h)
                loss_supervised = self.loss_fn.supervised_loss(h, h_supervised)
                
                # 生成器欺骗判别器
                results = self.model.train_adversarial(batch)
                loss_generator = self.loss_fn.generator_loss(results['d_fake'])
                
                # 联合损失
                g_loss = loss_recon + loss_supervised + loss_generator
                
                self.opt_embedder.zero_grad()
                self.opt_recovery.zero_grad()
                self.opt_generator.zero_grad()
                g_loss.backward()
                self.opt_embedder.step()
                self.opt_recovery.step()
                self.opt_generator.step()
                
                epoch_g_loss += loss_generator.item()
                epoch_d_loss += d_loss.item()
                n_batches += 1
            
            avg_g_loss = epoch_g_loss / n_batches
            avg_d_loss = epoch_d_loss / n_batches
            self.phase_c_generator_loss.append(avg_g_loss)
            self.phase_c_discriminator_loss.append(avg_d_loss)
            
            # 保存最佳模型
            if avg_g_loss < best_g_loss:
                best_g_loss = avg_g_loss
                torch.save(self.model.state_dict(), self.best_model_path)
                print(f"  [*] 保存最佳生成器模型 (G_loss: {best_g_loss:.6f})")
            
            if (epoch + 1) % 10 == 0:
                print(f"  [Epoch {epoch+1}/{epochs}] G_loss: {avg_g_loss:.4f}, D_loss: {avg_d_loss:.4f}")
            
            if (epoch + 1) % 50 == 0:
                self.save_checkpoint('phase_c', epoch + 1)
        
        self.save_checkpoint('phase_c', epochs)
        print(f"[Phase C] 完成")
    
    def train(self, data: np.ndarray, resume: bool = False):
        """完整三阶段训练"""
        # 创建数据集
        dataset = StockDataset(data)
        dataloader = DataLoader(
            dataset, 
            batch_size=self.config.batch_size, 
            shuffle=True,
            num_workers=0
        )
        
        # Phase A: 自编码器预训练
        self.train_phase_a(dataloader, self.config.phase_a_epochs, resume)
        
        # Phase B: 监督训练
        self.train_phase_b(dataloader, self.config.phase_b_epochs, resume)
        
        # Phase C: 联合对抗训练
        self.train_phase_c(dataloader, self.config.phase_c_epochs, resume)
        
        # 保存最终模型
        final_path = Path(self.config.model_dir) / "timegan_final.pt"
        torch.save(self.model.state_dict(), final_path)
        print(f"[保存] 最终模型已保存到 {final_path}")
        
        # 保存训练日志
        self.save_training_log()
    
    def save_training_log(self):
        """保存训练日志"""
        log_path = Path(self.config.output_dir) / "training_log.json"
        
        log = {
            'timestamp': datetime.now().isoformat(),
            'config': self.config.to_dict(),
            'phase_a_loss': self.phase_a_loss[-10:] if self.phase_a_loss else [],
            'phase_b_loss': self.phase_b_loss[-10:] if self.phase_b_loss else [],
            'phase_c_generator_loss': self.phase_c_generator_loss[-10:] if self.phase_c_generator_loss else [],
            'phase_c_discriminator_loss': self.phase_c_discriminator_loss[-10:] if self.phase_c_discriminator_loss else [],
            'final_phase_a_loss': self.phase_a_loss[-1] if self.phase_a_loss else None,
            'final_phase_b_loss': self.phase_b_loss[-1] if self.phase_b_loss else None,
            'final_phase_c_g_loss': self.phase_c_generator_loss[-1] if self.phase_c_generator_loss else None,
            'final_phase_c_d_loss': self.phase_c_discriminator_loss[-1] if self.phase_c_discriminator_loss else None
        }
        
        with open(log_path, 'w', encoding='utf-8') as f:
            json.dump(log, f, indent=2, ensure_ascii=False)
        
        print(f"[保存] 训练日志已保存到 {log_path}")


# ============================================================================
# 生成器
# ============================================================================

class SyntheticDataGenerator:
    """合成数据生成器"""
    
    def __init__(self, model: TimeGAN, device: str = "cpu"):
        self.model = model
        self.device = device
    
    def generate(self, n_samples: int) -> np.ndarray:
        """
        生成合成股票序列
        
        Args:
            n_samples: 生成样本数量
        
        Returns:
            generated_data: (n_samples, seq_len, feature_dim)
        """
        self.model.eval()
        
        with torch.no_grad():
            z = torch.randn(n_samples, self.model.config.seq_len, self.model.config.latent_dim)
            z = z.to(self.device)
            
            # 生成流程
            h_gen = self.model.generator(z)
            h_gen_supervised = self.model.supervisor(h_gen)
            x_gen = self.model.recovery(h_gen_supervised)
        
        return x_gen.cpu().numpy()
    
    def save(self, data: np.ndarray, output_path: Path):
        """保存生成的数据"""
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        np.save(output_path, data)
        print(f"[保存] 生成数据已保存到 {output_path}")
        print(f"[INFO] 形状: {data.shape}")


# ============================================================================
# 三级验证系统
# ============================================================================

class TimeGANValidator:
    """TimeGAN验证器 - 三级验证体系"""
    
    def __init__(self, real_data: np.ndarray, generated_data: np.ndarray):
        self.real_data = real_data
        self.generated_data = generated_data
    
    def validate_level1_statistics(self) -> Dict[str, Any]:
        """
        Level 1: 统计匹配验证
        检查生成数据的统计特性是否与真实数据匹配
        """
        print(f"\n{'='*60}")
        print(f"[Level 1] 统计匹配验证")
        print(f"{'='*60}")
        
        results = {}
        
        # 对每个特征分别验证
        feature_names = ['对数收益率', '成交量变化率', '滚动波动率']
        
        for feat_idx, feat_name in enumerate(feature_names):
            real_feat = self.real_data[:, :, feat_idx].flatten()
            gen_feat = self.generated_data[:, :, feat_idx].flatten()
            
            # 基本统计量
            real_mean, real_std = np.mean(real_feat), np.std(real_feat)
            gen_mean, gen_std = np.mean(gen_feat), np.std(gen_feat)
            
            # 偏度和峰度
            real_skew = self._skewness(real_feat)
            gen_skew = self._skewness(gen_feat)
            real_kurt = self._kurtosis(real_feat)
            gen_kurt = self._kurtosis(gen_feat)
            
            # 自相关函数 (ACF)
            real_acf = self._acf(real_feat, lags=20)
            gen_acf = self._acf(gen_feat, lags=20)
            acf_diff = np.mean(np.abs(real_acf - gen_acf))
            
            feat_result = {
                'mean': {'real': real_mean, 'generated': gen_mean, 'diff': abs(real_mean - gen_mean)},
                'std': {'real': real_std, 'generated': gen_std, 'diff': abs(real_std - gen_std)},
                'skewness': {'real': real_skew, 'generated': gen_skew, 'diff': abs(real_skew - gen_skew)},
                'kurtosis': {'real': real_kurt, 'generated': gen_kurt, 'diff': abs(real_kurt - gen_kurt)},
                'acf_mean_diff': acf_diff
            }
            
            results[feat_name] = feat_result
            
            print(f"\n  [{feat_name}]")
            print(f"    均值: 真实={real_mean:.4f}, 生成={gen_mean:.4f}, 差异={abs(real_mean - gen_mean):.4f}")
            print(f"    标准差: 真实={real_std:.4f}, 生成={gen_std:.4f}, 差异={abs(real_std - gen_std):.4f}")
            print(f"    偏度: 真实={real_skew:.4f}, 生成={gen_skew:.4f}")
            print(f"    峰度: 真实={real_kurt:.4f}, 生成={gen_kurt:.4f}")
            print(f"    ACF平均差异: {acf_diff:.4f}")
        
        # 总体评分
        all_diffs = []
        for feat_result in results.values():
            all_diffs.append(feat_result['mean']['diff'])
            all_diffs.append(feat_result['std']['diff'])
            all_diffs.append(feat_result['acf_mean_diff'])
        
        avg_diff = np.mean(all_diffs)
        results['overall_score'] = 1.0 / (1.0 + avg_diff)  # 越高越好
        
        print(f"\n  [总体评分] {results['overall_score']:.4f} (1.0为最佳)")
        
        return results
    
    def validate_level2_discriminator(self) -> Dict[str, Any]:
        """
        Level 2: 判别器测试
        训练一个独立分类器区分真假，期望准确率≈50%
        """
        print(f"\n{'='*60}")
        print(f"[Level 2] 判别器测试")
        print(f"{'='*60}")
        
        # 准备数据
        X_real = self.real_data.reshape(len(self.real_data), -1)
        X_gen = self.generated_data.reshape(len(self.generated_data), -1)
        
        # 确保数量一致
        n_samples = min(len(X_real), len(X_gen))
        X_real = X_real[:n_samples]
        X_gen = X_gen[:n_samples]
        
        # 创建标签
        y_real = np.ones(n_samples)
        y_gen = np.zeros(n_samples)
        
        # 合并数据
        X = np.vstack([X_real, X_gen])
        y = np.hstack([y_real, y_gen])
        
        # 打乱
        indices = np.random.permutation(len(X))
        X = X[indices]
        y = y[indices]
        
        # 划分训练/测试集
        split_idx = int(0.7 * len(X))
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        
        # 训练简单分类器 (逻辑回归)
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import StandardScaler
        
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        clf = LogisticRegression(max_iter=1000, random_state=42)
        clf.fit(X_train_scaled, y_train)
        
        # 评估
        accuracy = clf.score(X_test_scaled, y_test)
        
        print(f"  测试集准确率: {accuracy:.4f}")
        print(f"  期望范围: 45% - 55% (越接近50%越好)")
        
        if 0.45 <= accuracy <= 0.55:
            print(f"  [通过] 判别器无法有效区分真假数据")
            pass_test = True
        else:
            print(f"  [警告] 判别器能区分真假数据，生成质量可能有问题")
            pass_test = False
        
        return {
            'accuracy': accuracy,
            'pass_test': pass_test,
            'interpretation': '准确率越接近50%越好，表示生成数据难以被区分'
        }
    
    def validate_level3_downstream(self) -> Dict[str, Any]:
        """
        Level 3: 下游等价验证
        在真实和生成数据上训练同一预测模型，比较表现
        """
        print(f"\n{'='*60}")
        print(f"[Level 3] 下游等价验证")
        print(f"{'='*60}")
        
        results = {}
        
        # 使用收益率特征作为预测目标
        # 任务: 用前59天预测第60天的收益率
        feature_idx = 0  # 对数收益率
        
        # 准备数据
        real_sequences = self.real_data
        gen_sequences = self.generated_data
        
        # 创建训练数据 (X: 前59天, y: 第60天)
        def prepare_xy(sequences):
            X = sequences[:, :-1, :]  # (n, 59, 3)
            y = sequences[:, -1, feature_idx]  # (n,)
            return X.reshape(len(X), -1), y
        
        X_real, y_real = prepare_xy(real_sequences)
        X_gen, y_gen = prepare_xy(gen_sequences)
        
        # 划分训练/测试
        split_idx = int(0.7 * len(X_real))
        
        X_real_train, X_real_test = X_real[:split_idx], X_real[split_idx:]
        y_real_train, y_real_test = y_real[:split_idx], y_real[split_idx:]
        
        X_gen_train, X_gen_test = X_gen[:split_idx], X_gen[split_idx:]
        y_gen_train, y_gen_test = y_gen[:split_idx], y_gen[split_idx:]
        
        # 训练预测模型
        from sklearn.linear_model import Ridge
        from sklearn.metrics import mean_squared_error, r2_score
        
        # 在真实数据上训练
        model_real = Ridge(alpha=1.0)
        model_real.fit(X_real_train, y_real_train)
        pred_real_test = model_real.predict(X_real_test)
        mse_real = mean_squared_error(y_real_test, pred_real_test)
        r2_real = r2_score(y_real_test, pred_real_test)
        
        # 在生成数据上训练
        model_gen = Ridge(alpha=1.0)
        model_gen.fit(X_gen_train, y_gen_train)
        pred_gen_test = model_gen.predict(X_gen_test)
        mse_gen = mean_squared_error(y_gen_test, pred_gen_test)
        r2_gen = r2_score(y_gen_test, pred_gen_test)
        
        # 比较
        mse_diff = abs(mse_real - mse_gen)
        r2_diff = abs(r2_real - r2_gen)
        
        print(f"  在真实数据上训练:")
        print(f"    MSE: {mse_real:.6f}, R²: {r2_real:.4f}")
        print(f"  在生成数据上训练:")
        print(f"    MSE: {mse_gen:.6f}, R²: {r2_gen:.4f}")
        print(f"  差异: MSE差={mse_diff:.6f}, R²差={r2_diff:.4f}")
        
        # 评估等价性
        if mse_diff < 0.1 and r2_diff < 0.2:
            print(f"  [通过] 下游表现相近，生成数据可用于替代真实数据")
            pass_test = True
        else:
            print(f"  [警告] 下游表现差异较大")
            pass_test = False
        
        results = {
            'real_mse': mse_real,
            'gen_mse': mse_gen,
            'real_r2': r2_real,
            'gen_r2': r2_gen,
            'mse_diff': mse_diff,
            'r2_diff': r2_diff,
            'pass_test': pass_test
        }
        
        return results
    
    @staticmethod
    def _skewness(x: np.ndarray) -> float:
        """计算偏度"""
        mean = np.mean(x)
        std = np.std(x)
        if std < 1e-8:
            return 0.0
        return np.mean(((x - mean) / std) ** 3)
    
    @staticmethod
    def _kurtosis(x: np.ndarray) -> float:
        """计算峰度"""
        mean = np.mean(x)
        std = np.std(x)
        if std < 1e-8:
            return 0.0
        return np.mean(((x - mean) / std) ** 4) - 3
    
    @staticmethod
    def _acf(x: np.ndarray, lags: int = 20) -> np.ndarray:
        """计算自相关函数"""
        x = x - np.mean(x)
        var = np.var(x)
        if var < 1e-8:
            return np.zeros(lags)
        
        acf = np.array([np.correlate(x[:len(x)-lag], x[lag:])[0] / (len(x) - lag) / var 
                       for lag in range(1, lags + 1)])
        return acf


# ============================================================================
# 命令行接口
# ============================================================================

def train_mode(args):
    """训练模式"""
    print(f"\n{'#'*60}")
    print(f"# TimeGAN 训练模式")
    print(f"{'#'*60}")
    
    # 配置
    config = Config(
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        latent_dim=args.latent_dim,
        seq_len=args.seq_len,
        batch_size=args.batch_size,
        phase_a_epochs=args.phase_a_epochs,
        phase_b_epochs=args.phase_b_epochs,
        phase_c_epochs=args.phase_c_epochs,
        data_dir=args.data_dir,
        device=args.device
    )
    
    print(f"[配置]")
    for key, value in asdict(config).items():
        print(f"  {key}: {value}")
    
    # 加载数据
    data_dir = Path(args.data_dir) if args.data_dir else DATA_DIR
    print(f"\n[数据加载] 从 {data_dir}")
    data = load_and_preprocess(data_dir, seq_len=config.seq_len)
    
    # 训练
    trainer = TimeGANTrainer(config)
    trainer.train(data, resume=args.resume)
    
    print(f"\n[完成] 训练完成！")


def generate_mode(args):
    """生成模式"""
    print(f"\n{'#'*60}")
    print(f"# TimeGAN 生成模式")
    print(f"{'#'*60}")
    
    # 加载模型
    config = Config(seq_len=args.seq_len, device=args.device)
    model = TimeGAN(config)
    
    model_path = Path(args.model_path) if args.model_path else MODEL_DIR / "timegan_best.pt"
    
    if model_path.exists():
        model.load_state_dict(torch.load(model_path, map_location=args.device))
        print(f"[加载] 模型已从 {model_path} 加载")
    else:
        print(f"[警告] 模型文件不存在，使用随机初始化")
    
    model = model.to(args.device)
    
    # 生成数据
    generator = SyntheticDataGenerator(model, args.device)
    generated_data = generator.generate(args.num_stocks)
    
    print(f"[生成] 已生成 {args.num_stocks} 个序列")
    print(f"[形状] {generated_data.shape}")
    
    # 保存
    output_path = Path(args.output_dir) / "generated_data.npy"
    generator.save(generated_data, output_path)
    
    print(f"\n[完成] 生成完成！")


def validate_mode(args):
    """验证模式"""
    print(f"\n{'#'*60}")
    print(f"# TimeGAN 验证模式")
    print(f"{'#'*60}")
    
    # 加载数据
    output_dir = Path(args.data_dir) if args.data_dir else OUTPUT_DIR
    
    real_path = output_dir / "training_data.npy"
    gen_path = output_dir / "generated_data.npy"
    
    if not gen_path.exists():
        print(f"[错误] 生成数据不存在: {gen_path}")
        print(f"[提示] 请先运行生成模式")
        return
    
    # 加载
    real_data = np.load(real_path) if real_path.exists() else None
    gen_data = np.load(gen_path)
    
    if real_data is None:
        print(f"[警告] 真实训练数据不存在，使用模拟数据")
        real_data = generate_synthetic_data(n_samples=len(gen_data), seq_len=gen_data.shape[1])
    
    print(f"[数据] 真实数据: {real_data.shape}, 生成数据: {gen_data.shape}")
    
    # 验证
    validator = TimeGANValidator(real_data, gen_data)
    
    results = {}
    
    if args.level >= 1:
        results['level1'] = validator.validate_level1_statistics()
    
    if args.level >= 2:
        results['level2'] = validator.validate_level2_discriminator()
    
    if args.level >= 3:
        results['level3'] = validator.validate_level3_downstream()
    
    # 保存结果
    result_path = output_dir / "validation_results.json"
    with open(result_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"\n[保存] 验证结果已保存到 {result_path}")
    print(f"\n[完成] 验证完成！")


def status_mode(args):
    """状态模式"""
    print(f"\n{'#'*60}")
    print(f"# TimeGAN 状态")
    print(f"{'#'*60}")
    
    # 检查模型文件
    model_dir = Path(args.model_dir) if args.model_dir else MODEL_DIR
    output_dir = Path(args.output_dir) if args.output_dir else OUTPUT_DIR
    
    print(f"\n[模型目录] {model_dir}")
    print(f"[输出目录] {output_dir}")
    
    # 列出文件
    print(f"\n[模型文件]")
    for f in sorted(model_dir.glob("*.pt")):
        size_mb = f.stat().st_size / 1024 / 1024
        print(f"  {f.name} ({size_mb:.2f} MB)")
    
    if not any(model_dir.glob("*.pt")):
        print(f"  (无)")
    
    print(f"\n[输出文件]")
    for f in sorted(output_dir.glob("*.npy")):
        try:
            data = np.load(f)
            print(f"  {f.name} - 形状: {data.shape}")
        except:
            print(f"  {f.name}")
    
    if not any(output_dir.glob("*.npy")):
        print(f"  (无)")
    
    # 训练日志
    log_path = output_dir / "training_log.json"
    if log_path.exists():
        print(f"\n[训练日志]")
        with open(log_path, 'r', encoding='utf-8') as f:
            log = json.load(f)
        print(f"  时间: {log.get('timestamp', 'N/A')}")
        print(f"  Phase A 最终损失: {log.get('final_phase_a_loss', 'N/A')}")
        print(f"  Phase B 最终损失: {log.get('final_phase_b_loss', 'N/A')}")
        print(f"  Phase C G损失: {log.get('final_phase_c_g_loss', 'N/A')}")
        print(f"  Phase C D损失: {log.get('final_phase_c_d_loss', 'N/A')}")
    else:
        print(f"\n[训练日志] (无)")


def main():
    """主入口"""
    parser = argparse.ArgumentParser(
        description="A股市场数据生成器 - TimeGAN架构",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  # 训练生成器
  python market_generator.py --mode train --data-dir "stockdata" --epochs 200
  
  # 生成虚拟数据
  python market_generator.py --mode generate --num-stocks 100 --seq-len 60
  
  # 验证生成质量 (Level 1-3)
  python market_generator.py --mode validate --level 3
  
  # 查看状态
  python market_generator.py --mode status
        """
    )
    
    # 模式选择
    parser.add_argument('--mode', type=str, default='status',
                       choices=['train', 'generate', 'validate', 'status'],
                       help='运行模式')
    
    # 通用参数
    parser.add_argument('--data-dir', type=str, default=None,
                       help='数据目录路径')
    parser.add_argument('--model-dir', type=str, default=None,
                       help='模型保存目录')
    parser.add_argument('--output-dir', type=str, default=None,
                       help='输出目录')
    parser.add_argument('--device', type=str, default='cpu',
                       help='设备 (cpu/cuda)')
    
    # 训练参数
    parser.add_argument('--hidden-dim', type=int, default=24,
                       help='隐藏层维度')
    parser.add_argument('--num-layers', type=int, default=2,
                       help='GRU层数')
    parser.add_argument('--latent-dim', type=int, default=24,
                       help='潜在空间维度')
    parser.add_argument('--seq-len', type=int, default=60,
                       help='序列长度')
    parser.add_argument('--batch-size', type=int, default=64,
                       help='批大小')
    parser.add_argument('--phase-a-epochs', type=int, default=100,
                       help='Phase A训练轮数')
    parser.add_argument('--phase-b-epochs', type=int, default=100,
                       help='Phase B训练轮数')
    parser.add_argument('--phase-c-epochs', type=int, default=200,
                       help='Phase C训练轮数')
    parser.add_argument('--epochs', type=int, default=None,
                       help='所有阶段的总轮数 (会覆盖上述参数)')
    parser.add_argument('--resume', action='store_true',
                       help='从检查点恢复训练')
    
    # 生成参数
    parser.add_argument('--num-stocks', type=int, default=100,
                       help='生成股票数量')
    parser.add_argument('--model-path', type=str, default=None,
                       help='模型路径')
    
    # 验证参数
    parser.add_argument('--level', type=int, default=1, choices=[1, 2, 3],
                       help='验证级别 (1=统计, 2=判别器, 3=下游)')
    
    args = parser.parse_args()
    
    # 处理epochs参数
    if args.epochs is not None:
        args.phase_a_epochs = args.epochs
        args.phase_b_epochs = args.epochs
        args.phase_c_epochs = args.epochs
    
    # 执行对应模式
    if args.mode == 'train':
        train_mode(args)
    elif args.mode == 'generate':
        generate_mode(args)
    elif args.mode == 'validate':
        validate_mode(args)
    elif args.mode == 'status':
        status_mode(args)


if __name__ == '__main__':
    main()

"""
market_generator.py - C-TimeGAN 条件化市场数据生成器
====================================================
Phase 2: 生成器对抗

架构: 三层条件化生成 (宏观指数 → 板块指数 → 个股)
组件: Embedder + Recovery + Supervisor + Generator + Discriminator
训练: Phase A(自编码) → Phase B(有监督) → Phase C(对抗)

关键修复(v2.0):
  - 梯度链断裂: 拆为 forward_discriminator(detach) + forward_generator(不detach)
  - Supervisor优化器: Phase C补全 opt_supervisor.zero_grad()+step()
  - 训练数据量: --max-stocks 参数控制，默认0=全部

用法:
  python market_generator.py --mode train [--max-stocks 200] [--phase-a-epochs 100] ...
  python market_generator.py --mode validate [--level 3] [--model-path ...]
  python market_generator.py --mode generate --num-samples 1000
"""

import os, sys, argparse, json, time, warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, str(Path(__file__).parent))
from stock_config import (
    DATA_DIR, ADV_MODEL_DIR, ADV_DATA_DIR, RESULTS_DIR,
    GeneratorConfig, setup_logger, normalize_ohlcv, pct_change,
)
from stock_data_manager import DataAdapter

warnings.filterwarnings('ignore')
logger = setup_logger("Generator")

# ============================================================
# 网络组件
# ============================================================
class GRUCell(nn.Module):
    """带LayerNorm的GRU单元"""
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.gru = nn.GRUCell(input_dim, hidden_dim)
        self.ln = nn.LayerNorm(hidden_dim)
    
    def forward(self, x, h):
        h = self.gru(x, h)
        h = self.ln(h)
        return h

class Embedder(nn.Module):
    """时序编码器: (B, T, F) → (B, T, H)"""
    def __init__(self, cfg: GeneratorConfig):
        super().__init__()
        self.cfg = cfg
        self.rnn = nn.ModuleList()
        inp = cfg.feature_dim
        for _ in range(cfg.num_layers):
            self.rnn.append(GRUCell(inp, cfg.hidden_dim))
            inp = cfg.hidden_dim
    
    def forward(self, x):
        B, T, _ = x.shape
        h = x.new_zeros(B, self.cfg.hidden_dim)
        outputs = []
        for t in range(T):
            for layer in self.rnn:
                h = layer(x[:, t, :] if layer == self.rnn[0] else h, h)
            outputs.append(h.unsqueeze(1))
        return torch.cat(outputs, dim=1)  # (B, T, H)

class Recovery(nn.Module):
    """解码器: (B, T, H) → (B, T, F)"""
    def __init__(self, cfg: GeneratorConfig):
        super().__init__()
        self.cfg = cfg
        self.fc = nn.Linear(cfg.hidden_dim, cfg.feature_dim)
    
    def forward(self, h):
        return self.fc(h)  # (B, T, F)

class Supervisor(nn.Module):
    """有监督预测: h_t → h_{t+1}"""
    def __init__(self, cfg: GeneratorConfig):
        super().__init__()
        self.cfg = cfg
        self.rnn = nn.ModuleList()
        inp = cfg.hidden_dim
        for _ in range(cfg.num_layers):
            self.rnn.append(GRUCell(inp, cfg.hidden_dim))
            inp = cfg.hidden_dim
    
    def forward(self, h):
        B, T, _ = h.shape
        s = h.new_zeros(B, self.cfg.hidden_dim)
        outputs = []
        for t in range(T - 1):
            for layer in self.rnn:
                s = layer(h[:, t, :] if layer == self.rnn[0] else s, s)
            outputs.append(s.unsqueeze(1))
        # 最后一帧复制
        if len(outputs) < T - 1:
            outputs.append(s.unsqueeze(1))
        return torch.cat(outputs, dim=1)  # (B, T-1, H)

class Generator(nn.Module):
    """生成器: 噪声+条件 → 隐空间"""
    def __init__(self, cfg: GeneratorConfig, condition_dim: int = 0):
        super().__init__()
        self.cfg = cfg
        self.condition_dim = condition_dim
        inp = cfg.hidden_dim + condition_dim  # 噪声+条件
        self.rnn = nn.ModuleList()
        for _ in range(cfg.num_layers):
            self.rnn.append(GRUCell(inp, cfg.hidden_dim))
            inp = cfg.hidden_dim
        self.fc_z = nn.Linear(cfg.hidden_dim, cfg.hidden_dim)
    
    def forward(self, z, condition=None):
        """
        z: (B, T, H) 噪声
        condition: (B, condition_dim) 或 None
        """
        B, T, _ = z.shape
        h = z.new_zeros(B, self.cfg.hidden_dim)
        outputs = []
        for t in range(T):
            inp = z[:, t, :]
            # ★ 修复: condition=None时补零让维度匹配(Generator第一层input_size含condition_dim)
            if condition is not None:
                inp = torch.cat([inp, condition], dim=-1)
            elif self.condition_dim > 0:
                zero_cond = z.new_zeros(B, self.condition_dim)
                inp = torch.cat([inp, zero_cond], dim=-1)
            for layer in self.rnn:
                h = layer(inp if layer == self.rnn[0] else h, h)
            outputs.append(h.unsqueeze(1))
        return torch.cat(outputs, dim=1)  # (B, T, H)

class Discriminator(nn.Module):
    """判别器: (B, T, H) → (B, 1)"""
    def __init__(self, cfg: GeneratorConfig):
        super().__init__()
        self.cfg = cfg
        self.rnn = nn.ModuleList()
        inp = cfg.hidden_dim
        for _ in range(cfg.num_layers):
            self.rnn.append(GRUCell(inp, cfg.hidden_dim))
            inp = cfg.hidden_dim
        self.fc = nn.Sequential(
            nn.Linear(cfg.hidden_dim, cfg.hidden_dim // 2),
            nn.LeakyReLU(0.2),
            nn.Linear(cfg.hidden_dim // 2, 1),
        )
    
    def forward(self, h):
        B, T, _ = h.shape
        d = h.new_zeros(B, self.cfg.hidden_dim)
        for t in range(T):
            for layer in self.rnn:
                d = layer(h[:, t, :] if layer == self.rnn[0] else d, d)
        return self.fc(d)  # (B, 1)

# ============================================================
# 条件编码器 (三层条件化)
# ============================================================
class ConditionEncoder(nn.Module):
    """
    三层条件化: 宏观指数 → 板块指数 → 个股
    将三层信息编码为单一条件向量
    """
    def __init__(self, macro_dim=10, sector_dim=20, hidden=32):
        super().__init__()
        self.macro_enc = nn.Sequential(
            nn.Linear(macro_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
        )
        self.sector_enc = nn.Sequential(
            nn.Linear(sector_dim + hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
        )
        self.stock_enc = nn.Sequential(
            nn.Linear(hidden * 2, hidden),
            nn.ReLU(),
        )
        self.output_dim = hidden
    
    def forward(self, macro_feat=None, sector_feat=None):
        """
        macro_feat: (B, macro_dim) 或 None
        sector_feat: (B, sector_dim) 或 None
        """
        # 确定batch size
        if macro_feat is not None:
            B = macro_feat.shape[0]
            device = macro_feat.device
        elif sector_feat is not None:
            B = sector_feat.shape[0]
            device = sector_feat.device
        else:
            B = 1
            device = 'cpu'
        
        if macro_feat is not None:
            macro_h = self.macro_enc(macro_feat)
        else:
            macro_h = torch.zeros(B, self.macro_enc[-1].out_features, device=device)
        
        if sector_feat is not None:
            sector_inp = torch.cat([sector_feat, macro_h], dim=-1)
            sector_h = self.sector_enc(sector_inp)
        else:
            sector_h = torch.zeros(B, self.sector_enc[-1].out_features, device=device)
        
        # 最终条件 = macro + sector
        condition = self.stock_enc(torch.cat([macro_h, sector_h], dim=-1))
        return condition  # (B, hidden)

# ============================================================
# C-TimeGAN 主模型
# ============================================================
class CTimeGAN:
    """
    条件化TimeGAN: 三层条件化市场数据生成
    
    训练三阶段:
      Phase A: 自编码器 (E+R)
      Phase B: 有监督 (E+S)
      Phase C: 对抗 (E+R+S+G+D)
    """
    
    def __init__(self, cfg: GeneratorConfig = None, device: str = "auto"):
        self.cfg = cfg or GeneratorConfig()
        
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        
        logger.info(f"设备: {self.device}")
        
        # 构建网络
        H = self.cfg.hidden_dim
        F = self.cfg.feature_dim
        
        self.embedder = Embedder(self.cfg).to(self.device)
        self.recovery = Recovery(self.cfg).to(self.device)
        self.supervisor = Supervisor(self.cfg).to(self.device)
        
        # 条件编码器
        self.cond_encoder = ConditionEncoder(macro_dim=10, sector_dim=20, hidden=32).to(self.device)
        cond_dim = self.cond_encoder.output_dim
        
        self.generator = Generator(self.cfg, condition_dim=cond_dim).to(self.device)
        self.discriminator = Discriminator(self.cfg).to(self.device)
        
        # 参数量统计
        total_params = sum(p.numel() for m in [self.embedder, self.recovery, 
                         self.supervisor, self.generator, self.discriminator,
                         self.cond_encoder] for p in m.parameters())
        logger.info(f"模型参数总量: {total_params:,}")
    
    # ── 损失函数 ──
    def _mse(self, pred, target):
        return nn.MSELoss()(pred, target)
    
    def _bce(self, pred, target):
        return nn.BCEWithLogitsLoss()(pred, target)
    
    # ── 前向传播（分离版，防止梯度链断裂）──
    def forward_discriminator(self, x_real, z, condition=None):
        """
        判别器前向: detach生成器输出，防止梯度回流到G
        【关键修复】v1.x中用 h.detach() 导致Phase C梯度链断裂
        """
        h_real = self.embedder(x_real)
        h_fake = self.generator(z, condition)
        
        # 判别器只看detach后的隐表示
        d_real = self.discriminator(h_real.detach())
        d_fake = self.discriminator(h_fake.detach())
        
        return d_real, d_fake, h_real, h_fake
    
    def forward_generator(self, x_real, z, condition=None):
        """
        生成器前向: 不detach，梯度从D→S→G完整回传
        【关键修复】v1.x只有一个forward，Phase C时梯度链断裂
        """
        h_real = self.embedder(x_real)
        h_fake = self.generator(z, condition)        # 不detach
        h_sup = self.supervisor(h_fake)              # 不detach
        x_hat = self.recovery(h_fake)                # 不detach
        
        d_fake = self.discriminator(h_fake)          # 不detach
        d_real = self.discriminator(h_real.detach())  # 真实样本detach
        
        return d_real, d_fake, h_real, h_fake, h_sup, x_hat
    
    # ── 训练 ──
    def train(self, data_loader, val_data=None,
              save_dir: Path = ADV_MODEL_DIR):
        """完整三阶段训练"""
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info("=" * 60)
        logger.info("C-TimeGAN 训练开始")
        logger.info(f"  Phase A: {self.cfg.phase_a_epochs} epochs")
        logger.info(f"  Phase B: {self.cfg.phase_b_epochs} epochs")
        logger.info(f"  Phase C: {self.cfg.phase_c_epochs} epochs")
        logger.info("=" * 60)
        
        history = {'phase_a': [], 'phase_b': [], 'phase_c': []}
        
        # Phase A: 自编码器
        self._train_phase_a(data_loader, history)
        self._save(save_dir / "phase_a.pt")
        
        # Phase B: 有监督
        self._train_phase_b(data_loader, history)
        self._save(save_dir / "phase_b.pt")
        
        # Phase C: 对抗
        self._train_phase_c(data_loader, history)
        self._save(save_dir / "phase_c.pt")
        
        # 保存训练历史
        with open(save_dir / "train_history.json", 'w') as f:
            json.dump(history, f, indent=2)
        
        logger.info("训练完成！模型已保存")
        return history
    
    def _train_phase_a(self, data_loader, history):
        """Phase A: 自编码器 E + R"""
        logger.info("\n" + "=" * 40)
        logger.info("Phase A: 自编码器训练")
        logger.info("=" * 40)
        
        opt_er = optim.Adam(
            list(self.embedder.parameters()) + list(self.recovery.parameters()),
            lr=self.cfg.learning_rate
        )
        
        for epoch in range(self.cfg.phase_a_epochs):
            epoch_loss = 0.0
            n_batches = 0
            
            for batch in data_loader:
                x = batch[0].to(self.device).float()
                
                h = self.embedder(x)
                x_hat = self.recovery(h)
                
                loss = self._mse(x_hat, x)
                
                opt_er.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    list(self.embedder.parameters()) + list(self.recovery.parameters()),
                    max_norm=5.0
                )
                opt_er.step()
                
                epoch_loss += loss.item()
                n_batches += 1
            
            avg_loss = epoch_loss / max(n_batches, 1)
            history['phase_a'].append(avg_loss)
            
            log_every = max(1, self.cfg.phase_a_epochs // 10)
            if (epoch + 1) % log_every == 0 or epoch == 0:
                logger.info(f"  [A] Epoch {epoch+1}/{self.cfg.phase_a_epochs} | "
                           f"Loss: {avg_loss:.6f}")
    
    def _train_phase_b(self, data_loader, history):
        """Phase B: 有监督 E + S"""
        logger.info("\n" + "=" * 40)
        logger.info("Phase B: 有监督训练")
        logger.info("=" * 40)
        
        opt_s = optim.Adam(
            list(self.supervisor.parameters()),
            lr=self.cfg.learning_rate
        )
        
        for epoch in range(self.cfg.phase_b_epochs):
            epoch_loss = 0.0
            n_batches = 0
            
            for batch in data_loader:
                x = batch[0].to(self.device).float()
                
                with torch.no_grad():
                    h = self.embedder(x)
                
                h_sup = self.supervisor(h)
                # 目标: h_sup[:, t] ≈ h[:, t+1]
                loss = self._mse(h_sup, h[:, 1:, :])
                
                opt_s.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.supervisor.parameters(), max_norm=5.0)
                opt_s.step()
                
                epoch_loss += loss.item()
                n_batches += 1
            
            avg_loss = epoch_loss / max(n_batches, 1)
            history['phase_b'].append(avg_loss)
            
            log_every = max(1, self.cfg.phase_b_epochs // 10)
            if (epoch + 1) % log_every == 0 or epoch == 0:
                logger.info(f"  [B] Epoch {epoch+1}/{self.cfg.phase_b_epochs} | "
                           f"Loss: {avg_loss:.6f}")
    
    def _train_phase_c(self, data_loader, history):
        """
        Phase C: 对抗训练 E+R+S+G+D
        【关键修复】:
          1. 判别器步: 使用 forward_discriminator (detach)
          2. 生成器步: 使用 forward_generator (不detach)
          3. Supervisor优化器: 必须参与生成器步 zero_grad()+step()
        """
        logger.info("\n" + "=" * 40)
        logger.info("Phase C: 对抗训练")
        logger.info("=" * 40)
        
        lr = self.cfg.learning_rate
        
        # 五个优化器
        opt_e = optim.Adam(self.embedder.parameters(), lr=lr)
        opt_r = optim.Adam(self.recovery.parameters(), lr=lr)
        opt_s = optim.Adam(self.supervisor.parameters(), lr=lr)  # ← 关键：必须独立
        opt_g = optim.Adam(self.generator.parameters(), lr=lr)
        opt_d = optim.Adam(self.discriminator.parameters(), lr=lr * 0.1)  # ★ Bug#17修复: D学习率大幅降低(0.5→0.1)防碾压
        
        all_params = (list(self.embedder.parameters()) + 
                     list(self.recovery.parameters()) +
                     list(self.supervisor.parameters()) +
                     list(self.generator.parameters()))
        
        # ★ Bug#18修复: D训练频率控制 + 自适应G步数
        # 核心策略: D不是每批都训练，给G更多成长空间
        d_train_freq = 3          # 每隔3个batch才训练1次D (StyleGAN2同款策略)
        g_steps_per_d = 3         # D训练时G走3步; D不训练时G走1步
        d_warmup_epochs = 2       # Phase C前2个epoch不训练D, 让E+R+S+G先热身
        
        # ★ v2.3: 最佳模型保存 + 早停 (防D崩塌后G漂移)
        best_score = float('inf')
        best_state = None
        patience_counter = 0
        patience_limit = 20       # 连续20轮D-G平衡恶化则早停
        min_epochs_before_stop = 10  # 至少跑10轮才允许早停
        
        for epoch in range(self.cfg.phase_c_epochs):
            d_losses, g_losses = [], []
            n_batches = 0
            
            for batch in data_loader:
                x = batch[0].to(self.device).float()
                B = x.shape[0]
                
                # ── 判别器步 (隔d_train_freq批训练1次, 预热期跳过) ──
                train_d = (epoch >= d_warmup_epochs) and (n_batches % d_train_freq == 0)
                
                if train_d:
                    z = torch.randn(B, self.cfg.seq_len, self.cfg.hidden_dim,
                                   device=self.device)
                    d_real, d_fake, _, _ = self.forward_discriminator(x, z)
                    
                    d_loss_real = self._bce(d_real, torch.ones_like(d_real) * 0.9)
                    d_loss_fake = self._bce(d_fake, torch.ones_like(d_fake) * 0.1)
                    d_loss = d_loss_real + d_loss_fake
                    
                    opt_d.zero_grad()
                    d_loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.discriminator.parameters(), max_norm=5.0)
                    opt_d.step()
                    d_losses.append(d_loss.item())
                else:
                    d_losses.append(-1)  # 占位, 表示本批D未训练
                
                # ── 生成器步 ──
                n_g = g_steps_per_d if train_d else 1  # D训练时G多走几步追赶
                
                for _g_step in range(n_g):
                    z2 = torch.randn(B, self.cfg.seq_len, self.cfg.hidden_dim,
                                    device=self.device)
                    
                    d_real2, d_fake2, h_real, h_fake, h_sup, x_hat = \
                        self.forward_generator(x, z2)
                    
                    # G对抗损失
                    g_loss_adv = self._bce(d_fake2, torch.ones_like(d_fake2) * 0.9)
                    # 有监督损失
                    s_loss = self._mse(h_sup, h_real[:, 1:, :])
                    # 重建损失
                    r_loss = self._mse(x_hat, x)
                    # 总生成器损失
                    g_loss = g_loss_adv + s_loss + 0.5 * r_loss
                    
                    opt_e.zero_grad()
                    opt_r.zero_grad()
                    opt_s.zero_grad()
                    opt_g.zero_grad()
                    
                    g_loss.backward()
                    torch.nn.utils.clip_grad_norm_(all_params, max_norm=5.0)
                    
                    opt_e.step()
                    opt_r.step()
                    opt_s.step()
                    opt_g.step()
                
                g_losses.append(g_loss.item())
                n_batches += 1
            
            # ★ Bug#18: 过滤D未训练的占位(-1)
            valid_d = [d for d in d_losses if d >= 0]
            avg_d = np.mean(valid_d) if valid_d else -1
            avg_g = np.mean(g_losses)
            d_train_ratio = len(valid_d) / max(len(d_losses), 1)
            history['phase_c'].append({'d_loss': avg_d, 'g_loss': avg_g})
            
            log_every = max(1, self.cfg.phase_c_epochs // 10)
            if (epoch + 1) % log_every == 0 or epoch == 0:
                d_str = f"{avg_d:.4f}" if avg_d >= 0 else "warmup"
                logger.info(f"  [C] Epoch {epoch+1}/{self.cfg.phase_c_epochs} | "
                           f"D: {d_str} | G: {avg_g:.4f} | D训练比: {d_train_ratio:.0%}")
                
                # 梯度健康检查
                if avg_g > 50:
                    logger.warning(f"  ⚠ G_loss={avg_g:.2f} 异常偏高，可能梯度爆炸")
                if avg_d >= 0 and avg_d < 0.01:
                    logger.warning(f"  ⚠ D_loss={avg_d:.4f} 过低，判别器可能过强")
            
            # ★ v2.3: 最佳模型保存 + 早停
            if avg_d >= 0:
                # D-G平衡度评分: 越小越好
                # 理想状态: D≈1.0~1.5, G≈0.8~1.2, 差距小
                balance = abs(avg_d - avg_g) + abs(avg_d - 1.386)  # 1.386=ln4, 理想D值
                if balance < best_score:
                    best_score = balance
                    best_state = {
                        'embedder': {k: v.clone() for k, v in self.embedder.state_dict().items()},
                        'recovery': {k: v.clone() for k, v in self.recovery.state_dict().items()},
                        'supervisor': {k: v.clone() for k, v in self.supervisor.state_dict().items()},
                        'generator': {k: v.clone() for k, v in self.generator.state_dict().items()},
                        'discriminator': {k: v.clone() for k, v in self.discriminator.state_dict().items()},
                        'cond_encoder': {k: v.clone() for k, v in self.cond_encoder.state_dict().items()},
                    }
                    patience_counter = 0
                    if epoch >= min_epochs_before_stop:
                        logger.info(f"  ★ 最佳模型 @Epoch {epoch+1} | D={avg_d:.4f} G={avg_g:.4f} balance={balance:.4f}")
                else:
                    patience_counter += 1
                
                # 早停判断
                if (patience_counter >= patience_limit and 
                    epoch >= min_epochs_before_stop):
                    logger.info(f"\n  ⚡ 早停触发 @Epoch {epoch+1} | 连续{patience_limit}轮无改善")
                    logger.info(f"  最佳balance={best_score:.4f}, 当前balance={balance:.4f}")
                    break
            else:
                # warmup期间不评分，但保存初始状态
                if best_state is None and epoch == d_warmup_epochs - 1:
                    best_state = {
                        'embedder': {k: v.clone() for k, v in self.embedder.state_dict().items()},
                        'recovery': {k: v.clone() for k, v in self.recovery.state_dict().items()},
                        'supervisor': {k: v.clone() for k, v in self.supervisor.state_dict().items()},
                        'generator': {k: v.clone() for k, v in self.generator.state_dict().items()},
                        'discriminator': {k: v.clone() for k, v in self.discriminator.state_dict().items()},
                        'cond_encoder': {k: v.clone() for k, v in self.cond_encoder.state_dict().items()},
                    }
        
        # ★ v2.3: 恢复最佳模型 (而不是用崩塌后的最后一轮)
        if best_state is not None:
            self.embedder.load_state_dict(best_state['embedder'])
            self.recovery.load_state_dict(best_state['recovery'])
            self.supervisor.load_state_dict(best_state['supervisor'])
            self.generator.load_state_dict(best_state['generator'])
            self.discriminator.load_state_dict(best_state['discriminator'])
            self.cond_encoder.load_state_dict(best_state['cond_encoder'])
            logger.info(f"\n  ★ 已恢复Phase C最佳模型 (balance={best_score:.4f})")
    
    # ── 生成 ──
    @torch.no_grad()
    def generate(self, num_samples: int = 1000,
                  condition=None) -> np.ndarray:
        """
        生成市场数据
        condition: (B, condition_dim) 或 None
        返回: (num_samples, seq_len, feature_dim) numpy
        """
        self.generator.eval()
        self.recovery.eval()
        
        z = torch.randn(num_samples, self.cfg.seq_len, self.cfg.hidden_dim,
                        device=self.device)
        
        if condition is not None:
            if isinstance(condition, np.ndarray):
                condition = torch.FloatTensor(condition).to(self.device)
        
        h_fake = self.generator(z, condition)
        x_fake = self.recovery(h_fake)
        
        return x_fake.cpu().numpy()
    
    # ── 保存/加载 ──
    def _save(self, path):
        torch.save({
            'embedder': self.embedder.state_dict(),
            'recovery': self.recovery.state_dict(),
            'supervisor': self.supervisor.state_dict(),
            'generator': self.generator.state_dict(),
            'discriminator': self.discriminator.state_dict(),
            'cond_encoder': self.cond_encoder.state_dict(),
            'cfg': vars(self.cfg),
        }, path)
        logger.info(f"模型已保存: {path}")
    
    def load(self, path):
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        self.embedder.load_state_dict(ckpt['embedder'])
        self.recovery.load_state_dict(ckpt['recovery'])
        self.supervisor.load_state_dict(ckpt['supervisor'])
        self.generator.load_state_dict(ckpt['generator'])
        self.discriminator.load_state_dict(ckpt['discriminator'])
        self.cond_encoder.load_state_dict(ckpt['cond_encoder'])
        logger.info(f"模型已加载: {path}")

# ============================================================
# 三级验证
# ============================================================
class GeneratorValidator:
    """
    三级验证体系:
      L1: 统计特征对比 (均值/方差/偏度/峰度/自相关)
      L2: 开盘重点验证 (9:30-9:40 权重5x)
      L3: 排列检验 (迁移性门禁)
    """
    
    def __init__(self, cfg: GeneratorConfig = None):
        self.cfg = cfg or GeneratorConfig()
        self.logger = setup_logger("Validator")
    
    def _stats(self, data: np.ndarray) -> Dict:
        """计算统计特征 (N, T, F) → 统计dict"""
        flat = data.reshape(-1, data.shape[-1])
        return {
            'mean': flat.mean(axis=0),
            'std': flat.std(axis=0),
            'skew': self._skew(flat),
            'kurt': self._kurt(flat),
            'acf1': self._acf1(data),
        }
    
    def _skew(self, x):
        from scipy.stats import skew
        return skew(x, axis=0)
    
    def _kurt(self, x):
        from scipy.stats import kurtosis
        return kurtosis(x, axis=0)
    
    def _acf1(self, data):
        """一阶自相关 (对每只样本)"""
        acfs = []
        for i in range(min(len(data), 100)):
            for f in range(data.shape[-1]):
                s = data[i, :, f]
                if s.std() > 0:
                    acfs.append(np.corrcoef(s[:-1], s[1:])[0, 1])
        return np.mean(acfs) if acfs else 0.0
    
    def validate_l1(self, real: np.ndarray, fake: np.ndarray) -> Dict:
        """L1: 统计特征对比"""
        real_s = self._stats(real)
        fake_s = self._stats(fake)
        
        # 各特征相对误差
        errors = {}
        for key in ['mean', 'std']:
            r, f = real_s[key], fake_s[key]
            rel_err = np.abs(r - f) / (np.abs(r) + 1e-8)
            errors[key + '_rel_err'] = rel_err.mean()
        
        # KS检验
        from scipy.stats import ks_2samp
        ks_results = []
        for f in range(real.shape[-1]):
            r_flat = real[:, :, f].flatten()
            f_flat = fake[:, :, f].flatten()
            stat, pval = ks_2samp(r_flat, f_flat)
            ks_results.append({'feature': f, 'statistic': stat, 'p_value': pval})
        
        result = {
            'level': 'L1',
            'passed': all(e < 0.3 for e in errors.values()),
            'errors': errors,
            'ks_tests': ks_results,
            'skew_diff': np.abs(real_s['skew'] - fake_s['skew']).mean(),
            'kurt_diff': np.abs(real_s['kurt'] - fake_s['kurt']).mean(),
            'acf1_diff': np.abs(real_s['acf1'] - fake_s['acf1']),
        }
        
        self.logger.info(f"[L1] {'✓ PASS' if result['passed'] else '✗ FAIL'} | "
                        f"Mean_err={errors['mean_rel_err']:.4f} "
                        f"Std_err={errors['std_rel_err']:.4f}")
        return result
    
    def validate_l2(self, real: np.ndarray, fake: np.ndarray) -> Dict:
        """
        L2: 开盘重点验证
        9:30-9:40 (前2个5min bar) 权重5x
        9:40-9:50 权重2x
        其余 1x
        """
        T = real.shape[1]
        # 模拟开盘时段映射 (T个bar → 开盘权重)
        if T >= 48:  # 日内5分钟线
            weights = np.ones(T)
            weights[:2] = self.cfg.w930_940   # 9:30-9:40
            weights[2:4] = self.cfg.w940_950  # 9:40-9:50
        else:  # 日K线，用前3日模拟开盘效应
            weights = np.ones(T)
            weights[:3] = self.cfg.w930_940
        
        # 加权MSE
        weights_t = torch.FloatTensor(weights).unsqueeze(0).unsqueeze(-1)
        real_t = torch.FloatTensor(real)
        fake_t = torch.FloatTensor(fake)
        
        weighted_mse = ((real_t - fake_t) ** 2 * weights_t).mean().item()
        
        # 开盘时段专项统计
        opening_mse = ((real_t[:, :4] - fake_t[:, :4]) ** 2).mean().item()
        normal_mse = ((real_t[:, 4:] - fake_t[:, 4:]) ** 2).mean().item()
        
        result = {
            'level': 'L2',
            'passed': opening_mse < normal_mse * 2.0,  # 开盘MSE不超过普通2x
            'weighted_mse': weighted_mse,
            'opening_mse': opening_mse,
            'normal_mse': normal_mse,
            'opening_ratio': opening_mse / (normal_mse + 1e-8),
        }
        
        self.logger.info(f"[L2] {'✓ PASS' if result['passed'] else '✗ FAIL'} | "
                        f"开盘MSE={opening_mse:.6f} 普通MSE={normal_mse:.6f} "
                        f"比值={result['opening_ratio']:.2f}")
        return result
    
    def validate_l3(self, real: np.ndarray, fake: np.ndarray,
                     n_perm: int = None) -> Dict:
        """
        L3: 排列检验 (迁移性门禁)
        零假设H0: 生成数据与真实数据同分布
        检验: 如果排列p > 0.05，接受H0(生成数据足够真实)
        """
        from scipy.stats import ks_2samp
        
        n_perm = n_perm or 1000
        close_real = real[:, :, 3].flatten()  # close列
        close_fake = fake[:, :, 3].flatten()
        
        # 观测统计量
        obs_stat, _ = ks_2samp(close_real, close_fake)
        
        # 排列检验
        combined = np.concatenate([close_real[:len(close_fake)], close_fake])
        count = 0
        for _ in range(n_perm):
            np.random.shuffle(combined)
            s, _ = ks_2samp(combined[:len(close_fake)], combined[len(close_fake):])
            if s >= obs_stat:
                count += 1
        
        p_value = count / n_perm
        
        result = {
            'level': 'L3',
            'passed': p_value > 0.05,
            'observed_statistic': obs_stat,
            'p_value': p_value,
            'n_permutations': n_perm,
        }
        
        self.logger.info(f"[L3] {'✓ PASS' if result['passed'] else '✗ FAIL'} | "
                        f"KS={obs_stat:.4f} p={p_value:.4f}")
        return result
    
    def full_validation(self, real: np.ndarray, fake: np.ndarray,
                         level: int = 3) -> Dict:
        """执行完整验证"""
        results = {}
        results['L1'] = self.validate_l1(real, fake)
        
        if level >= 2:
            results['L2'] = self.validate_l2(real, fake)
        
        if level >= 3:
            results['L3'] = self.validate_l3(real, fake)
        
        # 总体判定
        all_passed = all(r['passed'] for r in results.values())
        results['overall'] = {
            'passed': all_passed,
            'levels_tested': level,
        }
        
        self.logger.info(f"\n{'='*40}")
        self.logger.info(f"验证结果: {'✓ 全部通过' if all_passed else '✗ 存在未通过'}")
        self.logger.info(f"{'='*40}")
        
        return results

# ============================================================
# 查重模块
# ============================================================
class Deduplicator:
    """
    生成数据查重: 三重把关
    1. DTW距离 < 阈值
    2. Pearson相关 > 0.85
    3. KS检验 p > 0.05
    全部通过才判定为"过于相似"
    """
    
    def __init__(self, dtw_thresh=2.0, pearson_thresh=0.85, ks_pval=0.05):
        self.dtw_thresh = dtw_thresh
        self.pearson_thresh = pearson_thresh
        self.ks_pval = ks_pval
    
    def check(self, generated: np.ndarray, real_pool: np.ndarray) -> Dict:
        """
        检查生成序列是否与真实序列过于相似
        generated: (N, T, F)
        real_pool: (M, T, F)
        """
        from scipy.stats import ks_2samp
        
        n_similar = 0
        similar_indices = []
        
        for i in range(len(generated)):
            gen_close = generated[i, :, 3]  # close列
            
            for j in range(len(real_pool)):
                real_close = real_pool[j, :, 3]
                
                # Pearson
                corr = np.corrcoef(gen_close, real_close)[0, 1]
                if abs(corr) < self.pearson_thresh:
                    continue
                
                # KS
                _, pval = ks_2samp(gen_close, real_close)
                if pval < self.ks_pval:
                    continue
                
                # DTW (简化版: 欧氏距离近似)
                dtw_dist = np.sqrt(((gen_close - real_close) ** 2).mean())
                if dtw_dist < self.dtw_thresh:
                    n_similar += 1
                    similar_indices.append((i, j, dtw_dist, corr))
                    break  # 一条生成匹配到一条真实即可
        
        return {
            'total_generated': len(generated),
            'similar_count': n_similar,
            'similar_ratio': n_similar / max(len(generated), 1),
            'similar_pairs': similar_indices[:10],  # 最多记录10条
        }

# ============================================================
# 命令行入口
# ============================================================
def main():
    parser = argparse.ArgumentParser(description="C-TimeGAN 市场生成器")
    parser.add_argument("--mode", required=True,
                        choices=["train", "validate", "generate", "retrain-c"])
    parser.add_argument("--data-dir", type=str, default=str(DATA_DIR))
    parser.add_argument("--model-path", type=str, default=str(ADV_MODEL_DIR / "phase_c.pt"))
    parser.add_argument("--pretrained-path", type=str, default="",
                        help="retrain-c模式: 加载Phase A+B的预训练模型路径")
    parser.add_argument("--max-stocks", type=int, default=0)
    parser.add_argument("--phase-a-epochs", type=int, default=100)
    parser.add_argument("--phase-b-epochs", type=int, default=100)
    parser.add_argument("--phase-c-epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--level", type=int, default=3,
                        help="验证级别 1/2/3")
    parser.add_argument("--num-samples", type=int, default=1000)
    parser.add_argument("--device", type=str, default="auto")
    
    args = parser.parse_args()
    
    cfg = GeneratorConfig(
        phase_a_epochs=args.phase_a_epochs,
        phase_b_epochs=args.phase_b_epochs,
        phase_c_epochs=args.phase_c_epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
    )
    
    if args.mode == "train":
        # 加载数据
        adapter = DataAdapter(Path(args.data_dir))
        stocks = adapter.list_stocks(min_length=50)
        if args.max_stocks > 0:
            stocks = stocks[:args.max_stocks]
        
        logger.info(f"加载 {len(stocks)} 只股票...")
        all_data, _ = adapter.load_batch(stocks)
        
        if len(all_data) == 0:
            logger.error("无可用数据！")
            return
        
        logger.info(f"数据形状: {all_data.shape}")
        
        # DataLoader
        dataset = TensorDataset(torch.FloatTensor(all_data))
        loader = DataLoader(dataset, batch_size=cfg.batch_size, shuffle=True, drop_last=True)
        
        # 训练
        model = CTimeGAN(cfg, device=args.device)
        model.train(loader, save_dir=ADV_MODEL_DIR)
    
    elif args.mode == "validate":
        adapter = DataAdapter(Path(args.data_dir))
        stocks = adapter.list_stocks(min_length=50)
        if args.max_stocks > 0:
            stocks = stocks[:args.max_stocks]
        
        real_data, _ = adapter.load_batch(stocks, max_seqs=1000)
        
        model = CTimeGAN(cfg, device=args.device)
        model.load(args.model_path)
        
        fake_data = model.generate(num_samples=len(real_data))
        
        validator = GeneratorValidator(cfg)
        results = validator.full_validation(real_data, fake_data, level=args.level)
        
        out = RESULTS_DIR / "validation_result.json"
        with open(out, 'w') as f:
            json.dump(results, f, indent=2, default=str)
    
    elif args.mode == "generate":
        model = CTimeGAN(cfg, device=args.device)
        model.load(args.model_path)
        
        fake = model.generate(num_samples=args.num_samples)
        
        out = ADV_DATA_DIR / f"generated_{args.num_samples}.npy"
        np.save(out, fake)
        logger.info(f"生成数据已保存: {out} | 形状{fake.shape}")
    
    elif args.mode == "retrain-c":
        # ★ v2.3: 只重训Phase C (复用已训好的Phase A+B)
        # 加载数据
        adapter = DataAdapter(Path(args.data_dir))
        stocks = adapter.list_stocks(min_length=50)
        if args.max_stocks > 0:
            stocks = stocks[:args.max_stocks]
        
        logger.info(f"加载 {len(stocks)} 只股票...")
        all_data, _ = adapter.load_batch(stocks)
        if len(all_data) == 0:
            logger.error("无可用数据！")
            return
        logger.info(f"数据形状: {all_data.shape}")
        
        dataset = TensorDataset(torch.FloatTensor(all_data))
        loader = DataLoader(dataset, batch_size=cfg.batch_size, shuffle=True, drop_last=True)
        
        # 加载预训练模型 (Phase A+B的权重)
        model = CTimeGAN(cfg, device=args.device)
        pretrained = args.pretrained_path or str(ADV_MODEL_DIR / "phase_c.pt")
        if Path(pretrained).exists():
            model.load(pretrained)
            logger.info(f"已加载预训练权重: {pretrained}")
        else:
            logger.error(f"预训练模型不存在: {pretrained}")
            return
        
        # 只跑Phase C (带v2.3最佳模型保存+早停)
        history = {'phase_a': [], 'phase_b': [], 'phase_c': []}
        model._train_phase_c(loader, history)
        model._save(ADV_MODEL_DIR / "phase_c.pt")
        logger.info(f"★ Phase C重训完成！模型已保存 (含最佳模型恢复+早停)")

if __name__ == "__main__":
    main()

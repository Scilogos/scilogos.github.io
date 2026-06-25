# 对抗学习量化系统 - 云端执行方案

## 📁 文件清单

| 文件 | 说明 |
|------|------|
| `stock_config_cloud.py` | 跨平台配置文件（Colab/Windows通用） |
| `arena_cloud.ipynb` | Google Colab笔记本（直接上传运行） |
| `one_click_arena.py` | Windows本地一键启动脚本 |
| `README.md` | 本文档 |

---

## 🚀 快速开始

### 方案一：Google Colab（推荐，免费GPU）

1. 打开 https://colab.research.google.com
2. 点击 **File → Upload notebook**
3. 上传 `arena_cloud.ipynb`
4. 点击 **Runtime → Run all** (或 Ctrl+F9)
5. 等待训练完成，下载结果

**Colab设置建议：**
- Runtime → Change runtime type → **Python3 + GPU** (免费T4)
- 训练约需 30分钟~2小时（取决于GPU）

---

### 方案二：Windows本地一键运行

1. 确保安装了Python 3.8+
2. 双击运行 `one_click_arena.py`
3. 脚本会自动：
   - 检测并安装依赖
   - 生成模拟数据
   - 从检查点恢复（如有）
   - 运行Arena训练

```bash
# 或者命令行运行
cd "C:\Users\HUAWEI\Desktop\Adversarial Learning"
python one_click_arena.py
```

---

## 📊 训练配置

| 参数 | 值 |
|------|------|
| 庄家策略 | 8种（拉高出货/洗盘/做空等） |
| 散户策略 | 8种（趋势/均值回归/突破等） |
| 对战组合 | 8×8 = 64组 |
| 每组轮数 | 500回合 |
| 数据量 | 2000条模拟股价序列 |
| 序列长度 | 30个交易日 |
| 特征 | OHLCV + 涨跌幅 (6维) |

---

## 🔧 核心功能

### 跨平台路径自动检测

```
if os.path.exists("/content"):      # Colab
    BASE_DIR = "/content/AdversarialLearning"
elif os.name == "nt":               # Windows
    BASE_DIR = "C:\Users\HUAWEI\Desktop\Adversarial Learning"
else:                              # Linux
    BASE_DIR = "~/AdversarialLearning"
```

### 断点续传

- 自动保存检查点：`checkpoint.json`
- 训练中断后再次运行，自动从断点恢复
- 无需手动指定恢复点

### 模拟数据生成（GBM模型）

```
使用几何布朗运动(Geometric Brownian Motion)生成逼真股价：
- 价格范围: 5~50元
- 日波动率: 1%~4%
- 涨跌停限制: ±10%
- 成交量: 对数正态分布
```

---

## 📂 输出文件

训练完成后，结果保存在 `stockresults/` 目录：

```
AdversarialLearning/
├── stockresults/
│   ├── arena_results.json      # 对抗结果汇总
│   ├── logs/                    # 训练日志
│   │   └── arena_YYYYMMDD_HHMMSS.log
│   └── ...
├── adversarial_data/
│   └── generated_2000.npy       # 训练数据
├── adversarial_model/          # 模型权重（可选）
└── checkpoint.json             # 检查点
```

---

## ⚠️ 注意事项

1. **GitHub访问问题**：如果GitHub raw文件无法下载，脚本会提示手动上传
2. **数据文件**：首次运行会生成2000条模拟数据（约2MB）
3. **训练时间**：完整训练可能需要数小时，建议使用Colab GPU加速
4. **Colab断连**：免费版Colab最长12小时，建议晚间运行

---

## 🆘 常见问题

**Q: 训练中途断开怎么办？**
A: 重新运行脚本，会自动从检查点恢复。

**Q: Colab显示内存不足？**
A: Runtime → Restart runtime，然后重新运行。

**Q: 如何查看训练进度？**
A: 检查 `stockresults/logs/` 下的日志文件。

**Q: 可以使用真实股票数据吗？**
A: 可以，将数据转为(N,30,6)格式的.npy文件，替换generated_2000.npy。

---

## 📝 手动操作步骤（可选）

如果需要手动配置环境：

```bash
# 1. 安装依赖
pip install torch numpy scipy pandas

# 2. 下载脚本
git clone <your-repo>
cd Adversarial\ Learning/dotpy

# 3. 生成数据
python -c "
import numpy as np
data = np.random.randn(2000, 30, 6)
np.save('../adversarial_data/generated_2000.npy', data)
"

# 4. 运行训练
python adversarial_env.py --mode arena --arena-episodes 500
```

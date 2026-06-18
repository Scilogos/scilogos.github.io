# 量化回测系统 v1.6 → v2.0 大修方案

## 一、目标
对 stocks2.py (6401行) 进行彻底大修，修复4个P0致命Bug + 5个P1准确性问题 + 引入元叙事模块，产出 stocks3.py。

## 二、原文件位置
- 源文件：stocks2_1781599059155_0_ahvl.py (6401行)
- 产出：stocks3.py（大修后的新版本）
- 输出目录：/app/data/所有对话/主对话/

## 三、P0致命Bug修复（必改，不改系统必崩）

### P0-1: BB_LGBMPredictor.train_and_predict 返回值长度不一致
- **位置**：约第1754行 `train_and_predict` 方法
- **问题**：正常返回5元素元组 `(prediction_1d, prediction_2d, prediction_3d, prediction_subtype, importance)`，异常返回3元素 `(np.array([0.33,0.34,0.33]), None, DataFrame())`
- **修复**：异常时也返回5元素，补齐 `prediction_3d=None` 和 `prediction_subtype=None`
- **代码**：`return np.array([0.33, 0.34, 0.33]), None, None, None, pd.DataFrame({'feature': [], 'importance': []})`

### P0-2: baostock连接生命周期完全失控
- **位置**：`baostock_batch_login/logout`（3437行） + `fetch_single_stock_data`（3397行） + `main()`中的多处调用
- **问题**：4个位置各自管理login/logout，连接状态不可预测，导致WinError 10054
- **修复**：使用上下文管理器统一管理连接生命周期
  ```python
  from contextlib import contextmanager
  
  @contextmanager
  def baostock_session():
      """baostock连接上下文管理器，确保login/logout配对"""
      if DEPENDENCY_STATUS['baostock']:
          try:
              bs.login()
              yield
          finally:
              try:
                  bs.logout()
              except:
                  pass
      else:
          yield
  ```
  在main()中用 `with baostock_session():` 包裹整个数据获取阶段，删除所有分散的login/logout调用

### P0-3: BB_BayesianFusion.fuse_predictions valid标记覆盖Bug
- **位置**：约第2025行 `fuse_predictions` 方法
- **问题**：`traditional_signals.get('valid', True)` 是全局字段，多个模型的valid标记用dict.update()平铺后互相覆盖，导致只检查最后一个模型的valid状态
- **修复**：将valid检查改为每个信号单独检查
  ```python
  # 旧代码（错误）：
  if traditional_signals.get('valid', True) is False:
      continue
  
  # 新代码（正确）：
  signal_valid_key = f'{signal_name}_valid'
  if not traditional_signals.get(signal_valid_key, True):
      continue
  ```
  同时在信号生成端，每个信号用独立key标记valid状态（如 `garch_valid`, `ewma_valid`等）

### P0-4: fetch_single_stock_data降级路径断裂
- **位置**：约第3397行
- **问题**：baostock查询失败后函数隐式返回None，后续的mock数据兜底逻辑永远无法触发
- **修复**：baostock失败后主动生成mock数据
  ```python
  def fetch_single_stock_data(stock_info, start_date, end_date):
      # ...baostock尝试...
      if success and result is not None and len(result) > 0:
          result['name'] = name
          result['code_std'] = code_std
          return result
      
      # 【P0-4修复】降级到mock数据
      _bb_print(f"[数据] ⚠ {name}({code}) 真实数据获取失败，使用模拟数据")
      mock_df = generate_mock_stock_data(code_std, name, start_date, end_date)
      return mock_df
  ```

## 四、P1准确性问题修复

### P1-1: 牛熊标签阈值过紧
- **位置**：BB_LGBMPredictor.create_labels（约1630行）
- **问题**：阈值0.008（0.8%），大部分交易日被错误标记为牛/熊，实际应该用分位数
- **修复**：改用收益率25/75分位数作为阈值
  ```python
  # 旧代码：
  threshold = 0.008
  
  # 新代码：
  threshold_up = returns.quantile(0.75)  # 约1.2%
  threshold_down = returns.quantile(0.25)  # 约-1.0%
  ```

### P1-2: ML模型双重时间分割
- **位置**：time_series_cv方法
- **问题**：train_test_split + TimeSeriesSplit双层分割，训练集仅64%可用数据
- **修复**：只用TimeSeriesSplit，删掉train_test_split
  ```python
  def time_series_cv(self, X, y, n_splits=5):
      tscv = TimeSeriesSplit(n_splits=n_splits)
      # 直接用tscv分割，不再额外train_test_split
  ```

### P1-3: 股票池分类错误
- **位置**：ALL_SECTORS_BB相关常量定义
- **问题**：拓普集团(汽车零部件)→金融板块，比亚迪(新能源车)→科技板块
- **修复**：
  - 拓普集团：从金融板块移到制造业/汽车板块
  - 比亚迪：从科技板块移到制造业/汽车板块或单独新能源板块

### P1-4: 指数RSI计算Bug
- **位置**：BB_FeatureExtractor或calculate_technical_indicators
- **问题**：RSI实现有bug，结果完全错误
- **修复**：使用标准RSI计算
  ```python
  def calculate_rsi(series, period=14):
      delta = series.diff()
      gain = delta.where(delta > 0, 0)
      loss = -delta.where(delta < 0, 0)
      avg_gain = gain.rolling(window=period, min_periods=period).mean()
      avg_loss = loss.rolling(window=period, min_periods=period).mean()
      rs = avg_gain / (avg_loss + 1e-10)
      rsi = 100 - (100 / (1 + rs))
      return rsi
  ```

### P1-5: 版本号不一致
- **问题**：文件内从v1.6到v1.8多处冲突
- **修复**：统一为 v2.0

## 五、元叙事模块引入（从彩票项目迁移）

### 5.1 市场生态系统模拟
将彩票的庄家-彩民生态系统替换为做市商-投资者生态系统：

```python
class MarketEcosystem:
    """市场生态系统模拟 - 从彩票元叙事迁移"""
    
    def __init__(self, maker_strategy="institutional"):
        self.maker_strategy = maker_strategy  # institutional/retail/mixed
        self.agents = []  # 多样化交易Agent
        self.history = []
    
    def simulate_round(self, market_state, agent_predictions):
        """模拟一轮市场博弈"""
        # 做市商根据策略调整报价
        # Agent根据预测调整仓位
        # 产生市场冲击和反馈
        pass

class TradingAgent:
    """交易Agent - 从彩票MetaAgent迁移"""
    
    AGENT_TYPES = [
        'momentum',      # 追涨杀跌
        'contrarian',    # 逆向
        'mean_revert',   # 均值回归
        'breakout',      # 突破
        'value',         # 价值
        'growth',        # 成长
        'quant',         # 量化
        'sentiment',     # 情绪
        'macro',         # 宏观
        'adaptive',      # 自适应（元学习）
    ]
    
    def __init__(self, agent_type, seed=42):
        self.agent_type = agent_type
        self.weights = self._init_weights()
        self.confidence = 0.5
        self.fitness = 0.0
    
    def predict_signal(self, features):
        """输出交易信号（买入/卖出/持有强度）"""
        score = features @ self.weights
        return np.tanh(score)  # 归一化到[-1, 1]
```

### 5.2 集成方式
- 在BB_BayesianFusion.fuse_predictions之后，增加元叙事信号
- 元叙事Agent群体输出综合信号，作为贝叶斯融合的额外先验
- 权重从0开始，验证有效后逐步提升

## 六、代码结构重构

将6401行单文件拆分为模块化结构（但产出仍为单文件，仅内部分区清晰）：

```
stocks3.py
├── [1] 配置与常量 (Config)          ~150行
├── [2] 数据层 (DataLayer)           ~800行  
│   ├── MootdxAdapter
│   ├── BB_DataFetcher
│   ├── baostock_session (新增)
│   └── fetch_single_stock_data (修复)
├── [3] 特征工程 (Features)          ~500行
│   ├── BB_FeatureExtractor
│   └── engineer_all_features (RSI修复)
├── [4] 传统模型 (Traditional)       ~400行
│   └── BB_TraditionalModels
├── [5] ML模型 (MLModels)            ~600行
│   ├── BB_LGBMPredictor (返回值修复+标签修复+CV修复)
│   └── BB_RealtimeAnalyzer
├── [6] 贝叶斯融合 (Fusion)          ~400行
│   └── BB_BayesianFusion (valid修复)
├── [7] 牛熊预判 (Regime)            ~500行
│   └── BB_RegimeAnalyzer
├── [8] 元叙事模块 (MetaNarrative)   ~400行 【新增】
│   ├── MarketEcosystem
│   ├── TradingAgent
│   └── MetaSignalGenerator
├── [9] 组合与风控 (Portfolio)        ~600行
│   ├── improved_kelly_criterion
│   ├── risk_parity_optimization
│   └── risk_control_check
├── [10] 输出与报告 (Output)         ~600行
│   ├── BB_ResultExporter
│   ├── generate_dragon_tiger_board
│   └── run_limit_up_report
├── [11] 主流程 (Main)               ~400行
│   └── main()
├── [12] 涨停/重仓 (Special)         ~400行
│   ├── run_limit_up_report
│   └── run_heavy_position_report
└── [13] 工具函数 (Utils)            ~200行
    ├── QuantErrorHandler
    └── safe_execute
```

## 七、执行要求

1. **输出文件**：`/app/data/所有对话/主对话/stocks3.py`
2. **必须是完整可运行的单文件**，包含所有import和功能
3. **P0修复是硬性要求**，缺一不可
4. **P1修复尽量全部完成**
5. **元叙事模块先搭骨架**，不需要完整实现，但接口和数据结构要定义好
6. **版本号统一为v2.0**
7. **保留所有原有功能**，不能删功能，只能修bug和加新功能
8. **中文注释保持风格一致**
9. **文件开头保留变更日志**，记录v1.6→v2.0所有改动

## 八、验证标准

大修完成后，脚本必须能：
1. `python3 stocks3.py` 不崩溃（至少到数据获取阶段）
2. P0-1: train_and_predict无论成功失败都返回5元素
3. P0-2: baostock连接管理统一，无分散login/logout
4. P0-3: 每个信号有独立valid标记
5. P0-4: fetch_single_stock_data绝不返回None

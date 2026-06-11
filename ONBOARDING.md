# 🧭 新手 Agent 入门模块

> 本文档面向任何接手 Logos 项目的新 AI Agent，提供完整上下文，使其无需阅读历史对话即可理解项目全貌并开始工作。

---

## 一、Logos 是什么

**Logos** 是 neko（主人）的个人品牌，统一命名旗下所有项目。当前核心框架为**四象计划**，包含四个独立运作的板块，全部须在 **2026 年暑假**期间完成。

| 象 | 名称 | 领域 | 一句话定位 |
|----|------|------|-----------|
| 文 | 文渊 | 文学创作 | 炼化旧作提取风格，辅助新作品创作 |
| 理 | 理境 | 生物信息 | 多模态融合科研，文献→实验→论文 |
| 武 | 武道 | 身体管理 | 数据驱动训练与营养，持续追踪 |
| 财 | 财略 | 量化经济 | 课程 Project：彩票博弈对照实验 |

---

## 二、站点与部署

### 2.1 线上站点

- **URL**：https://scilogos.github.io
- **GitHub 仓库**：`Scilogos/scilogos.github.io`
- **部署方式**：GitHub Pages（main 分支自动部署）

### 2.2 站点文件结构

```
scilogos.github.io/
├── index.html          ← 四象总览门户页
├── bg.jpg              ← 背景图（雨中白鹭）
├── wen/index.html      ← 文·文渊 子页面
├── li/index.html       ← 理·理境 子页面
├── wu/index.html       ← 武·武道 子页面
├── cai/index.html      ← 财·财略 项目详情页
├── cai/project-brief.md  ← 彩票博弈项目说明
└── cai/lottery-rules.md  ← 彩票规则汇编 v1.1（含数据获取指南）
```

### 2.3 如何推送文件到 GitHub

**重要：不能使用 `git push` 直连（会 403）。必须使用 GitHub API Content endpoint。**

```python
import requests, json, base64

TOKEN = "从 SECRET.md 获取 Classic Token"
REPO = "Scilogos/scilogos.github.io"
headers = {
    "Authorization": f"token {TOKEN}",
    "Accept": "application/vnd.github.v3+json"
}

# 1. 先 GET 获取已有文件的 SHA（更新已有文件时必须）
def get_sha(path):
    r = requests.get(f"https://api.github.com/repos/{REPO}/contents/{path}", headers=headers)
    if r.status_code == 200:
        return r.json()["sha"]
    return None

# 2. PUT 推送文件
def push_file(path, content_bytes, message, sha=None):
    data = {
        "message": message,
        "content": base64.b64encode(content_bytes).decode(),
    }
    if sha:
        data["sha"] = sha
    r = requests.put(f"https://api.github.com/repos/{REPO}/contents/{path}", headers=headers, json=data)
    return r.status_code, r.json()
```

- **Token 类型**：必须用 Classic Token（Fine-grained Token 无写入权限，已弃用）
- **Token 存储位置**：Agent 的 `SECRET.md` 文件中
- **Token 有效期**：约至 2026-09

---

## 三、项目空间

- **Coze 项目 ID**：`7649036488834875654`
- **项目文件操作**：使用 `coze agent file` CLI 命令
  - `coze agent file list --project-id <id> [--project-dir /path] [--depth 2]`
  - `coze agent file read --project-id <id> --project-file-path /path/file.md`
  - `coze agent file upload --project-id <id> --local-file-path <local> [--project-dir /dir]`
  - `coze agent file download --project-id <id> --project-file-path /path/file.md`
- **工作产出必须上传到项目空间**，确保项目成员可见
- **发送文件给用户**使用 `computer://` + 项目文件绝对路径

---

## 四、各象详情与当前状态

### 4.1 文·文渊

- **状态**：待启动
- **核心流程**：上传旧作 → Agent 深度炼化（风格画像）→ 建立风格档案 → 辅助新作品创作
- **关键文件**：`wen/index.html`（占位页，待内容填充）
- **待办**：等待 neko 上传旧作后启动

### 4.2 理·理境

- **状态**：待启动
- **核心流程**：上传文献 → 结构化笔记 → 方法论对比 → 实验 pipeline → 论文
- **研究方向**：多模态融合（生物信息学）
- **关键文件**：`li/index.html`（占位页，待内容填充）
- **待办**：等待 neko 上传文献或给出研究方向指引

### 4.3 武·武道

- **状态**：基础数据已录入，待制定训练计划
- **基础数据**：

| 指标 | 数值 | 录入日期 |
|------|------|----------|
| 体重 | 83 kg | 2026-06-11 |
| 体脂率 | 27% | 2026-06-11 |

- **核心流程**：上传身体数据 → 制定/调整训练计划 + 营养方案 → 持续追踪进展
- **关键文件**：`wu/index.html`（占位页，待内容填充）
- **待办**：后续将上传更多身体数据；当前可先基于已有数据制定初步方案

### 4.4 财·财略

- **状态**：框架搭建完成，待采集数据与实现模型
- **项目名称**：彩票博弈（量化经济与行为课程 Project）
- **小组规模**：4 人
- **研究问题**：彩票开奖是否倾向开少人买的号码？
- **实验设计**：
  - 实验组 A：深度学习（LSTM/Transformer）预测
  - 实验组 B：对抗学习（GAN 框架，彩民 vs 庄家博弈）
  - 统计检验比较两组差异
- **数据获取方案（已确定）**：优先使用**官方 JSON API**
  - 福彩 API：`https://www.cwl.gov.cn/cwl_admin/front/cwlkj/search/kjxx/findDrawNotice`
    - 参数：`name=ssq/3d/qlc`, `issueCount`, `dayStart`, `dayEnd`
    - 支持：双色球、福彩3D、七乐彩
  - 体彩 API：`https://webapi.sporttery.cn/gateway/lottery/getHistoryPageListV1.qry`
    - 参数：`gameNo=85/35/350/04`, `pageSize`, `pageNo`
    - 支持：大乐透、排列3/5、七星彩
  - 官方 API 免费、无需注册、JSON 格式，写几行 `requests` 即可批量拉取
  - 不推荐 `lotterycn` 包（v0.0.4, 2022，太旧）
- **推荐彩种**：排列3/福彩3D（结构简单数据量大）、双色球/大乐透（社会关注度高）
- **关键瓶颈**：官方不公开各号码具体投注量，需寻找代理变量
- **关键文件**：
  - `cai/project-brief.md` — 项目完整说明
  - `cai/lottery-rules.md` — 彩票规则汇编 v1.1（含数据获取指南与示例代码）
  - `cai/index.html` — 项目详情页

---

## 五、工作规范

### 5.1 工单系统 (WO)

- 所有任务使用 **WO-XXXX** 编号体系流转
- 已完成的工单：WO-0001（项目初始化与看板搭建）
- 工单状态：`OPEN` → `WIP` → `REVIEW` → `DONE`

### 5.2 部署原则

- **所有正式产出优先部署到 scilogos.github.io**，不依赖 Coze 平台临时链接
- 文件先推 GitHub，再上传项目空间

### 5.3 文档规范

- 文档写详细，确保没看到对话的其他 Agent 也能大致理解
- Markdown 格式为主，正式文档用 `.docx`
- 数据文件用 `.csv` / `.xlsx`

### 5.4 命名规范

- 文件名简短有意义，带扩展名
- 只用中英文、数字、下划线、短横线
- 严禁空格及特殊字符

---

## 六、技术栈与工具

| 类别 | 工具 | 说明 |
|------|------|------|
| 站点部署 | GitHub Pages + API 推送 | 不能用 git push，只能用 API |
| 项目文件 | `coze agent file` CLI | 项目空间文件管理 |
| 文件发送 | `computer://` 协议 | 发送文件给用户 |
| 数据采集 | Python + requests | 官方 JSON API 直接拉取 |
| 深度学习 | PyTorch / TensorFlow | 财略实验组 A |
| 对抗学习 | GAN 框架 | 财略实验组 B |
| 网页读取 | fetch_web | 读取已知 URL 内容 |
| 搜索 | search_web | 联网搜索实时信息 |

---

## 七、已做关键决策

| 决策 | 原因 |
|------|------|
| 用 GitHub API 而非 git push | git push 直连返回 403，API 方式成功 |
| 站点从单页改为多页结构 | 四象门户 + 四个子页面，各象独立 |
| Classic Token 优于 Fine-grained | Fine-grained 无写入权限 |
| 彩票数据用官方 API | 免费、权威、比 Python 包更可靠 |
| 不推荐 lotterycn 包 | 版本过旧（2022），生产级项目建议直接用 API |
| 推荐彩种排序 | 排列3/3D 首选（简单+数据量大），竞彩/刮刮乐不推荐 |
| 投注分布数据为关键瓶颈 | 官方不公开，文档列出 4 种替代方案 |

---

## 八、沟通偏好

- neko 偏好：专业术语后括注英文，复习内容生动详细
- 文档风格：详细完整，确保新 Agent 可独立理解
- 工作节奏：不 push，自然推进，但关键节点需确认
- 称呼：neko

---

*新手 Agent 入门模块 v1.0 · 2026-06-11 · 任何新 Agent 接手时请先阅读本文档*

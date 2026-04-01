# Research Literature Trace Agent

基于 LangGraph 的科研文献溯源 Agent。输入一篇 arXiv 论文链接，自动完成多轮引用溯源，输出一条最值得阅读的文献发展脉络与 Markdown 报告。

## 功能概览

- 自动获取 arXiv 论文元数据并下载 PDF 解析引用
- 多轮迭代溯源（可配置轮数 `m` 与每轮 top-k 数量）
- 用 LLM 提炼每篇论文的核心 idea、方法、解决的问题
- 用 networkx 构建引用图（支持节点复用，非树结构）
- LLM 评估选出最合理的一条文献发展脉络
- 输出 Markdown 报告 + JSON 图快照

## 目录结构

```
paper_agent/
├── main.py                        # 运行入口
├── config.example.yaml            # 配置示例文件
├── pyproject.toml
├── outputs/                       # 生成的 Markdown 报告
├── data/                          # 中间 JSON 图快照
└── src/paper_agent/
    ├── config/        settings.py  # 配置加载
    ├── llm/                        # LLM 抽象层
    ├── ingestion/                  # arXiv 元数据获取
    ├── parsing/                    # PDF 全文解析
    ├── summarization/              # 论文摘要提炼
    ├── retrieval/                  # 引用 arXiv ID 解析
    ├── ranking/                    # top-k 候选排序
    ├── graph_store/                # networkx 图封装
    ├── evaluation/                 # 最优脉络评估
    ├── reporting/                  # Markdown 报告生成
    └── tracing/                   # LangGraph Agent 编排
```

## 快速开始

### 1. 安装依赖

```bash
# 推荐使用 uv
uv sync
```

### 2. 配置 LLM

复制 `config.example.yaml` 为本地 `config.yaml`，再填写你的 LLM 服务信息：

```yaml
agent:
  max_rounds: 3   # 最大迭代轮数
  top_k: 3        # 每轮深入调查的候选文献数量

llm:
  protocol: openai            # openai | openai_compatible
  base_url: https://api.openai.com/v1
  api_key: ""                 # 也可通过环境变量设置
  model_id: gpt-4o

paths:
  outputs_dir: outputs
  data_dir: data
```

**通过环境变量设置 API Key（推荐）：**

```bash
export PAPER_AGENT_API_KEY=sk-xxxxxxxxxxxx
```

> 支持任意 OpenAI 兼容接口，例如 DeepSeek、Qwen、本地 vLLM 等，只需修改 `base_url` 和 `model_id`。

### 3. 运行

```bash
uv run python main.py https://arxiv.org/abs/2305.10601
```

**可选参数：**

```bash
uv run python main.py https://arxiv.org/abs/2305.10601 \
  --config config.yaml \
  --max-rounds 2 \
  --top-k 2
```

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `arxiv_url` | arXiv 论文链接或 ID | 必填 |
| `--config` | 配置文件路径 | 自动查找 config.yaml |
| `--max-rounds` | 最大迭代轮数（覆盖 config） | config 中的值 |
| `--top-k` | 每轮 top-k 候选数（覆盖 config） | config 中的值 |

### 4. 查看输出

- **报告**：`outputs/report_<arxiv_id>_<timestamp>.md`
- **图快照**：`data/<arxiv_id>_graph.json`

## 报告内容

生成的 Markdown 报告包含：

1. **最优文献脉络链** — 从根论文出发，LLM 评选的最合理发展路径
2. **脉络说明** — LLM 对该链路的解释（方法演进、问题驱动关系）
3. **阅读清单** — 链路中每篇论文的：
   - 标题、作者、年份
   - 核心 idea
   - 方法是什么
   - 试图解决什么问题
4. **无法获取的文献列表** — 不在 arXiv 上或获取失败的引用

## 配置不同 LLM 服务商

**DeepSeek：**
```yaml
llm:
  protocol: openai
  base_url: https://api.deepseek.com/v1
  model_id: deepseek-chat
```

**本地 Ollama：**
```yaml
llm:
  protocol: openai
  base_url: http://localhost:11434/v1
  api_key: ollama
  model_id: llama3
```

**Qwen（阿里云）：**
```yaml
llm:
  protocol: openai
  base_url: https://dashscope.aliyuncs.com/compatible-mode/v1
  model_id: qwen-max
```

## 迭代逻辑说明

```
输入论文
  └─ Round 1: 解析 PDF 引用 → LLM 选 top-k → 加入 frontier
       └─ Round 2: 对每个 frontier 论文重复上述过程
            └─ Round m: 达到最大轮数 → 停止
                 └─ LLM 评估所有已分析论文 → 选出最优脉络链 → 生成报告
```

- 同一篇论文在多条路径出现时，**复用**已有分析结果（图结构，非树）
- 无法在 arXiv 找到的文献会被记录在报告末尾，不影响其他分支继续迭代

## 依赖

- [LangGraph](https://github.com/langchain-ai/langgraph) — Agent 编排
- [arxiv](https://github.com/lukasschwab/arxiv.py) — arXiv API 客户端
- [pypdf](https://github.com/py-pdf/pypdf) — PDF 解析
- [networkx](https://networkx.org/) — 引用图结构
- [httpx](https://www.python-httpx.org/) — HTTP 客户端

# 基于 FastAPI、Streamlit、LangGraph、Milvus、Redis、MySQL 的扫地机器人知识库问答系统

本项目支持知识库检索、问答生成、会话历史保存、Redis 答案缓存，以及可选的高德地图 MCP 定位和天气工具调用。

## 目录

- [项目截图](#项目截图)
- [功能特性](#功能特性)
- [技术栈](#技术栈)
- [技术架构深度解析](#技术架构深度解析)
- [项目结构](#项目结构)
- [快速开始](#快速开始)
- [API 接口](#api-接口)

## 项目截图

### 前端首页

![前端首页](docs/images/示例1.png)

### 问答示例

![问答示例](docs/images/示例2.png)

### 问答示例

![问答示例](docs/images/示例3.png)

### 问答示例

![问答示例](docs/images/示例4.png)

## 功能特性

- 基于本地知识库进行扫地机器人相关问答
- 使用 BGE-M3 向量模型构建混合检索知识库
- 使用 Milvus 存储和检索向量数据
- 使用 LangGraph 编排 Agent 问答流程
- 使用 DashScope/Qwen 作为大语言模型
- 使用 MySQL 保存多轮会话历史
- 使用 Redis 缓存问答结果
- 提供 FastAPI 后端接口
- 提供 Streamlit 前端页面
- 可选接入高德地图 MCP 工具，支持定位和天气查询

## 技术栈

- Python 3.10+
- FastAPI
- Streamlit
- LangChain / LangGraph
- DashScope / Qwen
- BGE-M3 / FlagEmbedding
- Milvus
- Redis
- MySQL
- PyMySQL
- Pydantic

## 技术架构深度解析

### STEP 1 — 整体架构设计

项目采用“Streamlit 前端 + FastAPI 后端 + LangGraph Agent + 多存储组件”的轻量化分层架构。前端只负责交互展示，后端负责接口编排，Agent 层负责问题改写、路由、检索和最终回答生成，数据层分别承担知识检索、会话历史和答案缓存。

核心组件分工如下：

| 层级 | 技术 | 作用 |
|---|---|---|
| 前端交互层 | Streamlit | 提供聊天界面、会话控制、Redis 缓存开关和回答展示 |
| API 服务层 | FastAPI | 提供 `/chat/stream`、会话清空、知识追加、健康检查等接口 |
| Agent 编排层 | LangGraph | 编排问题改写、本地检索路由、LLM 路由、RAG/Chat 分支准备 |
| 大模型层 | DashScope/Qwen | 负责问题改写、意图判断和最终中文客服回答生成 |
| 检索层 | BGE-M3 + Milvus | 支持 dense/sparse 混合检索，召回知识库候选片段 |
| 重排层 | BGE Reranker | 对 Milvus 召回候选进行二次排序，提升上下文质量 |
| 会话层 | MySQL | 按 `session_id` 保存多轮会话历史 |
| 缓存层 | Redis | 缓存相同会话内相同问题的答案，减少重复模型调用 |
| 工具层 | MCP / 业务工具 | 可选调用定位、天气、故障诊断、保养计划、选购推荐等工具 |

整体问答流程如下：

```text
用户问题
   |
   v
Streamlit 前端
   |
   v
FastAPI 后端
   |
   v
检查 Redis 缓存
   |-- 命中: 返回缓存答案，并写入 MySQL 会话历史
   |
   |-- 未命中:
          |
          v
      LangGraph Agent
          |-- rewrite: 改写为检索查询
          |-- local_router: 本地检索并按分数路由
          |-- llm_router: 必要时使用 LLM 进行二次路由
          |-- prepare_rag / prepare_chat: 准备最终回答消息
          |
          v
      Qwen / ReAct 工具 Agent 生成回答
          |
          |-- 可选工具: 高德 MCP / 业务工具
          |-- 会话历史: MySQL
          |-- 答案缓存: Redis
          |
          v
      返回前端展示
```

这个设计的关键点是：LangGraph 负责回答前的决策流程，检索、路由、工具选择和回答生成各层职责清晰，便于调试和扩展。

### STEP 2 — FastAPI 接口与前端交互

项目通过 FastAPI 对前端提供问答、会话管理和知识追加能力。核心问答接口为：

```http
POST /chat/stream
```

请求体包含用户问题、会话 ID 和是否启用 Redis 缓存：

```json
{
  "question": "扫地机器人报 E13 怎么办？",
  "session_id": "demo-session",
  "use_cache": true
}
```

前端负责维护 `session_id`、聊天消息、后端地址和缓存开关。用户提交问题后，前端把这些信息发送给后端；后端完成缓存检查、会话读取、Agent 编排、知识库检索和回答生成，再把结果返回给前端展示。

### STEP 3 — LangGraph Agent 框架

Agent 的核心实现在 `app/agent.py`。项目使用 `StateGraph(AgentState)` 显式定义节点和条件边，当前图结构如下：

```text
rewrite
   |
   v
local_router
   |-- route = rag ---------> prepare_rag -----> END
   |
   `-- route = llm_router --> llm_router
                              |-- route = chat -> prepare_chat -> END
                              `-- route = rag  -> prepare_rag  -> END
```

各节点职责如下：

| 节点 | 输入 | 输出 | 作用 |
|---|---|---|---|
| `rewrite` | 原始问题、历史消息 | `rewritten_query` | 将用户问题改写成适合知识库检索的查询语句 |
| `local_router` | `rewritten_query` | `docs`、`route` | 调用 Milvus 检索和 rerank，根据最佳分数判断是否直接进入 RAG |
| `llm_router` | 原始问题 | `route` | 当本地检索信心不足时，让 LLM 判断走普通聊天还是 RAG |
| `prepare_chat` | 历史消息 | `answer_messages` | 准备普通聊天分支的最终模型输入 |
| `prepare_rag` | 问题、检索文档 | `answer_messages`、`docs` | 拼接知识库上下文，准备 RAG 分支的最终模型输入 |

`AgentState` 维护图中流转的状态：

```python
class AgentState(TypedDict, total=False):
    messages: List[BaseMessage]
    answer_messages: List[BaseMessage]
    question: str
    rewritten_query: str
    route: RouteName
    docs: List[Document]
```

其中 `route` 有三个取值：

- `rag`：问题需要知识库回答。
- `chat`：普通聊天，不需要知识库。
- `llm_router`：本地检索无法确定，需要 LLM 再判断。

需要注意的是，LangGraph 主要负责问题改写、路由和最终输入准备。回答生成在图执行完成后根据 `answer_messages` 继续执行：

```text
LangGraph.ainvoke(...)
   -> 得到 answer_messages
   -> Qwen / ReAct 工具 Agent 生成最终回答
```

这样设计可以让 LangGraph 保持清晰的流程编排职责，同时让最终回答阶段按是否需要工具调用选择不同的执行方式。

### STEP 4 — RAG 检索增强生成

项目通过 RAG（Retrieval-Augmented Generation）提升回答的准确性和可追溯性。

- 使用 `data/` 目录中的 TXT、PDF 文档作为原始知识来源
- 通过文档加载器读取原始文本
- 使用 `RecursiveCharacterTextSplitter` 将长文档拆分为适合检索的知识片段
- 使用 BGE-M3 生成 dense vector 和 sparse vector
- 将向量、原文片段、来源文件和创建时间写入 Milvus
- 用户提问时先检索相关知识片段，再交给大模型生成最终回答

知识库构建流程：

```text
data/ 文档
   |
   v
DocumentLoader 读取 TXT / PDF
   |
   v
MD5 去重
   |
   v
文本切分
   |
   v
BGE-M3 编码 dense_vector + sparse_vector
   |
   v
写入 Milvus Collection
```

在线检索流程：

```text
用户问题
   |
   v
rewrite 改写查询
   |
   v
BGE-M3 编码查询向量
   |
   v
Milvus hybrid_search
   |-- dense_vector: 语义相似度
   `-- sparse_vector: 关键词/稀疏特征
   |
   v
WeightedRanker 分数融合
   |
   v
BGE Reranker 二次重排
   |
   v
Top-K 文档进入 RAG Prompt
```

默认配置下，系统先召回 `RETRIEVAL_CANDIDATES=10` 条候选，再通过 reranker 返回 `RETRIEVAL_TOP_K=3` 条最相关片段给大模型。

### STEP 5 — 混合检索与重排机制

项目没有只使用单一向量检索，而是使用 BGE-M3 的 dense + sparse 双通道能力：

- dense vector：更擅长语义相近但字面不同的问题，例如“边刷不转”和“侧刷卡住”。
- sparse vector：更擅长保留关键词、型号、错误码，例如“E13”“滤网”“基站”等。

Milvus 中分别创建两个字段和索引：

- `dense_vector`: `FLOAT_VECTOR`
- `sparse_vector`: `SPARSE_FLOAT_VECTOR`

检索时通过 `hybrid_search()` 同时发起 dense 和 sparse 检索请求，再使用 `WeightedRanker(dense_weight, sparse_weight)` 做融合排序。融合后的候选仍不直接作为最终上下文，而是交给 BGE reranker 重新打分：

```text
Milvus 混合召回 10 条
   |
   v
BGE reranker 计算 query-document 相关性
   |
   v
按 rerank_score 排序
   |
   v
返回前 3 条
```

如果 reranker 依赖不可用，系统会自动降级为使用 Milvus 原始排序结果，保证服务仍可运行。

### STEP 6 — MySQL 会话历史

项目使用 MySQL 保存每个 `session_id` 对应的多轮对话。

核心表包括：

- `chat_sessions`：保存会话 ID、创建时间、更新时间。
- `chat_messages`：保存消息顺序、角色、文本内容和 LangChain message JSON。

每次新问题进入 Agent 前，系统会读取当前 `session_id` 的历史消息，并把历史消息与当前用户问题一起放入 `messages` 状态。最终回答生成完成后，系统会清空当前会话旧消息并写入最新的完整上下文：

```text
history.messages
   +
HumanMessage(question)
   +
AIMessage(answer)
   |
   v
写回 MySQL
```

前端的“清空当前会话”会调用：

```http
POST /sessions/{session_id}/clear
```

该接口会删除 MySQL 中当前会话的 `chat_messages`，但保留 `chat_sessions` 会话记录并更新时间。

### STEP 7 — Redis 答案缓存

Redis 用于缓存相同会话中的相同问题答案，避免重复调用检索和大模型。

缓存 key 格式为：

```text
chat_cache:{session_id}:{question_sha256}
```

`use_cache=true` 时：

```text
请求进入 /chat/stream
   |
   v
查询 Redis
   |-- 命中: 直接返回缓存答案，并写入 MySQL 历史
   |
   `-- 未命中: 走 LangGraph + LLM，完成后写入 Redis
```

`use_cache=false` 时，系统既不读取 Redis，也不写入 Redis，确保用户可以绕过缓存获取新回答。

### STEP 8 — MCP 与业务工具调用

项目接入高德地图 MCP 工具，用于扩展 Agent 的外部能力：

- `maps_ip_location`：根据 IP 获取用户所在城市
- `maps_weather`：根据城市查询天气信息
- 仅在用户问题确实需要位置或天气信息时调用
- MCP 工具由后端统一加载，并注入到 Agent 执行流程中

同时，项目还提供了无副作用的业务工具：

- `diagnose_fault`：根据故障现象生成排查建议
- `generate_maintenance_plan`：生成维护保养计划
- `recommend_robot_type`：根据使用场景推荐扫地机器人类型

这些工具会被注入到 `create_react_agent()` 中。LangGraph 主流程负责决定用户问题进入 `chat` 还是 `rag`，最终回答阶段的 ReAct Agent 再根据系统提示词和上下文判断是否调用工具。

为了避免工具误用，系统提示词中明确规定：

- 产品使用、维护、故障、选购类问题优先使用知识库上下文。
- 只有明确涉及位置、天气、本地环境时才调用高德 MCP。
- 普通聊天不调用地图和天气工具。

### STEP 9 — Streamlit Web 界面

前端使用 Streamlit 实现快速交互。

- 提供聊天式问答界面
- 支持会话 ID 管理
- 支持开启或关闭 Redis 答案缓存
- 与 FastAPI 后端通过 HTTP 接口通信

前端主要状态保存在 `st.session_state`：

- `session_id`：当前会话 ID
- `messages`：前端展示用消息列表
- `api_base_url`：后端地址
- `use_cache`：是否启用 Redis 答案缓存

当用户提交问题时，前端会调用后端问答接口，并将返回结果追加到聊天气泡中。

## 项目结构

```text
.
|-- app/                         # 后端核心代码
|   |-- __init__.py
|   |-- agent.py                  # Agent 编排与问答逻辑
|   |-- business_tools.py         # 业务工具
|   |-- config.py                 # 环境变量与配置读取
|   |-- history.py                # MySQL 会话历史管理
|   |-- knowledge_base.py         # 知识库构建与检索
|   |-- loaders.py                # TXT / PDF 文档加载
|   |-- main.py                   # FastAPI 应用入口
|   |-- mcp_client.py             # MCP 工具客户端
|   `-- redis_cache.py            # Redis 缓存
|-- frontend/
|   `-- streamlit_app.py          # Streamlit 前端入口
|-- scripts/
|   |-- init_mysql.py             # 初始化 MySQL 数据库
|   `-- build_knowledge_base.py   # 构建知识库
|-- data/                         # 原始知识库文档
|-- runtime/                      # 运行时记录
|-- .env.example                  # 环境变量示例
|-- requirements.txt              # Python 依赖
`-- README.md
```

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 配置环境变量

复制示例配置：

```bash
cp .env.example .env
```

然后修改 `.env`，填写自己的配置：

```env
DASHSCOPE_API_KEY=your_dashscope_api_key_here
AMAP_MAPS_API_KEY=your_amap_maps_api_key_here

MYSQL_HOST=localhost
MYSQL_PORT=3306
MYSQL_USER=your_mysql_user_here
MYSQL_PASSWORD=your_mysql_password_here
MYSQL_DATABASE=rag_agent

REDIS_ENABLED=true
REDIS_URL=redis://localhost:6379/0

MILVUS_URI=http://localhost:19530
MILVUS_COLLECTION=agent_knowledge
```

如果暂时不使用高德地图 MCP，可以设置：

```env
MCP_ENABLED=false
```

### 3. 启动外部服务

运行项目前需要确保以下服务可用：

- MySQL
- Redis
- Milvus

### 4. 初始化 MySQL

```bash
python scripts/init_mysql.py
```

### 5. 构建知识库

```bash
python scripts/build_knowledge_base.py
```

### 6. 启动后端

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

### 7. 启动前端

```bash
streamlit run frontend/streamlit_app.py
```

## API 接口

后端默认运行在：

```text
http://127.0.0.1:8000
```

常用接口：

- `GET /health`：健康检查
- `POST /chat/stream`：问答接口
- `POST /sessions/{session_id}/clear`：清空指定会话
- `POST /knowledge/documents`：手动添加知识文档

# RAG Agent 项目

基于 FastAPI、Streamlit、LangGraph、Milvus、Redis、MySQL 的扫地机器人知识库问答系统。

项目支持知识库检索、问答生成、会话历史保存、Redis 答案缓存，以及可选的高德地图 MCP 定位和天气工具调用。

## 目录

- [功能特性](#功能特性)
- [技术栈](#技术栈)
- [项目结构](#项目结构)
- [快速开始](#快速开始)
- [API 接口](#api-接口)

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
- `POST /chat/stream`：流式问答
- `POST /sessions/{session_id}/clear`：清空指定会话
- `POST /knowledge/documents`：手动添加知识文档

# RAG Agent 项目

基于 FastAPI、Streamlit、LangGraph、Milvus、Redis、MySQL 的扫地机器人知识库问答系统。

## 项目结构

```text
.
|-- app/          # 后端核心代码
|-- frontend/     # Streamlit 前端
|-- scripts/      # 初始化数据库、构建知识库脚本
|-- data/         # 知识库原始文档
|-- runtime/      # 运行时记录文件
`-- README.md
```

## 环境要求

运行项目前需要准备：

- Python 3.10+
- MySQL
- Redis
- Milvus
- DashScope API Key
- 高德地图 API Key，可选

## 安装依赖

```bash
pip install -r requirements.txt
```

## 配置环境变量

复制示例配置文件：

```bash
cp .env.example .env
```

然后修改 `.env`，填写自己的配置：

- `DASHSCOPE_API_KEY`
- `AMAP_MAPS_API_KEY`
- `MYSQL_USER`
- `MYSQL_PASSWORD`
- Redis 配置
- Milvus 配置

如果暂时不用高德地图 MCP，可以在 `.env` 中设置：

```env
MCP_ENABLED=false
```

## 初始化 MySQL

```bash
python scripts/init_mysql.py
```

## 构建知识库

```bash
python scripts/build_knowledge_base.py
```

## 启动后端

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

## 启动前端

```bash
streamlit run frontend/streamlit_app.py
```

## 提交说明

提交到 Git 时请不要提交以下文件：

- `.env`
- `__pycache__/`
- `*.pyc`
- 运行时缓存文件

建议提交 `.env.example`，用于说明项目需要哪些环境变量。

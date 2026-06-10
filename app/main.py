"""
RAG Agent 的 FastAPI 后端。

运行方式：
    uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
"""
import json
import uuid
from functools import lru_cache
from typing import Optional

from fastapi import Depends, FastAPI
from langchain_core.messages import AIMessage, HumanMessage
from pydantic import BaseModel, Field
from starlette.concurrency import run_in_threadpool
from starlette.responses import StreamingResponse

from app.agent import RagAgent
from app.config import RAGConfig
from app.knowledge_base import KnowledgeBaseBuilder
from app.mcp_client import McpClientManager
from app.redis_cache import RedisService


class ChatRequest(BaseModel):
    question: str = Field(..., min_length=1)
    session_id: Optional[str] = None
    user_ip: Optional[str] = None
    use_cache: bool = True


class AddDocumentRequest(BaseModel):
    content: str = Field(..., min_length=1)
    source: str = "api"


class AddDocumentResponse(BaseModel):
    result: str


@lru_cache(maxsize=1)
def get_config() -> RAGConfig:
    return RAGConfig()


@lru_cache(maxsize=1)
def get_redis_service() -> RedisService:
    return RedisService(get_config())


@lru_cache(maxsize=1)
def get_kb() -> KnowledgeBaseBuilder:
    return KnowledgeBaseBuilder(get_config())


async def load_mcp_tools():
    config = get_config()
    if not config.mcp_enabled:
        print("MCP tools disabled by MCP_ENABLED=false")
        return []

    manager = McpClientManager(config)
    location_tools = await manager.get_tools_for("location")
    weather_tools = await manager.get_tools_for("weather")
    return [*location_tools, *weather_tools]


app = FastAPI(
    title="扫地机器人智能客服 API",
    description="基于 FastAPI、LangGraph、Redis、MySQL、Milvus/BGE-M3 的后端服务。",
    version="2.0.0",
)


@app.on_event("startup")
async def startup_event():
    config = get_config()
    app.state.redis_service = get_redis_service()
    app.state.mcp_tools = await load_mcp_tools()
    user_ip = config.ip or await config.get_public_ip()
    if user_ip:
        print(f"[IP] {user_ip}")
    else:
        print("[IP] public IP unavailable")
    app.state.agent = RagAgent(config=config, mcp_tools=app.state.mcp_tools, user_ip=user_ip)


def get_agent() -> RagAgent:
    return app.state.agent


def get_redis() -> RedisService:
    return app.state.redis_service


@app.get("/health")
async def health(redis_service: RedisService = Depends(get_redis)):
    return {
        "status": "ok",
        "redis": redis_service.available,
        "vector_store": "Milvus",
        "agent": "LangGraph",
        "history_store": "MySQL",
    }


def _json_line(payload: dict) -> str:
    return json.dumps(payload, ensure_ascii=False) + "\n"


@app.post("/chat/stream")
async def chat_stream(
    payload: ChatRequest,
    agent: RagAgent = Depends(get_agent),
    redis_service: RedisService = Depends(get_redis),
):
    session_id = payload.session_id or uuid.uuid4().hex

    async def generate():
        yield _json_line(
            {
                "type": "meta",
                "session_id": session_id,
                "redis_available": redis_service.available,
            }
        )

        if payload.use_cache:
            cached_answer = redis_service.get_cached_answer(session_id, payload.question)
            if cached_answer:
                history = agent.history_manager.get(session_id)
                history.add_messages(
                    [
                        HumanMessage(content=payload.question),
                        AIMessage(content=cached_answer),
                    ]
                )
                yield _json_line({"type": "meta", "cached": True})
                for index in range(0, len(cached_answer), 24):
                    yield _json_line(
                        {
                            "type": "chunk",
                            "content": cached_answer[index : index + 24],
                        }
                    )
                yield _json_line({"type": "done"})
                return

        answer_parts = []
        try:
            async for chunk in agent.query_stream(payload.question, session_id=session_id):
                answer_parts.append(chunk)
                yield _json_line({"type": "chunk", "content": chunk})
        except Exception as exc:
            yield _json_line({"type": "error", "message": str(exc)})
            return

        answer = "".join(answer_parts)
        if payload.use_cache:
            redis_service.set_cached_answer(session_id, payload.question, answer)
        yield _json_line({"type": "done"})

    return StreamingResponse(generate(), media_type="application/x-ndjson")


@app.post("/sessions/{session_id}/clear")
async def clear_session(session_id: str, agent: RagAgent = Depends(get_agent)):
    agent.clear_history(session_id)
    return {"session_id": session_id, "cleared": True}


@app.post("/knowledge/documents", response_model=AddDocumentResponse)
async def add_document(payload: AddDocumentRequest):
    kb = get_kb()
    result = await run_in_threadpool(kb.add_text, payload.content, payload.source)
    return AddDocumentResponse(result=result)

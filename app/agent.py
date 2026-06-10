"""显式 LangGraph 流程：问题改写、路由、RAG 和流式回答。"""
from typing import AsyncIterator, List, Literal, Optional, TypedDict

from dotenv import load_dotenv
from langchain_community.chat_models.tongyi import ChatTongyi
from langchain_core.documents import Document
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_core.tools import BaseTool
from langgraph.graph import END, StateGraph
from langgraph.prebuilt import create_react_agent
from pydantic import BaseModel, Field

from app.business_tools import get_business_tools
from app.config import RAGConfig
from app.history import HistoryManager
from app.knowledge_base import KnowledgeBaseBuilder

load_dotenv()

RouteName = Literal["chat", "rag", "llm_router"]
FinalRouteName = Literal["chat", "rag"]


class RouterDecision(BaseModel):
    """LLM Router 的结构化分类结果。"""

    route: FinalRouteName = Field(
        description="只能选择 chat 或 rag。chat 表示普通聊天；rag 表示需要查询本地知识库。"
    )


class AgentState(TypedDict, total=False):
    messages: List[BaseMessage]
    answer_messages: List[BaseMessage]
    question: str
    rewritten_query: str
    route: RouteName
    docs: List[Document]


SYSTEM_PROMPT = """你是一个专业的扫地/拖地机器人中文客服助手。

规则：
- 始终使用中文回答，语气简洁、专业、像真实客服。
- 当用户询问维护保养、故障排查、选购建议、常见问题时，优先结合知识库上下文回答。
- 当用户只是寒暄、闲聊，或问题不需要知识库时，正常中文对话即可。
- 用户询问产品使用、维护保养、故障排查、选购建议、常见问题时，不要调用高德地图 MCP，优先使用知识库上下文回答。
- 只有当用户明确询问当前位置、所在地天气，或问题包含“这里”“本地”“天气”“气候”“下雨”“温度”等需要位置/天气信息的场景时，才调用高德地图 MCP 工具。
- 需要位置/天气时，先调用 maps_ip_location 获取城市，再按需调用 maps_weather；其他普通聊天不要调用 MCP。

{ip_hint}"""


class RagAgent:
    """包含 rewrite -> router -> chat/rag 的 LangGraph 流式 Agent。"""

    def __init__(
        self,
        config: Optional[RAGConfig] = None,
        mcp_tools: Optional[List[BaseTool]] = None,
        user_ip: str = "",
    ):
        self.config = config or RAGConfig()
        self.mcp_tools = mcp_tools or []
        self.business_tools = get_business_tools()
        self.answer_tools = [*self.mcp_tools, *self.business_tools]
        self.kb = KnowledgeBaseBuilder(self.config)
        self.retriever = self.kb.get_retriever()
        self.history_manager = HistoryManager(self.config)
        self.llm = ChatTongyi(
            model=self.config.chat_model,
            api_key=self.config.api_key,
            streaming=True,
        )
        self.router_llm = self.llm.with_structured_output(RouterDecision)
        ip_hint = (
            f"当前用户公网 IP 是：{user_ip}。"
            "当用户询问当前位置、所在地天气或与本地环境有关的问题时，"
            "请优先使用此 IP 调用 maps_ip_location，再按需要调用 maps_weather。"
            if user_ip
            else ""
        )
        self.system_prompt = SYSTEM_PROMPT.format(ip_hint=ip_hint)
        self.answer_graph = (
            create_react_agent(
                model=self.llm,
                tools=self.answer_tools,
                prompt=self.system_prompt,
            )
            if self.answer_tools
            else None
        )
        self.graph = self._build_graph()

    def _build_graph(self):
        workflow = StateGraph(AgentState)
        workflow.add_node("rewrite", self._rewrite_node)
        workflow.add_node("local_router", self._local_router_node)
        workflow.add_node("llm_router", self._llm_router_node)
        workflow.add_node("prepare_chat", self._prepare_chat_node)
        workflow.add_node("prepare_rag", self._prepare_rag_node)

        workflow.set_entry_point("rewrite")
        workflow.add_edge("rewrite", "local_router")
        workflow.add_conditional_edges(
            "local_router",
            self._select_route,
            {
                "rag": "prepare_rag",
                "llm_router": "llm_router",
            },
        )
        workflow.add_conditional_edges(
            "llm_router",
            self._select_route,
            {
                "chat": "prepare_chat",
                "rag": "prepare_rag",
            },
        )
        workflow.add_edge("prepare_chat", END)
        workflow.add_edge("prepare_rag", END)
        return workflow.compile()

    @staticmethod
    def _latest_human_text(messages: List[BaseMessage]) -> str:
        for message in reversed(messages):
            if isinstance(message, HumanMessage):
                return str(message.content)
        return ""

    @staticmethod
    def _format_docs(docs: List[Document]) -> str:
        parts = []
        for doc in docs:
            source = doc.metadata.get("source", "未知来源")
            score = doc.metadata.get("rerank_score", doc.metadata.get("retrieval_score", ""))
            score_text = f" score={score:.4f}" if isinstance(score, float) else ""
            parts.append(f"[source: {source}{score_text}]\n{doc.page_content}")
        return "\n\n---\n\n".join(parts)

    @staticmethod
    def _best_score(docs: List[Document]) -> float:
        if not docs:
            return 0.0
        score = docs[0].metadata.get("rerank_score", docs[0].metadata.get("retrieval_score", 0.0))
        try:
            return float(score)
        except (TypeError, ValueError):
            return 0.0

    @staticmethod
    def _select_route(state: AgentState) -> RouteName:
        return state.get("route", "chat")

    async def _rewrite_node(self, state: AgentState) -> AgentState:
        question = state.get("question") or self._latest_human_text(state["messages"])
        prompt = [
            SystemMessage(
                content=(
                    "请将用户问题改写成一句适合知识库检索的中文查询语句。"
                    "保留产品名、错误码、故障现象、场景限制等关键信息。"
                    "只返回改写后的查询语句，不要解释。"
                )
            ),
            HumanMessage(content=question),
        ]
        response = await self.llm.ainvoke(prompt)
        rewritten = str(response.content).strip() or question
        return {**state, "question": question, "rewritten_query": rewritten}

    async def _local_router_node(self, state: AgentState) -> AgentState:
        query = state["rewritten_query"]
        docs = self.retriever.invoke(query)
        if docs and self._best_score(docs) >= self.config.router_match_threshold:
            return {**state, "docs": docs, "route": "rag"}
        return {**state, "docs": docs, "route": "llm_router"}

    async def _llm_router_node(self, state: AgentState) -> AgentState:
        question = state["question"]
        decision = await self.router_llm.ainvoke(
            [
                SystemMessage(
                    content=(
                        "请判断用户请求应该进入哪个处理分支。\n"
                        "chat：寒暄、闲聊，或不需要项目知识库的问题。\n"
                        "rag：扫地/拖地机器人产品知识、维护保养、故障排查、选购建议、常见问题。\n"
                        "必须只在结构化字段 route 中选择 chat 或 rag。"
                    )
                ),
                HumanMessage(content=question),
            ]
        )
        route: FinalRouteName = decision.route
        return {**state, "route": route}

    async def _prepare_chat_node(self, state: AgentState) -> AgentState:
        if self.answer_graph is not None:
            answer_messages = state["messages"]
        else:
            answer_messages = [
                SystemMessage(content=self.system_prompt),
                *state["messages"],
            ]
        return {**state, "answer_messages": answer_messages}

    async def _prepare_rag_node(self, state: AgentState) -> AgentState:
        docs = state.get("docs") or self.retriever.invoke(state["rewritten_query"])
        answer_prompt = self._build_rag_answer_prompt(state["question"], docs)
        if self.answer_graph is not None:
            answer_messages = [
                *state["messages"],
                HumanMessage(content=answer_prompt),
            ]
        else:
            answer_messages = [
                SystemMessage(content=self.system_prompt),
                HumanMessage(content=answer_prompt),
            ]
        return {**state, "docs": docs, "answer_messages": answer_messages}

    def _build_rag_answer_prompt(self, question: str, docs: List[Document]) -> str:
        context = self._format_docs(docs)
        return (
            "请基于以下信息回答用户问题。知识库上下文必须优先采用；如果资料不足，请明确说明。\n"
            "你可以按需调用故障诊断、保养计划、选购推荐等业务工具。"
            "只有当问题确实需要当前位置或天气信息时，才调用高德地图 MCP 工具。\n\n"
            f"用户问题：{question}\n\n"
            f"知识库上下文：\n{context}"
        )

    @staticmethod
    def _chunk_text(content) -> str:
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts = []
            for item in content:
                if isinstance(item, str):
                    parts.append(item)
                elif isinstance(item, dict) and item.get("type") == "text":
                    parts.append(str(item.get("text", "")))
            return "".join(parts)
        return str(content or "")

    async def _stream_llm_messages(self, messages: List[BaseMessage]) -> AsyncIterator[str]:
        async for chunk in self.llm.astream(messages):
            text = self._chunk_text(getattr(chunk, "content", ""))
            if text:
                yield text

    async def _stream_answer_graph(self, payload: dict) -> AsyncIterator[str]:
        async for event in self.answer_graph.astream_events(payload, version="v2"):
            if event.get("event") != "on_chat_model_stream":
                continue
            chunk = event.get("data", {}).get("chunk")
            text = self._chunk_text(getattr(chunk, "content", ""))
            if text:
                yield text

    async def query_stream(
        self,
        question: str,
        session_id: str = "default",
    ) -> AsyncIterator[str]:
        history = self.history_manager.get(session_id)
        messages = history.messages + [HumanMessage(content=question)]
        state = await self.graph.ainvoke(
            {
                "messages": messages,
                "question": question,
            }
        )

        chunks = []
        if self.answer_graph is not None:
            stream = self._stream_answer_graph({"messages": state["answer_messages"]})
        else:
            stream = self._stream_llm_messages(state["answer_messages"])

        async for chunk in stream:
            chunks.append(chunk)
            yield chunk

        answer = "".join(chunks)
        history.clear()
        history.add_messages(messages + [AIMessage(content=answer)])

    def clear_history(self, session_id: str = "default") -> None:
        self.history_manager.get(session_id).clear()

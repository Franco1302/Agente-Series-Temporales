"""Nodo de recuperación de contexto teórico mediante el sistema RAG."""

from __future__ import annotations

import time
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from src.agent.state import AgentState
# Importamos la herramienta y su extractor lateral de contexto seguro
from src.tools.rag_tool import consultar_teoria, pop_last_retrieval

# Componentes de la infraestructura analítica del sistema
from src.observability.context import get_current_span, get_thread_id, get_trace_id, new_span_id
from src.observability.events import EVENT_RAG_RETRIEVAL, TraceEvent
from src.observability.logger import emit, is_enabled


def _extract_query(state: AgentState) -> str:
    """Devuelve la query refinada de la tool call si existe; si no, el último HumanMessage."""
    messages = state.get("messages", [])

    for msg in reversed(messages):
        if not isinstance(msg, AIMessage):
            continue
        tool_calls = getattr(msg, "tool_calls", None) or []
        if not tool_calls:
            continue
        call = tool_calls[0]
        name = call.get("name") if isinstance(call, dict) else getattr(call, "name", "")
        if name != "consultar_teoria":
            break
        args = call.get("args") if isinstance(call, dict) else getattr(call, "args", {})
        query = (args or {}).get("query")
        if isinstance(query, str) and query.strip():
            return query.strip()
        break

    for msg in reversed(messages):
        if isinstance(msg, HumanMessage):
            return str(msg.content).strip()

    return ""


def recuperar_contexto_node(state: AgentState) -> dict:
    """Ejecuta una búsqueda RAG, mide su latencia y emite métricas estructuradas."""
    messages = state.get("messages", [])
    query_text = _extract_query(state)

    tool_call_id: str | None = None
    for msg in reversed(messages):
        if not isinstance(msg, AIMessage):
            continue
        tool_calls = getattr(msg, "tool_calls", None) or []
        if tool_calls:
            call = tool_calls[0]
            tool_call_id = (
                call.get("id") if isinstance(call, dict) else getattr(call, "id", None)
            )
        break

    if not query_text or not tool_call_id:
        return {}

    t0 = time.perf_counter()
    
    # Invocación real síncrona de la herramienta RAG documental
    result = consultar_teoria.invoke({"query": query_text})
    
    duration_ms = (time.perf_counter() - t0) * 1000.0

    # Extraemos de forma limpia las métricas del RAG (Scores, Chunks, Inner Tokens)
    rag_metrics = pop_last_retrieval()

    # Si la observabilidad está activa y se recuperaron métricas, disparamos el evento local
    if is_enabled() and rag_metrics:
        emit(
            TraceEvent(
                trace_id=get_trace_id(),
                thread_id=get_thread_id(),
                name="rag.consultar_teoria",
                event_type=EVENT_RAG_RETRIEVAL,
                span_id=new_span_id(),
                parent_span_id=get_current_span(),
                duration_ms=duration_ms,
                attributes=rag_metrics
            )
        )

    if isinstance(result, str) and result.startswith("Error:"):
        return {"error_info": result}

    return {
        "messages": [
            ToolMessage(
                content=str(result),
                name="consultar_teoria",
                tool_call_id=tool_call_id,
            )
        ]
    }

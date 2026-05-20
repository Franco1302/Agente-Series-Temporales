"""Nodo de ejecución de herramientas: ejecuta las tool calls del LLM."""

from __future__ import annotations

import asyncio
import json
import time
from typing import Any, Optional

from langchain_core.messages import AIMessage, ToolMessage
from langgraph.prebuilt import ToolNode

from src.agent.state import AgentState
from src.agent.tools import AGENT_TOOLS
from src.observability.context import get_current_span, get_thread_id, get_trace_id, new_span_id
from src.observability.events import EVENT_API_HTTP, EVENT_TOOL_CALL_END, EVENT_TOOL_CALL_START, TraceEvent
from src.observability.logger import emit, is_enabled

_tool_node = ToolNode(AGENT_TOOLS)

def _run_tools_sync(state: AgentState) -> dict:
    """Ejecuta ToolNode de forma síncrona controlando el loop de asyncio."""
    coro = _tool_node.ainvoke(state)
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)
    return loop.run_until_complete(coro)

def _pop_observability(msg: ToolMessage) -> Optional[list[dict[str, Any]]]:
    """Extrae de forma directa la telemetría sin bucles redundantes."""
    if not msg or not msg.content:
        return None
    try:
        # Al venir del adaptador MCP estructurado, se trata directamente como JSON dict o string legible
        raw_content = msg.content
        data = json.loads(raw_content) if isinstance(raw_content, str) else raw_content
        
        if isinstance(data, list) and data and isinstance(data[0], dict):
            data = json.loads(data[0].get("text", "{}"))

        if isinstance(data, dict):
            http_logs = data.pop("_observability", None)
            msg.content = json.dumps(data, ensure_ascii=False)
            return http_logs if isinstance(http_logs, list) else None
    except Exception:
        pass
    return None

def tool_execution_node(state: AgentState) -> dict:
    """Orquesta las herramientas eliminando sobrecarga de código."""
    messages = state.get("messages", [])
    last_ai_msg = next((m for m in reversed(messages) if isinstance(m, AIMessage) and m.tool_calls), None)

    if not last_ai_msg:
        return _run_tools_sync(state)

    tool_call = last_ai_msg.tool_calls[0]
    tool_name = tool_call.get("name", "unknown")
    tool_call_id = tool_call.get("id")

    if is_enabled():
        emit(TraceEvent(get_trace_id(), get_thread_id(), f"tool.{tool_name}", EVENT_TOOL_CALL_START, 
                        new_span_id(), get_current_span(), 
                        attributes={"tool_name": tool_name, 
                                    "tool_call_id": tool_call_id
                                    }))

    t0 = time.perf_counter()
    try:
        result = _run_tools_sync(state)
        duration_ms = (time.perf_counter() - t0) * 1000.0
        result_messages = result.get("messages", [])

        if result_messages and isinstance(result_messages[-1], ToolMessage):
            last_msg = result_messages[-1]
            http_logs = _pop_observability(last_msg)
            is_ok = "error" not in str(last_msg.content).lower()

            if http_logs and is_enabled():
                for log in http_logs:
                    emit(TraceEvent(get_trace_id(), 
                                    get_thread_id(), f"api_http.{tool_name}", 
                                    EVENT_API_HTTP, new_span_id(), get_current_span(), 
                                    duration_ms=log.get("duration_ms"), 
                                    attributes={"tool_name": tool_name, 
                                                "method": log.get("method"), 
                                                "endpoint": log.get("endpoint"), 
                                                "status_code": log.get("status_code")}))

            if is_enabled():
                emit(TraceEvent(get_trace_id(), 
                                get_thread_id(), 
                                f"tool.{tool_name}", 
                                EVENT_TOOL_CALL_END, 
                                new_span_id(), 
                                get_current_span(), 
                                duration_ms=duration_ms, 
                                attributes={"tool_name": tool_name, 
                                            "tool_call_id": tool_call_id, 
                                            "ok": is_ok, "result_kind": ""
                                            "json" if is_ok else "error"}))

        return result
    except Exception as exc:
        if is_enabled():
            emit(TraceEvent(get_trace_id(), 
                            get_thread_id(), 
                            f"tool.{tool_name}", 
                            EVENT_TOOL_CALL_END, 
                            new_span_id(), 
                            get_current_span(), 
                            duration_ms=(time.perf_counter() - t0)*1000.0, 
                            attributes={"tool_name": tool_name, 
                                        "tool_call_id": tool_call_id, 
                                        "ok": False, 
                                        "result_kind": "error"}))
        raise exc
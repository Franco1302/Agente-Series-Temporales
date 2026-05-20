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


def _parse_tool_payload(msg: ToolMessage) -> tuple[Optional[dict], Optional[list[dict[str, Any]]]]:
    """Parsea el ToolMessage y separa la telemetría HTTP del resultado real.

    Parámetros:
        msg: ToolMessage devuelto por la ejecución de la herramienta.

    Retorno:
        Tupla ``(data, http_logs)`` donde ``data`` es el dict de
        resultado de la tool (ya sin la clave reservada
        ``_observability``) y ``http_logs`` la lista de peticiones HTTP
        recogidas en el servidor MCP. Reescribe ``msg.content`` para que
        el LLM nunca vea la telemetría. Devuelve ``(None, None)`` si el
        contenido no se puede interpretar como dict.
    """
    if not msg or not msg.content:
        return None, None
    try:
        raw = msg.content
        data = json.loads(raw) if isinstance(raw, str) else raw
        # El adaptador MCP estructurado envuelve el dict en [{"text": "..."}]
        if isinstance(data, list) and data and isinstance(data[0], dict):
            data = json.loads(data[0].get("text", "{}"))
        if not isinstance(data, dict):
            return None, None
        http_logs = data.pop("_observability", None)
        msg.content = json.dumps(data, ensure_ascii=False)
        return data, http_logs if isinstance(http_logs, list) else None
    except Exception:
        return None, None


def _classify_result(data: Optional[dict]) -> tuple[bool, str]:
    """Determina ``(ok, result_kind)`` a partir del dict de resultado.

    ``result_kind`` es una de las etiquetas ``csv``/``png``/``json``/
    ``error`` según los artefactos presentes: las tools MCP devuelven
    ``image_path`` para gráficas, ``output_path`` para CSVs generados y
    una clave ``error`` cuando la llamada falla.
    """
    if not isinstance(data, dict):
        return False, "error"
    if data.get("error"):
        return False, "error"
    if data.get("image_path"):
        return True, "png"
    if data.get("output_path"):
        return True, "csv"
    return True, "json"


def _emit_http_events(tool_name: str, http_logs: Optional[list[dict[str, Any]]]) -> None:
    """Emite un evento ``api_http`` por cada petición HTTP del servidor MCP."""
    for log in http_logs or []:
        emit(TraceEvent(
            trace_id=get_trace_id(),
            thread_id=get_thread_id(),
            name=f"api_http.{tool_name}",
            event_type=EVENT_API_HTTP,
            span_id=new_span_id(),
            parent_span_id=get_current_span(),
            duration_ms=log.get("duration_ms"),
            attributes={
                "tool_name": tool_name,
                "method": log.get("method"),
                "endpoint": log.get("endpoint"),
                "status_code": log.get("status_code"),
            },
        ))


def _emit_tool_end(
    tool_name: str,
    tool_call_id: Optional[str],
    duration_ms: float,
    ok: bool,
    result_kind: str,
) -> None:
    """Emite el evento ``tool_call_end`` con el resultado clasificado."""
    emit(TraceEvent(
        trace_id=get_trace_id(),
        thread_id=get_thread_id(),
        name=f"tool.{tool_name}",
        event_type=EVENT_TOOL_CALL_END,
        span_id=new_span_id(),
        parent_span_id=get_current_span(),
        duration_ms=duration_ms,
        attributes={
            "tool_name": tool_name,
            "tool_call_id": tool_call_id,
            "ok": ok,
            "result_kind": result_kind,
        },
    ))


def tool_execution_node(state: AgentState) -> dict:
    """Ejecuta las tool calls del LLM y emite los eventos de observabilidad.

    La instrumentación (eventos ``tool_call_start``/``tool_call_end`` y
    ``api_http``) se añade alrededor de la ejecución sin alterar la
    lógica: ``_parse_tool_payload`` se invoca siempre para retirar la
    telemetría inyectada por el servidor MCP, de modo que el LLM nunca la
    ve aunque el subsistema esté desactivado.
    """
    messages = state.get("messages", [])
    last_ai_msg = next(
        (m for m in reversed(messages) if isinstance(m, AIMessage) and m.tool_calls),
        None,
    )

    if not last_ai_msg:
        return _run_tools_sync(state)

    tool_call = last_ai_msg.tool_calls[0]
    tool_name = tool_call.get("name", "unknown")
    tool_call_id = tool_call.get("id")

    if is_enabled():
        emit(TraceEvent(
            trace_id=get_trace_id(),
            thread_id=get_thread_id(),
            name=f"tool.{tool_name}",
            event_type=EVENT_TOOL_CALL_START,
            span_id=new_span_id(),
            parent_span_id=get_current_span(),
            attributes={"tool_name": tool_name, "tool_call_id": tool_call_id},
        ))

    t0 = time.perf_counter()
    try:
        result = _run_tools_sync(state)
    except Exception:
        if is_enabled():
            _emit_tool_end(
                tool_name, tool_call_id,
                (time.perf_counter() - t0) * 1000.0,
                ok=False, result_kind="error",
            )
        raise

    duration_ms = (time.perf_counter() - t0) * 1000.0
    result_messages = result.get("messages", [])
    last_msg = result_messages[-1] if result_messages else None

    if isinstance(last_msg, ToolMessage):
        # Siempre: retira la telemetría del contenido (aunque obs. esté off).
        data, http_logs = _parse_tool_payload(last_msg)
        ok, result_kind = _classify_result(data)

        if is_enabled():
            _emit_http_events(tool_name, http_logs)
            _emit_tool_end(tool_name, tool_call_id, duration_ms, ok, result_kind)

    return result

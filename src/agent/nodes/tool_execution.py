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

# Imports de la infraestructura base de observabilidad
from src.observability.context import get_current_span, get_thread_id, get_trace_id, new_span_id
from src.observability.events import (
    EVENT_API_HTTP,
    EVENT_TOOL_CALL_END,
    EVENT_TOOL_CALL_START,
    TraceEvent,
)
from src.observability.logger import emit, is_enabled

# consultar_teoria se maneja en recuperar_contexto_node, no aquí.
# ToolNode despacha automáticamente el resto de herramientas por nombre.
_tool_node = ToolNode(AGENT_TOOLS)


def _run_tools_sync(state: AgentState) -> dict:
    """Ejecuta ToolNode aceptando tools async-only (como las MCP via stdio)."""
    coro = _tool_node.ainvoke(state)
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)
    return loop.run_until_complete(coro)


def _pop_observability(msg: ToolMessage) -> Optional[list[dict[str, Any]]]:
    """Parseo defensivo estricto para extraer y eliminar la telemetría del subproceso.
    
    Garantiza que el LLM jamás sufra contaminación de contexto por metadatos.
    """
    if not msg or not getattr(msg, "content", None):
        return None

    content_str = str(msg.content).strip()
    
    try:
        # Fast path: si no empieza y termina con llaves, no es un JSON de tool MCP exitoso
        if not (content_str.startswith("{") and content_str.endswith("}")):
            return None
            
        data = json.loads(content_str)
        if not isinstance(data, dict):
            return None

        # Extraemos la clave reservada inyectada por el hook httpx en el MCP
        http_logs = data.pop("_observability", None)

        if http_logs is not None:
            # Re-serializamos el diccionario limpio de vuelta al cuerpo del mensaje
            msg.content = json.dumps(data, ensure_ascii=False)
            return http_logs if isinstance(http_logs, list) else None

    except (json.JSONDecodeError, TypeError, AssertionError):
        # Fallo defensivo inocuo (ej. string plano de error). Seguimos adelante.
        pass
        
    return None


def _determine_result_kind(msg: ToolMessage) -> str:
    """Deduce analíticamente el tipo de resultado devuelto por la herramienta."""
    content = str(msg.content)
    if "error" in content.lower() or "exception" in content.lower() or content.startswith("Error:"):
        return "error"
    if "csv_path" in content:
        return "csv"
    if "chart_path" in content or "png" in content:
        return "png"
    return "json"


def tool_execution_node(state: AgentState) -> dict:
    """Ejecuta las tool calls pendientes en el último AIMessage del estado.

    Extrae de forma invisible la telemetría HTTP acoplada en el subproceso MCP,
    emite los eventos estructurados locales y limpia el contexto para el LLM.
    """
    messages = state.get("messages", [])

    # 1. Localizar la tool_call activa en el último mensaje emitido por el razonador
    last_ai_msg: Optional[AIMessage] = None
    for msg in reversed(messages):
        if isinstance(msg, AIMessage) and getattr(msg, "tool_calls", None):
            last_ai_msg = msg
            break

    # Si no hay llamadas pendientes por anomalía estructural, delegamos en el flujo nativo
    if not last_ai_msg or not last_ai_msg.tool_calls:
        try:
            result = _run_tools_sync(state)
            return {"messages": result.get("messages", []), "error_info": None}
        except Exception as exc:
            return {"messages": [ToolMessage(content=f"Error: {exc}", tool_call_id="error", name="error")], "error_info": str(exc)}

    # Extraemos metadatos analíticos de la llamada activa
    tool_call = last_ai_msg.tool_calls[0]
    tool_name = tool_call.get("name", "unknown_tool")
    tool_call_id = tool_call.get("id")

    trace_id = get_trace_id()
    thread_id = get_thread_id()
    current_span_id = get_current_span()

    # 2. Emitir EVENT_TOOL_CALL_START
    if is_enabled():
        emit(
            TraceEvent(
                trace_id=trace_id,
                thread_id=thread_id,
                name=f"tool.{tool_name}",
                event_type=EVENT_TOOL_CALL_START,
                span_id=new_span_id(),
                parent_span_id=current_span_id,
                attributes={
                    "tool_name": tool_name,
                    "tool_call_id": tool_call_id,
                    "args": str(tool_call.get("args", {}))[:200]
                }
            )
        )

    t0 = time.perf_counter()
    
    try:
        # 3. Invocar la ejecución real original a través de ToolNode
        result = _run_tools_sync(state)
        duration_ms = (time.perf_counter() - t0) * 1000.0
        
        result_messages = result.get("messages", [])
        result_kind = "json"

        if result_messages:
            last_msg = result_messages[-1]
            if isinstance(last_msg, ToolMessage):
                # 4. Extraer y remover de forma invisible la telemetría HTTP del JSON
                http_logs = _pop_observability(last_msg)
                result_kind = _determine_result_kind(last_msg)

                # 5. Si viajaba telemetría HTTP válida, disparamos eventos api_http individuales
                if http_logs and is_enabled():
                    for log in http_logs:
                        emit(
                            TraceEvent(
                                trace_id=trace_id,
                                thread_id=thread_id,
                                name=f"api_http.{tool_name}",
                                event_type=EVENT_API_HTTP,
                                span_id=new_span_id(),
                                parent_span_id=current_span_id,
                                duration_ms=log.get("duration_ms"),
                                attributes={
                                    "tool_name": tool_name,
                                    "method": log.get("method"),
                                    "endpoint": log.get("endpoint"),
                                    "status_code": log.get("status_code")
                                }
                            )
                    )

        # 6. Emitir EVENT_TOOL_CALL_END exitoso
        if is_enabled():
            emit(
                TraceEvent(
                    trace_id=trace_id,
                    thread_id=thread_id,
                    name=f"tool.{tool_name}",
                    event_type=EVENT_TOOL_CALL_END,
                    span_id=new_span_id(),
                    parent_span_id=current_span_id,
                    duration_ms=duration_ms,
                    attributes={
                        "tool_name": tool_name,
                        "tool_call_id": tool_call_id,
                        "ok": result_kind != "error",
                        "result_kind": result_kind
                    }
                )
            )

        return {
            "messages": result_messages,
            "error_info": None,
        }

    except Exception as exc:
        duration_ms = (time.perf_counter() - t0) * 1000.0
        error_msg = ToolMessage(content=f"Error al ejecutar herramienta: {exc}", tool_call_id=tool_call_id or "error", name=tool_name)
        
        # Emitir EVENT_TOOL_CALL_END fallido
        if is_enabled():
            emit(
                TraceEvent(
                    trace_id=trace_id,
                    thread_id=thread_id,
                    name=f"tool.{tool_name}",
                    event_type=EVENT_TOOL_CALL_END,
                    span_id=new_span_id(),
                    parent_span_id=current_span_id,
                    duration_ms=duration_ms,
                    attributes={"tool_name": tool_name, "tool_call_id": tool_call_id, "ok": false, "result_kind": "error"}
                )
            )
            
        return {
            "messages": [error_msg],
            "error_info": str(exc),
        }
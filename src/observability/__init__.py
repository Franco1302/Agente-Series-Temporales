"""Subsistema de observabilidad local: trazas JSON Lines por turno conversacional.

Uso típico desde el resto del código:

    from src.observability import emit, TraceEvent, EVENT_NODE_ENTER, get_trace_id

    emit(TraceEvent(
        trace_id=get_trace_id(),
        thread_id=get_thread_id(),
        name="razonador",
        event_type=EVENT_NODE_ENTER,
        attributes={...},
    ))

El subsistema se activa con ``OBSERVABILITY_ENABLED=true`` en ``.env``.
Si está desactivado, ``emit`` es un no-op y el agente funciona idéntico.
"""

from __future__ import annotations

from .context import (
    get_current_span,
    get_thread_id,
    get_trace_id,
    new_span_id,
    span,
    start_turn,
)
from .events import (
    EVENT_API_HTTP,
    EVENT_ERROR,
    EVENT_LLM_CALL,
    EVENT_NODE_ENTER,
    EVENT_NODE_EXIT,
    EVENT_PARAMS_INHERITED,
    EVENT_RAG_RETRIEVAL,
    EVENT_TOOL_CALL_END,
    EVENT_TOOL_CALL_START,
    EVENT_TURN_END,
    EVENT_TURN_START,
    TraceEvent,
)
from .llm_metrics import emit_llm_call, extract_llm_attributes
from .logger import configure, emit, is_enabled, log_file_path
from .reader import read_recent_thread_lines, read_trace_lines

__all__ = [
    "TraceEvent",
    "configure",
    "emit",
    "emit_llm_call",
    "extract_llm_attributes",
    "is_enabled",
    "log_file_path",
    "read_recent_thread_lines",
    "read_trace_lines",
    "span",
    "start_turn",
    "new_span_id",
    "get_trace_id",
    "get_thread_id",
    "get_current_span",
    "EVENT_TURN_START",
    "EVENT_TURN_END",
    "EVENT_NODE_ENTER",
    "EVENT_NODE_EXIT",
    "EVENT_LLM_CALL",
    "EVENT_TOOL_CALL_START",
    "EVENT_TOOL_CALL_END",
    "EVENT_API_HTTP",
    "EVENT_RAG_RETRIEVAL",
    "EVENT_PARAMS_INHERITED",
    "EVENT_ERROR",
]

"""Extracción de métricas del LLM y emisión de eventos `llm_call`.

Los campos extraídos siguen el esquema definido en el plan:
``model``, ``n_messages_in``, ``prompt_chars``, ``input_tokens``,
``output_tokens``, ``tokens_per_sec``, ``decided``, ``tool_name``.
Cuando alguno no se puede obtener (p. ej. el modelo
local no rellena ``response_metadata`` o falla la división por
``eval_duration=0``) se devuelve ``None`` en lugar de propagar la
excepción.
"""

from __future__ import annotations

from typing import Any, Optional

from .context import get_current_span, get_thread_id, get_trace_id, new_span_id
from .events import EVENT_LLM_CALL, TraceEvent
from .logger import emit, is_enabled


def _safe_get(mapping: Any, key: str) -> Any:
    """Lee ``mapping[key]`` tolerando ``None`` y objetos que no son dict."""
    if mapping is None:
        return None
    if isinstance(mapping, dict):
        return mapping.get(key)
    return getattr(mapping, key, None)


def _count_prompt_chars(messages: list) -> int:
    """Devuelve la suma de longitudes de ``.content`` de todos los mensajes.

    Soporta el caso (poco común con Ollama) en que ``content`` es una lista
    de bloques tipo ``[{"type": "text", "text": "..."}]``.
    """
    total = 0
    for msg in messages:
        content = getattr(msg, "content", None)
        if isinstance(content, str):
            total += len(content)
        elif isinstance(content, list):
            for part in content:
                if isinstance(part, dict):
                    total += len(str(part.get("text", "")))
                else:
                    total += len(str(part))
    return total


def _extract_first_tool_name(response: Any) -> Optional[str]:
    """Si la respuesta tiene tool_calls, devuelve el name del primero."""
    tool_calls = getattr(response, "tool_calls", None) or []
    if not tool_calls:
        return None
    first = tool_calls[0]
    if isinstance(first, dict):
        return first.get("name")
    return getattr(first, "name", None)


def extract_llm_attributes(
    *,
    messages: list,
    response: Any,
    model: Optional[str] = None,
) -> dict[str, Any]:
    """Construye el diccionario ``attributes`` de un evento ``llm_call``.

    Parámetros:
        messages: Lista pasada a ``llm.invoke(messages)``. Se usa para
            ``n_messages_in`` y ``prompt_chars``.
        response: Respuesta del LLM.
        model: Nombre del modelo (informativo). Si es ``None`` se intenta
            extraer del ``response_metadata`` del LLM.

    Retorno:
        Diccionario con las claves esperadas por el evento ``llm_call``.
    """
    has_tool_calls = bool(getattr(response, "tool_calls", None) or [])

    usage = getattr(response, "usage_metadata", None)
    meta = getattr(response, "response_metadata", None) or {}

    input_tokens = _safe_get(usage, "input_tokens")
    output_tokens = _safe_get(usage, "output_tokens")
    eval_count = _safe_get(meta, "eval_count")
    eval_duration = _safe_get(meta, "eval_duration")

    tokens_per_sec: Optional[float] = None
    if eval_count and eval_duration:
        try:
            tokens_per_sec = float(eval_count) / (float(eval_duration) / 1e9)
        except (ZeroDivisionError, TypeError, ValueError):
            tokens_per_sec = None

    decided = "tool_call" if has_tool_calls else "text"
    tool_name = _extract_first_tool_name(response)

    if model is None:
        model = _safe_get(meta, "model")

    return {
        "model": model,
        "n_messages_in": len(messages),
        "prompt_chars": _count_prompt_chars(messages),
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "tokens_per_sec": tokens_per_sec,
        "decided": decided,
        "tool_name": tool_name,
    }


def emit_llm_call(
    *,
    name: str,
    messages: list,
    response: Any,
    duration_ms: float,
    model: Optional[str] = None,
) -> None:
    """Emite un evento ``llm_call`` con todos los atributos estándar.

    El evento se cuelga del span actualmente activo (``parent_span_id``)
    sin modificar el ContextVar, de modo que no afecta al anidamiento de
    spans dentro del nodo decorado por :func:`traced_node`.

    Si el subsistema está apagado, la función es un no-op rápido.
    """
    if not is_enabled():
        return
    attributes = extract_llm_attributes(
        messages=messages,
        response=response,
        model=model,
    )
    emit(
        TraceEvent(
            trace_id=get_trace_id(),
            thread_id=get_thread_id(),
            name=name,
            event_type=EVENT_LLM_CALL,
            span_id=new_span_id(),
            parent_span_id=get_current_span(),
            duration_ms=duration_ms,
            attributes=attributes,
        )
    )

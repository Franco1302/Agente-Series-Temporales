"""Decoradores de instrumentación no invasiva para nodos LangGraph."""

from __future__ import annotations

import asyncio
import time
from functools import wraps
from typing import Any, Callable, Mapping

from .context import get_thread_id, get_trace_id, span
from .events import (
    EVENT_ERROR,
    EVENT_NODE_ENTER,
    EVENT_NODE_EXIT,
    TraceEvent,
)
from .logger import emit, is_enabled

NodeFn = Callable[..., Any]

# Tope para no volcar cadenas o estructuras grandes en el log
_MAX_SUMMARY_CHARS = 120
_MAX_DICT_KEYS = 8


def _summarize_value(value: Any) -> Any:
    """Resume un valor del estado para el log sin volcar contenidos pesados."""
    if value is None or isinstance(value, bool) or isinstance(value, (int, float)):
        return value
    if isinstance(value, str):
        return value if len(value) <= _MAX_SUMMARY_CHARS else value[: _MAX_SUMMARY_CHARS] + "..."
    if isinstance(value, Mapping):
        keys = list(value.keys())
        return {"_dict_keys": sorted(map(str, keys))[:_MAX_DICT_KEYS], "_n": len(keys)}
    if isinstance(value, (list, tuple, set)):
        return {"_collection_len": len(value), "_type": type(value).__name__}
    return type(value).__name__


def _compute_state_delta(state_before: Any, updates: Any) -> dict[str, Any]:
    """Calcula qué claves del ``AgentState`` modifica el nodo.

    Reglas:
        - Si ``updates`` no es un dict, se reporta solo el tipo devuelto.
        - Para ``messages`` (reducer ``add_messages``) se reporta el
          número de mensajes nuevos como ``messages_added``.
        - Para el resto de campos se comparan los valores antiguos con
          los nuevos y solo se incluyen los que cambian, con un resumen
          compacto (sin volcar el dict/lista entero).
    """
    if not isinstance(updates, Mapping):
        return {"_returned_type": type(updates).__name__}

    delta: dict[str, Any] = {}
    state_get = state_before.get if hasattr(state_before, "get") else (lambda *_: None)

    for key, new_val in updates.items():
        if key == "messages":
            delta["messages_added"] = len(new_val) if isinstance(new_val, list) else "?"
            continue
        try:
            old_val = state_get(key)
        except Exception:
            old_val = None
        try:
            unchanged = old_val == new_val
        except Exception:
            unchanged = False
        if not unchanged:
            delta[key] = _summarize_value(new_val)

    return delta


def _emit_enter(name: str, span_id: str, parent_id: str | None) -> dict[str, Any]:
    """Emite ``node_enter`` y devuelve el dict base reusable para los demás eventos."""
    base = {
        "trace_id": get_trace_id(),
        "thread_id": get_thread_id(),
        "name": name,
        "span_id": span_id,
        "parent_span_id": parent_id,
    }
    emit(
        TraceEvent(
            **base,
            event_type=EVENT_NODE_ENTER,
            attributes={"node": name},
        )
    )
    return base


def _emit_exit_ok(base: dict[str, Any], name: str, t0: float, state: Any, result: Any) -> None:
    """Emite ``node_exit`` con ``state_delta`` en el camino feliz."""
    duration_ms = (time.perf_counter() - t0) * 1000.0
    emit(
        TraceEvent(
            **base,
            event_type=EVENT_NODE_EXIT,
            duration_ms=duration_ms,
            attributes={
                "node": name,
                "status": "ok",
                "state_delta": _compute_state_delta(state, result),
            },
        )
    )


def _emit_exit_error(base: dict[str, Any], name: str, t0: float, exc: BaseException) -> None:
    """Emite ``error`` y ``node_exit`` (con ``status: error``) ante excepción."""
    duration_ms = (time.perf_counter() - t0) * 1000.0
    emit(
        TraceEvent(
            **base,
            event_type=EVENT_ERROR,
            duration_ms=duration_ms,
            attributes={
                "category": "runtime",
                "node": name,
                "exc_type": type(exc).__name__,
                "message": str(exc)[:_MAX_SUMMARY_CHARS],
            },
        )
    )
    emit(
        TraceEvent(
            **base,
            event_type=EVENT_NODE_EXIT,
            duration_ms=duration_ms,
            attributes={"node": name, "status": "error"},
        )
    )


def traced_node(name: str) -> Callable[[NodeFn], NodeFn]:
    """Devuelve un decorador que envuelve un nodo del grafo con observabilidad.

    Parámetros:
        name: Nombre canónico del nodo (debe coincidir con el usado en
            ``builder.add_node(name, ...)`` para que las trazas sean
            agregables por nodo con pandas).

    Comportamiento:
        - Si el subsistema está apagado el wrapper invoca la función
          original sin ningún overhead más allá de un ``if``.
        - Crea un span propio del nodo (``parent_span_id`` = span actual).
        - Emite ``node_enter`` al entrar, ``node_exit`` con
          ``duration_ms`` y ``state_delta`` al salir.
        - Si la función lanza, emite además un evento ``error`` con la
          categoría ``runtime``, y re-propaga la excepción para que el
          grafo siga su flujo habitual de manejo de errores.
        - Detecta automáticamente si ``fn`` es ``async def`` y devuelve
          un wrapper compatible (síncrono o corrutina). Hoy todos los
          nodos son síncronos pero esto evita romper el ``AgentState``
          si en el futuro alguno se refactoriza a ``async``.
    """

    def decorator(fn: NodeFn) -> NodeFn:
        if asyncio.iscoroutinefunction(fn):

            @wraps(fn)
            async def async_wrapper(state: Any, *args: Any, **kwargs: Any) -> Any:
                if not is_enabled():
                    return await fn(state, *args, **kwargs)

                with span(name) as (span_id, parent_id):
                    base = _emit_enter(name, span_id, parent_id)
                    t0 = time.perf_counter()
                    try:
                        result = await fn(state, *args, **kwargs)
                    except Exception as exc:
                        _emit_exit_error(base, name, t0, exc)
                        raise
                    _emit_exit_ok(base, name, t0, state, result)
                    return result

            return async_wrapper

        @wraps(fn)
        def sync_wrapper(state: Any, *args: Any, **kwargs: Any) -> Any:
            if not is_enabled():
                return fn(state, *args, **kwargs)

            with span(name) as (span_id, parent_id):
                base = _emit_enter(name, span_id, parent_id)
                t0 = time.perf_counter()
                try:
                    result = fn(state, *args, **kwargs)
                except Exception as exc:
                    _emit_exit_error(base, name, t0, exc)
                    raise
                _emit_exit_ok(base, name, t0, state, result)
                return result

        return sync_wrapper

    return decorator

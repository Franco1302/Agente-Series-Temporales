"""Propagación de identificadores de traza vía ContextVar.

Se utilizan ContextVars (no variables globales mutables) porque LangGraph
ejecuta nodos y ramificaciones sobre bucles de eventos asíncronos
(`ToolNode.ainvoke`, `graph.astream`). Cada tarea de asyncio recibe una
copia independiente del contexto, evitando que el estado de un span se
mezcle entre corrutinas concurrentes.

Para el anidamiento de spans se guarda el ``Token`` devuelto por
``ContextVar.set`` y se restaura con ``ContextVar.reset``; nunca se
asigna a mano el span padre.
"""

from __future__ import annotations

import uuid
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Iterator, Optional

_trace_id: ContextVar[Optional[str]] = ContextVar("tfg_trace_id", default=None)
_thread_id: ContextVar[Optional[str]] = ContextVar("tfg_thread_id", default=None)
_current_span: ContextVar[Optional[str]] = ContextVar("tfg_current_span", default=None)


def start_turn(thread_id: str) -> str:
    """Inicia un turno conversacional y devuelve el ``trace_id`` recién generado.

    Parámetros:
        thread_id: Identificador estable de la sesión Streamlit.

    Retorno:
        El ``trace_id`` (hex de 32 caracteres) que el resto del turno
        debe compartir. Se fija también en la ContextVar de hilo.
    """
    trace_id = uuid.uuid4().hex
    _trace_id.set(trace_id)
    _thread_id.set(thread_id)
    _current_span.set(None)
    return trace_id


def get_trace_id() -> Optional[str]:
    """Devuelve el ``trace_id`` activo en este contexto, o ``None``."""
    return _trace_id.get()


def get_thread_id() -> Optional[str]:
    """Devuelve el ``thread_id`` activo en este contexto, o ``None``."""
    return _thread_id.get()


def get_current_span() -> Optional[str]:
    """Devuelve el ``span_id`` actualmente activo (padre para spans nuevos)."""
    return _current_span.get()


def new_span_id() -> str:
    """Genera un identificador de span (hex de 16 caracteres)."""
    return uuid.uuid4().hex[:16]


@contextmanager
def span(name: str) -> Iterator[tuple[str, Optional[str]]]:
    """Crea un span anidado y lo activa durante el ``with``.

    Parámetros:
        name: Nombre lógico del span (informativo; no se persiste aquí).

    Yield:
        Una tupla ``(span_id, parent_span_id)`` para que el llamante la
        adjunte al evento que vaya a emitir.

    Notas de implementación:
        Se guarda el ``Token`` devuelto por ``set`` y se libera con
        ``reset`` en el ``finally``. Esto es asyncio-safe: si dos
        corrutinas crean spans en paralelo cada una ve su propio padre.
    """
    parent = _current_span.get()
    span_id = new_span_id()
    token = _current_span.set(span_id)
    try:
        yield span_id, parent
    finally:
        _current_span.reset(token)

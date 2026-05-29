"""Esquema tipado de eventos de observabilidad.

Los nombres de campo siguen la convención OpenTelemetry (``trace_id``,
``span_id``, ``parent_span_id``, ``name``, ``duration_ms``,
``attributes``) para poder exportar a Phoenix/OTel en el futuro sin
reescribir esta capa. El campo ``event_type`` es propio del proyecto y
agrupa los eventos por categoría analítica.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Optional

# Etiquetas oficiales de event_type — el resto del proyecto debe importarlas
# desde aquí para evitar typos.
EVENT_TURN_START = "turn_start"
EVENT_TURN_END = "turn_end"
EVENT_NODE_ENTER = "node_enter"
EVENT_NODE_EXIT = "node_exit"
EVENT_LLM_CALL = "llm_call"
EVENT_TOOL_CALL_START = "tool_call_start"
EVENT_TOOL_CALL_END = "tool_call_end"
EVENT_API_HTTP = "api_http"
EVENT_RAG_RETRIEVAL = "rag_retrieval"
EVENT_PARAMS_INHERITED = "params_inherited"
EVENT_ERROR = "error"


def _utc_now_iso() -> str:
    """Devuelve la marca temporal UTC actual en formato ISO 8601 con ms."""
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


@dataclass
class TraceEvent:
    """Evento individual de la traza.

    Atributos:
        trace_id: Identificador del turno (compartido por todos los
            eventos del mismo turno conversacional).
        thread_id: Identificador de la sesión Streamlit.
        name: Nombre descriptivo libre (p. ej. ``"razonador"``,
            ``"llm.invoke"``, ``"detect_drift"``).
        event_type: Una de las constantes ``EVENT_*`` definidas arriba.
        timestamp: ISO 8601 UTC; se autocompleta al instanciar.
        span_id: Identificador del span al que pertenece el evento.
        parent_span_id: Span padre (si el evento estaba dentro de otro).
        duration_ms: Duración asociada en milisegundos (cuando aplica).
        attributes: Diccionario libre con campos específicos por
            ``event_type``. Debe ser JSON-serializable.
    """

    trace_id: Optional[str]
    thread_id: Optional[str]
    name: str
    event_type: str
    timestamp: str = field(default_factory=_utc_now_iso)
    span_id: Optional[str] = None
    parent_span_id: Optional[str] = None
    duration_ms: Optional[float] = None
    attributes: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Devuelve la representación serializable del evento."""
        return asdict(self)

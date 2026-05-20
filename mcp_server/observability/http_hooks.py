from __future__ import annotations

import time
from contextvars import ContextVar
from typing import Any
import httpx

# Almacena de manera asíncronamente segura la lista de llamadas HTTP de la tool actual
_http_log: ContextVar[list[dict[str, Any]]] = ContextVar("mcp_http_log", default_factory=list)


def init_http_log() -> None:
    """Reinicia el log de peticiones HTTP para la ejecución actual de la tool."""
    _http_log.set([])


def flush_http_log() -> list[dict[str, Any]]:
    """Extrae las peticiones acumuladas y limpia la ContextVar."""
    logs = _http_log.get()
    _http_log.set([])
    return logs


def attach_observability(result_dict: dict[str, Any]) -> dict[str, Any]:
    """Inyecta de forma segura el log HTTP acumulado bajo la clave reservada '_observability'."""
    if not isinstance(result_dict, dict):
        return result_dict

    logs = flush_http_log()
    if logs:
        result_dict["_observability"] = logs
    return result_dict


# ── Hooks de Eventos para httpx.AsyncClient ──────────────────────────────────

async def record_request_start(request: httpx.Request) -> None:
    """Hook que se ejecuta inmediatamente antes de enviar la petición HTTP."""
    # Adjuntamos una marca temporal de inicio en las extensiones internas de la request
    request.extensions["tfg_start_time"] = time.perf_counter()


async def record_response_end(response: httpx.Response) -> None:
    """Hook que se ejecuta tras recibir completamente la respuesta del servidor API."""
    start_time = response.request.extensions.get("tfg_start_time")
    duration_ms = (time.perf_counter() - start_time) * 1000.0 if start_time else None

    # Extraemos el fragmento del endpoint eliminando parámetros de consulta pesados
    url = response.request.url
    endpoint = url.path

    log_entry = {
        "method": response.request.method,
        "endpoint": endpoint,
        "status_code": response.status_code,
        "duration_ms": round(duration_ms, 2) if duration_ms is not None else None,
    }

    # Añadir a la lista de logs de la ContextVar de la tarea actual
    try:
        _http_log.get().append(log_entry)
    except Exception:
        pass
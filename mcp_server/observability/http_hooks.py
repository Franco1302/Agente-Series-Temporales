import time
from contextvars import ContextVar
from typing import Any
import httpx

# ContextVar local al subproceso MCP (usamos default=None para corregir el fallo de default_factory)
_mcp_http_log: ContextVar[list[dict[str, Any]] | None] = ContextVar("mcp_http_log", default=None)


def init_mcp_http_log() -> None:
    """Inicializa o vacía el registro de peticiones HTTP para la ejecución actual."""
    _mcp_http_log.set([])


def flush_mcp_http_log() -> list[dict[str, Any]]:
    """Extrae las peticiones HTTP acumuladas y restablece la ContextVar."""
    try:
        logs = _mcp_http_log.get()
        if logs is None:
            logs = []
    except LookupError:
        logs = []
    
    # Limpiamos la variable de contexto para evitar fugas de memoria
    _mcp_http_log.set(None)
    return logs


def attach_observability(result_dict: dict[str, Any]) -> dict[str, Any]:
    """Inyecta el log de peticiones bajo la clave reservada '_observability'."""
    if not isinstance(result_dict, dict):
        return result_dict

    logs = flush_mcp_http_log()
    if logs:
        result_dict["_observability"] = logs
    return result_dict


# ── Hooks de Eventos Asíncronos para httpx.AsyncClient ───────────────────────

async def record_request_start(request: httpx.Request) -> None:
    """Hook que httpx ejecuta inmediatamente antes de enviar la petición."""
    request.extensions["tfg_start_time"] = time.perf_counter()


async def record_response_end(response: httpx.Response) -> None:
    """Hook que httpx ejecuta al recibir la respuesta completa desde la API REST."""
    start_time = response.request.extensions.get("tfg_start_time")
    duration_ms = (time.perf_counter() - start_time) * 1000.0 if start_time else None

    endpoint = response.request.url.path

    log_entry = {
        "method": response.request.method,
        "endpoint": endpoint,
        "status_code": response.status_code,
        "duration_ms": round(duration_ms, 2) if duration_ms is not None else None,
    }

    try:
        current_log = _mcp_http_log.get()
        # Si no se ha inicializado para esta tarea, lo creamos dinámicamente
        if not isinstance(current_log, list):
            current_log = []
            _mcp_http_log.set(current_log)
            
        current_log.append(log_entry)
    except Exception:
        pass
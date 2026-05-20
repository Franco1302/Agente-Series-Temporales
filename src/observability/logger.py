"""Logger singleton del subsistema de observabilidad.

Escribe una línea JSON por evento en ``data/logs/agent.jsonl`` mediante
un ``RotatingFileHandler``. La función ``emit`` está blindada: cualquier
excepción dentro del subsistema queda contenida y JAMÁS interrumpe el
turno conversacional (criterio de aceptación del Bloque 1).
"""

from __future__ import annotations

import json
import logging
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path
from threading import Lock
from typing import Optional

from .events import TraceEvent

_LOGGER_NAME = "tfg.observability"
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_LOG_DIR = _PROJECT_ROOT / "data" / "logs"
_LOG_FILE = _LOG_DIR / "agent.jsonl"
_MAX_BYTES = 10 * 1024 * 1024  # 10 MB
_BACKUP_COUNT = 5

_state_lock = Lock()
_initialized: bool = False
_logger: Optional[logging.Logger] = None
_enabled: bool = False


class _JsonLineFormatter(logging.Formatter):
    """Serializa cada ``LogRecord`` como una única línea JSON.

    El payload se transporta en ``record.event_payload`` (un dict listo
    para serializar). Si no está, se vuelca el mensaje plano para
    compatibilidad con cualquier log accidental que llegue al logger.
    """

    def format(self, record: logging.LogRecord) -> str:
        payload = getattr(record, "event_payload", None)
        if payload is None:
            payload = {"message": record.getMessage(), "level": record.levelname}
        return json.dumps(payload, ensure_ascii=False, default=str)


def _build_logger(level_name: str, *, also_to_stderr: bool) -> logging.Logger:
    """Construye el logger con rotating file handler y formato JSON."""
    logger = logging.getLogger(_LOGGER_NAME)
    logger.setLevel(getattr(logging, level_name.upper(), logging.INFO))
    logger.propagate = False
    # Idempotencia: limpiar handlers preexistentes si se reconfigura
    for handler in list(logger.handlers):
        logger.removeHandler(handler)

    _LOG_DIR.mkdir(parents=True, exist_ok=True)

    file_handler = RotatingFileHandler(
        _LOG_FILE,
        maxBytes=_MAX_BYTES,
        backupCount=_BACKUP_COUNT,
        encoding="utf-8",
    )
    file_handler.setFormatter(_JsonLineFormatter())
    logger.addHandler(file_handler)

    if also_to_stderr:
        stderr_handler = logging.StreamHandler(sys.stderr)
        stderr_handler.setFormatter(_JsonLineFormatter())
        logger.addHandler(stderr_handler)

    return logger


def configure(*, enabled: bool, level: str = "INFO") -> None:
    """Inicializa explícitamente el subsistema (idempotente y thread-safe).

    Parámetros:
        enabled: Si es ``False``, ``emit`` se convierte en no-op.
        level: Nivel mínimo (``"DEBUG"``, ``"INFO"``, …). Con
            ``"DEBUG"`` se duplica la salida a ``stderr``.
    """
    global _initialized, _logger, _enabled
    with _state_lock:
        _enabled = bool(enabled)
        _initialized = True
        if _enabled:
            _logger = _build_logger(level, also_to_stderr=(level.upper() == "DEBUG"))
        else:
            _logger = None


def _ensure_initialized() -> None:
    """Si nadie llamó a ``configure``, lee la configuración del entorno una vez."""
    global _initialized
    if _initialized:
        return
    try:
        # Import perezoso para evitar dependencia circular en tiempo de import
        from src.config.llm_config import load_observability_settings

        settings = load_observability_settings()
        configure(enabled=settings.enabled, level=settings.log_level)
    except Exception as e:
        import traceback
        traceback.print_exc()
        # No conseguimos leer la config: marcamos como intentado y nos quedamos
        # apagados. No hay que reintentar en cada emit().
        with _state_lock:
            _initialized = True
            _logger = None


def emit(event: TraceEvent) -> None:
    """Escribe un evento como línea JSON.

    Esta función NUNCA propaga excepciones: si algo falla dentro del
    subsistema (disco lleno, permisos, JSON no serializable…) se ignora
    silenciosamente para no afectar la conversación.
    """
    try:
        _ensure_initialized()
        if not _enabled or _logger is None:
            return
        _logger.info("event", extra={"event_payload": event.to_dict()})
    except Exception:
        return


def is_enabled() -> bool:
    """Indica si el subsistema está activo (útil para `if is_enabled(): ...`)."""
    _ensure_initialized()
    return _enabled


def log_file_path() -> Path:
    """Ruta absoluta del fichero JSONL principal."""
    return _LOG_FILE.resolve()

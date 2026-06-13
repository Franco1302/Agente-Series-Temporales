"""Settings del servidor MCP cargados desde variables de entorno."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ServerSettings:
    api_url: str
    workspace_dir: Path
    request_timeout: float
    connect_timeout: float
    log_level: str


def load_settings() -> ServerSettings:
    api_url = os.getenv("DRIFT_API_URL", "http://localhost:8017").rstrip("/")
    workspace = Path(os.getenv("MCP_WORKSPACE_DIR", "data/temp_uploads")).resolve()
    workspace.mkdir(parents=True, exist_ok=True)
    return ServerSettings(
        api_url=api_url,
        workspace_dir=workspace,
        request_timeout=float(os.getenv("DRIFT_API_TIMEOUT", "600")),
        connect_timeout=float(os.getenv("DRIFT_API_CONNECT_TIMEOUT", "5")),
        log_level=os.getenv("MCP_LOG_LEVEL", "INFO"),
    )

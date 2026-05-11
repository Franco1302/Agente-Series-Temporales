"""Carga de tools MCP como BaseTool de LangChain mediante stdio.

Se invoca una vez al construir el grafo. La conexión persiste mientras el
proceso del agente esté vivo (mismo subproceso = misma sesión MCP).
"""

from __future__ import annotations

import asyncio
import os
import sys
from functools import lru_cache
from pathlib import Path

from langchain_core.tools import BaseTool
from langchain_mcp_adapters.client import MultiServerMCPClient

PROJECT_ROOT = Path(__file__).resolve().parents[3]


@lru_cache(maxsize=1)
def _get_mcp_client() -> MultiServerMCPClient:
    """Inicializa el cliente MCP que arranca el servidor como subproceso stdio."""
    return MultiServerMCPClient(
        {
            "drift-mcp": {
                "command": sys.executable,
                "args": ["-m", "mcp_server.server"],
                "transport": "stdio",
                "cwd": str(PROJECT_ROOT),
                "env": {
                    **os.environ,
                    "PYTHONPATH": str(PROJECT_ROOT),
                },
            }
        }
    )


def load_mcp_tools_sync() -> list[BaseTool]:
    """Wrapper síncrono usable desde el grafo (build_agent_graph se ejecuta en main thread)."""
    client = _get_mcp_client()
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            # En contexto async (raro en build_agent_graph) lanzar un loop nuevo
            return asyncio.run(client.get_tools())
    except RuntimeError:
        pass
    return asyncio.run(client.get_tools())

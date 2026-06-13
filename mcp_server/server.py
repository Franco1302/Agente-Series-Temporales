"""Punto de entrada del servidor MCP."""

from __future__ import annotations

import logging
import sys

from mcp_server.config import load_settings
from mcp_server.instance import mcp

settings = load_settings()
logging.basicConfig(
    level=settings.log_level,
    stream=sys.stderr,
    format="[mcp_server] %(levelname)s: %(message)s",
)

# Registro de herramientas (cada módulo decora con @mcp.tool sobre la instancia importada).
from mcp_server.tools import synthetic, drift, augment, exogenous, forecast  # noqa: E402, F401


def main() -> None:
    """Lanza el servidor en modo stdio."""
    logging.info("Iniciando servidor MCP. API URL: %s", settings.api_url)
    logging.info("Workspace: %s", settings.workspace_dir)
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()

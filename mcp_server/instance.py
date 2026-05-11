"""Instancia compartida de FastMCP.

Vive en su propio módulo para evitar duplicación de la instancia cuando
`mcp_server.server` se ejecuta como `python -m mcp_server.server`
(en ese caso `sys.modules['__main__']` y `sys.modules['mcp_server.server']`
son módulos distintos).
"""

from __future__ import annotations

from mcp.server.fastmcp import FastMCP

mcp = FastMCP("drift-mcp-server")

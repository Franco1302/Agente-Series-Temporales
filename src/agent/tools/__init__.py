"""Herramientas disponibles para el agente LangGraph.

Las 8 tools de cálculo se obtienen del servidor MCP (subproceso stdio).
La tool RAG `consultar_teoria` se mantiene como BaseTool local.

Los mocks (`mock_drift.py`, `mock_synthetic.py`, `mock_augment.py`) se
conservan en el paquete por si hace falta volver atrás sin la API, pero
no están incluidos en AGENT_TOOLS.
"""

from src.agent.tools.mcp_loader import load_mcp_tools_sync
from src.tools.rag_tool import consultar_teoria

_MCP_TOOLS = load_mcp_tools_sync()

AGENT_TOOLS = [*_MCP_TOOLS, consultar_teoria]

__all__ = ["AGENT_TOOLS"]

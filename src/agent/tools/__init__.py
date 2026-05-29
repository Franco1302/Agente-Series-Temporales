"""Herramientas disponibles para el agente LangGraph.

Las 8 tools de cálculo se obtienen del servidor MCP (subproceso stdio).
La tool RAG `consultar_teoria` se mantiene como BaseTool local.
"""

from src.agent.tools.mcp_loader import load_mcp_tools_sync
from src.tools.rag_tool import consultar_teoria

_MCP_TOOLS = load_mcp_tools_sync()

AGENT_TOOLS = [*_MCP_TOOLS, consultar_teoria]

__all__ = ["AGENT_TOOLS"]

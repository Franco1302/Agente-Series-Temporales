"""Nodo de ejecución de herramientas: ejecuta las tool calls del LLM."""

from __future__ import annotations

import asyncio

from langchain_core.messages import ToolMessage
from langgraph.prebuilt import ToolNode

from src.agent.state import AgentState
from src.agent.tools import AGENT_TOOLS

# consultar_teoria se maneja en recuperar_contexto_node, no aquí.
# ToolNode despacha automáticamente el resto de herramientas por nombre.
_tool_node = ToolNode(AGENT_TOOLS)


def _run_tools_sync(state: AgentState) -> dict:
    """Ejecuta ToolNode aceptando tools async-only (como las MCP via stdio).

    Las tools MCP cargadas por langchain_mcp_adapters son `StructuredTool`
    sin implementación sync, así que ToolNode.invoke() falla con
    `NotImplementedError: StructuredTool does not support sync invocation`.
    Hay que pasar por `ainvoke` y ejecutarlo dentro de un event loop.

    Si ya estamos dentro de un loop activo (caso `graph.astream`), lanzamos
    la corrutina en ese loop; si no, abrimos uno nuevo con `asyncio.run`.
    """
    coro = _tool_node.ainvoke(state)
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)
    # Hay un loop corriendo; lo programamos sin bloquearlo (escenario raro
    # en el flujo sync del agente, pero contemplado).
    return loop.run_until_complete(coro)


def tool_execution_node(state: AgentState) -> dict:
    """Ejecuta las tool calls pendientes en el último AIMessage del estado.

    Delega la ejecución real en ToolNode de LangGraph (vía ainvoke, ver
    `_run_tools_sync`). En caso de excepción, captura el error y lo
    registra en `error_info` para que route_after_tool desvíe el flujo a
    gestionar_error.
    """
    try:
        result = _run_tools_sync(state)
        return {
            "messages": result.get("messages", []),
            "error_info": None,
        }
    except Exception as exc:
        error_msg = ToolMessage(
            content=f"Error al ejecutar herramienta: {exc}",
            tool_call_id="error",
            name="error",
        )
        return {
            "messages": [error_msg],
            "error_info": str(exc),
        }

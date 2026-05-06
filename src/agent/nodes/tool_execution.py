"""Nodo de ejecución de herramientas: ejecuta las tool calls del LLM."""

from __future__ import annotations

from langchain_core.messages import ToolMessage
from langgraph.prebuilt import ToolNode

from src.agent.state import AgentState
from src.agent.tools import AGENT_TOOLS

# consultar_teoria se maneja en recuperar_contexto_node, no aquí.
# ToolNode despacha automáticamente el resto de herramientas por nombre.
_tool_node = ToolNode(AGENT_TOOLS)


def tool_execution_node(state: AgentState) -> dict:
    """Ejecuta las tool calls pendientes en el último AIMessage del estado.

    Delega la ejecución real en ToolNode de LangGraph. En caso de excepción,
    captura el error y lo registra en `error_info` para que route_after_tool
    desvíe el flujo a gestionar_error.
    """
    try:
        result = _tool_node.invoke(state)
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

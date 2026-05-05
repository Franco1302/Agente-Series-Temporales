"""Nodo de ejecución de herramientas: ejecuta las tool calls del LLM."""

from __future__ import annotations

from langchain_core.messages import AIMessage, ToolMessage
from langgraph.prebuilt import ToolNode

from src.agent.state import AgentState
from src.agent.tools import AGENT_TOOLS

# ToolNode de LangGraph gestiona automáticamente la ejecución de tool calls
# presentes en el último AIMessage, produciendo ToolMessages con los resultados.
_tool_node = ToolNode(AGENT_TOOLS)


def tool_execution_node(state: AgentState) -> dict:
    """Ejecuta las tool calls pendientes en el último AIMessage del estado.

    Delega la ejecución real en ToolNode de LangGraph, que busca la herramienta
    por nombre, llama a su función con los argumentos recibidos y construye
    ToolMessages con los resultados.

    Además de añadir los ToolMessages al historial, actualiza `tool_results`
    con un dict {nombre_herramienta: resultado} para que otros nodos puedan
    acceder al último resultado sin parsear el historial completo.
    """
    result = _tool_node.invoke(state)

    tool_results: dict = dict(state.get("tool_results") or {})

    for msg in result.get("messages", []):
        if isinstance(msg, ToolMessage):
            tool_results[msg.name] = msg.content

    return {
        "messages": result.get("messages", []),
        "tool_results": tool_results,
    }

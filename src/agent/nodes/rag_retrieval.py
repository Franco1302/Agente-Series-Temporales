"""Nodo de recuperación de contexto teórico mediante el sistema RAG."""

from __future__ import annotations

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from src.agent.state import AgentState
from src.tools.rag_tool import consultar_teoria


def _extract_query(state: AgentState) -> str:
    """Devuelve la query refinada de la tool call si existe; si no, el último HumanMessage."""
    messages = state.get("messages", [])

    # 1) Preferir la query refinada que el LLM puso en la tool call
    for msg in reversed(messages):
        if not isinstance(msg, AIMessage):
            continue
        tool_calls = getattr(msg, "tool_calls", None) or []
        if not tool_calls:
            continue
        call = tool_calls[0]
        name = call.get("name") if isinstance(call, dict) else getattr(call, "name", "")
        if name != "consultar_teoria":
            break
        args = call.get("args") if isinstance(call, dict) else getattr(call, "args", {})
        query = (args or {}).get("query")
        if isinstance(query, str) and query.strip():
            return query.strip()
        break

    # 2) Fallback: último HumanMessage
    for msg in reversed(messages):
        if isinstance(msg, HumanMessage):
            return str(msg.content).strip()

    return ""


def recuperar_contexto_node(state: AgentState) -> dict:
    """Ejecuta una búsqueda RAG y devuelve el resultado como ToolMessage.

    Toma la consulta refinada que el LLM emitió en la tool call de
    `consultar_teoria`, llama a la herramienta y emite el ToolMessage
    de cierre asociado al `tool_call_id`. El razonador, en su siguiente
    pasada, leerá ese ToolMessage como respuesta natural a su tool call
    y sintetizará la respuesta final.

    Sin el ToolMessage, ChatOllama lanza el error
    "tool call without a corresponding tool message" cuando el ciclo
    vuelve al razonador.
    """
    messages = state.get("messages", [])
    query_text = _extract_query(state)

    # Localizar el id de la tool call para el ToolMessage de cierre
    tool_call_id: str | None = None
    for msg in reversed(messages):
        if not isinstance(msg, AIMessage):
            continue
        tool_calls = getattr(msg, "tool_calls", None) or []
        if tool_calls:
            call = tool_calls[0]
            tool_call_id = (
                call.get("id") if isinstance(call, dict) else getattr(call, "id", None)
            )
        break

    if not query_text or not tool_call_id:
        return {}

    result = consultar_teoria.invoke({"query": query_text})

    if isinstance(result, str) and result.startswith("Error:"):
        return {"error_info": result}

    return {
        "messages": [
            ToolMessage(
                content=str(result),
                name="consultar_teoria",
                tool_call_id=tool_call_id,
            )
        ]
    }

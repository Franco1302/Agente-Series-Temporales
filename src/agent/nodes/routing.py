"""Funciones de enrutamiento condicional entre nodos del grafo."""

from __future__ import annotations

from langchain_core.messages import AIMessage

from src.agent.state import AgentState


def route_after_reasoning(state: AgentState) -> str:
    """Decide el siguiente nodo tras reasoning_node.

    Reglas:
    - Si el último mensaje es un AIMessage con tool_calls → param_validation_node
      (primero se validan los parámetros antes de ejecutar).
    - En cualquier otro caso → END (el LLM respondió directamente).
    """
    messages = state.get("messages", [])
    if not messages:
        return "END"

    last = messages[-1]
    if not isinstance(last, AIMessage):
        return "END"

    tool_calls = getattr(last, "tool_calls", None) or []
    if tool_calls:
        return "param_validation_node"

    return "END"


def route_after_validation(state: AgentState) -> str:
    """Decide el siguiente nodo tras param_validation_node.

    Reglas:
    - Si hay parámetros pendientes (`pending_params` no vacío) → END
      (el nodo ya generó un mensaje pidiendo los datos al usuario).
    - Si no hay parámetros pendientes → tool_execution_node
      (todos los argumentos están completos, se puede ejecutar).
    """
    pending = state.get("pending_params") or []
    if pending:
        return "END"

    return "tool_execution_node"

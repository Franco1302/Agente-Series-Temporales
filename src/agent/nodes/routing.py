"""Funciones de enrutamiento condicional para la topología cíclica del grafo."""

from __future__ import annotations

from langchain_core.messages import AIMessage

from src.agent.state import AgentState

_PARAM_ERROR_KEYWORDS = ("parámetro", "argumento", "falta", "missing", "required")


def route_after_razonador(state: AgentState) -> str:
    """Decide el siguiente nodo tras razonador (4 destinos posibles).

    - Si el LLM emitió tool_call para consultar_teoria → recuperar_contexto
    - Si el LLM emitió tool_call pero faltan parámetros  → solicitar_parametros
    - Si el LLM emitió tool_call completa               → ejecutar_herramienta
    - Si el LLM respondió directamente (sin tool_calls)  → generar_respuesta
    """
    messages = state.get("messages", [])
    if not messages:
        return "generar_respuesta"

    last = messages[-1]
    if not isinstance(last, AIMessage):
        return "generar_respuesta"

    tool_calls = getattr(last, "tool_calls", None) or []
    if not tool_calls:
        return "generar_respuesta"

    call = tool_calls[0]
    tool_name = call.get("name", "") if isinstance(call, dict) else getattr(call, "name", "")

    if tool_name == "consultar_teoria":
        return "recuperar_contexto"

    # razonador_node ya evaluó si faltan parámetros y lo registró en pending_tool
    if state.get("pending_tool"):
        return "solicitar_parametros"

    return "ejecutar_herramienta"


def route_after_tool(state: AgentState) -> str:
    """Decide el siguiente nodo tras ejecutar_herramienta.

    - Si tool_execution_node capturó un error → gestionar_error
    - Si todo fue bien                         → razonador (ciclo ReAct)
    """
    if state.get("error_info"):
        return "gestionar_error"
    return "razonador"


def route_after_error(state: AgentState) -> str:
    """Decide el siguiente nodo tras gestionar_error.

    - Si se alcanzó el límite de errores → generar_respuesta (abortar)
    - Si el error parece de parámetros   → solicitar_parametros
    - Por defecto                         → solicitar_parametros (más seguro que abortar)
    """
    error_count = state.get("error_count", 0)
    if error_count >= 3:
        return "generar_respuesta"

    error_info = (state.get("error_info") or "").lower()
    if any(kw in error_info for kw in _PARAM_ERROR_KEYWORDS):
        return "solicitar_parametros"

    return "solicitar_parametros"

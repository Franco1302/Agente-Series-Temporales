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
    - Si el LLM respondió directamente (sin tool_calls)  → fin (END)

    El razonador ya produjo la respuesta final; volver a invocar al LLM en
    `generar_respuesta` doblaría el tiempo y suele devolver content vacío
    porque el modelo ve que ya hay un AIMessage con la respuesta.
    `generar_respuesta` queda reservado para el camino de error.
    """
    messages = state.get("messages", [])
    if not messages:
        return "fin"

    last = messages[-1]
    if not isinstance(last, AIMessage):
        return "fin"

    tool_calls = getattr(last, "tool_calls", None) or []
    if not tool_calls:
        return "fin"

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

    - Si se alcanzó el límite de errores → generar_respuesta (abortar con disculpa)
    - Si el error parece de parámetros Y hay un pending_tool con datos que
      recoger → solicitar_parametros (pedir datos al usuario)
    - Por defecto (errores no recuperables: timeouts, conexión, runtime, o
      errores de parámetros sin pending_tool) → generar_respuesta
    """
    error_count = state.get("error_count", 0)
    if error_count >= 3:
        return "generar_respuesta"

    # Solo tiene sentido pedir parámetros si hay una tool pendiente que los
    # espera. Un error de validación de una tool ya ejecutada (p. ej. aridad de
    # trend_params) llega sin pending_tool: enrutarlo a solicitar_parametros
    # produciría un fin silencioso, así que lo reportamos vía generar_respuesta.
    error_info = (state.get("error_info") or "").lower()
    if state.get("pending_tool") and any(kw in error_info for kw in _PARAM_ERROR_KEYWORDS):
        return "solicitar_parametros"

    return "generar_respuesta"

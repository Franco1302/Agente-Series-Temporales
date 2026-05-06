"""Nodo de gestión de errores: clasifica fallos y decide si recuperar o abortar."""

from __future__ import annotations

from langchain_core.messages import AIMessage

from src.agent.state import AgentState

_MAX_ERRORS = 3

_CRITICAL_KEYWORDS = ("timeout", "connection", "crítico", "fatal", "unavailable")
_PARAM_KEYWORDS = ("parámetro", "argumento", "falta", "missing", "required")


def gestionar_error_node(state: AgentState) -> dict:
    """Intercepta un fallo de tool_execution_node, clasifica el error y prepara el estado.

    Incrementa `error_count`. Si se supera el límite o el error es crítico, añade
    un AIMessage explicativo para que generar_respuesta lo presente al usuario.
    En errores de parámetros, no añade mensaje (lo hará solicitar_parametros).

    La decisión de enrutamiento real la toma route_after_error en routing.py,
    que lee `error_count` y `error_info` del estado devuelto por este nodo.
    """
    error_count = state.get("error_count", 0) + 1
    error_info = state.get("error_info") or ""

    is_critical = (
        error_count >= _MAX_ERRORS
        or any(kw in error_info.lower() for kw in _CRITICAL_KEYWORDS)
    )

    updates: dict = {"error_count": error_count}

    if is_critical:
        updates["messages"] = [
            AIMessage(
                content=(
                    f"Ha ocurrido un error que impide continuar: {error_info}\n\n"
                    "Por favor, comprueba que Ollama está en ejecución y vuelve a intentarlo."
                )
            )
        ]

    return updates

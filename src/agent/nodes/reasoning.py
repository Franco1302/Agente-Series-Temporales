"""Nodo de razonamiento: invoca el LLM con las herramientas enlazadas."""

from __future__ import annotations

from langchain_core.messages import SystemMessage

from src.agent.prompts import build_system_prompt
from src.agent.state import AgentState
from src.agent.tools import AGENT_TOOLS
from src.config.llm_config import get_llm_with_tools

_MAX_ITERATIONS = 5


def reasoning_node(state: AgentState) -> dict:
    """Invoca el LLM con el estado actual y devuelve el mensaje de respuesta.

    El LLM decide en este nodo si responder directamente, pedir parámetros
    al usuario o emitir una tool call. La decisión queda codificada en el
    campo `tool_calls` del AIMessage resultante.

    Si se supera el límite de iteraciones, devuelve un mensaje de parada
    sin invocar el LLM para evitar bucles infinitos.
    """
    iteration = state.get("iteration_count", 0)

    if iteration >= _MAX_ITERATIONS:
        from langchain_core.messages import AIMessage
        return {
            "messages": [
                AIMessage(
                    content=(
                        "He alcanzado el límite de iteraciones en esta consulta. "
                        "Por favor, reformula tu petición o simplifica la tarea."
                    )
                )
            ],
            "iteration_count": iteration,
        }

    uploaded_path = state.get("uploaded_file_path")
    if uploaded_path:
        from pathlib import Path
        p = Path(uploaded_path)
        file_info = {
            "file_name": p.name,
            "file_path": uploaded_path,
            "file_size_kb": p.stat().st_size / 1024 if p.exists() else 0.0,
        }
        system_prompt = build_system_prompt(has_uploaded_file=True, uploaded_file_info=file_info)
    else:
        system_prompt = build_system_prompt(has_uploaded_file=False)

    messages = list(state["messages"])

    # Inyecta el system prompt solo si el primer mensaje no es ya un SystemMessage
    if not messages or not isinstance(messages[0], SystemMessage):
        messages = [SystemMessage(content=system_prompt)] + messages

    llm = get_llm_with_tools(AGENT_TOOLS)
    response = llm.invoke(messages)

    return {
        "messages": [response],
        "iteration_count": iteration + 1,
    }

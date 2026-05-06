"""Nodo terminal de síntesis: genera la respuesta final para el usuario."""

from __future__ import annotations

from langchain_core.messages import SystemMessage

from src.agent.prompts import build_system_prompt
from src.agent.state import AgentState
from src.config.llm_config import get_llm_with_tools


def generar_respuesta_node(state: AgentState) -> dict:
    """Sintetiza la respuesta final sin emitir nuevas tool calls.

    Se activa cuando razonador responde directamente, cuando gestionar_error
    detecta un fallo crítico, o cuando el ciclo RAG ha terminado y el razonador
    necesita entregar el resultado al usuario.

    Llama al LLM sin herramientas enlazadas para garantizar que la respuesta
    sea texto plano y no desencadene otro ciclo de ejecución.
    Resetea `error_count` y `error_info` en la salida limpia.
    """
    csv_path = state.get("csv_path")
    csv_metadata = state.get("csv_metadata")
    rag_context = state.get("rag_context")

    system_prompt = build_system_prompt(csv_path=csv_path, csv_metadata=csv_metadata)

    messages = list(state["messages"])

    # Inyectar system prompt si aún no está como primer mensaje
    if not messages or not isinstance(messages[0], SystemMessage):
        messages = [SystemMessage(content=system_prompt)] + messages

    # Inyectar contexto RAG como segundo SystemMessage si está disponible
    if rag_context:
        rag_msg = SystemMessage(content=f"CONTEXTO TEÓRICO RECUPERADO:\n{rag_context}")
        messages = [messages[0], rag_msg] + messages[1:]

    llm = get_llm_with_tools([])
    response = llm.invoke(messages)

    return {
        "messages": [response],
        "error_count": 0,
        "error_info": None,
    }

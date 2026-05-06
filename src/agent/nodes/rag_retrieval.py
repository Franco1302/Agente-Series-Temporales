"""Nodo de recuperación de contexto teórico mediante el sistema RAG."""

from __future__ import annotations

from langchain_core.messages import HumanMessage

from src.agent.state import AgentState
from src.tools.rag_tool import consultar_teoria


def recuperar_contexto_node(state: AgentState) -> dict:
    """Ejecuta una búsqueda RAG y almacena el resultado en `rag_context`.

    Extrae la consulta del último HumanMessage del historial, llama a
    `consultar_teoria` y devuelve el contexto recuperado para que
    razonador lo inyecte como SystemMessage en el siguiente ciclo.

    Este nodo NO invoca el LLM directamente: la síntesis ya ocurre
    dentro de consultar_teoria. Su única responsabilidad es
    popular `rag_context`.
    """
    messages = state.get("messages", [])

    # Buscar el último HumanMessage del historial
    query_text = ""
    for msg in reversed(messages):
        if isinstance(msg, HumanMessage):
            query_text = str(msg.content)
            break

    if not query_text:
        return {"rag_context": None}

    result = consultar_teoria.invoke({"query": query_text})

    if isinstance(result, str) and result.startswith("Error:"):
        return {"error_info": result, "rag_context": None}

    return {"rag_context": str(result)}

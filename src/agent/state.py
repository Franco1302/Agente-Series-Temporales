"""Definición del estado compartido del agente LangGraph."""

from __future__ import annotations

from typing import Annotated, Any

from langgraph.graph.message import add_messages
from typing_extensions import TypedDict


class AgentState(TypedDict):
    """Estado completo que fluye a través de todos los nodos del grafo.

    LangGraph pasa este dict entre nodos; cada nodo puede leer y devolver
    una actualización parcial. El campo `messages` usa `add_messages` como
    reducer, lo que garantiza que los mensajes se acumulen en lugar de
    sobreescribirse.
    """

    # Historial completo de mensajes (HumanMessage, AIMessage, ToolMessage…)
    # add_messages acumula en lugar de reemplazar cuando se hace una actualización parcial.
    messages: Annotated[list, add_messages]

    # Últimos resultados de herramientas ejecutadas: {nombre_herramienta: resultado}
    tool_results: dict[str, Any]

    # Reservado para contexto RAG inyectado antes de razonar (vacío en esta iteración)
    rag_context: str

    # Parámetros detectados como faltantes en la última tool call
    pending_params: list[str]

    # Ruta al CSV cargado por el usuario desde el sidebar (None si no hay fichero)
    uploaded_file_path: str | None

    # Contador de iteraciones del ciclo ReAct para evitar bucles infinitos
    iteration_count: int

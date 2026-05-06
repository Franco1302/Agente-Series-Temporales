"""Construcción y compilación del grafo LangGraph del agente."""

from __future__ import annotations

from functools import lru_cache

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph

from src.agent.nodes import (
    generar_respuesta_node,
    gestionar_error_node,
    razonador_node,
    recuperar_contexto_node,
    route_after_error,
    route_after_razonador,
    route_after_tool,
    solicitar_parametros_node,
    tool_execution_node,
)
from src.agent.state import AgentState


@lru_cache(maxsize=1)
def build_agent_graph():
    """Construye y compila el grafo del agente con persistencia en memoria.

    Implementa la arquitectura cíclica ReAct con 6 nodos:
        razonador → ejecutar_herramienta → razonador  (ciclo ReAct)
        razonador → recuperar_contexto → razonador    (ciclo RAG)
        razonador → solicitar_parametros → END
        razonador → generar_respuesta → END
        ejecutar_herramienta → gestionar_error → solicitar_parametros | generar_respuesta

    Se cachea con lru_cache para evitar reconstruirlo en cada petición de Streamlit.

    Returns:
        Grafo compilado listo para invocar con .invoke() o .stream().
    """
    builder = StateGraph(AgentState)

    # ── Registro de nodos ──────────────────────────────────────────────────────
    builder.add_node("razonador", razonador_node)
    builder.add_node("ejecutar_herramienta", tool_execution_node)
    builder.add_node("solicitar_parametros", solicitar_parametros_node)
    builder.add_node("gestionar_error", gestionar_error_node)
    builder.add_node("recuperar_contexto", recuperar_contexto_node)
    builder.add_node("generar_respuesta", generar_respuesta_node)

    # ── Arista de entrada ──────────────────────────────────────────────────────
    builder.add_edge(START, "razonador")

    # ── Enrutamiento desde razonador (4 destinos) ─────────────────────────────
    builder.add_conditional_edges(
        "razonador",
        route_after_razonador,
        {
            "ejecutar_herramienta": "ejecutar_herramienta",
            "solicitar_parametros": "solicitar_parametros",
            "recuperar_contexto": "recuperar_contexto",
            "generar_respuesta": "generar_respuesta",
        },
    )

    # ── Ciclo ReAct: éxito → razonador, error → gestionar_error ───────────────
    builder.add_conditional_edges(
        "ejecutar_herramienta",
        route_after_tool,
        {
            "razonador": "razonador",
            "gestionar_error": "gestionar_error",
        },
    )

    # ── Recuperación de error: reintento o abortar ────────────────────────────
    builder.add_conditional_edges(
        "gestionar_error",
        route_after_error,
        {
            "solicitar_parametros": "solicitar_parametros",
            "generar_respuesta": "generar_respuesta",
        },
    )

    # ── Ciclo RAG: siempre vuelve al razonador con el contexto poblado ─────────
    builder.add_edge("recuperar_contexto", "razonador")

    # ── Nodos terminales ───────────────────────────────────────────────────────
    builder.add_edge("solicitar_parametros", END)
    builder.add_edge("generar_respuesta", END)

    # ── Compilación con persistencia por hilo de conversación ─────────────────
    memory = MemorySaver()
    return builder.compile(checkpointer=memory)

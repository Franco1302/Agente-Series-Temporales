"""Construcción y compilación del grafo LangGraph del agente."""

from __future__ import annotations

from functools import lru_cache

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph

from src.agent.nodes import (
    param_validation_node,
    reasoning_node,
    route_after_reasoning,
    route_after_validation,
    tool_execution_node,
)
from src.agent.state import AgentState


@lru_cache(maxsize=1)
def build_agent_graph():
    """Construye y compila el grafo del agente con persistencia en memoria.

    El grafo implementa el ciclo ReAct:
        START → reasoning → [validación de params] → ejecución → reasoning → …

    Se cachea con lru_cache para evitar reconstruirlo en cada petición de Streamlit.

    Returns:
        Grafo compilado listo para invocar con .invoke() o .stream().
    """
    builder = StateGraph(AgentState)

    # ── Registro de nodos ──────────────────────────────────────────────────────
    builder.add_node("reasoning_node", reasoning_node)
    builder.add_node("param_validation_node", param_validation_node)
    builder.add_node("tool_execution_node", tool_execution_node)

    # ── Arista de entrada ──────────────────────────────────────────────────────
    builder.add_edge(START, "reasoning_node")

    # ── Enrutamiento tras razonar ──────────────────────────────────────────────
    # Si el LLM emitió tool_calls → param_validation_node
    # Si respondió directamente  → END
    builder.add_conditional_edges(
        "reasoning_node",
        route_after_reasoning,
        {
            "param_validation_node": "param_validation_node",
            "END": END,
        },
    )

    # ── Enrutamiento tras validar parámetros ───────────────────────────────────
    # Si faltan parámetros → END (ya se envió mensaje de solicitud al usuario)
    # Si todo completo     → tool_execution_node
    builder.add_conditional_edges(
        "param_validation_node",
        route_after_validation,
        {
            "tool_execution_node": "tool_execution_node",
            "END": END,
        },
    )

    # ── Ciclo ReAct: tras ejecutar la tool, volver a razonar ──────────────────
    builder.add_edge("tool_execution_node", "reasoning_node")

    # ── Compilación con persistencia por hilo de conversación ─────────────────
    memory = MemorySaver()
    return builder.compile(checkpointer=memory)

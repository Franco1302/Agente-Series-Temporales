"""Tests de las directivas que fuerzan la tool call (Parte 2).

Tras introducir el nodo router, la DECISIÓN de forzar vive en ``razonador_node``
guiada por ``state['intent']`` (ver ``tests/test_router.py``); las DIRECTIVAS que
empujan al modelo cuantizado a emitir el JSON en vez de responder en prosa son
funciones puras (sin LLM) verificadas aquí.
"""

from __future__ import annotations

from langchain_core.messages import SystemMessage

from src.agent.nodes.reasoning import (
    _build_action_directive,
    _build_theory_directive,
)


def test_action_directive_pushes_tool_call() -> None:
    msg = _build_action_directive()
    assert isinstance(msg, SystemMessage)
    body = msg.content.lower()
    # Debe empujar a emitir el JSON y prohibir preguntar en texto.
    assert "json" in body or "tool call" in body
    assert "no preguntes" in body


def test_theory_directive_names_consultar_teoria() -> None:
    msg = _build_theory_directive()
    assert isinstance(msg, SystemMessage)
    assert "consultar_teoria" in msg.content
    assert "query" in msg.content.lower()

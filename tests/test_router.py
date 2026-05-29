"""Tests del nodo router (`clasificar_intencion`) — Parte 2.

El router clasifica la intención del turno en una sola palabra. Aquí se verifica
el parser puro (sin LLM) y los cortocircuitos deterministas del nodo, que NO
gastan una llamada al LLM: recogida de parámetros en curso (``pending_tool``) e
historial sin petición del usuario.
"""

from __future__ import annotations

import pytest
from langchain_core.messages import HumanMessage

from src.agent.nodes.router import (
    VALID_INTENTS,
    clasificar_intencion_node,
    parse_intent,
)


@pytest.mark.parametrize("raw, expected", [
    ("analisis", "analisis"),
    ("Analisis.", "analisis"),
    ("Intención: análisis", "analisis"),   # tilde + ruido
    ("teoria", "teoria"),
    ("Teoría", "teoria"),
    ("texto", "texto"),
    ("TEXTO", "texto"),
    ("ni idea", "texto"),                  # fallback seguro
    ("", "texto"),
])
def test_parse_intent_maps_to_valid_intent(raw: str, expected: str) -> None:
    assert parse_intent(raw) == expected
    assert parse_intent(raw) in VALID_INTENTS


def test_node_shortcircuits_when_collecting_params() -> None:
    """Con una recogida de parámetros en curso, el router no clasifica: devuelve
    ``continuacion`` sin invocar al LLM (el razonador hace replay/merge)."""
    state = {"pending_tool": "detect_drift", "messages": [HumanMessage(content="0.2")]}
    assert clasificar_intencion_node(state) == {"intent": "continuacion"}


def test_node_returns_texto_when_no_human_message() -> None:
    """Sin petición del usuario en el historial, no se clasifica: ``texto``."""
    state = {"pending_tool": None, "messages": []}
    assert clasificar_intencion_node(state) == {"intent": "texto"}

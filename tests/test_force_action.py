"""Tests del detector que decide cuándo forzar la tool call (Parte 2).

razonador_node reintenta forzando la tool call cuando el modelo responde en prosa a
una petición de acción clara. Aquí se verifican las funciones puras (sin LLM)
_should_force_action y _looks_like_action_request.
"""

from __future__ import annotations

import pytest

from src.agent.nodes.reasoning import (
    _looks_like_action_request,
    _should_force_action,
)


@pytest.mark.parametrize("text", [
    "Generame una serie sintetica",
    "Genera datos sintéticos",
    "Hazme un forecast",
    "Detecta drift en mis datos",
    "Aumenta mi serie con más observaciones",
    "Crea una variable exógena con PCA",
])
def test_action_requests_detected(text):
    assert _looks_like_action_request(text) is True


@pytest.mark.parametrize("text", [
    "¿Qué es el data drift?",
    "Explícame la diferencia entre KS y JS",
    "¿Para qué sirve un forecast?",   # teoría aunque mencione 'forecast'
    "¿Qué puedes hacer?",
    "Hola, buenas",
    "",
])
def test_non_action_requests_ignored(text):
    assert _looks_like_action_request(text) is False


def test_force_gate_requires_csv_for_file_tools():
    # Sin CSV: solo se fuerza la generación sintética (no necesita fichero).
    assert _should_force_action("Generame una serie sintetica", csv_loaded=False) is True
    assert _should_force_action("Hazme un forecast", csv_loaded=False) is False
    # Con CSV: se fuerza cualquier acción (forecast incluido).
    assert _should_force_action("Hazme un forecast", csv_loaded=True) is True


def test_force_gate_never_fires_for_theory_or_capabilities():
    assert _should_force_action("¿Qué es el drift?", csv_loaded=True) is False
    assert _should_force_action("¿Qué puedes hacer?", csv_loaded=True) is False

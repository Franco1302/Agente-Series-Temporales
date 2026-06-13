"""Snapshot del prompt por defecto + sanity checks de las variantes de ablación.

El prompt por defecto (todos los flags a True) debe ser byte-exact al previo al
refactor; si alguien cambia el texto de cualquier bloque, el test cae con un diff.
Los snapshots viven en tests/fixtures/prompt_snapshots/; para regenerar uno, bórralo
y deja que el test lo reescriba tras confirmar que el cambio es intencional.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from src.agent.prompts.system_prompts import (
    FEWSHOT_EXAMPLES,
    RULE_NO_INVENT,
    RULE_THEORY_TOOL_BEHAVIOR,
    RULE_THEORY_TOOL_REGLAS,
    PromptAblation,
    build_system_prompt,
)

SNAPSHOTS = Path(__file__).parent / "fixtures" / "prompt_snapshots"


def _read_snapshot(name: str) -> str:
    """Lee el snapshot eliminando el trailing newline que añade el fichero."""
    return (SNAPSHOTS / f"{name}.txt").read_text(encoding="utf-8").rstrip("\n")


# ── Snapshots del prompt por defecto ────────────────────────────────────────


@pytest.mark.parametrize(
    "snapshot_name, kwargs",
    [
        ("no_file", {}),
        (
            "with_file",
            {
                "csv_path": "/home/franco/Documentos/TFG/data/temp_uploads/example.csv",
                "csv_metadata": {"columns": ["Indice", "valor"], "rows": 200},
            },
        ),
        ("with_explain_drift", {"tool_result_to_explain": "detect_drift"}),
        ("with_explain_forecast", {"tool_result_to_explain": "forecast_time_series"}),
        # consultar_teoria NO es analítica → no añade el bloque EXPLICACIÓN
        ("with_explain_rag_no_extra", {"tool_result_to_explain": "consultar_teoria"}),
    ],
)
def test_default_prompt_matches_snapshot(snapshot_name: str, kwargs: dict) -> None:
    """El prompt por defecto (ablation=None) coincide byte-a-byte con el snapshot."""
    expected = _read_snapshot(snapshot_name)
    got = build_system_prompt(**kwargs)
    assert got == expected, (
        f"El prompt cambió respecto al snapshot '{snapshot_name}'. "
        "Si el cambio es intencional, regenera el snapshot."
    )


def test_explicit_default_ablation_equals_implicit() -> None:
    """Pasar PromptAblation() explícitamente o None produce el mismo prompt."""
    p_implicit = build_system_prompt()
    p_explicit = build_system_prompt(ablation=PromptAblation())
    assert p_implicit == p_explicit


# ── Comportamiento de los flags de ablación ─────────────────────────────────


def test_no_invent_off_removes_rule() -> None:
    """Con include_no_invent=False la regla anti-invención desaparece del prompt."""
    full = build_system_prompt()
    ablated = build_system_prompt(ablation=PromptAblation(include_no_invent=False))
    assert RULE_NO_INVENT in full
    assert RULE_NO_INVENT not in ablated
    # El resto sigue presente (sanity).
    assert RULE_THEORY_TOOL_BEHAVIOR in ablated
    assert RULE_THEORY_TOOL_REGLAS in ablated


def test_theory_tool_off_removes_both_directives_and_swaps_tool9() -> None:
    """Con include_theory_tool=False desaparecen las dos directivas y la descripción de la tool nº 9 se sustituye por la variante neutra."""
    full = build_system_prompt()
    ablated = build_system_prompt(ablation=PromptAblation(include_theory_tool=False))

    assert RULE_THEORY_TOOL_BEHAVIOR in full
    assert RULE_THEORY_TOOL_REGLAS in full
    assert RULE_THEORY_TOOL_BEHAVIOR not in ablated
    assert RULE_THEORY_TOOL_REGLAS not in ablated

    # La descripción enfática ("SIEMPRE para preguntas teóricas...") se sustituye por la neutra ("Recupera contexto teórico...").
    assert "SIEMPRE para preguntas teóricas" in full
    assert "SIEMPRE para preguntas teóricas" not in ablated
    assert "Recupera contexto teórico" in ablated
    # La tool sigue listada en el prompt (necesaria para que el binding del LLM la encuentre).
    assert "consultar_teoria" in ablated


def test_fewshot_off_removes_examples_block() -> None:
    """Con include_fewshot=False los dos ejemplos desaparecen del bloque EXPLICACIÓN DE RESULTADOS, pero la cabecera y las etiquetas se mantienen."""
    full = build_system_prompt(tool_result_to_explain="detect_drift")
    ablated = build_system_prompt(
        tool_result_to_explain="detect_drift",
        ablation=PromptAblation(include_fewshot=False),
    )
    assert FEWSHOT_EXAMPLES in full
    assert FEWSHOT_EXAMPLES not in ablated
    assert "**RESULTADO:**" in ablated  # la cabecera del bloque se conserva
    assert "**INTERPRETACIÓN:**" in ablated
    assert "**SIGUIENTE PASO:**" in ablated


def test_all_off_still_produces_valid_prompt() -> None:
    """Apagar las tres reglas a la vez no rompe la composición del prompt."""
    ablated = build_system_prompt(
        tool_result_to_explain="detect_drift",
        ablation=PromptAblation(
            include_no_invent=False,
            include_theory_tool=False,
            include_fewshot=False,
        ),
    )
    # Estructura mínima (rol, comportamiento, herramientas, explicar, fichero "ninguno") sin ninguna de las reglas.
    for header in ("IDIOMA:", "COMPORTAMIENTO:", "HERRAMIENTAS:",
                   "EXPLICACIÓN DE RESULTADOS:", "FICHERO ACTIVO: ninguno."):
        assert header in ablated, f"Falta cabecera '{header}' en el prompt ablacionado"
    assert RULE_NO_INVENT not in ablated
    assert RULE_THEORY_TOOL_BEHAVIOR not in ablated
    assert FEWSHOT_EXAMPLES not in ablated

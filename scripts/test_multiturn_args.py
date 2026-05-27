"""Test multi-turno del bug de pérdida de args entre turnos.

Reproduce el escenario donde el LLM local cuantizado "olvida" los args de
la tool call en el turno 2 (después de responder a una pregunta de parámetros
faltantes). El fix `_merge_with_pending_params` debe recuperar los args de
`pending_params` y completar la tool call sin volver a preguntar.

Se ejecuta en dos partes:

  1. Tests instrumentales con LLM mockeado que recorren TODAS las tools del
     mapa `TOOL_REQUIRED_PARAMS` y comprueban que, si el LLM olvida args
     entre turnos, el fix los fusiona desde `pending_params`.

  2. Pruebas E2E con LLM y API MCP reales para 2 tools representativas
     (detect_drift y generate_synthetic_distribution).

Ejecutar:
    PYTHONPATH=. python scripts/test_multiturn_args.py
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from unittest.mock import patch

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from src.agent.graph import build_agent_graph
from src.agent.nodes import reasoning as reasoning_mod
from src.agent.nodes.param_request import TOOL_REQUIRED_PARAMS


# ── Casos de "olvido del LLM" por tool ───────────────────────────────────────
#
# Para cada tool: qué args ya estaban recogidos en `pending_params` y qué
# arg nuevo emite el LLM en su tool call (olvidando el resto).

@dataclass
class ForgetfulCase:
    tool: str
    already_collected: dict[str, Any]
    llm_emits_only: dict[str, Any]

    @property
    def expected_merged(self) -> dict[str, Any]:
        merged = dict(self.already_collected)
        merged.update(self.llm_emits_only)
        return merged


_CASES: list[ForgetfulCase] = [
    ForgetfulCase(
        tool="detect_drift",
        already_collected={"file_path": "data/temp_uploads/ARMA-DIST-fin.csv", "method": "KS"},
        llm_emits_only={"index_column": "Indice"},
    ),
    ForgetfulCase(
        tool="generate_synthetic_distribution",
        already_collected={
            "start_date": "2024-01-01",
            "frequency": "D",
            "distribution_type": 1,
        },
        llm_emits_only={"distribution_params": [0.0, 1.0]},
    ),
    ForgetfulCase(
        tool="generate_synthetic_arma",
        already_collected={"start_date": "2024-01-01"},
        llm_emits_only={"frequency": "D"},
    ),
    ForgetfulCase(
        tool="generate_synthetic_periodic",
        already_collected={
            "start_date": "2024-01-01",
            "frequency": "D",
            "period_length": 7,
            "pattern_type": 1,
        },
        llm_emits_only={"distribution_type": 1, "distribution_params": [0.0, 1.0]},
    ),
    ForgetfulCase(
        tool="generate_synthetic_trend",
        already_collected={
            "start_date": "2024-01-01",
            "frequency": "D",
            "trend_type": 1,
        },
        llm_emits_only={"trend_params": [0.5, 1.0]},
    ),
    ForgetfulCase(
        tool="augment_time_series",
        already_collected={
            "file_path": "data/temp_uploads/ARMA-DIST-fin.csv",
            "index_column": "Indice",
            "strategy": "normal",
        },
        llm_emits_only={"size": 50, "frequency": "D"},
    ),
    ForgetfulCase(
        tool="create_exogenous_variable",
        already_collected={
            "file_path": "data/temp_uploads/ARMA-DIST-fin.csv",
            "index_column": "Indice",
        },
        llm_emits_only={"new_column_name": "y", "relation": "linear"},
    ),
    ForgetfulCase(
        tool="forecast_time_series",
        already_collected={
            "file_path": "data/temp_uploads/ARMA-DIST-fin.csv",
            "index_column": "Indice",
            "target_column": "valor",
        },
        llm_emits_only={"forecast_steps": 30},
    ),
]


def _make_fake_llm(tool: str, args: dict[str, Any]):
    """Devuelve una factory que el patcher usará como `get_llm_with_tools`."""
    forgetful = AIMessage(
        content="",
        tool_calls=[{
            "name": tool,
            "args": args,
            "id": "call_mock",
            "type": "tool_call",
        }],
    )

    class _FakeLLM:
        def invoke(self, _messages):
            return forgetful

    return lambda _tools: _FakeLLM()


def _run_forgetful_case(case: ForgetfulCase) -> tuple[bool, str]:
    """Ejecuta razonador_node con un LLM mockeado que olvida args.

    Returns (ok, detalle).
    """
    state_in: dict[str, Any] = {
        "messages": [
            HumanMessage(content=f"prepara {case.tool}"),
            AIMessage(content="Necesito más datos."),
            HumanMessage(content="aquí van"),
        ],
        "csv_path": case.already_collected.get("file_path"),
        "csv_metadata": None,
        "pending_tool": case.tool,
        "pending_params": dict(case.already_collected),
        "optionals_confirmed_for": None,
        "rag_context": None,
        "error_count": 0,
        "error_info": None,
    }

    with patch.object(
        reasoning_mod, "get_llm_with_tools",
        _make_fake_llm(case.tool, case.llm_emits_only),
    ):
        updates = reasoning_mod.razonador_node(state_in)

    out_msg = (updates.get("messages") or [None])[0]
    if not isinstance(out_msg, AIMessage):
        return False, "no se generó AIMessage de salida"

    tcs = getattr(out_msg, "tool_calls", None) or []
    if not tcs:
        return False, "la salida no tiene tool_calls"

    final_args = tcs[0]["args"] if isinstance(tcs[0], dict) else getattr(tcs[0], "args", {})

    required = TOOL_REQUIRED_PARAMS.get(case.tool, [])
    obligatorios_aportados = [r for r in required if r in case.expected_merged]
    faltan_en_call = [r for r in obligatorios_aportados if final_args.get(r) in (None, "", [])]

    if faltan_en_call:
        return False, (
            f"la tool call perdió {faltan_en_call}; args finales: {final_args}; "
            f"pending_params: {updates.get('pending_params')}"
        )

    pending_after = updates.get("pending_params") or {}
    if updates.get("pending_tool") is not None:
        faltan_en_pending = [r for r in obligatorios_aportados if pending_after.get(r) in (None, "", [])]
        if faltan_en_pending:
            return False, f"pending_params perdió {faltan_en_pending}"

    return True, f"args fusionados: {final_args}"


# ── Parte 2: E2E con LLM real ────────────────────────────────────────────────


def _stream_turn(graph, input_state: dict, config: dict) -> dict:
    last_ai_text: str = ""
    tool_calls_seen: list[tuple[str, dict]] = []
    tool_results: list[tuple[str, str]] = []
    nodos: list[str] = []

    for event in graph.stream(input_state, config=config):
        node_name = next(iter(event))
        if node_name.startswith("__"):
            continue
        nodos.append(node_name)
        node_output = event[node_name] or {}
        messages_out = node_output.get("messages", []) or []
        for msg in messages_out:
            if isinstance(msg, AIMessage):
                tcs = getattr(msg, "tool_calls", None) or []
                for tc in tcs:
                    name = tc.get("name") if isinstance(tc, dict) else getattr(tc, "name", None)
                    args = tc.get("args") if isinstance(tc, dict) else getattr(tc, "args", {})
                    tool_calls_seen.append((name or "?", args or {}))
                if not tcs and isinstance(msg.content, str) and msg.content.strip():
                    last_ai_text = msg.content
            elif isinstance(msg, ToolMessage):
                tool_results.append((getattr(msg, "name", "?"), str(msg.content)[:200]))

    state = graph.get_state(config).values
    return {
        "ai_text": last_ai_text,
        "tool_calls": tool_calls_seen,
        "tool_results": tool_results,
        "pending_tool": state.get("pending_tool"),
        "pending_params": state.get("pending_params"),
        "nodos": nodos,
    }


def _print_turn(label: str, result: dict) -> None:
    print(f"\n──── {label} ────")
    print(f"  nodos: {' → '.join(result['nodos'])}")
    for name, args in result["tool_calls"]:
        print(f"  tool_call: {name} args={args}")
    for name, content in result["tool_results"]:
        print(f"  tool_result {name}: {content[:120]}…")
    if result["ai_text"]:
        preview = result["ai_text"][:240].replace("\n", " ")
        print(f"  AI: {preview}…")
    print(f"  STATE pending_tool={result['pending_tool']} pending_params={result['pending_params']}")


def _e2e_drift() -> bool:
    csv_path = "data/temp_uploads/ARMA-DIST-fin.csv"
    if not Path(csv_path).exists():
        print(f"  SKIP: no existe {csv_path}")
        return True

    build_agent_graph.cache_clear()
    graph = build_agent_graph()
    config: dict[str, Any] = {"configurable": {"thread_id": "e2e-drift"}}

    t1 = _stream_turn(
        graph,
        {
            "messages": [HumanMessage(content="Quiero detectar drift en mis datos.")],
            "csv_path": None,
            "error_count": 0,
        },
        config,
    )
    _print_turn("E2E drift TURNO 1", t1)

    t2 = _stream_turn(
        graph,
        {"messages": [HumanMessage(content=(
            f"El fichero es '{csv_path}' y la columna índice es 'Indice'. Usa el método KS."
        ))]},
        config,
    )
    _print_turn("E2E drift TURNO 2", t2)

    pending_params = t2["pending_params"] or {}
    ejecuto = any(name == "detect_drift" for name, _ in t2["tool_results"])
    pending_ok = (
        t2["pending_tool"] == "detect_drift"
        and all(pending_params.get(k) for k in ("file_path", "index_column", "method"))
    )
    if not ejecuto and not pending_ok:
        print("  FAIL E2E drift: turno 2 no completó los 3 obligatorios.")
        return False
    print("  OK E2E drift.")
    return True


def _e2e_synthetic() -> bool:
    """E2E de generación sintética en 2 turnos.

    Turno 1: petición incompleta ("genera una serie sintética normal").
    Turno 2: usuario aporta los parámetros faltantes uno a uno.
    El fix debe evitar que el agente vuelva a pedir lo que ya se dio antes.
    """
    build_agent_graph.cache_clear()
    graph = build_agent_graph()
    config: dict[str, Any] = {"configurable": {"thread_id": "e2e-synth"}}

    t1 = _stream_turn(
        graph,
        {
            "messages": [HumanMessage(content="Genera una serie temporal sintética.")],
            "csv_path": None,
            "error_count": 0,
        },
        config,
    )
    _print_turn("E2E synth TURNO 1", t1)

    t2 = _stream_turn(
        graph,
        {"messages": [HumanMessage(content=(
            "Empieza el 2024-01-01, frecuencia diaria, distribución normal "
            "tipo 1 con parámetros [0.0, 1.0]."
        ))]},
        config,
    )
    _print_turn("E2E synth TURNO 2", t2)

    pending_params = t2["pending_params"] or {}
    requeridos = ("start_date", "frequency", "distribution_type", "distribution_params")
    pending_ok = (
        t2["pending_tool"] == "generate_synthetic_distribution"
        and all(pending_params.get(k) not in (None, "", []) for k in requeridos)
    )
    if not pending_ok:
        print(f"  FAIL E2E synth: turno 2 no completó los 4 obligatorios. pending_params={pending_params}")
        return False

    # Turno 3: aportamos periods. El agente debe completar el XOR y ejecutar.
    t3 = _stream_turn(
        graph,
        {"messages": [HumanMessage(content="Usa 100 periods.")]},
        config,
    )
    _print_turn("E2E synth TURNO 3 (aporta periods=100)", t3)

    ejecuto = any(name == "generate_synthetic_distribution" for name, _ in t3["tool_results"])
    # Buscamos un tool_result sin error
    sin_error = any(
        name == "generate_synthetic_distribution" and "error" not in content.lower()
        for name, content in t3["tool_results"]
    )
    if not ejecuto:
        print("  FAIL E2E synth: turno 3 no ejecutó la tool tras aportar periods.")
        return False
    if not sin_error:
        print("  FAIL E2E synth: la tool se ejecutó pero la API devolvió error.")
        return False
    print("  OK E2E synth: serie sintética generada sin errores.")
    return True


# ── Main ─────────────────────────────────────────────────────────────────────


def main() -> int:
    print("═══ TESTS DEL FIX DE PÉRDIDA DE ARGS ENTRE TURNOS ═══")

    print("\n── PARTE 1: tests mockeados por tool ──")
    fallos: list[str] = []
    for case in _CASES:
        ok, detalle = _run_forgetful_case(case)
        marca = "OK " if ok else "FAIL"
        print(f"  [{marca}] {case.tool}: {detalle}")
        if not ok:
            fallos.append(case.tool)

    print(f"\n  Resumen parte 1: {len(_CASES) - len(fallos)}/{len(_CASES)} tools OK")

    print("\n── PARTE 2: E2E con LLM real ──")
    e2e_drift_ok = _e2e_drift()
    e2e_synth_ok = _e2e_synthetic()

    print("\n═══ RESUMEN GLOBAL ═══")
    print(f"  Mockeados      : {len(_CASES) - len(fallos)}/{len(_CASES)}")
    print(f"  E2E drift      : {'OK' if e2e_drift_ok else 'FAIL'}")
    print(f"  E2E synth      : {'OK' if e2e_synth_ok else 'FAIL'}")

    return 0 if (not fallos and e2e_drift_ok and e2e_synth_ok) else 1


if __name__ == "__main__":
    sys.exit(main())

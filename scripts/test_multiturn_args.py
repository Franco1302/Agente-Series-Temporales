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


def _e2e_followup_trend() -> bool:
    """E2E de continuidad entre herramientas sintéticas.

    Turno 1: genera distribución normal con todos los parámetros explícitos.
    Turno extra(s): confirma tunables si el agente los pregunta.
    Turno 2: "Ahora otra igual pero con tendencia lineal." → debe invocar
    generate_synthetic_trend con start_date/frequency/periods heredados del
    turno anterior (vía session_facts.by_param + _inherit_from_session),
    sin volver a pedirlos. Solo debería faltar trend_params (y posiblemente
    trend_type si el LLM no lo infiere de "lineal").

    Criterio de fallo: start_date o frequency ausentes en pending_params tras
    el turno 2, lo que indicaría que la Capa 2 no arrastró los temporales.
    """
    build_agent_graph.cache_clear()
    graph = build_agent_graph()
    config: dict[str, Any] = {"configurable": {"thread_id": "e2e-followup-trend"}}

    t1 = _stream_turn(
        graph,
        {
            "messages": [HumanMessage(content=(
                "Genera una serie sintética con distribución normal (tipo 1, "
                "parámetros [0.0, 1.0]), desde 2024-01-01, frecuencia diaria, 100 periodos."
            ))],
            "csv_path": None,
            "error_count": 0,
        },
        config,
    )
    _print_turn("E2E followup-trend TURNO 1", t1)

    # Confirmar opcionales si el agente los pide (máx. 3 rondas para evitar bucle)
    n_extra = 0
    while t1["pending_tool"] == "generate_synthetic_distribution" and n_extra < 3:
        t1 = _stream_turn(
            graph,
            {"messages": [HumanMessage(content="Usa los valores por defecto.")]},
            config,
        )
        _print_turn(f"E2E followup-trend TURNO 1+{n_extra + 1} (confirma opcionales)", t1)
        n_extra += 1

    ejecuto_t1 = any(
        name == "generate_synthetic_distribution" for name, _ in t1["tool_results"]
    )
    if not ejecuto_t1:
        print("  FAIL E2E followup-trend: turno 1 no ejecutó generate_synthetic_distribution.")
        return False

    state_after_t1 = graph.get_state(config).values
    sf = state_after_t1.get("session_facts") or {}
    by_param = sf.get("by_param") or {}
    if not (by_param.get("start_date") or {}).get("value"):
        print(
            f"  FAIL E2E followup-trend: session_facts.by_param no tiene start_date tras turno 1. "
            f"by_param={by_param}"
        )
        return False
    print(f"  INFO turno 1 ok. by_param={by_param}")

    # Turno 2: continuidad con herramienta diferente
    t2 = _stream_turn(
        graph,
        {"messages": [HumanMessage(content="Ahora otra igual pero con tendencia lineal.")]},
        config,
    )
    _print_turn("E2E followup-trend TURNO 2", t2)

    # El agente debe haber elegido generate_synthetic_trend
    tool_names_t2 = [name for name, _ in t2["tool_calls"]]
    pending_t2 = t2["pending_tool"]
    if "generate_synthetic_trend" not in tool_names_t2 and pending_t2 != "generate_synthetic_trend":
        print(
            f"  FAIL E2E followup-trend: turno 2 no eligió generate_synthetic_trend. "
            f"tool_calls={tool_names_t2}, pending_tool={pending_t2}"
        )
        return False

    # start_date y frequency deben estar presentes en args o en pending_params,
    # NO deben haberse perdido (lo cual indicaría que la Capa 2 no los arrastró)
    args_t2: dict = {}
    for name, args in t2["tool_calls"]:
        if name == "generate_synthetic_trend":
            args_t2 = args
            break
    pending_params_t2 = t2["pending_params"] or {}

    faltan_temporales = []
    for param in ("start_date", "frequency"):
        en_args = args_t2.get(param) not in (None, "", [])
        en_pending = pending_params_t2.get(param) not in (None, "", [])
        if not en_args and not en_pending:
            faltan_temporales.append(param)

    if faltan_temporales:
        print(
            f"  FAIL E2E followup-trend: la pasada de herencia no arrastró "
            f"{faltan_temporales} desde session_facts.by_param. "
            f"args={args_t2}, pending_params={pending_params_t2}"
        )
        return False

    # El único param que debería faltar es trend_params (y quizá trend_type si
    # el LLM no lo infirió de "lineal"); start_date/frequency/periods NO deben faltar.
    print(
        f"  OK E2E followup-trend: generate_synthetic_trend con temporales arrastrados. "
        f"args={args_t2}, pending={pending_params_t2}"
    )
    return True


def _e2e_synth_to_drift() -> bool:
    """E2E de herencia cross-tool: generación sintética → detect_drift.

    Turno 1: genera una distribución sintética que produce un CSV en disco.
      → session_facts.by_param queda con start_date, frequency, periods (de la
        familia temporal_window) y, si la API expone la ruta como argumento de
        ejecución posterior, también file_path/index_column.
    Turno 2: "Ahora detecta drift sobre esos datos con KS." → debe invocar
      detect_drift heredando los campos de la familia data_source que ya
      estén en la sesión. Solo deberían faltar los que dependen del Turno 2
      (method, posiblemente index_column si no se inyectó en el Turno 1).

    Criterio: la pasada de herencia (_inherit_from_session) debe haber
    rellenado al menos algún campo de data_source o el usuario debe haberlos
    repetido explícitamente. Si tras el Turno 2 el agente vuelve a pedir
    parámetros que SÍ están en session_facts.by_param Y que detect_drift
    acepta, la herencia falló.
    """
    build_agent_graph.cache_clear()
    graph = build_agent_graph()
    config: dict[str, Any] = {"configurable": {"thread_id": "e2e-synth-to-drift"}}

    t1 = _stream_turn(
        graph,
        {
            "messages": [HumanMessage(content=(
                "Genera una serie sintética con distribución normal (tipo 1, "
                "parámetros [0.0, 1.0]), desde 2024-01-01, frecuencia diaria, 200 periodos."
            ))],
            "csv_path": None,
            "error_count": 0,
        },
        config,
    )
    _print_turn("E2E synth→drift TURNO 1", t1)

    # Confirmación de opcionales si aparece
    n_extra = 0
    while t1["pending_tool"] == "generate_synthetic_distribution" and n_extra < 3:
        t1 = _stream_turn(
            graph,
            {"messages": [HumanMessage(content="Usa los valores por defecto.")]},
            config,
        )
        _print_turn(f"E2E synth→drift TURNO 1+{n_extra + 1}", t1)
        n_extra += 1

    ejecuto_t1 = any(
        name == "generate_synthetic_distribution" for name, _ in t1["tool_results"]
    )
    if not ejecuto_t1:
        print("  FAIL E2E synth→drift: turno 1 no ejecutó la herramienta sintética.")
        return False

    state_after_t1 = graph.get_state(config).values
    sf = state_after_t1.get("session_facts") or {}
    by_param = sf.get("by_param") or {}
    print(f"  INFO session_facts.by_param tras T1: {by_param}")

    # Turno 2: pedir drift sobre los datos generados, repitiendo solo lo que
    # la sintética no llegó a fijar en by_param.
    t2 = _stream_turn(
        graph,
        {"messages": [HumanMessage(content=(
            "Ahora detecta drift sobre esos datos con el método KS. La columna índice es 'Indice'."
        ))]},
        config,
    )
    _print_turn("E2E synth→drift TURNO 2", t2)

    pending_params_t2 = t2["pending_params"] or {}
    tool_names_t2 = [name for name, _ in t2["tool_calls"]]
    pending_t2 = t2["pending_tool"]

    if "detect_drift" not in tool_names_t2 and pending_t2 != "detect_drift":
        print(
            f"  FAIL E2E synth→drift: turno 2 no eligió detect_drift. "
            f"tool_calls={tool_names_t2}, pending_tool={pending_t2}"
        )
        return False

    # Comprobar que el método quedó fijado en la tool call o en pending_params.
    args_t2: dict = {}
    for name, args in t2["tool_calls"]:
        if name == "detect_drift":
            args_t2 = args
            break
    method_ok = (
        args_t2.get("method") in ("KS", "ks", "Ks")
        or pending_params_t2.get("method") in ("KS", "ks", "Ks")
    )
    if not method_ok:
        print(
            f"  FAIL E2E synth→drift: method=KS no quedó fijado en la tool call. "
            f"args={args_t2}, pending={pending_params_t2}"
        )
        return False

    print(
        f"  OK E2E synth→drift: detect_drift invocado con method=KS. "
        f"args={args_t2}, pending={pending_params_t2}"
    )
    return True


def _e2e_drift_to_forecast() -> bool:
    """E2E de herencia cross-tool: detect_drift → forecast_time_series.

    Turno 1: detectar drift sobre un CSV real con file_path/index_column
      explícitos. Tras la ejecución exitosa, file_path e index_column quedan
      registrados en session_facts.by_param dentro de la familia data_source.
    Turno 2: "Ahora hazme un forecast a 30 pasos sobre la columna 'valor'."
      → forecast_time_series debe heredar file_path e index_column de la
      sesión, sin que el usuario los repita. Solo debería pedir
      target_column y forecast_steps (que el usuario sí aporta), y los
      tunables si aplica.

    Criterio: el turno 2 ejecuta (o queda pending solo por tunables) sin
    volver a pedir file_path ni index_column.
    """
    csv_path = "data/temp_uploads/ARMA-DIST-fin.csv"
    if not Path(csv_path).exists():
        print(f"  SKIP E2E drift→forecast: no existe {csv_path}")
        return True

    build_agent_graph.cache_clear()
    graph = build_agent_graph()
    config: dict[str, Any] = {"configurable": {"thread_id": "e2e-drift-to-forecast"}}

    t1 = _stream_turn(
        graph,
        {
            "messages": [HumanMessage(content=(
                f"Detecta drift en el fichero '{csv_path}', columna índice 'Indice', método KS."
            ))],
            "csv_path": csv_path,
            "error_count": 0,
        },
        config,
    )
    _print_turn("E2E drift→forecast TURNO 1", t1)

    n_extra = 0
    while t1["pending_tool"] == "detect_drift" and n_extra < 3:
        t1 = _stream_turn(
            graph,
            {"messages": [HumanMessage(content="Usa los valores por defecto.")]},
            config,
        )
        _print_turn(f"E2E drift→forecast TURNO 1+{n_extra + 1}", t1)
        n_extra += 1

    ejecuto_t1 = any(name == "detect_drift" for name, _ in t1["tool_results"])
    if not ejecuto_t1:
        print("  FAIL E2E drift→forecast: turno 1 no ejecutó detect_drift.")
        return False

    state_after_t1 = graph.get_state(config).values
    sf = state_after_t1.get("session_facts") or {}
    by_param = sf.get("by_param") or {}
    file_in_session = (by_param.get("file_path") or {}).get("value")
    idx_in_session = (by_param.get("index_column") or {}).get("value")
    if not (file_in_session and idx_in_session):
        print(
            f"  FAIL E2E drift→forecast: tras T1 falta file_path/index_column en session_facts. "
            f"by_param={by_param}"
        )
        return False
    print(f"  INFO session_facts.by_param tras T1: {by_param}")

    # Turno 2: forecast sin repetir file_path ni index_column.
    t2 = _stream_turn(
        graph,
        {"messages": [HumanMessage(content=(
            "Ahora hazme un forecast a 30 pasos sobre la columna 'valor'."
        ))]},
        config,
    )
    _print_turn("E2E drift→forecast TURNO 2", t2)

    tool_names_t2 = [name for name, _ in t2["tool_calls"]]
    pending_t2 = t2["pending_tool"]
    if "forecast_time_series" not in tool_names_t2 and pending_t2 != "forecast_time_series":
        print(
            f"  FAIL E2E drift→forecast: turno 2 no eligió forecast_time_series. "
            f"tool_calls={tool_names_t2}, pending_tool={pending_t2}"
        )
        return False

    args_t2: dict = {}
    for name, args in t2["tool_calls"]:
        if name == "forecast_time_series":
            args_t2 = args
            break
    pending_params_t2 = t2["pending_params"] or {}

    faltan_data_source: list[str] = []
    for param in ("file_path", "index_column"):
        en_args = args_t2.get(param) not in (None, "", [])
        en_pending = pending_params_t2.get(param) not in (None, "", [])
        if not en_args and not en_pending:
            faltan_data_source.append(param)

    if faltan_data_source:
        print(
            f"  FAIL E2E drift→forecast: herencia falló para {faltan_data_source}. "
            f"args={args_t2}, pending={pending_params_t2}"
        )
        return False

    print(
        f"  OK E2E drift→forecast: forecast_time_series con data_source heredado. "
        f"args={args_t2}, pending={pending_params_t2}"
    )
    return True


def _e2e_override_explicit() -> bool:
    """E2E de sobrescritura explícita del usuario sobre un valor heredado.

    Turno 1: genera serie con frequency='D'.
    Turno 2: "Ahora otra con tendencia lineal, pero mensual." → frequency
      debe pasar a 'M', NO heredar el 'D' del turno anterior. El resto de
      temporales (start_date, periods) sí se heredan.

    Criterio: en el turno 2, la tool call efectiva (o pending_params) tiene
    frequency='M', no 'D'. Si frequency hereda 'D' al pisar la intención del
    usuario, la pasada de herencia es incorrecta.
    """
    build_agent_graph.cache_clear()
    graph = build_agent_graph()
    config: dict[str, Any] = {"configurable": {"thread_id": "e2e-override-explicit"}}

    t1 = _stream_turn(
        graph,
        {
            "messages": [HumanMessage(content=(
                "Genera una serie sintética con distribución normal (tipo 1, parámetros "
                "[0.0, 1.0]), desde 2024-01-01, frecuencia diaria, 100 periodos."
            ))],
            "csv_path": None,
            "error_count": 0,
        },
        config,
    )
    _print_turn("E2E override-explicit TURNO 1", t1)

    n_extra = 0
    while t1["pending_tool"] == "generate_synthetic_distribution" and n_extra < 3:
        t1 = _stream_turn(
            graph,
            {"messages": [HumanMessage(content="Usa los valores por defecto.")]},
            config,
        )
        _print_turn(f"E2E override-explicit TURNO 1+{n_extra + 1}", t1)
        n_extra += 1

    if not any(name == "generate_synthetic_distribution" for name, _ in t1["tool_results"]):
        print("  FAIL E2E override-explicit: turno 1 no ejecutó la sintética.")
        return False

    t2 = _stream_turn(
        graph,
        {"messages": [HumanMessage(content=(
            "Ahora otra con tendencia lineal, pero mensual."
        ))]},
        config,
    )
    _print_turn("E2E override-explicit TURNO 2", t2)

    args_t2: dict = {}
    for name, args in t2["tool_calls"]:
        if name == "generate_synthetic_trend":
            args_t2 = args
            break
    pending_params_t2 = t2["pending_params"] or {}

    # frequency efectiva
    freq_effective = args_t2.get("frequency") or pending_params_t2.get("frequency")
    if freq_effective not in ("M", "ME", "MS"):
        print(
            f"  FAIL E2E override-explicit: frequency no fue sobrescrita por el usuario. "
            f"esperado 'M'/'ME'/'MS', obtenido frequency={freq_effective!r}. "
            f"args={args_t2}, pending={pending_params_t2}"
        )
        return False

    print(
        f"  OK E2E override-explicit: frequency={freq_effective!r} respeta la intención del usuario."
    )
    return True


def _e2e_no_invent_no_session() -> bool:
    """E2E de no-regresión de RULE_NO_INVENT cuando NO hay sesión previa.

    Sin ejecuciones anteriores, una petición vaga ("genera una serie sintética")
    debe llevar al sistema a pedir los parámetros obligatorios. La pasada de
    herencia no tiene de dónde rellenar (session_facts vacío), así que la regla
    anti-invención sigue gobernando: el agente NO debe imaginarse start_date,
    frequency, distribution_type ni distribution_params.

    Criterio: tras un único turno vago, el estado queda pending con la tool
    sintética y la lista de obligatorios faltantes no debería estar vacía
    (o la respuesta es texto pidiendo aclaración).
    """
    build_agent_graph.cache_clear()
    graph = build_agent_graph()
    config: dict[str, Any] = {"configurable": {"thread_id": "e2e-no-invent"}}

    t1 = _stream_turn(
        graph,
        {
            "messages": [HumanMessage(content="Genera una serie temporal sintética.")],
            "csv_path": None,
            "error_count": 0,
        },
        config,
    )
    _print_turn("E2E no-invent TURNO 1", t1)

    pending_tool = t1["pending_tool"] or ""
    pending_params = t1["pending_params"] or {}

    # Si el agente ejecutó la tool con args inventados, es FAIL.
    ejecuto_t1 = any(name.startswith("generate_synthetic_") for name, _ in t1["tool_results"])
    if ejecuto_t1:
        print("  FAIL E2E no-invent: el agente ejecutó la sintética inventando args.")
        return False

    if not pending_tool.startswith("generate_synthetic_"):
        # Aceptamos también la rama "pide aclaración en texto" del prompt.
        if t1["ai_text"]:
            print("  OK E2E no-invent: el agente pidió aclaración en texto sin inventar.")
            return True
        print(
            f"  FAIL E2E no-invent: ni pending sintético ni texto aclaratorio. "
            f"pending_tool={pending_tool}, ai_text={t1['ai_text']!r}"
        )
        return False

    # En pending_params NO deben aparecer valores inventados para los obligatorios.
    obligatorios = TOOL_REQUIRED_PARAMS.get(pending_tool, [])
    inventados = [p for p in obligatorios if pending_params.get(p) not in (None, "", [])]
    if inventados:
        print(
            f"  FAIL E2E no-invent: pending_params contiene valores inventados para "
            f"{inventados}: {pending_params}"
        )
        return False

    print(
        f"  OK E2E no-invent: pending sin valores inventados. pending_tool={pending_tool}, "
        f"pending_params={pending_params}"
    )
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
    e2e_followup_ok = _e2e_followup_trend()

    print("\n── PARTE 3: E2E de herencia genérica cross-tool ──")
    e2e_synth_to_drift_ok = _e2e_synth_to_drift()
    e2e_drift_to_forecast_ok = _e2e_drift_to_forecast()
    e2e_override_ok = _e2e_override_explicit()
    e2e_no_invent_ok = _e2e_no_invent_no_session()

    print("\n═══ RESUMEN GLOBAL ═══")
    print(f"  Mockeados            : {len(_CASES) - len(fallos)}/{len(_CASES)}")
    print(f"  E2E drift            : {'OK' if e2e_drift_ok else 'FAIL'}")
    print(f"  E2E synth            : {'OK' if e2e_synth_ok else 'FAIL'}")
    print(f"  E2E followup-trend   : {'OK' if e2e_followup_ok else 'FAIL'}")
    print(f"  E2E synth→drift      : {'OK' if e2e_synth_to_drift_ok else 'FAIL'}")
    print(f"  E2E drift→forecast   : {'OK' if e2e_drift_to_forecast_ok else 'FAIL'}")
    print(f"  E2E override-explicit: {'OK' if e2e_override_ok else 'FAIL'}")
    print(f"  E2E no-invent        : {'OK' if e2e_no_invent_ok else 'FAIL'}")

    all_e2e_ok = (
        e2e_drift_ok and e2e_synth_ok and e2e_followup_ok
        and e2e_synth_to_drift_ok and e2e_drift_to_forecast_ok
        and e2e_override_ok and e2e_no_invent_ok
    )
    return 0 if (not fallos and all_e2e_ok) else 1


if __name__ == "__main__":
    sys.exit(main())

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

    return lambda _tools, tool_choice=None: _FakeLLM()


def _run_forgetful_case(case: ForgetfulCase) -> tuple[bool, str]:
    """Ejecuta razonador_node con un LLM mockeado que olvida args.

    Returns (ok, detalle).
    """
    # El segundo HumanMessage simula lo que el usuario teclea en el turno 2.
    # Debe incluir evidencia textual de cada tipo de campo que el LLM emite
    # (fecha, frecuencia, métodos, tipos de distribución/tendencia, listas
    # numéricas, enteros, columnas) porque la defensa anti-invención de
    # `_strip_invented_args` exige respaldo textual para cada uno. Sin esto,
    # los args legítimos del turno 2 se interpretan como inventados.
    user_reply = (
        "aquí van los datos: fecha 2024-01-01, frecuencia diaria, "
        "distribución normal lineal con parámetros [0.0, 1.0], "
        "tipo de patrón amplitud, método KS kolmogorov, "
        "estrategia normal, relación PCA lineal, "
        "columna índice 'Indice', columna nueva 'y', columna objetivo 'valor', "
        "30 pasos, doce observaciones."
    )
    state_in: dict[str, Any] = {
        "messages": [
            HumanMessage(content=f"prepara {case.tool}"),
            AIMessage(content="Necesito más datos."),
            HumanMessage(content=user_reply),
        ],
        "csv_path": case.already_collected.get("file_path"),
        "csv_metadata": {"columns": ["Indice", "valor"]},
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


def _e2e_synthetic_followup() -> bool:
    """E2E de reproducción del bug de seguimiento con herencia (cross-tool).

    Reproduce el flujo exacto reportado:
      1. "Genera una serie sintética con distribución normal" → pide params.
      2. "Desde 2024-01-01, 365 puntos diarios, μ=50 y σ=10" → pide periods (XOR).
      3. "Usa 365 periods" → ejecuta generate_synthetic_distribution y puebla
         session_facts.by_param con la familia temporal_window.
      4. "Ahora con los parámetros anteriores quiero que me crees una pero con
         tendencia lineal" → es OTRA tool (generate_synthetic_trend) que debe
         heredar start_date/frequency/periods.

    El bug: en el turno 4 el agente responde en texto plano SIN emitir tool_call.
    Aserción: el turno 4 debe producir un tool_call de generate_synthetic_trend
    (ejecutado o a la espera de obligatorios nuevos vía pending_tool). FALLA si
    el último mensaje es texto plano sin ninguna tool_call.

    Discriminación de causa (sin instrumentación extra):
      * tool_calls vacío + ai_text  → H-B1 (el LLM no emitió tool_call).
      * tool_call presente con args vacíos → H-B2 (_strip_invented_args).
    """
    build_agent_graph.cache_clear()
    graph = build_agent_graph()
    config: dict[str, Any] = {"configurable": {"thread_id": "e2e-synth-followup"}}

    t1 = _stream_turn(
        graph,
        {
            "messages": [HumanMessage(content="Genera una serie sintética con distribución normal.")],
            "csv_path": None,
            "error_count": 0,
        },
        config,
    )
    _print_turn("E2E followup TURNO 1", t1)

    t2 = _stream_turn(
        graph,
        {"messages": [HumanMessage(content=(
            "Desde 2024-01-01, 365 puntos diarios, distribución normal tipo 1 "
            "con parámetros [50.0, 10.0]."
        ))]},
        config,
    )
    _print_turn("E2E followup TURNO 2", t2)

    t3 = _stream_turn(
        graph,
        {"messages": [HumanMessage(content="Usa 365 periods.")]},
        config,
    )
    _print_turn("E2E followup TURNO 3 (aporta periods=365)", t3)

    ejecuto_dist = any(
        name == "generate_synthetic_distribution" and "error" not in content.lower()
        for name, content in t3["tool_results"]
    )
    if not ejecuto_dist:
        print(
            "  SKIP E2E followup: turnos 1-3 no ejecutaron la distribución "
            "(¿API analítica caída?). No se puede reproducir la herencia."
        )
        return True

    # Turno 4: el turno del bug. Otra tool (tendencia) reutilizando params previos.
    t4 = _stream_turn(
        graph,
        {"messages": [HumanMessage(content=(
            "Ahora con los parámetros anteriores quiero que me crees una pero "
            "con tendencia lineal."
        ))]},
        config,
    )
    _print_turn("E2E followup TURNO 4 (tendencia, hereda params)", t4)

    trend_call = any(name == "generate_synthetic_trend" for name, _ in t4["tool_calls"])
    trend_pending = t4["pending_tool"] == "generate_synthetic_trend"
    trend_executed = any(name == "generate_synthetic_trend" for name, _ in t4["tool_results"])

    if not (trend_call or trend_pending or trend_executed):
        print(
            "  FAIL E2E followup: turno 4 respondió SIN emitir tool_call de "
            f"generate_synthetic_trend. Causa probable: H-B1 (texto plano). "
            f"ai_text={t4['ai_text'][:120]!r}"
        )
        return False

    # El bug del transcript era un error XOR ("periods o end_date, pero no
    # ambos"): la herencia metía periods aunque el LLM ya hubiera puesto
    # end_date. Verificamos sobre los args de la tool call de tendencia (o el
    # pending) que NO conviven ambos miembros del grupo horizon.
    trend_args = next((a for n, a in t4["tool_calls"] if n == "generate_synthetic_trend"), None)
    if trend_args is None:
        trend_args = t4["pending_params"] if trend_pending else {}
    nonempty = lambda v: v not in (None, "", [])
    if nonempty(trend_args.get("periods")) and nonempty(trend_args.get("end_date")):
        print(f"  FAIL E2E followup: la tendencia lleva periods Y end_date (XOR roto): {trend_args}")
        return False

    print(
        "  OK E2E followup: turno 4 emitió la tool de tendencia heredando params "
        "sin romper el XOR."
    )
    return True


# ── Main ─────────────────────────────────────────────────────────────────────


def _check_message_quality() -> bool:
    """Verifica que los mensajes 'Para continuar necesito…' usan descripciones MCP.

    Tras el refactor, las descripciones y defaults del usuario se leen del schema
    de cada tool MCP (`Field(description=…, default=…)`) en vez de un dict
    hand-coded. Esta verificación se asegura de que:

      * El mensaje incluye el nombre de cada parámetro obligatorio.
      * Cada parámetro tiene una descripción no trivial (len > 10).
      * No aparece el fallback `el valor de '<param>'` en NINGÚN param (señal
        de que la descripción de MCP no estaba presente y se perdió contexto).
      * No hay artefactos tipo `PydanticUndefined` o `None` como texto literal.
    """
    from src.agent.nodes.param_request import (
        TOOL_REQUIRED_PARAMS,
        _format_required_message,
        get_missing_alternative_groups,
        get_param_description,
    )

    print("\n── PARTE 3: calidad de mensajes auto-generados ──")
    fail = False
    for tool, required in sorted(TOOL_REQUIRED_PARAMS.items()):
        if not required:
            continue
        msg = _format_required_message(tool, list(required), get_missing_alternative_groups(tool, {}))
        problemas: list[str] = []
        for param in required:
            if f"**{param}**" not in msg:
                problemas.append(f"falta nombre '{param}'")
            desc = get_param_description(tool, param)
            if desc.startswith("el valor de '"):
                problemas.append(f"sin descripción MCP para '{param}'")
            elif len(desc) < 10:
                problemas.append(f"descripción demasiado corta para '{param}': {desc!r}")
        if "PydanticUndefined" in msg or "None" in msg.split():
            problemas.append("artefacto Pydantic/None en mensaje")
        marca = "OK " if not problemas else "FAIL"
        resumen = " | ".join(problemas) if problemas else f"{len(required)} params, mensaje legible"
        print(f"  [{marca}] {tool}: {resumen}")
        if problemas:
            fail = True
    return not fail


def _check_inheritance_xor() -> bool:
    """La herencia desde session_facts no debe romper un grupo XOR (oneof_group).

    Determinista, sin LLM. Si el LLM ya fijó ``end_date``, heredar ``periods``
    de la sesión dejaría ambos miembros del grupo ``horizon`` y la API fallaría
    con "periods o end_date, pero no ambos".
    """
    from src.agent.nodes.reasoning import _inherit_from_session

    print("\n── PARTE 5: herencia respeta XOR (oneof_group) ──")
    facts = {"by_param": {
        "start_date": {"value": "2024-01-01"},
        "frequency": {"value": "D"},
        "periods": {"value": 365},
    }}
    fail = False

    # Caso 1: LLM ya puso end_date → NO heredar periods; sí start_date/frequency.
    enriched, _ = _inherit_from_session("generate_synthetic_trend", {"end_date": "2024-06-30"}, facts)
    if "periods" in enriched:
        print(f"  [FAIL] heredó periods pese a end_date fijado: {enriched}")
        fail = True
    elif not (enriched.get("start_date") and enriched.get("frequency")):
        print(f"  [FAIL] no heredó start_date/frequency: {enriched}")
        fail = True
    else:
        print("  [OK ] end_date fijado → periods no heredado; start_date/frequency sí")

    # Caso 2: sin miembro del grupo fijado → periods se hereda con normalidad.
    enriched2, _ = _inherit_from_session("generate_synthetic_trend", {}, facts)
    if enriched2.get("periods") != 365:
        print(f"  [FAIL] no heredó periods cuando el grupo estaba libre: {enriched2}")
        fail = True
    else:
        print("  [OK ] grupo libre → periods heredado")

    return not fail


def _check_followup_detector() -> bool:
    """El detector de seguimiento heredante cubre las variantes del usuario."""
    from src.agent.nodes.reasoning import _is_inheriting_followup

    print("\n── PARTE 6: detector de seguimiento heredante ──")
    facts = {"by_param": {"start_date": {"value": "2024-01-01"}}}
    positivos = [
        "Ahora con los parámetros anteriores quiero una con tendencia lineal.",
        "Ahora generame una con los parametros de tiempo anteriores, PERO CON TENDENCIA LINEAL",
        "COge los anteriores",
        "usa los mismos parámetros pero con tendencia",
    ]
    fail = False
    for txt in positivos:
        if not _is_inheriting_followup(txt, facts):
            print(f"  [FAIL] no detectó seguimiento: {txt!r}")
            fail = True
    # Negativo claro: sin params en sesión no debe disparar aunque mencione "anteriores".
    if _is_inheriting_followup("los resultados anteriores", {}):
        print("  [FAIL] disparó sin params heredables en sesión")
        fail = True
    if not fail:
        print(f"  [OK ] {len(positivos)} frases detectadas; negativo sin facts no dispara")
    return not fail


def _check_tool_error_surfaced() -> bool:
    """Un error de tool ({"error": …}) debe enrutar a gestionar_error, no fingir éxito.

    Determinista, sin LLM ni API: mockeamos la ejecución de la tool para que
    devuelva un ToolMessage de error y comprobamos que tool_execution_node setea
    ``error_info`` (y NO actualiza session_facts), y que route_after_tool desvía
    a "gestionar_error".
    """
    from langchain_core.messages import AIMessage, ToolMessage
    from src.agent.nodes import tool_execution as te
    from src.agent.nodes.routing import route_after_tool

    print("\n── PARTE 7: error de tool se reporta (no se finge éxito) ──")
    fail = False

    ai = AIMessage(content="", tool_calls=[{
        "name": "generate_synthetic_trend", "args": {"trend_type": 3, "trend_params": [0.1]},
        "id": "call_x", "type": "tool_call",
    }])
    err_tm = ToolMessage(
        content='{"error": "Error de la API (500) en generate_synthetic_trend: list index out of range"}',
        name="generate_synthetic_trend", tool_call_id="call_x",
    )
    state = {"messages": [ai], "session_facts": {"by_param": {"start_date": {"value": "2024-01-01"}}}}

    orig = te._run_tools_sync
    te._run_tools_sync = lambda _s: {"messages": [err_tm]}
    try:
        out = te.tool_execution_node(state)
    finally:
        te._run_tools_sync = orig

    if not out.get("error_info"):
        print(f"  [FAIL] no se seteó error_info en un resultado de error: {out.keys()}")
        fail = True
    elif "session_facts" in out:
        print("  [FAIL] se actualizó session_facts pese al error")
        fail = True
    else:
        print(f"  [OK ] error_info seteado: {out['error_info'][:60]!r}")

    # route_after_tool con error_info → gestionar_error
    dest = route_after_tool({**state, **out, "messages": state["messages"] + [err_tm]})
    if dest != "gestionar_error":
        print(f"  [FAIL] route_after_tool fue a {dest!r}, esperaba 'gestionar_error'")
        fail = True
    else:
        print("  [OK ] route_after_tool → gestionar_error")

    # Caso éxito: limpia error_info/error_count
    ok_tm = ToolMessage(
        content='{"output_path": "/tmp/x.csv", "rows_generated": 10, "summary": "ok"}',
        name="generate_synthetic_trend", tool_call_id="call_x",
    )
    te._run_tools_sync = lambda _s: {"messages": [ok_tm]}
    try:
        out_ok = te.tool_execution_node({**state, "error_info": "viejo", "error_count": 2})
    finally:
        te._run_tools_sync = orig
    if out_ok.get("error_info") is not None or out_ok.get("error_count") != 0:
        print(f"  [FAIL] éxito no limpió estado de error: error_info={out_ok.get('error_info')} count={out_ok.get('error_count')}")
        fail = True
    else:
        print("  [OK ] éxito limpia error_info/error_count")

    return not fail


def _check_trend_arity() -> bool:
    """trend_params debe validar su aridad según trend_type (contrato de la API).

    Contrato real: tipos 1/3/4 exigen exactamente 2 coeficientes; el 2 exige ≥1.
    Sin esta validación la API devolvía un IndexError 500 al recibir 1 coef.
    """
    import pydantic
    from mcp_server.tools.synthetic import GenerateTrendInput

    print("\n── PARTE 8: aridad de trend_params (contrato API) ──")
    casos = [(3, [0.1], False), (3, [0.1, 0.2], True), (1, [1.0], False),
             (1, [1.0, 2.0], True), (2, [1.0], True), (2, [1, 2, 3], True), (4, [1.0], False)]
    fail = False
    for tt, params, should_ok in casos:
        try:
            GenerateTrendInput(start_date="2024-01-01", frequency="D", periods=10,
                               trend_type=tt, trend_params=params)
            ok = True
        except pydantic.ValidationError:
            ok = False
        if ok != should_ok:
            print(f"  [FAIL] tipo={tt} params={params} valid={ok} (esperado {should_ok})")
            fail = True
    if not fail:
        print(f"  [OK ] {len(casos)} casos de aridad validados correctamente")
    return not fail


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

    quality_ok = _check_message_quality()
    xor_ok = _check_inheritance_xor()
    detector_ok = _check_followup_detector()
    tool_err_ok = _check_tool_error_surfaced()
    trend_arity_ok = _check_trend_arity()

    print("\n── PARTE 4: E2E con LLM real ──")
    e2e_drift_ok = _e2e_drift()
    e2e_synth_ok = _e2e_synthetic()
    e2e_followup_ok = _e2e_synthetic_followup()

    print("\n═══ RESUMEN GLOBAL ═══")
    print(f"  Mockeados      : {len(_CASES) - len(fallos)}/{len(_CASES)}")
    print(f"  Calidad mensaje: {'OK' if quality_ok else 'FAIL'}")
    print(f"  Herencia XOR   : {'OK' if xor_ok else 'FAIL'}")
    print(f"  Detector follow: {'OK' if detector_ok else 'FAIL'}")
    print(f"  Error de tool  : {'OK' if tool_err_ok else 'FAIL'}")
    print(f"  Aridad trend   : {'OK' if trend_arity_ok else 'FAIL'}")
    print(f"  E2E drift      : {'OK' if e2e_drift_ok else 'FAIL'}")
    print(f"  E2E synth      : {'OK' if e2e_synth_ok else 'FAIL'}")
    print(f"  E2E followup   : {'OK' if e2e_followup_ok else 'FAIL'}")

    return 0 if (
        not fallos and quality_ok and xor_ok and detector_ok and tool_err_ok
        and trend_arity_ok and e2e_drift_ok and e2e_synth_ok and e2e_followup_ok
    ) else 1


if __name__ == "__main__":
    sys.exit(main())

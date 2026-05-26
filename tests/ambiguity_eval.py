"""Batería de ambigüedad (Tarea 2 del brief del Capítulo 6).

Mide cómo se comporta el agente ante peticiones incompletas o ambiguas frente
al modelo y a las herramientas reales (no se *mockea* nada). Produce dos CSVs
en ``tests/results/``:

  * ``ambiguity_scorecard.csv``  — una fila por (caso, repetición).
  * ``ambiguity_summary.csv``    — una fila por ``case_type`` y un agregado
                                   global con las cuatro tasas (invention,
                                   correct_ask, over_ask, routing_accuracy).

Cada fila lleva ``run_id``, ``model``, ``temperature`` y ``commit`` para que
sea reproducible y comparable entre corridas.

Casos
-----
Definidos en ``tests/data/ambiguity_cases.json`` con taxonomía:

  * ``missing_param``    — petición a la que faltan datos obligatorios.
  * ``ambiguous_tool``   — petición compatible con varias herramientas.
  * ``theory_vs_op``     — pregunta teórica (debe ir por ``recuperar_contexto``).
  * ``out_of_scope``     — petición fuera del dominio (debe declinar en texto).
  * ``implicit_context`` — petición que reusa el turno anterior; el harness
                           reproduce el turno previo en el mismo ``thread_id``.

Métricas
--------
  * ``invention_rate``   = fracción de filas donde el LLM emitió una tool call
                          incluyendo algún parámetro listado en
                          ``forbidden_invent`` de su caso. **Debe tender a 0.**
  * ``correct_ask_rate`` = entre las filas con ``must_ask=true``, fracción
                          donde el agente alcanzó ``solicitar_parametros``.
  * ``over_ask_rate``    = entre las filas con ``must_ask=false``, fracción
                          donde el agente alcanzó ``solicitar_parametros``.
  * ``routing_accuracy`` = fracción donde el nodo alcanzado coincide con
                          ``expected_route``.

Etiquetas de ``reached_route``: ``solicitar_parametros``,
``recuperar_contexto``, ``ejecutar_herramienta``, ``fin_texto`` (texto sin
tool call). El clasificador inspecciona la secuencia de nodos del ``stream``
del grafo y prioriza en este orden.

Uso
---
  python -m tests.ambiguity_eval --model qwen2.5:3b-instruct-q4_K_M --reps 5
  python -m tests.ambiguity_eval --reps 1 --cases MP-01 TH-02   # smoke test
  python -m tests.ambiguity_eval --reps 5 --csv-out other.csv --summary-out s.csv

Requiere Ollama corriendo y la API MCP arrancada (las tools se conectan al
servidor real). La batería NO mockea ni el LLM ni las herramientas, por
diseño: queremos medir el comportamiento end-to-end.
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import json
import os
import subprocess
import sys
import time
import uuid
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CASES_FILE = PROJECT_ROOT / "tests" / "data" / "ambiguity_cases.json"
RESULTS_DIR = PROJECT_ROOT / "tests" / "results"
DEFAULT_SCORECARD = RESULTS_DIR / "ambiguity_scorecard.csv"
DEFAULT_SUMMARY = RESULTS_DIR / "ambiguity_summary.csv"


# ── Definición y carga de casos ─────────────────────────────────────────────


@dataclass
class AmbiguityCase:
    id: str
    case_type: str
    prompt: str
    expected_route: str
    must_ask: bool
    forbidden_invent: list[str] = field(default_factory=list)
    prev_turn: str | None = None


def load_cases(path: Path = CASES_FILE) -> list[AmbiguityCase]:
    """Carga la lista de casos desde el JSON, ignorando claves con prefijo '_'."""
    raw = json.loads(path.read_text(encoding="utf-8"))
    cases: list[AmbiguityCase] = []
    for c in raw["cases"]:
        cases.append(AmbiguityCase(
            id=c["id"],
            case_type=c["case_type"],
            prompt=c["prompt"],
            expected_route=c["expected_route"],
            must_ask=c["must_ask"],
            forbidden_invent=list(c.get("forbidden_invent", []) or []),
            prev_turn=c.get("prev_turn"),
        ))
    return cases


# ── Trazado del grafo y clasificación del nodo alcanzado ────────────────────


@dataclass
class TurnTrace:
    visited_nodes: list[str]
    tool_call_name: str | None
    tool_call_args: dict[str, Any]
    final_text: str


def _extract_tool_call(message: Any) -> tuple[str | None, dict]:
    """Devuelve (name, args) del primer tool_call de un AIMessage, o (None, {})."""
    tcs = getattr(message, "tool_calls", None) or []
    if not tcs:
        return None, {}
    call = tcs[0]
    if isinstance(call, dict):
        return call.get("name"), (call.get("args") or {})
    return getattr(call, "name", None), (getattr(call, "args", {}) or {})


def _stream_turn(graph, state: dict, config: dict) -> TurnTrace:
    """Ejecuta un turno del grafo y captura visitas, tool_call y texto final."""
    from langchain_core.messages import AIMessage

    visited: list[str] = []
    tc_name: str | None = None
    tc_args: dict = {}
    final_text = ""

    for event in graph.stream(state, config=config):
        for node_name, output in event.items():
            if node_name.startswith("__"):
                continue
            visited.append(node_name)
            messages = (output or {}).get("messages", []) or []
            for m in messages:
                if isinstance(m, AIMessage):
                    name, args = _extract_tool_call(m)
                    if name is not None and tc_name is None:
                        tc_name, tc_args = name, args
                    # Texto final: AIMessage sin tool_calls y con contenido.
                    if not getattr(m, "tool_calls", None) and isinstance(m.content, str) \
                            and m.content.strip():
                        final_text = m.content
    return TurnTrace(visited, tc_name, tc_args, final_text)


# El orden de prioridad refleja el grafo: si pasamos por solicitar_parametros
# es porque el flujo terminó pidiendo datos; si pasamos por recuperar_contexto
# pero acabamos en texto, fue una respuesta teórica vía RAG; etc.
_ROUTE_PRIORITY = ("solicitar_parametros", "recuperar_contexto", "ejecutar_herramienta")


def _classify_route(trace: TurnTrace) -> str:
    """Devuelve la etiqueta de ``reached_route`` a partir de los nodos visitados.

    Si el flujo no pasó por ninguno de los nodos prioritarios, el turno acabó
    en texto plano y devolvemos ``fin_texto``.
    """
    visited = set(trace.visited_nodes)
    for label in _ROUTE_PRIORITY:
        if label in visited:
            return label
    return "fin_texto"


# ── Fixture CSV reutilizable (solo se construye una vez por corrida) ────────


_FIXTURE_CSV_PATH: str | None = None


def _build_fixture_csv() -> str:
    """Genera un CSV real con ``generate_synthetic_distribution`` y devuelve su ruta.

    Sigue el principio de `feedback_test_data_via_api`: los inputs se generan
    llamando a la propia API en vez de duplicar lógica en numpy/pandas.
    """
    global _FIXTURE_CSV_PATH
    if _FIXTURE_CSV_PATH is not None and Path(_FIXTURE_CSV_PATH).exists():
        return _FIXTURE_CSV_PATH

    from mcp_server.tools.synthetic import generate_synthetic_distribution

    result = asyncio.run(generate_synthetic_distribution(
        start_date="2024-01-01",
        periods=200,
        frequency="D",
        distribution_type=1,
        distribution_params=[0.0, 1.0],
        column_name="valor",
    ))
    if "error" in result:
        raise RuntimeError(f"No se pudo preparar el CSV fixture: {result['error']}")
    _FIXTURE_CSV_PATH = result["output_path"]
    return _FIXTURE_CSV_PATH


# ── Ejecución de un caso (con replay del turno previo si aplica) ────────────


def _needs_csv(case: AmbiguityCase) -> bool:
    return "{fixture_csv}" in (case.prev_turn or "") or "{fixture_csv}" in case.prompt


def _render(text: str, csv_fixture: str) -> str:
    return text.replace("{fixture_csv}", csv_fixture)


def run_one(
    case: AmbiguityCase,
    csv_fixture: str,
    ablation: Any = None,
    verbose: bool = False,
) -> dict:
    """Ejecuta una repetición del caso y devuelve un dict con las métricas.

    El parámetro ``ablation`` se reserva para la Tarea 3 (estudio de
    ablación): cuando es no-None se pasa a ``build_system_prompt`` vía un
    monkey-patch local. Hoy el grafo construye el prompt directamente desde
    ``razonador_node``, así que el harness lo monkey-patchea sólo durante el
    turno; con ``ablation=None`` el flujo es el mismo que en producción.
    """
    from langchain_core.messages import HumanMessage
    from src.agent.graph import build_agent_graph

    needs_csv = _needs_csv(case)
    csv_path_state: str | None = csv_fixture if needs_csv else None

    # Reconstruir el grafo para que reciba el LLM con los env vars vigentes
    # (ver ``main`` — cachea get_chat_ollama y el grafo).
    build_agent_graph.cache_clear()
    graph = build_agent_graph()
    thread_id = f"amb-{case.id}-{uuid.uuid4().hex[:6]}"
    config = {"configurable": {"thread_id": thread_id}}

    _maybe_patch = _patch_ablation_context(ablation)
    with _maybe_patch:
        if case.prev_turn:
            prev_text = _render(case.prev_turn, csv_fixture)
            prev_state = {
                "messages": [HumanMessage(content=prev_text)],
                "csv_path": csv_path_state,
                "error_count": 0,
            }
            try:
                _stream_turn(graph, prev_state, config)
            except Exception as exc:  # noqa: BLE001
                if verbose:
                    print(f"    [WARN] prev_turn falló: {exc!s}")

        prompt = _render(case.prompt, csv_fixture)
        state = {
            "messages": [HumanMessage(content=prompt)],
            "csv_path": csv_path_state,
            "error_count": 0,
        }

        t0 = time.perf_counter()
        try:
            trace = _stream_turn(graph, state, config)
            error_msg = ""
        except Exception as exc:  # noqa: BLE001
            duration = time.perf_counter() - t0
            return {
                "case_id": case.id, "case_type": case.case_type,
                "must_ask": case.must_ask,
                "reached_route": "error",
                "invented_params": False,
                "asked_param": False,
                "over_asked": False,
                "correct": False,
                "tool_call_name": "",
                "tool_call_args": "{}",
                "duration_s": round(duration, 2),
                "error": str(exc)[:200],
            }
        duration = time.perf_counter() - t0

    reached = _classify_route(trace)
    invented = (
        bool(trace.tool_call_name) and bool(case.forbidden_invent)
        and any(k in trace.tool_call_args for k in case.forbidden_invent)
    )
    asked = reached == "solicitar_parametros"
    over_asked = asked and not case.must_ask
    correct = (
        reached == case.expected_route
        and not invented
        and (asked or not case.must_ask)
    )

    return {
        "case_id": case.id,
        "case_type": case.case_type,
        "must_ask": case.must_ask,
        "reached_route": reached,
        "invented_params": invented,
        "asked_param": asked,
        "over_asked": over_asked,
        "correct": correct,
        "tool_call_name": trace.tool_call_name or "",
        "tool_call_args": json.dumps(trace.tool_call_args, ensure_ascii=False),
        "duration_s": round(duration, 2),
        "error": error_msg,
    }


# ── Monkey-patch de ablación (sólo se usa desde scripts/ablation_eval.py) ───


class _NoopContext:
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False


def _patch_ablation_context(ablation: Any):
    """Devuelve un context manager que parchea ``build_system_prompt`` para que
    use la ablación pasada, si es no-None. Si es None, devuelve un no-op.

    El parche se aplica al módulo de razonamiento, que es quien hoy llama a
    ``build_system_prompt`` para inyectar el SystemMessage. Mantener este
    parche local al turno evita contaminación entre repeticiones.
    """
    if ablation is None:
        return _NoopContext()

    from src.agent.nodes import reasoning as _reasoning
    from src.agent.prompts import system_prompts as _sp

    original = _reasoning.build_system_prompt

    def _patched(csv_path=None, csv_metadata=None, tool_result_to_explain=None,
                 ablation=ablation, _orig=_sp.build_system_prompt):
        return _orig(
            csv_path=csv_path,
            csv_metadata=csv_metadata,
            tool_result_to_explain=tool_result_to_explain,
            ablation=ablation,
        )

    class _Ctx:
        def __enter__(self):
            _reasoning.build_system_prompt = _patched
            return self

        def __exit__(self, exc_type, exc, tb):
            _reasoning.build_system_prompt = original
            return False

    return _Ctx()


# ── Agregación y persistencia ───────────────────────────────────────────────


_SCORECARD_FIELDS = [
    "run_id", "model", "temperature", "commit",
    "case_id", "case_type", "repetition",
    "reached_route", "invented_params", "asked_param", "over_asked", "correct",
    "tool_call_name", "tool_call_args", "duration_s", "error",
]

_SUMMARY_FIELDS = [
    "run_id", "model", "temperature", "commit",
    "case_type", "n_runs", "n_must_ask", "n_not_must_ask",
    "invention_rate", "correct_ask_rate", "over_ask_rate", "routing_accuracy",
    "mean_duration_s",
]


def _write_scorecard(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=_SCORECARD_FIELDS)
        writer.writeheader()
        for r in rows:
            writer.writerow({
                "run_id": r["run_id"],
                "model": r["model"],
                "temperature": r["temperature"],
                "commit": r["commit"],
                "case_id": r["case_id"],
                "case_type": r["case_type"],
                "repetition": r["repetition"],
                "reached_route": r["reached_route"],
                "invented_params": int(bool(r["invented_params"])),
                "asked_param": int(bool(r["asked_param"])),
                "over_asked": int(bool(r["over_asked"])),
                "correct": int(bool(r["correct"])),
                "tool_call_name": r["tool_call_name"] or "",
                "tool_call_args": r["tool_call_args"],
                "duration_s": r["duration_s"],
                "error": r["error"],
            })


def compute_summary(
    rows: list[dict],
    run_id: str,
    model: str,
    temperature: float,
    commit: str,
) -> list[dict]:
    """Calcula las cuatro tasas por case_type y un total global."""
    buckets: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        buckets[r["case_type"]].append(r)
    buckets["total"] = list(rows)

    summary: list[dict] = []
    for case_type, rs in buckets.items():
        n = len(rs)
        if n == 0:
            continue
        must_ask_rows = [r for r in rs if r["must_ask"]]
        not_must_ask_rows = [r for r in rs if not r["must_ask"]]

        inv_rate = sum(1 for r in rs if r["invented_params"]) / n
        if must_ask_rows:
            correct_ask = sum(1 for r in must_ask_rows if r["asked_param"]) / len(must_ask_rows)
        else:
            correct_ask = float("nan")
        if not_must_ask_rows:
            over_ask = sum(1 for r in not_must_ask_rows if r["asked_param"]) / len(not_must_ask_rows)
        else:
            over_ask = float("nan")
        # routing_accuracy: cada caso conoce su expected_route, pero aquí
        # `r["correct"]` no garantiza solo routing (incluye also invention/ask).
        # Re-calculamos sobre `reached_route == expected_route` por fila.
        # Para hacerlo necesitamos expected_route; lo recuperamos del caso.
        routing_correct = sum(1 for r in rs if r.get("routing_correct")) / n
        mean_dur = sum(r["duration_s"] for r in rs) / n

        summary.append({
            "run_id": run_id, "model": model, "temperature": temperature, "commit": commit,
            "case_type": case_type,
            "n_runs": n,
            "n_must_ask": len(must_ask_rows),
            "n_not_must_ask": len(not_must_ask_rows),
            "invention_rate": round(inv_rate, 4),
            "correct_ask_rate": round(correct_ask, 4) if correct_ask == correct_ask else "",
            "over_ask_rate": round(over_ask, 4) if over_ask == over_ask else "",
            "routing_accuracy": round(routing_correct, 4),
            "mean_duration_s": round(mean_dur, 2),
        })
    return summary


def _write_summary(summary_rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=_SUMMARY_FIELDS)
        writer.writeheader()
        writer.writerows(summary_rows)


def _print_summary(summary_rows: list[dict]) -> None:
    sep = "=" * 95
    print(f"\n{sep}")
    print("RESUMEN AGREGADO (por case_type)")
    print(sep)
    headers = ("case_type", "n", "inv", "cor.ask", "over.ask", "rout.acc", "mean(s)")
    print(f"  {headers[0]:<18} {headers[1]:>4}  {headers[2]:>6} {headers[3]:>8} "
          f"{headers[4]:>9} {headers[5]:>9} {headers[6]:>8}")
    print("-" * 95)
    for r in summary_rows:
        ca = r["correct_ask_rate"] if r["correct_ask_rate"] != "" else "n/a"
        oa = r["over_ask_rate"] if r["over_ask_rate"] != "" else "n/a"
        print(f"  {r['case_type']:<18} {r['n_runs']:>4}  "
              f"{r['invention_rate']:>6.3f} {str(ca):>8} {str(oa):>9} "
              f"{r['routing_accuracy']:>9.3f} {r['mean_duration_s']:>8.2f}")
    print(sep)


# ── CLI ─────────────────────────────────────────────────────────────────────


def _git_short_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], cwd=PROJECT_ROOT, text=True,
        ).strip()
    except Exception:  # noqa: BLE001
        return "unknown"


def _apply_ollama_env(model: str, temperature: float) -> None:
    """Exporta OLLAMA_MODEL/OLLAMA_TEMPERATURE y limpia las cachés del LLM
    para que ``get_chat_ollama`` recree el cliente con los nuevos parámetros.
    """
    os.environ["OLLAMA_MODEL"] = model
    os.environ["OLLAMA_TEMPERATURE"] = str(temperature)

    from src.agent.graph import build_agent_graph
    from src.config.llm_config import get_chat_ollama

    get_chat_ollama.cache_clear()
    build_agent_graph.cache_clear()


def _silence_observability() -> None:
    """Apaga el subsistema de observabilidad para que la corrida no inunde
    stdout/stderr con JSON. Se usa por defecto en la batería; el opt-in
    ``--observability`` permite volver a activarla cuando interese trazar.
    """
    try:
        from src.observability.logger import configure as _obs_configure
        _obs_configure(enabled=False)
    except Exception:  # noqa: BLE001
        pass


def run_battery(
    model: str,
    temperature: float,
    reps: int,
    case_filter: list[str] | None = None,
    ablation: Any = None,
    scorecard_path: Path | None = DEFAULT_SCORECARD,
    summary_path: Path | None = DEFAULT_SUMMARY,
    run_id: str | None = None,
    extra_row_fields: dict | None = None,
    observability: bool = False,
) -> tuple[list[dict], list[dict]]:
    """Ejecuta la batería entera y persiste scorecard + summary.

    Devuelve (rows, summary_rows) por si el llamador (ej. ablation_eval) los
    quiere acumular antes de volcar a disco.
    """
    _apply_ollama_env(model, temperature)
    if not observability:
        _silence_observability()

    cases = load_cases()
    if case_filter:
        cases = [c for c in cases if c.id in case_filter]
        if not cases:
            print(f"[ambiguity_eval] No hay casos que coincidan con {case_filter!r}.")
            return [], []

    csv_fixture = _build_fixture_csv()
    commit_sha = _git_short_sha()
    rid = run_id or f"{model.replace(':', '_').replace('/', '_')}-{int(time.time())}"

    print(f"[ambiguity_eval] model={model} temperature={temperature} "
          f"reps={reps} commit={commit_sha}")
    print(f"[ambiguity_eval] casos={len(cases)}  total runs={len(cases) * reps}  "
          f"run_id={rid}")
    if ablation is not None:
        print(f"[ambiguity_eval] ablation={ablation!r}")

    rows: list[dict] = []
    expected_by_id = {c.id: c.expected_route for c in cases}

    for case in cases:
        for rep in range(1, reps + 1):
            print(f"  [{case.id}] rep {rep}/{reps}", end="", flush=True)
            r = run_one(case, csv_fixture, ablation=ablation)
            r["run_id"] = rid
            r["model"] = model
            r["temperature"] = temperature
            r["commit"] = commit_sha
            r["repetition"] = rep
            r["routing_correct"] = r["reached_route"] == expected_by_id[case.id]
            if extra_row_fields:
                r.update(extra_row_fields)
            rows.append(r)
            mark = "OK" if r["correct"] else "--"
            print(f"  [{mark}] route={r['reached_route']:<22} "
                  f"tool={r['tool_call_name'] or '-':<26} "
                  f"inv={int(r['invented_params'])} ask={int(r['asked_param'])} "
                  f"({r['duration_s']}s)")

    if scorecard_path is not None:
        _write_scorecard(rows, scorecard_path)
        print(f"\n  Scorecard escrito en {scorecard_path}")

    summary = compute_summary(rows, rid, model, temperature, commit_sha)
    if summary_path is not None:
        _write_summary(summary, summary_path)
        print(f"  Summary escrito en   {summary_path}")
    _print_summary(summary)

    return rows, summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Batería de ambigüedad para el agente LangGraph."
    )
    parser.add_argument(
        "--model", default=os.getenv("OLLAMA_MODEL", "qwen2.5:3b-instruct-q4_K_M"),
        help="Modelo de Ollama a evaluar (default: $OLLAMA_MODEL o qwen2.5:3b-instruct-q4_K_M).",
    )
    parser.add_argument(
        "--temperature", type=float, default=float(os.getenv("OLLAMA_TEMPERATURE", "0.1")),
        help="Temperatura del LLM (default: $OLLAMA_TEMPERATURE o 0.1).",
    )
    parser.add_argument(
        "--reps", type=int, default=5,
        help="Repeticiones por caso para medir no-determinismo (default: 5).",
    )
    parser.add_argument(
        "--cases", nargs="+", default=None,
        help="Filtra por IDs (ej. --cases MP-01 TH-02). Default: todos.",
    )
    parser.add_argument(
        "--csv-out", type=Path, default=DEFAULT_SCORECARD,
        help=f"Ruta del scorecard granular (default: {DEFAULT_SCORECARD}).",
    )
    parser.add_argument(
        "--summary-out", type=Path, default=DEFAULT_SUMMARY,
        help=f"Ruta del summary agregado (default: {DEFAULT_SUMMARY}).",
    )
    parser.add_argument(
        "--observability", action="store_true",
        help="Activa el subsistema de observabilidad durante la corrida "
             "(por defecto se silencia para no ensuciar stdout).",
    )
    args = parser.parse_args()

    rows, summary = run_battery(
        model=args.model,
        temperature=args.temperature,
        reps=args.reps,
        case_filter=args.cases,
        scorecard_path=args.csv_out,
        summary_path=args.summary_out,
        observability=args.observability,
    )

    # Código de salida: 0 si routing_accuracy global >= 0.6, 1 si no.
    total_row = next((r for r in summary if r["case_type"] == "total"), None)
    if total_row is None:
        sys.exit(2)
    ok = total_row["routing_accuracy"] >= 0.6 and total_row["invention_rate"] <= 0.2
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()

"""Benchmark de tiempos del flujo RAG: coste de la doble sintesis de LLM.

Contexto (PlanMejoraRAG, Paso 5): hoy ``consultar_teoria`` recupera contexto y
**sintetiza una respuesta con el LLM** (sintesis interna); despues el razonador
del grafo **vuelve a sintetizar** sobre esa respuesta. Son dos pasadas de LLM
por turno teorico. El Paso 5 elimina la sintesis interna. Este script mide el
caso ANTES y DESPUES para cuantificar la mejora.

Caso medido: la consulta teorica del caso 5 de ``scripts/test_agent.py``
(Kolmogorov-Smirnov), que dispara ``consultar_teoria``.

Que mide:
  1. Micro-medicion aislada (sin grafo):
       - retrieval_ms      : tiempo de ``recuperar_documentos`` (solo recuperacion).
       - tool_total_ms     : tiempo de ``consultar_teoria.invoke`` (recuperacion +
                             sintesis interna).
       - inner_synthesis_ms: tool_total_ms - retrieval_ms ~= coste de la pasada
                             de LLM que el Paso 5 elimina.
  2. Turno completo contra el grafo del agente, con observabilidad activa. De la
     traza de logs extrae la duracion por nodo (``recuperar_contexto``,
     ``razonador``) y el total del turno.

Uso:
    python -m tests.rag_timing_benchmark --label antes_paso5
    python -m tests.rag_timing_benchmark --label despues_paso5   # tras el Paso 5

Guarda en ``docs/rag_evaluation/`` (datos para la memoria del TFG):
    timing_<label>.json   -> reporte de tiempos completo.
    trace_<label>.jsonl   -> traza de logs del turno (trazabilidad reproducible).
    timing_summary.csv    -> una fila por etiqueta: tabla antes/despues.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
EVAL_DIR = PROJECT_ROOT / "docs" / "rag_evaluation"
TIMING_SUMMARY = EVAL_DIR / "timing_summary.csv"

# Caso en particular: consulta teorica del caso 5 de scripts/test_agent.py.
CASO_QUERY = "Que es el test de Kolmogorov-Smirnov y para que sirve en la deteccion de drift?"

SUMMARY_FIELDS = [
    "label", "timestamp", "repeats", "retrieval_ms", "inner_synthesis_ms",
    "tool_total_ms", "recuperar_contexto_ms", "razonador_total_ms", "turn_total_ms",
    "query",
]


def _median(values: list[float]) -> float:
    """Mediana robusta; 0.0 si la lista esta vacia."""
    return statistics.median(values) if values else 0.0


def _measure_retrieval(query: str, repeats: int) -> list[float]:
    """Tiempos (ms) de la recuperacion pura (``recuperar_documentos``)."""
    from src.rag_engine.hybrid import recuperar_documentos

    keep_top = int(os.getenv("RAG_KEEP_TOP") or 3)
    times: list[float] = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        recuperar_documentos(query, top_k=keep_top)
        times.append((time.perf_counter() - t0) * 1000.0)
    return times


def _measure_tool(query: str, repeats: int) -> list[float]:
    """Tiempos (ms) de ``consultar_teoria.invoke`` (recuperacion + sintesis interna)."""
    from src.tools.rag_tool import consultar_teoria

    times: list[float] = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        consultar_teoria.invoke({"query": query})
        times.append((time.perf_counter() - t0) * 1000.0)
    return times


def _run_graph_turn(query: str) -> tuple[str, float, list[dict]]:
    """Ejecuta un turno completo contra el grafo y devuelve (trace_id, turn_ms, eventos)."""
    from langchain_core.messages import HumanMessage

    from src.agent.graph import build_agent_graph
    from src.observability import (
        EVENT_TURN_END,
        EVENT_TURN_START,
        TraceEvent,
        emit,
        log_file_path,
        read_trace_lines,
        start_turn,
    )

    build_agent_graph.cache_clear()
    graph = build_agent_graph()
    thread_id = f"timing-{int(time.time() * 1000)}"
    config = {"configurable": {"thread_id": thread_id}}

    trace_id = start_turn(thread_id)
    emit(TraceEvent(trace_id=trace_id, thread_id=thread_id,
                    name="timing_case", event_type=EVENT_TURN_START))

    input_state = {"messages": [HumanMessage(content=query)], "csv_path": None, "error_count": 0}
    t0 = time.perf_counter()
    for _ in graph.stream(input_state, config=config):
        pass
    turn_ms = (time.perf_counter() - t0) * 1000.0

    emit(TraceEvent(trace_id=trace_id, thread_id=thread_id, name="fin_timing_case",
                    event_type=EVENT_TURN_END, attributes={"status": "ok"}))

    events = read_trace_lines(log_file_path(), trace_id=trace_id, max_events=2000)
    return trace_id, turn_ms, events


def _node_durations(events: list[dict]) -> dict[str, list[float]]:
    """Agrupa las duraciones (ms) de los eventos node_exit por nombre de nodo."""
    durations: dict[str, list[float]] = {}
    for event in events:
        if event.get("event_type") != "node_exit":
            continue
        node = (event.get("attributes") or {}).get("node") or event.get("name") or "?"
        duration = event.get("duration_ms")
        if duration is not None:
            durations.setdefault(node, []).append(float(duration))
    return durations


def _llm_calls(events: list[dict]) -> list[dict]:
    """Extrae un resumen de los eventos llm_call de la traza."""
    calls: list[dict] = []
    for event in events:
        if event.get("event_type") != "llm_call":
            continue
        attrs = event.get("attributes") or {}
        calls.append({
            "name": event.get("name"),
            "duration_ms": event.get("duration_ms"),
            "decided": attrs.get("decided"),
            "tool_name": attrs.get("tool_name"),
            "prompt_chars": attrs.get("prompt_chars"),
            "output_tokens": attrs.get("output_tokens"),
        })
    return calls


def run_benchmark(label: str, repeats: int) -> int:
    """Ejecuta el benchmark completo y persiste los artefactos. Devuelve exit code."""
    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Benchmark de tiempos RAG  |  label={label}  repeats={repeats}")
    print(f"Caso: {CASO_QUERY}\n")

    # Warm-up: carga el modelo en Ollama y el indice BM25 (no se mide).
    print("Warm-up (carga de modelo / indice BM25)...")
    try:
        from src.tools.rag_tool import consultar_teoria
        consultar_teoria.invoke({"query": CASO_QUERY})
    except Exception as exc:  # noqa: BLE001
        print(f"[ERROR] Fallo el warm-up: {exc}")
        print("        Comprueba que Ollama este levantado y la base vectorial ingerida.")
        return 1

    # 1. Micro-medicion aislada.
    print(f"Micro-medicion aislada ({repeats} repeticiones)...")
    retrieval_times = _measure_retrieval(CASO_QUERY, repeats)
    tool_times = _measure_tool(CASO_QUERY, repeats)
    retrieval_ms = _median(retrieval_times)
    tool_ms = _median(tool_times)
    inner_synthesis_ms = max(0.0, tool_ms - retrieval_ms)

    # 2. Turno completo contra el grafo.
    print(f"Turno(s) completo(s) contra el grafo ({repeats} repeticiones)...")
    turn_times: list[float] = []
    reco_times: list[float] = []
    razonador_times: list[float] = []
    last_events: list[dict] = []
    last_trace_id = ""
    for idx in range(repeats):
        trace_id, turn_ms, events = _run_graph_turn(CASO_QUERY)
        node_durations = _node_durations(events)
        turn_times.append(turn_ms)
        reco_times.append(sum(node_durations.get("recuperar_contexto", [])))
        razonador_times.append(sum(node_durations.get("razonador", [])))
        last_events, last_trace_id = events, trace_id
        print(f"  run {idx + 1}: turno={turn_ms:.0f} ms  "
              f"recuperar_contexto={reco_times[-1]:.0f} ms  "
              f"razonador(total)={razonador_times[-1]:.0f} ms")

    report = {
        "label": label,
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "query": CASO_QUERY,
        "repeats": repeats,
        "micro_aislado_ms": {
            "retrieval": {"median": round(retrieval_ms, 1), "samples": [round(t, 1) for t in retrieval_times]},
            "tool_total": {"median": round(tool_ms, 1), "samples": [round(t, 1) for t in tool_times]},
            "inner_synthesis_estimado": round(inner_synthesis_ms, 1),
        },
        "grafo_ms": {
            "turn_total": {"median": round(_median(turn_times), 1), "samples": [round(t, 1) for t in turn_times]},
            "recuperar_contexto": {"median": round(_median(reco_times), 1), "samples": [round(t, 1) for t in reco_times]},
            "razonador_total": {"median": round(_median(razonador_times), 1), "samples": [round(t, 1) for t in razonador_times]},
        },
        "trace_representativa": {
            "trace_id": last_trace_id,
            "llm_calls": _llm_calls(last_events),
            "n_eventos": len(last_events),
        },
    }

    # Persistencia de artefactos.
    timing_json = EVAL_DIR / f"timing_{label}.json"
    timing_json.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    trace_jsonl = EVAL_DIR / f"trace_{label}.jsonl"
    with trace_jsonl.open("w", encoding="utf-8") as handle:
        for event in last_events:
            handle.write(json.dumps(event, ensure_ascii=False) + "\n")

    _upsert_summary(report)

    # Resumen por consola.
    print("\n" + "=" * 64)
    print(f"RESULTADO  (label={label}, medianas de {repeats} repeticiones)")
    print("=" * 64)
    print(f"  recuperacion pura          : {retrieval_ms:8.0f} ms")
    print(f"  consultar_teoria (total)   : {tool_ms:8.0f} ms")
    print(f"  -> sintesis interna LLM    : {inner_synthesis_ms:8.0f} ms   <- la elimina el Paso 5")
    print(f"  nodo recuperar_contexto    : {_median(reco_times):8.0f} ms")
    print(f"  nodo razonador (total)     : {_median(razonador_times):8.0f} ms")
    print(f"  turno completo             : {_median(turn_times):8.0f} ms")
    print("=" * 64)
    print(f"\nArtefactos para la memoria:")
    print(f"  - {timing_json.relative_to(PROJECT_ROOT)}")
    print(f"  - {trace_jsonl.relative_to(PROJECT_ROOT)}")
    print(f"  - {TIMING_SUMMARY.relative_to(PROJECT_ROOT)} (tabla antes/despues)")

    _print_delta(label)
    return 0


def _summary_row(report: dict) -> dict:
    """Construye la fila de summary a partir del reporte."""
    return {
        "label": report["label"],
        "timestamp": report["timestamp"],
        "repeats": report["repeats"],
        "retrieval_ms": f"{report['micro_aislado_ms']['retrieval']['median']:.1f}",
        "inner_synthesis_ms": f"{report['micro_aislado_ms']['inner_synthesis_estimado']:.1f}",
        "tool_total_ms": f"{report['micro_aislado_ms']['tool_total']['median']:.1f}",
        "recuperar_contexto_ms": f"{report['grafo_ms']['recuperar_contexto']['median']:.1f}",
        "razonador_total_ms": f"{report['grafo_ms']['razonador_total']['median']:.1f}",
        "turn_total_ms": f"{report['grafo_ms']['turn_total']['median']:.1f}",
        "query": report["query"],
    }


def _upsert_summary(report: dict) -> None:
    """Inserta o actualiza (por etiqueta) la fila de timing_summary.csv."""
    rows: list[dict] = []
    if TIMING_SUMMARY.exists():
        with TIMING_SUMMARY.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))

    new_row = _summary_row(report)
    replaced = False
    for idx, row in enumerate(rows):
        if row.get("label") == report["label"]:
            rows[idx] = new_row
            replaced = True
            break
    if not replaced:
        rows.append(new_row)

    with TIMING_SUMMARY.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=SUMMARY_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def _print_delta(current_label: str) -> None:
    """Si existen ambas etiquetas antes/despues, imprime el delta de tiempos."""
    if not TIMING_SUMMARY.exists():
        return
    with TIMING_SUMMARY.open(newline="", encoding="utf-8") as handle:
        rows = {row["label"]: row for row in csv.DictReader(handle)}
    antes, despues = rows.get("antes_paso5"), rows.get("despues_paso5")
    if not (antes and despues):
        return
    print("\nDELTA antes_paso5 -> despues_paso5:")
    for field in ("inner_synthesis_ms", "recuperar_contexto_ms", "razonador_total_ms", "turn_total_ms"):
        before, after = float(antes[field]), float(despues[field])
        delta = after - before
        pct = (delta / before * 100.0) if before else 0.0
        print(f"  {field:<22}: {before:8.0f} -> {after:8.0f} ms  ({delta:+.0f} ms, {pct:+.1f} %)")


def main() -> None:
    """Punto de entrada CLI."""
    parser = argparse.ArgumentParser(description="Benchmark de tiempos del flujo RAG.")
    parser.add_argument(
        "--label", required=True,
        help="Etiqueta de la corrida (p. ej. antes_paso5, despues_paso5).",
    )
    parser.add_argument(
        "--repeats", type=int, default=3,
        help="Repeticiones para promediar por mediana (def. 3).",
    )
    args = parser.parse_args()

    if args.repeats <= 0:
        print("[ERROR] --repeats debe ser mayor que 0.")
        sys.exit(2)

    sys.exit(run_benchmark(label=args.label, repeats=args.repeats))


if __name__ == "__main__":
    main()

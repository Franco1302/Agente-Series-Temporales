"""Estudio de ablación del *system prompt* (Tarea 3 del brief Cap. 6).

Cuantifica cuánto aporta cada regla del prompt repitiendo la batería de
ambigüedad (Tarea 2) con variantes del prompt (Tarea 1). Variantes:

  * ``baseline``         — todas las reglas activas.
  * ``no_invent_off``    — quita la regla anti-invención de parámetros.
  * ``fewshot_off``      — quita los dos ejemplos few-shot del bloque
                          EXPLICACIÓN DE RESULTADOS.
  * ``theory_tool_off``  — quita las dos directivas que obligan a usar
                          ``consultar_teoria`` para teoría.

Protocolo
---------
Mismo modelo, misma temperatura, mismos casos y mismo ``N`` que la Tarea 2,
para que las cuatro variantes sean comparables. Si la corrida ``baseline``
no se vuelve a hacer aquí (porque ya la lanzaste con la batería normal),
puedes saltársela con ``--variants no_invent_off fewshot_off theory_tool_off``;
el delta se mira contra el ``baseline`` del fichero ``ambiguity_summary.csv``.

Salidas
-------
  * ``tests/results/ablation_scorecard.csv``  — concatena las filas de las
    variantes con una columna extra ``variant``.
  * ``tests/results/ablation_summary.csv``    — matriz ``variant`` × métricas,
    además del desglose por ``case_type``. La diferencia frente a la fila
    ``baseline`` es la contribución de cada regla.

Uso
---
  python -m scripts.ablation_eval --model qwen2.5:3b-instruct-q4_K_M --reps 5
  python -m scripts.ablation_eval --variants no_invent_off fewshot_off --reps 5
  python -m scripts.ablation_eval --reps 1 --cases MP-01 TH-02  # smoke
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = PROJECT_ROOT / "tests" / "results"
DEFAULT_SCORECARD = RESULTS_DIR / "ablation_scorecard.csv"
DEFAULT_SUMMARY = RESULTS_DIR / "ablation_summary.csv"

# Importes perezosos en funciones cuando dependen del entorno (LLM/MCP). Pero
# PromptAblation y el harness no inician nada al importar, así que se pueden
# traer aquí arriba sin coste.
from src.agent.prompts.system_prompts import PromptAblation
from tests.ambiguity_eval import (
    _git_short_sha,
    compute_summary,
    run_battery,
)


_VARIANTS: dict[str, PromptAblation] = {
    "baseline":         PromptAblation(),
    "no_invent_off":    PromptAblation(include_no_invent=False),
    "fewshot_off":      PromptAblation(include_fewshot=False),
    "theory_tool_off":  PromptAblation(include_theory_tool=False),
}


_SCORECARD_FIELDS = [
    "run_id", "model", "temperature", "commit", "variant",
    "case_id", "case_type", "repetition",
    "reached_route", "invented_params", "asked_param", "over_asked", "correct",
    "tool_call_name", "tool_call_args", "duration_s", "error",
]

_SUMMARY_FIELDS = [
    "run_id", "model", "temperature", "commit", "variant",
    "case_type", "n_runs", "n_must_ask", "n_not_must_ask",
    "invention_rate", "correct_ask_rate", "over_ask_rate", "routing_accuracy",
    "mean_duration_s",
]


def _write_combined_scorecard(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=_SCORECARD_FIELDS)
        writer.writeheader()
        for r in rows:
            writer.writerow({
                "run_id": r["run_id"], "model": r["model"],
                "temperature": r["temperature"], "commit": r["commit"],
                "variant": r["variant"],
                "case_id": r["case_id"], "case_type": r["case_type"],
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


def _write_combined_summary(summary_rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=_SUMMARY_FIELDS)
        writer.writeheader()
        writer.writerows(summary_rows)


def _load_external_baseline(
    summary_csv: Path = PROJECT_ROOT / "tests" / "results" / "ambiguity_summary.csv",
) -> dict | None:
    """Carga la fila ``total`` del summary de la batería de ambigüedad (Tarea 2).

    Sirve para comparar ablaciones contra un baseline previo cuando no se
    re-ejecuta ``baseline`` en la corrida actual. Devuelve None si el fichero
    no existe o no contiene la fila ``total`` con ``variant`` ausente (la
    batería de ambigüedad no escribe la columna ``variant``).
    """
    if not summary_csv.exists():
        return None
    try:
        with summary_csv.open(encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            for row in reader:
                if row.get("case_type") == "total":
                    return row
    except Exception:  # noqa: BLE001
        return None
    return None


def _print_matrix(summary_rows: list[dict], variants: list[str]) -> None:
    """Imprime una matriz comparativa variant × métrica usando la fila ``total``.

    Si ``baseline`` no se ha ejecutado en esta corrida, intenta cargarlo desde
    ``tests/results/ambiguity_summary.csv`` para que el delta siga siendo
    comparable contra el baseline existente.
    """
    by_variant = {r["variant"]: r for r in summary_rows if r["case_type"] == "total"}

    baseline = by_variant.get("baseline")
    if baseline is None:
        external = _load_external_baseline()
        if external is not None:
            try:
                baseline = {
                    "variant": "baseline (ambiguity_summary.csv)",
                    "invention_rate": float(external["invention_rate"]),
                    "correct_ask_rate": (
                        float(external["correct_ask_rate"])
                        if external.get("correct_ask_rate") not in (None, "")
                        else ""
                    ),
                    "over_ask_rate": (
                        float(external["over_ask_rate"])
                        if external.get("over_ask_rate") not in (None, "")
                        else ""
                    ),
                    "routing_accuracy": float(external["routing_accuracy"]),
                }
            except (TypeError, ValueError):
                baseline = None

    sep = "=" * 78
    print(f"\n{sep}")
    print("ABLATION — Matriz variant × métricas (sobre el total de casos)")
    print(sep)
    hdr = f"  {'variant':<28}  {'inv':>6}  {'cor.ask':>8}  {'over.ask':>9}  {'rout.acc':>9}"
    print(hdr)
    print("-" * 78)

    if baseline is not None and baseline["variant"].startswith("baseline ("):
        # Baseline cargado del fichero externo: ponlo en cabecera para referencia
        ca = baseline["correct_ask_rate"] if baseline["correct_ask_rate"] != "" else "n/a"
        oa = baseline["over_ask_rate"] if baseline["over_ask_rate"] != "" else "n/a"
        print(f"  {baseline['variant']:<28}  {baseline['invention_rate']:>6.3f}  "
              f"{str(ca):>8}  {str(oa):>9}  {baseline['routing_accuracy']:>9.3f}")

    for v in variants:
        r = by_variant.get(v)
        if r is None:
            continue
        ca = r["correct_ask_rate"] if r["correct_ask_rate"] != "" else "n/a"
        oa = r["over_ask_rate"] if r["over_ask_rate"] != "" else "n/a"
        print(f"  {v:<28}  {r['invention_rate']:>6.3f}  {str(ca):>8}  "
              f"{str(oa):>9}  {r['routing_accuracy']:>9.3f}")
    print("-" * 78)

    if baseline is not None:
        print("  Delta frente a baseline (positivo = peor para 'inv', mejor para 'rout.acc'):")
        for v in variants:
            if v == "baseline":
                continue
            r = by_variant.get(v)
            if r is None:
                continue
            d_inv = r["invention_rate"] - baseline["invention_rate"]
            d_rout = r["routing_accuracy"] - baseline["routing_accuracy"]
            print(f"  {v:<28}  Δinv={d_inv:+.3f}  Δrout.acc={d_rout:+.3f}")
    print(sep)


def main() -> None:
    parser = argparse.ArgumentParser(description="Estudio de ablación del prompt.")
    parser.add_argument(
        "--model", default=os.getenv("OLLAMA_MODEL", "qwen2.5:3b-instruct-q4_K_M"),
    )
    parser.add_argument(
        "--temperature", type=float, default=float(os.getenv("OLLAMA_TEMPERATURE", "0.1")),
    )
    parser.add_argument("--reps", type=int, default=5)
    parser.add_argument(
        "--cases", nargs="+", default=None,
        help="Filtra por IDs (ej. --cases MP-01 TH-02).",
    )
    parser.add_argument(
        "--variants", nargs="+", choices=list(_VARIANTS), default=list(_VARIANTS),
        help="Variantes a ejecutar (default: las 4).",
    )
    parser.add_argument("--csv-out", type=Path, default=DEFAULT_SCORECARD)
    parser.add_argument("--summary-out", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument(
        "--observability", action="store_true",
        help="Activa el subsistema de observabilidad durante la corrida.",
    )
    args = parser.parse_args()

    commit_sha = _git_short_sha()
    # run_id compartido entre todas las variantes para que se identifique como
    # una sola sesión de ablación.
    rid = f"ablation-{args.model.replace(':', '_').replace('/', '_')}-{int(time.time())}"

    print(f"[ablation_eval] model={args.model} temperature={args.temperature} "
          f"reps={args.reps} commit={commit_sha}")
    print(f"[ablation_eval] variantes: {args.variants}")

    all_rows: list[dict] = []
    all_summary: list[dict] = []

    for variant_name in args.variants:
        ablation = _VARIANTS[variant_name]
        print(f"\n{'#' * 70}")
        print(f"#  variant = {variant_name}   ablation = {ablation}")
        print(f"{'#' * 70}")

        rows, _ = run_battery(
            model=args.model,
            temperature=args.temperature,
            reps=args.reps,
            case_filter=args.cases,
            ablation=ablation,
            scorecard_path=None,         # se omite la escritura por variante
            summary_path=None,
            run_id=rid,
            extra_row_fields={"variant": variant_name},
            observability=args.observability,
        )
        # Cada fila ya lleva 'variant' por extra_row_fields, pero por defensa:
        for r in rows:
            r.setdefault("variant", variant_name)

        variant_summary = compute_summary(rows, rid, args.model, args.temperature, commit_sha)
        for s in variant_summary:
            s["variant"] = variant_name

        all_rows.extend(rows)
        all_summary.extend(variant_summary)

    _write_combined_scorecard(all_rows, args.csv_out)
    print(f"\n  Scorecard escrito en {args.csv_out}")
    _write_combined_summary(all_summary, args.summary_out)
    print(f"  Summary escrito en   {args.summary_out}")

    _print_matrix(all_summary, args.variants)

    # Código de salida: 0 si baseline tiene routing_accuracy >= 0.6 (sanity).
    baseline = next(
        (r for r in all_summary if r["variant"] == "baseline" and r["case_type"] == "total"),
        None,
    )
    if baseline is None:
        print("\n[ablation_eval] aviso: no se ejecutó la variante baseline.")
        sys.exit(0)
    sys.exit(0 if baseline["routing_accuracy"] >= 0.6 else 1)


if __name__ == "__main__":
    main()

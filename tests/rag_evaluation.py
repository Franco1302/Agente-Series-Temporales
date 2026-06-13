"""Bateria de evaluacion de la recuperacion RAG (precision@k / recall@k / MRR).

Dataset de 20 preguntas de DECISION ("que metodo uso cuando..."): un chunk cuenta
como relevante (proxy lexico) si contiene alguno de los terminos esperados; el total
de relevantes se obtiene escaneando la coleccion Chroma. Las preguntas sin soporte
(total_relevant==0) se marcan SIN-CORPUS y se excluyen de las medias. Con --label la
corrida se persiste en docs/rag_evaluation/ (snapshot por pregunta + summary
antes/despues) y se compara contra --compare-to.

Uso:
    python -m tests.rag_evaluation --label baseline
    python -m tests.rag_evaluation --k 8 --label hybrid
"""

from __future__ import annotations

import argparse
import csv
import os
import re
import sys
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
EVAL_DIR = PROJECT_ROOT / "docs" / "rag_evaluation"
SUMMARY_CSV = EVAL_DIR / "summary.csv"
DEFAULT_K = 5

SUMMARY_FIELDS = [
    "label", "timestamp", "search_type", "k",
    "n_questions", "n_supported",
    "precision_at_k", "recall_at_k", "mrr",
]

# Dataset de evaluacion: (pregunta de decision, terminos de metodo esperados), lo mas distintivos posible.
EVAL_DATASET: list[tuple[str, tuple[str, ...]]] = [
    # --- Deteccion de drift ---------------------------------------------------
    ("Que test de drift uso con datos univariantes",
     ("kolmogorov", "jensen", "psi", "cusum")),
    ("Que metodo de deteccion de drift uso con datos multivariantes",
     ("mewma", "hotelling")),
    ("Tengo varias variables correlacionadas, que test de drift aplico",
     ("mewma", "hotelling")),
    ("Quiero detectar drift gradual y acumulado en una serie, que metodo uso",
     ("cusum", "mewma")),
    ("Cuando conviene Kolmogorov-Smirnov frente a Jensen-Shannon para drift",
     ("kolmogorov", "jensen")),
    ("Que test comparo distribuciones de probabilidad para detectar drift",
     ("jensen", "kolmogorov", "psi")),
    ("Necesito un indice de estabilidad de poblacion para medir drift",
     ("psi", "estabilidad")),
    ("Como detecto un cambio abrupto en la media de una senal",
     ("cusum",)),
    ("Diferencia entre deteccion de drift univariante y multivariante",
     ("univariante", "multivariante", "mewma", "hotelling")),
    ("Que prueba estadistica uso para drift en una sola variable continua",
     ("kolmogorov", "smirnov")),
    # --- Modelos de forecast --------------------------------------------------
    ("Que modelo de forecast uso si mi serie tiene estacionalidad fuerte",
     ("sarimax", "prophet")),
    ("Tengo regresores exogenos, que modelo de prediccion elijo",
     ("sarimax",)),
    ("Cuando conviene Prophet frente a SARIMAX",
     ("prophet", "sarimax")),
    ("Quiero forecasting autoregresivo, que modelo uso",
     ("forecaster", "autoreg")),
    ("Que modelo de prediccion de series temporales recomiendas",
     ("sarimax", "prophet", "forecaster")),
    # --- Estrategias de aumentacion ------------------------------------------
    ("Que estrategia de aumentacion de datos preserva la estructura armonica",
     ("harmonic", "armonic")),
    ("Quiero aumentar datos generando ruido con el metodo de Box-Muller",
     ("muller",)),
    ("Que estrategia de aumentacion uso para replicar muestras existentes",
     ("duplicate", "duplicad")),
    ("Como aumento datos preservando las propiedades estadisticas originales",
     ("statistical", "estadistic")),
    ("Que estrategias de aumentacion de datos existen",
     ("muller", "harmonic", "duplicate", "statistical")),
]


@dataclass
class QuestionResult:
    """Resultado de una pregunta del dataset de evaluacion."""

    question: str
    expected_terms: tuple[str, ...]
    total_relevant: int
    hits: int
    precision: float
    recall: float | None
    mrr: float

    @property
    def supported(self) -> bool:
        """True si el corpus contiene al menos un chunk relevante."""
        return self.total_relevant > 0


def _is_relevant(content: str, expected_terms: tuple[str, ...]) -> bool:
    """Proxy lexico: el chunk es relevante si contiene algun termino esperado."""
    lowered = content.lower()
    return any(term in lowered for term in expected_terms)


def _count_corpus_relevant(corpus: list[str], expected_terms: tuple[str, ...]) -> int:
    """Cuenta cuantos chunks de todo el corpus son relevantes para la pregunta."""
    return sum(1 for content in corpus if _is_relevant(content, expected_terms))


def _evaluate_question(
    question: str,
    expected_terms: tuple[str, ...],
    retrieved: list[str],
    corpus: list[str],
    k: int,
) -> QuestionResult:
    """Calcula precision@k, recall@k y MRR para una pregunta."""
    total_relevant = _count_corpus_relevant(corpus, expected_terms)
    relevance_flags = [_is_relevant(content, expected_terms) for content in retrieved]
    hits = sum(relevance_flags)

    precision = hits / k if k > 0 else 0.0
    recall = (hits / total_relevant) if total_relevant > 0 else None

    mrr = 0.0
    for rank, is_rel in enumerate(relevance_flags, start=1):
        if is_rel:
            mrr = 1.0 / rank
            break

    return QuestionResult(
        question=question,
        expected_terms=expected_terms,
        total_relevant=total_relevant,
        hits=hits,
        precision=precision,
        recall=recall,
        mrr=mrr,
    )


def _retrieve(recuperar_fn, query: str, k: int) -> list[str]:
    """Recupera los chunks para la query via hybrid.recuperar_documentos, que respeta RAG_SEARCH_TYPE (similarity | mmr | hybrid)."""
    documents = recuperar_fn(query, top_k=k)
    return [doc.page_content or "" for doc in documents]


def _load_corpus(vector_store) -> list[str]:
    """Devuelve el texto de todos los chunks persistidos en la coleccion."""
    data = vector_store.get(include=["documents"])
    documents = data.get("documents") or []
    return [text or "" for text in documents]


def _current_search_type() -> str:
    """Lee el modo de busqueda configurado (RAG_SEARCH_TYPE), default 'similarity'."""
    return (os.getenv("RAG_SEARCH_TYPE") or "similarity").strip().lower() or "similarity"


def _means(results: list[QuestionResult]) -> dict[str, float]:
    """Calcula las medias de las metricas sobre las preguntas con soporte."""
    supported = [r for r in results if r.supported]
    if not supported:
        return {"precision": 0.0, "recall": 0.0, "mrr": 0.0, "n": 0}
    n = len(supported)
    return {
        "precision": sum(r.precision for r in supported) / n,
        "recall": sum((r.recall or 0.0) for r in supported) / n,
        "mrr": sum(r.mrr for r in supported) / n,
        "n": n,
    }


def _emit_observability(results: list[QuestionResult], k: int) -> bool:
    """Registra un evento rag_retrieval por pregunta si la observabilidad esta activa; devuelve True si se emitieron. Cualquier fallo se ignora."""
    try:
        from src.observability import (
            EVENT_RAG_RETRIEVAL,
            TraceEvent,
            emit,
            get_thread_id,
            get_trace_id,
            is_enabled,
            new_span_id,
            start_turn,
        )

        if not is_enabled():
            return False

        start_turn("rag-evaluation")
        for result in results:
            emit(
                TraceEvent(
                    trace_id=get_trace_id(),
                    thread_id=get_thread_id(),
                    name="rag_eval.question",
                    event_type=EVENT_RAG_RETRIEVAL,
                    span_id=new_span_id(),
                    attributes={
                        "query": result.question,
                        "k": k,
                        "total_relevant": result.total_relevant,
                        "hits": result.hits,
                        "precision_at_k": round(result.precision, 4),
                        "recall_at_k": (
                            round(result.recall, 4) if result.recall is not None else None
                        ),
                        "mrr": round(result.mrr, 4),
                    },
                )
            )
        return True
    except Exception:  # noqa: BLE001
        return False


def _slug(label: str) -> str:
    """Normaliza una etiqueta para usarla como nombre de fichero seguro."""
    cleaned = re.sub(r"[^A-Za-z0-9_-]+", "-", label.strip()).strip("-")
    return cleaned or "sin-etiqueta"


def _write_snapshot(label: str, results: list[QuestionResult], k: int) -> Path:
    """Persiste el detalle por pregunta de una corrida etiquetada."""
    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    path = EVAL_DIR / f"snapshot_{_slug(label)}.csv"
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            ["question", "expected_terms", "k", "total_relevant", "hits",
             "precision_at_k", "recall_at_k", "mrr"]
        )
        for r in results:
            writer.writerow([
                r.question,
                ";".join(r.expected_terms),
                k,
                r.total_relevant,
                r.hits,
                f"{r.precision:.4f}",
                "" if r.recall is None else f"{r.recall:.4f}",
                f"{r.mrr:.4f}",
            ])
    return path


def _read_summary() -> list[dict[str, str]]:
    """Lee las filas del summary.csv (lista vacia si no existe)."""
    if not SUMMARY_CSV.exists():
        return []
    with SUMMARY_CSV.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _upsert_summary(
    label: str,
    results: list[QuestionResult],
    k: int,
    search_type: str,
) -> None:
    """Inserta o actualiza (por etiqueta) la fila de medias en summary.csv."""
    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    means = _means(results)
    new_row = {
        "label": label,
        "timestamp": datetime.now(UTC).isoformat(timespec="seconds"),
        "search_type": search_type,
        "k": str(k),
        "n_questions": str(len(results)),
        "n_supported": str(means["n"]),
        "precision_at_k": f"{means['precision']:.4f}",
        "recall_at_k": f"{means['recall']:.4f}",
        "mrr": f"{means['mrr']:.4f}",
    }

    rows = _read_summary()
    replaced = False
    for idx, row in enumerate(rows):
        if row.get("label") == label:
            rows[idx] = new_row
            replaced = True
            break
    if not replaced:
        rows.append(new_row)

    with SUMMARY_CSV.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=SUMMARY_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def _summary_row(label: str) -> dict[str, str] | None:
    """Devuelve la fila de summary.csv con la etiqueta indicada, si existe."""
    for row in _read_summary():
        if row.get("label") == label:
            return row
    return None


def _print_table(results: list[QuestionResult]) -> None:
    """Imprime la tabla por pregunta."""
    print(f"\n{'#':>2}  {'pregunta':<52} {'rel':>4} {'hit':>4} "
          f"{'P@k':>7} {'R@k':>7} {'MRR':>7}")
    print("-" * 90)
    for idx, r in enumerate(results, start=1):
        question = r.question if len(r.question) <= 52 else r.question[:49] + "..."
        recall_str = "  N/A  " if r.recall is None else f"{r.recall:7.3f}"
        flag = "" if r.supported else "  <- SIN-CORPUS"
        print(f"{idx:>2}  {question:<52} {r.total_relevant:>4} {r.hits:>4} "
              f"{r.precision:7.3f} {recall_str} {r.mrr:7.3f}{flag}")
    print("-" * 90)


def _print_summary(
    results: list[QuestionResult],
    k: int,
    compare_to: dict[str, str] | None,
) -> None:
    """Imprime las medias y, si hay corrida de referencia, el delta frente a ella."""
    means = _means(results)
    unsupported = [r for r in results if not r.supported]

    print(f"\nMEDIAS (k={k}, sobre {means['n']}/{len(results)} preguntas con soporte):")
    print(f"  precision@k : {means['precision']:.4f}")
    print(f"  recall@k    : {means['recall']:.4f}")
    print(f"  MRR         : {means['mrr']:.4f}")

    if unsupported:
        print(f"\n{len(unsupported)} pregunta(s) SIN-CORPUS (0 chunks relevantes, "
              f"excluidas de las medias):")
        for r in unsupported:
            print(f"  - {r.question}  [esperaba: {', '.join(r.expected_terms)}]")

    if compare_to is not None:
        ref_label = compare_to.get("label", "?")
        print(f"\nDELTA vs corrida '{ref_label}' "
              f"(search_type={compare_to.get('search_type', '?')}, "
              f"k={compare_to.get('k', '?')}):")
        for metric, key in (("precision", "precision_at_k"),
                            ("recall", "recall_at_k"),
                            ("mrr", "mrr")):
            try:
                ref_value = float(compare_to.get(key, ""))
            except (TypeError, ValueError):
                continue
            delta = means[metric] - ref_value
            sign = "+" if delta >= 0 else ""
            print(f"  {key:<13} : {ref_value:.4f} -> {means[metric]:.4f} "
                  f"({sign}{delta:.4f})")


def run_evaluation(
    k: int = DEFAULT_K,
    label: str | None = None,
    compare_to: str = "baseline",
) -> int:
    """Ejecuta la bateria completa de evaluacion RAG. Devuelve un codigo de salida."""
    try:
        from src.rag_engine.hybrid import recuperar_documentos
        from src.rag_engine.retriever import get_vector_store
    except Exception as exc:  # noqa: BLE001
        print(f"[ERROR] No se pudo importar el subsistema RAG: {exc}")
        return 1

    try:
        vector_store = get_vector_store()
        corpus = _load_corpus(vector_store)
    except Exception as exc:  # noqa: BLE001
        print(f"[ERROR] No se pudo abrir la base vectorial: {exc}")
        print("        Ejecuta primero: PYTHONPATH=. python src/rag_engine/ingest.py")
        return 1

    if not corpus:
        print("[ERROR] La coleccion Chroma esta vacia. Re-ejecuta la ingesta.")
        return 1

    search_type = _current_search_type()
    print(f"Corpus: {len(corpus)} chunks | dataset: {len(EVAL_DATASET)} preguntas "
          f"| k={k} | search_type={search_type}")

    results: list[QuestionResult] = []
    for question, expected_terms in EVAL_DATASET:
        try:
            retrieved = _retrieve(recuperar_documentos, question, k)
        except Exception as exc:  # noqa: BLE001
            print(f"[ERROR] Fallo recuperando '{question}': {exc}")
            return 1
        results.append(
            _evaluate_question(question, expected_terms, retrieved, corpus, k)
        )

    _print_table(results)

    # La referencia de comparacion no debe ser la propia corrida actual.
    reference = None if compare_to == label else _summary_row(compare_to)
    _print_summary(results, k, reference)

    emitted = _emit_observability(results, k)
    if emitted:
        print("\nObservabilidad activa: eventos 'rag_retrieval' registrados.")

    if label:
        snapshot_path = _write_snapshot(label, results, k)
        _upsert_summary(label, results, k, search_type)
        print(f"\nSnapshot '{label}' guardado:")
        print(f"  - detalle: {snapshot_path.relative_to(PROJECT_ROOT)}")
        print(f"  - medias : {SUMMARY_CSV.relative_to(PROJECT_ROOT)} (tabla antes/despues)")
    else:
        print("\n(corrida sin --label: no se ha persistido ningun snapshot)")

    return 0


def main() -> None:
    """Punto de entrada CLI."""
    parser = argparse.ArgumentParser(description="Evaluacion de recuperacion RAG.")
    parser.add_argument(
        "--k", type=int, default=DEFAULT_K,
        help=f"Numero de chunks recuperados por pregunta (def. {DEFAULT_K}).",
    )
    parser.add_argument(
        "--label", type=str, default=None,
        help="Etiqueta de la corrida; la persiste en docs/rag_evaluation/ "
             "(p. ej. baseline, mmr, hybrid).",
    )
    parser.add_argument(
        "--compare-to", type=str, default="baseline",
        help="Etiqueta de la corrida de referencia para el delta (def. baseline).",
    )
    args = parser.parse_args()

    if args.k <= 0:
        print("[ERROR] --k debe ser mayor que 0.")
        sys.exit(2)

    sys.exit(run_evaluation(k=args.k, label=args.label, compare_to=args.compare_to))


if __name__ == "__main__":
    main()

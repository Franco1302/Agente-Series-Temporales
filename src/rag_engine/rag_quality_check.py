"""Pruebas basicas de calidad para el flujo RAG (ejecutar: .venv/bin/python src/rag_engine/rag_quality_check.py)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.tools.rag_tool import consultar_teoria


@dataclass(frozen=True)
class RagTestCase:
    """Define un caso de prueba para evaluar recuperacion + sintesis."""

    name: str
    query: str
    expected_terms: list[str]


TEST_CASES: list[RagTestCase] = [
    RagTestCase(
        name="Drift y metodos estadisticos",
        query="Que es data drift y que metodos estadisticos menciona el TFG para detectarlo?",
        expected_terms=["kolmogorov", "cusum", "mewma", "drift"],
    ),
    RagTestCase(
        name="Series temporales sinteticas",
        query="Como aborda el TFG la generacion de series temporales sinteticas?",
        expected_terms=["sintet", "series temporales", "generacion", "autoregres"],
    ),
    RagTestCase(
        name="Modelos predictivos",
        query="Que modelos predictivos de series temporales se utilizan?",
        expected_terms=["sarimax", "prophet", "modelo", "predic"],
    ),
]


def _count_term_hits(text: str, expected_terms: list[str]) -> int:
    """Cuenta cuantas palabras esperadas aparecen en el texto."""
    lowered_text = text.lower()
    return sum(1 for term in expected_terms if term.lower() in lowered_text)


def run_quality_check() -> int:
    """Ejecuta pruebas y devuelve codigo de salida (0 ok, 1 fallos)."""
    fail_count = 0

    for idx, test_case in enumerate(TEST_CASES, start=1):
        print(f"\n[{idx}] Caso: {test_case.name}")
        print(f"Consulta: {test_case.query}")

        try:
            output = consultar_teoria.invoke({"query": test_case.query})
        except Exception as exc:
            print(f"Resultado: FAIL (excepcion inesperada: {exc})")
            fail_count += 1
            continue

        if output.startswith("Error:"):
            print(f"Resultado: FAIL (tool devolvio error: {output})")
            fail_count += 1
            continue

        if "Fuentes consultadas:" not in output:
            print("Resultado: FAIL (salida sin bloque de fuentes)")
            fail_count += 1
            continue

        hits = _count_term_hits(output, test_case.expected_terms)
        if hits == 0:
            print("Resultado: WARN (sin terminos esperados en la respuesta)")
            print("Vista previa:")
            print(output[:600])
            fail_count += 1
            continue

        print(f"Resultado: PASS (terminos detectados: {hits})")

    print("\nResumen de calidad RAG")
    print(f"Total casos: {len(TEST_CASES)}")
    print(f"Fallos: {fail_count}")

    return 1 if fail_count > 0 else 0


if __name__ == "__main__":
    raise SystemExit(run_quality_check())

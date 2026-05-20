"""Pruebas end-to-end del agente LangGraph con scoring ponderado.

Cada caso de prueba se puntua sobre 100 con cuatro criterios:
    - tool_correcta (40): el agente invoco la tool MCP esperada (o ninguna)
    - args_correctos (25): los argumentos JSON coinciden con lo esperado
    - terminos_clave (20): la respuesta natural contiene conceptos esperados
    - cita_artefactos (15): cuando la tool genera CSV/PNG, el resultado cita
      `output_path` / `image_path` o un path concreto en `data/temp_uploads/`

Si `genera_artefacto=False`, el criterio de artefactos no aplica y se suma
automaticamente. El total por caso siempre es sobre 100.

Ejecutar desde la raiz del proyecto:
    python -m scripts.test_agent                          # casos 1 y 2 por defecto
    python -m scripts.test_agent --caso 1                 # un caso concreto
    python -m scripts.test_agent --todos                  # todos los casos
    python -m scripts.test_agent --todos --csv-out tests/results/agent_scorecard.csv

Requiere: Ollama corriendo + backend MCP en localhost:8017 (las tools MCP se
ejecutan de verdad contra la API real).
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import json
import sys
import textwrap
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from src.observability import start_turn, emit, TraceEvent, EVENT_TURN_START, EVENT_TURN_END

from scripts._scoring_utils import comparar_args
from src.agent.graph import build_agent_graph


# ── Pesos del scoring por criterio ───────────────────────────────────────────

W_TOOL = 40
W_ARGS = 25
W_TERMS = 20
W_ARTIFACT = 15
W_TOTAL = W_TOOL + W_ARGS + W_TERMS + W_ARTIFACT  # 100


@dataclass
class TestCase:
    """Definicion de un caso de prueba del agente, con expectativas para scoring."""

    nombre: str
    mensaje: str
    csv_path: str | None = None
    # Tool MCP esperada; None si NO se debe invocar ninguna tool
    tool_esperada: str | None = None
    # Argumentos esperados (subset de lo que el LLM deberia incluir)
    args_esperados: dict[str, Any] = field(default_factory=dict)
    # Terminos que deben aparecer en la respuesta final (al menos uno)
    terminos_esperados: list[str] = field(default_factory=list)
    # Si True, la tool deberia generar un artefacto (CSV/PNG) que el agente cite
    genera_artefacto: bool = False


# ── Fixture de CSV real, generado una sola vez desde la propia API ────────────

_FIXTURE_CSV_PATH: str | None = None


def _build_fixture_csv() -> str:
    """Genera un CSV real con `generate_synthetic_distribution` y devuelve su path.

    Se llama a la tool MCP por importacion directa (sin pasar por el grafo),
    asi reutilizamos la logica real y evitamos duplicar generacion en numpy.
    """
    global _FIXTURE_CSV_PATH
    if _FIXTURE_CSV_PATH is not None and Path(_FIXTURE_CSV_PATH).exists():
        return _FIXTURE_CSV_PATH

    from mcp_server.tools.synthetic import generate_synthetic_distribution

    result = asyncio.run(
        generate_synthetic_distribution(
            start_date="2024-01-01",
            periods=200,
            frequency="D",
            distribution_type=1,  # Normal
            distribution_params=[0.0, 1.0],
            column_name="valor",
        )
    )
    if "error" in result:
        raise RuntimeError(f"No se pudo preparar el CSV de prueba: {result['error']}")
    _FIXTURE_CSV_PATH = result["output_path"]
    return _FIXTURE_CSV_PATH


def _casos() -> list[TestCase]:
    """Construye la lista de casos. Funcion para diferir la creacion del CSV."""
    csv_path = _build_fixture_csv()
    return [
        # 1. Drift con todos los parametros (camino feliz tool MCP)
        TestCase(
            nombre="Deteccion de drift con parametros completos",
            mensaje=(
                f"Analiza el drift en el fichero '{csv_path}' usando la columna "
                f"'Indice' como indice temporal, con el metodo KS y umbral 0.05."
            ),
            csv_path=csv_path,
            tool_esperada="detect_drift",
            args_esperados={
                "file_path": csv_path,
                "index_column": "Indice",
                "method": "KS",
                "threshold": 0.05,
            },
            terminos_esperados=["drift", "kolmogorov", "ks", "umbral", "0.05", "no detectado", "detectado"],
            genera_artefacto=False,
        ),
        # 2. Parametros incompletos -> el agente invoca la tool con args vacios
        #    y el grafo dispara solicitar_parametros (comportamiento descrito en
        #    src/agent/prompts/system_prompts.py BEHAVIOR_BLOCK).
        TestCase(
            nombre="Parametros incompletos - agente pide datos",
            mensaje="Genera una serie temporal sintetica.",
            csv_path=None,
            tool_esperada="generate_synthetic_distribution",
            args_esperados={},  # se espera args vacios o muy parciales
            terminos_esperados=[
                "start_date", "periods", "frequency", "distribution", "necesito", "proporcion", "datos",
            ],
            genera_artefacto=False,
        ),
        # 3. Pregunta general -> RAG (consultar_teoria es la tool esperada para
        #    preguntas teoricas; el prompt instruye SIEMPRE usarla).
        TestCase(
            nombre="Pregunta general sobre drift",
            mensaje="Que es el data drift y cuando debo preocuparme por el?",
            csv_path=None,
            tool_esperada="consultar_teoria",
            args_esperados={},
            terminos_esperados=["drift", "distribucion", "datos", "modelo", "cambio"],
            genera_artefacto=False,
        ),
        # 4. Augment - genera un CSV nuevo (artefacto)
        TestCase(
            nombre="Aumentacion normal de la serie",
            mensaje=(
                f"Aumenta el fichero '{csv_path}' con la estrategia 'normal' "
                f"anadiendo 50 nuevas observaciones, usando 'Indice' como indice "
                f"y frecuencia diaria."
            ),
            csv_path=csv_path,
            tool_esperada="augment_time_series",
            args_esperados={
                "file_path": csv_path,
                "index_column": "Indice",
                "strategy": "normal",
                "size": 50,
                "frequency": "D",
            },
            terminos_esperados=["augment", "aument", "filas", "csv", "observaciones"],
            genera_artefacto=True,
        ),
        # 5. RAG - consulta teorica
        TestCase(
            nombre="Consulta teorica RAG - Kolmogorov-Smirnov",
            mensaje="Que es el test de Kolmogorov-Smirnov y para que sirve en la deteccion de drift?",
            csv_path=None,
            tool_esperada="consultar_teoria",
            args_esperados={},  # query libre, solo verificamos que se llame la tool
            terminos_esperados=["kolmogorov", "drift", "distribu"],
            genera_artefacto=False,
        ),
        # 6. Recuperacion de error: file_path vacio -> el agente invoca
        #    detect_drift con file_path vacio y el validador dispara
        #    solicitar_parametros para pedir la ruta.
        TestCase(
            nombre="Recuperacion de error de parametros vacios",
            mensaje="Analiza el drift en el fichero '' sobre la columna 'valor'.",
            csv_path=None,
            tool_esperada="detect_drift",
            args_esperados={},
            terminos_esperados=["ruta", "fichero", "necesito", "proporcion", "file_path"],
            genera_artefacto=False,
        ),
        # 7. Generacion sintetica con grafica - artefacto PNG
        TestCase(
            nombre="Generacion sintetica con grafica (PNG)",
            mensaje=(
                "Genera una serie temporal sintetica con distribucion normal "
                "(mu=0, sigma=1) empezando el 2024-01-01, con 100 periodos diarios, "
                "y dame ademas la grafica."
            ),
            csv_path=None,
            tool_esperada="generate_synthetic_distribution",
            args_esperados={
                "start_date": "2024-01-01",
                "frequency": "D",
                "distribution_type": 1,
                "periods": 100,
                "with_plot": True,
            },
            terminos_esperados=["serie", "generad", "csv", "png", "grafica", "100"],
            genera_artefacto=True,
        ),
        # 8. Forecast SARIMAX - artefacto CSV
        TestCase(
            nombre="Forecast SARIMAX sobre la serie",
            mensaje=(
                f"Haz un forecast SARIMAX a 30 pasos sobre el fichero '{csv_path}' "
                f"usando 'Indice' como indice y 'valor' como columna objetivo, "
                f"con frecuencia diaria."
            ),
            csv_path=csv_path,
            tool_esperada="forecast_time_series",
            args_esperados={
                "file_path": csv_path,
                "index_column": "Indice",
                "target_column": "valor",
                "model": "sarimax",
                "forecast_steps": 30,
            },
            terminos_esperados=["forecast", "sarimax", "prediccion", "30", "csv"],
            genera_artefacto=True,
        ),
    ]


# ── Extraccion de invocaciones de tool desde el stream del grafo ──────────────


def _extract_tool_call_from_messages(messages: list) -> tuple[str | None, dict]:
    """Devuelve el (nombre, args) de la primera tool_call que aparezca, si la hay."""
    for msg in messages:
        if isinstance(msg, AIMessage):
            tool_calls = getattr(msg, "tool_calls", None) or []
            for call in tool_calls:
                if isinstance(call, dict):
                    return call.get("name"), call.get("args", {}) or {}
                return getattr(call, "name", None), getattr(call, "args", {}) or {}
    return None, {}


def _decode_tool_message_payload(content: Any) -> dict | None:
    """Devuelve el dict del ToolMessage soportando los dos formatos habituales.

    - LangChain nativo: `content` es un string JSON con el dict de la tool.
    - MCP (langchain_mcp_adapters): `content` es una lista de partes
      `[{"type": "text", "text": "{\"output_path\": ...}"}]`.
    """
    if isinstance(content, str):
        try:
            payload = json.loads(content)
        except (json.JSONDecodeError, TypeError):
            return None
        return payload if isinstance(payload, dict) else None

    if isinstance(content, list):
        for part in content:
            if not isinstance(part, dict):
                continue
            text = part.get("text")
            if not isinstance(text, str):
                continue
            try:
                payload = json.loads(text)
            except (json.JSONDecodeError, TypeError):
                continue
            if isinstance(payload, dict):
                return payload
    return None


def _artifacts_from_tool_messages(tool_messages: list[ToolMessage]) -> list[str]:
    """Extrae rutas concretas de artefactos (output_path / image_path) de los ToolMessage."""
    artifacts: list[str] = []
    for tm in tool_messages:
        payload = _decode_tool_message_payload(tm.content)
        if payload is None:
            continue
        for key in ("output_path", "image_path"):
            value = payload.get(key)
            if isinstance(value, str) and value:
                artifacts.append(value)
    return artifacts


def _response_cites_artifact(response: str, artifacts: list[str]) -> bool:
    """True si la respuesta cita literalmente uno de los paths de artefacto (o su nombre)."""
    if not response or not artifacts:
        return False
    response_l = response.lower()
    for path in artifacts:
        if not path:
            continue
        if path.lower() in response_l:
            return True
        # Tambien aceptamos que cite solo el nombre de fichero (los modelos a
        # veces resumen sin la ruta completa)
        name = Path(path).name.lower()
        if name and name in response_l:
            return True
    return False


# ── Ejecucion y scoring por caso ──────────────────────────────────────────────


@dataclass
class CaseScore:
    nombre: str
    tool_invocada: str | None
    score_tool: int
    score_args: int
    score_terms: int
    score_artifact: int
    total: int
    detalle: str  # mensaje corto explicando aciertos/fallos


def _score_case(caso: TestCase, verbose: bool = True) -> CaseScore:
    """Ejecuta el caso contra el grafo y calcula los 4 sub-scores."""
    separador = "-" * 60
    print(f"\n{separador}")
    print(f"CASO: {caso.nombre}")
    print(separador)
    print(f"Mensaje:  {textwrap.shorten(caso.mensaje, width=80)}")
    print(f"Fichero:  {caso.csv_path or '(ninguno)'}")
    print()

    build_agent_graph.cache_clear()
    graph = build_agent_graph()
    thread_id = f"score-{caso.nombre[:20].replace(' ', '-')}"
    config = {"configurable": {"thread_id": thread_id}}

    trace_id = start_turn(thread_id)
    emit(TraceEvent(
        trace_id=trace_id,
        thread_id=thread_id,
        name="test_case",
        event_type=EVENT_TURN_START
    ))

    input_state: dict[str, Any] = {
        "messages": [HumanMessage(content=caso.mensaje)],
        "csv_path": caso.csv_path,
        "error_count": 0,
    }

    final_response = ""
    tool_messages: list[ToolMessage] = []
    invoked_name: str | None = None
    invoked_args: dict = {}

    try:
        for event in graph.stream(input_state, config=config):
            node_name = next(iter(event))
            if node_name.startswith("__"):
                continue
            node_output = event[node_name] or {}
            messages_out: list = node_output.get("messages", []) or []

            if verbose:
                print(f"  -> Nodo: {node_name}")

            # Tool call la cogemos del primer AIMessage con tool_calls que veamos
            if invoked_name is None:
                cand_name, cand_args = _extract_tool_call_from_messages(messages_out)
                if cand_name is not None:
                    invoked_name, invoked_args = cand_name, cand_args
                    if verbose:
                        print(f"     Tool call: {invoked_name} args={invoked_args}")

            for msg in messages_out:
                if isinstance(msg, AIMessage):
                    tool_calls = getattr(msg, "tool_calls", None) or []
                    if not tool_calls and msg.content:
                        final_response = msg.content
                        if verbose:
                            preview = textwrap.shorten(str(msg.content), width=120)
                            print(f"     Respuesta: {preview}")
                elif isinstance(msg, ToolMessage):
                    tool_messages.append(msg)
                    if verbose:
                        preview = textwrap.shorten(str(msg.content), width=100)
                        print(f"     Resultado de {msg.name}: {preview}")
                        
        emit(TraceEvent(
            trace_id=trace_id,
            thread_id=thread_id,
            name="fin_test_case",
            event_type=EVENT_TURN_END,
            attributes={"status": "ok"}
        ))
    except (ConnectionError, RuntimeError) as exc:
        print(f"\n  ERROR DE CONEXION: {exc}")
        emit(TraceEvent(
            trace_id=trace_id,
            thread_id=thread_id,
            name="error_test_case",
            event_type=EVENT_TURN_END,
            attributes={"status": "error", "error": str(exc)}
        ))
        return CaseScore(caso.nombre, None, 0, 0, 0, 0, 0, f"ConnectionError: {exc}")
    except Exception as exc:
        print(f"\n  ERROR INESPERADO: {type(exc).__name__}: {exc}")
        emit(TraceEvent(
            trace_id=trace_id,
            thread_id=thread_id,
            name="error_test_case",
            event_type=EVENT_TURN_END,
            attributes={"status": "error", "error": str(exc)}
        ))
        return CaseScore(caso.nombre, None, 0, 0, 0, 0, 0, f"{type(exc).__name__}: {exc}")

    # ── Sub-score 1: tool correcta (40) ─────────────────────────────────────
    score_tool = W_TOOL if invoked_name == caso.tool_esperada else 0
    tool_detalle = f"tool={invoked_name!r} esperada={caso.tool_esperada!r}"

    # ── Sub-score 2: args correctos (25) ────────────────────────────────────
    if caso.args_esperados:
        aciertos, total = comparar_args(invoked_args, caso.args_esperados)
        # Damos los 25 pts solo si TODOS los args esperados coinciden
        score_args = W_ARGS if aciertos == total and total > 0 else int(round(W_ARGS * aciertos / total)) if total else 0
        args_detalle = f"args={aciertos}/{total}"
    else:
        # No hay args esperados (RAG libre / no-tool): se concede el peso entero
        score_args = W_ARGS
        args_detalle = "args=N/A"

    # ── Sub-score 3: terminos clave en respuesta (20) ───────────────────────
    response_l = (final_response or "").lower()
    encontrados = [t for t in caso.terminos_esperados if t.lower() in response_l]
    if caso.terminos_esperados:
        score_terms = W_TERMS if encontrados else 0
        terms_detalle = f"terms={len(encontrados)}/{len(caso.terminos_esperados)}"
    else:
        score_terms = W_TERMS
        terms_detalle = "terms=N/A"

    # ── Sub-score 4: cita artefactos (15) ───────────────────────────────────
    artifacts = _artifacts_from_tool_messages(tool_messages)
    if caso.genera_artefacto:
        if not artifacts:
            score_artifact = 0
            art_detalle = "artifact=NO GENERADO"
        elif _response_cites_artifact(final_response, artifacts):
            score_artifact = W_ARTIFACT
            art_detalle = "artifact=citado"
        else:
            score_artifact = 0
            art_detalle = f"artifact=no_citado ({len(artifacts)} disponibles)"
    else:
        score_artifact = W_ARTIFACT  # N/A -> auto-pasa
        art_detalle = "artifact=N/A"

    total = score_tool + score_args + score_terms + score_artifact
    detalle = " | ".join([tool_detalle, args_detalle, terms_detalle, art_detalle])

    print()
    print(f"  Tool   : {score_tool}/{W_TOOL}    ({tool_detalle})")
    print(f"  Args   : {score_args}/{W_ARGS}    ({args_detalle})")
    print(f"  Terms  : {score_terms}/{W_TERMS}    ({terms_detalle})")
    print(f"  Artif. : {score_artifact}/{W_ARTIFACT}    ({art_detalle})")
    print(f"  TOTAL  : {total}/{W_TOTAL}")

    return CaseScore(
        nombre=caso.nombre,
        tool_invocada=invoked_name,
        score_tool=score_tool,
        score_args=score_args,
        score_terms=score_terms,
        score_artifact=score_artifact,
        total=total,
        detalle=detalle,
    )


def _print_summary(resultados: list[CaseScore]) -> None:
    separador = "=" * 70
    print(f"\n{separador}")
    print("RESUMEN DE SCORING")
    print(separador)
    print(f"{'#':>2}  {'Caso':<45}  {'Tool':>5}  {'Args':>5}  {'Terms':>5}  {'Art':>4}  {'TOT':>4}")
    for i, r in enumerate(resultados, start=1):
        nombre = textwrap.shorten(r.nombre, width=45)
        print(
            f"{i:>2}  {nombre:<45}  {r.score_tool:>5}  {r.score_args:>5}  "
            f"{r.score_terms:>5}  {r.score_artifact:>4}  {r.total:>4}"
        )
    total_obt = sum(r.total for r in resultados)
    total_max = W_TOTAL * len(resultados)
    pct = (total_obt / total_max * 100.0) if total_max else 0.0
    print(separador)
    print(f"  Global: {total_obt}/{total_max} pts = {pct:.1f}%")
    print(separador)


def _write_scorecard_csv(out_path: Path, resultados: list[CaseScore]) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow([
            "caso", "tool_invocada",
            "score_tool", "score_args", "score_terms", "score_artifact",
            "total", "detalle",
        ])
        for r in resultados:
            writer.writerow([
                r.nombre, r.tool_invocada or "",
                r.score_tool, r.score_args, r.score_terms, r.score_artifact,
                r.total, r.detalle,
            ])
    total = sum(r.total for r in resultados)
    max_total = W_TOTAL * len(resultados)
    pct = (total / max_total * 100.0) if max_total else 0.0
    print(f"\n  Scorecard escrito en {out_path} ({total}/{max_total} = {pct:.1f}%)")


# ── CLI ───────────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Pruebas end-to-end del agente LangGraph con scoring ponderado."
    )
    grupo = parser.add_mutually_exclusive_group()
    grupo.add_argument(
        "--caso", type=int, metavar="N",
        help="Ejecuta un caso concreto (1-based).",
    )
    grupo.add_argument(
        "--todos", action="store_true",
        help="Ejecuta todos los casos de prueba.",
    )
    parser.add_argument(
        "--silencioso", action="store_true",
        help="Suprime el detalle de nodos y mensajes intermedios.",
    )
    parser.add_argument(
        "--csv-out", type=Path, default=None,
        help="Ruta de salida del scorecard CSV (ej. tests/results/agent_scorecard.csv).",
    )
    args = parser.parse_args()

    verbose = not args.silencioso

    casos = _casos()
    if args.caso:
        if not (1 <= args.caso <= len(casos)):
            parser.error(f"--caso debe estar entre 1 y {len(casos)}")
        casos_a_ejecutar = [casos[args.caso - 1]]
    elif args.todos:
        casos_a_ejecutar = casos
    else:
        casos_a_ejecutar = casos[:2]
        print(f"(Ejecutando casos 1 y 2 por defecto. Usa --todos para los {len(casos)} casos.)")

    resultados: list[CaseScore] = []
    for caso in casos_a_ejecutar:
        score = _score_case(caso, verbose=verbose)
        resultados.append(score)

    if len(resultados) > 1:
        _print_summary(resultados)

    if args.csv_out is not None:
        _write_scorecard_csv(args.csv_out, resultados)

    # Codigo de salida: 0 si total global >= 70%, 1 si no
    total = sum(r.total for r in resultados)
    max_total = W_TOTAL * len(resultados)
    pct = (total / max_total * 100.0) if max_total else 0.0
    sys.exit(0 if pct >= 70 else 1)


if __name__ == "__main__":
    main()

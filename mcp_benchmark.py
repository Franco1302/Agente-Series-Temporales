"""Banco de modelos Ollama para tool-calling sobre las herramientas MCP REALES.

Mide, por cada modelo y caso, si el LLM emite la tool call esperada con los
argumentos del usuario, midiendo además **invention_rate**: la fracción de
casos ambiguos en los que el modelo se inventa parámetros que el usuario no
mencionó.

------
  * ``resultados_benchmark.csv`` — una fila por (modelo, caso) con tiempo,
    tokens/s, decisión del LLM y si inventó parámetros.
  * Tabla en consola ordenada por score global, ahora con la columna
    ``invention rate``.

Uso
---
    python mcp_benchmark.py
    python mcp_benchmark.py --modelos qwen2.5:3b-instruct-q4_K_M
    python mcp_benchmark.py --casos 1 5   # acota TC-01 y TC-05
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path

try:
    import ollama
    from tabulate import tabulate
except ImportError:
    print("ERROR: Instala las dependencias del benchmark:")
    print("  pip install ollama tabulate")
    sys.exit(1)

# Necesario para que ``from src...`` funcione cuando se ejecuta como script
# desde la raíz del proyecto (sin PYTHONPATH=.). El propio CLAUDE.md aconseja
# PYTHONPATH=., pero el fichero se ha invocado tradicionalmente sin él.
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts._scoring_utils import evaluar_tool_call as _evaluar_tool_call


# ── Configuración ─────────────────────────────────────────────────────────────

OLLAMA_HOST = "http://localhost:11434"

MODELS = [
    "qwen2.5:3b-instruct-q4_K_M",
    "qwen2.5:7b-instruct-q4_K_M",
    "llama3.1:8b-instruct-q4_K_M",
]


# ── Construcción de las tool specs reales desde el MCP ─────────────────────


def _resolve_parameters(args_schema) -> dict:
    """Devuelve el JSON Schema de los parámetros de una BaseTool de LangChain.

    En este proyecto, ``langchain_mcp_adapters`` rellena ``args_schema`` con un
    dict (no con una clase Pydantic), porque el schema viene del lado MCP.
    Mantenemos compatibilidad con ambos formatos por robustez.
    """
    if args_schema is None:
        return {"type": "object", "properties": {}}
    if isinstance(args_schema, dict):
        return args_schema
    if hasattr(args_schema, "model_json_schema"):
        try:
            return args_schema.model_json_schema()
        except Exception:  # noqa: BLE001
            pass
    return {"type": "object", "properties": {}}


def build_mcp_tool_specs() -> list[dict]:
    """Carga las tools reales del agente (8 MCP + 1 RAG) y las convierte al
    formato que Ollama acepta en ``client.chat(tools=...)``.

    Importar ``AGENT_TOOLS`` arranca el subproceso MCP por stdio (igual que
    cuando arranca el grafo del agente). Es un coste de ~1-2 segundos
    aceptable para un script de benchmark.
    """
    from src.agent.tools import AGENT_TOOLS  # spawn lazy del MCP subprocess

    specs: list[dict] = []
    for t in AGENT_TOOLS:
        parameters = _resolve_parameters(getattr(t, "args_schema", None))
        description = (t.description or t.name).strip()
        # Ollama trunca descripciones muy largas; recortar a 1000 chars
        if len(description) > 1000:
            description = description[:1000].rstrip()
        specs.append({
            "type": "function",
            "function": {
                "name": t.name,
                "description": description,
                "parameters": parameters,
            },
        })
    return specs


# Las tool specs se construyen sólo una vez al iniciar el script
MCP_TOOLS: list[dict] = []  # se rellena en main()


# ── Casos de prueba contra las tools REALES ────────────────────────────────

# Para cada caso "happy path" damos `valores_esperados`: el evaluador medirá
# qué porcentaje del JSON coincide con los kwargs que el usuario escribió.
# Para los casos ambiguos `forbidden_invent` lista los parámetros que el
# usuario NO mencionó: si aparecen en la tool call, es invención.

TEST_CASES = [
    # ── Happy path ─────────────────────────────────────────────────────────
    {
        "id": "TC-01",
        "nombre": "Drift KS con argumentos completos",
        "prompt": (
            "Analiza el drift en el fichero '/tmp/ventas.csv' usando la columna "
            "'Indice' como índice temporal y el método KS, con umbral 0.05."
        ),
        "herramienta_esperada": "detect_drift",
        "args_requeridos": ["file_path", "index_column", "method"],
        "valores_esperados": {
            "file_path": "/tmp/ventas.csv",
            "index_column": "Indice",
            "method": "KS",
            "threshold": 0.05,
        },
        "forbidden_invent": [],
    },
    {
        "id": "TC-02",
        "nombre": "Generación sintética Normal con parámetros completos",
        "prompt": (
            "Genera una serie temporal sintética con distribución normal mu=0 "
            "sigma=1, empezando el 2024-01-01, frecuencia diaria, 200 períodos."
        ),
        "herramienta_esperada": "generate_synthetic_distribution",
        "args_requeridos": ["start_date", "frequency", "distribution_type", "distribution_params"],
        "valores_esperados": {
            "start_date": "2024-01-01",
            "frequency": "D",
            "distribution_type": 1,
            "periods": 200,
        },
        "forbidden_invent": [],
    },
    {
        "id": "TC-03",
        "nombre": "Augmentation normal con argumentos completos",
        "prompt": (
            "Aumenta el fichero '/tmp/ventas.csv' con la estrategia 'normal' "
            "añadiendo 50 nuevas observaciones, usando 'Indice' como índice "
            "y frecuencia diaria."
        ),
        "herramienta_esperada": "augment_time_series",
        "args_requeridos": ["file_path", "index_column", "strategy", "size", "frequency"],
        "valores_esperados": {
            "file_path": "/tmp/ventas.csv",
            "index_column": "Indice",
            "strategy": "normal",
            "size": 50,
            "frequency": "D",
        },
        "forbidden_invent": [],
    },
    {
        "id": "TC-04",
        "nombre": "Forecast SARIMAX con argumentos completos",
        "prompt": (
            "Haz un forecast SARIMAX a 30 pasos sobre el fichero '/tmp/ventas.csv', "
            "usando 'Indice' como índice y 'valor' como columna objetivo, frecuencia diaria."
        ),
        "herramienta_esperada": "forecast_time_series",
        "args_requeridos": ["file_path", "index_column", "target_column", "model", "forecast_steps"],
        "valores_esperados": {
            "file_path": "/tmp/ventas.csv",
            "index_column": "Indice",
            "target_column": "valor",
            "model": "sarimax",
            "forecast_steps": 30,
        },
        "forbidden_invent": [],
    },

    # ── Casos ambiguos: medir invention_rate ───────────────────────────────
    {
        "id": "TC-05",
        "nombre": "AMBIGUO — 'Detecta drift' sin parámetros",
        "prompt": "Detecta drift en mis datos.",
        "herramienta_esperada": "detect_drift",
        # No medimos args_requeridos en casos ambiguos: el agente NO debe pasar
        # parámetros (los pedirá el grafo). Marcamos la lista como vacía para
        # que `args_requeridos_presentes` valga True por convención y no
        # arrastre el score.
        "args_requeridos": [],
        "valores_esperados": {},
        "forbidden_invent": ["file_path", "index_column", "method", "threshold"],
    },
    {
        "id": "TC-06",
        "nombre": "AMBIGUO — 'Genera datos sintéticos' sin parámetros",
        "prompt": "Genera datos sintéticos.",
        # Hay 4 tools `generate_synthetic_*`; cualquier elección sin parámetros
        # exigidos cuenta como tool-correcta para no penalizar arbitrariamente.
        # Tomamos `generate_synthetic_distribution` como canónica (la más
        # frecuente) y, en consecuencia, lo importante de este caso es la
        # invención.
        "herramienta_esperada": "generate_synthetic_distribution",
        "args_requeridos": [],
        "valores_esperados": {},
        "forbidden_invent": [
            "start_date", "frequency", "distribution_type", "distribution_params",
            "periods", "end_date",
        ],
    },
    {
        "id": "TC-07",
        "nombre": "AMBIGUO — 'Aumenta los datos' sin parámetros",
        "prompt": "Aumenta los datos.",
        "herramienta_esperada": "augment_time_series",
        "args_requeridos": [],
        "valores_esperados": {},
        "forbidden_invent": ["file_path", "index_column", "strategy", "size", "frequency"],
    },
]


SYSTEM_PROMPT = (
    "Eres un asistente experto en análisis de series temporales y detección "
    "de drift. Cuando el usuario te pida realizar una operación, invoca la "
    "herramienta más apropiada. REGLA CRÍTICA: pasa SOLO los parámetros que "
    "el usuario haya escrito explícitamente en su mensaje y OMITE el resto. "
    "Mejor una tool call con arguments={} que con parámetros inventados."
)


# ── Evaluación ────────────────────────────────────────────────────────────────


def _extract_call_name_and_args(call) -> tuple[str | None, dict]:
    """Extrae (name, args) de un tool_call de Ollama, robusto frente a tipos."""
    nombre = getattr(call.function, "name", None) if hasattr(call, "function") else None

    args_raw = getattr(call.function, "arguments", {}) if hasattr(call, "function") else {}
    if isinstance(args_raw, str):
        try:
            args = json.loads(args_raw)
        except json.JSONDecodeError:
            args = {}
    elif isinstance(args_raw, dict):
        args = args_raw
    else:
        args = {}
    return nombre, args


def evaluar_tool_call(tool_calls, caso: dict) -> dict:
    """Devuelve métricas + flag de invención para un tool call de Ollama."""
    forbidden = caso.get("forbidden_invent") or []
    if not tool_calls:
        return {
            "herramienta_invocada": None,
            "herramienta_correcta": False,
            "args_requeridos_presentes": False,
            "precision_args_pct": 0.0,
            "inventado": False,                # sin tool call no hay invención
            "tool_call_args_json": "{}",
        }

    nombre, args = _extract_call_name_and_args(tool_calls[0])

    eval_dict = _evaluar_tool_call(
        nombre_invocado=nombre,
        args_obtenidos=args,
        nombre_esperado=caso["herramienta_esperada"],
        args_requeridos=caso["args_requeridos"],
        valores_esperados=caso["valores_esperados"],
    )

    invento = bool(forbidden) and any(k in args for k in forbidden)

    return {
        "herramienta_invocada": eval_dict["herramienta_invocada"],
        "herramienta_correcta": eval_dict["herramienta_correcta"],
        "args_requeridos_presentes": eval_dict["args_requeridos_presentes"],
        "precision_args_pct": eval_dict["precision_args_pct"],
        "inventado": invento,
        "tool_call_args_json": json.dumps(args, ensure_ascii=False),
    }


# ── Benchmark ─────────────────────────────────────────────────────────────────


def ejecutar_caso(client, modelo: str, caso: dict) -> dict:
    """Ejecuta un caso de prueba contra un modelo y retorna métricas."""
    mensajes = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": caso["prompt"]},
    ]

    inicio = time.perf_counter()
    error_msg = None
    response = None

    try:
        response = client.chat(
            model=modelo,
            messages=mensajes,
            tools=MCP_TOOLS,
            options={"temperature": 0.0},
        )
    except Exception as exc:  # noqa: BLE001
        error_msg = str(exc)

    duracion = round(time.perf_counter() - inicio, 3)

    if error_msg or response is None:
        return {
            "modelo": modelo,
            "caso_id": caso["id"],
            "caso_nombre": caso["nombre"],
            "tiempo_s": duracion,
            "tokens_por_s": 0.0,
            "herramienta_invocada": "ERROR",
            "herramienta_correcta": False,
            "json_valido": False,
            "args_requeridos_presentes": False,
            "precision_args_pct": 0.0,
            "inventado": False,
            "tool_call_args_json": "{}",
            "tiene_forbidden_invent": bool(caso.get("forbidden_invent")),
            "error": error_msg or "Sin respuesta",
        }

    # Tokens por segundo desde metadata de Ollama
    eval_count = getattr(response, "eval_count", 0) or 0
    eval_duration_ns = getattr(response, "eval_duration", 0) or 0
    tps = round(eval_count / (eval_duration_ns / 1e9), 2) if eval_duration_ns > 0 else 0.0

    tool_calls = getattr(response.message, "tool_calls", None)
    json_valido = bool(tool_calls)

    evaluacion = evaluar_tool_call(tool_calls, caso)

    return {
        "modelo": modelo,
        "caso_id": caso["id"],
        "caso_nombre": caso["nombre"],
        "tiempo_s": duracion,
        "tokens_por_s": tps,
        "herramienta_invocada": evaluacion["herramienta_invocada"],
        "herramienta_correcta": evaluacion["herramienta_correcta"],
        "json_valido": json_valido,
        "args_requeridos_presentes": evaluacion["args_requeridos_presentes"],
        "precision_args_pct": evaluacion["precision_args_pct"],
        "inventado": evaluacion["inventado"],
        "tool_call_args_json": evaluacion["tool_call_args_json"],
        "tiene_forbidden_invent": bool(caso.get("forbidden_invent")),
        "error": "",
    }


def verificar_modelo_disponible(client, modelo: str) -> bool:
    try:
        modelos_locales = [m.model for m in client.list().models]
        return any(modelo in m for m in modelos_locales)
    except Exception:  # noqa: BLE001
        return False


def calcular_score_global(resultados: list[dict]) -> dict[str, float]:
    """Score ponderado 40/30/30 por modelo (no se altera por la Tarea 4).

    score = herramienta_correcta × 40 + args_requeridos_presentes × 30 +
            precision_args_pct × 0.30
    """
    scores: dict[str, list[float]] = defaultdict(list)
    for r in resultados:
        if r["error"]:
            scores[r["modelo"]].append(0.0)
            continue
        s = (
            (40.0 if r["herramienta_correcta"] else 0.0)
            + (30.0 if r["args_requeridos_presentes"] else 0.0)
            + (r["precision_args_pct"] * 0.30)
        )
        scores[r["modelo"]].append(s)
    return {m: round(sum(v) / len(v), 1) for m, v in scores.items()}


def calcular_invention_rate(resultados: list[dict]) -> dict[str, float]:
    """Fracción de casos con ``forbidden_invent`` no vacía donde el modelo
    metió en la tool call alguno de los parámetros prohibidos.

    Solo entran al denominador los casos cuyo set ``forbidden_invent`` es no
    vacío (los demás no aportan información sobre invención). Si un modelo no
    tiene ninguno de esos casos en su corrida, devolvemos ``nan`` para que
    la celda salga vacía en la tabla.
    """
    by_model: dict[str, list[dict]] = defaultdict(list)
    for r in resultados:
        if r.get("tiene_forbidden_invent"):
            by_model[r["modelo"]].append(r)
    rates: dict[str, float] = {}
    for modelo, rs in by_model.items():
        if not rs:
            rates[modelo] = float("nan")
            continue
        n_inv = sum(1 for r in rs if r["inventado"])
        rates[modelo] = round(n_inv / len(rs), 3)
    return rates


# ── CLI ───────────────────────────────────────────────────────────────────────


def main() -> None:
    global MCP_TOOLS

    parser = argparse.ArgumentParser(description="Benchmark MCP tool calling (tools reales)")
    parser.add_argument(
        "--modelos", nargs="+", default=MODELS,
        help="Lista de modelos a evaluar (default: todos en MODELS)",
    )
    parser.add_argument(
        "--casos", nargs="+", type=int,
        help="Sufijos numéricos de los casos a ejecutar, ej: --casos 1 5 (default: todos)",
    )
    parser.add_argument(
        "--salida", default="resultados_benchmark.csv",
        help="Nombre del archivo CSV de salida",
    )
    args = parser.parse_args()

    # Carga las tool specs reales (arranca el subproceso MCP)
    print("[mcp_benchmark] cargando tool specs desde AGENT_TOOLS (MCP)…")
    MCP_TOOLS = build_mcp_tool_specs()
    print(f"[mcp_benchmark] {len(MCP_TOOLS)} tools cargadas: "
          f"{', '.join(t['function']['name'] for t in MCP_TOOLS)}")

    client = ollama.Client(host=OLLAMA_HOST)

    modelos_a_probar: list[str] = []
    for m in args.modelos:
        if verificar_modelo_disponible(client, m):
            modelos_a_probar.append(m)
        else:
            print(f"  [AVISO] Modelo no encontrado localmente, se omite: {m}")

    if not modelos_a_probar:
        print("ERROR: Ningún modelo disponible. Ejecuta setup_models.sh primero.")
        sys.exit(1)

    casos_a_probar = TEST_CASES
    if args.casos:
        casos_a_probar = [c for c in TEST_CASES if int(c["id"].split("-")[1]) in args.casos]

    print(f"\n{'=' * 70}")
    print(f"  MCP Tool Calling Benchmark  —  {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print(f"{'=' * 70}")
    print(f"  Modelos: {len(modelos_a_probar)}  |  Casos: {len(casos_a_probar)}  "
          f"|  Total iter: {len(modelos_a_probar) * len(casos_a_probar)}")
    print(f"{'=' * 70}\n")

    resultados: list[dict] = []
    total = len(modelos_a_probar) * len(casos_a_probar)
    idx = 0

    for modelo in modelos_a_probar:
        for caso in casos_a_probar:
            idx += 1
            print(f"[{idx:02d}/{total}] {modelo} | {caso['id']} — {caso['nombre']}")
            r = ejecutar_caso(client, modelo, caso)
            resultados.append(r)

            estado = "OK" if r["herramienta_correcta"] else "--"
            inv_marker = "  INV!" if r["inventado"] else ""
            print(
                f"        [{estado}] Tool: {r['herramienta_invocada'] or 'ninguna':<35} "
                f"{r['tiempo_s']}s | {r['tokens_por_s']} t/s | "
                f"prec.args: {r['precision_args_pct']}%{inv_marker}"
            )
            if r["error"]:
                print(f"        ERROR: {r['error']}")
        print()

    # ── CSV ───────────────────────────────────────────────────────────────────
    salida = Path(args.salida)
    campos = [
        "modelo", "caso_id", "caso_nombre", "tiempo_s", "tokens_por_s",
        "herramienta_invocada", "herramienta_correcta", "json_valido",
        "args_requeridos_presentes", "precision_args_pct",
        "inventado", "tiene_forbidden_invent", "tool_call_args_json", "error",
    ]
    with salida.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=campos)
        writer.writeheader()
        writer.writerows(resultados)
    print(f"Resultados guardados en: {salida.resolve()}\n")

    # ── Tabla resumen (40/30/30 intacto + columna invention rate) ─────────────
    scores = calcular_score_global(resultados)
    inventiones = calcular_invention_rate(resultados)

    resumen = []
    for modelo in modelos_a_probar:
        mrs = [r for r in resultados if r["modelo"] == modelo]
        n = len(mrs) or 1
        inv = inventiones.get(modelo, float("nan"))
        inv_str = "n/a" if inv != inv else f"{inv * 100:.1f}%"  # NaN check
        resumen.append([
            modelo,
            f"{sum(r['tiempo_s'] for r in mrs) / n:.2f}",
            f"{sum(r['tokens_por_s'] for r in mrs) / n:.1f}",
            f"{sum(1 for r in mrs if r['herramienta_correcta'])}/{n}",
            f"{sum(1 for r in mrs if r['args_requeridos_presentes'])}/{n}",
            f"{sum(r['precision_args_pct'] for r in mrs) / n:.1f}%",
            inv_str,
            f"{scores[modelo]}/100",
        ])

    resumen.sort(key=lambda x: float(x[-1].split("/")[0]), reverse=True)

    print(tabulate(
        resumen,
        headers=[
            "Modelo", "Tiempo medio(s)", "t/s medio",
            "Tool correcta", "Args presentes", "Precisión args",
            "Invention rate", "Score global",
        ],
        tablefmt="rounded_outline",
    ))

    ganador = resumen[0][0]
    print(f"\nModelo recomendado: {ganador}  (score: {resumen[0][-1]}, "
          f"invention rate: {resumen[0][-2]})\n")


if __name__ == "__main__":
    main()

"""Banco de modelos Ollama para tool-calling sobre las herramientas MCP REALES.

Mide cada modelo en **dos ejes** complementarios, porque la calidad real del
agente no se captura con un solo disparo:

* **Eje A — single-turn (modelo en crudo).** Una sola petición a Ollama con las
  tool specs reales. Mide si el modelo elige la tool correcta, extrae los args y,
  sobre todo, su **tendencia intrínseca a inventar** parámetros que el usuario no
  dio. Necesario: a veces el usuario lo da todo en un turno.

* **Eje B — casos de uso multiturno por el grafo real.** Cada CASO DE USO del
  Capítulo 5 (drift con RAG, generación sintética encadenada, aumentación +
  forecast SARIMAX) se corre como un GUION de turnos a través de
  ``build_agent_graph`` (con la API real), intercambiando el modelo por corrida.
  Mide lo que el usuario vive de verdad a lo largo de una conversación completa:
  turno a turno, ¿el agente **ejecuta** la tool esperada con los args fusionados
  y heredados, o **alucina** una respuesta en prosa sin ejecutar nada? Captura los
  fallos que el Eje A nunca ve: bypass de RAG, seguimiento heredante fallido,
  encadenamiento de artefactos roto.

La invención es protagonista en ambos ejes y **erosiona el score**:
  - Eje A: ``invention_rate`` (cuánto inventa el modelo solo).
  - Eje B: ``invention_leak`` (args inventados que SOBREVIVEN a la defensa
    ``_strip_invented_args`` y llegan a ejecutarse — debe ser 0).

Salida
------
  * ``data/benchmarks/resultados_benchmark.csv`` — una fila por caso del Eje A y una fila por
    TURNO PUNTUADO del Eje B (``<caso>·<tool>``), con el eje, las métricas y los
    flags de invención.
  * Tres tablas en consola: Eje A, Eje B y ranking combinado (lo decide el Eje B).

Uso
---
    python mcp_benchmark.py                       # ambos ejes, todos los modelos
    python mcp_benchmark.py --eje a               # solo single-turn
    python mcp_benchmark.py --eje b               # solo multiturno (necesita API)
    python mcp_benchmark.py --modelos qwen2.5:3b-instruct-q4_K_M
    python mcp_benchmark.py --casos 1 5           # acota TC-01 y TC-05 (Eje A)

Requisitos
----------
    Eje A: Ollama levantado.
    Eje B: Ollama + DRIFT_API (``DRIFT_API_URL``, default :8017) levantadas.
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import json
import os
import sys
import time
import uuid
from collections import defaultdict
from datetime import datetime
from functools import lru_cache
from pathlib import Path

try:
    import ollama
    from tabulate import tabulate
except ImportError:
    print("ERROR: Instala las dependencias del benchmark:")
    print("  pip install ollama tabulate")
    sys.exit(1)

# Necesario para que ``from src...`` funcione cuando se ejecuta como script
# desde la raíz del proyecto (sin PYTHONPATH=.).
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts._scoring_utils import comparar_args, evaluar_tool_call as _evaluar_tool_call


# ── Configuración ─────────────────────────────────────────────────────────────

OLLAMA_HOST = "http://localhost:11434"

MODELS = [
    "granite4.1:3b",
    "qwen2.5:3b-instruct-q4_K_M",
    "qwen2.5:3b-instruct-q8_0",
    "qwen2.5:7b-instruct-q4_K_M",
    "llama3.1:8b-instruct-q4_K_M",
]


# ── Pesos del score (AJÚSTALOS AQUÍ) ──────────────────────────────────────────
#
# Cada eje produce un sub-score base de 0..100; la invención lo erosiona como
# multiplicador (1 - tasa); el ranking final combina ambos ejes. Todo lo de
# abajo es un dial: cámbialo y vuelve a correr.

# Eje A (single-turn): reparto del base de cada caso.
W_A_TOOL = 50.0   # eligió la tool correcta
W_A_ARGS = 30.0   # los args obligatorios están presentes
W_A_PREC = 20.0   # % de valores correctos (precisión de args)

# Eje B (multiturno): reparto del base de cada escenario.
W_B_COMPLETION = 60.0   # la tool correcta se EJECUTÓ con los args obligatorios
W_B_NOHALLUC = 20.0     # NO alucinó un éxito sin ejecutar
W_B_PREC = 20.0         # % de valores correctos en la llamada ejecutada

# Combinación de ejes en el ranking final. El multiturno manda.
W_AXIS_A = 0.35
W_AXIS_B = 0.65

# La invención entra como multiplicador erosivo del base: score = base * (1-tasa).
# Con esto, un modelo que inventa en la mitad de los casos pierde la mitad del
# score; si inventa siempre, su score colapsa a 0. (Si lo quieres aún más duro,
# eleva la tasa a una potencia < 1 antes de multiplicar.)


# ── Eje A: construcción de las tool specs reales desde el MCP ──────────────────


def _resolve_parameters(args_schema) -> dict:
    """Devuelve el JSON Schema de los parámetros de una BaseTool de LangChain."""
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
    """Carga las tools reales del agente (8 MCP + 1 RAG) y las pasa al formato
    que Ollama acepta en ``client.chat(tools=...)``.

    Importar ``AGENT_TOOLS`` arranca el subproceso MCP por stdio (igual que el
    grafo del agente).
    """
    from src.agent.tools import AGENT_TOOLS  # spawn lazy del MCP subprocess

    specs: list[dict] = []
    for t in AGENT_TOOLS:
        parameters = _resolve_parameters(getattr(t, "args_schema", None))
        description = (t.description or t.name).strip()
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


# Las tool specs (Eje A) se construyen una sola vez en main()
MCP_TOOLS: list[dict] = []


# ── Eje A: casos de prueba single-turn ─────────────────────────────────────────

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
            "usando 'Indice' como índice, frecuencia diaria."
        ),
        "herramienta_esperada": "forecast_time_series",
        "args_requeridos": ["file_path", "index_column", "model", "forecast_steps"],
        "valores_esperados": {
            "file_path": "/tmp/ventas.csv",
            "index_column": "Indice",
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
        "args_requeridos": [],
        "valores_esperados": {},
        "forbidden_invent": ["file_path", "index_column", "method", "threshold"],
    },
    {
        "id": "TC-06",
        "nombre": "AMBIGUO — 'Genera datos sintéticos' sin parámetros",
        "prompt": "Genera datos sintéticos.",
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
    "NO INVENTES PARAMETROS, ¡ES MEJOR NO PASAR ARGUMENTOS QUE INVENTARLOS!"
)


# ── Eje A: evaluación ──────────────────────────────────────────────────────────


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
    """Devuelve métricas + flag de invención para un tool call de Ollama (Eje A)."""
    forbidden = caso.get("forbidden_invent") or []
    if not tool_calls:
        return {
            "herramienta_invocada": None,
            "herramienta_correcta": False,
            "args_requeridos_presentes": False,
            "precision_args_pct": 0.0,
            "inventado": False,
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


def ejecutar_caso(client, modelo: str, caso: dict) -> dict:
    """Ejecuta un caso single-turn (Eje A) contra un modelo y retorna métricas."""
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
        return _fila_a(modelo, caso, duracion, 0.0, error=error_msg or "Sin respuesta")

    eval_count = getattr(response, "eval_count", 0) or 0
    eval_duration_ns = getattr(response, "eval_duration", 0) or 0
    tps = round(eval_count / (eval_duration_ns / 1e9), 2) if eval_duration_ns > 0 else 0.0

    tool_calls = getattr(response.message, "tool_calls", None)
    evaluacion = evaluar_tool_call(tool_calls, caso)

    return _fila_a(modelo, caso, duracion, tps, evaluacion=evaluacion, json_valido=bool(tool_calls))


def _fila_a(modelo, caso, duracion, tps, evaluacion=None, json_valido=False, error="") -> dict:
    """Normaliza una fila del Eje A al esquema común del CSV."""
    ev = evaluacion or {}
    return {
        "eje": "A",
        "modelo": modelo,
        "caso_id": caso["id"],
        "caso_nombre": caso["nombre"],
        "tiempo_s": duracion,
        "tokens_por_s": tps,
        "herramienta_invocada": ev.get("herramienta_invocada") if not error else "ERROR",
        "herramienta_correcta": ev.get("herramienta_correcta", False),
        "json_valido": json_valido,
        "args_requeridos_presentes": ev.get("args_requeridos_presentes", False),
        "precision_args_pct": ev.get("precision_args_pct", 0.0),
        "inventado": ev.get("inventado", False),
        "tiene_forbidden_invent": bool(caso.get("forbidden_invent")),
        "tool_call_args_json": ev.get("tool_call_args_json", "{}"),
        # campos del Eje B (vacíos aquí)
        "herramienta_ejecutada": "",
        "completion_success": "",
        "hallucinated": "",
        "invention_leak": "",
        "rounds": "",
        "error": error,
    }


# ── Eje B: multiturno por el grafo real ────────────────────────────────────────

# Marcadores estables de los mensajes que emite solicitar_parametros_node. Sirven
# para distinguir "el agente está recogiendo params" (legítimo) de "el agente
# alucinó un éxito en prosa" (el fallo que cazamos).
_PARAM_REQUEST_MARKER = "Para continuar necesito algunos datos adicionales"
_OPTIONAL_CONFIRM_MARKER = "parámetros opcionales que aún no has indicado"
_CONFIRM_REPLY = "sí, usa los valores por defecto"


# Cada escenario reproduce un CASO DE USO completo del Capítulo 5 como un GUION
# de turnos (``script``) corrido por el grafo real, en un único thread (la
# memoria de sesión se acumula entre turnos). Cada turno es:
#   * de CONTEXTO (sin ``expect``): subida de CSV, pregunta teórica que dispara
#     una petición de parámetros, etc. Se envía pero no puntúa.
#   * PUNTUADO (con ``expect``): se espera que ``tool`` se ejecute con sus
#     ``required`` presentes; ``valores`` mide la precisión y ``forbidden`` la
#     fuga de invención. Genera una fila del Eje B.
#
# ``valores`` admite el centinela ``"$AUGMENTED$"`` (solo UC-03): el driver lo
# sustituye por la ruta real del CSV aumentado capturada del turno anterior,
# para medir si el forecast reutiliza el artefacto encadenado.
#
# ``csv_kind`` selecciona el fixture (ver ``_bootstrap_fixtures``): "drift" es
# una serie diaria; "monthly" una serie mensual (para augment+forecast SARIMAX).

MULTITURN_SCENARIOS = [
    {
        "id": "UC-01",
        "nombre": "Caso 1 · Drift: RAG + elicitación de umbral + ejecución",
        "needs_csv": True,
        "csv_kind": "drift",
        "script": [
            {"user": "He cargado un fichero con una serie temporal. ¿Lo tienes disponible?"},
            {"user": "¿Qué es el PSI?",
             "expect": {"tool": "consultar_teoria",
                        "required": ["query"], "valores": {}, "forbidden": []}},
            {"user": "Detecta drift con PSI."},
            {"user": "Usa los valores por defecto.",
             "expect": {"tool": "detect_drift",
                        "required": ["file_path", "index_column", "method"],
                        "valores": {"method": "PSI"}, "forbidden": []}},
        ],
    },
    {
        "id": "UC-02",
        "nombre": "Caso 2 · Sintético: distribución + tendencia reutilizando contexto",
        "needs_csv": False,
        "csv_kind": None,
        "script": [
            {"user": "Genera una serie sintética con distribución normal."},
            {"user": "Desde 2024-01-01, 365 periodos diarios, con media 50 y desviación 10.",
             "expect": {"tool": "generate_synthetic_distribution",
                        "required": ["start_date", "frequency",
                                     "distribution_type", "distribution_params"],
                        "valores": {"start_date": "2024-01-01", "frequency": "D",
                                    "distribution_type": 1},
                        "forbidden": ["end_date"]}},
            {"user": "Ahora otra igual pero con tendencia lineal de pendiente 0.5 e "
                     "intercepto 0, reutilizando los mismos parámetros.",
             "expect": {"tool": "generate_synthetic_trend",
                        "required": ["start_date", "frequency",
                                     "trend_type", "trend_params"],
                        "valores": {"start_date": "2024-01-01", "frequency": "D",
                                    "trend_type": 1},
                        "forbidden": ["end_date"]}},
        ],
    },
    {
        "id": "UC-03",
        "nombre": "Caso 3 · Aumentación + forecast SARIMAX encadenados",
        "needs_csv": True,
        "csv_kind": "monthly",
        "script": [
            {"user": "He subido mi serie de ventas mensuales."},
            {"user": "Amplía mi serie con 12 puntos más usando distribución normal "
                     "y frecuencia mensual.",
             "expect": {"tool": "augment_time_series",
                        "required": ["file_path", "index_column",
                                     "strategy", "size", "frequency"],
                        "valores": {"strategy": "normal", "size": 12, "frequency": "M"},
                        "forbidden": []}},
            {"user": "Ahora predice 6 pasos con SARIMAX, frecuencia mensual, sobre "
                     "los datos aumentados.",
             "expect": {"tool": "forecast_time_series",
                        "required": ["file_path", "index_column", "forecast_steps"],
                        "valores": {"model": "sarimax", "forecast_steps": 6,
                                    "frequency": "M", "file_path": "$AUGMENTED$"},
                        "forbidden": []}},
        ],
    },
]


def _swap_model(modelo: str) -> None:
    """Reapunta el agente al modelo dado para la siguiente corrida del grafo.

    El modelo sale de ``OLLAMA_MODEL`` y ``get_chat_ollama`` está cacheado, así
    que basta con reescribir el env y limpiar la caché: el grafo (también
    cacheado) recoge el cliente nuevo en el próximo nodo. Fijamos temperatura 0
    para reproducibilidad.
    """
    from src.config import llm_config

    os.environ["OLLAMA_MODEL"] = modelo
    os.environ["OLLAMA_TEMPERATURE"] = "0"
    llm_config.get_chat_ollama.cache_clear()


@lru_cache(maxsize=None)
def _schema_props(tool_name: str) -> dict:
    from src.agent.tools import AGENT_TOOLS

    tool = next((t for t in AGENT_TOOLS if t.name == tool_name), None)
    schema = getattr(tool, "args_schema", None) if tool else None
    if isinstance(schema, dict):
        return schema.get("properties", {}) or {}
    return {}


def _schema_default(tool_name: str, param: str):
    """Default declarado en el schema (None si no tiene). Se usa para no marcar
    como invención un valor que en realidad es el default legítimo del flujo."""
    return _schema_props(tool_name).get(param, {}).get("default")


def _make_fixture(
    start_date: str,
    frequency: str,
    periods: int,
    column_name: str,
) -> tuple[str | None, str | None]:
    """Genera un CSV fixture llamando a la propia tool sintética (API real).

    Devuelve ``(ruta, columna_indice)`` o ``(None, None)`` si falla (p. ej. la
    API está caída). Reutilizar la tool en vez de fabricar el CSV a mano sigue la
    norma del proyecto (los inputs de test salen del propio sistema).
    """
    from src.agent.tools import AGENT_TOOLS

    tool = next((t for t in AGENT_TOOLS if t.name == "generate_synthetic_distribution"), None)
    if tool is None:
        return None, None

    args = {
        "start_date": start_date,
        "frequency": frequency,
        "distribution_type": 1,
        "distribution_params": [50.0, 10.0],
        "periods": periods,
        "column_name": column_name,
        "with_plot": False,
    }
    try:
        raw = asyncio.run(tool.ainvoke(args))
    except Exception as exc:  # noqa: BLE001
        print(f"  [AVISO] No se pudo generar el CSV fixture (¿API caída?): {exc}")
        return None, None

    data = _coerce_tool_payload(raw)
    path = data.get("output_path") if isinstance(data, dict) else None
    if not path or not Path(path).exists():
        print(f"  [AVISO] La tool sintética no devolvió un output_path válido: {data}")
        return None, None

    return path, _first_csv_column(path)


def _bootstrap_fixtures() -> dict[str, tuple[str, str]]:
    """Construye los fixtures CSV con nombre que consumen los escenarios.

    Devuelve ``{csv_kind: (ruta, columna_indice)}`` solo con los que se
    generaron correctamente. "drift" es una serie diaria genérica (a detect_drift
    le basta un CSV válido); "monthly" es mensual, para que el forecast SARIMAX
    de UC-03 use ``frequency='M'`` coherente con los datos.
    """
    specs = {
        "drift": dict(start_date="2024-01-01", frequency="D", periods=300, column_name="valor"),
        "monthly": dict(start_date="2018-01-01", frequency="M", periods=72, column_name="ventas"),
    }
    fixtures: dict[str, tuple[str, str]] = {}
    for kind, spec in specs.items():
        path, index_col = _make_fixture(**spec)
        if path and index_col:
            fixtures[kind] = (path, index_col)
            print(f"  fixture '{kind}': {path}  (columna índice: {index_col})")
    return fixtures


def _csv_metadata(path: str | None) -> dict | None:
    """Metadata mínima del CSV (columnas + nº de filas) como hace la UI.

    El razonador la usa para validar ``index_column`` contra las columnas reales
    (``_check_existing_column``); sin ella el agente descartaría la columna que
    infiere y la pediría por texto, rompiendo el guion. ``dtypes`` se deja vacío:
    no interviene en esa validación.
    """
    if not path:
        return None
    try:
        with open(path, newline="", encoding="utf-8") as f:
            reader = csv.reader(f)
            header = next(reader)
            rows = sum(1 for _ in reader)
    except Exception:  # noqa: BLE001
        return None
    return {"columns": [str(c) for c in header], "rows": rows, "dtypes": {}}


def _coerce_tool_payload(raw):
    """Normaliza el retorno de una tool MCP (dict, str JSON o lista MCP) a dict."""
    data = raw
    if isinstance(data, str):
        try:
            data = json.loads(data)
        except json.JSONDecodeError:
            return {}
    if isinstance(data, list) and data and isinstance(data[0], dict):
        try:
            data = json.loads(data[0].get("text", "{}"))
        except json.JSONDecodeError:
            return {}
    return data if isinstance(data, dict) else {}


def _first_csv_column(path: str) -> str | None:
    try:
        with open(path, newline="", encoding="utf-8") as f:
            header = next(csv.reader(f))
        return header[0] if header else None
    except Exception:  # noqa: BLE001
        return None


def _last_ai_text(messages) -> str:
    """Texto del último AIMessage con contenido (la síntesis o la petición)."""
    from langchain_core.messages import AIMessage

    for m in reversed(messages):
        if isinstance(m, AIMessage):
            content = m.content
            if isinstance(content, list):
                content = " ".join(str(x) for x in content)
            if content and content.strip():
                return content
    return ""


def _is_param_request(text: str) -> bool:
    return _PARAM_REQUEST_MARKER in (text or "")


def _is_optional_confirmation(text: str) -> bool:
    return _OPTIONAL_CONFIRM_MARKER in (text or "")


def _executed_calls(messages) -> list[tuple[str, dict, bool]]:
    """Pares (tool_name, args, ok) de las tools que REALMENTE se ejecutaron.

    Una tool se ejecutó si hay un ToolMessage cuyo ``tool_call_id`` casa con el
    ``id`` de un tool_call emitido por un AIMessage. ``ok`` es False si el payload
    del ToolMessage trae una clave ``error`` (la API falló).
    """
    from langchain_core.messages import AIMessage, ToolMessage

    id_to_call: dict[str, tuple[str, dict]] = {}
    for m in messages:
        if isinstance(m, AIMessage):
            for tc in (m.tool_calls or []):
                cid = tc.get("id")
                if cid:
                    id_to_call[cid] = (tc.get("name", ""), tc.get("args", {}) or {})

    executed: list[tuple[str, dict, bool]] = []
    for m in messages:
        if isinstance(m, ToolMessage):
            cid = getattr(m, "tool_call_id", None)
            name, args = id_to_call.get(cid, (getattr(m, "name", "") or "", {}))
            executed.append((name, args, _toolmsg_ok(m)))
    return executed


def _toolmsg_ok(msg) -> bool:
    data = _coerce_tool_payload(getattr(msg, "content", None))
    if isinstance(data, dict):
        return not data.get("error")
    return True  # ilegible → asumimos que corrió


def _augment_output_path(messages) -> str | None:
    """Ruta del CSV generado por el último augment ejecutado, o None.

    Se usa para encadenar UC-03: el forecast debería operar sobre este artefacto,
    no sobre el CSV original.
    """
    from langchain_core.messages import ToolMessage

    for m in reversed(messages):
        if isinstance(m, ToolMessage) and getattr(m, "name", "") == "augment_time_series":
            data = _coerce_tool_payload(getattr(m, "content", None))
            path = data.get("output_path") if isinstance(data, dict) else None
            return path if path else None
    return None


def _advance(graph, config, user_text, active_csv, csv_meta, scored) -> tuple[dict | None, int, str]:
    """Envía un turno del usuario y devuelve ``(estado, rondas, error)``.

    Para un turno PUNTUADO (``scored``), si el agente responde con una
    confirmación de opcionales (umbral, ruido…) que el guion no contesta
    explícitamente, el driver auto-responde "usa los defaults" hasta 2 veces para
    que la tool llegue a ejecutarse. En turnos de contexto NO se auto-confirma:
    la confirmación la responde el siguiente turno del guion (p. ej. UC-01).
    """
    from langchain_core.messages import HumanMessage

    rounds = 0
    state = None
    next_text = user_text
    for _ in range(3):
        try:
            state = graph.invoke(
                {"messages": [HumanMessage(content=next_text)],
                 "csv_path": active_csv, "csv_metadata": csv_meta},
                config=config,
            )
        except Exception as exc:  # noqa: BLE001
            return state, rounds, str(exc)
        rounds += 1
        if not scored:
            break
        if _is_optional_confirmation(_last_ai_text(state.get("messages", []))):
            next_text = _CONFIRM_REPLY
            continue
        break
    return state, rounds, ""


def ejecutar_caso_uso(modelo: str, escenario: dict, fixtures: dict) -> list[dict]:
    """Corre un caso de uso (guion de turnos) por el grafo real y mide cada turno
    puntuado. Devuelve una fila del Eje B por turno con ``expect``.

    Un único thread acumula la memoria de sesión entre turnos, igual que la UI.
    Las ejecuciones se atribuyen al turno que las provocó comparando la lista de
    tools ejecutadas antes y después de cada turno.
    """
    from src.agent.graph import build_agent_graph

    graph = build_agent_graph()
    config = {"configurable": {"thread_id": uuid.uuid4().hex}}

    csv_path, index_col = fixtures.get(escenario.get("csv_kind")) if escenario.get("needs_csv") else (None, None)
    active_csv = csv_path
    csv_meta = _csv_metadata(csv_path)
    fmt = {"csv": active_csv or "", "index_col": index_col or "fecha"}

    rows: list[dict] = []
    prev_executed: list = []
    augmented_path: str | None = None
    rounds_acc = 0

    inicio = time.perf_counter()
    for step in escenario["script"]:
        scored = bool(step.get("expect"))
        state, rounds, error_msg = _advance(
            graph, config, step["user"].format(**fmt), active_csv, csv_meta, scored
        )
        rounds_acc += rounds

        messages = (state or {}).get("messages", [])
        executed = _executed_calls(messages)
        new_exec = executed[len(prev_executed):]
        prev_executed = executed

        if augmented_path is None:
            augmented_path = _augment_output_path(messages)

        if scored:
            duracion = round(time.perf_counter() - inicio, 3)
            rows.append(_fila_caso_uso(
                modelo, escenario, step, new_exec, messages,
                augmented_path, rounds_acc, duracion, error_msg,
            ))
            inicio = time.perf_counter()
        if error_msg:
            break

    return rows


def _fila_caso_uso(
    modelo, escenario, step, new_exec, messages, augmented_path, rounds, duracion, error_msg,
) -> dict:
    """Normaliza la métrica de un turno puntuado a la fila común del Eje B."""
    expect = step["expect"]
    expected = expect["tool"]
    required = expect["required"]
    forbidden = expect.get("forbidden") or []

    # Sustituye el centinela $AUGMENTED$ por la ruta encadenada real (o lo retira
    # si aún no hay augment ejecutado), para medir la reutilización del artefacto.
    valores = dict(expect["valores"])
    if valores.get("file_path") == "$AUGMENTED$":
        if augmented_path:
            valores["file_path"] = augmented_path
        else:
            valores.pop("file_path", None)

    exp_calls = [(a, ok) for n, a, ok in new_exec if n == expected]
    tool_executed = bool(exp_calls)
    exp_args, exp_ok = exp_calls[-1] if exp_calls else ({}, False)

    required_present = bool(exp_args) and all(k in exp_args for k in required)
    completion = tool_executed and required_present and exp_ok

    aciertos, total = comparar_args(exp_args, valores)
    precision = round((aciertos / total) * 100, 1) if total else (100.0 if completion else 0.0)

    leak = any(
        k in exp_args and exp_args[k] != _schema_default(expected, k) for k in forbidden
    )

    last_text = _last_ai_text(messages)
    hallucinated = (
        not tool_executed
        and bool(last_text.strip())
        and not _is_param_request(last_text)
        and not _is_optional_confirmation(last_text)
    )

    return {
        "eje": "B",
        "modelo": modelo,
        "caso_id": f"{escenario['id']}·{expected}",
        "caso_nombre": escenario["nombre"],
        "tiempo_s": duracion,
        "tokens_por_s": 0.0,
        "herramienta_invocada": (new_exec[-1][0] if new_exec else None),
        "herramienta_correcta": tool_executed,
        "json_valido": tool_executed,
        "args_requeridos_presentes": required_present,
        "precision_args_pct": precision,
        "inventado": leak,
        "tiene_forbidden_invent": bool(forbidden),
        "tool_call_args_json": json.dumps(exp_args, ensure_ascii=False),
        "herramienta_ejecutada": expected if tool_executed else "",
        "completion_success": completion,
        "hallucinated": hallucinated,
        "invention_leak": leak,
        "rounds": rounds,
        "error": error_msg,
    }


# ── Disponibilidad de modelos / API ────────────────────────────────────────────


def verificar_modelo_disponible(client, modelo: str) -> bool:
    try:
        modelos_locales = [m.model for m in client.list().models]
        return any(modelo in m for m in modelos_locales)
    except Exception:  # noqa: BLE001
        return False


def _api_disponible() -> bool:
    """Comprueba que DRIFT_API responde (necesaria para el Eje B)."""
    import urllib.request
    from mcp_server.config import load_settings

    url = load_settings().api_url
    for probe in ("/docs", "/openapi.json", "/"):
        try:
            with urllib.request.urlopen(url + probe, timeout=3) as resp:
                if getattr(resp, "status", 200) < 500:
                    return True
        except Exception:  # noqa: BLE001
            continue
    return False


# ── Scoring ────────────────────────────────────────────────────────────────────


def _rows_por_eje(resultados, eje):
    return [r for r in resultados if r["eje"] == eje]


def calcular_score_eje_a(resultados) -> dict[str, dict]:
    """Sub-score del Eje A por modelo: base (capacidad) erosionado por invención."""
    base_by: dict[str, list[float]] = defaultdict(list)
    inv_by: dict[str, list[float]] = defaultdict(list)
    for r in _rows_por_eje(resultados, "A"):
        if r["error"]:
            base_by[r["modelo"]].append(0.0)
        else:
            base = (
                (W_A_TOOL if r["herramienta_correcta"] else 0.0)
                + (W_A_ARGS if r["args_requeridos_presentes"] else 0.0)
                + (r["precision_args_pct"] * W_A_PREC / 100.0)
            )
            base_by[r["modelo"]].append(base)
        if r["tiene_forbidden_invent"]:
            inv_by[r["modelo"]].append(1.0 if r["inventado"] else 0.0)

    out: dict[str, dict] = {}
    for m, bases in base_by.items():
        base = sum(bases) / len(bases)
        inv = inv_by[m]
        rate = (sum(inv) / len(inv)) if inv else 0.0
        out[m] = {"base": round(base, 1), "inv_rate": round(rate, 3), "score": round(base * (1 - rate), 1)}
    return out


def calcular_score_eje_b(resultados) -> dict[str, dict]:
    """Sub-score del Eje B por modelo: base (completar + no-alucinar + precisión)
    erosionado por la fuga de invención."""
    base_by: dict[str, list[float]] = defaultdict(list)
    leak_by: dict[str, list[float]] = defaultdict(list)
    hall_by: dict[str, list[float]] = defaultdict(list)
    for r in _rows_por_eje(resultados, "B"):
        if r["error"]:
            base_by[r["modelo"]].append(0.0)
        else:
            base = (
                (W_B_COMPLETION if r["completion_success"] else 0.0)
                + (0.0 if r["hallucinated"] else W_B_NOHALLUC)
                + (r["precision_args_pct"] * W_B_PREC / 100.0)
            )
            base_by[r["modelo"]].append(base)
        leak_by[r["modelo"]].append(1.0 if r["invention_leak"] else 0.0)
        hall_by[r["modelo"]].append(1.0 if r["hallucinated"] else 0.0)

    out: dict[str, dict] = {}
    for m, bases in base_by.items():
        base = sum(bases) / len(bases)
        leak = leak_by[m]
        rate = (sum(leak) / len(leak)) if leak else 0.0
        hall = hall_by[m]
        hall_rate = (sum(hall) / len(hall)) if hall else 0.0
        out[m] = {
            "base": round(base, 1),
            "leak_rate": round(rate, 3),
            "halluc_rate": round(hall_rate, 3),
            "score": round(base * (1 - rate), 1),
        }
    return out


def calcular_score_final(score_a: dict, score_b: dict) -> dict[str, float]:
    """Combina los dos ejes. Si a un modelo le falta un eje, usa el disponible."""
    modelos = set(score_a) | set(score_b)
    final: dict[str, float] = {}
    for m in modelos:
        a = score_a.get(m, {}).get("score")
        b = score_b.get(m, {}).get("score")
        if a is not None and b is not None:
            final[m] = round(W_AXIS_A * a + W_AXIS_B * b, 1)
        elif a is not None:
            final[m] = round(a, 1)
        else:
            final[m] = round(b or 0.0, 1)
    return final


# ── CLI ─────────────────────────────────────────────────────────────────────────

_CSV_FIELDS = [
    "eje", "modelo", "caso_id", "caso_nombre", "tiempo_s", "tokens_por_s",
    "herramienta_invocada", "herramienta_correcta", "json_valido",
    "args_requeridos_presentes", "precision_args_pct", "inventado",
    "tiene_forbidden_invent", "tool_call_args_json",
    "herramienta_ejecutada", "completion_success", "hallucinated",
    "invention_leak", "rounds", "error",
]


def _run_eje_a(client, modelos, casos) -> list[dict]:
    print("\n── Eje A · single-turn (modelo en crudo) ─────────────────────────────")
    resultados: list[dict] = []
    total = len(modelos) * len(casos)
    idx = 0
    for modelo in modelos:
        for caso in casos:
            idx += 1
            print(f"[A {idx:02d}/{total}] {modelo} | {caso['id']} — {caso['nombre']}")
            r = ejecutar_caso(client, modelo, caso)
            resultados.append(r)
            estado = "OK" if r["herramienta_correcta"] else "--"
            inv = "  INV!" if r["inventado"] else ""
            print(
                f"        [{estado}] {r['herramienta_invocada'] or 'ninguna':<34} "
                f"{r['tiempo_s']}s | prec.args {r['precision_args_pct']}%{inv}"
            )
            if r["error"]:
                print(f"        ERROR: {r['error']}")
        print()
    return resultados


def _run_eje_b(modelos, escenarios) -> list[dict]:
    print("\n── Eje B · casos de uso multiturno por el grafo real (API real) ───────")
    fixtures = _bootstrap_fixtures()

    usables = [
        s for s in escenarios
        if not s.get("needs_csv") or s.get("csv_kind") in fixtures
    ]
    omitidos = [s["id"] for s in escenarios if s not in usables]
    if omitidos:
        print(f"  [AVISO] Sin fixture: se omiten {', '.join(omitidos)}.")

    resultados: list[dict] = []
    total = len(modelos) * len(usables)
    idx = 0
    for modelo in modelos:
        _swap_model(modelo)
        for esc in usables:
            idx += 1
            print(f"[B {idx:02d}/{total}] {modelo} | {esc['id']} — {esc['nombre']}")
            filas = ejecutar_caso_uso(modelo, esc, fixtures)
            resultados.extend(filas)
            for r in filas:
                estado = "OK" if r["completion_success"] else "--"
                flags = ""
                if r["hallucinated"]:
                    flags += "  ALUCINÓ!"
                if r["invention_leak"]:
                    flags += "  LEAK!"
                print(
                    f"        [{estado}] {r['caso_id']:<40} "
                    f"prec {r['precision_args_pct']}%{flags}"
                )
                if r["error"]:
                    print(f"        ERROR: {r['error']}")
        print()
    return resultados


def _tabla_eje_a(resultados, modelos, scores) -> None:
    rows = []
    for m in modelos:
        mrs = _rows_por_eje([r for r in resultados if r["modelo"] == m], "A")
        if not mrs:
            continue
        n = len(mrs)
        s = scores.get(m, {})
        rows.append([
            m,
            f"{sum(1 for r in mrs if r['herramienta_correcta'])}/{n}",
            f"{sum(1 for r in mrs if r['args_requeridos_presentes'])}/{n}",
            f"{sum(r['precision_args_pct'] for r in mrs) / n:.1f}%",
            f"{s.get('inv_rate', 0) * 100:.1f}%",
            f"{s.get('base', 0)}",
            f"{s.get('score', 0)}",
        ])
    rows.sort(key=lambda x: float(x[-1]), reverse=True)
    print("\nEJE A · single-turn")
    print(tabulate(
        rows,
        headers=["Modelo", "Tool correcta", "Args present.", "Precisión",
                 "Invent. rate", "Base", "Score A"],
        tablefmt="rounded_outline",
    ))


def _tabla_eje_b(resultados, modelos, scores) -> None:
    rows = []
    for m in modelos:
        mrs = _rows_por_eje([r for r in resultados if r["modelo"] == m], "B")
        if not mrs:
            continue
        n = len(mrs)
        s = scores.get(m, {})
        rows.append([
            m,
            f"{sum(1 for r in mrs if r['completion_success'])}/{n}",
            f"{sum(1 for r in mrs if r['hallucinated'])}/{n}",
            f"{sum(1 for r in mrs if r['invention_leak'])}/{n}",
            f"{sum(r['precision_args_pct'] for r in mrs) / n:.1f}%",
            f"{s.get('base', 0)}",
            f"{s.get('score', 0)}",
        ])
    rows.sort(key=lambda x: float(x[-1]), reverse=True)
    print("\nEJE B · multiturno (grafo real)")
    print(tabulate(
        rows,
        headers=["Modelo", "Completó", "Alucinó", "Invent. LEAK", "Precisión",
                 "Base", "Score B"],
        tablefmt="rounded_outline",
    ))


def _tabla_final(modelos, score_a, score_b, final) -> None:
    rows = []
    for m in modelos:
        if m not in final:
            continue
        rows.append([
            m,
            f"{score_a.get(m, {}).get('score', '—')}",
            f"{score_b.get(m, {}).get('score', '—')}",
            f"{final[m]}",
        ])
    rows.sort(key=lambda x: float(x[-1]), reverse=True)
    print(f"\nRANKING FINAL  (={W_AXIS_A:g}·A + {W_AXIS_B:g}·B)")
    print(tabulate(
        rows,
        headers=["Modelo", "Score A", "Score B", "FINAL"],
        tablefmt="rounded_outline",
    ))
    if rows:
        print(f"\nModelo recomendado: {rows[0][0]}  (final: {rows[0][-1]}/100)\n")


def main() -> None:
    global MCP_TOOLS

    parser = argparse.ArgumentParser(description="Benchmark MCP tool calling (2 ejes)")
    parser.add_argument("--modelos", nargs="+", default=MODELS,
                        help="Modelos a evaluar (default: todos en MODELS)")
    parser.add_argument("--casos", nargs="+", type=int,
                        help="Sufijos de los casos del Eje A, ej: --casos 1 5")
    parser.add_argument("--eje", choices=["a", "b", "ambos"], default="ambos",
                        help="Qué eje(s) correr (default: ambos)")
    parser.add_argument("--salida", default="data/benchmarks/resultados_benchmark.csv",
                        help="CSV de salida")
    args = parser.parse_args()

    print("[mcp_benchmark] cargando tool specs desde AGENT_TOOLS (MCP)…")
    MCP_TOOLS = build_mcp_tool_specs()
    print(f"[mcp_benchmark] {len(MCP_TOOLS)} tools: "
          f"{', '.join(t['function']['name'] for t in MCP_TOOLS)}")

    client = ollama.Client(host=OLLAMA_HOST)

    modelos = [m for m in args.modelos if verificar_modelo_disponible(client, m)]
    for m in args.modelos:
        if m not in modelos:
            print(f"  [AVISO] Modelo no encontrado localmente, se omite: {m}")
    if not modelos:
        print("ERROR: Ningún modelo disponible.")
        sys.exit(1)

    casos = TEST_CASES
    if args.casos:
        casos = [c for c in TEST_CASES if int(c["id"].split("-")[1]) in args.casos]

    correr_a = args.eje in ("a", "ambos")
    correr_b = args.eje in ("b", "ambos")

    if correr_b and not _api_disponible():
        print("\n  [AVISO] DRIFT_API no responde: las métricas del Eje B no serán "
              "fiables (completar fallará). Levanta la API o usa --eje a.")

    print(f"\n{'=' * 70}")
    print(f"  MCP Tool Calling Benchmark  —  {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print(f"  Modelos: {len(modelos)}  |  Ejes: {args.eje}")
    print(f"{'=' * 70}")

    resultados: list[dict] = []
    if correr_a:
        resultados += _run_eje_a(client, modelos, casos)
    if correr_b:
        resultados += _run_eje_b(modelos, MULTITURN_SCENARIOS)

    # ── CSV ───────────────────────────────────────────────────────────────────
    salida = Path(args.salida)
    with salida.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=_CSV_FIELDS)
        writer.writeheader()
        for r in resultados:
            writer.writerow({k: r.get(k, "") for k in _CSV_FIELDS})
    print(f"Resultados guardados en: {salida.resolve()}")

    # ── Tablas ────────────────────────────────────────────────────────────────
    score_a = calcular_score_eje_a(resultados) if correr_a else {}
    score_b = calcular_score_eje_b(resultados) if correr_b else {}

    if correr_a:
        _tabla_eje_a(resultados, modelos, score_a)
    if correr_b:
        _tabla_eje_b(resultados, modelos, score_b)
    if correr_a and correr_b:
        final = calcular_score_final(score_a, score_b)
        _tabla_final(modelos, score_a, score_b, final)


if __name__ == "__main__":
    main()

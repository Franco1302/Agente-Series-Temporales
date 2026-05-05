"""
Banco de pruebas para comparar modelos Ollama en escenarios de tool calling MCP.

Salida: resultados_benchmark.csv + tabla resumen en consola.
Uso:    python mcp_benchmark.py [--modelos modelo1 modelo2 ...]
"""

import argparse
import csv
import json
import sys
import time
from datetime import datetime
from pathlib import Path

try:
    import ollama
    from tabulate import tabulate
except ImportError:
    print("ERROR: Instala las dependencias del benchmark:")
    print("  pip install -r requirements_benchmark.txt")
    sys.exit(1)


# ── Configuración ─────────────────────────────────────────────────────────────

OLLAMA_HOST = "http://localhost:11434"

MODELS = [
    "qwen2.5:7b-instruct-q4_K_M",
    "llama3.1:8b-instruct-q4_K_M",
    "qwen2.5-coder:7b-instruct-q4_K_M",
    "mistral:7b-instruct-q4_K_M",
]

# ── Herramientas MCP simuladas ────────────────────────────────────────────────

MCP_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "detectar_drift_ks",
            "description": (
                "Detecta drift estadístico entre dos distribuciones de datos "
                "usando el test de Kolmogorov-Smirnov. Retorna el p-valor y si "
                "se detectó drift significativo."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "serie_referencia": {
                        "type": "string",
                        "description": "Ruta al archivo CSV con los datos históricos de referencia.",
                    },
                    "serie_actual": {
                        "type": "string",
                        "description": "Ruta al archivo CSV con los datos actuales a evaluar.",
                    },
                    "columna": {
                        "type": "string",
                        "description": "Nombre de la columna numérica a analizar.",
                    },
                    "umbral_pvalor": {
                        "type": "number",
                        "description": "Umbral de p-valor para considerar drift (default: 0.05).",
                    },
                },
                "required": ["serie_referencia", "serie_actual", "columna"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "generar_serie_sintetica",
            "description": (
                "Genera una serie temporal sintética con la distribución "
                "estadística especificada."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "distribucion": {
                        "type": "string",
                        "enum": ["normal", "uniforme", "poisson"],
                        "description": "Tipo de distribución estadística.",
                    },
                    "n_puntos": {
                        "type": "integer",
                        "description": "Número de puntos de datos a generar.",
                    },
                    "semilla": {
                        "type": "integer",
                        "description": "Semilla para reproducibilidad (opcional).",
                    },
                },
                "required": ["distribucion", "n_puntos"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "aumentar_datos_relacion_lineal",
            "description": (
                "Agrega columnas derivadas a un CSV usando relaciones lineales "
                "configurables entre columnas existentes."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "archivo_csv": {
                        "type": "string",
                        "description": "Ruta al archivo CSV de entrada.",
                    },
                    "columna_origen": {
                        "type": "string",
                        "description": "Columna fuente para calcular la relación lineal.",
                    },
                    "nombre_nueva_columna": {
                        "type": "string",
                        "description": "Nombre de la nueva columna a crear.",
                    },
                    "factor": {
                        "type": "number",
                        "description": "Factor multiplicativo de la relación lineal.",
                    },
                    "offset": {
                        "type": "number",
                        "description": "Desplazamiento aditivo (default: 0).",
                    },
                },
                "required": ["archivo_csv", "columna_origen", "nombre_nueva_columna", "factor"],
            },
        },
    },
]

# ── Casos de prueba ───────────────────────────────────────────────────────────

TEST_CASES = [
    {
        "id": "TC-01",
        "nombre": "Drift detection - args completos",
        "prompt": (
            "Necesito que analices si existe drift estadístico entre los datos "
            "del archivo 'ventas_enero.csv' y 'ventas_febrero.csv'. "
            "Concretamente en la columna 'precio'. Usa un umbral de p-valor de 0.05."
        ),
        "herramienta_esperada": "detectar_drift_ks",
        "args_requeridos": ["serie_referencia", "serie_actual", "columna"],
        "valores_esperados": {
            "serie_referencia": "ventas_enero.csv",
            "serie_actual": "ventas_febrero.csv",
            "columna": "precio",
        },
    },
    {
        "id": "TC-02",
        "nombre": "Generación sintética - args mínimos",
        "prompt": (
            "Genera una serie temporal sintética de 500 puntos con distribución normal."
        ),
        "herramienta_esperada": "generar_serie_sintetica",
        "args_requeridos": ["distribucion", "n_puntos"],
        "valores_esperados": {
            "distribucion": "normal",
            "n_puntos": 500,
        },
    },
    {
        "id": "TC-03",
        "nombre": "Aumento de datos - relación lineal",
        "prompt": (
            "Toma el archivo 'dataset.csv' y agrega una nueva columna llamada "
            "'precio_iva' que sea la columna 'precio_base' multiplicada por 1.21."
        ),
        "herramienta_esperada": "aumentar_datos_relacion_lineal",
        "args_requeridos": ["archivo_csv", "columna_origen", "nombre_nueva_columna", "factor"],
        "valores_esperados": {
            "archivo_csv": "dataset.csv",
            "columna_origen": "precio_base",
            "nombre_nueva_columna": "precio_iva",
            "factor": 1.21,
        },
    },
    {
        "id": "TC-04",
        "nombre": "Herramienta correcta entre ambigüedad",
        "prompt": (
            "Quiero comparar la distribución estadística de 'temperaturas_2023.csv' "
            "contra 'temperaturas_2024.csv' en la columna 'temp_max' para ver si "
            "hay cambios significativos."
        ),
        "herramienta_esperada": "detectar_drift_ks",
        "args_requeridos": ["serie_referencia", "serie_actual", "columna"],
        "valores_esperados": {
            "serie_referencia": "temperaturas_2023.csv",
            "serie_actual": "temperaturas_2024.csv",
            "columna": "temp_max",
        },
    },
]

SYSTEM_PROMPT = (
    "Eres un asistente experto en análisis de datos y detección de drift estadístico. "
    "Cuando el usuario te pida realizar una operación, usa SIEMPRE la herramienta más "
    "apropiada disponible. No respondas en texto plano si hay una herramienta aplicable. "
    "Extrae con precisión los parámetros del mensaje del usuario."
)


# ── Evaluación ────────────────────────────────────────────────────────────────

def _normalizar(valor):
    """Normaliza valores para comparación flexible (str/int/float)."""
    if isinstance(valor, str):
        return valor.strip().lower()
    return valor


def evaluar_tool_call(tool_calls, caso: dict) -> dict:
    """Evalúa si el tool call cumple los criterios del caso de prueba."""
    if not tool_calls:
        return {
            "herramienta_invocada": None,
            "herramienta_correcta": False,
            "args_requeridos_presentes": False,
            "precision_args_pct": 0.0,
        }

    # Tomar el primer tool call (ReAct espera uno por turno)
    call = tool_calls[0]
    nombre_tool = getattr(call.function, "name", None)

    # Extraer argumentos
    args_raw = getattr(call.function, "arguments", {})
    if isinstance(args_raw, str):
        try:
            args = json.loads(args_raw)
        except json.JSONDecodeError:
            args = {}
    else:
        args = args_raw or {}

    herramienta_correcta = nombre_tool == caso["herramienta_esperada"]

    # Verificar args requeridos presentes
    requeridos = caso["args_requeridos"]
    presentes = all(k in args for k in requeridos)

    # Precisión de valores
    esperados = caso["valores_esperados"]
    aciertos = sum(
        1 for k, v in esperados.items()
        if k in args and _normalizar(args[k]) == _normalizar(v)
    )
    precision = round((aciertos / len(esperados)) * 100, 1) if esperados else 0.0

    return {
        "herramienta_invocada": nombre_tool,
        "herramienta_correcta": herramienta_correcta,
        "args_requeridos_presentes": presentes,
        "precision_args_pct": precision,
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
    except Exception as exc:
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
            "error": error_msg or "Sin respuesta",
        }

    # Tokens por segundo desde metadata de Ollama
    eval_count = getattr(response, "eval_count", 0) or 0
    eval_duration_ns = getattr(response, "eval_duration", 0) or 0
    tps = round(eval_count / (eval_duration_ns / 1e9), 2) if eval_duration_ns > 0 else 0.0

    tool_calls = getattr(response.message, "tool_calls", None)
    json_valido = tool_calls is not None and len(tool_calls) > 0

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
        "error": "",
    }


def verificar_modelo_disponible(client, modelo: str) -> bool:
    try:
        modelos_locales = [m.model for m in client.list().models]
        return any(modelo in m for m in modelos_locales)
    except Exception:
        return False


def calcular_score_global(resultados: list[dict]) -> dict[str, float]:
    """Score ponderado por modelo: herramienta_correcta×40 + args×30 + precision×30."""
    from collections import defaultdict
    scores = defaultdict(list)
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


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Benchmark MCP tool calling para modelos Ollama")
    parser.add_argument(
        "--modelos", nargs="+", default=MODELS,
        help="Lista de modelos a evaluar (default: todos en MODELS)"
    )
    parser.add_argument(
        "--casos", nargs="+", type=int,
        help="IDs de casos a ejecutar, ej: --casos 1 3 (default: todos)"
    )
    parser.add_argument(
        "--salida", default="resultados_benchmark.csv",
        help="Nombre del archivo CSV de salida"
    )
    args = parser.parse_args()

    client = ollama.Client(host=OLLAMA_HOST)

    # Filtrar modelos disponibles
    modelos_a_probar = []
    for m in args.modelos:
        if verificar_modelo_disponible(client, m):
            modelos_a_probar.append(m)
        else:
            print(f"  [AVISO] Modelo no encontrado localmente, se omite: {m}")

    if not modelos_a_probar:
        print("ERROR: Ningún modelo disponible. Ejecuta setup_models.sh primero.")
        sys.exit(1)

    # Filtrar casos
    casos_a_probar = TEST_CASES
    if args.casos:
        casos_a_probar = [c for c in TEST_CASES if int(c["id"].split("-")[1]) in args.casos]

    print(f"\n{'='*65}")
    print(f"  MCP Tool Calling Benchmark  —  {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print(f"{'='*65}")
    print(f"  Modelos: {len(modelos_a_probar)}  |  Casos: {len(casos_a_probar)}")
    print(f"  Total iteraciones: {len(modelos_a_probar) * len(casos_a_probar)}")
    print(f"{'='*65}\n")

    resultados = []
    total = len(modelos_a_probar) * len(casos_a_probar)
    idx = 0

    for modelo in modelos_a_probar:
        for caso in casos_a_probar:
            idx += 1
            print(f"[{idx:02d}/{total}] {modelo} | {caso['id']} — {caso['nombre']}")
            r = ejecutar_caso(client, modelo, caso)
            resultados.append(r)

            estado = "✓" if r["herramienta_correcta"] else "✗"
            print(
                f"        {estado} Tool: {r['herramienta_invocada'] or 'ninguna'} | "
                f"{r['tiempo_s']}s | {r['tokens_por_s']} t/s | "
                f"Precisión args: {r['precision_args_pct']}%"
            )
            if r["error"]:
                print(f"        ERROR: {r['error']}")
        print()

    # ── CSV ───────────────────────────────────────────────────────────────────
    salida = Path(args.salida)
    campos = [
        "modelo", "caso_id", "caso_nombre", "tiempo_s", "tokens_por_s",
        "herramienta_invocada", "herramienta_correcta", "json_valido",
        "args_requeridos_presentes", "precision_args_pct", "error",
    ]
    with salida.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=campos)
        writer.writeheader()
        writer.writerows(resultados)
    print(f"Resultados guardados en: {salida.resolve()}\n")

    # ── Tabla resumen ─────────────────────────────────────────────────────────
    scores = calcular_score_global(resultados)

    resumen = []
    for modelo in modelos_a_probar:
        mrs = [r for r in resultados if r["modelo"] == modelo]
        resumen.append([
            modelo,
            f"{sum(r['tiempo_s'] for r in mrs) / len(mrs):.2f}",
            f"{sum(r['tokens_por_s'] for r in mrs) / len(mrs):.1f}",
            f"{sum(1 for r in mrs if r['herramienta_correcta'])}/{len(mrs)}",
            f"{sum(1 for r in mrs if r['args_requeridos_presentes'])}/{len(mrs)}",
            f"{sum(r['precision_args_pct'] for r in mrs) / len(mrs):.1f}%",
            f"{scores[modelo]}/100",
        ])

    resumen.sort(key=lambda x: float(x[-1].split("/")[0]), reverse=True)

    print(tabulate(
        resumen,
        headers=[
            "Modelo", "Tiempo medio(s)", "t/s medio",
            "Tool correcta", "Args presentes", "Precisión args", "Score global",
        ],
        tablefmt="rounded_outline",
    ))

    ganador = resumen[0][0]
    print(f"\nModelo recomendado: {ganador}  (score: {resumen[0][-1]})\n")


if __name__ == "__main__":
    main()

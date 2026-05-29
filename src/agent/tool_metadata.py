"""Registro único de metadatos del agente por herramienta.

Aquí vive SOLO lo que es estrategia de prompting/UX de *este* agente y que NO
puede derivarse del schema MCP (la fuente de verdad del contrato de las tools):

  * ``TOOL_TRIGGERS``            — frase de PROPÓSITO ("úsala cuando…") de cada
    tool, para que el modelo razone por intención en vez de pattern-matchear
    palabras clave. La misma frase alimenta el prompt del nodo router.
  * ``MUST_CITE_FIELDS``        — campos de SALIDA que deben citarse en la
    respuesta (no están en el schema de entrada).
  * ``TOOL_ORDER``             — orden visual de las tools en el prompt.

Los metadatos de *parámetro* (descripciones, defaults, enums, evidencia
anti-invención, tunables, grupos XOR) NO viven aquí: se declaran en
``mcp_server/tools/`` con ``Field``/``json_schema_extra`` y el agente los deriva.

Añadir una herramienta nueva ⇒ definirla en ``mcp_server/tools/`` y añadir su
entrada de prompting aquí: un único fichero agente-side que tocar.
"""

from __future__ import annotations

from src.agent.tools import AGENT_TOOLS

# Herramientas RAG (no analíticas): sus respuestas son teóricas, en prosa, y se
# excluyen del formato RESULTADO / INTERPRETACIÓN / SIGUIENTE PASO.
RAG_TOOL_NAMES: frozenset[str] = frozenset({"consultar_teoria"})

# Herramientas analíticas = todas las cargadas vía MCP (lo que no es RAG). Se
# deriva de ``AGENT_TOOLS`` para que el set se mantenga solo al añadir o quitar
# tools en el servidor MCP, en vez de enumerarlas a mano.
ANALYTICAL_TOOL_NAMES: frozenset[str] = (
    frozenset(t.name for t in AGENT_TOOLS) - RAG_TOOL_NAMES
)

# Orden visual de las 8 tools en el bloque HERRAMIENTAS del prompt. El orden de
# ``AGENT_TOOLS`` depende del subproceso MCP y no es estable entre reloads, así
# que lo fijamos explícitamente aquí.
TOOL_ORDER: list[str] = [
    "generate_synthetic_distribution",
    "generate_synthetic_arma",
    "generate_synthetic_periodic",
    "generate_synthetic_trend",
    "detect_drift",
    "augment_time_series",
    "create_exogenous_variable",
    "forecast_time_series",
]

TOOL_TRIGGERS: dict[str, str] = {
    "generate_synthetic_distribution": (
        "el usuario quiere crear datos nuevos desde cero siguiendo una "
        "distribución estadística (normal, uniforme, Poisson…), sin partir de un fichero"
    ),
    "generate_synthetic_arma": (
        "el usuario quiere generar una serie con dependencia temporal tipo "
        "ARMA/AR/MA (autocorrelación, memoria temporal)"
    ),
    "generate_synthetic_periodic": (
        "el usuario quiere generar una serie con un patrón estacional o cíclico "
        "que se repite cada cierto periodo"
    ),
    "generate_synthetic_trend": (
        "el usuario quiere generar una serie con una tendencia creciente o "
        "decreciente (lineal, polinómica o exponencial)"
    ),
    "detect_drift": (
        "el usuario quiere saber si la distribución de una serie ha cambiado o "
        "si hay drift / pérdida de estabilidad"
    ),
    "augment_time_series": (
        "el usuario quiere ampliar o aumentar una serie existente con más observaciones"
    ),
    "create_exogenous_variable": (
        "el usuario quiere añadir una variable o columna nueva derivada de la "
        "serie (p. ej. por PCA o correlación)"
    ),
    "forecast_time_series": (
        "el usuario quiere predecir o pronosticar valores futuros de una serie "
        "(p. ej. con SARIMAX)"
    ),
}

# Campos deterministas (string/entero) de SALIDA que deben citarse literalmente
# en el bloque RESULTADO de cada herramienta analítica. Se usan para verificar
# la fidelidad numérica de la síntesis (RF-11). Los floats (p-valores, métricas)
# se omiten a propósito: el modelo puede redondearlos legítimamente y un
# substring-match daría falsos negativos. No derivable del schema de entrada.
MUST_CITE_FIELDS: dict[str, tuple[str, ...]] = {
    "detect_drift": ("drift_label", "method_used"),
    "forecast_time_series": ("model_used",),
    "augment_time_series": ("new_rows", "strategy_used"),
    "create_exogenous_variable": ("new_column_name", "relation_used"),
    "generate_synthetic_distribution": ("rows_generated",),
    "generate_synthetic_arma": ("rows_generated",),
    "generate_synthetic_periodic": ("rows_generated",),
    "generate_synthetic_trend": ("rows_generated",),
}

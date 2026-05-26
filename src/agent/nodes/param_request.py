"""Nodo de recogida guiada de parámetros para herramientas incompletas."""

from __future__ import annotations

from langchain_core.messages import AIMessage

from src.agent.state import AgentState

# Parámetros obligatorios por herramienta. Los Optional/con default de la firma
# Pydantic NO se incluyen aquí — solo lo que la tool no puede ejecutarse sin.
TOOL_REQUIRED_PARAMS: dict[str, list[str]] = {
    "generate_synthetic_distribution": [
        "start_date", "frequency", "distribution_type", "distribution_params",
    ],
    "generate_synthetic_arma": [
        "start_date", "frequency",
    ],
    "generate_synthetic_periodic": [
        "start_date", "frequency", "period_length", "pattern_type",
        "distribution_type", "distribution_params",
    ],
    "generate_synthetic_trend": [
        "start_date", "frequency", "trend_type", "trend_params",
    ],
    "detect_drift": ["file_path", "index_column", "method"],
    "augment_time_series": [
        "file_path", "index_column", "strategy", "size", "frequency",
    ],
    "create_exogenous_variable": [
        "file_path", "index_column", "new_column_name", "relation",
    ],
    "forecast_time_series": [
        "file_path", "index_column", "target_column", "forecast_steps",
    ],
}

_PARAM_DESCRIPTIONS: dict[str, str] = {
    # Comunes
    "file_path": "la ruta al fichero CSV",
    "index_column": "el nombre de la columna que actúa como índice temporal del CSV",
    "start_date": "la fecha de inicio en formato YYYY-MM-DD (ej. 2024-01-01)",
    "frequency": "la frecuencia temporal: 'D' (diaria), 'W' (semanal), 'M' (mensual), 'h' (horaria), 'min' o 's'",
    # Distribución
    "distribution_type": (
        "el código de distribución (1=Normal, 2=Binomial, 3=Poisson, 4=Geométrica, "
        "7=Uniforme, 9=Exponencial, 10=Gamma, 11=Beta, 17=Aleatorio, etc.)"
    ),
    "distribution_params": "los parámetros de la distribución como lista (ej. [0.0, 1.0] para Normal mu=0 sigma=1)",
    # Periódica
    "period_length": "cada cuántas observaciones se repite el patrón (entero positivo)",
    "pattern_type": "tipo de patrón cíclico: 1=variación de amplitud, 2=variación de cantidad de elementos",
    # Tendencia
    "trend_type": "el código del tipo de tendencia (lineal, polinómica, exponencial, etc.)",
    "trend_params": "los coeficientes que definen la tendencia como lista de números",
    # Drift
    "method": (
        "el método de detección de drift: KS (Kolmogorov-Smirnov), JS (Jensen-Shannon), "
        "PSI (Population Stability Index), CUSUM, MEWMA o HOTELLING"
    ),
    # Augmentación
    "strategy": (
        "la estrategia de aumentación: 'normal', 'muller', 'duplicate', 'harmonic' o 'statistical'"
    ),
    "size": "el número de observaciones nuevas a generar (entero positivo)",
    # Exógenas
    "new_column_name": "el nombre de la nueva columna que se creará",
    "relation": (
        "el tipo de relación: 'pca', 'correlation', 'covariance', 'linear' o 'polynomial'"
    ),
    # Forecast
    "target_column": "el nombre de la columna a predecir",
    "forecast_steps": "el número de pasos futuros a predecir (entero positivo)",
}


def get_missing_params(tool_name: str, provided_args: dict) -> list[str]:
    """Devuelve los parámetros obligatorios que faltan o están vacíos en provided_args."""
    required = TOOL_REQUIRED_PARAMS.get(tool_name, [])
    return [
        p for p in required
        if p not in provided_args or provided_args[p] is None or provided_args[p] == ""
    ]


def solicitar_parametros_node(state: AgentState) -> dict:
    """Genera un mensaje pidiendo al usuario los parámetros faltantes de la herramienta pendiente.

    Se activa cuando razonador detecta que el LLM intentó llamar a una herramienta
    sin proporcionar todos los argumentos obligatorios.

    No limpia `pending_tool` ni `pending_params`; eso ocurrirá en razonador cuando
    el usuario responda y la llamada pueda completarse.
    """
    pending_tool = state.get("pending_tool")
    if not pending_tool:
        return {}

    collected = state.get("pending_params") or {}
    missing = get_missing_params(pending_tool, collected)

    if not missing:
        return {}

    lines: list[str] = [
        "Para continuar necesito algunos datos adicionales. Por favor, proporciona:"
    ]
    for param in missing:
        desc = _PARAM_DESCRIPTIONS.get(param, f"el valor de '{param}'")
        lines.append(f"  • **{param}**: {desc}")

    return {"messages": [AIMessage(content="\n".join(lines))]}

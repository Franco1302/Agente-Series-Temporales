"""Nodo de recogida guiada de parámetros para herramientas incompletas."""

from __future__ import annotations

from langchain_core.messages import AIMessage

from src.agent.state import AgentState

# Mismos parámetros obligatorios que en el antiguo param_validation.py.
# Los parámetros con valor por defecto en la firma de la herramienta no se incluyen.
TOOL_REQUIRED_PARAMS: dict[str, list[str]] = {
    "detect_drift_kolmogorov_smirnov": ["file_path", "reference_column"],
    "generate_synthetic_series": [
        "start_date",
        "periods",
        "frequency",
        "distribution_type",
        "distribution_params",
    ],
    "augment_data_linear_relation": [
        "file_path",
        "index_column",
        "new_column_name",
        "slope",
        "intercept",
    ],
}

_PARAM_DESCRIPTIONS: dict[str, str] = {
    "file_path": "la ruta al fichero CSV",
    "reference_column": "el nombre de la columna a analizar",
    "start_date": "la fecha de inicio en formato YYYY-MM-DD (ej. 2023-01-01)",
    "periods": "el número de periodos a generar (número entero positivo)",
    "frequency": "la frecuencia temporal: 'D' (diaria), 'W' (semanal), 'M' (mensual), 'H' (horaria)",
    "distribution_type": "el tipo de distribución: 0=Normal, 1=Uniforme, 2=Poisson, 3=Exponencial",
    "distribution_params": "los parámetros de la distribución como lista (ej. [0.0, 1.0] para Normal)",
    "index_column": "el nombre de la columna existente que actuará como variable independiente",
    "new_column_name": "el nombre de la nueva columna que se creará",
    "slope": "la pendiente de la relación lineal (número decimal)",
    "intercept": "el término independiente de la relación lineal (número decimal)",
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

"""Nodo de validación de parámetros: detecta tool calls con argumentos incompletos."""

from __future__ import annotations

from langchain_core.messages import AIMessage

from src.agent.state import AgentState

# Mapa de parámetros obligatorios por herramienta.
# Los parámetros con valor por defecto NO se incluyen (no son obligatorios).
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

# Descripciones en español de cada parámetro para mensajes de solicitud claros.
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


def _describe_param(param: str) -> str:
    return _PARAM_DESCRIPTIONS.get(param, f"el valor de '{param}'")


def param_validation_node(state: AgentState) -> dict:
    """Valida que las tool calls del último AIMessage tengan todos los parámetros obligatorios.

    Si detecta parámetros faltantes, genera un AIMessage pidiendo al usuario
    exactamente qué información necesita y vacía las tool_calls para que el grafo
    termine en END en lugar de ejecutar una herramienta incompleta.

    Si todos los parámetros están presentes, devuelve el estado sin modificaciones
    para que el flujo continúe hacia tool_execution_node.
    """
    messages = state.get("messages", [])
    if not messages:
        return {}

    last_message = messages[-1]

    # Si el último mensaje no es un AIMessage con tool_calls, no hay nada que validar.
    if not isinstance(last_message, AIMessage):
        return {}

    tool_calls = getattr(last_message, "tool_calls", None) or []
    if not tool_calls:
        return {}

    missing_by_tool: dict[str, list[str]] = {}

    for call in tool_calls:
        tool_name = call.get("name", "") if isinstance(call, dict) else getattr(call, "name", "")
        args = call.get("args", {}) if isinstance(call, dict) else getattr(call, "args", {})

        required = TOOL_REQUIRED_PARAMS.get(tool_name, [])
        missing = [p for p in required if p not in args or args[p] is None or args[p] == ""]

        if missing:
            missing_by_tool[tool_name] = missing

    if not missing_by_tool:
        # Todo correcto: el nodo no modifica el estado.
        return {"pending_params": []}

    # Construye un mensaje de solicitud claro para el usuario.
    lines: list[str] = [
        "Para continuar necesito algunos datos adicionales. Por favor, proporciona:"
    ]
    all_missing: list[str] = []

    for tool_name, missing_params in missing_by_tool.items():
        all_missing.extend(missing_params)
        for param in missing_params:
            lines.append(f"  • **{param}**: {_describe_param(param)}")

    request_message = AIMessage(content="\n".join(lines))

    return {
        "messages": [request_message],
        "pending_params": all_missing,
    }

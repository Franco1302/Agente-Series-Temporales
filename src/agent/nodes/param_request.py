"""Nodo de recogida guiada de parámetros para herramientas incompletas.

Cubre dos casos:
  * Parámetros OBLIGATORIOS ausentes → pregunta directa por cada uno.
  * Parámetros OPCIONALES "tunables" ausentes (umbrales, bins, coeficientes…)
    → confirmación al usuario con los valores por defecto listados, para
    evitar ejecutar la API con valores que el usuario no eligió.
"""

from __future__ import annotations

import re
from functools import lru_cache

from langchain_core.messages import AIMessage

from src.agent.state import AgentState
from src.agent.tools import AGENT_TOOLS


def _derive_required_from_schema(tool) -> list[str]:
    """Lee el schema Pydantic de la tool y devuelve los parámetros sin default.

    Pydantic v2 expone `model_fields[name].is_required()` para distinguir los
    obligatorios. Cuando el schema es un dict (JSON Schema crudo), usamos su
    clave `required`. Si la tool no expone schema (caso raro), devolvemos [].
    """
    schema = getattr(tool, "args_schema", None)
    if schema is None:
        return []
    if hasattr(schema, "model_fields"):
        return [n for n, f in schema.model_fields.items() if f.is_required()]
    if isinstance(schema, dict):
        return list(schema.get("required", []))
    return []


@lru_cache(maxsize=1)
def _build_required_map() -> dict[str, list[str]]:
    """Construye el mapa tool_name → required leyendo `AGENT_TOOLS` una sola vez.

    La fuente de verdad pasa a ser el schema de cada tool MCP/LC. Si alguien
    añade o quita un parámetro obligatorio en `mcp_server/tools/`, el agente
    lo detecta automáticamente sin tocar este módulo. Eliminamos así el riesgo
    de "drift" entre el dict hand-coded y la firma real de la API.
    """
    return {t.name: _derive_required_from_schema(t) for t in AGENT_TOOLS}


# Compatibilidad: `TOOL_REQUIRED_PARAMS` sigue exponiéndose para llamadores
# externos (tests, scripts) pero su contenido se deriva automáticamente.
TOOL_REQUIRED_PARAMS: dict[str, list[str]] = _build_required_map()


# ── Grupos "alternativos" (XOR) que el schema no expresa ────────────────────
#
# Algunas tools requieren "exactamente uno de" varios parámetros — restricción
# que vive como código imperativo dentro de la API (p. ej. `_resolve_horizon`
# de `mcp_server/tools/synthetic.py`). El schema Pydantic los marca como
# Optional, así que la verificación automática los pasaría por alto y la
# API rechazaría la llamada con un error opaco para el usuario.
#
# Cada entrada es `tool_name → [grupo1, grupo2, …]` y cada grupo es una lista
# de nombres de parámetros entre los cuales al menos uno debe estar presente.
TOOL_ALTERNATIVE_GROUPS: dict[str, list[list[str]]] = {
    "generate_synthetic_distribution": [["periods", "end_date"]],
    "generate_synthetic_arma": [["periods", "end_date"]],
    "generate_synthetic_periodic": [["periods", "end_date"]],
    "generate_synthetic_trend": [["periods", "end_date"]],
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
    # Horizonte temporal (XOR en las tools sintéticas)
    "periods": "el número de observaciones a generar (entero positivo)",
    "end_date": "la fecha de fin en formato YYYY-MM-DD",
}


# ── Parámetros opcionales "tunables" ────────────────────────────────────────
#
# Son los parámetros con default que afectan materialmente al algoritmo
# (umbrales, número de bins, coeficientes…). Cuando el usuario no los fija,
# preferimos preguntar antes que ejecutar a ciegas con los defaults.
#
# Se excluyen los cosméticos (`with_plot`, `column_name`, `return_metrics`)
# porque no cambian el resultado analítico.

_DETECT_DRIFT_TUNABLES_BY_METHOD: dict[str, list[str]] = {
    "KS": ["threshold"],
    "JS": ["threshold"],
    "PSI": ["threshold", "num_bins"],
    "CUSUM": ["threshold", "drift_cusum"],
    "MEWMA": ["min_instances", "alpha", "lambd"],
    "HOTELLING": ["min_instances", "alpha"],
}

_AUGMENT_TUNABLES_BY_STRATEGY: dict[str, list[str]] = {
    "duplicate": ["duplication_factor", "perturbation_std"],
    "statistical": ["statistical_type"],
}

_EXOGENOUS_TUNABLES_BY_RELATION: dict[str, list[str]] = {
    "linear": ["coefficients"],
    "polynomial": ["coefficients"],
}

_TOOL_TUNABLES_STATIC: dict[str, list[str]] = {
    "forecast_time_series": ["frequency", "model"],
    "generate_synthetic_arma": [
        "constant", "noise_std", "seasonality",
        "ar_coefficients", "ma_coefficients",
    ],
    "generate_synthetic_trend": ["noise"],
}

# Descripciones + default-as-text para cada tunable. Se usa para construir el
# mensaje de confirmación que ve el usuario.
_OPTIONAL_PARAM_INFO: dict[str, tuple[str, str]] = {
    "threshold": (
        "umbral de decisión del test",
        "específico del método (KS=0.05, JS=0.2, PSI=0.25, CUSUM=1.5)",
    ),
    "num_bins": ("número de bins del histograma (PSI)", "10"),
    "drift_cusum": ("término de deriva del CUSUM", "0.5"),
    "min_instances": ("número de observaciones iniciales del límite de control", "100"),
    "lambd": ("parámetro de suavizado del MEWMA", "0.5"),
    "alpha": ("nivel de significación del límite de control", "0.05"),
    "duplication_factor": ("proporción de filas duplicadas", "0.5"),
    "perturbation_std": ("desviación estándar del ruido añadido al duplicar", "0.1"),
    "statistical_type": ("tipo de estadístico para la estrategia 'statistical'", "1"),
    "coefficients": (
        "lista de coeficientes para la relación lineal/polinómica",
        "se calcularán automáticamente a partir de los datos",
    ),
    "frequency": ("frecuencia temporal de la serie", "'D' (diaria)"),
    "model": ("modelo de predicción", "'sarimax'"),
    "constant": ("término constante c del modelo ARMA", "0.0"),
    "noise_std": ("desviación estándar del ruido blanco", "1.0"),
    "seasonality": ("periodo de estacionalidad", "0 (sin estacionalidad)"),
    "ar_coefficients": ("coeficientes AR del modelo", "[] (sin componente AR)"),
    "ma_coefficients": ("coeficientes MA del modelo", "[] (sin componente MA)"),
    "noise": ("magnitud del ruido aditivo gaussiano sobre la tendencia", "0.0"),
}


def _is_empty(value: object) -> bool:
    """True si el valor cuenta como ausente (None, cadena vacía o lista vacía)."""
    return value is None or value == "" or value == []


def get_missing_params(tool_name: str, provided_args: dict) -> list[str]:
    """Devuelve los parámetros obligatorios que faltan o están vacíos en provided_args."""
    required = TOOL_REQUIRED_PARAMS.get(tool_name, [])
    return [p for p in required if _is_empty(provided_args.get(p))]


def get_missing_alternative_groups(tool_name: str, provided_args: dict) -> list[list[str]]:
    """Devuelve los grupos "uno-de" que ningún parámetro satisface.

    Un grupo está satisfecho si AL MENOS uno de sus parámetros tiene valor.
    La regla "no ambos" (XOR estricto) la valida la propia API: aquí solo
    pedimos lo mínimo para que la llamada salga adelante.
    """
    groups = TOOL_ALTERNATIVE_GROUPS.get(tool_name, [])
    return [
        group for group in groups
        if all(_is_empty(provided_args.get(p)) for p in group)
    ]


def get_tunable_params(tool_name: str, provided_args: dict) -> list[str]:
    """Devuelve los parámetros opcionales relevantes para esta llamada.

    Algunos tools tienen tunables que dependen del valor de otro parámetro
    (p. ej. `detect_drift` los tunables cambian según `method`); esta función
    encapsula esa lógica para que el razonador no la conozca.
    """
    if tool_name == "detect_drift":
        return list(_DETECT_DRIFT_TUNABLES_BY_METHOD.get(provided_args.get("method"), []))
    if tool_name == "augment_time_series":
        return list(_AUGMENT_TUNABLES_BY_STRATEGY.get(provided_args.get("strategy"), []))
    if tool_name == "create_exogenous_variable":
        return list(_EXOGENOUS_TUNABLES_BY_RELATION.get(provided_args.get("relation"), []))
    return list(_TOOL_TUNABLES_STATIC.get(tool_name, []))


def get_missing_tunable_params(tool_name: str, provided_args: dict) -> list[str]:
    """Devuelve los tunables que no han sido fijados explícitamente por el usuario."""
    tunables = get_tunable_params(tool_name, provided_args)
    return [
        p for p in tunables
        if p not in provided_args or provided_args[p] is None or provided_args[p] == ""
    ]


def _format_required_message(
    missing: list[str],
    missing_groups: list[list[str]] | None = None,
) -> str:
    lines: list[str] = [
        "Para continuar necesito algunos datos adicionales. Por favor, proporciona:"
    ]
    for param in missing:
        desc = _PARAM_DESCRIPTIONS.get(param, f"el valor de '{param}'")
        lines.append(f"  • **{param}**: {desc}")
    for group in missing_groups or []:
        alternativas = " **o** ".join(f"**{p}** ({_PARAM_DESCRIPTIONS.get(p, p)})" for p in group)
        lines.append(f"  • Uno de: {alternativas}")
    return "\n".join(lines)


def _format_optional_confirmation_message(tool_name: str, missing_tunable: list[str]) -> str:
    lines: list[str] = [
        f"Voy a ejecutar **{tool_name}** y hay varios parámetros opcionales que aún no has indicado. "
        "Estos son sus valores por defecto:",
        "",
    ]
    for param in missing_tunable:
        desc, default = _OPTIONAL_PARAM_INFO.get(
            param, (f"el parámetro '{param}'", "valor por defecto del sistema")
        )
        lines.append(f"  • **{param}** ({desc}) → default: `{default}`")
    lines.append("")
    lines.append(
        "Responde **sí** (o «usa los defaults») para ejecutar con esos valores, o "
        "indícame los que quieras cambiar con la sintaxis `nombre=valor` "
        "(ejemplo: `threshold=0.3, num_bins=20`)."
    )
    return "\n".join(lines)


# ── Parser de la respuesta del usuario a la confirmación de opcionales ──────
#
# El razonador llama a estos helpers cuando el estado indica que estamos
# esperando la confirmación del usuario sobre los tunables. Es determinista
# para evitar que el LLM local (qwen) pierda los args originales al re-emitir
# la tool call.

# "no", "cancela", "olvida", "déjalo", "para" como palabra suelta o "no quiero/
# no ejecutes/no detectes" detectan intención de cancelar.
_CANCEL_PATTERN = re.compile(
    r"\b(cancela|olvida|d[eé]jalo|para)\b|"
    r"\bno\s+(quiero|deseo|hagas|ejecutes|detectes|lo\s+hagas|continues|sigas)\b",
    re.IGNORECASE,
)

# Captura pares nombre=valor o nombre: valor. Acepta números (con signo y
# decimales/notación científica), listas entre corchetes y strings simples.
_VALUE_PAIR_RE = re.compile(
    r"(?P<name>[A-Za-z_][A-Za-z_0-9]*)\s*[=:]\s*"
    r"(?P<value>\[[^\]]*\]|'[^']*'|\"[^\"]*\"|[\-+]?\d+\.?\d*(?:[eE][\-+]?\d+)?|[A-Za-z_][A-Za-z_0-9]*)"
)


def is_cancel_intent(user_message: str) -> bool:
    """Detecta si el mensaje del usuario expresa intención de cancelar la ejecución."""
    return bool(_CANCEL_PATTERN.search(user_message))


def parse_optional_values(user_message: str, tunables: list[str]) -> dict:
    """Extrae pares `nombre=valor` del mensaje del usuario para los tunables conocidos.

    Solo se retienen los nombres presentes en la lista de tunables (evita que
    palabras sueltas como "default=1" contaminen otros parámetros). Los valores
    numéricos se convierten a int/float; las listas y strings entrecomillados se
    parsean al tipo correspondiente.
    """
    found: dict[str, object] = {}
    for match in _VALUE_PAIR_RE.finditer(user_message):
        name = match.group("name")
        if name not in tunables:
            continue

        raw = match.group("value").strip()

        # Lista de números o strings
        if raw.startswith("[") and raw.endswith("]"):
            inner = raw[1:-1].strip()
            if not inner:
                found[name] = []
                continue
            items: list[object] = []
            for piece in (p.strip() for p in inner.split(",")):
                try:
                    items.append(float(piece) if "." in piece or "e" in piece.lower() else int(piece))
                except ValueError:
                    items.append(piece.strip("'\""))
            found[name] = items
            continue

        # String entrecomillado
        if (raw.startswith("'") and raw.endswith("'")) or (raw.startswith('"') and raw.endswith('"')):
            found[name] = raw[1:-1]
            continue

        # Numérico
        try:
            if "." in raw or "e" in raw.lower():
                found[name] = float(raw)
            else:
                found[name] = int(raw)
            continue
        except ValueError:
            pass

        # Fallback: string sin comillas
        found[name] = raw

    return found


def solicitar_parametros_node(state: AgentState) -> dict:
    """Pide al usuario los datos que faltan para ejecutar la herramienta pendiente.

    Dos modos:
      * Si faltan parámetros OBLIGATORIOS, pregunta por cada uno.
      * Si todos los obligatorios están y solo faltan TUNABLES, lista los
        defaults y pide confirmación.

    No limpia `pending_tool` ni `pending_params`; el razonador los reseteará
    cuando el usuario responda y se pueda completar la llamada.
    """
    pending_tool = state.get("pending_tool")
    if not pending_tool:
        return {}

    collected = state.get("pending_params") or {}

    missing_required = get_missing_params(pending_tool, collected)
    missing_groups = get_missing_alternative_groups(pending_tool, collected)
    if missing_required or missing_groups:
        return {"messages": [AIMessage(
            content=_format_required_message(missing_required, missing_groups),
        )]}

    missing_tunable = get_missing_tunable_params(pending_tool, collected)
    if missing_tunable:
        return {"messages": [AIMessage(
            content=_format_optional_confirmation_message(pending_tool, missing_tunable),
        )]}

    return {}

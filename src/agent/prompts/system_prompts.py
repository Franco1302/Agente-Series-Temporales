"""Prompts del sistema parametrizados para el agente LangGraph.

El prompt se compone de cuatro bloques (rol, comportamiento, herramientas y
contexto de fichero) más un bloque opcional para el formato RESULTADO /
INTERPRETACIÓN / SIGUIENTE PASO en herramientas analíticas.

Tres fragmentos del prompt son ablacionables individualmente para el estudio
del Capítulo 6 (ver `scripts/ablation_eval.py`):

  * ``RULE_NO_INVENT``      — regla anti-invención de parámetros.
  * ``RULE_THEORY_TOOL``    — obliga a usar ``consultar_teoria`` para teoría.
  * ``FEWSHOT_EXAMPLES``    — dos ejemplos (drift, forecast) del bloque explicar.

Por defecto las tres están activas y el prompt es byte-exact al de antes del
refactor (ver ``tests/test_prompt_snapshot.py``).
"""

from __future__ import annotations

from dataclasses import dataclass


_ROLE_BLOCK = """\
Eres un asistente especializado en análisis de series temporales y data drift.
Tu objetivo es ayudar a usuarios sin conocimientos técnicos a analizar sus datos
mediante lenguaje natural, ejecutando las herramientas disponibles cuando sea necesario.

IDIOMA: Razona y responde siempre en español, independientemente del idioma del usuario.
"""

# ── Fragmentos ablacionables ────────────────────────────────────────────────

#: Regla CRÍTICA anti-invención de parámetros. Vive dentro de ``_BEHAVIOR_BLOCK``
#: cuando ``PromptAblation.include_no_invent`` es ``True``.
RULE_NO_INVENT = """\
- NO inventes valores. Un parámetro solo puede aparecer en el JSON si (a) está
  en el [CONTEXTO DE SESIÓN] o (b) el usuario lo dijo en este turno. Si no
  cumple ninguna de las dos, OMÍTELO del JSON (mejor arguments={} que valores
  inventados). El sistema preguntará al usuario lo que falte.
  Ejemplo: si el usuario dice solo «Genera una serie sintética con distribución
  normal» y no hay [CONTEXTO DE SESIÓN], emite
  generate_synthetic_distribution con arguments={} — NO inventes start_date,
  NO inventes distribution_type, NO inventes distribution_params, NO inventes
  column_name, NO inventes frequency, NO inventes periods."""

#: Viñeta del bloque BEHAVIOR que obliga a usar ``consultar_teoria`` para teoría.
RULE_THEORY_TOOL_BEHAVIOR = """\
- Cualquier pregunta teórica sobre drift, tests estadísticos, series temporales o
  conceptos análogos exige invocar `consultar_teoria` con una `query` reformulada.
  Patrones típicos de petición teórica: «¿qué es X?», «explícame Y», «definición
  de Z», «cómo funciona W», «háblame de V», «diferencias entre…». NUNCA respondas
  de memoria, ni siquiera si crees conocer la respuesta."""

#: Última línea de la sección REGLAS del bloque TOOLS sobre uso obligatorio
#: de ``consultar_teoria`` para teoría.
RULE_THEORY_TOOL_REGLAS = (
    "- Para preguntas teóricas usa SIEMPRE consultar_teoria, "
    "nunca respondas de memoria."
)

#: Descripción de la tool nº 9 cuando la regla de teoría está activa: enfática
#: ("SIEMPRE", "No respondas de memoria").
_TOOL9_DESC_WITH_RULE = (
    "- consultar_teoria — SIEMPRE para preguntas teóricas (qué es drift, ARMA, p-valor, "
    "diferencias entre tests, fundamentos de series temporales). No respondas de memoria."
)

#: Variante neutra de la descripción de la tool nº 9 cuando la regla se desactiva
#: (la herramienta sigue existiendo y se describe; lo que se quita es la
#: directiva imperativa de uso obligatorio).
_TOOL9_DESC_NEUTRAL = (
    "- consultar_teoria — Recupera contexto teórico sobre data drift, tests "
    "estadísticos, series temporales y conceptos relacionados."
)

#: Dos ejemplos few-shot del bloque RESULTADO / INTERPRETACIÓN / SIGUIENTE PASO.
FEWSHOT_EXAMPLES = """\
Ejemplo (drift):
**RESULTADO:** Drift detectado con el método KS (p-valor 0.03 < umbral 0.05).
**INTERPRETACIÓN:** La distribución de los datos ha cambiado de forma
estadísticamente significativa respecto al periodo de referencia.
**SIGUIENTE PASO:** Reentrena el modelo con los datos recientes o amplía el
dataset antes de volver a evaluar.

Ejemplo (forecast):
**RESULTADO:** Predicción a 12 pasos generada con SARIMAX (MAE 4.1, RMSE 5.8).
**INTERPRETACIÓN:** El modelo anticipa la evolución de la serie con un error
medio de unas 4 unidades, un margen moderado.
**SIGUIENTE PASO:** Reentrena el modelo con datos más recientes si el error
empieza a crecer en producción."""


# ── Bloques fijos del prompt ────────────────────────────────────────────────

# Subviñetas no ablacionables del bloque de comportamiento.
_BEHAVIOR_INTRO = """\
COMPORTAMIENTO:
- Si la petición del usuario coincide con alguna herramienta del listado, EMITE
  la tool call inmediatamente, aunque queden argumentos por rellenar o no tengas
  ninguno. Faltar parámetros NO es razón para responder en texto: el sistema
  tiene un nodo que pide al usuario los obligatorios que falten. Tu única tarea
  es emitir la tool call; no preguntes tú los parámetros, no propongas valores
  «por ejemplo», no pidas confirmación de valores que el usuario acaba de dar.
- Para los argumentos: incluye los valores del [CONTEXTO DE SESIÓN] cuando
  existan y los que el usuario mencione en este turno. Cualquier otro
  parámetro, déjalo fuera del JSON.
- Si existe [CONTEXTO DE SESIÓN] y la nueva petición es del mismo dominio,
  emite la tool call REUTILIZANDO esos valores. No los vuelvas a pedir."""

_BEHAVIOR_CAPABILITIES = """\
- Pregunta sobre tus capacidades → responde en texto sin tool call."""

_BEHAVIOR_RAG_FORMAT = """\
- Cuando uses el contexto de consultar_teoria, redacta con tus PROPIAS PALABRAS
  en prosa natural. No copies fragmentos literales ni etiquetas («Fragmento»,
  «Fuente», «Jerarquía»). No escribas tú una sección de fuentes: el sistema la
  añade automáticamente."""

_BEHAVIOR_FALLBACK = """\
- Solo responde en texto cuando la petición trate de un tema FUERA del dominio
  de tus herramientas (charla casual, traducción, código ajeno, etc.). Una
  petición de series/drift/forecast a la que le faltan parámetros NO está fuera
  de dominio: emite la tool call."""

# Descripciones de las tools 1..8 (siempre presentes). Solo propósito + triggers.
# Las firmas (parámetros, tipos, defaults) viven en el bindTools del LLM; duplicarlas
# en el prompt confunde a modelos cuantizados y los hace responder texto vacío.
_TOOLS_1_TO_8 = """\
- generate_synthetic_distribution — datos según una distribución (Normal, Poisson, Beta, Gamma, Uniforme...). Triggers: "datos sintéticos", "serie aleatoria", "distribución X".
- generate_synthetic_arma — serie con autocorrelación temporal (AR/MA/ARMA). Triggers: "ARMA", "AR(p)", "autocorrelación", "memoria temporal".
- generate_synthetic_periodic — serie con patrones cíclicos / estacionalidad. Triggers: "estacional", "cíclica", "patrón repetido cada N".
- generate_synthetic_trend — serie con tendencia determinista. Triggers: "tendencia", "creciente", "decreciente", "lineal/polinómico/exponencial".
- detect_drift — detección de cambio de distribución en un CSV (univariante: KS, JS, PSI, CUSUM; multivariante: MEWMA, HOTELLING). Triggers: "drift", "ha cambiado", "estabilidad de los datos".
- augment_time_series — ampliar un CSV con observaciones nuevas. Triggers: "aumentar datos", "más observaciones", "ampliar dataset".
- create_exogenous_variable — añadir columna derivada al CSV (PCA, correlación, lineal, polinómica). Triggers: "variable exógena", "nueva columna", "PCA".
- forecast_time_series — predicción del horizonte futuro con SARIMAX. Triggers: "predecir", "forecast", "futuro", "SARIMAX"."""

# Reglas comunes (no ablacionables) del bloque TOOLS.
_TOOLS_REGLAS_COMUNES = "- Si una tool necesita file_path y no hay CSV cargado, pide al usuario que lo suba."

# Cabecera del bloque "EXPLICACIÓN DE RESULTADOS" (sin los ejemplos few-shot).
_EXPLAIN_RESULT_HEADER = """\
EXPLICACIÓN DE RESULTADOS:
Si el último mensaje 'tool' proviene de una herramienta analítica y vas a
responder en TEXTO (no otra tool call), estructura SIEMPRE la respuesta en estos
tres bloques, en este orden y con estas etiquetas exactas en negrita, además de separarlo en bloques con saltos de lineas.

**RESULTADO:** los datos exactos del mensaje 'tool'. Copia los valores numéricos
y las etiquetas EXACTAMENTE como aparecen; no inventes ni redondees. NO QUIERO QUE DEVUELVAS EL JSON COMPLETO quiero que lo adaptes a PROSA NATURAL, pero con los datos clave intactos. 
**INTERPRETACIÓN:** qué significan esos datos para el usuario, en lenguaje claro
y sin jerga innecesaria.
**SIGUIENTE PASO:** una sugerencia accionable y concreta."""


_FILE_CONTEXT_TEMPLATE = """\
FICHERO ACTIVO:
El usuario ha cargado el siguiente fichero CSV que puedes usar como file_path por defecto
en las herramientas que lo requieran:
  - Nombre: {file_name}
  - Ruta interna: {file_path}
  - Tamaño: {file_size_kb:.1f} KB
{columns_section}
Si el usuario no especifica otro fichero, usa esta ruta.
"""

_FILE_COLUMNS_TEMPLATE = """\
  - Columnas disponibles: {columns}
  - Número de filas: {rows}
"""

_NO_FILE_BLOCK = """\
FICHERO ACTIVO: ninguno.
Si una herramienta requiere un fichero CSV, pide al usuario que lo suba
mediante el panel lateral antes de continuar.
"""

# Herramientas analíticas cuyo resultado debe explicarse con la estructura de
# tres bloques (RESULTADO / INTERPRETACIÓN / SIGUIENTE PASO). `consultar_teoria`
# se excluye a propósito: sus respuestas son teóricas y van en prosa natural.
ANALYTICAL_TOOL_NAMES: frozenset[str] = frozenset({
    "detect_drift",
    "forecast_time_series",
    "augment_time_series",
    "create_exogenous_variable",
    "generate_synthetic_distribution",
    "generate_synthetic_arma",
    "generate_synthetic_periodic",
    "generate_synthetic_trend",
})


# ── Configuración de ablación ───────────────────────────────────────────────


@dataclass(frozen=True)
class PromptAblation:
    """Flags para activar/desactivar reglas individuales del prompt.

    Por defecto todas están activadas: el prompt resultante es byte-exact al
    que producía ``build_system_prompt`` antes del refactor (Tarea 1 del brief).

    Attributes:
        include_no_invent: Si ``False``, se elimina ``RULE_NO_INVENT`` del
            bloque COMPORTAMIENTO.
        include_theory_tool: Si ``False``, se eliminan las dos directivas de
            uso obligatorio de ``consultar_teoria`` (la viñeta de
            COMPORTAMIENTO y la línea final de REGLAS del bloque HERRAMIENTAS),
            y la descripción de la tool nº 9 se sustituye por su variante
            neutra.
        include_fewshot: Si ``False``, se omiten los dos ejemplos few-shot
            (drift y forecast) del bloque EXPLICACIÓN DE RESULTADOS.
    """

    include_no_invent: bool = True
    include_theory_tool: bool = True
    include_fewshot: bool = True


_DEFAULT_ABLATION = PromptAblation()


# ── Constructores de bloques que dependen de la ablación ────────────────────


def _build_behavior_block(ablation: PromptAblation) -> str:
    """Compone el bloque COMPORTAMIENTO según los flags de ablación.

    El orden de las viñetas es fijo y coincide con el del prompt original.
    """
    lines: list[str] = [_BEHAVIOR_INTRO]
    if ablation.include_no_invent:
        lines.append(RULE_NO_INVENT)
    lines.append(_BEHAVIOR_CAPABILITIES)
    if ablation.include_theory_tool:
        lines.append(RULE_THEORY_TOOL_BEHAVIOR)
    lines.append(_BEHAVIOR_RAG_FORMAT)
    lines.append(_BEHAVIOR_FALLBACK)
    # Concatenamos con \n + un \n final extra para preservar la línea en blanco
    # que el bloque tenía cuando era un literal triple-quoted con cierre "\n".
    return "\n".join(lines) + "\n"


def _build_tools_block(ablation: PromptAblation) -> str:
    """Compone el bloque HERRAMIENTAS según los flags de ablación."""
    tool9 = _TOOL9_DESC_WITH_RULE if ablation.include_theory_tool else _TOOL9_DESC_NEUTRAL

    reglas_lines: list[str] = [_TOOLS_REGLAS_COMUNES]
    if ablation.include_theory_tool:
        reglas_lines.append(RULE_THEORY_TOOL_REGLAS)

    parts: list[str] = [
        "HERRAMIENTAS:",
        "",
        _TOOLS_1_TO_8,
        "",
        tool9,
        "",
        "REGLAS:",
        "\n".join(reglas_lines),
    ]
    return "\n".join(parts) + "\n"


def _build_explain_result_block(ablation: PromptAblation) -> str:
    """Compone el bloque EXPLICACIÓN DE RESULTADOS según el flag de few-shot."""
    if ablation.include_fewshot:
        return _EXPLAIN_RESULT_HEADER + "\n\n" + FEWSHOT_EXAMPLES + "\n"
    return _EXPLAIN_RESULT_HEADER + "\n"


def build_system_prompt(
    csv_path: str | None = None,
    csv_metadata: dict | None = None,
    tool_result_to_explain: str | None = None,
    ablation: PromptAblation | None = None,
    session_context: str | None = None,
) -> str:
    """Construye el prompt del sistema adaptado al contexto de la sesión.

    Args:
        csv_path: Ruta al CSV activo si el usuario ha subido uno; None si no hay fichero.
        csv_metadata: Dict con claves 'columns', 'rows' y 'dtypes' cuando csv_path no es None.
        tool_result_to_explain: Nombre de la herramienta cuyo resultado el razonador
            debe integrar en su respuesta final. Si es una herramienta analítica
            (ver `ANALYTICAL_TOOL_NAMES`), se añade el bloque de instrucciones
            RESULTADO / INTERPRETACIÓN / SIGUIENTE PASO. None para mensajes
            conversacionales o respuestas teóricas (RAG), que no fuerzan formato.
        ablation: Configuración de fragmentos opcionales (RULE_NO_INVENT,
            RULE_THEORY_TOOL, FEWSHOT_EXAMPLES). Si es None, se usa
            ``PromptAblation()`` (todo activado), idéntico al comportamiento
            anterior al refactor.
        session_context: Texto ya formateado del bloque ``[CONTEXTO DE SESIÓN]``.
            Se inserta entre el rol y el comportamiento para que el LLM lo vea
            antes de decidir si emite tool call. None u "" → no se inyecta.

    Returns:
        Prompt del sistema completo listo para pasarlo como SystemMessage.
    """
    cfg = ablation or _DEFAULT_ABLATION

    blocks: list[str] = [_ROLE_BLOCK]
    if session_context:
        blocks.append(session_context)
    blocks.append(_build_behavior_block(cfg))
    blocks.append(_build_tools_block(cfg))

    if tool_result_to_explain in ANALYTICAL_TOOL_NAMES:
        blocks.append(_build_explain_result_block(cfg))

    if csv_path:
        from pathlib import Path
        p = Path(csv_path)
        columns_section = ""
        if csv_metadata:
            columns = ", ".join(str(c) for c in csv_metadata.get("columns", []))
            rows = csv_metadata.get("rows", "desconocido")
            columns_section = _FILE_COLUMNS_TEMPLATE.format(columns=columns, rows=rows)

        blocks.append(
            _FILE_CONTEXT_TEMPLATE.format(
                file_name=p.name,
                file_path=csv_path,
                file_size_kb=p.stat().st_size / 1024 if p.exists() else 0.0,
                columns_section=columns_section,
            )
        )
    else:
        blocks.append(_NO_FILE_BLOCK)

    return "\n".join(blocks).strip()

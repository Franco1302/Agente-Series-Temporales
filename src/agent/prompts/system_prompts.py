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
- REGLA — no inventes valores por defecto que el usuario no haya escrito.
  Si dudas, OMITE el parámetro del JSON: arguments={} es preferible a valores
  inventados. EXCEPCIÓN: si el usuario delega explícitamente ("usa los
  defaults", "cualquiera", "lo que sea", "como tú quieras", "no me importa"),
  propón TÚ valores razonables, justifica brevemente tu elección y completa
  la tool call — eso es delegación, no invención."""

#: Viñeta del bloque BEHAVIOR que obliga a usar ``consultar_teoria`` para teoría.
RULE_THEORY_TOOL_BEHAVIOR = """\
- Para cualquier pregunta teórica sobre data drift, tests estadísticos, series
  temporales o conceptos relacionados, invoca SIEMPRE la herramienta consultar_teoria
  con una `query` reformulada y precisa que capture lo que el usuario quiere saber."""

#: Última línea de la sección REGLAS del bloque TOOLS sobre uso obligatorio
#: de ``consultar_teoria`` para teoría.
RULE_THEORY_TOOL_REGLAS = (
    "- Para preguntas teóricas usa SIEMPRE consultar_teoria, "
    "nunca respondas de memoria."
)

#: Descripción de la tool nº 9 cuando la regla de teoría está activa: enfática
#: ("SIEMPRE", "No respondas de memoria").
_TOOL9_DESC_WITH_RULE = """\
9. consultar_teoria — SIEMPRE para preguntas teóricas (qué es drift, ARMA, p-valor,
   diferencias entre tests, fundamentos de series temporales). No respondas de
   memoria; usa esta herramienta. Requiere: query."""

#: Variante neutra de la descripción de la tool nº 9 cuando la regla se desactiva
#: (la herramienta sigue existiendo y se describe; lo que se quita es la
#: directiva imperativa de uso obligatorio).
_TOOL9_DESC_NEUTRAL = """\
9. consultar_teoria — Recupera contexto teórico sobre data drift, tests
   estadísticos, series temporales y conceptos relacionados. Requiere: query."""

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
- Cuando la petición del usuario coincida con una herramienta, INVOCA la herramienta
  siempre, incluso si faltan parámetros obligatorios. Pasa SOLO los parámetros que el
  usuario haya escrito EXPLÍCITAMENTE en su mensaje y OMITE COMPLETAMENTE el resto.
  El sistema validará los argumentos: si falta alguno, pedirá al usuario los datos
  automáticamente. NO redactes preguntas sobre parámetros faltantes en el contenido
  del mensaje: emite la tool call y deja que el grafo se encargue de la validación."""

_BEHAVIOR_CAPABILITIES = """\
- Si el usuario hace una pregunta sobre tus capacidades o sobre cómo usarte, responde
  directamente sin invocar ninguna herramienta."""

_BEHAVIOR_RAG_FORMAT = """\
- El contexto que devuelve consultar_teoria es material de referencia interno:
  redacta tu respuesta con tus PROPIAS PALABRAS y en prosa natural. NO copies los
  fragmentos literalmente ni reproduzcas etiquetas del contexto como «Fragmento»,
  «Fuente», «Jerarquía» o «Fuentes consultadas». NO escribas tú una sección de
  fuentes: el sistema añade la cita automáticamente al final de la respuesta."""

_BEHAVIOR_FALLBACK = """\
- Si la petición del usuario es genuinamente ambigua y no encaja con ninguna
  herramienta, pide aclaración en texto plano sin emitir tool call."""

# ── Descripciones de las tools 1..8 (generación dinámica) ──────────────────
#
# Antes este bloque era un literal hand-coded que duplicaba información del
# schema MCP (descripción, parámetros requeridos, valores enum). Ahora se
# genera leyendo `tool.description` y `tool.args_schema` de cada herramienta
# cargada vía MCP. Solo se mantienen hand-coded las dos piezas que el schema
# no expresa:
#
#   * `TOOL_TRIGGERS`: frases que disparan la elección de cada tool (guía
#     al LLM, no derivable).
#   * `TOOL_EXTRAS`: notas adicionales que ayudan al modelo a usar bien la
#     tool (enumeraciones complementarias, opcionales destacados, etc.).
#
# El orden visual de las 8 tools queda fijado por `_TOOL_ORDER`. El orden
# de `AGENT_TOOLS` depende del subproceso MCP y no es estable a través de
# reloads, así que no podemos usarlo directamente.

_TOOL_ORDER: list[str] = [
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
    "generate_synthetic_distribution": '"datos sintéticos", "serie aleatoria", "distribución X"',
    "generate_synthetic_arma": '"ARMA", "AR(p)", "autocorrelación", "memoria temporal"',
    "generate_synthetic_periodic": '"estacional", "cíclica", "patrón repetido cada N"',
    "generate_synthetic_trend": '"tendencia", "creciente", "decreciente lineal/polinómico/exponencial"',
    "detect_drift": '"drift", "ha cambiado", "estabilidad de los datos"',
    "augment_time_series": '"aumentar datos", "más observaciones", "ampliar dataset"',
    "create_exogenous_variable": '"variable exógena", "nueva columna", "PCA", "correlación"',
    "forecast_time_series": '"predecir", "forecast", "futuro", "SARIMAX"',
}

# Notas opcionales que NO viven en el schema (enumeraciones extra, etc.).
TOOL_EXTRAS: dict[str, str] = {
    "generate_synthetic_distribution": "Distribuciones: Normal, Poisson, Beta, Gamma, Uniforme, etc. (códigos 1-17).",
    "generate_synthetic_arma": "Opcionales: ar_coefficients, ma_coefficients.",
    "detect_drift": "Univariantes: KS, JS, PSI, CUSUM. Multivariantes: MEWMA, HOTELLING.",
    "forecast_time_series": "Modelo: SARIMAX.",
}

# Por tool, el parámetro requerido cuyo enum queremos mostrar inline en la
# línea "Requiere:" para que el LLM sepa de antemano los valores válidos.
_REQUIRED_PARAM_WITH_ENUM: dict[str, str] = {
    "detect_drift": "method",
    "augment_time_series": "strategy",
    "create_exogenous_variable": "relation",
}


def _build_tools_1_to_8() -> str:
    """Genera la sección "1.…8." del bloque HERRAMIENTAS desde la metadata MCP.

    Para cada tool en `_TOOL_ORDER` lee del schema:
      * `tool.description` (primer párrafo del docstring) como texto principal.
      * lista de parámetros requeridos (auto-derivada del schema).
      * `enum` del parámetro principal (method/strategy/relation) cuando aplica.
      * grupos XOR de `TOOL_ALTERNATIVE_GROUPS` para la nota "Pasa X O Y".

    Combina lo anterior con `TOOL_TRIGGERS` (hand-coded) y `TOOL_EXTRAS`
    (hand-coded). El resultado es byte-distinto del literal anterior, por lo
    que los snapshots de prompt deben regenerarse.
    """
    # Import dentro de la función para evitar ciclos al cargar el módulo.
    from src.agent.nodes.param_request import (
        TOOL_ALTERNATIVE_GROUPS,
        TOOL_REQUIRED_PARAMS,
        get_param_enum,
        get_tool_description,
    )

    entries: list[str] = []
    for idx, name in enumerate(_TOOL_ORDER, start=1):
        desc = get_tool_description(name)
        triggers = TOOL_TRIGGERS.get(name, "")

        enum_param = _REQUIRED_PARAM_WITH_ENUM.get(name)
        required: list[str] = []
        for p in TOOL_REQUIRED_PARAMS.get(name, []):
            if p == enum_param:
                values = get_param_enum(name, p)
                if values:
                    required.append(f"{p} ∈ {{{', '.join(values)}}}")
                    continue
            required.append(p)
        required_str = ", ".join(required)

        xor_note = ""
        for group in TOOL_ALTERNATIVE_GROUPS.get(name, []):
            xor_note = f" Pasa {' O '.join(group)}, no ambos."

        entry_lines = [
            f"{idx}. {name} — {desc}",
            f"   Triggers: {triggers}.",
            f"   Requiere: {required_str}.{xor_note}",
        ]
        extras = TOOL_EXTRAS.get(name)
        if extras:
            entry_lines.append(f"   {extras}")
        entries.append("\n".join(entry_lines))

    return "\n\n".join(entries)

# Reglas comunes (no ablacionables) del bloque TOOLS.
_TOOLS_REGLAS_COMUNES = """\
- Si la tool requiere file_path y no hay CSV cargado: pide al usuario que lo suba.
- No inventes parámetros opcionales: si dudas, omítelos del JSON."""

# Cabecera del bloque "EXPLICACIÓN DE RESULTADOS" (sin los ejemplos few-shot).
_EXPLAIN_RESULT_HEADER = """\
EXPLICACIÓN DE RESULTADOS:
Si el último mensaje 'tool' es de una herramienta analítica y respondes en
TEXTO, estructura la respuesta en tres bloques con estas etiquetas en negrita:

**RESULTADO:** los datos exactos del mensaje 'tool' en prosa (no JSON crudo).
Copia los valores numéricos sin inventar ni redondear.
**INTERPRETACIÓN:** qué significan esos valores para el caso concreto del
usuario. Razona conectando los números con su petición; no es un resumen.
**SIGUIENTE PASO:** sugerencia accionable y concreta.

Varía el lenguaje y la longitud en cada respuesta; evita reusar frases."""


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
        _build_tools_1_to_8(),
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

    Returns:
        Prompt del sistema completo listo para pasarlo como SystemMessage.
    """
    cfg = ablation or _DEFAULT_ABLATION

    blocks: list[str] = [
        _ROLE_BLOCK,
        _build_behavior_block(cfg),
        _build_tools_block(cfg),
    ]

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

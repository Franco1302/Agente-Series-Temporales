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


# Los grupos "alternativos" (XOR) viven en este módulo como
# ``TOOL_ALTERNATIVE_GROUPS``, derivados del schema vía
# ``_build_alternative_groups_map`` (ver más abajo, justo después de
# ``_iter_properties``).


# ── Lectura del schema de la tool (descripciones, defaults, enums) ──────────
#
# Antes había un dict `_PARAM_DESCRIPTIONS` (y otro `_OPTIONAL_PARAM_INFO` con
# defaults) que repetía a mano lo que cada tool MCP ya declara en su
# `Field(description=…, default=…)`. Esa duplicación había generado bugs por
# desincronización. Ahora se lee directamente del schema de cada tool.

@lru_cache(maxsize=1)
def _tools_by_name() -> dict[str, object]:
    return {t.name: t for t in AGENT_TOOLS}


def _get_tool(tool_name: str):
    return _tools_by_name().get(tool_name)


def _iter_properties(tool) -> dict[str, dict]:
    """Normaliza el schema de la tool a `{param: {description, default, enum, type}}`.

    Soporta los dos formatos que conviven en AGENT_TOOLS:
      * tools MCP: `args_schema` es un dict JSON Schema crudo.
      * tools LangChain locales (consultar_teoria): `args_schema` es un
        Pydantic BaseModel con `model_fields`.
    """
    if tool is None:
        return {}
    schema = getattr(tool, "args_schema", None)
    if schema is None:
        return {}
    if isinstance(schema, dict):
        return dict(schema.get("properties", {}))
    if hasattr(schema, "model_fields"):
        from pydantic.fields import PydanticUndefined
        out: dict[str, dict] = {}
        for name, field in schema.model_fields.items():
            entry: dict = {}
            if getattr(field, "description", None):
                entry["description"] = field.description
            default = getattr(field, "default", PydanticUndefined)
            if default is not PydanticUndefined:
                entry["default"] = default
            # Propaga claves del json_schema_extra (oneof_group, etc.) para que
            # los consumidores del agente lo lean del mismo `_iter_properties`
            # tanto si la tool es MCP (dict crudo) como Pydantic (BaseModel).
            extra = getattr(field, "json_schema_extra", None)
            if isinstance(extra, dict):
                for k, v in extra.items():
                    entry.setdefault(k, v)
            out[name] = entry
        return out
    return {}


# ── Grupos "alternativos" (XOR) derivados del schema ────────────────────────
#
# Fuente de verdad: cada parámetro miembro de un grupo "uno-de" se declara con
# ``Field(json_schema_extra={"oneof_group": "<nombre>"})`` en su tool MCP. El
# nombre del grupo agrupa los miembros entre sí dentro del mismo tool; un tool
# puede declarar varios grupos (ej. "horizon", "scale", …). El agente lee este
# metadato del JSON Schema vía ``_iter_properties`` y construye el mapa
# automáticamente: añadir un grupo XOR nuevo en cualquier tool MCP no
# requiere tocar este módulo.


@lru_cache(maxsize=1)
def _build_alternative_groups_map() -> dict[str, list[list[str]]]:
    """Construye ``tool_name -> [[param, …], …]`` leyendo ``oneof_group`` del schema."""
    result: dict[str, list[list[str]]] = {}
    for tool in AGENT_TOOLS:
        groups: dict[str, list[str]] = {}
        for name, info in _iter_properties(tool).items():
            grp = info.get("oneof_group") if isinstance(info, dict) else None
            if isinstance(grp, str) and grp:
                groups.setdefault(grp, []).append(name)
        # Orden alfabético dentro de cada grupo: la salida es estable y no
        # depende del orden de definición de los Field en la tool MCP.
        non_trivial = [sorted(members) for members in groups.values() if len(members) >= 2]
        if non_trivial:
            result[tool.name] = sorted(non_trivial)
    return result


# Compatibilidad: la constante sigue exponiéndose para tests/scripts externos,
# pero su contenido viene del schema MCP (no se mantiene a mano).
TOOL_ALTERNATIVE_GROUPS: dict[str, list[list[str]]] = _build_alternative_groups_map()


def _format_default_value(value: object) -> str:
    """Convierte el default a un texto legible para el usuario."""
    if value is None:
        return "sin valor por defecto"
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, str):
        return f"'{value}'"
    if isinstance(value, (int, float)):
        return str(value)
    if isinstance(value, list):
        return str(value) if value else "[]"
    return str(value)


def get_param_description(tool_name: str, param: str) -> str:
    """Descripción de un parámetro leída del schema de la tool MCP/LC.

    Devuelve un fallback genérico si la tool no expone descripción para ese
    parámetro (la solución preferida es enriquecer el `Field(description=…)`).
    """
    props = _iter_properties(_get_tool(tool_name))
    desc = props.get(param, {}).get("description")
    if isinstance(desc, str) and desc.strip():
        return desc.strip()
    return f"el valor de '{param}'"


def get_param_default_text(tool_name: str, param: str, provided_args: dict | None = None) -> str:
    """Texto legible del default de un parámetro, leído del schema de la tool.

    Maneja tres formas de default, todas derivadas de `json_schema_extra`:

      * ``default_by={"on": <disc>, "map": {...}}`` — default condicional al
        valor de otro parámetro (p. ej. `threshold` según `method`). Si
        ``provided_args`` permite resolver el discriminador, devuelve el valor
        concreto; si no, lista el mapa.
      * ``default_note=<texto>`` — el default no es un valor fijo sino una
        explicación (p. ej. "se calcularán automáticamente…").
      * ``default=<valor>`` — el `Field(default=…)` normal del schema.

    Como último recurso devuelve «sin valor por defecto».
    """
    props = _iter_properties(_get_tool(tool_name))
    entry = props.get(param, {})
    if not isinstance(entry, dict):
        return "sin valor por defecto"

    default_by = entry.get("default_by")
    if isinstance(default_by, dict):
        disc = default_by.get("on")
        mapping = default_by.get("map") or {}
        chosen = (provided_args or {}).get(disc)
        if chosen in mapping:
            return f"{_format_default_value(mapping[chosen])} (por {disc}='{chosen}')"
        pares = ", ".join(f"{k}={v}" for k, v in mapping.items())
        return f"según {disc} ({pares})"

    note = entry.get("default_note")
    if isinstance(note, str) and note:
        return note

    if "default" not in entry:
        return "sin valor por defecto"
    return _format_default_value(entry["default"])


def get_param_enum(tool_name: str, param: str) -> list[str]:
    """Valores permitidos por un parámetro Literal[...], o [] si no aplica."""
    props = _iter_properties(_get_tool(tool_name))
    enum = props.get(param, {}).get("enum")
    if isinstance(enum, list):
        return [str(v) for v in enum]
    return []


def get_tool_description(tool_name: str) -> str:
    """Primer párrafo de la descripción de la tool (su docstring en MCP)."""
    tool = _get_tool(tool_name)
    desc = (getattr(tool, "description", "") or "").strip()
    return desc.split("\n\n", 1)[0].strip()


def get_field_metadata(tool_name: str) -> dict[str, dict]:
    """Devuelve `{param: {description, default, enum, evidence, …}}` de la tool.

    Expone el schema normalizado (incluidas las claves de `json_schema_extra`)
    para que otros nodos del agente —p. ej. la defensa anti-invención en
    `reasoning.py`— deriven su metadata del schema MCP sin duplicarla.
    """
    return _iter_properties(_get_tool(tool_name))


# ── Parámetros opcionales "tunables" ────────────────────────────────────────
#
# Son los parámetros con default que afectan materialmente al algoritmo
# (umbrales, número de bins, coeficientes…). Cuando el usuario no los fija,
# preferimos preguntar antes que ejecutar a ciegas con los defaults.
#
# Se excluyen los cosméticos (`with_plot`, `column_name`, `return_metrics`)
# porque no cambian el resultado analítico.
#
# La fuente de verdad es el schema MCP: cada parámetro tunable se marca con
# ``json_schema_extra``:
#   * ``{"tunable": true}`` — siempre relevante (p. ej. forecast.frequency).
#   * ``{"tunable_if": {"<discriminador>": [valores…]}}`` — relevante solo
#     cuando otro parámetro toma ciertos valores (p. ej. drift.num_bins solo
#     aplica si method=='PSI'). El nombre del discriminador viaja en el propio
#     metadato, así que este módulo no necesita conocer method/strategy/relation.

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

    Genérico y schema-driven: recorre las propiedades de la tool e incluye las
    marcadas con ``tunable`` (siempre) o ``tunable_if`` (cuando el valor actual
    del discriminador coincide). El razonador no conoce qué parámetro es el
    discriminador de cada tool: esa relación vive en el schema MCP.
    """
    result: list[str] = []
    for name, info in _iter_properties(_get_tool(tool_name)).items():
        if not isinstance(info, dict):
            continue
        if info.get("tunable") is True:
            result.append(name)
            continue
        cond = info.get("tunable_if")
        if isinstance(cond, dict):
            for disc, values in cond.items():
                if isinstance(values, (list, tuple)) and provided_args.get(disc) in values:
                    result.append(name)
                    break
    return result


def get_missing_tunable_params(tool_name: str, provided_args: dict) -> list[str]:
    """Devuelve los tunables que no han sido fijados explícitamente por el usuario."""
    tunables = get_tunable_params(tool_name, provided_args)
    return [
        p for p in tunables
        if p not in provided_args or provided_args[p] is None or provided_args[p] == ""
    ]


def _describe_param_with_options(tool_name: str, param: str) -> str:
    """Descripción del parámetro + sus opciones válidas (enum) si las tiene.

    Como el system prompt ya no lista los parámetros de cada herramienta
    (el agente usa schema fino), este nodo es el único punto donde el usuario
    ve los valores admitidos. Si el parámetro es un enum (``Literal[...]``),
    los añadimos para que la pregunta sea autosuficiente.
    """
    desc = get_param_description(tool_name, param)
    enum = get_param_enum(tool_name, param)
    if enum:
        return f"{desc} (opciones: {', '.join(enum)})"
    return desc


def _format_required_message(
    tool_name: str,
    missing: list[str],
    missing_groups: list[list[str]] | None = None,
) -> str:
    lines: list[str] = [
        "Para continuar necesito algunos datos adicionales. Por favor, proporciona:"
    ]
    for param in missing:
        lines.append(f"  • **{param}**: {_describe_param_with_options(tool_name, param)}")
    for group in missing_groups or []:
        alternativas = " **o** ".join(
            f"**{p}** ({_describe_param_with_options(tool_name, p)})" for p in group
        )
        lines.append(f"  • Uno de: {alternativas}")
    return "\n".join(lines)


def _format_optional_confirmation_message(
    tool_name: str,
    missing_tunable: list[str],
    provided_args: dict | None = None,
) -> str:
    lines: list[str] = [
        f"Voy a ejecutar **{tool_name}** y hay varios parámetros opcionales que aún no has indicado. "
        "Estos son sus valores por defecto:",
        "",
    ]
    for param in missing_tunable:
        desc = get_param_description(tool_name, param)
        default = get_param_default_text(tool_name, param, provided_args)
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
            content=_format_required_message(pending_tool, missing_required, missing_groups),
        )]}

    missing_tunable = get_missing_tunable_params(pending_tool, collected)
    if missing_tunable:
        return {"messages": [AIMessage(
            content=_format_optional_confirmation_message(pending_tool, missing_tunable, collected),
        )]}

    return {}

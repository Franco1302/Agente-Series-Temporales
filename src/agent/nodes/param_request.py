"""Nodo de recogida guiada de parámetros: pide los obligatorios ausentes y confirma los opcionales (tunables) antes de ejecutar."""

from __future__ import annotations

import re
from functools import lru_cache

from langchain_core.messages import AIMessage

from src.agent.state import AgentState
from src.agent.tools import AGENT_TOOLS


def _derive_required_from_schema(tool) -> list[str]:
    """Lee el schema Pydantic de la tool y devuelve los parámetros sin default."""
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
    """Construye el mapa tool_name → required leyendo AGENT_TOOLS una sola vez."""
    return {t.name: _derive_required_from_schema(t) for t in AGENT_TOOLS}


# Se expone para llamadores externos (tests, scripts) pero su contenido se deriva del schema.
TOOL_REQUIRED_PARAMS: dict[str, list[str]] = _build_required_map()


@lru_cache(maxsize=1)
def _tools_by_name() -> dict[str, object]:
    return {t.name: t for t in AGENT_TOOLS}


def _get_tool(tool_name: str):
    return _tools_by_name().get(tool_name)


def _iter_properties(tool) -> dict[str, dict]:
    """Normaliza el schema de la tool a {param: {description, default, enum, type}}."""
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
            # Propaga las claves de json_schema_extra (oneof_group, etc.) tanto para tools MCP como Pydantic.
            extra = getattr(field, "json_schema_extra", None)
            if isinstance(extra, dict):
                for k, v in extra.items():
                    entry.setdefault(k, v)
            out[name] = entry
        return out
    return {}


# Grupos XOR derivados del schema: cada miembro se declara con Field(json_schema_extra={"oneof_group": "<nombre>"}).
@lru_cache(maxsize=1)
def _build_alternative_groups_map() -> dict[str, list[list[str]]]:
    """Construye tool_name -> [[param, …], …] leyendo oneof_group del schema."""
    result: dict[str, list[list[str]]] = {}
    for tool in AGENT_TOOLS:
        groups: dict[str, list[str]] = {}
        for name, info in _iter_properties(tool).items():
            grp = info.get("oneof_group") if isinstance(info, dict) else None
            if isinstance(grp, str) and grp:
                groups.setdefault(grp, []).append(name)
        # Orden alfabético dentro de cada grupo para una salida estable.
        non_trivial = [sorted(members) for members in groups.values() if len(members) >= 2]
        if non_trivial:
            result[tool.name] = sorted(non_trivial)
    return result


# Se expone para tests/scripts externos, pero su contenido viene del schema MCP.
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
    """Descripción de un parámetro leída del schema de la tool, o un fallback genérico si no la expone."""
    props = _iter_properties(_get_tool(tool_name))
    desc = props.get(param, {}).get("description")
    if isinstance(desc, str) and desc.strip():
        return desc.strip()
    return f"el valor de '{param}'"


def get_param_default_text(tool_name: str, param: str, provided_args: dict | None = None) -> str:
    """Texto legible del default de un parámetro: default_by condicional, default_note textual o el default normal del schema."""
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
    """Devuelve el schema normalizado {param: {description, default, enum, evidence, …}} para que otros nodos deriven su metadata sin duplicarla."""
    return _iter_properties(_get_tool(tool_name))


# Tunables: opcionales con default que afectan al algoritmo (umbrales, bins, coeficientes…); se marcan en el schema con
# json_schema_extra {"tunable": true} o {"tunable_if": {"<disc>": [valores]}} y se confirman con el usuario antes de ejecutar.
def _is_empty(value: object) -> bool:
    """True si el valor cuenta como ausente (None, cadena vacía o lista vacía)."""
    return value is None or value == "" or value == []


def get_missing_params(tool_name: str, provided_args: dict) -> list[str]:
    """Devuelve los parámetros obligatorios que faltan o están vacíos en provided_args."""
    required = TOOL_REQUIRED_PARAMS.get(tool_name, [])
    return [p for p in required if _is_empty(provided_args.get(p))]


def get_missing_alternative_groups(tool_name: str, provided_args: dict) -> list[list[str]]:
    """Devuelve los grupos "uno-de" que ningún parámetro satisface; la regla XOR estricta la valida la propia API."""
    groups = TOOL_ALTERNATIVE_GROUPS.get(tool_name, [])
    return [
        group for group in groups
        if all(_is_empty(provided_args.get(p)) for p in group)
    ]


def get_tunable_params(tool_name: str, provided_args: dict) -> list[str]:
    """Devuelve los opcionales relevantes: los marcados tunable (siempre) o tunable_if (si el valor del discriminador coincide)."""
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
    """Descripción del parámetro más sus opciones válidas (enum) si las tiene; es el único punto donde el usuario ve los valores admitidos."""
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


# Parser determinista de la respuesta del usuario a la confirmación de opcionales (evita que el LLM local pierda los args originales al re-emitir la tool call).

# Detecta intención de cancelar: "cancela/olvida/déjalo/para" o "no quiero/no ejecutes/…".
_CANCEL_PATTERN = re.compile(
    r"\b(cancela|olvida|d[eé]jalo|para)\b|"
    r"\bno\s+(quiero|deseo|hagas|ejecutes|detectes|lo\s+hagas|continues|sigas)\b",
    re.IGNORECASE,
)

# Captura pares nombre=valor o nombre: valor (números con signo/decimales/notación científica, listas entre corchetes y strings simples).
_VALUE_PAIR_RE = re.compile(
    r"(?P<name>[A-Za-z_][A-Za-z_0-9]*)\s*[=:]\s*"
    r"(?P<value>\[[^\]]*\]|'[^']*'|\"[^\"]*\"|[\-+]?\d+\.?\d*(?:[eE][\-+]?\d+)?|[A-Za-z_][A-Za-z_0-9]*)"
)


def is_cancel_intent(user_message: str) -> bool:
    """Detecta si el mensaje del usuario expresa intención de cancelar la ejecución."""
    return bool(_CANCEL_PATTERN.search(user_message))


def parse_optional_values(user_message: str, tunables: list[str]) -> dict:
    """Extrae pares nombre=valor del mensaje, solo para los tunables conocidos, convirtiendo números, listas y strings entrecomillados a su tipo."""
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
    """Pide los datos que faltan: pregunta por los obligatorios ausentes o, si solo faltan tunables, lista los defaults y pide confirmación."""
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

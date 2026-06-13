"""Nodo de razonamiento: invoca el LLM con las herramientas enlazadas."""

from __future__ import annotations

import json
import re
import time
import unicodedata
import uuid
from functools import lru_cache
from typing import Callable

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

from src.agent.nodes.param_request import (
    TOOL_ALTERNATIVE_GROUPS,
    get_field_metadata,
    get_missing_alternative_groups,
    get_missing_params,
    get_missing_tunable_params,
    get_tunable_params,
    is_cancel_intent,
    parse_optional_values,
)
from src.agent.param_families import INHERITABLE_PARAMS
from src.agent.prompts import ANALYTICAL_TOOL_NAMES, build_system_prompt
from src.agent.state import AgentState
from src.agent.tool_metadata import MUST_CITE_FIELDS
from src.agent.tools import AGENT_TOOLS
from src.config.llm_config import get_llm_with_tools, load_ollama_settings, thin_tool_schemas
from src.observability import emit_llm_call

_MAX_ERRORS = 3

_KNOWN_TOOL_NAMES = {t.name for t in AGENT_TOOLS}

# Schema cacheado: nombres de parámetros que la firma de cada tool acepta (para descartar herencias que el ToolNode rechazaría).
_TOOL_ACCEPTED_PARAMS: dict[str, frozenset[str]] = {}
for _t in AGENT_TOOLS:
    _schema = getattr(_t, "args_schema", None)
    if _schema is not None and hasattr(_schema, "model_fields"):
        _TOOL_ACCEPTED_PARAMS[_t.name] = frozenset(_schema.model_fields.keys())
    elif isinstance(_schema, dict):
        _TOOL_ACCEPTED_PARAMS[_t.name] = frozenset((_schema.get("properties") or {}).keys())
    else:
        _TOOL_ACCEPTED_PARAMS[_t.name] = frozenset()
def _last_tool_message_name(messages: list) -> str | None:
    """Devuelve el nombre del último ToolMessage del historial, o None."""
    for msg in reversed(messages):
        if isinstance(msg, ToolMessage):
            return getattr(msg, "name", None)
        if isinstance(msg, (AIMessage,)):
            # AIMessage de texto plano (sin tool_calls): el ciclo anterior ya cerró.
            if not getattr(msg, "tool_calls", None):
                return None
    return None


def _build_fuentes_section(messages: list) -> str | None:
    """Extrae el bloque «Fuentes consultadas» del último ToolMessage de consultar_teoria."""
    for msg in reversed(messages):
        if isinstance(msg, ToolMessage) and getattr(msg, "name", None) == "consultar_teoria":
            content = msg.content if isinstance(msg.content, str) else str(msg.content)
            marker = "Fuentes consultadas:"
            idx = content.find(marker)
            if idx == -1:
                return None
            lineas = [
                ln.strip()
                for ln in content[idx + len(marker):].splitlines()
                if ln.strip().startswith("-")
            ]
            return "Fuentes:\n" + "\n".join(lineas) if lineas else None
    return None


def _append_fuentes(response: AIMessage, messages: list) -> None:
    """Anexa de forma determinista la sección Fuentes a la respuesta final."""
    content = response.content
    if not isinstance(content, str) or not content.strip():
        return
    if "Fuentes:" in content:  # el modelo ya la incluyó: no duplicar
        return
    fuentes = _build_fuentes_section(messages)
    if fuentes:
        response.content = content.rstrip() + "\n\n" + fuentes


def _last_analytical_tool_message(messages: list) -> ToolMessage | None:
    """Devuelve el último ToolMessage de una herramienta analítica, o None."""
    for msg in reversed(messages):
        if isinstance(msg, ToolMessage):
            if getattr(msg, "name", None) in ANALYTICAL_TOOL_NAMES:
                return msg
            return None
        if isinstance(msg, AIMessage) and not getattr(msg, "tool_calls", None):
            return None
    return None


def _extract_must_cite_facts(tool_msg: ToolMessage) -> list[str]:
    """Extrae del ToolMessage los valores deterministas que deben aparecer citados."""
    try:
        raw = tool_msg.content
        data = json.loads(raw) if isinstance(raw, str) else raw
        if not isinstance(data, dict):
            return []
        fields = MUST_CITE_FIELDS.get(getattr(tool_msg, "name", "") or "", ())
        facts: list[str] = []
        for key in fields:
            value = data.get(key)
            if isinstance(value, bool):
                continue
            if isinstance(value, (str, int)):
                texto = str(value).strip()
                if texto:
                    facts.append(texto)
        return facts
    except Exception:  # noqa: BLE001 — la verificación nunca debe abortar el grafo
        return []


def _missing_facts(response_text: str, facts: list[str]) -> list[str]:
    """Devuelve los facts que NO aparecen como substring (case-insensitive)."""
    texto = response_text.lower()
    return [f for f in facts if f.lower() not in texto]


def _verify_and_repair(response: AIMessage, messages: list) -> AIMessage:
    """Verifica la fidelidad numérica y reintenta una vez (sin tools) si la síntesis omite valores del ToolMessage; conserva el reintento solo si mejora."""
    try:
        content = response.content
        if not isinstance(content, str) or not content.strip():
            return response

        tool_msg = _last_analytical_tool_message(messages)
        if tool_msg is None:
            return response

        facts = _extract_must_cite_facts(tool_msg)
        missing = _missing_facts(content, facts)
        if not missing:
            return response

        correccion = SystemMessage(content=(
            "Tu borrador de respuesta omitió estos valores exactos del resultado "
            f"de la herramienta: {', '.join(missing)}. Genera de nuevo la respuesta "
            "completa con los tres bloques (**RESULTADO:**, **INTERPRETACIÓN:**, "
            "**SIGUIENTE PASO:**) y cita esos valores literalmente dentro de "
            "**RESULTADO:**."
        ))
        llm = get_llm_with_tools([])
        t0 = time.perf_counter()
        retry = llm.invoke(messages + [correccion])
        duration_ms = (time.perf_counter() - t0) * 1000.0
        emit_llm_call(
            name="razonador.fidelidad_retry",
            messages=messages + [correccion],
            response=retry,
            duration_ms=duration_ms,
        )

        if (
            isinstance(retry.content, str)
            and retry.content.strip()
            and len(_missing_facts(retry.content, facts)) < len(missing)
        ):
            return retry
        return response
    except Exception:  # noqa: BLE001 — la verificación nunca debe abortar el grafo
        return response


# Defensa anti-invención: el modelo cuantizado rellena campos por sentido común ante peticiones vagas. Para cada (tool, arg)
# de _FIELD_EVIDENCE_MAP, si la evidencia no aparece en el historial del usuario el arg se elimina y se pide como ausente.


# Patrones de evidencia textual 

_DATE_EVIDENCE_RE = re.compile(
    r"\b(?:19|20)\d{2}\b"
    r"|\b(?:fecha|inicio|desde|comienza|empieza|empezando|partir\s+de|hasta|fin)"
    r"|\b(?:ener|febrer|marzo|abril|mayo|junio|julio|agost|septiembr|octubr|noviembr|diciembr)",
    re.IGNORECASE,
)

_FREQUENCY_KEYWORDS: tuple[str, ...] = (
    "diari", "semanal", "mensu", "trimestr", "anual", "año", "anos",
    "horari", "hora", "minuto", "segundo",
    "cada hora", "cada día", "cada dia", "cada semana", "cada mes", "cada año",
    "freq", "frecuencia",
)

# Dígito o número en palabras (desde "dos") cuenta como evidencia; "uno/una/un" se excluye por ser artículo indefinido y dar falsos positivos.
_INTEGER_EVIDENCE_RE = re.compile(
    r"\b\d+\b"
    r"|\b(?:dos|tres|cuatro|cinco|seis|siete|ocho|nueve|diez|once|doce|"
    r"trece|catorce|quince|diecis[eé]is|veinte|treinta|cuarenta|cincuenta|"
    r"sesenta|setenta|ochenta|noventa|cien|mil)\b",
    re.IGNORECASE,
)

# Listas numéricas: corchetes con dígitos, decimales, pares por coma, asignaciones letra=número o palabras de parámetro.
_NUMERIC_LIST_EVIDENCE_RE = re.compile(
    r"[\[\(]\s*[-+]?\d"
    r"|\b-?\d+[\.,]\d+\b"
    r"|\b-?\d+\s*,\s*-?\d+\b"
    r"|\b[a-záéíóú]{1,6}\s*=\s*-?\d"
    r"|\b(?:par[aá]metros?|coeficientes?|valores?|param|lista|sigma|desv|"
    r"mu|media|lambda|prob|probabilidad)\b",
    re.IGNORECASE,
)

# Frases con las que el usuario delega la elección de un valor ("usa los defaults", "cualquiera", …): desactivan la defensa anti-invención ese turno para no entrar en bucle.
_DELEGATION_KEYWORDS: tuple[str, ...] = (
    "default", "predetermin", "est[aá]ndar", "t[ií]pico", "habitual",
    "lo que sea", "lo que t[uú] quieras", "lo que quieras",
    "lo que prefieras", "como sea", "como t[uú] quieras", "como quieras",
    "que t[uú] quieras", "que quieras",
    "cualquier", "no importa", "no me importa", "me da igual",
    "elige t[uú]", "elige tu", "decide t[uú]", "decide tu",
    "tu eliges", "t[uú] eliges", "tu decides", "t[uú] decides",
    "gen[eé]ralos t[uú]", "haz t[uú]", "genera t[uú]",
)


def _user_delegates(user_text: str) -> bool:
    """True si el usuario delegó explícitamente la elección de algún valor."""
    return _has_any_keyword(user_text, _DELEGATION_KEYWORDS)


def _build_delegation_directive(
    pending_tool: str | None,
    missing: list[str],
) -> SystemMessage:
    """Directiva que se inyecta al LLM cuando el usuario delega."""
    if pending_tool and missing:
        body = (
            f"El usuario acaba de delegar explícitamente la elección de los "
            f"parámetros que faltan para {pending_tool}. DEBES invocar la "
            f"herramienta AHORA emitiendo el JSON de la tool call. Para los "
            f"parámetros faltantes ({', '.join(missing)}) propón TÚ valores "
            f"razonables y justifica brevemente tu elección en una frase "
            f"antes del JSON. NO preguntes nada al usuario: ya ha dicho que "
            f"confía en tu criterio."
        )
    elif pending_tool:
        body = (
            f"El usuario ha delegado. DEBES invocar {pending_tool} AHORA con "
            f"los parámetros ya recogidos; no preguntes más."
        )
    else:
        body = (
            "El usuario acaba de delegar explícitamente la elección de "
            "parámetros. DEBES invocar AHORA la herramienta más adecuada a "
            "su petición, emitiendo el JSON de tool call con valores "
            "razonables que tú elijas. Justifica brevemente tu elección en "
            "una frase antes del JSON. NO preguntes al usuario."
        )
    return SystemMessage(content=body)


# Frases con las que el usuario pide una operación NUEVA reutilizando parámetros de un turno anterior ("los mismos parámetros pero…", "como antes pero…").
_FOLLOWUP_REUSE_KEYWORDS: tuple[str, ...] = (
    "anteriores", "de antes", "los de antes",
    "mismos par[áa]metros", "esos par[áa]metros", "los mismos", "las mismas",
    "como antes", "igual que antes", "reutiliz", "reusa", "vuelve a usar",
    "c[oó]ge los", "usa los", "con esos",
)


def _is_inheriting_followup(user_text: str, session_facts: dict) -> bool:
    """True si el usuario pide algo nuevo reutilizando parámetros ya conocidos: requiere by_param no vacío y una referencia explícita a reutilizar valores previos."""
    by_param = (session_facts or {}).get("by_param") or {}
    if not by_param:
        return False
    return _has_any_keyword(user_text, _FOLLOWUP_REUSE_KEYWORDS)


def _build_followup_directive(session_facts: dict) -> SystemMessage:
    """Directiva que fuerza la tool call en un seguimiento que hereda parámetros: indica al LLM que emita el JSON ahora; la herencia genérica rellenará los parámetros."""
    known = ", ".join(sorted((session_facts or {}).get("by_param", {}).keys()))
    body = (
        "El usuario pide una operación NUEVA reutilizando parámetros ya "
        "conocidos en esta conversación (ver [CONTEXTO DE SESIÓN]"
        + (f": {known}" if known else "")
        + "). DEBES identificar la herramienta adecuada a su nueva petición y "
        "emitir AHORA el JSON de la tool call, heredando esos parámetros. NO "
        "respondas en texto ni repitas la plantilla de petición de datos: si "
        "algún parámetro obligatorio nuevo falta de verdad, emite igualmente la "
        "tool call con lo que tengas y el sistema lo pedirá."
    )
    return SystemMessage(content=body)


# Petición de acción analítica que el modelo debe resolver con tool call: si responde en prosa pidiendo parámetros,
# reintentamos una vez forzando la llamada con contexto compacto + directiva (tool_choice="any" solo no basta con este modelo).
_ACTION_INTENT_KEYWORDS: tuple[str, ...] = (
    "sint[eé]tic", "serie aleatoria", "datos aleatorios", "simula",
    "genera", "gener[aá]", "crea", "cr[eé]a",
    "drift", "deriva", "ha cambiado", "estabilidad",
    "aument", "ampliar", "m[aá]s observaciones", "m[aá]s datos",
    "forecast", "predic", "predec", "pron[oó]stic", "sarimax", "futuro",
    "detecta", "ex[oó]gena", "correlaci", "covar",
)

# Marcadores de pregunta teórica o de capacidades: si aparecen, NO forzamos la tool call (se resuelven por el flujo normal o en texto plano).
_NON_ACTION_MARKERS: tuple[str, ...] = (
    "qu[eé] es", "qu[eé] son", "qu[eé] significa", "explica", "expl[ií]ca",
    "diferencia", "c[oó]mo funciona", "para qu[eé] sirve", "concepto",
    "definici[oó]n", "qu[eé] puedes", "qu[eé] sabes", "ay[uú]dame", "ayuda",
)


def _looks_like_action_request(text: str) -> bool:
    """True si el mensaje es una petición de acción analítica clara (intención de acción y no una pregunta teórica/de capacidades). Gate del reintento que fuerza la tool call."""
    if not text:
        return False
    if _has_any_keyword(text, _NON_ACTION_MARKERS):
        return False
    return _has_any_keyword(text, _ACTION_INTENT_KEYWORDS)


# Marcadores de generación que NO necesita fichero (las 4 generate_synthetic_*): sin CSV solo forzamos la tool call para estas.
_NO_FILE_GENERATION_MARKERS: tuple[str, ...] = (
    "sint[eé]tic", "serie aleatoria", "datos aleatorios", "simula",
)


def _should_force_action(text: str, csv_loaded: bool) -> bool:
    """True si debemos reintentar forzando la tool call: requiere intención de acción y, sin CSV, que sea una generación sintética."""
    if not _looks_like_action_request(text):
        return False
    return bool(csv_loaded) or _has_any_keyword(text, _NO_FILE_GENERATION_MARKERS)


def _build_action_directive() -> SystemMessage:
    """Directiva que fuerza la tool call ante una petición de acción en prosa: empuja al LLM a emitir el JSON ya, sugiriendo generate_synthetic_distribution si la serie no se concreta."""
    return SystemMessage(content=(
        "El usuario pide una operación analítica. DEBES emitir AHORA el JSON de la "
        "tool call de la herramienta más adecuada a su petición (p. ej. un "
        "'forecast' → forecast_time_series; una 'serie' o 'datos sintéticos' sin "
        "concretar el tipo → generate_synthetic_distribution). Si hay FICHERO "
        "ACTIVO, usa su ruta como file_path. Pasa SOLO los parámetros que el "
        "usuario haya escrito explícitamente; si faltan, emite igualmente la tool "
        "call con lo que tengas (arguments={} si no hay ninguno) y el sistema "
        "pedirá el resto. NO preguntes parámetros en texto."
    ))


# Pregunta teórica que el agente debe fundamentar con RAG: si el modelo responde de su conocimiento sin consultar la documentación,
# sintetizamos de forma determinista la llamada a consultar_teoria para garantizar grounding y citas.
_THEORY_QUESTION_MARKERS: tuple[str, ...] = (
    "qu[eé] es", "qu[eé] son", "qu[eé] significa", "qu[eé] mide",
    "explica", "expl[ií]ca", "explicame", "expl[ií]came",
    "definici[oó]n", "concepto", "en qu[eé] consiste",
    "diferencia entre", "diferencias entre",
    "c[oó]mo funciona", "para qu[eé] sirve", "por qu[eé] se usa",
)

# Si la pregunta referencia los DATOS del usuario (CSV, columnas, resultados…) NO es teórica: la resuelve el flujo normal, no el RAG.
_DATA_REFERENCE_MARKERS: tuple[str, ...] = (
    "mi csv", "mis datos", "mi fichero", "mi archivo", "mi serie", "mi dataset",
    "el fichero", "el archivo", "este dataset",
    "columna", "fila", "sub[ií]", "cargu[eé]",
    "resultado", "predicci", "gr[aá]fica",
)


def _looks_like_theory_question(text: str) -> bool:
    """True si el mensaje es una pregunta conceptual sobre teoría (marcador conceptual y sin referencias a los datos del usuario). Gate del forzado de consultar_teoria."""
    if not text:
        return False
    if _has_any_keyword(text, _DATA_REFERENCE_MARKERS):
        return False
    return _has_any_keyword(text, _THEORY_QUESTION_MARKERS)


# Mapas de palabras-clave para los enums semánticos: detectar que el usuario nombró alguna opción relacionada.
_DISTRIBUTION_NAMES: tuple[str, ...] = (
    "normal", "gaussian", "gauss", "poisson", "uniform", "uniforme", "beta",
    "gamma", "exponen", "binomial", "chi", "student", "t-student",
    "geomet", "lognor", "weibull", "tipo", "distribu",
)
_TREND_NAMES: tuple[str, ...] = (
    "lineal", "linear", "polinom", "polynom", "exponen", "logarit",
    "constante", "constant", "potencia", "power", "sinusoid", "seno",
    "tendencia", "crecient", "decrecient",
)
_PATTERN_NAMES: tuple[str, ...] = (
    "amplitud", "cantidad", "patr[oó]n", "patron", "variaci[oó]n",
    "variacion", "ciclo", "estacional",
)
_DRIFT_METHOD_NAMES: tuple[str, ...] = (
    "ks", "kolmogorov", "smirnov",
    "js", "jensen", "shannon",
    "psi",
    "cusum", "suma acumul",
    "mewma", "ewma",
    "hotelling", "t2", "t²", "t cuadrado",
    "m[eé]todo", "metodo", "test", "univar", "multivar",
)
_AUGMENT_STRATEGY_NAMES: tuple[str, ...] = (
    "normal", "muller", "box-muller", "duplica", "duplicar",
    "harmoni", "arm[oó]ni", "statistical", "estad[ií]sti",
    "estrategia",
)
_EXOGENOUS_RELATION_NAMES: tuple[str, ...] = (
    "correla", "covar", "relaci[oó]n",
)
# Mención (a favor o en contra) de la gráfica/imagen: evidencia que habilita respetar el with_plot del modelo; sin ella se descarta y aplica el default True del schema.
_PLOT_PREF_NAMES: tuple[str, ...] = (
    "gr[aá]fic", "grafic", "imagen", "im[aá]genes", "plot", "visualiz",
    "dibuj", "png", "chart", "figura", "represent",
)


def _has_any_keyword(text: str, keywords: tuple[str, ...]) -> bool:
    """True si alguna keyword (regex parcial, case-insensitive) aparece en text."""
    for kw in keywords:
        if re.search(kw, text, re.IGNORECASE):
            return True
    return False


# ── Checkers por tipo de campo ──────────────────────────────────────────────

def _check_date(user_text: str, value: object, state: dict) -> bool:
    return bool(_DATE_EVIDENCE_RE.search(user_text))


def _check_freq(user_text: str, value: object, state: dict) -> bool:
    text = user_text.lower()
    return any(kw in text for kw in _FREQUENCY_KEYWORDS)


def _check_integer(user_text: str, value: object, state: dict) -> bool:
    return bool(_INTEGER_EVIDENCE_RE.search(user_text))


def _check_numeric_list(user_text: str, value: object, state: dict) -> bool:
    return bool(_NUMERIC_LIST_EVIDENCE_RE.search(user_text))


def _check_distribution_kind(user_text: str, value: object, state: dict) -> bool:
    return _has_any_keyword(user_text, _DISTRIBUTION_NAMES)


def _check_trend_kind(user_text: str, value: object, state: dict) -> bool:
    return _has_any_keyword(user_text, _TREND_NAMES)


def _check_pattern_kind(user_text: str, value: object, state: dict) -> bool:
    return _has_any_keyword(user_text, _PATTERN_NAMES)


def _check_drift_method(user_text: str, value: object, state: dict) -> bool:
    return _has_any_keyword(user_text, _DRIFT_METHOD_NAMES)


def _check_augment_strategy(user_text: str, value: object, state: dict) -> bool:
    return _has_any_keyword(user_text, _AUGMENT_STRATEGY_NAMES)


def _check_exogenous_relation(user_text: str, value: object, state: dict) -> bool:
    return _has_any_keyword(user_text, _EXOGENOUS_RELATION_NAMES)


def _check_plot_pref(user_text: str, value: object, state: dict) -> bool:
    """Evidencia para with_plot: si el usuario no menciona la gráfica se descarta el valor del modelo (→ default True); con mención se respeta ("sin gráfica" → False)."""
    return _has_any_keyword(user_text, _PLOT_PREF_NAMES)


def _check_existing_column(user_text: str, value: object, state: dict) -> bool:
    """Columna que debe existir en el CSV activo (index_column, target_column).

    Acepta cuando el valor aparece literalmente en el texto del usuario O
    cuando coincide (case-insensitive) con alguna columna del CSV cargado.
    Si no hay CSV ni mención textual, se considera invención.
    """
    if not isinstance(value, str) or not value:
        return True
    if value.lower() in user_text.lower():
        return True
    csv_meta = (state.get("csv_metadata") or {}) if state else {}
    columns = csv_meta.get("columns") or []
    return any(str(c).lower() == value.lower() for c in columns)


def _check_new_column(user_text: str, value: object, state: dict) -> bool:
    """Columna NUEVA a crear (create_exogenous_variable.new_column_name).

    El usuario debe haber escrito el nombre exacto. No vale fallback contra
    csv_metadata porque la columna aún no existe.
    """
    if not isinstance(value, str) or not value:
        return True
    return value.lower() in user_text.lower()


_CHECKS: dict[str, Callable[[str, object, dict], bool]] = {
    "date": _check_date,
    "freq": _check_freq,
    "integer": _check_integer,
    "numeric_list": _check_numeric_list,
    "distribution_kind": _check_distribution_kind,
    "trend_kind": _check_trend_kind,
    "pattern_kind": _check_pattern_kind,
    "drift_method": _check_drift_method,
    "augment_strategy": _check_augment_strategy,
    "exogenous_relation": _check_exogenous_relation,
    "existing_column": _check_existing_column,
    "new_column": _check_new_column,
    "plot_pref": _check_plot_pref,
}


# Mapa (tool, arg) → tipo de evidencia, derivado del schema: por cada tool analítica se recoge la clave evidence de sus Field.
# file_path queda fuera a propósito (la sección FICHERO ACTIVO del prompt lo proporciona desde el estado).
def _build_field_evidence_map() -> dict[str, dict[str, str]]:
    out: dict[str, dict[str, str]] = {}
    for tool_name in ANALYTICAL_TOOL_NAMES:
        evidence: dict[str, str] = {}
        for param, info in get_field_metadata(tool_name).items():
            ev = info.get("evidence") if isinstance(info, dict) else None
            if isinstance(ev, str) and ev:
                evidence[param] = ev
        if evidence:
            out[tool_name] = evidence
    return out


_FIELD_EVIDENCE_MAP: dict[str, dict[str, str]] = _build_field_evidence_map()


def _aggregate_human_text(messages: list) -> str:
    """Concatena el contenido de todos los HumanMessage del historial."""
    parts: list[str] = []
    for msg in messages:
        if isinstance(msg, HumanMessage):
            content = msg.content
            parts.append(content if isinstance(content, str) else str(content))
    return " ".join(parts)


def _strip_invented_args(response: AIMessage, state: dict) -> AIMessage:
    """Elimina de la tool call los args inventados sin evidencia en el texto del usuario; los recogidos en turnos previos se restauran luego en el merge. Recibe state porque algunos checkers consultan csv_metadata."""
    tool_calls = getattr(response, "tool_calls", None) or []
    if not tool_calls:
        return response

    call = tool_calls[0]
    tool_name = call.get("name", "") if isinstance(call, dict) else getattr(call, "name", "")
    type_map = _FIELD_EVIDENCE_MAP.get(tool_name)
    if not type_map:
        return response

    args = dict(call.get("args", {}) if isinstance(call, dict) else getattr(call, "args", {}) or {})
    messages = state.get("messages", []) if state else []
    user_text = _aggregate_human_text(messages)
    if not user_text:
        return response

    # Bypass por delegación: si el ÚLTIMO mensaje del usuario delega, la defensa se desactiva ese turno y dejamos que el LLM proponga un valor.
    last_human = ""
    for msg in reversed(messages):
        if isinstance(msg, HumanMessage):
            last_human = msg.content if isinstance(msg.content, str) else str(msg.content)
            break
    if last_human and _user_delegates(last_human):
        return response

    stripped: list[str] = []
    for arg_name, field_type in type_map.items():
        if arg_name not in args:
            continue
        if args[arg_name] in (None, "", []):
            continue
        checker = _CHECKS.get(field_type)
        if checker is None:
            continue
        if not checker(user_text, args[arg_name], state or {}):
            del args[arg_name]
            stripped.append(arg_name)

    if not stripped:
        return response

    call_id = (
        call.get("id") if isinstance(call, dict) else getattr(call, "id", None)
    ) or f"call_{uuid.uuid4().hex[:12]}"
    return AIMessage(
        content=response.content,
        tool_calls=[{
            "name": tool_name,
            "args": args,
            "id": call_id,
            "type": "tool_call",
        }],
        additional_kwargs=getattr(response, "additional_kwargs", {}) or {},
    )


# Resolución determinista de enums "código=nombre" (distribution_type, …): el modelo a veces yerra el código entero con schema fino.
# Parsea el mapa código→nombre de la propia descripción del schema y, si el usuario nombró una opción, fija el código canónico.

_ENUM_CODE_PATTERN = re.compile(r"(\d+)\s*=\s*([A-Za-zÀ-ÿ]+)")

# Alias de idioma (el schema dice "Normal"; el usuario puede decir "gaussiana"): metadato lingüístico, no del contrato de la API.
_ENUM_NAME_ALIASES: dict[str, dict[str, tuple[str, ...]]] = {
    "distribution_type": {
        "normal": ("gaussian", "gauss", "gaussiana", "gaussiano"),
        "uniforme": ("uniform",),
        "exponencial": ("exponential",),
        "tstudent": ("t-student", "t student", "student"),
        "chicuadrado": ("chi-cuadrado", "chi cuadrado", "ji cuadrado"),
        "geometrica": ("geometric",),
    },
    "trend_type": {
        "lineal": ("linear", "recta"),
        "polinomica": ("polynomial", "polinomial", "polinomio"),
        "logaritmica": ("logarithmic", "logaritmo"),
    },
}


def _strip_accents(text: str) -> str:
    """Minúsculas sin tildes, para casar nombres independientemente de la grafía."""
    nfd = unicodedata.normalize("NFD", text.lower())
    return "".join(c for c in nfd if unicodedata.category(c) != "Mn")


@lru_cache(maxsize=1)
def _enum_code_maps() -> dict[str, dict[str, int]]:
    """{param: {nombre_sin_tildes: código}} derivado de las descripciones del schema (pares "N=Nombre", la más rica entre las tools), más los alias de _ENUM_NAME_ALIASES."""
    best: dict[str, dict[str, int]] = {}
    for tool in AGENT_TOOLS:
        for param, info in get_field_metadata(tool.name).items():
            if not isinstance(info, dict):
                continue
            pairs = _ENUM_CODE_PATTERN.findall(info.get("description", "") or "")
            if len(pairs) < 2:
                continue
            names = [_strip_accents(name) for _, name in pairs]
            # Si dos códigos comparten el primer término (p. ej. pattern_type: ambos "variación") no es resoluble por nombre: descartamos.
            if len(set(names)) != len(names):
                continue
            mapping = {name: int(code) for name, (code, _) in zip(names, pairs)}
            if len(mapping) > len(best.get(param, {})):
                best[param] = mapping

    for param, aliases in _ENUM_NAME_ALIASES.items():
        mapping = best.get(param)
        if not mapping:
            continue
        for canonical, syns in aliases.items():
            code = mapping.get(_strip_accents(canonical))
            if code is None:
                continue
            for syn in syns:
                mapping.setdefault(_strip_accents(syn), code)
    return best


def _match_enum(user_text: str, name_to_code: dict[str, int]) -> int | None:
    """Código canónico que el usuario nombró, o None: devuelve un código solo si exactamente un valor del enum aparece en el texto."""
    norm = _strip_accents(user_text)
    found = {code for name, code in name_to_code.items() if name in norm}
    return next(iter(found)) if len(found) == 1 else None


def _resolve_enum_codes(response: AIMessage, state: AgentState) -> AIMessage:
    """Corrige/fija los enums código=nombre de la tool call según lo que el usuario nombró (solo en parámetros que la tool acepta y con evidencia textual inequívoca)."""
    tool_calls = getattr(response, "tool_calls", None) or []
    if not tool_calls:
        return response

    call = tool_calls[0]
    tool_name = call.get("name", "") if isinstance(call, dict) else getattr(call, "name", "")
    accepted = _TOOL_ACCEPTED_PARAMS.get(tool_name, frozenset())
    relevant = {p: m for p, m in _enum_code_maps().items() if p in accepted}
    if not relevant:
        return response

    user_text = _aggregate_human_text(state.get("messages", []) if state else [])
    if not user_text:
        return response

    args = dict(call.get("args", {}) if isinstance(call, dict) else getattr(call, "args", {}) or {})
    changed = False
    for param, name_to_code in relevant.items():
        resolved = _match_enum(user_text, name_to_code)
        if resolved is not None and args.get(param) != resolved:
            args[param] = resolved
            changed = True

    if not changed:
        return response

    call_id = (
        call.get("id") if isinstance(call, dict) else getattr(call, "id", None)
    ) or f"call_{uuid.uuid4().hex[:12]}"
    return AIMessage(
        content=response.content,
        tool_calls=[{
            "name": tool_name,
            "args": args,
            "id": call_id,
            "type": "tool_call",
        }],
        additional_kwargs=getattr(response, "additional_kwargs", {}) or {},
    )


def _drop_empty_args(args: dict) -> dict:
    """Elimina los args con valor ausente (None, "" o []): los placeholders vacíos del modelo (p. ej. end_date="" junto a periods) llegarían a la API como "provistos" y la harían fallar con "no ambos"."""
    return {k: v for k, v in args.items() if v not in (None, "", [])}


def _merge_with_pending_params(response: AIMessage, state: AgentState) -> AIMessage:
    """Fusiona los args ya recogidos en pending_params con la nueva tool call (el modelo suele re-emitir solo el último parámetro y olvidar los originales). No fusiona si la tool call es para otra tool distinta a la pendiente."""
    tool_calls = getattr(response, "tool_calls", None) or []
    if not tool_calls:
        return response

    pending_tool = state.get("pending_tool")
    pending_params = state.get("pending_params") or {}
    if not pending_tool or not pending_params:
        return response

    call = tool_calls[0]
    tool_name = call.get("name", "") if isinstance(call, dict) else getattr(call, "name", "")
    if tool_name != pending_tool:
        return response

    new_args = call.get("args", {}) if isinstance(call, dict) else getattr(call, "args", {})
    merged = dict(pending_params)
    for k, v in (new_args or {}).items():
        if v not in (None, ""):
            merged[k] = v

    if merged == new_args:
        return response

    call_id = (
        call.get("id") if isinstance(call, dict) else getattr(call, "id", None)
    ) or f"call_{uuid.uuid4().hex[:12]}"
    return AIMessage(
        content=response.content,
        tool_calls=[{
            "name": tool_name,
            "args": merged,
            "id": call_id,
            "type": "tool_call",
        }],
        additional_kwargs=getattr(response, "additional_kwargs", {}) or {},
    )


def _last_human_message_content(messages: list) -> str:
    """Devuelve el contenido del último HumanMessage como string, o cadena vacía."""
    for msg in reversed(messages):
        if isinstance(msg, HumanMessage):
            content = msg.content
            if isinstance(content, str):
                return content
            return str(content)
    return ""


def _replay_pending_tool_call(state: AgentState) -> dict:
    """Construye la tool call de forma determinista tras la confirmación de opcionales, reusando pending_params + los nombre=valor que el usuario indique. Si el usuario cancela, aborta y limpia el estado pendiente."""
    pending_tool = state.get("pending_tool") or ""
    base_args = dict(state.get("pending_params") or {})
    user_msg = _last_human_message_content(state.get("messages", []))

    tunables = get_tunable_params(pending_tool, base_args)
    explicit = parse_optional_values(user_msg, tunables) if user_msg else {}

    # Cancela solo si hay negativa Y ningún valor explícito ("no quiero ejecutarla" cancela; "no quiero defaults, usa threshold=0.3" sigue).
    if user_msg and not explicit and is_cancel_intent(user_msg):
        return {
            "messages": [AIMessage(
                content=(
                    "De acuerdo, cancelo la ejecución de **{tool}**. "
                    "¿Qué te gustaría hacer en su lugar?"
                ).format(tool=pending_tool)
            )],
            "pending_tool": None,
            "pending_params": None,
            "optionals_confirmed_for": None,
        }

    final_args = {**base_args, **explicit}

    synthetic_call = {
        "name": pending_tool,
        "args": final_args,
        "id": f"call_{uuid.uuid4().hex[:12]}",
        "type": "tool_call",
    }
    tool_call_msg = AIMessage(content="", tool_calls=[synthetic_call])

    return {
        "messages": [tool_call_msg],
        "pending_tool": None,
        "pending_params": None,
        "optionals_confirmed_for": None,
    }


def _trim_to_recent_turns(messages: list, max_turns: int) -> list:
    """Devuelve solo los últimos max_turns turnos (un turno empieza en un HumanMessage); reduce que el LLM imite formatos antiguos. La memoria de parámetros vive en session_facts, que no se trunca."""
    if max_turns <= 0 or not messages:
        return list(messages)
    human_indices = [i for i, m in enumerate(messages) if isinstance(m, HumanMessage)]
    if len(human_indices) <= max_turns:
        return list(messages)
    cut = human_indices[-max_turns]
    return list(messages[cut:])


def _build_session_facts_hint(session_facts: dict) -> str:
    """Genera el bloque [CONTEXTO DE SESIÓN] (una línea por parámetro heredable) que se inyecta en el system prompt; "" si by_param está vacío."""
    by_param = (session_facts or {}).get("by_param") or {}
    if not by_param:
        return ""
    lines = [
        "[CONTEXTO DE SESIÓN]",
        "Parámetros ya conocidos en esta conversación (úsalos en cualquier tool call que los acepte):",
    ]
    for name, fact in by_param.items():
        value = fact.get("value")
        rendered = f'"{value}"' if isinstance(value, str) else str(value)
        lines.append(f"  {name} = {rendered}")
    # Línea en blanco final para separar visualmente del bloque siguiente.
    return "\n".join(lines) + "\n"


def _inherit_from_session(
    tool_name: str,
    args: dict,
    session_facts: dict,
) -> tuple[dict, list[dict]]:
    """Rellena args con parámetros heredables de la sesión: solo los de INHERITABLE_PARAMS que la tool acepte y que el LLM no haya emitido ya. Devuelve (args_enriquecidos, breadcrumbs) con cada herencia aplicada."""
    nonempty = lambda v: v not in (None, "", [])  # noqa: E731
    by_param = (session_facts or {}).get("by_param") or {}
    accepted = _TOOL_ACCEPTED_PARAMS.get(tool_name, frozenset())
    groups = TOOL_ALTERNATIVE_GROUPS.get(tool_name, [])

    def _xor_sibling_already_set(param: str) -> bool:
        """True si otro miembro del grupo XOR de param ya está fijado (heredar periods con end_date ya emitido haría fallar la API con "no ambos")."""
        for grp in groups:
            if param in grp:
                return any(nonempty(enriched.get(m)) for m in grp if m != param)
        return False

    enriched = dict(args)
    inherited: list[dict] = []
    for name, fact in by_param.items():
        if name not in INHERITABLE_PARAMS:
            continue
        if name not in accepted:
            continue
        if nonempty(enriched.get(name)):
            continue
        if _xor_sibling_already_set(name):
            continue
        value = fact.get("value")
        if not nonempty(value):
            continue
        enriched[name] = value
        inherited.append({
            "name": name,
            "value": value,
            "source_tool": fact.get("source_tool"),
        })
    return enriched, inherited


def razonador_node(state: AgentState) -> dict:
    """Invoca el LLM y decide la próxima acción: si la tool call tiene parámetros incompletos registra pending_tool/pending_params. Anti-bucle: si el último ToolMessage es de consultar_teoria, retira esa tool del bind para forzar la síntesis."""
    error_count = state.get("error_count", 0)

    if error_count >= _MAX_ERRORS:
        return {
            "messages": [
                AIMessage(
                    content=(
                        "He alcanzado el límite de reintentos en esta consulta. "
                        "Por favor, reformula tu petición o simplifica la tarea."
                    )
                )
            ],
        }

    # Atajo determinista: si esperábamos la confirmación de opcionales, reconstruimos la tool call sin pasar por el LLM (evita que re-emita sin los obligatorios).
    pending_tool_state = state.get("pending_tool")
    if pending_tool_state and state.get("optionals_confirmed_for") == pending_tool_state:
        return _replay_pending_tool_call(state)

    csv_path = state.get("csv_path")
    csv_metadata = state.get("csv_metadata")

    # Truncamos el historial a los últimos N turnos (CHAT_MAX_CONTEXT_TURNS); los parámetros heredables viven en session_facts, no en el historial.
    max_turns = load_ollama_settings().max_context_turns
    messages = _trim_to_recent_turns(state["messages"], max_turns)

    # Construimos [CONTEXTO DE SESIÓN] con los parámetros heredables; si el modelo lo ignora, la pasada de herencia rellena los args al final.
    session_facts = state.get("session_facts") or {}
    facts_hint = _build_session_facts_hint(session_facts) if session_facts else ""

    # Si el último ToolMessage es de una tool analítica, el prompt añade el bloque RESULTADO / INTERPRETACIÓN / SIGUIENTE PASO (RF-11).
    last_tool = _last_tool_message_name(messages)
    system_prompt = build_system_prompt(
        csv_path=csv_path,
        csv_metadata=csv_metadata,
        tool_result_to_explain=last_tool,
        session_context=facts_hint or None,
    )

    # Inyectar system prompt si el primer mensaje no es ya un SystemMessage
    if not messages or not isinstance(messages[0], SystemMessage):
        messages = [SystemMessage(content=system_prompt)] + messages

    # Directiva de delegación: si el último HumanMessage delega, añadimos una SystemMessage que fuerza al LLM a invocar la tool con valores que él proponga.
    # Guarda anti-bucle: las directivas que FUERZAN una tool call solo actúan al inicio del turno (last_tool sin fijar); con tool ejecutada toca sintetizar, no forzar otra.
    last_human_text = _last_human_message_content(messages)
    force_tool_call = False
    if not last_tool and last_human_text and _user_delegates(last_human_text):
        pending_params = state.get("pending_params") or {}
        missing = (
            get_missing_params(pending_tool_state, pending_params)
            if pending_tool_state else []
        )
        # Inyectar siempre que haya delegación: la directiva se adapta a si hay pending_tool y a qué params faltan.
        messages = messages + [_build_delegation_directive(pending_tool_state, missing)]
    elif not last_tool and last_human_text and _is_inheriting_followup(last_human_text, session_facts):
        # Seguimiento que reutiliza parámetros de un turno anterior. Con el historial largo Ollama ignora tool_choice="any" e imita la plantilla de petición de datos.
        # La cura es recrear el primer turno: contexto COMPACTO (system prompt con [CONTEXTO DE SESIÓN] + última petición + directiva) y tool_choice="any".
        messages = [
            messages[0],  # SystemMessage (rol + [CONTEXTO DE SESIÓN] + comportamiento)
            HumanMessage(content=last_human_text),
            _build_followup_directive(session_facts),
        ]
        force_tool_call = True

    # Selección de tools: si el último ToolMessage es de consultar_teoria, la excluimos del bind para impedir bucles RAG → RAG.
    if last_tool == "consultar_teoria":
        tools_for_bind = [t for t in AGENT_TOOLS if t.name != "consultar_teoria"]
    else:
        tools_for_bind = AGENT_TOOLS

    # Schema fino: el modelo ve nombre + descripción de cada tool analítica (para SELECCIONARLA) pero no sus parámetros, así no puede rellenar args; consultar_teoria se exime.
    bind_payload = thin_tool_schemas(tools_for_bind, ANALYTICAL_TOOL_NAMES)
    llm = get_llm_with_tools(bind_payload, tool_choice="any" if force_tool_call else None)
    t0 = time.perf_counter()
    raw_response = llm.invoke(messages)
    duration_ms = (time.perf_counter() - t0) * 1000.0
    response = raw_response

    # Red de seguridad: si el modelo respondió en prosa a una petición de acción clara (sin tool previa ni recogida en curso), reintentamos una vez forzando la tool call.
    if (
        not getattr(response, "tool_calls", None)
        and not last_tool
        and not pending_tool_state
        and _should_force_action(last_human_text, bool(csv_path))
    ):
        forced_llm = get_llm_with_tools(bind_payload, tool_choice="any")
        compact = [messages[0], HumanMessage(content=last_human_text), _build_action_directive()]
        t_force = time.perf_counter()
        forced_raw = forced_llm.invoke(compact)
        forced = forced_raw
        emit_llm_call(
            name="razonador.force_action_retry",
            messages=compact,
            response=forced,
            duration_ms=(time.perf_counter() - t_force) * 1000.0,
        )
        if getattr(forced, "tool_calls", None):
            response = forced

    # Red de seguridad RAG: si el usuario hizo una pregunta teórica y el modelo respondió en prosa, sintetizamos la llamada a consultar_teoria de forma determinista para garantizar el grounding.
    # Mismas guardas que el force-action (solo al inicio de ciclo); la query es la propia pregunta del usuario.
    if (
        not getattr(response, "tool_calls", None)
        and not last_tool
        and not pending_tool_state
        and "consultar_teoria" in _KNOWN_TOOL_NAMES
        and _looks_like_theory_question(last_human_text)
    ):
        response = AIMessage(
            content="",
            tool_calls=[{
                "name": "consultar_teoria",
                "args": {"query": last_human_text},
                "id": f"call_{uuid.uuid4().hex[:12]}",
                "type": "tool_call",
            }],
        )

    # Anti-invención: retira args sin respaldo en el historial. Va ANTES del merge: un valor legítimo de un turno previo lo repone el merge; los inventados caen y se piden.
    response = _strip_invented_args(response, state)
    # Si el turno anterior pidió obligatorios, recuperamos los args ya conocidos para que el LLM solo aporte los nuevos.
    response = _merge_with_pending_params(response, state)
    # Canonicaliza los enums código=nombre según lo que el usuario nombró (corrige un código mal mapeado). Va tras el merge para ver también el de un turno anterior.
    response = _resolve_enum_codes(response, state)

    # Evento llm_call (tokens, tokens/s, parser de fallback); no-op si la observabilidad está apagada.
    emit_llm_call(
        name="razonador.llm",
        messages=messages,
        response=response,
        duration_ms=duration_ms,
    )

    updates: dict = {
        "messages": [response],
    }

    # Detectar si hay una tool call y si le faltan parámetros
    tool_calls = getattr(response, "tool_calls", None) or []

    # Citas trazables: al cerrar el ciclo RAG con la respuesta final, se anexa la sección Fuentes a partir del contexto de consultar_teoria.
    if not tool_calls and last_tool == "consultar_teoria":
        _append_fuentes(response, messages)
    elif not tool_calls and last_tool in ANALYTICAL_TOOL_NAMES:
        # Fidelidad numérica (RF-11): verifica que los valores del ToolMessage aparecen en la síntesis y reintenta una vez si faltan.
        response = _verify_and_repair(response, messages)
        updates["messages"] = [response]

    if tool_calls:
        call = tool_calls[0]
        tool_name = call.get("name", "") if isinstance(call, dict) else getattr(call, "name", "")
        args = call.get("args", {}) if isinstance(call, dict) else getattr(call, "args", {})

        # Herencia genérica desde session_facts: rellena los parámetros de familias semánticas ya establecidos que la tool acepte. consultar_teoria se exime.
        session_facts = state.get("session_facts") or {}
        if tool_name != "consultar_teoria":
            enriched_args, inherited = _inherit_from_session(tool_name, args, session_facts)
            if inherited:
                call_id = (
                    call.get("id") if isinstance(call, dict) else getattr(call, "id", None)
                ) or f"call_{uuid.uuid4().hex[:12]}"
                response = AIMessage(
                    content=response.content,
                    tool_calls=[{
                        "name": tool_name,
                        "args": enriched_args,
                        "id": call_id,
                        "type": "tool_call",
                    }],
                    additional_kwargs=getattr(response, "additional_kwargs", {}) or {},
                )
                updates["messages"] = [response]
                args = enriched_args

        # Descarta los placeholders vacíos antes de evaluar faltantes o ejecutar (end_date="" junto a periods haría fallar la API con "no ambos").
        cleaned = _drop_empty_args(args)
        if cleaned != args:
            call_id = (
                call.get("id") if isinstance(call, dict) else getattr(call, "id", None)
            ) or f"call_{uuid.uuid4().hex[:12]}"
            response = AIMessage(
                content=response.content,
                tool_calls=[{
                    "name": tool_name,
                    "args": cleaned,
                    "id": call_id,
                    "type": "tool_call",
                }],
                additional_kwargs=getattr(response, "additional_kwargs", {}) or {},
            )
            updates["messages"] = [response]
            args = cleaned

        missing = get_missing_params(tool_name, args)
        missing_groups = get_missing_alternative_groups(tool_name, args)

        if missing or missing_groups:
            # Faltan obligatorios o un grupo "uno-de": pedirlos al usuario. La confirmación de opcionales irá en el siguiente ciclo.
            updates["pending_tool"] = tool_name
            updates["pending_params"] = args
        else:
            # Obligatorios completos. Antes de ejecutar confirmamos los tunables no fijados, una sola vez por tool_name (optionals_confirmed_for).
            already_confirmed = state.get("optionals_confirmed_for") == tool_name
            missing_tunable = get_missing_tunable_params(tool_name, args)

            if missing_tunable and not already_confirmed:
                updates["pending_tool"] = tool_name
                updates["pending_params"] = args
                updates["optionals_confirmed_for"] = tool_name
            else:
                updates["pending_tool"] = None
                updates["pending_params"] = None
                updates["optionals_confirmed_for"] = None

    return updates

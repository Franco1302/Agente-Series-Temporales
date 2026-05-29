"""Nodo de razonamiento: invoca el LLM con las herramientas enlazadas."""

from __future__ import annotations

import json
import re
import time
import uuid
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

# Schema cacheado de cada tool: nombres de parámetros que su firma acepta.
# Se usa para descartar la herencia de un parámetro si la tool destino no lo
# declara (evita inyectar args inválidos que el ToolNode rechazaría).
_TOOL_ACCEPTED_PARAMS: dict[str, frozenset[str]] = {}
for _t in AGENT_TOOLS:
    _schema = getattr(_t, "args_schema", None)
    if _schema is not None and hasattr(_schema, "model_fields"):
        _TOOL_ACCEPTED_PARAMS[_t.name] = frozenset(_schema.model_fields.keys())
    elif isinstance(_schema, dict):
        _TOOL_ACCEPTED_PARAMS[_t.name] = frozenset((_schema.get("properties") or {}).keys())
    else:
        _TOOL_ACCEPTED_PARAMS[_t.name] = frozenset()
_JSON_TOOLCALL_RE = re.compile(
    r'\{[^{}]*"name"\s*:\s*"(?P<name>[\w\-]+)"[^{}]*"(?:arguments|args|parameters)"\s*:\s*(?P<args>\{.*?\})\s*\}',
    re.DOTALL,
)


def _coerce_text_toolcall(message: AIMessage) -> AIMessage:
    """Promueve una tool call emitida como JSON en `content` a `tool_calls`.

    Algunos modelos cuantizados (p. ej. qwen2.5-coder q4) no respetan el
    protocolo nativo de tool calling de Ollama y emiten el objeto
    {"name": ..., "arguments": {...}} dentro de `content`. Este helper detecta
    ese patrón, sintetiza un tool_call estándar y limpia el content para
    que el resto del grafo (route_after_razonador, ToolNode) lo trate como
    una tool call legítima.
    """
    if getattr(message, "tool_calls", None):
        return message

    content = message.content
    if not isinstance(content, str) or "name" not in content:
        return message

    for match in _JSON_TOOLCALL_RE.finditer(content):
        name = match.group("name")
        if name not in _KNOWN_TOOL_NAMES:
            continue
        try:
            args = json.loads(match.group("args"))
        except json.JSONDecodeError:
            continue
        if not isinstance(args, dict):
            continue

        synthetic_call = {
            "name": name,
            "args": args,
            "id": f"call_{uuid.uuid4().hex[:12]}",
            "type": "tool_call",
        }
        return AIMessage(
            content="",
            tool_calls=[synthetic_call],
            additional_kwargs=getattr(message, "additional_kwargs", {}) or {},
        )

    return message


def _last_tool_message_name(messages: list) -> str | None:
    """Devuelve el nombre del último ToolMessage del historial, o None."""
    for msg in reversed(messages):
        if isinstance(msg, ToolMessage):
            return getattr(msg, "name", None)
        if isinstance(msg, (AIMessage,)):
            # Si vemos un AIMessage de texto plano (sin tool_calls), el ciclo
            # anterior ya cerró. No buscamos más atrás.
            if not getattr(msg, "tool_calls", None):
                return None
    return None


def _build_fuentes_section(messages: list) -> str | None:
    """Extrae el bloque «Fuentes consultadas» del último ToolMessage de consultar_teoria.

    Devuelve la sección ya formateada como «Fuentes:\\n- ...» lista para anexar,
    o None si no se encuentra el bloque.
    """
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
    """Anexa de forma determinista la sección Fuentes a la respuesta final.

    Paso 6 del PlanMejoraRAG: la instrucción del prompt no basta con el modelo
    3B local. Copiar las fuentes del output de ``consultar_teoria`` garantiza la
    cita y evita que el modelo invente referencias. No-op si la respuesta ya
    incluye la sección o si no hay bloque de fuentes que citar.
    """
    content = response.content
    if not isinstance(content, str) or not content.strip():
        return
    if "Fuentes:" in content:  # el modelo ya la incluyó: no duplicar
        return
    fuentes = _build_fuentes_section(messages)
    if fuentes:
        response.content = content.rstrip() + "\n\n" + fuentes


def _last_analytical_tool_message(messages: list) -> ToolMessage | None:
    """Devuelve el último ToolMessage de una herramienta analítica, o None.

    Replica la lógica de corte de `_last_tool_message_name`: si encuentra un
    AIMessage de texto plano antes que un ToolMessage, el ciclo anterior ya
    cerró y no se sigue buscando.
    """
    for msg in reversed(messages):
        if isinstance(msg, ToolMessage):
            if getattr(msg, "name", None) in ANALYTICAL_TOOL_NAMES:
                return msg
            return None
        if isinstance(msg, AIMessage) and not getattr(msg, "tool_calls", None):
            return None
    return None


def _extract_must_cite_facts(tool_msg: ToolMessage) -> list[str]:
    """Extrae del ToolMessage los valores deterministas que deben aparecer citados.

    El contenido ya es JSON (lo reescribe `_parse_tool_payload`). Solo se
    recogen valores string o entero; cualquier fallo de parseo devuelve una
    lista vacía para no romper el flujo.
    """
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
    """Verifica la fidelidad numérica de la síntesis y reintenta una vez si falla.

    Si la respuesta omite algún valor determinista del ToolMessage analítico,
    reinvoca el LLM (sin herramientas, para forzar texto) con un mensaje
    correctivo. Conserva el reintento solo si reduce los valores omitidos; en
    caso contrario mantiene la respuesta original. Cualquier excepción devuelve
    la respuesta original: la verificación nunca debe romper el grafo.
    """
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
            response_raw=retry,
            response_final=retry,
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


# ── Defensa contra invención de args sin evidencia en el mensaje del usuario ─
#
# Aunque `RULE_NO_INVENT` lo prohíbe en el system prompt, el modelo local
# cuantizado tiende a rellenar campos "rellenables por sentido común" (fechas,
# frecuencias, métodos, tipos numéricos, listas de parámetros, enteros,
# nombres de columna) ante peticiones vagas. Cuando la tool call llega con un
# valor inventado, `get_missing_params` no lo detecta como ausente y la tool
# se ejecuta con datos que el usuario nunca eligió.
#
# Esta defensa post-LLM revisa la primera tool call y, para cada (tool, arg)
# del `_FIELD_EVIDENCE_MAP`, aplica el checker del tipo correspondiente. Si la
# evidencia no aparece en el historial de mensajes del usuario, el arg se
# elimina y `get_missing_params` lo recogerá como ausente para que el grafo
# lo pida explícitamente.
#
# Cobertura: todas las herramientas analíticas (sintéticas, drift, augment,
# exogenous, forecast). `file_path` no se vigila aquí: la sección FICHERO
# ACTIVO del prompt suministra la ruta de forma legítima desde el estado.


# ── Patrones de evidencia textual ───────────────────────────────────────────

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

# Cualquier dígito o número español escrito en palabras (desde "dos") cuenta
# como evidencia. "uno/una/un" se excluye a propósito: en español funciona
# casi siempre como artículo indefinido ("una serie", "un análisis"), no como
# numeral, y daría muchos falsos positivos sobre `periods`/`forecast_steps`.
_INTEGER_EVIDENCE_RE = re.compile(
    r"\b\d+\b"
    r"|\b(?:dos|tres|cuatro|cinco|seis|siete|ocho|nueve|diez|once|doce|"
    r"trece|catorce|quince|diecis[eé]is|veinte|treinta|cuarenta|cincuenta|"
    r"sesenta|setenta|ochenta|noventa|cien|mil)\b",
    re.IGNORECASE,
)

# Listas numéricas: corchetes/paréntesis con dígitos, decimales sueltos, pares
# separados por coma, patrones de asignación letra=número ("p = 30", "mu=0",
# "lambda=5"), o palabras explícitas para referirse a parámetros.
_NUMERIC_LIST_EVIDENCE_RE = re.compile(
    r"[\[\(]\s*[-+]?\d"
    r"|\b-?\d+[\.,]\d+\b"
    r"|\b-?\d+\s*,\s*-?\d+\b"
    r"|\b[a-záéíóú]{1,6}\s*=\s*-?\d"
    r"|\b(?:par[aá]metros?|coeficientes?|valores?|param|lista|sigma|desv|"
    r"mu|media|lambda|prob|probabilidad)\b",
    re.IGNORECASE,
)

# Frases que el usuario emplea cuando delega la elección de un valor al sistema
# ("usa los defaults", "cualquiera", "como tú quieras", "lo que sea"). Cuando
# alguna aparece en el historial, la defensa anti-invención se desactiva por
# completo para ese turno: el usuario ha aceptado explícitamente que el LLM
# proponga un valor sensato, y bloquearlo provoca bucles infinitos de la forma
# usuario→"usa defaults"→agente→"dame el valor"→usuario→"cualquiera"→…
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
    """True si el usuario delegó explícitamente la elección de algún valor.

    No intentamos identificar a qué campo se refiere la delegación: si la
    palabra aparece en el historial, asumimos que el usuario quiere que el
    LLM proponga valores sensatos. Es preferible un ejecución con valores
    razonables a un bucle de preguntas.
    """
    return _has_any_keyword(user_text, _DELEGATION_KEYWORDS)


def _build_delegation_directive(
    pending_tool: str | None,
    missing: list[str],
) -> SystemMessage:
    """Directiva que se inyecta al LLM cuando el usuario delega.

    El objetivo NO es construir la tool call por código, sino forzar al LLM a
    invocar la herramienta razonando y proponiendo él mismo los valores. Con
    qwen cuantizado, sin este empujón explícito el modelo tiende a pedir
    aclaraciones en texto plano incluso después de "usa los defaults" — la
    directiva le indica que está autorizado a decidir y que debe emitir el
    JSON ya.
    """
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


# Frases con las que el usuario pide una operación NUEVA reutilizando parámetros
# de un turno anterior ("ahora con los parámetros anteriores…", "los mismos
# parámetros pero…", "como antes pero…"). A diferencia de la delegación, aquí no
# cede la elección de un valor: pide reutilizar lo ya establecido en la sesión.
# El modelo cuantizado, ante este seguimiento cross-tool, tiende a responder en
# prosa imitando la plantilla "Para continuar necesito…" en vez de emitir el
# JSON; la directiva de abajo lo corrige.
_FOLLOWUP_REUSE_KEYWORDS: tuple[str, ...] = (
    "anteriores", "de antes", "los de antes",
    "mismos par[áa]metros", "esos par[áa]metros", "los mismos", "las mismas",
    "como antes", "igual que antes", "reutiliz", "reusa", "vuelve a usar",
    "c[oó]ge los", "usa los", "con esos",
)


def _is_inheriting_followup(user_text: str, session_facts: dict) -> bool:
    """True si el usuario pide algo nuevo reutilizando parámetros ya conocidos.

    Solo se considera seguimiento heredante si (a) hay parámetros heredables en
    sesión (``session_facts['by_param']`` no vacío) y (b) el texto referencia
    explícitamente reutilizar valores previos. Ser estricto en (b) evita que la
    directiva se dispare en preguntas teóricas u otras peticiones no analíticas.
    """
    by_param = (session_facts or {}).get("by_param") or {}
    if not by_param:
        return False
    return _has_any_keyword(user_text, _FOLLOWUP_REUSE_KEYWORDS)


def _build_followup_directive(session_facts: dict) -> SystemMessage:
    """Directiva que fuerza la tool call en un seguimiento que hereda parámetros.

    Mismo patrón que ``_build_delegation_directive``: no construimos la tool call
    por código, solo indicamos al LLM que debe emitir el JSON ahora en lugar de
    responder en texto. La herencia genérica (``_inherit_from_session``) rellenará
    los parámetros de las familias semánticas; si falta un obligatorio nuevo de la
    tool destino, la validación de parámetros lo pedirá de forma normal.
    """
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


# ── Petición de acción analítica que el modelo debe resolver con tool call ───
#
# El modelo local a veces responde en PROSA pidiendo parámetros cuando debería
# emitir la tool call y dejar que `solicitar_parametros_node` los recoja de forma
# estructurada (con sus opciones). Pasa sobre todo con peticiones vagas de
# generación ("genérame una serie sintética"): el modelo trata el tipo como algo
# que debe saber antes de llamar. Cuando detectamos una petición de acción clara
# y el modelo NO emitió tool call, reintentamos UNA vez forzando la llamada con
# contexto compacto + una directiva (mismo patrón que el seguimiento heredante).
# `tool_choice="any"` solo no basta con este modelo: la directiva es lo que de
# verdad lo empuja a emitir el JSON.
_ACTION_INTENT_KEYWORDS: tuple[str, ...] = (
    "sint[eé]tic", "serie aleatoria", "datos aleatorios", "simula",
    "genera", "gener[aá]", "crea", "cr[eé]a",
    "drift", "deriva", "ha cambiado", "estabilidad",
    "aument", "ampliar", "m[aá]s observaciones", "m[aá]s datos",
    "forecast", "predic", "predec", "pron[oó]stic", "sarimax", "futuro",
    "detecta", "ex[oó]gena", "pca", "correlaci",
)

# Marcadores de pregunta teórica o de capacidades: si aparecen, NO forzamos
# (la teoría la enruta consultar_teoria por el flujo normal; las preguntas de
# capacidades y los saludos se responden en texto plano).
_NON_ACTION_MARKERS: tuple[str, ...] = (
    "qu[eé] es", "qu[eé] son", "qu[eé] significa", "explica", "expl[ií]ca",
    "diferencia", "c[oó]mo funciona", "para qu[eé] sirve", "concepto",
    "definici[oó]n", "qu[eé] puedes", "qu[eé] sabes", "ay[uú]dame", "ayuda",
)


def _looks_like_action_request(text: str) -> bool:
    """True si el mensaje es una petición de acción analítica clara.

    Positivo si menciona alguna intención de acción (generar, detectar,
    aumentar, predecir, crear variable…) y NO es una pregunta teórica ni de
    capacidades. Gate del reintento que fuerza la tool call.
    """
    if not text:
        return False
    if _has_any_keyword(text, _NON_ACTION_MARKERS):
        return False
    return _has_any_keyword(text, _ACTION_INTENT_KEYWORDS)


# Marcadores de generación que NO necesita fichero (las 4 generate_synthetic_*).
# Si no hay CSV cargado solo forzamos la tool call para estas: forzar una tool
# que requiere file_path sin CSV haría que el nodo pidiese "file_path" por texto,
# cuando el fichero se sube por el panel lateral (no se teclea la ruta).
_NO_FILE_GENERATION_MARKERS: tuple[str, ...] = (
    "sint[eé]tic", "serie aleatoria", "datos aleatorios", "simula",
)


def _should_force_action(text: str, csv_loaded: bool) -> bool:
    """True si debemos reintentar forzando la tool call para esta petición.

    Requiere intención de acción y, si no hay CSV, que sea una generación
    sintética (que no necesita fichero).
    """
    if not _looks_like_action_request(text):
        return False
    return bool(csv_loaded) or _has_any_keyword(text, _NO_FILE_GENERATION_MARKERS)


def _build_action_directive() -> SystemMessage:
    """Directiva que fuerza la tool call ante una petición de acción en prosa.

    No construye la tool call por código: empuja al LLM a emitir el JSON ya, con
    los parámetros explícitos que tenga (o ``arguments={}``), en vez de preguntar
    en texto. Para 'serie/datos sintéticos' sin concretar, sugiere la opción por
    defecto (generate_synthetic_distribution) para resolver la ambigüedad de tool.
    """
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


# Mapas de palabras-clave para los enums semánticos (el LLM hace traducción
# legítima «normal» → 1, «kolmogorov-smirnov» → KS; queremos detectar que
# el usuario dijo *alguna* palabra relacionada).
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
    "pca", "principal", "correla", "covar", "lineal", "linear",
    "polinom", "polynom", "relaci[oó]n",
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
}


# ── Mapa (tool, arg) → tipo de evidencia, derivado del schema ───────────────
#
# Antes era un literal hand-coded que repetía a mano lo que cada Field de la
# tool MCP ya declara en `json_schema_extra={"evidence": "<tipo>"}`. Ahora se
# deriva: para cada herramienta analítica se leen sus propiedades y se recoge la
# clave `evidence` de las que la declaren. Añadir/cambiar un campo invent-prone
# en `mcp_server/tools/` no requiere tocar este módulo.
#
# `file_path` queda fuera a propósito (no lleva `evidence`): la sección FICHERO
# ACTIVO del prompt lo proporciona desde el estado.
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
    """Elimina de la tool call los args inventados sin evidencia en el texto del usuario.

    Se ejecuta antes de `_merge_with_pending_params`, por lo que solo ve los
    args que el LLM acaba de emitir (no los recogidos en turnos previos). Si
    en pending_params hay un valor legítimo para el campo, se restaurará en
    el siguiente paso del pipeline; los inventados quedan fuera y caen al
    nodo de solicitud de parámetros.

    Recibe `state` (no solo `messages`) porque algunos checkers consultan
    `csv_metadata` para validar nombres de columna contra el CSV cargado.
    """
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

    # Bypass por delegación: si el usuario acaba de delegar ("usa los defaults",
    # "cualquiera", "como tú quieras"…), la defensa se desactiva ese turno y
    # dejamos que el LLM proponga un valor sensato. Mirar SOLO el último mensaje
    # del usuario evita que una delegación de un turno anterior arrastre permisos
    # a turnos posteriores no relacionados.
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


def _merge_with_pending_params(response: AIMessage, state: AgentState) -> AIMessage:
    """Fusiona los args ya recogidos en `pending_params` con la nueva tool call.

    Cuando el turno anterior pidió parámetros obligatorios al usuario, los
    modelos locales cuantizados a menudo re-emiten la tool call con SOLO el
    parámetro que el usuario acaba de aportar y olvidan los args originales
    (file_path, method, etc.). Esto provoca un bucle pregunta → respuesta →
    pregunta porque el razonador vuelve a detectar obligatorios ausentes.

    La fusión conserva todos los valores no vacíos previamente recogidos y deja
    que los args del LLM sobrescriban únicamente cuando aportan un valor real.
    Si la tool call es para una tool distinta a la pendiente (el usuario cambió
    de intención), no se fusiona nada.
    """
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
    """Construye la tool call de forma determinista tras la confirmación del usuario.

    Se invoca cuando el razonador detecta que el turno actual responde a una
    pregunta de confirmación de parámetros opcionales (`optionals_confirmed_for`
    está fijado y coincide con `pending_tool`). En lugar de re-invocar al LLM
    (que con modelos locales cuantizados suele "olvidar" los args originales
    y vuelve a pedirlos), se reconstruye la tool call a partir de
    `pending_params` + cualquier `nombre=valor` que el usuario haya indicado
    explícitamente en su respuesta.

    Si el usuario expresa cancelación ("no quiero", "olvídalo", etc.), se
    aborta la operación y se limpia el estado pendiente.
    """
    pending_tool = state.get("pending_tool") or ""
    base_args = dict(state.get("pending_params") or {})
    user_msg = _last_human_message_content(state.get("messages", []))

    tunables = get_tunable_params(pending_tool, base_args)
    explicit = parse_optional_values(user_msg, tunables) if user_msg else {}

    # Cancelación solo si el usuario expresa negativa Y no aporta valores
    # explícitos (ej. "no quiero ejecutarla" cancela; "no quiero defaults,
    # usa threshold=0.3" sigue y aplica los valores).
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
        "rag_context": None,
        "pending_tool": None,
        "pending_params": None,
        "optionals_confirmed_for": None,
    }


def _trim_to_recent_turns(messages: list, max_turns: int) -> list:
    """Devuelve solo los últimos ``max_turns`` turnos de usuario y sus respuestas.

    Un turno empieza con un HumanMessage y abarca todo lo que viene hasta el
    siguiente HumanMessage. Conservar el contexto reciente reduce que el LLM
    cuantizado imite formatos repetidos de turnos antiguos; la memoria de
    parámetros se preserva igualmente en ``session_facts`` (que NO se trunca).
    """
    if max_turns <= 0 or not messages:
        return list(messages)
    human_indices = [i for i, m in enumerate(messages) if isinstance(m, HumanMessage)]
    if len(human_indices) <= max_turns:
        return list(messages)
    cut = human_indices[-max_turns]
    return list(messages[cut:])


def _build_session_facts_hint(session_facts: dict) -> str:
    """Genera el bloque ``[CONTEXTO DE SESIÓN]`` con los parámetros heredables.

    Versión compacta: una línea por parámetro con su valor. Se inyecta en el
    system prompt para que el LLM sepa qué argumentos ya están establecidos en
    la sesión y no los vuelva a pedir ni los invente. Si ``by_param`` está
    vacío devuelve "" y el prompt se construye sin el bloque.
    """
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
    """Rellena ``args`` con parámetros heredables que ya estén en la sesión.

    Política (silenciosa y determinista):
      * Solo se rellena un parámetro si pertenece a ``INHERITABLE_PARAMS``
        (familias semánticas declaradas en ``src.agent.param_families``).
      * Solo se rellena si la tool destino acepta ese parámetro según su
        ``args_schema``. Evita inyectar args inválidos.
      * NUNCA sobrescribe un valor que el LLM ya emitió. Si el usuario
        redefinió un parámetro en el turno actual, se respeta su intención.

    Devuelve ``(args_enriquecidos, breadcrumbs)`` donde ``breadcrumbs`` es una
    lista de dicts ``{name, value, source_tool}`` con cada herencia aplicada.

    Esta pasada COMPLEMENTA a ``_strip_invented_args``: aquel quita los args
    inventados sin respaldo; este rellena los que el usuario sí estableció en
    turnos anteriores. Orden recomendado: strip → merge_pending → inherit.
    """
    nonempty = lambda v: v not in (None, "", [])  # noqa: E731
    by_param = (session_facts or {}).get("by_param") or {}
    accepted = _TOOL_ACCEPTED_PARAMS.get(tool_name, frozenset())
    groups = TOOL_ALTERNATIVE_GROUPS.get(tool_name, [])

    def _xor_sibling_already_set(param: str) -> bool:
        """True si otro miembro del grupo XOR de ``param`` ya está fijado.

        Sin esto, heredar (p. ej.) ``periods`` desde la sesión cuando el LLM ya
        emitió ``end_date`` deja ambos miembros del grupo ``horizon`` y la API
        rechaza la llamada ("periods o end_date, pero no ambos").
        """
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
    """Invoca el LLM con el estado actual y decide la próxima acción.

    Detecta si la respuesta del LLM incluye una tool call con parámetros
    incompletos y, en ese caso, registra `pending_tool` y `pending_params`
    para que el router desvíe el flujo a solicitar_parametros.

    Defensa anti-bucle: si el último ToolMessage del historial proviene de
    `consultar_teoria`, se retira esa herramienta del bind para que el LLM
    no pueda reinvocarla en la síntesis. El `ToolMessage` ya contiene la
    respuesta del RAG, así que el modelo debe sintetizar en texto.
    """
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

    # Atajo determinista: si veníamos esperando la confirmación de opcionales,
    # reconstruimos la tool call sin pasar por el LLM. Esto evita que el modelo
    # local re-emita la tool sin los args obligatorios (file_path, index_column,
    # method, etc.) y termine pidiéndolos otra vez en bucle.
    pending_tool_state = state.get("pending_tool")
    if pending_tool_state and state.get("optionals_confirmed_for") == pending_tool_state:
        return _replay_pending_tool_call(state)

    csv_path = state.get("csv_path")
    csv_metadata = state.get("csv_metadata")

    # Truncamos el historial a los últimos N turnos del usuario
    # (CHAT_MAX_CONTEXT_TURNS en .env). El contexto real de la sesión
    # (parámetros heredables) vive en session_facts y se inyecta más abajo
    # via [CONTEXTO DE SESIÓN]; el historial solo aporta el detalle de la
    # conversación reciente.
    max_turns = load_ollama_settings().max_context_turns
    messages = _trim_to_recent_turns(state["messages"], max_turns)

    # Construimos [CONTEXTO DE SESIÓN] con los parámetros heredables ya
    # establecidos. `build_system_prompt` lo inyecta arriba (entre rol y
    # comportamiento) para que el LLM lo vea antes de decidir si emite tool
    # call. Si el modelo lo ignora, la pasada de herencia (más abajo) rellena
    # igualmente los args al final.
    session_facts = state.get("session_facts") or {}
    facts_hint = _build_session_facts_hint(session_facts) if session_facts else ""

    # Si el último ToolMessage proviene de una herramienta analítica, el prompt
    # incorpora el bloque RESULTADO / INTERPRETACIÓN / SIGUIENTE PASO para que la
    # síntesis final cumpla RF-11. `build_system_prompt` ignora los nombres que no
    # sean analíticos (p. ej. consultar_teoria), así que esto no afecta al RAG.
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

    # Directiva de delegación: si el último HumanMessage delega ("usa los
    # defaults", "cualquiera", "como tú quieras"…), añadimos una SystemMessage
    # al final del historial que fuerza al LLM a invocar la tool con valores
    # razonables propuestos por él. Sin este empujón el modelo cuantizado
    # responde en texto plano y entra en bucle de preguntas. El LLM sigue
    # razonando: nosotros solo le indicamos que está autorizado a decidir.
    last_human_text = _last_human_message_content(messages)
    force_tool_call = False
    if last_human_text and _user_delegates(last_human_text):
        pending_params = state.get("pending_params") or {}
        missing = (
            get_missing_params(pending_tool_state, pending_params)
            if pending_tool_state else []
        )
        # Inyectar SIEMPRE que haya delegación: si pending_tool_state es None
        # la directiva indica al LLM que identifique él la herramienta; si hay
        # pending_tool con missing, le dice qué params completar; si no falta
        # nada, le dice que invoque ya con lo que hay.
        messages = messages + [_build_delegation_directive(pending_tool_state, missing)]
    elif last_human_text and _is_inheriting_followup(last_human_text, session_facts):
        # Seguimiento que reutiliza parámetros de un turno anterior ("ahora con
        # los parámetros anteriores, una con tendencia…"). El modelo cuantizado
        # tiende a responder en texto imitando la plantilla de petición de datos
        # que vio en turnos previos, sin emitir el JSON (H-B1). Ni la directiva
        # de prompt ni `tool_choice="any"` bastan con el historial completo:
        # Ollama ignora el tool_choice cuando el contexto es largo y contiene
        # esas plantillas. La cura es recrear las condiciones del primer turno
        # (que sí funciona): invocar con un contexto COMPACTO — solo el system
        # prompt (que ya incluye [CONTEXTO DE SESIÓN] con los params heredables),
        # la última petición del usuario y la directiva — más `tool_choice="any"`.
        # El historial pesado no aporta aquí: los params viven en session_facts.
        messages = [
            messages[0],  # SystemMessage (rol + [CONTEXTO DE SESIÓN] + comportamiento)
            HumanMessage(content=last_human_text),
            _build_followup_directive(session_facts),
        ]
        force_tool_call = True

    # Selección de tools: si el último ToolMessage es de consultar_teoria,
    # excluimos esa tool del bind para impedir bucles RAG → RAG → RAG.
    if last_tool == "consultar_teoria":
        tools_for_bind = [t for t in AGENT_TOOLS if t.name != "consultar_teoria"]
    else:
        tools_for_bind = AGENT_TOOLS

    # Schema fino: el modelo ve nombre + descripción de cada herramienta analítica
    # (suficiente para SELECCIONARLA) pero no las descripciones/enums/defaults de
    # sus parámetros ni cuáles son obligatorios. Así no puede "rellenar" args y la
    # recogida se centraliza en solicitar_parametros_node, que lee el schema REAL.
    # consultar_teoria se exime (su `query` la reformula el LLM legítimamente).
    bind_payload = thin_tool_schemas(tools_for_bind, ANALYTICAL_TOOL_NAMES)
    llm = get_llm_with_tools(bind_payload, tool_choice="any" if force_tool_call else None)
    t0 = time.perf_counter()
    raw_response = llm.invoke(messages)
    duration_ms = (time.perf_counter() - t0) * 1000.0
    response = _coerce_text_toolcall(raw_response)

    # Red de seguridad: si el modelo respondió en PROSA a una petición de acción
    # clara (y no estamos sintetizando tras una tool ni en mitad de una recogida
    # de parámetros), reintentamos UNA vez forzando la tool call con contexto
    # compacto + directiva. Así la petición de parámetros se centraliza en el
    # nodo (estructurada, con opciones) en vez de salir como prosa improvisada.
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
        forced = _coerce_text_toolcall(forced_raw)
        emit_llm_call(
            name="razonador.force_action_retry",
            messages=compact,
            response_raw=forced_raw,
            response_final=forced,
            duration_ms=(time.perf_counter() - t_force) * 1000.0,
        )
        if getattr(forced, "tool_calls", None):
            response = forced

    # Anti-invención (RULE_NO_INVENT defensiva): retira args sin respaldo en el
    # historial del usuario (fechas, frecuencias, métodos, listas numéricas,
    # nombres de columna, etc.). Debe ir ANTES del merge: si pending_params
    # tenía un valor legítimo recogido en un turno anterior, el merge posterior
    # lo repondrá; los inventados de novo caen y forzarán a solicitar_parametros.
    response = _strip_invented_args(response, state)
    # Si el turno anterior pidió params obligatorios, recuperamos los args ya
    # conocidos para que el LLM solo necesite aportar los nuevos. Sin esto, los
    # modelos cuantizados pierden el contexto y reabren el bucle de preguntas.
    response = _merge_with_pending_params(response, state)

    # Evento llm_call: tokens, tokens/s y si actuó el parser de fallback.
    # No-op cuando el subsistema de observabilidad está apagado.
    emit_llm_call(
        name="razonador.llm",
        messages=messages,
        response_raw=raw_response,
        response_final=response,
        duration_ms=duration_ms,
    )

    updates: dict = {
        "messages": [response],
        "rag_context": None,
    }

    # Detectar si hay una tool call y si le faltan parámetros
    tool_calls = getattr(response, "tool_calls", None) or []

    # Citas trazables (Paso 6): cuando el razonador cierra el ciclo RAG con la
    # respuesta final (texto, sin tool call), se anexa la sección Fuentes de
    # forma determinista a partir del contexto que devolvió consultar_teoria.
    if not tool_calls and last_tool == "consultar_teoria":
        _append_fuentes(response, messages)
    elif not tool_calls and last_tool in ANALYTICAL_TOOL_NAMES:
        # Fidelidad numérica (RF-11): verifica que los valores deterministas del
        # ToolMessage analítico aparecen en la síntesis; si faltan, reintenta la
        # generación una sola vez. Nunca rompe el flujo si la verificación falla.
        response = _verify_and_repair(response, messages)
        updates["messages"] = [response]

    if tool_calls:
        call = tool_calls[0]
        tool_name = call.get("name", "") if isinstance(call, dict) else getattr(call, "name", "")
        args = call.get("args", {}) if isinstance(call, dict) else getattr(call, "args", {})

        # Herencia genérica desde session_facts: rellena los parámetros de las
        # familias semánticas (temporal_window, data_source, series_identity)
        # que ya estén establecidos en sesión y que la tool destino acepte en su
        # firma. Complementa a _strip_invented_args: aquel ya retiró los args
        # inventados; este restaura los que el usuario sí estableció antes.
        # Excepción: consultar_teoria se exime — su `query` la reformula el LLM
        # libremente y no hay herencia útil entre turnos teóricos.
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

        missing = get_missing_params(tool_name, args)
        missing_groups = get_missing_alternative_groups(tool_name, args)

        if missing or missing_groups:
            # Faltan obligatorios o un grupo "uno-de" (p. ej. periods|end_date
            # en las tools sintéticas): pedirlos al usuario. La confirmación
            # de opcionales se hará en el siguiente ciclo, una vez resueltos.
            updates["pending_tool"] = tool_name
            updates["pending_params"] = args
        else:
            # Obligatorios completos. Antes de ejecutar, confirmamos con el
            # usuario los tunables (umbrales, bins, coeficientes…) que no haya
            # fijado explícitamente. Solo preguntamos una vez por tool_name:
            # `optionals_confirmed_for` se setea aquí y se limpia al proceder
            # a la ejecución en el siguiente turno.
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

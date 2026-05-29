"""Nodo de razonamiento: invoca el LLM con las herramientas enlazadas."""

from __future__ import annotations

import json
import re
import time
import uuid

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

from src.agent.nodes.param_request import (
    get_missing_alternative_groups,
    get_missing_params,
    get_missing_tunable_params,
    get_tunable_params,
    is_cancel_intent,
    parse_optional_values,
)
from src.agent.param_families import INHERITABLE_PARAMS, SEMANTIC_ENUMS
from src.agent.prompts import ANALYTICAL_TOOL_NAMES, build_system_prompt
from src.agent.state import AgentState
from src.agent.tools import AGENT_TOOLS
from src.config.llm_config import get_llm_with_tools, load_ollama_settings
from src.observability import (
    EVENT_PARAMS_INHERITED,
    TraceEvent,
    emit,
    emit_llm_call,
    get_current_span,
    get_thread_id,
    get_trace_id,
    new_span_id,
)

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


# Campos deterministas (string/entero) que deben citarse literalmente en el
# bloque RESULTADO de cada herramienta analítica. Se usan para verificar la
# fidelidad numérica de la síntesis (RF-11). Los floats (p-valores, métricas)
# se omiten a propósito: el modelo puede redondearlos legítimamente y un
# substring-match daría falsos negativos.
_MUST_CITE_FIELDS: dict[str, tuple[str, ...]] = {
    "detect_drift": ("drift_label", "method_used"),
    "forecast_time_series": ("model_used",),
    "augment_time_series": ("new_rows", "strategy_used"),
    "create_exogenous_variable": ("new_column_name", "relation_used"),
    "generate_synthetic_distribution": ("rows_generated",),
    "generate_synthetic_arma": ("rows_generated",),
    "generate_synthetic_periodic": ("rows_generated",),
    "generate_synthetic_trend": ("rows_generated",),
}


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
        fields = _MUST_CITE_FIELDS.get(getattr(tool_msg, "name", "") or "", ())
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


def _build_session_facts_hint(session_facts: dict) -> str:
    """Genera el bloque ``[CONTEXTO DE SESIÓN]`` con los parámetros heredables.

    Versión compacta: una línea por parámetro con su valor. El nombre de la
    tool origen se omite (ruido para el LLM); lo importante es que el valor
    ya está disponible y debe usarse en cualquier tool call que lo pida.
    """
    by_param = session_facts.get("by_param") or {}
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
    # Línea en blanco final para separar del bloque COMPORTAMIENTO siguiente.
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
    lista de dicts ``{name, value, source_tool}`` con cada herencia aplicada,
    para emisión en observabilidad.
    """
    nonempty = lambda v: v not in (None, "", [])  # noqa: E731
    by_param = (session_facts or {}).get("by_param") or {}
    accepted = _TOOL_ACCEPTED_PARAMS.get(tool_name, frozenset())

    enriched = dict(args)
    inherited: list[dict] = []
    for name, fact in by_param.items():
        if name not in INHERITABLE_PARAMS:
            continue
        if name not in accepted:
            continue
        if nonempty(enriched.get(name)):
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


def _retry_if_text_with_active_session(
    response: AIMessage,
    messages: list,
    session_facts: dict,
    last_tool: str | None,
    tools_for_bind: list,
) -> AIMessage:
    """Retry sin keywords cuando el LLM responde texto con sesión activa.

    Política:
      * Si el LLM ya emitió tool_call → no-op.
      * Si NO hay parámetros en sesión → no-op (no hay nada que recordar).
      * Si el ciclo viene de una tool analítica (cierre con
        RESULTADO/INTERPRETACIÓN/SIGUIENTE PASO) o de consultar_teoria
        (síntesis del RAG) → no-op: en esos casos responder texto es lo
        correcto.
      * En el resto, reintenta UNA vez con un SystemMessage corrector que
        recuerda el contexto. Si el retry no emite tool_call, mantenemos la
        respuesta original (nunca rompemos el flujo).

    No examina el contenido del HumanMessage del usuario: depende solo del
    estado del agente (hay sesión + el LLM eligió texto + no estamos cerrando
    un ciclo de tool).
    """
    if getattr(response, "tool_calls", None):
        return response
    by_param = (session_facts or {}).get("by_param") or {}
    if not by_param:
        return response
    if last_tool in ANALYTICAL_TOOL_NAMES or last_tool == "consultar_teoria":
        return response
    by_param_listing = ", ".join(
        f"{k}={v.get('value')!r}" for k, v in by_param.items()
    )
    try:
        correccion = SystemMessage(content=(
            "STOP. Has respondido texto pero NO debías. Tienes parámetros ya "
            f"establecidos en sesión: {by_param_listing}. La petición del usuario "
            "se traduce a una de tus herramientas (probablemente del mismo dominio "
            "que la anterior). Tu único output válido AHORA es una tool call. "
            "Reutiliza los parámetros de sesión y emite la tool call sin redactar "
            "texto. Si faltan datos que el usuario NO ha dado, el sistema los "
            "pedirá automáticamente."
        ))
        llm = get_llm_with_tools(tools_for_bind)
        t0 = time.perf_counter()
        retry = llm.invoke(messages + [correccion])
        duration_ms = (time.perf_counter() - t0) * 1000.0
        retry = _coerce_text_toolcall(retry)
        emit_llm_call(
            name="razonador.session_retry",
            messages=messages + [correccion],
            response_raw=retry,
            response_final=retry,
            duration_ms=duration_ms,
        )
        if getattr(retry, "tool_calls", None):
            return retry
        return response
    except Exception:  # noqa: BLE001 — el retry nunca debe romper el flujo
        return response


def _value_appears_in_text(value, text: str) -> bool:
    """¿Aparece el valor (en alguna representación textual) en ``text``?

    Compara case-insensitive. Para listas/tuplas exige que CADA elemento aparezca
    en el texto. Vacíos (None/""/[]) NO se consideran respaldados.
    """
    if value is None or value == "" or value == []:
        return False
    text_l = text.lower()
    if isinstance(value, bool):
        return str(value).lower() in text_l
    if isinstance(value, str):
        return value.lower() in text_l
    if isinstance(value, (int, float)):
        return str(value) in text_l
    if isinstance(value, (list, tuple)):
        return all(_value_appears_in_text(v, text) for v in value)
    return False


def _sanitize_open_data_args(
    args: dict,
    session_facts: dict,
    last_human_msg: str,
) -> tuple[dict, list[str]]:
    """Descarta args sin respaldo (saneamiento de invención de datos abiertos).

    Política:
      - Los enums semánticos (``SEMANTIC_ENUMS``) pasan siempre: el LLM puede
        traducir "normal" → 1, "diaria" → 'D', etc.
      - Cualquier otro arg solo se mantiene si su valor está respaldado por
        (a) ``session_facts.by_param`` con ese mismo valor, o (b) aparece como
        substring en el último HumanMessage del usuario.
      - Vacíos (None, "", []) se conservan tal cual: ``solicitar_parametros``
        los detecta como faltantes y los pide.

    Devuelve ``(args_saneados, descartados)``.
    """
    by_param = (session_facts or {}).get("by_param") or {}
    sanitized = dict(args)
    descartados: list[str] = []
    for name in list(sanitized.keys()):
        if name in SEMANTIC_ENUMS:
            continue
        value = sanitized[name]
        if value in (None, "", []):
            continue
        fact = by_param.get(name)
        if fact and fact.get("value") == value:
            continue
        if last_human_msg and _value_appears_in_text(value, last_human_msg):
            continue
        del sanitized[name]
        descartados.append(name)
    return sanitized, descartados


def _emit_params_inherited(tool_name: str, inherited: list[dict]) -> None:
    """Emite un evento de observabilidad por cada pasada de herencia no vacía."""
    if not inherited:
        return
    emit(TraceEvent(
        trace_id=get_trace_id(),
        thread_id=get_thread_id(),
        name=f"inherit.{tool_name}",
        event_type=EVENT_PARAMS_INHERITED,
        span_id=new_span_id(),
        parent_span_id=get_current_span(),
        attributes={
            "tool_name": tool_name,
            "inherited": inherited,
        },
    ))


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

    # Truncamos el historial a los últimos N turnos del usuario (CHAT_MAX_CONTEXT_TURNS
    # en .env). El contexto real de la sesión (parámetros heredables) vive en
    # session_facts y se inyecta más abajo via [CONTEXTO DE SESIÓN]; el historial
    # solo aporta el detalle de la conversación reciente.
    max_turns = load_ollama_settings().max_context_turns
    messages = _trim_to_recent_turns(state["messages"], max_turns)

    # Capa 1 (advisory): construimos [CONTEXTO DE SESIÓN] con los parámetros
    # heredables ya establecidos. `build_system_prompt` lo inyecta arriba (entre
    # rol y comportamiento) para que el LLM lo vea antes de decidir si emite
    # tool call. Si el modelo lo ignora, la Capa 2 (post-procesado en
    # `_inherit_from_session`) rellena igualmente los args al final.
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

    # Selección de tools: si el último ToolMessage es de consultar_teoria,
    # excluimos esa tool del bind para impedir bucles RAG → RAG → RAG.
    if last_tool == "consultar_teoria":
        tools_for_bind = [t for t in AGENT_TOOLS if t.name != "consultar_teoria"]
    else:
        tools_for_bind = AGENT_TOOLS

    llm = get_llm_with_tools(tools_for_bind)
    t0 = time.perf_counter()
    raw_response = llm.invoke(messages)
    # Robustez ante fallos del modelo cuantizado: con qwen2.5:3b q4 a veces la
    # respuesta vuelve vacía (sin content y sin tool_calls). Un único reintento
    # suele resolverlo; si insiste, el fallback más abajo evita el turno mudo.
    if not getattr(raw_response, "tool_calls", None) and not (
        isinstance(raw_response.content, str) and raw_response.content.strip()
    ):
        raw_response = llm.invoke(messages)
    duration_ms = (time.perf_counter() - t0) * 1000.0
    response = _coerce_text_toolcall(raw_response)
    # Fallback final: si tras el retry sigue vacío, mensaje amable al usuario
    # para que la UI no muestre "(El agente no produjo una respuesta de texto.)".
    if not getattr(response, "tool_calls", None) and not (
        isinstance(response.content, str) and response.content.strip()
    ):
        response = AIMessage(content=(
            "No he conseguido generar una respuesta para tu petición. "
            "¿Puedes reformularla con un poco más de detalle?"
        ))
    # Si el turno anterior pidió params obligatorios, recuperamos los args ya
    # conocidos para que el LLM solo necesite aportar los nuevos. Sin esto, los
    # modelos cuantizados pierden el contexto y reabren el bucle de preguntas.
    response = _merge_with_pending_params(response, state)

    # Retry sin keywords: si el LLM respondió texto y hay parámetros en sesión
    # (ya hubo una ejecución analítica), le damos una segunda oportunidad
    # recordándole el contexto. No examina el HumanMessage; depende solo del
    # estado del agente.
    response = _retry_if_text_with_active_session(
        response, messages, session_facts, last_tool, tools_for_bind
    )

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

        # Saneamiento de invención (datos abiertos): los modelos cuantizados
        # tienden a rellenar args plausibles del schema (start_date='2024-01-01',
        # periods=365, etc.) aunque RULE_NO_INVENT esté en el prompt. Descartamos
        # cualquier valor que no esté respaldado por la sesión ni por el mensaje
        # del usuario. Los enums semánticos (distribution_type, trend_type, ...)
        # se conservan: el LLM tiene permiso para traducir "normal" → 1.
        # Excepción: `consultar_teoria` se exime del saneamiento porque su
        # parámetro `query` está pensado para que el LLM lo REFORMULE en una
        # consulta óptima para el RAG; descartar esa reformulación rompe la tool.
        user_msg = _last_human_message_content(messages)
        if tool_name == "consultar_teoria":
            descartados: list[str] = []
        else:
            args, descartados = _sanitize_open_data_args(args, session_facts, user_msg)
        if descartados:
            call_id = (
                call.get("id") if isinstance(call, dict) else getattr(call, "id", None)
            ) or f"call_{uuid.uuid4().hex[:12]}"
            response = AIMessage(
                content=response.content,
                tool_calls=[{
                    "name": tool_name,
                    "args": args,
                    "id": call_id,
                    "type": "tool_call",
                }],
                additional_kwargs=getattr(response, "additional_kwargs", {}) or {},
            )
            updates["messages"] = [response]

        # Capa 2 — pasada determinista de herencia genérica.
        # Rellena cualquier parámetro de una familia heredable (temporal_window,
        # data_source, series_identity) que ya esté establecido en la sesión y
        # que la tool destino acepte en su firma. Nunca pisa lo que el LLM ya
        # emitió. Los parámetros de dominio (method, trend_type, distribution_*,
        # strategy, model…) no entran porque no están en INHERITABLE_PARAMS;
        # si faltan, los pedirá solicitar_parametros al usuario.
        enriched_args, inherited = _inherit_from_session(tool_name, args, session_facts)
        if inherited:
            _emit_params_inherited(tool_name, inherited)
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

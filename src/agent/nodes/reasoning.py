"""Nodo de razonamiento: invoca el LLM con las herramientas enlazadas."""

from __future__ import annotations

import json
import re
import time
import uuid

from langchain_core.messages import AIMessage, SystemMessage, ToolMessage

from src.agent.nodes.param_request import TOOL_REQUIRED_PARAMS
from src.agent.prompts import build_system_prompt
from src.agent.state import AgentState
from src.agent.tools import AGENT_TOOLS
from src.config.llm_config import get_llm_with_tools
from src.observability import emit_llm_call

_MAX_ERRORS = 3

_KNOWN_TOOL_NAMES = {t.name for t in AGENT_TOOLS}
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

    csv_path = state.get("csv_path")
    csv_metadata = state.get("csv_metadata")

    system_prompt = build_system_prompt(csv_path=csv_path, csv_metadata=csv_metadata)

    messages = list(state["messages"])

    # Inyectar system prompt si el primer mensaje no es ya un SystemMessage
    if not messages or not isinstance(messages[0], SystemMessage):
        messages = [SystemMessage(content=system_prompt)] + messages

    # Selección de tools: si el último ToolMessage es de consultar_teoria,
    # excluimos esa tool del bind para impedir bucles RAG → RAG → RAG.
    last_tool = _last_tool_message_name(messages)
    if last_tool == "consultar_teoria":
        tools_for_bind = [t for t in AGENT_TOOLS if t.name != "consultar_teoria"]
    else:
        tools_for_bind = AGENT_TOOLS

    llm = get_llm_with_tools(tools_for_bind)
    t0 = time.perf_counter()
    raw_response = llm.invoke(messages)
    duration_ms = (time.perf_counter() - t0) * 1000.0
    response = _coerce_text_toolcall(raw_response)

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

    if tool_calls:
        call = tool_calls[0]
        tool_name = call.get("name", "") if isinstance(call, dict) else getattr(call, "name", "")
        args = call.get("args", {}) if isinstance(call, dict) else getattr(call, "args", {})

        required = TOOL_REQUIRED_PARAMS.get(tool_name, [])
        missing = [p for p in required if p not in args or args[p] is None or args[p] == ""]

        if missing:
            updates["pending_tool"] = tool_name
            updates["pending_params"] = args
        else:
            updates["pending_tool"] = None
            updates["pending_params"] = None

    return updates

"""Nodo de razonamiento: invoca el LLM con las herramientas enlazadas."""

from __future__ import annotations

import json
import re
import uuid

from langchain_core.messages import AIMessage, SystemMessage, ToolMessage

from src.agent.nodes.param_request import TOOL_REQUIRED_PARAMS
from src.agent.prompts import build_system_prompt
from src.agent.state import AgentState
from src.agent.tools import AGENT_TOOLS
from src.config.llm_config import get_llm_with_tools

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
    if _last_tool_message_name(messages) == "consultar_teoria":
        tools_for_bind = [t for t in AGENT_TOOLS if t.name != "consultar_teoria"]
    else:
        tools_for_bind = AGENT_TOOLS

    llm = get_llm_with_tools(tools_for_bind)
    response = llm.invoke(messages)
    response = _coerce_text_toolcall(response)

    updates: dict = {
        "messages": [response],
        "rag_context": None,
    }

    # Detectar si hay una tool call y si le faltan parámetros
    tool_calls = getattr(response, "tool_calls", None) or []
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

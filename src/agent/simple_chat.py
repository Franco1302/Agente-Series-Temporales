"""Puente mínimo de chat entre la UI de Streamlit y el cliente LLM local."""

from __future__ import annotations

from typing import Literal, Sequence, TypedDict

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage

from src.config.llm_config import get_chat_ollama

Role = Literal["user", "assistant"]


class ChatTurn(TypedDict):
    """Mensaje individual intercambiado entre usuario y asistente."""

    role: Role
    content: str


def _map_turn_to_message(turn: ChatTurn) -> BaseMessage | None:
    """Convierte un turno de chat en un objeto de mensaje de LangChain."""
    content = turn["content"].strip()
    if not content:
        return None

    if turn["role"] == "user":
        return HumanMessage(content=content)
    if turn["role"] == "assistant":
        return AIMessage(content=content)

    return None


def generate_chat_response(
    history: Sequence[ChatTurn],
    system_prompt: str | None = None,
) -> str:
    """Envía el historial al LLM y devuelve la respuesta del asistente.

    Esta función mantiene una interfaz reducida a propósito porque será
    el punto de integración para la orquestación con LangGraph en próximas fases.
    """
    if not history:
        raise ValueError("El historial debe contener al menos un mensaje de usuario.")

    messages: list[BaseMessage] = []

    if system_prompt and system_prompt.strip():
        messages.append(SystemMessage(content=system_prompt.strip()))

    for turn in history:
        message = _map_turn_to_message(turn)
        if message is not None:
            messages.append(message)

    if not messages or not isinstance(messages[-1], HumanMessage):
        raise ValueError("El historial debe terminar con un mensaje de usuario antes de inferir.")

    llm = get_chat_ollama()

    try:
        response = llm.invoke(messages)
    except Exception as exc:
        raise RuntimeError(
            "Falló la invocación al LLM. Verifica disponibilidad del modelo y conectividad con Ollama."
        ) from exc

    content = response.content
    if isinstance(content, str):
        return content.strip()

    if isinstance(content, list):
        return "".join(str(part) for part in content).strip()

    return str(content).strip()

"""Interfaz de chat en Streamlit para el asistente conectado a Ollama local."""

from __future__ import annotations

import os
from typing import cast

import streamlit as st

from src.agent.simple_chat import ChatTurn, generate_chat_response
from src.config.llm_config import load_ollama_settings


def _read_positive_int_env(variable_name: str, default_value: int) -> int:
    """Lee un entero positivo desde entorno y aplica un valor por defecto si es invalido."""
    raw_value = os.getenv(variable_name)
    if raw_value is None or not raw_value.strip():
        return default_value

    try:
        parsed_value = int(raw_value)
    except ValueError:
        return default_value

    return parsed_value if parsed_value > 0 else default_value


def _init_page() -> None:
    """Configura metadatos de la página y encabezados principales."""
    st.set_page_config(
        page_title="Asistente IA de Data Drift",
        page_icon=":bar_chart:",
        layout="wide",
    )
    st.title("Asistente IA de Data Drift")
    st.caption(
        "Fase 1: interfaz de chat local conectada a Ollama. "
        "RAG, LangGraph y la API se incorporarán en fases posteriores."
    )


def _get_chat_history() -> list[ChatTurn]:
    """Inicializa y recupera el historial de chat persistido."""
    if "chat_history" not in st.session_state:
        st.session_state["chat_history"] = []

    return cast(list[ChatTurn], st.session_state["chat_history"])


def _get_chat_summary() -> str:
    """Inicializa y recupera el resumen acumulado de contexto antiguo."""
    if "chat_summary" not in st.session_state:
        st.session_state["chat_summary"] = ""

    return cast(str, st.session_state["chat_summary"])


def _get_summary_cursor() -> int:
    """Indica hasta que indice del historial ya fue resumido."""
    if "summary_cursor" not in st.session_state:
        st.session_state["summary_cursor"] = 0

    return cast(int, st.session_state["summary_cursor"])


def _reset_chat_state() -> None:
    """Reinicia historial y resumen de conversacion."""
    st.session_state["chat_history"] = []
    st.session_state["chat_summary"] = ""
    st.session_state["summary_cursor"] = 0


def _render_history(history: list[ChatTurn]) -> None:
    """Renderiza todos los mensajes previos en formato chat."""
    for message in history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])


def _compact_text(text: str, max_chars: int = 140) -> str:
    """Normaliza y recorta texto para evitar que el resumen crezca demasiado rapido."""
    compact_text = " ".join(text.split())
    if len(compact_text) <= max_chars:
        return compact_text
    return f"{compact_text[: max_chars - 3].rstrip()}..."


def _append_turns_to_summary(
    current_summary: str,
    turns_to_archive: list[ChatTurn],
    max_summary_chars: int,
) -> str:
    """Agrega turnos antiguos a un resumen incremental con tamano maximo."""
    if not turns_to_archive:
        return current_summary

    new_lines: list[str] = []
    for turn in turns_to_archive:
        role = "Usuario" if turn["role"] == "user" else "Asistente"
        new_lines.append(f"- {role}: {_compact_text(turn['content'])}")

    new_summary_block = "\n".join(new_lines)
    merged_summary = (
        f"{current_summary}\n{new_summary_block}".strip()
        if current_summary.strip()
        else new_summary_block
    )

    if len(merged_summary) <= max_summary_chars:
        return merged_summary

    return merged_summary[-max_summary_chars:].lstrip()


def _build_inference_history(history: list[ChatTurn], max_context_turns: int) -> list[ChatTurn]:
    """Devuelve solo la ventana reciente del historial para inferencia."""
    if len(history) <= max_context_turns:
        return list(history)
    return list(history[-max_context_turns:])


def _compose_system_prompt(base_prompt: str, summary: str) -> str:
    """Combina prompt base con resumen de contexto antiguo."""
    clean_base_prompt = base_prompt.strip()
    if not summary.strip():
        return clean_base_prompt

    return (
        f"{clean_base_prompt}\n\n"
        "Resumen de contexto anterior (comprimido):\n"
        f"{summary.strip()}\n\n"
        "Si hay conflicto, prioriza los mensajes recientes."
    )


def _render_sidebar(
    max_context_turns: int,
    max_summary_chars: int,
    current_summary_chars: int,
) -> None:
    """Muestra configuración activa de ejecución para diagnóstico rápido."""
    st.sidebar.header("Entorno de ejecución")

    try:
        settings = load_ollama_settings()
        st.sidebar.write(f"Base URL: {settings.base_url}")
        st.sidebar.write(f"Modelo: {settings.model}")
        st.sidebar.write(f"Temperatura: {settings.temperature}")
    except Exception as exc:
        st.sidebar.warning(f"Error de configuración: {exc}")

    st.sidebar.divider()
    st.sidebar.subheader("Memoria conversacional")
    st.sidebar.write(f"Turnos recientes enviados al LLM: {max_context_turns}")
    st.sidebar.write(f"Límite del resumen: {max_summary_chars} caracteres")
    st.sidebar.write(f"Resumen actual: {current_summary_chars} caracteres")

    if st.sidebar.button("Limpiar conversación"):
        _reset_chat_state()
        st.rerun()


def main() -> None:
    """Ejecuta la aplicación de Streamlit."""
    _init_page()

    history = _get_chat_history()
    chat_summary = _get_chat_summary()
    summary_cursor = _get_summary_cursor()

    max_context_turns = _read_positive_int_env("CHAT_MAX_CONTEXT_TURNS", 8)
    max_summary_chars = _read_positive_int_env("CHAT_SUMMARY_MAX_CHARS", 1400)

    _render_sidebar(
        max_context_turns=max_context_turns,
        max_summary_chars=max_summary_chars,
        current_summary_chars=len(chat_summary),
    )

    _render_history(history)
    # Es la información que se muestra en el chat, donde escribe el usuario
    user_prompt = st.chat_input("Pregunta lo que necesites")
    if not user_prompt:
        return

    user_turn: ChatTurn = {"role": "user", "content": user_prompt}
    history.append(user_turn)

    with st.chat_message("user"):
        st.markdown(user_prompt)

    with st.chat_message("assistant"):
        with st.spinner("Consultando LLM local..."):
            try:
                archive_limit = max(0, len(history) - max_context_turns)
                if summary_cursor < archive_limit:
                    turns_to_archive = history[summary_cursor:archive_limit]
                    chat_summary = _append_turns_to_summary(
                        current_summary=chat_summary,
                        turns_to_archive=turns_to_archive,
                        max_summary_chars=max_summary_chars,
                    )
                    st.session_state["chat_summary"] = chat_summary
                    st.session_state["summary_cursor"] = archive_limit

                inference_history = _build_inference_history(history, max_context_turns)

                system_prompt = os.getenv(
                    "CHAT_SYSTEM_PROMPT",
                    "Eres un asistente de IA útil especializado en análisis de data drift.",
                )
                composed_system_prompt = _compose_system_prompt(system_prompt, chat_summary)

                assistant_reply = generate_chat_response(
                    history=inference_history,
                    system_prompt=composed_system_prompt,
                )
                st.markdown(assistant_reply)
            except Exception as exc:
                assistant_reply = (
                    "Error de conexión con el LLM local. Revisa el servicio de Ollama, "
                    "el modelo y la configuración de .env."
                )
                st.error(f"{assistant_reply}\n\nDetalle técnico: {exc}")

    assistant_turn: ChatTurn = {"role": "assistant", "content": assistant_reply}
    history.append(assistant_turn)


if __name__ == "__main__":
    main()

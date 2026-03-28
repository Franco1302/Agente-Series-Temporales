"""Interfaz de chat en Streamlit para el asistente conectado a Ollama local."""

from __future__ import annotations

import os
from typing import cast

import streamlit as st

from src.agent.simple_chat import ChatTurn, generate_chat_response
from src.config.llm_config import load_ollama_settings


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


def _render_history(history: list[ChatTurn]) -> None:
    """Renderiza todos los mensajes previos en formato chat."""
    for message in history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])


def _render_sidebar() -> None:
    """Muestra configuración activa de ejecución para diagnóstico rápido."""
    st.sidebar.header("Entorno de ejecución")

    try:
        settings = load_ollama_settings()
        st.sidebar.write(f"Base URL: {settings.base_url}")
        st.sidebar.write(f"Modelo: {settings.model}")
        st.sidebar.write(f"Temperatura: {settings.temperature}")
    except Exception as exc:
        st.sidebar.warning(f"Error de configuración: {exc}")


def main() -> None:
    """Ejecuta la aplicación de Streamlit."""
    _init_page()
    _render_sidebar()

    history = _get_chat_history()
    _render_history(history)

    user_prompt = st.chat_input("Pregunta sobre señales de drift, variables o datasets...")
    if not user_prompt:
        return

    user_turn: ChatTurn = {"role": "user", "content": user_prompt}
    history.append(user_turn)

    with st.chat_message("user"):
        st.markdown(user_prompt)

    with st.chat_message("assistant"):
        with st.spinner("Consultando LLM local..."):
            try:
                system_prompt = os.getenv(
                    "CHAT_SYSTEM_PROMPT",
                    "Eres un asistente de IA útil especializado en análisis de data drift.",
                )
                assistant_reply = generate_chat_response(
                    history=history,
                    system_prompt=system_prompt,
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

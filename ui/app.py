"""Interfaz de chat en Streamlit para el agente LangGraph con Tool Calling."""

from __future__ import annotations

import json
import uuid
from pathlib import Path
from typing import cast

import pandas as pd
import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from src.agent.graph import build_agent_graph
from src.config.llm_config import load_ollama_settings

# Directorio donde se guardan los ficheros subidos por el usuario.
_UPLOADS_DIR = Path(__file__).resolve().parents[1] / "data" / "temp_uploads"


# ── Inicialización de sesión ────────────────────────────────────────────────

def _get_thread_id() -> str:
    """Devuelve el thread_id de la sesión actual, creándolo si no existe."""
    if "thread_id" not in st.session_state:
        st.session_state["thread_id"] = str(uuid.uuid4())
    return cast(str, st.session_state["thread_id"])


def _get_chat_history() -> list[dict]:
    """Inicializa y recupera el historial de display de la sesión."""
    if "chat_history" not in st.session_state:
        st.session_state["chat_history"] = []
    return cast(list[dict], st.session_state["chat_history"])


def _get_csv_path() -> str | None:
    """Devuelve la ruta al CSV activo de la sesión, o None si no hay ninguno."""
    return cast(str | None, st.session_state.get("csv_path"))


def _get_csv_metadata() -> dict | None:
    """Devuelve las columnas/filas/dtypes del CSV activo si están en sesión."""
    return cast(dict | None, st.session_state.get("csv_metadata"))


def _compute_csv_metadata(csv_path: str) -> dict | None:
    """Lee la cabecera + dtypes del CSV para que el LLM conozca las columnas reales.

    Sin esta info el modelo inventa nombres de columna (`fecha_index`, etc.)
    y la API responde 400. Hacer un read_csv completo es asumible: los CSVs
    de la UI son pequeños y solo se ejecuta una vez al subir el fichero.
    """
    try:
        df = pd.read_csv(csv_path)
    except Exception:  # noqa: BLE001 — si falla, mejor seguir sin metadata
        return None
    return {
        "columns": [str(c) for c in df.columns],
        "rows": len(df),
        "dtypes": {str(c): str(df[c].dtype) for c in df.columns},
    }


def _reset_chat_state() -> None:
    """Reinicia el historial y genera un nuevo thread_id para la sesión."""
    st.session_state["chat_history"] = []
    # Nuevo thread_id = nueva conversación limpia en MemorySaver
    st.session_state["thread_id"] = str(uuid.uuid4())
    # csv_path / csv_metadata se conservan a propósito: el usuario suele
    # querer mantener su fichero activo entre conversaciones.


# ── Gestión del fichero CSV ─────────────────────────────────────────────────

def _save_uploaded_file(uploaded_file) -> str:
    """Guarda el fichero subido en data/temp_uploads/ y devuelve su ruta."""
    _UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
    dest = _UPLOADS_DIR / uploaded_file.name
    dest.write_bytes(uploaded_file.getbuffer())
    return str(dest)


# ── Renderizado ─────────────────────────────────────────────────────────────

def _init_page() -> None:
    """Configura metadatos de la página y encabezados principales."""
    st.set_page_config(
        page_title="Asistente IA de Series Temporales",
        page_icon=":bar_chart:",
        layout="wide",
    )
    st.title("Asistente IA de Series Temporales")
    st.caption(
        "Agente LangGraph con Tool Calling y RAG — "
        "Sube un CSV en el panel lateral y pregunta lo que necesites. "
        "También puedes consultar teoría sobre data drift y series temporales."
    )


def _render_history(history: list[dict]) -> None:
    """Renderiza el historial de mensajes en formato chat."""
    for turn in history:
        with st.chat_message(turn["role"]):
            st.markdown(turn["content"])


def _render_sidebar() -> None:
    """Muestra el panel lateral con el uploader de CSV, info de Ollama y controles."""
    st.sidebar.header("Fichero de datos")

    uploaded_file = st.sidebar.file_uploader(
        "Sube tu CSV para analizarlo",
        type=["csv"],
        help="El fichero se guardará temporalmente y su ruta se pasará al agente.",
    )

    if uploaded_file is not None:
        saved_path = _save_uploaded_file(uploaded_file)
        st.session_state["csv_path"] = saved_path
        st.session_state["csv_metadata"] = _compute_csv_metadata(saved_path)
        file_size_kb = Path(saved_path).stat().st_size / 1024
        meta = _get_csv_metadata()
        cols_preview = ", ".join(meta["columns"]) if meta else "?"
        st.sidebar.success(
            f"**{uploaded_file.name}** cargado  \n"
            f"{file_size_kb:.1f} KB · `{saved_path}`  \n"
            f"Columnas: `{cols_preview}`"
        )
    elif _get_csv_path():
        active_path = _get_csv_path()
        st.sidebar.info(f"Fichero activo: `{Path(active_path).name}`")

    st.sidebar.divider()
    st.sidebar.header("Entorno de ejecución")

    try:
        settings = load_ollama_settings()
        st.sidebar.write(f"**Modelo:** {settings.model}")
        st.sidebar.write(f"**Base URL:** {settings.base_url}")
        st.sidebar.write(f"**Temperatura:** {settings.temperature}")
    except Exception as exc:
        st.sidebar.warning(f"Error de configuración: {exc}")

    st.sidebar.divider()
    st.sidebar.subheader("Conversación")
    thread_id = _get_thread_id()
    st.sidebar.caption(f"Thread ID: `{thread_id[:8]}…`")

    if st.sidebar.button("Limpiar conversación"):
        _reset_chat_state()
        st.rerun()


# ── Streaming del agente ────────────────────────────────────────────────────

def _extract_tool_name_from_ai_message(msg: AIMessage) -> str | None:
    """Extrae el nombre de la primera tool call de un AIMessage, si existe."""
    tool_calls = getattr(msg, "tool_calls", None) or []
    if not tool_calls:
        return None
    first = tool_calls[0]
    if isinstance(first, dict):
        return first.get("name")
    return getattr(first, "name", None)


def _run_agent_streaming(user_prompt: str, csv_path: str | None) -> str:
    """Invoca el grafo con streaming y renderiza el progreso mediante st.status().

    Devuelve el texto final de respuesta del agente para añadirlo al historial.
    """
    graph = build_agent_graph()
    thread_id = _get_thread_id()
    config = {"configurable": {"thread_id": thread_id}}

    input_state = {
        "messages": [HumanMessage(content=user_prompt)],
        "csv_path": csv_path,
        "csv_metadata": _get_csv_metadata(),
        "error_count": 0,
    }

    final_response = ""
    tool_messages: list[ToolMessage] = []

    with st.status("Procesando tu petición…", expanded=True) as status:
        try:
            for event in graph.stream(input_state, config=config):
                node_name = next(iter(event))

                # Ignorar eventos internos de LangGraph
                if node_name.startswith("__"):
                    continue

                node_output: dict = event[node_name]
                messages_out: list = node_output.get("messages", [])

                if node_name == "razonador":
                    for msg in messages_out:
                        if not isinstance(msg, AIMessage):
                            continue
                        tool_name = _extract_tool_name_from_ai_message(msg)
                        if tool_name:
                            status.write(f"🧠 Razonando: invocando **{tool_name}**…")
                        elif msg.content:
                            status.write("🧠 Razonando, generando respuesta…")
                            # El razonador ya produjo la respuesta final.
                            # El grafo enruta directamente a END en este caso.
                            final_response = msg.content

                elif node_name == "ejecutar_herramienta":
                    for msg in messages_out:
                        if isinstance(msg, ToolMessage):
                            status.write(f"Ejecutando herramienta: **{msg.name}**…")
                            tool_messages.append(msg)

                elif node_name == "solicitar_parametros":
                    for msg in messages_out:
                        if isinstance(msg, AIMessage) and msg.content:
                            final_response = msg.content

                elif node_name == "recuperar_contexto":
                    status.write("Consultando base de conocimiento…")

                elif node_name == "gestionar_error":
                    status.write("Gestionando error, reintentando…")

                elif node_name == "generar_respuesta":
                    for msg in messages_out:
                        if isinstance(msg, AIMessage) and msg.content:
                            status.write("Resultado obtenido, generando respuesta…")
                            final_response = msg.content

            status.update(label="✅ Listo", state="complete", expanded=False)

        except (ConnectionError, RuntimeError) as exc:
            status.update(label="❌ Error de conexión", state="error", expanded=True)
            raise exc

    artifacts = _collect_tool_artifacts(tool_messages, csv_path)
    _render_generated_artifacts(artifacts)
    return final_response


# ── Auto-render de artefactos generados por las tools MCP ──────────────────

_ARTIFACT_KEYS = ("output_path", "image_path")


def _collect_tool_artifacts(
    tool_messages: list[ToolMessage], csv_path: str | None
) -> list[Path]:
    """Extrae rutas a PNG/CSV de los ToolMessage emitidos por las tools MCP.

    Las tools devuelven dicts del tipo {"output_path": "...", "image_path": "..."}
    que llegan en `ToolMessage.content` en uno de dos formatos:
      - LangChain nativo: string JSON con el dict serializado.
      - MCP (langchain_mcp_adapters): lista de partes
        `[{"type": "text", "text": "{...}"}]`.
    """
    seen: set[Path] = set()
    ordered: list[Path] = []

    for msg in tool_messages:
        raw = msg.content
        if not raw:
            continue

        payload: dict | None = None
        if isinstance(raw, dict):
            payload = raw
        elif isinstance(raw, str):
            try:
                parsed = json.loads(raw)
                if isinstance(parsed, dict):
                    payload = parsed
            except (json.JSONDecodeError, TypeError):
                payload = None
        elif isinstance(raw, list):
            for part in raw:
                if not isinstance(part, dict):
                    continue
                text = part.get("text")
                if not isinstance(text, str):
                    continue
                try:
                    parsed = json.loads(text)
                except (json.JSONDecodeError, TypeError):
                    continue
                if isinstance(parsed, dict):
                    payload = parsed
                    break

        if payload is None:
            continue

        for key in _ARTIFACT_KEYS:
            value = payload.get(key)
            if not isinstance(value, str) or not value:
                continue
            if csv_path and value == csv_path:
                continue  # no re-render el CSV original subido por el usuario
            path = Path(value)
            if "temp_uploads" not in path.parts:
                continue
            if path in seen:
                continue
            seen.add(path)
            ordered.append(path)

    return ordered


def _render_generated_artifacts(artifacts: list[Path]) -> None:
    """Renderiza inline cada PNG (st.image) y CSV (preview + descarga)."""
    for p in artifacts:
        if not p.exists():
            continue
        suffix = p.suffix.lower()
        if suffix == ".png":
            st.image(str(p), caption=p.name, use_container_width=True)
        elif suffix == ".csv":
            with st.expander(f"Vista previa: {p.name}"):
                try:
                    import pandas as pd
                    df = pd.read_csv(p, nrows=20)
                    st.dataframe(df, use_container_width=True)
                    st.download_button(
                        label=f"Descargar {p.name}",
                        data=p.read_bytes(),
                        file_name=p.name,
                        mime="text/csv",
                    )
                except Exception as exc:  # noqa: BLE001
                    st.warning(f"No se pudo previsualizar {p.name}: {exc}")


# ── Bucle principal ─────────────────────────────────────────────────────────

def main() -> None:
    """Ejecuta la aplicación de Streamlit."""
    _init_page()
    _render_sidebar()

    history = _get_chat_history()
    _render_history(history)

    user_prompt = st.chat_input("Pregunta lo que necesites")
    if not user_prompt:
        return

    history.append({"role": "user", "content": user_prompt})
    with st.chat_message("user"):
        st.markdown(user_prompt)

    with st.chat_message("assistant"):
        try:
            assistant_reply = _run_agent_streaming(
                user_prompt=user_prompt,
                csv_path=_get_csv_path(),
            )
            if assistant_reply:
                st.markdown(assistant_reply)
            else:
                assistant_reply = "(El agente no produjo una respuesta de texto.)"
                st.warning(assistant_reply)

        except Exception as exc:
            assistant_reply = (
                "Error de conexión con el LLM local. Revisa el servicio de Ollama, "
                "el modelo y la configuración de .env."
            )
            st.error(f"{assistant_reply}\n\nDetalle técnico: {exc}")

    history.append({"role": "assistant", "content": assistant_reply})


if __name__ == "__main__":
    main()

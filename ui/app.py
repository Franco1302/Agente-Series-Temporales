"""Interfaz de chat en Streamlit para el agente LangGraph con Tool Calling."""

from __future__ import annotations

import json
import uuid
from pathlib import Path
from typing import cast

import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from src.agent.graph import build_agent_graph
from src.config.llm_config import load_ollama_settings

_UPLOADS_DIR = Path(__file__).resolve().parents[1] / "data" / "temp_uploads"

_NODE_ICONS = {
    "reasoning_node": "🧠",
    "param_validation_node": "🔍",
    "tool_execution_node": "🔧",
}


# ── Inicialización de sesión ────────────────────────────────────────────────

def _get_thread_id() -> str:
    if "thread_id" not in st.session_state:
        st.session_state["thread_id"] = str(uuid.uuid4())
    return cast(str, st.session_state["thread_id"])


def _get_chat_history() -> list[dict]:
    if "chat_history" not in st.session_state:
        st.session_state["chat_history"] = []
    return cast(list[dict], st.session_state["chat_history"])


def _get_uploaded_file_path() -> str | None:
    return cast(str | None, st.session_state.get("uploaded_file_path"))


def _reset_chat_state() -> None:
    st.session_state["chat_history"] = []
    st.session_state["thread_id"] = str(uuid.uuid4())
    st.session_state["debug_trace"] = []
    st.session_state["debug_graph_state"] = {}


def _init_debug_state() -> None:
    if "debug_trace" not in st.session_state:
        st.session_state["debug_trace"] = []
    if "debug_graph_state" not in st.session_state:
        st.session_state["debug_graph_state"] = {}


# ── Gestión del fichero CSV ─────────────────────────────────────────────────

def _save_uploaded_file(uploaded_file) -> str:
    _UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
    dest = _UPLOADS_DIR / uploaded_file.name
    dest.write_bytes(uploaded_file.getbuffer())
    return str(dest)


# ── Renderizado del debug ───────────────────────────────────────────────────

def _render_trace_into(placeholder, trace: list[dict], graph_state: dict) -> None:
    """Sobreescribe el placeholder del sidebar con la traza actual.

    Se llama tras cada evento del grafo para actualizar el debug en tiempo real.
    Usar `placeholder.container()` reemplaza completamente el contenido previo.
    """
    with placeholder.container():
        if not trace:
            st.caption("Esperando eventos del grafo…")
            return

        for i, step in enumerate(trace):
            node = step["node"]
            icon = _NODE_ICONS.get(node, "⬜")

            with st.expander(f"{icon} Paso {i + 1}: `{node}`", expanded=True):
                if node == "reasoning_node":
                    tool_calls = step.get("tool_calls", [])
                    response = step.get("response", "")
                    if tool_calls:
                        st.write("**Tool calls decididas:**")
                        for tc in tool_calls:
                            args_str = json.dumps(tc["args"], indent=2, ensure_ascii=False)
                            st.code(f"{tc['name']}(\n{args_str}\n)", language="json")
                    elif response:
                        st.write("**Respuesta directa:**")
                        preview = response[:400] + ("…" if len(response) > 400 else "")
                        st.text(preview)
                    else:
                        st.caption("Procesando…")

                elif node == "param_validation_node":
                    missing = step.get("missing_params", [])
                    if missing:
                        st.error(f"Parámetros faltantes: `{', '.join(missing)}`")
                    else:
                        st.success("Todos los parámetros validados.")

                elif node == "tool_execution_node":
                    tool_name = step.get("tool_name", "?")
                    result = step.get("result")
                    st.write(f"**Herramienta:** `{tool_name}`")
                    if result is not None:
                        st.json(result)

        if graph_state:
            st.divider()
            iterations = graph_state.get("iteration_count", 0)
            st.metric("Iteraciones ReAct", iterations)
            pending = graph_state.get("pending_params", [])
            if pending:
                st.warning(f"Params pendientes: `{', '.join(pending)}`")
            tool_results = graph_state.get("tool_results", {})
            if tool_results:
                with st.expander("Tool results acumulados"):
                    st.json(tool_results)


def _render_sidebar() -> None:
    st.sidebar.header("Fichero de datos")

    uploaded_file = st.sidebar.file_uploader(
        "Sube tu CSV para analizarlo",
        type=["csv"],
        help="El fichero se guardará temporalmente y su ruta se pasará al agente.",
    )

    if uploaded_file is not None:
        saved_path = _save_uploaded_file(uploaded_file)
        st.session_state["uploaded_file_path"] = saved_path
        file_size_kb = Path(saved_path).stat().st_size / 1024
        st.sidebar.success(
            f"**{uploaded_file.name}** cargado  \n"
            f"{file_size_kb:.1f} KB · `{saved_path}`"
        )
    elif _get_uploaded_file_path():
        active_path = _get_uploaded_file_path()
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

    # ── Panel de debug ──────────────────────────────────────────────────────
    st.sidebar.divider()
    debug_on = st.sidebar.toggle("Debug LangGraph", value=False)
    st.session_state["debug_on"] = debug_on

    if debug_on:
        st.sidebar.subheader("Traza de ejecución")
        # Creamos el placeholder AQUÍ, en el sidebar, en este punto del layout.
        # _run_agent_streaming() lo sobreescribirá en tiempo real con cada evento.
        placeholder = st.sidebar.empty()
        st.session_state["debug_placeholder"] = placeholder

        # Si ya hay una traza de la petición anterior, la mostramos mientras
        # el usuario no ha enviado nada nuevo.
        existing_trace = st.session_state.get("debug_trace", [])
        existing_state = st.session_state.get("debug_graph_state", {})
        _render_trace_into(placeholder, existing_trace, existing_state)


# ── Streaming del agente ────────────────────────────────────────────────────

def _parse_tool_call(tc) -> dict:
    if isinstance(tc, dict):
        return {"name": tc.get("name", "?"), "args": tc.get("args", {})}
    return {"name": getattr(tc, "name", "?"), "args": getattr(tc, "args", {})}


def _parse_tool_result(content: str | dict) -> dict | str:
    if isinstance(content, dict):
        return content
    try:
        return json.loads(content)
    except (json.JSONDecodeError, TypeError):
        return content


def _run_agent_streaming(user_prompt: str, uploaded_file_path: str | None) -> str:
    """Invoca el grafo con stream_mode=["updates","messages"] para ver tokens y nodos.

    - Eventos "messages": tokens individuales del LLM → se muestran en un
      placeholder dentro del bubble de chat con cursor animado ▌.
    - Eventos "updates": nodo completado → se actualiza el sidebar de debug
      y el st.status() con la descripción de lo que hizo el nodo.
    El placeholder de streaming se borra al terminar; el caller renderiza el
    markdown final limpio.
    """
    graph = build_agent_graph()
    thread_id = _get_thread_id()
    config = {"configurable": {"thread_id": thread_id}}

    input_state = {
        "messages": [HumanMessage(content=user_prompt)],
        "uploaded_file_path": uploaded_file_path,
        "iteration_count": 0,
        "pending_params": [],
    }

    final_response = ""
    reasoning_count = 0
    debug_trace: list[dict] = []
    debug_graph_state: dict = {}
    streaming_text = ""

    placeholder = st.session_state.get("debug_placeholder") if st.session_state.get("debug_on") else None
    if placeholder:
        _render_trace_into(placeholder, [], {})

    # Placeholder dentro del bubble de chat para los tokens en streaming.
    # Se actualiza token a token y se borra al terminar.
    token_placeholder = st.empty()

    with st.status("Iniciando agente…", expanded=True) as status:
        try:
            for event in graph.stream(
                input_state,
                config=config,
                stream_mode=["updates", "messages"],
            ):
                mode, data = event

                # ── Tokens del LLM en tiempo real ───────────────────────────
                if mode == "messages":
                    chunk, metadata = data
                    node = metadata.get("langgraph_node", "")
                    content = getattr(chunk, "content", "")
                    # Solo mostramos tokens del reasoning_node (el LLM hablando,
                    # no mensajes internos de herramientas).
                    if node == "reasoning_node" and isinstance(content, str) and content:
                        streaming_text += content
                        token_placeholder.markdown(streaming_text + " ▌")
                    continue

                # ── Eventos por nodo (updates) ───────────────────────────────
                if mode != "updates":
                    continue

                node_name = next(iter(data))

                if node_name.startswith("__"):
                    continue

                node_output: dict = data[node_name]
                messages_out: list = node_output.get("messages", [])

                if "iteration_count" in node_output:
                    debug_graph_state["iteration_count"] = node_output["iteration_count"]

                # ── reasoning_node ──────────────────────────────────────────
                if node_name == "reasoning_node":
                    reasoning_count += 1
                    step: dict = {"node": node_name, "tool_calls": [], "response": ""}

                    for msg in messages_out:
                        if not isinstance(msg, AIMessage):
                            continue
                        raw_tool_calls = getattr(msg, "tool_calls", None) or []
                        if raw_tool_calls:
                            tool_names = ", ".join(tc.get("name", "?") if isinstance(tc, dict) else getattr(tc, "name", "?") for tc in raw_tool_calls)
                            status.update(label=f"🧠 reasoning_node — llamando a `{tool_names}`")
                            step["tool_calls"] = [_parse_tool_call(tc) for tc in raw_tool_calls]
                        elif msg.content:
                            label = "🧠 reasoning_node — analizando petición" if reasoning_count == 1 else "🧠 reasoning_node — interpretando resultado"
                            status.update(label=label)
                            final_response = msg.content
                            step["response"] = msg.content

                    # Borramos el streaming preview; el texto final lo renderiza el caller
                    streaming_text = ""
                    token_placeholder.empty()
                    debug_trace.append(step)

                # ── param_validation_node ───────────────────────────────────
                elif node_name == "param_validation_node":
                    pending: list = node_output.get("pending_params", [])
                    step = {"node": node_name, "missing_params": pending}

                    for msg in messages_out:
                        if isinstance(msg, AIMessage) and msg.content:
                            final_response = msg.content

                    if pending:
                        status.update(label=f"🔍 param_validation_node — faltan: `{', '.join(pending)}`")
                    else:
                        status.update(label="🔍 param_validation_node — parámetros OK")

                    debug_trace.append(step)
                    debug_graph_state["pending_params"] = pending

                # ── tool_execution_node ─────────────────────────────────────
                elif node_name == "tool_execution_node":
                    for msg in messages_out:
                        if not isinstance(msg, ToolMessage):
                            continue
                        status.update(label=f"🔧 tool_execution_node — ejecutando `{msg.name}`")
                        result = _parse_tool_result(msg.content)
                        debug_trace.append({
                            "node": node_name,
                            "tool_name": msg.name,
                            "result": result,
                        })

                    tool_results = node_output.get("tool_results", {})
                    if tool_results:
                        debug_graph_state["tool_results"] = tool_results

                # Sidebar se actualiza tras cada nodo completado
                if placeholder:
                    _render_trace_into(placeholder, debug_trace, debug_graph_state)

            status.update(label="✅ Respuesta generada", state="complete", expanded=False)

        except (ConnectionError, RuntimeError) as exc:
            token_placeholder.empty()
            status.update(label="❌ Error de conexión", state="error", expanded=True)
            raise exc

    st.session_state["debug_trace"] = debug_trace
    st.session_state["debug_graph_state"] = debug_graph_state

    return final_response


# ── Renderizado del chat ────────────────────────────────────────────────────

def _init_page() -> None:
    st.set_page_config(
        page_title="Asistente IA de Series Temporales",
        page_icon=":bar_chart:",
        layout="wide",
    )
    st.title("Asistente IA de Series Temporales")
    st.caption(
        "Agente LangGraph con Tool Calling — "
        "Sube un CSV en el panel lateral y pregunta lo que necesites."
    )


def _render_history(history: list[dict]) -> None:
    for turn in history:
        with st.chat_message(turn["role"]):
            st.markdown(turn["content"])


# ── Bucle principal ─────────────────────────────────────────────────────────

def main() -> None:
    _init_page()
    _init_debug_state()
    _render_sidebar()  # crea el placeholder del debug ANTES del streaming

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
                uploaded_file_path=_get_uploaded_file_path(),
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

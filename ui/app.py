"""Interfaz de chat en Streamlit para el agente LangGraph con Tool Calling."""

from __future__ import annotations

import json
import re
import time
import uuid
from pathlib import Path
from typing import cast

import pandas as pd
import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from src.agent.graph import build_agent_graph
from src.config.llm_config import load_ollama_settings
from src.observability import (
    EVENT_TURN_END,
    EVENT_TURN_START,
    TraceEvent,
    emit,
    is_enabled,
    log_file_path,
    read_recent_thread_lines,
    read_trace_lines,
    start_turn,
)

# Directorio donde se guardan los ficheros subidos por el usuario.
_UPLOADS_DIR = Path(__file__).resolve().parents[1] / "data" / "temp_uploads"

# Directorio que Streamlit sirve como estático (relativo al entrypoint ui/app.py,
# expuesto en la URL `app/static/...`). Requiere enableStaticServing=true en
# .streamlit/config.toml. Ahí se publican los artefactos generados para que la
# respuesta del agente pueda enlazarlos como descargas reales.
_STATIC_DIR = Path(__file__).resolve().parent / "static"


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
            # unsafe_allow_html: el historial conserva las anclas <a download> que
            # _linkify_artifacts inserta en las respuestas (descarga directa).
            st.markdown(turn["content"], unsafe_allow_html=True)


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
        if active_path:
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
    
    if is_enabled():
        st.sidebar.markdown("---")
        st.sidebar.subheader("📊 Historial de Trazas (Sesión)")
        
        recent_logs = read_recent_thread_lines(log_file_path(), thread_id=thread_id, tail_lines=30)
        if recent_logs:
            df_recent = pd.DataFrame(recent_logs)
            st.sidebar.dataframe(
                df_recent[["name", "event_type", "duration_ms"]],
                height=200,
                use_container_width=True
            )
        else:
            st.sidebar.caption("Esperando interacciones para poblar el log...")
            
        if Path(log_file_path()).exists():
            with open(log_file_path(), "rb") as file_data:
                st.sidebar.download_button(
                    label="📥 Descargar agent.jsonl",
                    data=file_data,
                    file_name="tfg_agent_observability.jsonl",
                    mime="application/jsonlines",
                    use_container_width=True
                )

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


def _run_agent_streaming(user_prompt: str, csv_path: str | None) -> tuple[str, str | None]:
    """Invoca el grafo con streaming y renderiza el progreso mediante st.status().

    Devuelve la tupla (texto_respuesta_del_agente, trace_id) para añadirlo al historial
    y para habilitar el reproductor de trazas.
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
    
    trace_id = start_turn(thread_id)
    turn_t0 = time.perf_counter()
    emit(TraceEvent(
        trace_id=trace_id,
        thread_id=thread_id,
        name="interaccion_usuario",
        event_type=EVENT_TURN_START,
        attributes={
            "user_message_len": len(user_prompt),
            "has_csv": csv_path is not None,
        },
    ))

    # Métricas agregadas del turno para el evento turn_end.
    n_nodes = 0
    final_decision = "answer"

    with st.status("Procesando tu petición…", expanded=True) as status:
        try:
            for event in graph.stream(input_state, config=config): # type: ignore
                node_name = next(iter(event))

                # Ignorar eventos internos de LangGraph
                if node_name.startswith("__"):
                    continue

                n_nodes += 1

                node_output: dict = event[node_name]
                messages_out: list = node_output.get("messages", [])

                if node_name == "razonador":
                    for msg in messages_out:
                        if not isinstance(msg, AIMessage):
                            continue
                        tool_name = _extract_tool_name_from_ai_message(msg)
                        if tool_name:
                            status.write(f"🧠 Razonando: invocando **{tool_name}**…")
                        elif isinstance(msg.content, str) and msg.content:
                            status.write("🧠 Razonando, generando respuesta…")
                            # El razonador ya produjo la respuesta final.
                            # El grafo enruta directamente a END en este caso.
                            final_response = str(msg.content)

                elif node_name == "ejecutar_herramienta":
                    for msg in messages_out:
                        if isinstance(msg, ToolMessage):
                            status.write(f"Ejecutando herramienta: **{msg.name}**…")
                            tool_messages.append(msg)

                elif node_name == "solicitar_parametros":
                    final_decision = "ask_params"
                    for msg in messages_out:
                        if isinstance(msg, AIMessage) and isinstance(msg.content, str) and msg.content:
                            final_response = str(msg.content)

                elif node_name == "recuperar_contexto":
                    status.write("Consultando base de conocimiento…")

                elif node_name == "gestionar_error":
                    status.write("Gestionando error, reintentando…")

                elif node_name == "generar_respuesta":
                    for msg in messages_out:
                        if isinstance(msg, AIMessage) and isinstance(msg.content, str) and msg.content:
                            status.write("Resultado obtenido, generando respuesta…")
                            final_response = str(msg.content)

            status.update(label="✅ Listo", state="complete", expanded=False)

        except (ConnectionError, RuntimeError) as exc:
            status.update(label="❌ Error de conexión", state="error", expanded=True)
            total_ms = (time.perf_counter() - turn_t0) * 1000.0
            emit(TraceEvent(
                trace_id=trace_id,
                thread_id=thread_id,
                name="fin_turno",
                event_type=EVENT_TURN_END,
                duration_ms=total_ms,
                attributes={
                    "n_nodes": n_nodes,
                    "total_duration_ms": total_ms,
                    "final_decision": "error",
                    "error": str(exc),
                },
            ))
            raise exc

    artifacts = _collect_tool_artifacts(tool_messages, csv_path)
    if is_enabled():
        _render_artifact_debug(tool_messages, artifacts, csv_path)
    _render_generated_artifacts(artifacts)

    # Convierte las menciones a ficheros en la respuesta en enlaces de descarga
    # reales (app/static/...) antes de devolverla: así se renderiza y se guarda
    # en el historial ya con los enlaces funcionales, no con rutas locales muertas.
    final_response = _linkify_artifacts(final_response, artifacts)

    total_ms = (time.perf_counter() - turn_t0) * 1000.0
    emit(TraceEvent(
        trace_id=trace_id,
        thread_id=thread_id,
        name="fin_turno",
        event_type=EVENT_TURN_END,
        duration_ms=total_ms,
        attributes={
            "n_nodes": n_nodes,
            "total_duration_ms": total_ms,
            "final_decision": final_decision,
        },
    ))
    return final_response, trace_id


# ── Auto-render de artefactos generados por las tools MCP ──────────────────

_ARTIFACT_KEYS = ("output_path", "image_path")


def _collect_tool_artifacts(tool_messages: list[ToolMessage], csv_path: str | None) -> list[Path]:
    """Extrae de forma limpia y directa las rutas de archivos generadas por las herramientas."""
    ordered: list[Path] = []
    for msg in tool_messages:
        if not msg.content:
            continue
        try:
            # Intentar cargar directo, si es string JSON o ya es diccionario
            content = msg.content
            payload = json.loads(str(content)) if isinstance(content, str) else content
            
            if isinstance(payload, list) and len(payload) > 0 and isinstance(payload[0], dict):
                payload = json.loads(str(payload[0].get("text", "{}")))

            if isinstance(payload, dict):
                for key in ("output_path", "image_path"):
                    val = payload.get(key)
                    if val and isinstance(val, str) and val != csv_path:
                        p = Path(val)
                        if p.exists() and p not in ordered:
                            ordered.append(p)
        except Exception:
            continue
    return ordered


def _render_artifact_debug(
    tool_messages: list[ToolMessage],
    artifacts: list[Path],
    csv_path: str | None,
) -> None:
    """Muestra en pantalla el payload parseado de cada ToolMessage y las rutas detectadas.

    Panel de depuración: confirma qué llega realmente desde las tools MCP
    (output_path / image_path) y si los ficheros existen en disco. Solo se
    renderiza con el modo de observabilidad activo (`is_enabled()`).
    """
    with st.expander("🐞 DEBUG artefactos"):
        if not tool_messages:
            st.caption("No se ejecutó ninguna herramienta en este turno.")
        for i, msg in enumerate(tool_messages):
            st.markdown(f"**ToolMessage #{i}** — `{msg.name}`")
            try:
                content = msg.content
                payload = json.loads(str(content)) if isinstance(content, str) else content
                if isinstance(payload, list) and len(payload) > 0 and isinstance(payload[0], dict):
                    payload = json.loads(str(payload[0].get("text", "{}")))
            except Exception as exc:  # noqa: BLE001
                st.write(f"⚠️ No se pudo parsear el contenido: {exc}")
                continue
            st.json(payload if isinstance(payload, dict) else {"_raw": str(payload)})
            if isinstance(payload, dict):
                for key in ("output_path", "image_path"):
                    val = payload.get(key)
                    if val:
                        existe = Path(str(val)).exists()
                        st.write(f"`{key}` → `{val}` · exists={existe}")
        st.markdown(f"**Artefactos detectados:** {[str(p) for p in artifacts]}")
        st.caption(f"csv_path activo: `{csv_path}`")


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


def _publish_to_static(path: Path) -> str:
    """Copia un artefacto al directorio estático servido y devuelve su URL.

    Streamlit sirve `_STATIC_DIR` bajo la ruta `app/static/`. Copiamos solo si
    el destino no existe o está desactualizado, y devolvemos la URL relativa
    `app/static/<nombre>` lista para usar en un enlace Markdown.
    """
    _STATIC_DIR.mkdir(parents=True, exist_ok=True)
    dest = _STATIC_DIR / path.name
    if not dest.exists() or dest.stat().st_mtime < path.stat().st_mtime:
        dest.write_bytes(path.read_bytes())
    return f"app/static/{path.name}"


_IMAGE_SUFFIXES = (".png", ".jpg", ".jpeg", ".gif", ".svg", ".webp")


def _linkify_artifacts(text: str, artifacts: list[Path]) -> str:
    """Limpia en la respuesta del agente las menciones a ficheros generados.

    Por cada artefacto reescribe su aparición —imagen/enlace Markdown, ruta
    absoluta (con o sin prefijo ``file:``) o nombre en texto plano— evitando que
    quede colgando el prefijo de directorio interno (``/home/.../temp_uploads/``):

    * Imágenes (PNG…): se ELIMINAN del texto. Ya se muestran inline con
      ``st.image`` en ``_render_generated_artifacts``; además el agente las emitía
      como ``![](file:/ruta/abs)``, que el navegador no carga (icono roto).
    * Resto (CSV…): se sustituye por un ancla de descarga directa
      ``<a href="app/static/<nombre>" download="<nombre>">`` con SOLO el nombre.

    Requiere renderizar con ``unsafe_allow_html=True``. Usa un centinela
    intermedio por índice (no contiene el nombre) para no re-tocar el resultado.
    """
    if not text or not artifacts:
        return text
    for i, p in enumerate(artifacts):
        name = p.name
        abs_path = str(p)
        sentinel = f"\x00{i}\x00"
        # 1) Imagen Markdown que referencia el artefacto (por nombre o ruta).
        text = re.sub(r"!\[[^\]]*\]\([^)]*" + re.escape(name) + r"[^)]*\)", sentinel, text)
        # 2) Enlace Markdown que lo referencia.
        text = re.sub(r"\[[^\]]*\]\([^)]*" + re.escape(name) + r"[^)]*\)", sentinel, text)
        # 3) Ruta absoluta (con o sin prefijo file:) y, por último, el nombre suelto.
        text = text.replace("file:" + abs_path, sentinel)
        text = text.replace(abs_path, sentinel)
        text = text.replace(name, sentinel)
        # 4) Sustitución final: las imágenes se quitan (van por st.image); el resto
        #    se convierte en ancla de descarga limpia (sin la ruta interna).
        if p.suffix.lower() in _IMAGE_SUFFIXES:
            replacement = ""
        else:
            url = _publish_to_static(p)
            replacement = f'<a href="{url}" download="{name}">{name}</a>'
        text = text.replace(sentinel, replacement)
    return text


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
            assistant_reply, trace_id = _run_agent_streaming(
                user_prompt=user_prompt,
                csv_path=_get_csv_path(),
            )
            if assistant_reply:
                # unsafe_allow_html: render de las anclas <a download> de descarga
                # directa que inserta _linkify_artifacts (ver más abajo).
                st.markdown(assistant_reply, unsafe_allow_html=True)
            else:
                assistant_reply = "(El agente no produjo una respuesta de texto.)"
                st.warning(assistant_reply)
                
            if is_enabled():
                with st.expander("🛠️ Ver traza analítica del turno (OpenTelemetry compatible)"):
                    lineas_turno = read_trace_lines(log_file_path(), trace_id=trace_id)
                    if lineas_turno:
                        df_turno = pd.DataFrame(lineas_turno)
                        df_render = df_turno[["timestamp", "name", "event_type", "duration_ms", "attributes"]].copy()
                        st.dataframe(df_render, use_container_width=True)
                    else:
                        st.caption("No se localizaron trazas en disco para este identificador.")

        except Exception as exc:
            assistant_reply = (
                "Error de conexión con el LLM local. Revisa el servicio de Ollama, "
                "el modelo y la configuración de .env."
            )
            st.error(f"{assistant_reply}\n\nDetalle técnico: {exc}")

    history.append({"role": "assistant", "content": assistant_reply})


if __name__ == "__main__":
    main()

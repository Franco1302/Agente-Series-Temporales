# Flujo detallado de la aplicación

## Estado actual de integración (2026-04-21)

| Componente | Estado |
|---|---|
| Chat directo con Ollama (`simple_chat.py`) | Operativo — marcado como obsoleto, mantenido como referencia |
| Agente LangGraph con Tool Calling | **Implementado y activo** |
| RAG con base vectorial Chroma | Implementado, no integrado aún en el grafo del agente |
| Servidor MCP real | Pendiente (iteración futura) |
| API real de series temporales | Pendiente (iteración futura) |

---

## Componentes involucrados

### Capa de interfaz
- `ui/app.py` — Streamlit: renderizado, file uploader, streaming de pasos del agente.

### Capa de agente
- `src/agent/graph.py` — construcción y compilación del grafo LangGraph con MemorySaver.
- `src/agent/state.py` — `AgentState` TypedDict: estado compartido entre nodos.
- `src/agent/nodes/reasoning.py` — nodo de razonamiento: invoca el LLM con tools enlazadas.
- `src/agent/nodes/tool_execution.py` — nodo de ejecución: lanza `ToolNode` de LangGraph.
- `src/agent/nodes/param_validation.py` — nodo de validación: detecta parámetros faltantes.
- `src/agent/nodes/routing.py` — funciones de enrutamiento condicional entre nodos.
- `src/agent/prompts/system_prompts.py` — `build_system_prompt()`: prompt parametrizado por contexto.
- `src/agent/tools/` — herramientas mock de las tres operaciones de la API futura.

### Capa de configuración
- `src/config/llm_config.py` — cliente ChatOllama cacheado + `get_llm_with_tools()`.

### Capa RAG (implementada, pendiente de integración en el grafo)
- `src/rag_engine/ingest.py`, `src/rag_engine/retriever.py`, `src/tools/rag_tool.py`.

---

## Flujo de arranque de la app

1. Streamlit ejecuta `ui/app.py`.
2. `build_agent_graph()` construye y compila el grafo (una sola vez, cacheado con `lru_cache`).
3. `load_ollama_settings()` lee `.env` y valida la configuración de Ollama.
4. Se construye la página, el sidebar y se inicializa el `thread_id` de sesión en `session_state`.

---

## Flujo por mensaje del usuario

```
Usuario escribe → st.chat_input
        │
        ▼
  Se añade al chat_history de display
        │
        ▼
  _run_agent_streaming()
        │
        ├── graph.stream(input_state, config={thread_id})
        │       │
        │       ▼
        │   reasoning_node ──────────────────────────────────────────────┐
        │       │                                                         │
        │       │ ¿AIMessage con tool_calls?                              │
        │       ├── No → END (respuesta directa)                         │
        │       └── Sí → param_validation_node                           │
        │                   │                                             │
        │                   │ ¿Faltan parámetros?                         │
        │                   ├── Sí → END (mensaje de solicitud)          │
        │                   └── No → tool_execution_node                 │
        │                               │                                 │
        │                               └── ToolMessages → reasoning_node ┘
        │                                                 (ciclo ReAct, máx. 5 iter.)
        │
        ▼
  st.status() muestra pasos en tiempo real
        │
        ▼
  Respuesta final → st.markdown() + chat_history
```

### Detalle de cada paso

1. El usuario escribe en `st.chat_input`.
2. Se añade el turno `user` a `chat_history` (lista de display en `session_state`).
3. Se construye `input_state` con el `HumanMessage`, la ruta al CSV activo e `iteration_count=0`.
4. Se invoca `graph.stream(input_state, config={"configurable": {"thread_id": ...}})`.
5. LangGraph fusiona el input con el checkpoint existente del hilo (`MemorySaver`).
6. `reasoning_node` inyecta el `SystemMessage` (con o sin info de fichero) y llama al LLM con tools enlazadas.
7. Si el LLM emite `tool_calls`:
   - `param_validation_node` comprueba cada parámetro contra `TOOL_REQUIRED_PARAMS`.
   - Si alguno falta: genera `AIMessage` de solicitud → `END`.
   - Si todo completo: `tool_execution_node` ejecuta la herramienta mock y devuelve `ToolMessage`.
   - El ciclo vuelve a `reasoning_node` para interpretar el resultado.
8. Si el LLM responde texto directamente: `END`.
9. Los eventos del stream actualizan `st.status()` en tiempo real con iconos de progreso.
10. La respuesta final se muestra con `st.markdown()` y se guarda en `chat_history`.

---

## Flujo del file uploader

1. El usuario sube un CSV en `st.file_uploader` del sidebar.
2. `_save_uploaded_file()` escribe el fichero en `data/temp_uploads/{nombre_original}`.
3. La ruta absoluta se guarda en `st.session_state["uploaded_file_path"]`.
4. `_run_agent_streaming()` pasa la ruta en el campo `uploaded_file_path` del `AgentState`.
5. `reasoning_node` detecta el fichero y construye el `SystemMessage` con el bloque de fichero activo, incluyendo nombre, ruta y tamaño.
6. El LLM puede usar la ruta directamente como argumento `file_path` de las herramientas.

---

## Flujo RAG de ingesta (sin cambios)

1. `ingest.py` extrae el PDF fuente a Markdown con `pymupdf4llm`.
2. Segmenta por encabezados Markdown y luego por tamaño (`chunk_size=1000`, `overlap=200`).
3. Genera embeddings con `nomic-embed-text` vía Ollama.
4. Persiste en Chroma bajo `data/vector_db`.

## Flujo RAG de consulta (implementado, pendiente de integración en el grafo)

1. `consultar_teoria_drift(query)` recupera los `top_k` documentos más relevantes.
2. Re-rankea por overlap léxico local.
3. Construye contexto estructurado con metadatos de sección.
4. Sintetiza respuesta con instrucciones de grounding estricto.
5. Devuelve respuesta + bloque `Fuentes consultadas`.

---

## Manejo de errores

| Origen | Comportamiento |
|---|---|
| Ollama no disponible | `_check_ollama_connection()` lanza `ConnectionError`; la UI muestra mensaje amigable con detalle técnico |
| Iteraciones excedidas (`>5`) | `reasoning_node` genera `AIMessage` de parada y el grafo termina en `END` |
| Parámetros faltantes | `param_validation_node` genera mensaje de solicitud al usuario; no se lanza excepción |
| Base vectorial ausente | `retriever.py` lanza `FileNotFoundError` explícito |
| Excepción general en stream | Capturada en `_run_agent_streaming()`, `st.status` cambia a estado `error` |

---

## Notas sobre caché

- `get_chat_ollama()` — `@lru_cache(maxsize=1)`: un solo cliente ChatOllama por proceso.
- `get_llm_with_tools()` — sin caché: `bind_tools` puede variar; reutiliza el cliente base cacheado.
- `build_agent_graph()` — `@lru_cache(maxsize=1)`: el grafo se compila una sola vez por proceso Streamlit, evitando reconstrucciones en cada rerun.

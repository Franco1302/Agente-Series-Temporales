# Flujo detallado de ejecucion (chat + RAG)

## Estado actual de integracion

- La interfaz en Streamlit funciona en modo chat directo con Ollama.
- El backend RAG esta implementado y operativo por scripts/herramientas.
- La UI todavia no invoca de forma directa la herramienta RAG.

## Componentes involucrados

- ui/app.py: interfaz de chat y ciclo de interaccion en Streamlit.
- src/agent/simple_chat.py: puente entre historial de chat y mensajes del LLM.
- src/config/llm_config.py: carga de .env, validacion de parametros y cliente ChatOllama.
- src/rag_engine/ingest.py: pipeline de ingesta PDF -> Markdown -> chunks -> embeddings -> Chroma.
- src/rag_engine/retriever.py: construccion de retriever local en modo lectura sobre Chroma.
- src/tools/rag_tool.py: consulta RAG con retrieval, re-ranking, sintesis y fuentes.
- src/rag_engine/rag_quality_check.py: pruebas basicas de calidad para retrieval + sintesis.

## Flujo de arranque de la app (chat)

1. Se ejecuta Streamlit con el target ui/app.py.
2. Se importan generate_chat_response y load_ollama_settings.
3. Al importar llm_config.py se intenta cargar .env desde la raiz del proyecto.
4. Se construye la pagina y el sidebar con parametros activos de Ollama y memoria conversacional.

## Flujo por mensaje en la UI (chat directo)

1. El usuario escribe en st.chat_input(...).
2. Se agrega el turno user al historial en session_state.
3. La UI calcula ventana deslizante de historial reciente (CHAT_MAX_CONTEXT_TURNS).
4. Los turnos antiguos salen de la ventana y se compactan en chat_summary.
5. Se compone el prompt de sistema base + resumen comprimido.
6. La UI llama a generate_chat_response(history_reciente, prompt_compuesto).
7. simple_chat transforma turnos a mensajes LangChain (HumanMessage/AIMessage/SystemMessage).
8. Se valida que el ultimo mensaje sea del usuario.
9. Se obtiene cliente con get_chat_ollama().
10. get_chat_ollama() valida entorno, conecta con Ollama (/api/tags) y crea ChatOllama.
11. Se ejecuta llm.invoke(messages).
12. La respuesta se normaliza a texto y se guarda como turno assistant.

## Flujo RAG de ingesta

1. ingest.py carga OLLAMA_BASE_URL desde .env.
2. Extrae PDF fuente a Markdown con pymupdf4llm.
3. Aplica split semantico por encabezados (#, ##, ###).
4. Aplica split recursivo por tamano (chunk_size=1000, chunk_overlap=200).
5. Enriquece metadata de cada chunk (source, chunk_id, header_1..3).
6. Genera embeddings con OllamaEmbeddings (nomic-embed-text).
7. Persiste documentos vectorizados en data/vector_db mediante Chroma.

## Flujo RAG de consulta (tool consultar_teoria_drift)

1. Valida query de entrada.
2. Lee parametros de recuperacion desde entorno (RAG_TOP_K, RAG_KEEP_TOP).
3. Construye retriever Chroma y recupera top_k documentos.
4. Aplica re-ranking lexico local por overlap de tokens.
5. Conserva keep_top documentos y arma contexto estructurado con metadatos.
6. Recorta el contexto si supera el limite interno de caracteres.
7. Invoca LLM con instrucciones de respuesta anclada a evidencia.
8. Devuelve respuesta final + bloque Fuentes consultadas.

## Flujo de validacion de calidad RAG

1. rag_quality_check.py define casos de prueba con terminos esperados.
2. Cada caso llama a consultar_teoria_drift.invoke({"query": ...}).
3. Se valida presencia de bloque Fuentes consultadas.
4. Se cuentan coincidencias de terminos esperados.
5. Se reporta PASS/WARN/FAIL y resumen final de fallos.

## Manejo de errores

- Si falla carga de .env o conexion a Ollama, se propaga error controlado.
- Si no existe base vectorial, retriever.py devuelve FileNotFoundError explicito.
- Si falla retrieval o sintesis en rag_tool.py, se devuelve mensaje de error en texto plano.
- La UI captura excepciones del chat y muestra mensaje amigable.

## Nota sobre cache

get_chat_ollama() usa lru_cache(maxsize=1). Esto evita recrear el cliente en cada turno y mejora estabilidad/performance del chat y de la sintesis RAG.

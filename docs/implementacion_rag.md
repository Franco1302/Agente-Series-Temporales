# Implementacion RAG local


## Estado actual (2026-04-21)

- Implementado:
  - Ingesta de conocimiento desde PDF a base vectorial local.
  - Retriever sobre Chroma en modo lectura.
  - Tool de consulta con retrieval + re-ranking + síntesis + fuentes.
  - Script de chequeo básico de calidad.
- Pendiente:
  - Integrar `consultar_teoria_drift` como herramienta del agente LangGraph.
  - Poblar `AgentState.rag_context` con fragmentos relevantes antes de `reasoning_node`.

## Modulos y responsabilidades

- src/rag_engine/ingest.py
  - Ejecuta la ingesta completa: PDF -> Markdown -> chunks -> embeddings -> Chroma.
- src/rag_engine/retriever.py
  - Construye el retriever conectado a data/vector_db.
- src/tools/rag_tool.py
  - Implementa consultar_teoria_drift(query) y entrega respuesta final con fuentes.
- src/rag_engine/rag_quality_check.py
  - Ejecuta casos de prueba para evaluar salida del RAG.

## Variables de entorno relevantes

- OLLAMA_BASE_URL
  - Endpoint de Ollama (ej. <http://localhost:11434>).
- OLLAMA_MODEL
  - Modelo de chat para sintetizar respuesta final.
- OLLAMA_TEMPERATURE
  - Temperatura de generacion del modelo de chat.
- OLLAMA_REQUEST_TIMEOUT
  - Timeout para validacion de conectividad.
- RAG_TOP_K
  - Numero de documentos recuperados por similitud.
- RAG_KEEP_TOP
  - Numero de documentos conservados tras re-ranking.

Nota: RAG_SUMMARY_MAX_SENTENCES y RAG_SUMMARY_MAX_CHARS estan en .env.example pero no participan en el flujo actual.

## Flujo tecnico de ingesta

1. ingest.py valida .env y carga OLLAMA_BASE_URL.
2. Extrae PDF de data/knowledge_base a Markdown con pymupdf4llm.
3. Segmenta por encabezados Markdown (#, ##, ###).
4. Segmenta por tamano (chunk_size=1000, chunk_overlap=200).
5. Agrega metadata de trazabilidad (source, chunk_id, header_1..3).
6. Genera embeddings con nomic-embed-text via Ollama.
7. Persiste en Chroma bajo data/vector_db.

## Flujo tecnico de consulta

1. consultar_teoria_drift valida query.
2. Recupera top_k documentos con retriever Chroma.
3. Reordena por overlap lexico para priorizar evidencia relevante.
4. Construye contexto estructurado por fragmentos con metadata.
5. Limita longitud del contexto para controlar costos y latencia.
6. Sintetiza respuesta con instrucciones de grounding estricto.
7. Devuelve salida final con bloque Fuentes consultadas.

## Dependencias principales

- langchain-ollama
- langchain-chroma
- langchain-text-splitters
- pymupdf4llm
- python-dotenv

## Ejecucion paso a paso

### 1) Preparar entorno

```bash
cd /home/franco/Documentos/TFG
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

### 2) Verificar Ollama y modelos

```bash
ollama serve
ollama pull llama3.1:8b
ollama pull nomic-embed-text
```

### 3) Ejecutar ingesta

```bash
cd /home/franco/Documentos/TFG
source .venv/bin/activate
PYTHONPATH=. python src/rag_engine/ingest.py
```

Resultado esperado:

- Mensajes de extraccion y chunking.
- Mensaje final de guardado en data/vector_db.

### 4) Ejecutar chequeo de calidad

```bash
cd /home/franco/Documentos/TFG
source .venv/bin/activate
PYTHONPATH=. python src/rag_engine/rag_quality_check.py
```

Resultado esperado:

- Casos con PASS/WARN/FAIL.
- Resumen final de total de casos y fallos.

## Criterios de validacion minima

- Existe data/vector_db con contenido persistido.
- La tool devuelve una respuesta no vacia.
- La salida incluye seccion Fuentes consultadas.
- El chequeo de calidad reporta 0 fallos o solo warns aceptables.

## Problemas frecuentes

- Error de conexion con Ollama:
  - Revisar OLLAMA_BASE_URL y que ollama serve este activo.
- Base vectorial ausente o vacia:
  - Ejecutar primero src/rag_engine/ingest.py.
- Import path al ejecutar scripts:
  - Ejecutar desde raiz del proyecto con PYTHONPATH=.

## Integración pendiente con el agente LangGraph

`ui/app.py` ya usa el agente LangGraph. El siguiente paso es conectar el RAG al grafo:

1. Añadir `consultar_teoria_drift` a `AGENT_TOOLS` en `src/agent/tools/__init__.py`.
2. Añadir sus parámetros a `TOOL_REQUIRED_PARAMS` en `src/agent/nodes/param_validation.py`.
3. Decidir si el contexto RAG se inyecta proactivamente (pre-retrieval en un nodo anterior a `reasoning_node`) o reactivamente (el LLM invoca la tool cuando la necesita).
4. Definir formato de respuesta en chat para mostrar fuentes de forma legible.
5. Probar latencia, calidad y manejo de errores en la experiencia conversacional.

# Orquestador IA para Data Drift

Estado actual (abril 2026):

- Chat local en Streamlit conectado a Ollama.
- Implementacion RAG local disponible en backend (ingesta, retriever y tool).
- Integracion directa entre la UI y el flujo RAG aun pendiente.

## Alcance implementado

### Chat UI

- Interfaz en Streamlit con historial conversacional.
- Ventana de contexto reciente + resumen incremental en sesion.
- Invocacion al LLM local por medio de un puente desacoplado.

### RAG backend

- Ingesta de PDF tecnico a base vectorial Chroma local.
- Segmentacion semantica por encabezados Markdown y chunking con overlap.
- Recuperacion por similitud con embeddings en Ollama (nomic-embed-text).
- Re-ranking lexico local para priorizar fragmentos relevantes.
- Sintesis final con LLM y salida con bloque de fuentes.
- Script de chequeo basico de calidad para consultas de referencia.

## Arquitectura actual

- ui/: interfaz de usuario en Streamlit.
- src/agent/: logica del chat y transformacion de historial.
- src/config/: carga y validacion de variables de entorno, cliente Ollama.
- src/rag_engine/: ingesta, retriever y validacion de calidad RAG.
- src/tools/: herramienta consultar_teoria_drift para retrieval + sintesis.
- data/: conocimiento fuente, uploads temporales y persistencia vectorial.

## Requisitos previos

- Python 3.10 o superior.
- Entorno virtual (.venv) creado.
- Ollama instalado y en ejecucion.
- Modelo de chat descargado en Ollama.
- Modelo de embeddings disponible en Ollama: nomic-embed-text.

Comprobacion rapida:

```bash
python3 --version
ollama --version
```

## Configuracion inicial

Ejecuta estos comandos desde la raiz del proyecto (TFG/):

```bash
# 1) Activar entorno virtual
source .venv/bin/activate

# 2) Instalar dependencias del proyecto
python -m pip install --upgrade pip
pip install -r requirements.txt

# 3) Crear archivo de entorno local
cp .env.example .env
```

## Configurar .env

Variables usadas por chat:

- OLLAMA_BASE_URL: URL de Ollama local. Recomendado: <http://localhost:11434>
- OLLAMA_MODEL: nombre exacto del modelo de chat instalado en Ollama.
- OLLAMA_TEMPERATURE: temperatura de generacion (0.0 a 2.0).
- OLLAMA_REQUEST_TIMEOUT: timeout en segundos para chequeo de conexion.
- CHAT_SYSTEM_PROMPT: prompt de sistema por defecto para el asistente.
- CHAT_MAX_CONTEXT_TURNS: cantidad de turnos recientes enviados al LLM en cada inferencia.
- CHAT_SUMMARY_MAX_CHARS: tamano maximo del resumen incremental de turnos antiguos.

Variables usadas por RAG:

- RAG_TOP_K: cantidad inicial de documentos recuperados por similitud.
- RAG_KEEP_TOP: cantidad final de documentos tras re-ranking lexico.

Variables presentes en .env.example y no usadas en el flujo actual:

- RAG_SUMMARY_MAX_SENTENCES
- RAG_SUMMARY_MAX_CHARS

## Flujo operativo RAG (backend)

### 1) Construir base vectorial

```bash
cd /ruta/a/TFG
source .venv/bin/activate
PYTHONPATH=. python src/rag_engine/ingest.py
```

Resultado esperado:

- Se procesa el PDF tecnico en data/knowledge_base.
- Se persisten embeddings y metadatos en data/vector_db.

### 2) Ejecutar chequeo basico de calidad

```bash
cd /ruta/a/TFG
source .venv/bin/activate
PYTHONPATH=. python src/rag_engine/rag_quality_check.py
```

Resultado esperado:

- Pruebas PASS/WARN por caso de consulta.
- Resumen final con numero de fallos.

## Iniciar la app de chat

Usa dos terminales.

### Terminal A: Ollama

```bash
# Si el modelo no existe aun
ollama pull llama3.1:8b
ollama pull nomic-embed-text

# Levantar servicio local
ollama serve
```

### Terminal B: Streamlit

```bash
cd /ruta/a/TFG
source .venv/bin/activate
PYTHONPATH=. streamlit run ui/app.py
```

Nota importante: la UI actual usa el flujo de chat directo y no invoca todavia la tool RAG.

## Documentacion

- Flujo de app y flujo RAG: [docs/flujo_app.md](docs/flujo_app.md)
- Implementacion RAG (detalle tecnico): [docs/implementacion_rag.md](docs/implementacion_rag.md)
- Registro de errores y soluciones: [docs/errores_y_soluciones.md](docs/errores_y_soluciones.md)

Cada error nuevo detectado en desarrollo debe registrarse en docs/errores_y_soluciones.md.

## Estructura del repositorio

```text
TFG/
├── docs/
│   ├── errores_y_soluciones.md
│   ├── flujo_app.md
│   └── implementacion_rag.md
├── .env.example
├── README.md
├── requirements.txt
├── data/
│   ├── knowledge_base/
│   ├── temp_uploads/
│   └── vector_db/
├── src/
│   ├── agent/
│   │   └── simple_chat.py
│   ├── config/
│   │   └── llm_config.py
│   ├── rag_engine/
│   │   ├── ingest.py
│   │   ├── rag_quality_check.py
│   │   └── retriever.py
│   └── tools/
│       └── rag_tool.py
└── ui/
   └── app.py
```

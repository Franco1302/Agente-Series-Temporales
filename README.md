# Orquestador IA para Data Drift

Estado actual (mayo 2026):

- Chat local en Streamlit conectado a Ollama.
- Agente LangGraph ReAct con 6 nodos y persistencia por hilo.
- **RAG integrado en el grafo** vía la tool `consultar_teoria`.
- **Integración MCP con la API analítica drift-detection** (rama `integracion-mcp`): el agente expone 8 tools reales que llaman a `http://localhost:8017` a través de un servidor MCP local.
- Suite de tests pytest: 39 tests (34 unitarios mockeados con respx + 5 de integración contra la API real).

## Alcance implementado

### Chat UI y agente

- Interfaz en Streamlit con historial conversacional y carga de CSV vía sidebar.
- Streaming visual del progreso del agente con `st.status()`.
- Auto-renderizado inline de los artefactos (PNG/CSV) generados por las tools MCP.
- Grafo LangGraph con nodos `razonador`, `ejecutar_herramienta`, `solicitar_parametros`, `gestionar_error`, `recuperar_contexto`, `generar_respuesta`.

### Tools MCP (8 herramientas)

Expuestas por el paquete `mcp_server/` y arrancadas como subproceso stdio desde el agente:

| Tool | Familia | Endpoint base |
|---|---|---|
| `generate_synthetic_distribution` | Generación sintética | `GET /Datos/distribucion/{fin\|periodos}` |
| `generate_synthetic_arma` | Generación sintética | `GET /Datos/ARMA/{fin\|periodos}` |
| `generate_synthetic_periodic` | Generación sintética | `GET /Datos/periodicos/{fin\|periodos}` |
| `generate_synthetic_trend` | Generación sintética | `GET /Datos/tendencia/{fin\|periodos}` |
| `detect_drift` | Detección de drift | `POST /Deteccion/{KS\|JS\|PSI\|CUSUM\|MEWMA\|HOTELLING}` |
| `augment_time_series` | Aumentación de datos | `POST /Aumentar/{Normal\|Muller\|Duplicado\|Armonico\|Estadistica}` |
| `create_exogenous_variable` | Variables exógenas | `POST /Variables/{PCA\|Correlacion\|Covarianza\|Lineal\|Polinomico}` |
| `forecast_time_series` | Predicción | `POST /Datos/{Sarimax\|Prophet\|ForecasterAutoreg}` (+ `/Error/...`) |

Detalle de cada tool en [docs/arquitectura_mcp.md](docs/arquitectura_mcp.md).

### RAG backend

- Ingesta de PDF técnico a base vectorial Chroma local.
- Segmentación semántica por encabezados Markdown y chunking con overlap.
- Recuperación por similitud con embeddings en Ollama (`nomic-embed-text`).
- Re-ranking léxico local para priorizar fragmentos relevantes.
- Síntesis final con LLM y salida con bloque de fuentes.
- Tool `consultar_teoria` integrada en el grafo (nodo `recuperar_contexto`).

## Arquitectura actual

```
Streamlit ──► Agente LangGraph ──► Cliente MCP (subproceso stdio)
                                          │
                                          ▼
                                  Servidor MCP (FastMCP)
                                          │ httpx multipart
                                          ▼
                          API drift-detection (FastAPI :8017, Docker)
```

- `ui/` — interfaz de usuario en Streamlit.
- `src/agent/` — grafo LangGraph (nodos, routing, prompts, tools loader).
- `src/agent/tools/mcp_loader.py` — arranque del subproceso MCP.
- `src/config/` — carga y validación de variables de entorno, cliente Ollama.
- `src/rag_engine/` — ingesta, retriever y validación de calidad RAG.
- `src/tools/rag_tool.py` — herramienta `consultar_teoria`.
- `mcp_server/` — servidor MCP con las 8 tools que llaman a la API analítica.
- `tests/` — suite pytest (unit + integration).
- `data/` — conocimiento fuente, uploads temporales (workspace compartido con MCP) y persistencia vectorial.

## Requisitos previos

- Python 3.12 (testado con 3.12.3).
- Entorno virtual (`.venv`) creado.
- Ollama instalado y en ejecución.
- Modelo de chat con tool-calling descargado en Ollama (recomendado `llama3.1:8b-instruct-q4_K_M` o `qwen2.5:3b-instruct-q4_K_M` para iterar más rápido).
- Modelo de embeddings disponible en Ollama: `nomic-embed-text`.
- Docker con la imagen `drift-detection-api:latest` construida (la API se gestiona aparte).

Comprobación rápida:

```bash
python3 --version
ollama --version
docker --version
curl http://localhost:8017/        # debe responder cuando la API esté arrancada
```

## Configuración inicial

Desde la raíz del proyecto (TFG/):

```bash
# 1) Activar entorno virtual
source .venv/bin/activate

# 2) Instalar dependencias de runtime
python -m pip install --upgrade pip
pip install -r requirements.txt

# 3) (Opcional) Dependencias de tests
pip install -r requirements-dev.txt

# 4) Crear archivo de entorno local
cp .env.example .env
```

## Configurar .env

Variables usadas por chat:

- `OLLAMA_BASE_URL` — URL de Ollama local. Recomendado: `http://localhost:11434`.
- `OLLAMA_MODEL` — nombre exacto del modelo de chat instalado en Ollama.
- `OLLAMA_TEMPERATURE` — temperatura de generación (0.0 a 2.0).
- `OLLAMA_REQUEST_TIMEOUT` — timeout en segundos para llamadas a Ollama.
- `CHAT_SYSTEM_PROMPT` — prompt de sistema por defecto del asistente.
- `CHAT_MAX_CONTEXT_TURNS` — turnos recientes enviados al LLM en cada inferencia.
- `CHAT_SUMMARY_MAX_CHARS` — tamaño del resumen incremental de turnos antiguos.

Variables usadas por RAG:

- `RAG_TOP_K` — cantidad inicial de documentos recuperados.
- `RAG_KEEP_TOP` — cantidad final tras re-ranking léxico.

Variables MCP / API analítica:

- `DRIFT_API_URL` — URL base de la API. Default `http://localhost:8017`.
- `DRIFT_API_TIMEOUT` — read timeout (default 60 s).
- `DRIFT_API_CONNECT_TIMEOUT` — connect timeout (default 5 s).
- `MCP_WORKSPACE_DIR` — directorio donde el server guarda CSV/PNG. Default `data/temp_uploads`.
- `MCP_LOG_LEVEL` — nivel de log del server MCP. Default `INFO`.

## Flujo operativo

### 1) Construir base vectorial (única vez tras cambiar el PDF fuente)

```bash
source .venv/bin/activate
PYTHONPATH=. python src/rag_engine/ingest.py
```

### 2) Chequeo rápido de calidad RAG (opcional)

```bash
PYTHONPATH=. python src/rag_engine/rag_quality_check.py
```

### 3) Tests automáticos

```bash
PYTHONPATH=. pytest tests/ -q                # full suite (39 tests)
PYTHONPATH=. pytest -m integration -v        # solo integración (necesita la API en 8017)
PYTHONPATH=. pytest -m "not integration" -q  # solo unitarios
```

## Iniciar la app completa

Tres terminales:

### Terminal A — API analítica (Docker)

```bash
docker run --rm -p 8017:8017 drift-detection-api:latest
```

### Terminal B — Ollama

```bash
# Si los modelos no existen aún:
ollama pull llama3.1:8b-instruct-q4_K_M
ollama pull nomic-embed-text

# Levantar servicio:
ollama serve
```

### Terminal C — Streamlit + agente + servidor MCP

```bash
source .venv/bin/activate
PYTHONPATH=. streamlit run ui/app.py
```

El subproceso MCP arranca automáticamente al construir el grafo la primera vez. Abrir el navegador en `http://localhost:8501`.

## Comprobaciones rápidas

```bash
curl http://localhost:8017/      # API analítica responde
curl http://localhost:11434/      # Ollama responde
PYTHONPATH=. python -c "from src.agent.tools import AGENT_TOOLS; print(len(AGENT_TOOLS), [t.name for t in AGENT_TOOLS])"
# debe imprimir: 9 ['generate_synthetic_distribution', ..., 'consultar_teoria']
```

## Documentación

- Paso a paso de la integración MCP: [docs/integracion_mcp.md](docs/integracion_mcp.md).
- Arquitectura MCP y catálogo de las 8 tools: [docs/arquitectura_mcp.md](docs/arquitectura_mcp.md).
- Arquitectura general del agente: [docs/arquitectura_agente.md](docs/arquitectura_agente.md).
- Flujo de app y flujo RAG: [docs/flujo_app.md](docs/flujo_app.md).
- Implementación RAG (detalle técnico): [docs/implementacion_rag.md](docs/implementacion_rag.md).
- Registro de errores y soluciones: [docs/errores_y_soluciones.md](docs/errores_y_soluciones.md).

Cada error nuevo detectado en desarrollo debe registrarse en `docs/errores_y_soluciones.md`.

## Estructura del repositorio

```text
TFG/
├── docs/
│   ├── arquitectura_agente.md
│   ├── arquitectura_mcp.md         ← nuevo (rama integracion-mcp)
│   ├── bitacora_optimizacion_2026-05-06.md
│   ├── errores_y_soluciones.md
│   ├── flujo_app.md
│   ├── implementacion_rag.md
│   └── integracion_mcp.md          ← nuevo (rama integracion-mcp)
├── mcp_server/                     ← nuevo (rama integracion-mcp)
│   ├── instance.py
│   ├── config.py
│   ├── http_client.py
│   ├── file_utils.py
│   ├── errors.py
│   ├── server.py
│   └── tools/
│       ├── drift.py
│       ├── synthetic.py
│       ├── augment.py
│       ├── exogenous.py
│       └── forecast.py
├── src/
│   ├── agent/
│   │   ├── graph.py
│   │   ├── state.py
│   │   ├── nodes/
│   │   ├── prompts/
│   │   └── tools/
│   │       ├── __init__.py         ← AGENT_TOOLS = MCP tools + consultar_teoria
│   │       ├── mcp_loader.py       ← nuevo
│   │       ├── mock_drift.py       (conservado, no usado)
│   │       ├── mock_synthetic.py   (conservado, no usado)
│   │       └── mock_augment.py     (conservado, no usado)
│   ├── config/
│   │   └── llm_config.py
│   ├── rag_engine/
│   │   ├── ingest.py
│   │   ├── rag_quality_check.py
│   │   └── retriever.py
│   └── tools/
│       └── rag_tool.py
├── tests/                          ← nuevo (rama integracion-mcp)
│   ├── conftest.py
│   ├── test_mcp_server_unit.py
│   ├── test_mcp_tools_integration.py
│   └── fixtures/
│       └── sample_drift.csv
├── ui/
│   └── app.py
├── data/
│   ├── knowledge_base/
│   ├── temp_uploads/               # workspace compartido entre Streamlit y MCP
│   └── vector_db/
├── .env.example
├── pytest.ini                      ← nuevo
├── README.md
├── requirements.txt
└── requirements-dev.txt            ← nuevo
```

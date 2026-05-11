# Integración MCP — paso a paso

Documento de referencia de lo realizado en la rama `integracion-mcp`. Sustituye las 3 tools mock del agente por **8 tools reales** que llaman a la API analítica `drift-detection` (FastAPI, en `http://localhost:8017`) a través de un **servidor MCP local** arrancado como subproceso stdio.

---

## Estado antes de la integración

| Componente | Antes de la rama |
|---|---|
| Tools de cálculo del agente | 3 mocks deterministas: `detect_drift_kolmogorov_smirnov`, `generate_synthetic_series`, `augment_data_linear_relation` |
| Tool RAG (`consultar_teoria`) | Operativa e integrada en el grafo |
| Capa de transporte hacia la API real | Inexistente |
| Tests automáticos | Solo el chequeo de calidad del RAG |

## Estado tras la integración

| Componente | Después de la rama |
|---|---|
| Tools de cálculo del agente | **8 tools reales** vía MCP: 4 de generación sintética, 1 de detección de drift, 1 de aumentación, 1 de variables exógenas, 1 de forecast |
| Servidor MCP | Paquete `mcp_server/` que expone las 8 tools y traduce a `httpx` sobre `http://localhost:8017` |
| Tool RAG | Sin cambios, sigue en `AGENT_TOOLS` |
| Tests automáticos | 39 tests pytest: 34 unitarios (mockeados con `respx`) + 5 de integración contra la API real |

---

## Arquitectura resultante

```
┌──────────────── HOST (máquina del desarrollador) ───────────────┐
│                                                                  │
│   ┌────────────┐   in-process   ┌──────────────────────────┐    │
│   │ Streamlit  │◄───────────────┤   Agente LangGraph       │    │
│   │ ui/app.py  │                │   (build_agent_graph)    │    │
│   └────────────┘                └──────────┬───────────────┘    │
│                                            │ ToolNode            │
│                                            ▼                      │
│                                 ┌──────────────────────────┐    │
│                                 │ Cliente MCP (subproceso) │    │
│                                 │ MultiServerMCPClient     │    │
│                                 └──────────┬───────────────┘    │
│                                            │ JSON-RPC stdio      │
│                                            ▼                      │
│                                 ┌──────────────────────────┐    │
│                                 │ Servidor MCP             │    │
│                                 │ mcp_server.server        │    │
│                                 │ (FastMCP, 8 tools)       │    │
│                                 └──────────┬───────────────┘    │
│                                            │ httpx multipart     │
│   ┌────────────┐                           ▼                      │
│   │   Ollama   │              ┌────────────────────────────┐    │
│   │ :11434     │              │ API drift-detection        │    │
│   └────────────┘              │ FastAPI :8017 (Docker)     │    │
│                                └────────────────────────────┘    │
└──────────────────────────────────────────────────────────────────┘
```

Detalle visual de los componentes y el catálogo completo de las 8 tools: ver [docs/arquitectura_mcp.md](arquitectura_mcp.md).

---

## Trabajo realizado por fases

### Fase 0 — Preparación

| Archivo | Cambio |
|---|---|
| `requirements.txt` | +`mcp>=1.2.0`, `langchain-mcp-adapters>=0.0.7`, `httpx>=0.27.0` |
| `requirements-dev.txt` | Nuevo: `pytest`, `pytest-asyncio`, `respx` |
| `.env.example` | +bloque `DRIFT_API_URL`, `DRIFT_API_TIMEOUT`, `DRIFT_API_CONNECT_TIMEOUT`, `MCP_WORKSPACE_DIR`, `MCP_LOG_LEVEL` |

### Fase 1 — Esqueleto del servidor MCP

Nuevo paquete `mcp_server/`:

| Archivo | Responsabilidad |
|---|---|
| `mcp_server/__init__.py` | Docstring del paquete |
| `mcp_server/instance.py` | Instancia compartida `mcp = FastMCP("drift-mcp-server")` |
| `mcp_server/config.py` | `load_settings()` lee env vars y devuelve `ServerSettings` (frozen dataclass) |
| `mcp_server/http_client.py` | `get_client(settings)` asynccontextmanager con timeouts |
| `mcp_server/file_utils.py` | `deterministic_filename()`, `open_csv_for_upload()`, `save_streaming_response()` |
| `mcp_server/errors.py` | `translate_exception()`: convierte excepciones httpx/IO en strings legibles por el LLM |
| `mcp_server/server.py` | Entry point: importa los 5 módulos de tools y llama `mcp.run(transport="stdio")` |
| `mcp_server/tools/__init__.py` | Vacío (sólo agrupa el subpaquete) |

**Detalle técnico (Fase 1, ajuste posterior a runtime error)**: al arrancar como `python -m mcp_server.server`, Python ejecuta el módulo dos veces — una como `__main__` y otra como `mcp_server.server`. Si la instancia `mcp` vive en `server.py`, los submódulos de tools registran sus tools en la copia `mcp_server.server.mcp` mientras `main()` corre `mcp.run()` sobre la copia `__main__.mcp` (vacía). Resultado: 0 tools al cliente. Resuelto extrayendo `mcp = FastMCP(...)` a `mcp_server/instance.py`, módulo neutro al que ambas copias importan.

### Fase 2 — Implementación de las 8 tools

| Archivo | Tools | Endpoint(s) base | Tipo respuesta |
|---|---|---|---|
| `mcp_server/tools/drift.py` | `detect_drift` | `POST /Deteccion/{KS\|JS\|PSI\|CUSUM\|MEWMA\|HOTELLING}` | JSON |
| `mcp_server/tools/synthetic.py` | `generate_synthetic_distribution`, `_arma`, `_periodic`, `_trend` | `GET /Datos/{distribucion\|ARMA\|periodicos\|tendencia}/{fin\|periodos}` (+ `/Plot/...`) | CSV (+ PNG) |
| `mcp_server/tools/augment.py` | `augment_time_series` | `POST /Aumentar/{Normal\|Muller\|Duplicado\|Armonico\|Estadistica}` | CSV |
| `mcp_server/tools/exogenous.py` | `create_exogenous_variable` | `POST /Variables/{PCA\|Correlacion\|Covarianza\|Lineal\|Polinomico}` | CSV |
| `mcp_server/tools/forecast.py` | `forecast_time_series` | `POST /Datos/{Sarimax\|Prophet\|ForecasterAutoreg}` (+ `/Error/...`) | CSV (+ métricas) |

Patrón canónico de cada tool:

1. **Firma plana** con `Annotated[T, Field(description=...)]` — el LLM ve los parámetros al nivel raíz, no envueltos en un objeto `input`.
2. Dentro de la función, se construye el `XInput(**locals())` (modelo Pydantic interno) para validación y mapeo.
3. Un helper `_build_query_params(inp)` traduce los campos del modelo a los query params reales de la API (los nombres cambian: `index_column → indice`, `frequency → freq`, etc.).
4. Llamada `httpx` async con `multipart/form-data` cuando hay CSV de entrada.
5. La respuesta CSV o PNG se guarda en `data/temp_uploads/` con nombre determinista (hash SHA-1 de los argumentos).
6. La tool **nunca** devuelve binarios al LLM: sólo `output_path`, `image_path`, `summary` y campos descriptivos.
7. `try/except` que delega a `translate_exception(...)` y devuelve `{"error": "..."}` legible.

Reglas específicas:

- **`end_date xor periods`** en las 4 tools de generación: si el LLM pasa ambos o ninguno, se devuelve un string de error explicativo.
- **Mapeo `method → query params` en `detect_drift`**: cada método estadístico tiene su set de params propio (KS sólo `threshold_ks`; PSI añade `num_bins`; CUSUM añade `drift_cusum`; MEWMA/HOTELLING usan `min_instances`, `alpha`, `lambd`).
- **Mapeo `strategy → endpoint` en `augment_time_series`**: `normal/muller/harmonic` aceptan `size`; `duplicate` requiere `duplication_factor` + `perturbation_std` y NO acepta `size`; `statistical` requiere `tipo` + `num`.

### Fase 3 — Integración con el agente LangGraph

| Archivo | Cambio |
|---|---|
| `src/agent/tools/mcp_loader.py` (nuevo) | `load_mcp_tools_sync()` arranca el subproceso vía `MultiServerMCPClient` con stdio, cacheado con `@lru_cache(maxsize=1)` |
| `src/agent/tools/__init__.py` | Reemplazo: `AGENT_TOOLS = [*load_mcp_tools_sync(), consultar_teoria]`. Los ficheros `mock_*.py` se conservan intactos como rollback |
| `src/agent/nodes/param_request.py` | `TOOL_REQUIRED_PARAMS` y `_PARAM_DESCRIPTIONS` extendidos con los nombres y campos de las 8 tools. La función `solicitar_parametros_node` sin cambios |
| `src/agent/prompts/system_prompts.py` | `_TOOLS_BLOCK` reemplazado por una versión telegráfica con las 9 tools (8 MCP + `consultar_teoria`), triggers y obligatorios |
| `ui/app.py` | +función `_render_generated_artifacts(response_text, csv_path)` que detecta con regex rutas a `data/temp_uploads/*.{png,csv}` en la respuesta y las renderiza inline (PNG con `st.image`, CSV con `st.dataframe` + `st.download_button`) |

### Fase 4 — Tests

| Archivo | Cobertura |
|---|---|
| `tests/conftest.py` | Fixture `isolated_workspace` (autouse): exporta `DRIFT_API_URL=http://testserver`, redirige `MCP_WORKSPACE_DIR` a `tmp_path` y parchea `_SETTINGS` en cada módulo de tool. Fixture `sample_csv` |
| `tests/test_mcp_server_unit.py` | 34 tests: traducción de errores, `_build_query_params` de las 5 familias, end-to-end con `respx` (happy path + error path por cada tool), exclusión `end_date xor periods` |
| `tests/test_mcp_tools_integration.py` | 5 tests marcados con `@pytest.mark.integration` contra `http://localhost:8017` real: generación pura, drift sobre output, aumentación, fichero inexistente, drift sobre CSV fixture |
| `tests/fixtures/sample_drift.csv` | CSV de 20 filas (ts + valor) para los tests de integración |
| `pytest.ini` | `asyncio_mode = auto`, marker `integration` |

Ejecución:

```bash
PYTHONPATH=. pytest tests/ -q                # full suite (39 tests, 1s)
PYTHONPATH=. pytest -m integration -v        # solo integración (5 tests, ~1s)
PYTHONPATH=. pytest -m "not integration" -q  # solo unit (34 tests)
```

---

## Bug detectado y arreglado en pruebas E2E

Durante la prueba manual en Streamlit, el agente se quedaba "pensando" indefinidamente y rechazaba los parámetros varias veces aunque el usuario los hubiera dado bien.

**Causa raíz.** Las tools MCP se habían declarado en una primera iteración como `async def detect_drift(input: DetectDriftInput) -> dict`. FastMCP genera entonces un JSON Schema con la forma:

```json
{
  "$defs": {"DetectDriftInput": {...}},
  "properties": {"input": {"$ref": "#/$defs/DetectDriftInput"}}
}
```

El LLM tiene que emitir tool_calls con la forma `{"input": {"file_path": "...", "method": "PSI"}}`. Pero `src/agent/nodes/reasoning.py:142` revisa cada parámetro obligatorio al **nivel raíz** de `args`:

```python
required = TOOL_REQUIRED_PARAMS.get(tool_name, [])
missing = [p for p in required if p not in args or args[p] is None or args[p] == ""]
```

Como `"file_path" not in {"input": {...}}` siempre era cierto, el grafo enrutaba siempre a `solicitar_parametros` aunque el LLM hubiera pasado todos los datos correctos. Cuando el LLM por azar emitía la forma plana `{"file_path": "...", ...}`, la validación pasaba pero la tool MCP fallaba porque su firma esperaba un dict envuelto en `input`.

**Solución.** Aplanar las firmas de las 8 tools:

```python
@mcp.tool()
async def detect_drift(
    file_path: Annotated[str, Field(description="Ruta local al CSV en data/temp_uploads/.")],
    index_column: Annotated[str, Field(description="Nombre de la columna índice del CSV.")],
    method: Annotated[Literal["KS", "JS", "PSI", "CUSUM", "MEWMA", "HOTELLING"], Field(...)],
    inicio: Annotated[int, Field(...)] = 1,
    threshold: Annotated[Optional[float], Field(...)] = None,
    ...
) -> dict:
    inp = DetectDriftInput(file_path=file_path, index_column=index_column, ...)
    # resto igual
```

Tras el cambio:
- El JSON Schema visto por el cliente MCP tiene todos los params al nivel raíz, `$defs: false`.
- `TOOL_REQUIRED_PARAMS["detect_drift"] = ["file_path", "index_column", "method"]` se valida correctamente contra `args = {"file_path": ..., "method": "PSI"}`.
- Los modelos Pydantic internos (`DetectDriftInput`, `AugmentTimeSeriesInput`, etc.) se conservan para reusar los helpers `_build_query_params` y para los tests unitarios.

**Verificación E2E manual** con `qwen2.5:3b-instruct-q4_K_M`: el grafo visita `razonador → ejecutar_herramienta → razonador → recuperar_contexto → razonador` y devuelve respuesta en ~62 s, sin atascos ni reintentos de pedir parámetros.

---

## Cómo arrancar el sistema completo

Tres terminales (la API se gestiona por separado, no hay `docker-compose.yml` en esta rama):

```bash
# Terminal 1 — API drift-detection
docker run --rm -p 8017:8017 drift-detection-api:latest

# Terminal 2 — Ollama
ollama serve

# Terminal 3 — Agente + Streamlit
cd /home/franco/Documentos/TFG
source .venv/bin/activate
pip install -r requirements.txt -r requirements-dev.txt   # primera vez
PYTHONPATH=. streamlit run ui/app.py
```

El subproceso MCP arranca automáticamente al construir el grafo (la primera vez que Streamlit invoca `build_agent_graph()`).

Comprobación rápida:

```bash
curl http://localhost:8017/      # API responde con {"Contenido": "..."}
curl http://localhost:11434/     # Ollama responde
PYTHONPATH=. pytest tests/ -q    # 39 tests pasan
```

---

## Fuera del scope de esta rama

Lo siguiente queda pendiente para iteraciones posteriores:

- `docker-compose.yml` que orqueste el contenedor de la API.
- Documentación operativa adicional: `README_INTEGRATION.md`.
- Actualización de `mcp_benchmark.py` a las 8 tools reales (hoy sigue apuntando a los 3 mocks).
- Benchmark formal de modelos Ollama y elección del modelo por defecto.
- Detalle de baja latencia: el cliente `langchain-mcp-adapters` abre una nueva sesión MCP por cada tool call (con su cold-start). Si la latencia se vuelve insoportable, valorar transporte `streamable_http` con el server arrancado como proceso largo.

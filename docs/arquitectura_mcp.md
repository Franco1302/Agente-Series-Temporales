# Arquitectura MCP del agente

Este documento describe los componentes que participan en la ejecución de una herramienta analítica desde la UI hasta la API real, los contratos de las 8 tools expuestas por el servidor MCP, y los puntos de configuración.

Para el paso a paso del trabajo realizado, ver [docs/integracion_mcp.md](integracion_mcp.md).

---

## 1. Vista de despliegue

```
┌──────────────── HOST (máquina del desarrollador) ───────────────────┐
│                                                                      │
│   ┌────────────┐   in-process   ┌──────────────────────────────┐    │
│   │ Streamlit  │◄───────────────┤   Agente LangGraph           │    │
│   │ ui/app.py  │                │   build_agent_graph()        │    │
│   │            │                │                              │    │
│   │ • upload   │                │   ┌────────────────────┐     │    │
│   │ • status() │                │   │ razonador          │     │    │
│   │ • render   │                │   │ ejecutar_hta       │     │    │
│   │   PNG/CSV  │                │   │ solicitar_params   │     │    │
│   └────────────┘                │   │ gestionar_error    │     │    │
│        ▲                        │   │ recuperar_contexto │     │    │
│        │ progreso               │   │ generar_respuesta  │     │    │
│        │                        │   └──────────┬─────────┘     │    │
│        │                        └──────────────┼───────────────┘    │
│        │                                       │ ToolNode            │
│        │                                       ▼                      │
│        │                            ┌──────────────────────────┐    │
│        │                            │ Cliente MCP (subproceso) │    │
│        │                            │ MultiServerMCPClient     │    │
│        │                            │ stdio / JSON-RPC         │    │
│        │                            └──────────┬───────────────┘    │
│        │                                       │                      │
│        │                                       ▼                      │
│        │                            ┌──────────────────────────┐    │
│        │                            │ Servidor MCP             │    │
│        │                            │ mcp_server.server        │    │
│        │                            │ FastMCP("drift-mcp-...")  │    │
│        │                            │                          │    │
│        │                            │ 8 tools @mcp.tool()      │    │
│        │                            │ + httpx async client     │    │
│        │                            └──────────┬───────────────┘    │
│        │                                       │ HTTP multipart      │
│   ┌────────────┐                               ▼                      │
│   │   Ollama   │             ┌────────────────────────────────────┐ │
│   │ :11434     │             │ Contenedor Docker                  │ │
│   │ qwen/llama │             │ drift-detection-api:latest         │ │
│   │            │             │ FastAPI :8017                      │ │
│   └────────────┘             └────────────────────────────────────┘ │
└──────────────────────────────────────────────────────────────────────┘
```

---

## 2. Componentes nuevos (paquete `mcp_server/`)

```
mcp_server/
├── __init__.py          paquete vacío
├── instance.py          mcp = FastMCP("drift-mcp-server")  ←  instancia compartida
├── config.py            ServerSettings (frozen) + load_settings()
├── http_client.py       get_client(settings) asynccontextmanager
├── file_utils.py        deterministic_filename, open_csv_for_upload
├── errors.py            translate_exception(exc, tool_name) → str legible
├── server.py            entry point: main() → mcp.run(transport="stdio")
└── tools/
    ├── __init__.py
    ├── drift.py         detect_drift
    ├── synthetic.py     generate_synthetic_{distribution,arma,periodic,trend}
    ├── augment.py       augment_time_series
    ├── exogenous.py     create_exogenous_variable
    └── forecast.py      forecast_time_series
```

Importante: la instancia `mcp` vive en `mcp_server.instance`, no en `mcp_server.server`. Esto evita la duplicación de módulo que ocurre cuando se ejecuta `python -m mcp_server.server` y los submódulos hacen `from mcp_server.server import mcp` (Python ejecuta el módulo dos veces — `__main__` y `mcp_server.server`).

---

## 3. Flujo de una invocación end-to-end

Ejemplo: el usuario escribe en Streamlit *"Detecta drift en `data/temp_uploads/sample.csv` con el método PSI sobre la columna índice `ts`"*.

```
1. Streamlit (_run_agent_streaming)
   └─→ graph.stream({messages: [HumanMessage("Detecta drift...")], csv_path: "..."})

2. razonador_node
   └─→ build_system_prompt(csv_path=...)           # incluye bloque FICHERO ACTIVO
   └─→ llm.invoke(messages)                         # qwen2.5:3b o llama3.1:8b
   └─→ AIMessage(tool_calls=[{name: "detect_drift",
                              args: {file_path: "...", index_column: "ts", method: "PSI"}}])
   └─→ get_missing_params("detect_drift", args) → []
   └─→ pending_tool = None                          # todo OK

3. route_after_razonador
   └─→ "ejecutar_herramienta"

4. tool_execution_node  (ToolNode de LangGraph)
   └─→ la BaseTool de detect_drift se ejecuta:
       └─→ langchain_mcp_adapters abre sesión stdio con el subproceso MCP
       └─→ envía JSON-RPC tools/call con args planos
       └─→ Servidor MCP recibe → ejecuta detect_drift(file_path=..., method="PSI")
           └─→ open_csv_for_upload(file_path)
           └─→ _build_query_params() → {"indice": "ts", "threshold_psi": 0.25, ...}
           └─→ httpx.post("/Deteccion/PSI", params=..., files={"file": ...})
           └─→ recibe JSON {"Drift": "...", "reporte": {...}}
           └─→ return {drift_detected, summary, ...}
       └─→ ToolMessage(content=json.dumps(result), name="detect_drift")

5. route_after_tool → "razonador"

6. razonador_node (segunda vuelta)
   └─→ LLM ve el ToolMessage con el resultado
   └─→ AIMessage(content="Se ha detectado drift en la columna...")  ←  sin tool_calls

7. route_after_razonador → "fin"

8. Streamlit
   └─→ _render_generated_artifacts(final_response, csv_path)
       └─→ detecta rutas a *.png / *.csv en data/temp_uploads/
       └─→ render inline (st.image, st.dataframe + download_button)
```

---

## 4. Catálogo de las 8 tools

Convenciones comunes:

- Los parámetros se exponen al LLM al **nivel raíz** (schema plano, sin envoltorio `input`).
- `file_path` referencia siempre un fichero local dentro de `data/temp_uploads/`.
- La tool nunca devuelve binarios al LLM. CSV → `{"output_path": "..."}`. PNG → `{"image_path": "..."}`.
- Todas las respuestas incluyen un campo `summary` con descripción humana.
- En caso de excepción, la tool devuelve `{"error": "..."}` legible (nunca crash del server).

### 4.1. `detect_drift`

**Endpoint API:** `POST /Deteccion/{KS|JS|PSI|CUSUM|MEWMA|HOTELLING}`

| Parámetro | Tipo | Obligatorio | Notas |
|---|---|---|---|
| `file_path` | `str` | sí | Ruta al CSV |
| `index_column` | `str` | sí | Columna índice temporal del CSV |
| `method` | `Literal["KS","JS","PSI","CUSUM","MEWMA","HOTELLING"]` | sí | Tipo de test |
| `inicio` | `int` | no (=1) | Índice de comienzo |
| `threshold` | `float?` | no | Default por método: KS 0.05, JS 0.2, PSI 0.25, CUSUM 1.5 |
| `num_bins` | `int?` | no | Solo PSI (default 10) |
| `drift_cusum` | `float?` | no | Solo CUSUM (default 0.5) |
| `min_instances` | `int?` | no | Solo MEWMA/HOTELLING (default 100) |
| `lambd` | `float?` | no | Solo MEWMA (default 0.5) |
| `alpha` | `float?` | no | Solo MEWMA/HOTELLING (default 0) |

**Respuesta:** `{drift_detected, drift_label, per_column_report, method_used, parameters_used, summary}`.

### 4.2. `generate_synthetic_distribution`

**Endpoint API:** `GET /Datos/distribucion/{fin|periodos}` (+ `/Plot/distribucion/...` si `with_plot=True`).

| Parámetro | Tipo | Obligatorio | Notas |
|---|---|---|---|
| `start_date` | `str` | sí | `'YYYY-MM-DD'` |
| `frequency` | `Literal["B","D","W","M","Q","Y","h","min","s"]` | sí | |
| `distribution_type` | `int` 1–17 | sí | 1=Normal, 3=Poisson, 7=Uniforme, 9=Exponencial, etc. |
| `distribution_params` | `list[float]` | sí | Params de la distribución |
| `end_date` | `str?` | xor `periods` | Excluyente con `periods` |
| `periods` | `int?` | xor `end_date` | Excluyente con `end_date` |
| `column_name` | `str` | no (="valor") | |
| `with_plot` | `bool` | no | Si True, también genera PNG |

**Respuesta:** `{output_path, rows_generated, image_path, summary}`.

### 4.3. `generate_synthetic_arma`

**Endpoint API:** `GET /Datos/ARMA/{fin|periodos}`.

Params obligatorios: `start_date`, `frequency`. Opcionales: `constant`, `noise_std`, `seasonality`, `ar_coefficients`, `ma_coefficients`, `end_date xor periods`, `column_name`, `with_plot`.

**Respuesta:** `{output_path, rows_generated, image_path, model_spec, summary}`.

### 4.4. `generate_synthetic_periodic`

**Endpoint API:** `GET /Datos/periodicos/{fin|periodos}`.

Params obligatorios: `start_date`, `frequency`, `distribution_type`, `distribution_params`, `period_length`, `pattern_type` (1=amplitud, 2=cantidad).

**Respuesta:** `{output_path, rows_generated, image_path, summary}`.

### 4.5. `generate_synthetic_trend`

**Endpoint API:** `GET /Datos/tendencia/{fin|periodos}`.

Params obligatorios: `start_date`, `frequency`, `trend_type` (1=lineal, etc.), `trend_params`. Opcional: `noise`.

**Respuesta:** `{output_path, rows_generated, image_path, summary}`.

### 4.6. `augment_time_series`

**Endpoint API:** `POST /Aumentar/{Normal|Muller|Duplicado|Armonico|Estadistica}`.

| Parámetro | Tipo | Obligatorio | Notas |
|---|---|---|---|
| `file_path` | `str` | sí | |
| `index_column` | `str` | sí | |
| `strategy` | `Literal["normal","muller","duplicate","harmonic","statistical"]` | sí | |
| `size` | `int` | sí | Observaciones nuevas |
| `frequency` | `Literal` | sí | |
| `duplication_factor` | `float?` | no | Sólo `duplicate` (default 0.5) |
| `perturbation_std` | `float?` | no | Sólo `duplicate` (default 0.1) |
| `statistical_type` | `int?` | no | Sólo `statistical` (default 1) |

Mapeo `strategy → endpoint`:

| `strategy` | Endpoint | Params extra del endpoint |
|---|---|---|
| `normal` | `/Aumentar/Normal` | `size` |
| `muller` | `/Aumentar/Muller` | `size` |
| `harmonic` | `/Aumentar/Armonico` | `size` |
| `duplicate` | `/Aumentar/Duplicado` | `duplication_factor`, `perturbation_std` (NO acepta `size`) |
| `statistical` | `/Aumentar/Estadistica` | `tipo`, `num` (mapea desde `size`) |

**Respuesta:** `{output_path, new_rows, strategy_used, image_path, summary}`.

### 4.7. `create_exogenous_variable`

**Endpoint API:** `POST /Variables/{PCA|Correlacion|Covarianza|Lineal|Polinomico}`.

| Parámetro | Tipo | Obligatorio | Notas |
|---|---|---|---|
| `file_path` | `str` | sí | |
| `index_column` | `str` | sí | |
| `new_column_name` | `str` | sí | |
| `relation` | `Literal["pca","correlation","covariance","linear","polynomial"]` | sí | |
| `coefficients` | `list[float]?` | sólo `linear`/`polynomial` | linear → `[slope, intercept]`; polynomial → `[c0, c1, c2, ...]` |

**Respuesta:** `{output_path, new_column_name, relation_used, image_path, summary}`.

### 4.8. `forecast_time_series`

**Endpoint API:** `POST /Datos/{Sarimax|Prophet|ForecasterAutoreg}` (+ `/Error/...` si `return_metrics=True`).

| Parámetro | Tipo | Obligatorio | Notas |
|---|---|---|---|
| `file_path` | `str` | sí | |
| `index_column` | `str` | sí | |
| `target_column` | `str` | sí | Sólo descriptivo; la API toma la última columna |
| `model` | `Literal["sarimax","prophet","forecaster_autoreg"]` | sí | |
| `forecast_steps` | `int` | sí | Horizonte (mapea a `size`) |
| `frequency` | `Literal` | no (="D") | Debe coincidir con el CSV |
| `regressor` | `str?` | sólo `forecaster_autoreg` | Default `"RandomForestRegressor"` |
| `return_metrics` | `bool` | no (=True) | Hace segunda llamada a `/Error/...` |

**Respuesta:** `{output_path, metrics, model_used, image_path, summary}`.

---

## 5. Configuración (variables de entorno)

| Variable | Default | Descripción |
|---|---|---|
| `DRIFT_API_URL` | `http://localhost:8017` | URL base de la API drift-detection |
| `DRIFT_API_TIMEOUT` | `60` | Read timeout en segundos |
| `DRIFT_API_CONNECT_TIMEOUT` | `5` | Connect timeout en segundos |
| `MCP_WORKSPACE_DIR` | `data/temp_uploads` | Carpeta donde el server guarda CSV/PNG generados |
| `MCP_LOG_LEVEL` | `INFO` | Nivel de log del server (stderr) |

Estos values se leen en `mcp_server/config.py::load_settings()`, que produce un `ServerSettings` frozen dataclass. Cada módulo de tool cachea su propia copia en una variable `_SETTINGS` al import. Los tests parchean esa variable directamente (ver `tests/conftest.py`).

---

## 6. Modelo de datos del workspace

Todos los artefactos (uploads del usuario y salidas del MCP) viven en `data/temp_uploads/`:

```
data/temp_uploads/
├── <upload_del_usuario>.csv               # subido vía sidebar de Streamlit
├── distribucion_<hash8>.csv               # output de generate_synthetic_distribution
├── distribucion_<hash8>.png               # plot opcional
├── arma_<hash8>.csv                       # output de generate_synthetic_arma
├── augment_normal_<hash8>.csv             # output de augment_time_series strategy="normal"
├── exogenous_pca_<hash8>.csv              # output de create_exogenous_variable relation="pca"
├── forecast_sarimax_<hash8>.csv           # output de forecast_time_series model="sarimax"
└── ...
```

El hash es SHA-1 truncado a 8 caracteres calculado sobre los argumentos no triviales de la tool. Esto da naming determinista: dos invocaciones idénticas escriben el mismo fichero (caching natural sin lógica explícita).

---

## 7. Limitaciones conocidas

- **Cold-start del subproceso MCP por tool call.** `langchain-mcp-adapters` v0.2.2 abre una nueva sesión stdio por cada invocación de tool. Añade ~3 s de overhead por llamada. Con modelos 3B encadenando varias tools, el flujo puede tardar 60 s. Mitigación futura: transporte `streamable_http` con el server arrancado como proceso largo aparte.
- **`target_column` en forecast no se envía.** Los endpoints `/Datos/{Sarimax|Prophet|...}` de la API no aceptan `target_column` como query param: deducen la columna a predecir del CSV. El campo se conserva en la firma de la tool por claridad para el LLM y por aparecer en `summary`, pero no afecta la llamada.
- **`base_column` en exógenas no se envía.** Los endpoints `/Variables/...` no aceptan `base_column`: la API decide. La tool no expone ese campo.

# Plan: cierre de integración MCP + scoring de tests

Documento vivo. Recoge el plan acordado para cerrar la integración MCP y añadir
un sistema de tests con porcentaje de acierto a dos niveles (API y agente).
Actualizar la sección **Estado** conforme se avance.

---

## Contexto

La integración MCP en la rama `integracion-mcp` está funcional: 5 tools MCP
(drift, synthetic, augment, exogenous, forecast) + 1 tool local de RAG
(`consultar_teoria`), expuestas al grafo LangGraph vía `MultiServerMCPClient`
(`src/agent/tools/mcp_loader.py`). Tres huecos antes de poder evaluar
objetivamente el sistema:

1. **El CSV viaja como ruta** porque el backend HTTP en `localhost:8017` espera
   `multipart/form-data`. El UI guarda el upload en `data/temp_uploads/` y pasa
   el path al grafo (`ui/app.py:50-55, 149`). **Es correcto**, no requiere
   cambio — solo documentarlo.
2. **Los artefactos generados (PNG/CSV) no se ven en la UI** aunque existen en
   disco. Las tools devuelven `{"output_path": ..., "image_path": ...}` dentro
   de `ToolMessage.content`, pero el UI antes solo buscaba rutas con regex en
   el texto final del LLM. Si el modelo no citaba literalmente la ruta, no se
   renderizaba nada.
3. **No hay tests con porcentaje de acierto**. Existen: `mcp_benchmark.py`
   (single-turn tool-calling, no agente completo), `scripts/test_agent.py`
   (grafo completo, pasa/falla binario) y la suite de `tests/` (unitarios
   mockeados + integración real con asserts de existencia). Falta poder
   afirmar "el agente acierta al X% en el escenario Y" y "el backend responde
   al Z% como contrato en el escenario W".

---

## Parte A — Render de artefactos parseando ToolMessage en el UI ✅ COMPLETADO

**Decisión**: el UI parsea `ToolMessage.content` directamente; no depende de
que el LLM cite la ruta.

**Cambios realizados en `ui/app.py`**:

- Eliminado `_ARTIFACT_PATH_PATTERN` (el regex que escaneaba el texto del LLM).
- Añadido `_collect_tool_artifacts(tool_messages, csv_path) -> list[Path]`:
  parsea `msg.content` con `json.loads()` (con fallback si no es JSON), extrae
  `output_path`/`image_path` que apunten a `data/temp_uploads/` y devuelve la
  lista deduplicada.
- En `_run_agent_streaming`, los `ToolMessage` emitidos durante el streaming
  se acumulan en `tool_messages` y se pasan a `_render_generated_artifacts()`
  después de cerrar el `st.status`.
- `_render_generated_artifacts(artifacts: list[Path])` mantiene el render:
  `st.image` para PNG, expander + dataframe + download_button para CSV.

**Por qué este enfoque**: el contrato del MCP server ya emite los paths
estructurados (`mcp_server/tools/augment.py:138-146`,
`mcp_server/tools/forecast.py:145-154`, etc.). Parsear el dict es robusto
frente a cambios en cómo el LLM redacta su respuesta y sobrevive a modelos
pequeños que omiten detalles.

---

## Parte B — Tests API determinísticos con scoring 🟡 EN PROGRESO

**Decisión** (modificada respecto al plan inicial tras feedback del usuario):
**NO** se generan CSV pre-construidos con numpy. En su lugar, los tests usan
las propias tools de generación del MCP (`generate_synthetic_*`) como input
para los tests downstream. Esto evita duplicar la lógica de generación en
Python y testea el camino real que ven los usuarios.

Ver memoria: `feedback_test_data_via_api.md` ("Para tests de integración del
MCP/agente, generar inputs llamando a las propias tools del sistema").

### Implementación

**Estructura**:
```
tests/
  api_contracts/
    __init__.py
    conftest.py              ← hooks pytest + fixtures session-scoped
    test_api_contracts.py    ← 9 tests con @pytest.mark.weight()
  results/
    .gitignore               ← ignora *.csv generados al ejecutar
```

**`tests/api_contracts/conftest.py`** ✅ creado:

- Sobrescribe el autouse `isolated_workspace` del conftest padre con un no-op
  (porque allí monkeypatchea por test; aquí queremos workspace de sesión).
- `session_workspace` (autouse, scope=session): crea un workspace compartido
  con `tmp_path_factory`, exporta `DRIFT_API_URL` y `MCP_WORKSPACE_DIR`, y
  parchea `_SETTINGS` en los módulos de tools.
- `ensure_api_alive` (autouse, scope=session): salta toda la suite si la API
  no responde.
- Fixtures session-scoped que generan los CSV input una sola vez:
  - `trend_csv_path` → `generate_synthetic_trend` (lineal, slope=0.1) → drift YES
  - `stable_csv_path` → `generate_synthetic_distribution` (Normal[0,1]) → drift NO
  - `periodic_csv_path` → `generate_synthetic_periodic` (período 30) → input forecast
- Hooks `pytest_configure` + `pytest_runtest_makereport` + `pytest_sessionfinish`
  que registran outcomes con `@pytest.mark.weight(N)` y emiten al final
  `tests/results/api_scorecard.csv` con porcentaje agregado.

**`tests/api_contracts/test_api_contracts.py`** ✅ creado (9 tests, peso total
110 puntos):

| Test | Tool MCP | Peso |
|---|---|---|
| `test_synthetic_returns_expected_keys` | generate_synthetic_distribution | 10 |
| `test_synthetic_with_plot_emits_image_path` | generate_synthetic_distribution | 10 |
| `test_drift_ks_detects_on_trend` | detect_drift (KS sobre trend) | 20 |
| `test_drift_ks_no_drift_on_stable` | detect_drift (KS sobre estable) | 15 |
| `test_drift_method_routing` | detect_drift (KS/JS/PSI) | 10 |
| `test_drift_error_on_missing_file` | detect_drift (path inválido) | 5 |
| `test_augment_extends_csv` | augment_time_series | 15 |
| `test_exogenous_linear_adds_column` | create_exogenous_variable | 10 |
| `test_forecast_sarimax_on_periodic` | forecast_time_series | 15 |

### Pendiente en Parte B

- ⏳ **Ejecutar contra la API real** y validar que pasan: `ollama serve` no es
  necesario, pero sí el backend `drift-detection` en `localhost:8017`.
- ⏳ Ajustar umbrales/asserts si algún test falla por razones legítimas (ej.
  RNG del backend produce false positive en KS sobre estable).

Comando:
```bash
PYTHONPATH=. pytest -m integration tests/api_contracts -v
cat tests/results/api_scorecard.csv
```

---

## Parte C — Tests del agente con scoring ponderado ⏳ PENDIENTE

**Decisión**: extender `scripts/test_agent.py` con scoring 40/25/20/15 = 100 por
caso. Los 4 criterios (todos seleccionados por el usuario):

| Criterio | Peso | Descripción |
|---|---|---|
| Tool correcta invocada | 40 | El agente llamó la tool MCP esperada (o ninguna si era RAG-only) |
| Args correctos y completos | 25 | Los argumentos JSON coinciden con lo esperado |
| Términos clave en respuesta | 20 | La respuesta natural contiene los conceptos esperados |
| Cita rutas de artefactos | 15 | Si la tool generó CSV/PNG, la respuesta cita output_path/image_path |

### Tareas

#### C1: helpers compartidos de scoring ⏳

Extraer `evaluar_tool_call` y `_comparar_args` de `mcp_benchmark.py:341-355`
a `scripts/_scoring_utils.py` para reutilizar sin duplicar.

#### C2: agent tests con scoring ⏳

Cambios en `scripts/test_agent.py`:

- Extender `TestCase` añadiendo:
  - `tool_esperada: str | None`
  - `args_esperados: dict`
  - `genera_artefacto: bool`
- Reemplazar `_run_case` con `_score_case(caso) -> dict` que devuelve los 4
  scores parciales y el total.
- Reescribir `_print_summary` con tabla de % por caso + global.
- Añadir flag CLI `--csv-out tests/results/agent_scorecard.csv`.
- Actualizar los 6 casos existentes en `CASOS` con los campos nuevos.
- Añadir 2-3 casos que ejerciten generación de artefactos (augment con CSV,
  forecast con PNG) para cubrir el criterio "cita rutas".

Comando objetivo:
```bash
python -m scripts.test_agent --todos --csv-out tests/results/agent_scorecard.csv
```

---

## Verificación end-to-end ⏳ PENDIENTE

### 1. UI artifacts (Parte A)
```bash
# Terminal 1
ollama serve
# Terminal 2
PYTHONPATH=. streamlit run ui/app.py
```
- Subir un CSV y pedir generación sintética; verificar que el PNG/CSV
  resultante se renderiza inline sin que el LLM cite la ruta.
- Pedir augment / forecast del CSV subido; confirmar previews descargables.

### 2. API scorecard (Parte B)
```bash
# Asegurar backend MCP corriendo en localhost:8017
PYTHONPATH=. pytest -m integration tests/api_contracts -v
cat tests/results/api_scorecard.csv
```
Esperado: % global ≥ 95% (tolerancias absorben ruido estadístico legítimo).

### 3. Agent scorecard (Parte C)
```bash
python -m scripts.test_agent --todos --csv-out tests/results/agent_scorecard.csv
```
Esperado: tabla por caso + % global. Con qwen2.5-instruct, ≥ 90% según
benchmarks previos en `resultados_benchmark.csv`.

### 4. Tests unitarios sin regresión
```bash
pytest tests/test_mcp_server_unit.py -v
```

---

## Estado actual (actualizar al avanzar)

- ✅ **Parte A**: UI parsea ToolMessage. Sintaxis validada, falta probar end-to-end con UI viva.
- 🟡 **Parte B**: conftest + tests creados (9 tests, 110 pts total). Falta ejecutar contra API real y ajustar.
- ⏳ **Parte C1**: helpers compartidos sin extraer.
- ⏳ **Parte C2**: scoring del agente sin implementar.
- ⏳ **Verificación**: pendiente las cuatro fases.

---

## Archivos clave referenciados

- `ui/app.py` — UI Streamlit (Parte A modificada)
- `mcp_server/tools/{drift,synthetic,augment,exogenous,forecast}.py` — contratos MCP
- `mcp_server/file_utils.py:11-34` — `open_csv_for_upload`, `deterministic_filename`
- `src/agent/tools/mcp_loader.py:21-50` — carga de tools MCP en el grafo
- `tests/api_contracts/conftest.py` — fixtures + scoring (NUEVO)
- `tests/api_contracts/test_api_contracts.py` — 9 tests ponderados (NUEVO)
- `scripts/test_agent.py:22-203` — base del scoring del agente (Parte C)
- `mcp_benchmark.py:341-355` — fórmula de scoring reutilizable
- `tests/test_mcp_server_unit.py` — patrón pytest + respx (referencia)
- `tests/test_mcp_tools_integration.py` — patrón integration (referencia)
- `pytest.ini` — `integration` marker

## Fuera de alcance

- No se cambia el system_prompt para forzar citación de rutas (descartado).
- No se toca `mcp_benchmark.py` (otro propósito: benchmark single-turn de modelos).
- No se refactorizan las tools MCP ni el backend.
- No se añaden tests de RAG con scoring (cubierto por `rag_quality_check.py`).

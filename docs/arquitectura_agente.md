# Arquitectura del agente LangGraph

## Visión general

El agente implementa el patrón **ReAct** (Reason + Act) usando LangGraph. En cada turno de conversación, el LLM razona sobre la petición del usuario, decide si necesita ejecutar una herramienta y, si es así, la ejecuta y vuelve a razonar con el resultado hasta producir una respuesta final.

```
                    ┌─────────────────────────────────────┐
                    │           GRAFO DEL AGENTE          │
                    │                                     │
  HumanMessage ──▶ START ──▶ reasoning_node              │
                    │              │                      │
                    │    ┌─────────┴──────────┐           │
                    │    │ ¿tool_calls?        │           │
                    │    ├── No ──────────────▶ END        │
                    │    └── Sí ──▶ param_validation_node │
                    │                   │                  │
                    │         ┌─────────┴───────────┐      │
                    │         │ ¿parámetros OK?      │      │
                    │         ├── No ───────────────▶ END   │
                    │         └── Sí ──▶ tool_execution_node│
                    │                        │              │
                    │                        └──────────────┘
                    │                    (ciclo, máx. 5 iter.)
                    └─────────────────────────────────────┘
```

---

## AgentState

Definido en `src/agent/state.py`. Es el dict que fluye entre todos los nodos.

| Campo | Tipo | Descripción |
|---|---|---|
| `messages` | `Annotated[list, add_messages]` | Historial completo. El reducer `add_messages` acumula en lugar de sobrescribir |
| `tool_results` | `dict[str, Any]` | Últimos resultados de tools ejecutadas: `{nombre: resultado}` |
| `rag_context` | `str` | Reservado para contexto RAG futuro (vacío en esta iteración) |
| `pending_params` | `list[str]` | Parámetros detectados como faltantes en la última tool call |
| `uploaded_file_path` | `str \| None` | Ruta al CSV activo subido desde el sidebar |
| `iteration_count` | `int` | Contador de ciclos ReAct; fuerza END si supera `_MAX_ITERATIONS` |

---

## Nodos

### `reasoning_node` (`src/agent/nodes/reasoning.py`)

Responsabilidad: invocar el LLM con el estado actual y decidir la siguiente acción.

- Inyecta el `SystemMessage` al inicio del historial si no existe ya.
- El system prompt se construye dinámicamente con `build_system_prompt(has_uploaded_file, file_info)`.
- Llama a `get_llm_with_tools(AGENT_TOOLS)` para obtener el LLM con tools enlazadas.
- Si `iteration_count >= _MAX_ITERATIONS` (5), devuelve un mensaje de parada sin invocar el LLM.
- Incrementa `iteration_count` en cada ejecución.

### `param_validation_node` (`src/agent/nodes/param_validation.py`)

Responsabilidad: validar que los argumentos de las tool calls son completos antes de ejecutar.

- Compara los argumentos de cada tool call contra `TOOL_REQUIRED_PARAMS`.
- Si falta algún parámetro: genera un `AIMessage` en español pidiendo exactamente lo que falta, con descripciones legibles por el usuario.
- Si todo está completo: devuelve `pending_params=[]` sin modificar el estado.
- Impide que el agente ejecute herramientas con argumentos incompletos o inventados.

**`TOOL_REQUIRED_PARAMS`** (en `param_validation.py`):

```python
{
    "detect_drift_kolmogorov_smirnov": ["file_path", "reference_column"],
    "generate_synthetic_series": ["start_date", "periods", "frequency",
                                  "distribution_type", "distribution_params"],
    "augment_data_linear_relation": ["file_path", "index_column",
                                     "new_column_name", "slope", "intercept"],
}
```

### `tool_execution_node` (`src/agent/nodes/tool_execution.py`)

Responsabilidad: ejecutar las tool calls aprobadas por `param_validation_node`.

- Delega en `ToolNode` de LangGraph, que localiza la herramienta por nombre y la invoca con los argumentos.
- Actualiza `tool_results` con `{nombre_herramienta: contenido_ToolMessage}`.
- Produce `ToolMessage` objects que el LLM leerá en el siguiente ciclo de razonamiento.

---

## Funciones de enrutamiento (`src/agent/nodes/routing.py`)

### `route_after_reasoning(state) → str`

| Condición | Destino |
|---|---|
| Último mensaje es `AIMessage` con `tool_calls` | `"param_validation_node"` |
| Cualquier otro caso | `"END"` |

### `route_after_validation(state) → str`

| Condición | Destino |
|---|---|
| `pending_params` no vacío | `"END"` |
| `pending_params` vacío | `"tool_execution_node"` |

---

## Herramientas mock (`src/agent/tools/`)

Las tres herramientas simulan la futura API real de series temporales. Los resultados son **deterministas por entrada** (usan `random.Random(hash(args))`) para facilitar el debugging.

| Herramienta | Módulo | Simula |
|---|---|---|
| `detect_drift_kolmogorov_smirnov` | `mock_drift.py` | Test KS: devuelve `ks_statistic`, `p_value`, `drift_detected` |
| `generate_synthetic_series` | `mock_synthetic.py` | Genera serie: devuelve `series_id`, `summary_stats`, `file_path_generated` |
| `augment_data_linear_relation` | `mock_augment.py` | Aumenta CSV: devuelve `augmented_file_path`, `rows_generated`, `formula` |

La lista `AGENT_TOOLS` exportada desde `src/agent/tools/__init__.py` es la única fuente de herramientas para el grafo. Para añadir una herramienta nueva basta con:
1. Crear el módulo con `@tool`.
2. Importarlo y añadirlo a `AGENT_TOOLS`.
3. Añadir su entrada a `TOOL_REQUIRED_PARAMS` si tiene parámetros obligatorios.

---

## Persistencia de conversaciones

Se usa `MemorySaver` de LangGraph como checkpointer. Cada sesión de Streamlit tiene un `thread_id` único (UUID) almacenado en `st.session_state`. Esto garantiza que:

- Cada usuario tiene su propio hilo de conversación aislado.
- El historial persiste entre reruns de Streamlit dentro de la misma sesión del navegador.
- Al pulsar "Limpiar conversación" se genera un nuevo `thread_id`, abandonando el checkpoint anterior.
- `MemorySaver` es en memoria: el historial se pierde al reiniciar el proceso Streamlit.

> Para persistencia entre reinicios del servidor, sustituir `MemorySaver` por `SqliteSaver` o `PostgresSaver` de LangGraph (iteración futura).

---

## System prompt (`src/agent/prompts/system_prompts.py`)

`build_system_prompt(has_uploaded_file, uploaded_file_info)` ensambla el prompt a partir de cuatro bloques:

1. **`_ROLE_BLOCK`** — rol del agente, idioma español forzado.
2. **`_BEHAVIOR_BLOCK`** — instrucciones anti-alucinación y de petición de parámetros.
3. **`_TOOLS_BLOCK`** — descripción de cada herramienta y cuándo usarla.
4. **`_FILE_CONTEXT_TEMPLATE` / `_NO_FILE_BLOCK`** — info del CSV activo (o aviso de que no hay ninguno).

---

## Configuración del LLM (`src/config/llm_config.py`)

| Función | Caché | Descripción |
|---|---|---|
| `get_chat_ollama()` | `@lru_cache(maxsize=1)` | Cliente base, valida conexión con Ollama |
| `get_llm_with_tools(tools)` | Sin caché | Aplica `.bind_tools(tools)` sobre el cliente cacheado |

El modelo configurado en `.env` debe soportar **Tool Calling nativo**. Para este proyecto se usa `qwen2.5:7b`.

---

## Integración con Streamlit (`ui/app.py`)

```
session_state:
  ├── thread_id          UUID de la sesión → config del grafo
  ├── chat_history       Lista de dicts {role, content} → display únicamente
  └── uploaded_file_path Ruta al CSV activo → pasada al AgentState
```

El `chat_history` en `session_state` es solo para renderizar la UI. El historial real de mensajes vive en el checkpoint de `MemorySaver` indexado por `thread_id`. Ambas estructuras se sincronizan manualmente al añadir cada turno.

---

## Próximas integraciones planificadas

| Iteración | Cambio |
|---|---|
| RAG en el agente | Añadir `consultar_teoria_drift` a `AGENT_TOOLS` y popular `rag_context` antes de `reasoning_node` |
| Servidor MCP | Sustituir herramientas mock por clients MCP reales manteniendo la misma interfaz `@tool` |
| API real de series temporales | Reemplazar las funciones mock por llamadas HTTP reales; los docstrings y parámetros se mantienen |
| Persistencia entre sesiones | Sustituir `MemorySaver` por `SqliteSaver` con ruta configurable en `.env` |

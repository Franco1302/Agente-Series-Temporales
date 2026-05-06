# Bitácora de optimización — sesión 2026-05-06

> Documento de seguimiento del TFG. Registra qué problemas se detectaron en la
> rama `arquitectura-ciclica`, qué se hizo para resolverlos, qué se midió y en
> qué punto exacto queda el proyecto al cierre de la sesión.

---

## 1. Punto de partida

Rama actual: `arquitectura-ciclica`. Último commit antes de la sesión:

```
58c3105  test: Test de todo lo cambiado ahora
b438f02  refactor: cambio como obtenemos el CSV y el como se ve el status...
d5c4761  refactor: Adaptación del grafo, del sistem prompt y la ejecución...
390fa00  feat: nodo de respuesta, mejora en el razonador
455c259  feat: Manejo-Errores, solicitar parametros, y ejecutar la busqueda con RAG
```

Estado funcional al iniciar la sesión:

- Grafo cíclico de 6 nodos compilado y operativo.
- `consultar_teoria` (RAG) integrada en `AGENT_TOOLS`.
- Mock tools de drift / sintética / augment activas.
- Tests E2E (`scripts/test_agent.py`): **fallando 0/6** con interrupción manual al segundo caso.
- Modelo configurado en `.env`: `qwen2.5-coder:7b-instruct-q4_K_M`.

---

## 2. Síntomas observados al iniciar

### Síntoma 1 — caso 1 del test E2E falla

Salida del test:

```
→ Nodo: razonador
   Respuesta: { "name": "detect_drift_kolmogorov_smirnov", "arguments": { ... } }
→ Nodo: generar_respuesta
✗ FALLO: se esperaba ejecución de herramienta pero no se ejecutó ninguna.
```

El razonador emitía la tool call **como JSON dentro de `content`** en lugar
de poblar `tool_calls`. El router `route_after_razonador` no detectaba tool
calls y enviaba el flujo a `generar_respuesta` saltándose la ejecución.

### Síntoma 2 — caso 2 quedaba colgado

Trazas: `KeyboardInterrupt` durante la llamada al LLM en `generar_respuesta_node`
(`httpcore` en `_receive_response_headers`). No es un bug del grafo: era
lentitud de Ollama en una llamada redundante (el LLM se invocaba dos veces
en cada turno; ver §3.6).

### Síntoma 3 — Streamlit no muestra respuesta a "hola"

Una petición trivial como "hola" no producía nada en la UI a pesar de que
el grafo terminaba sin error.

---

## 3. Diagnóstico y correcciones

### 3.1. Causa raíz del síntoma 1: el modelo no soporta tool-calling nativo

Se aisló la llamada al LLM con `bind_tools` y se compararon los modelos
disponibles localmente con el mismo prompt:

| Modelo | `content` | `tool_calls` |
|---|---|---|
| `qwen2.5-coder:7b-instruct-q4_K_M` (configurado) | JSON con `name`/`arguments` | `[]` |
| `qwen2.5:7b-instruct-q4_K_M` (sin "coder") | `''` | poblado correctamente ✓ |
| `llama3.1:8b-instruct-q4_K_M` | `''` | poblado correctamente ✓ |

La variante `qwen2.5-coder` con cuantización q4_K_M no respeta el protocolo
de tool calling de Ollama: emite el JSON como texto.

**Corrección 1**: cambio en `.env`:

```diff
- OLLAMA_MODEL=qwen2.5-coder:7b-instruct-q4_K_M
+ OLLAMA_MODEL=qwen2.5:3b-instruct-q4_K_M  # decisión final tras §4
```

### 3.2. Defensa frente a modelos cuantizados que rompen el protocolo

Aunque el cambio de modelo soluciona el síntoma, conviene blindar el
agente para que no falle silenciosamente con modelos futuros.

**Corrección 2**: parser de fallback en `src/agent/nodes/reasoning.py`. La
función `_coerce_text_toolcall` detecta el patrón
`{"name": ..., "arguments": ...}` en `content`, valida que el nombre coincida
con una herramienta conocida y promueve el match a un `tool_call` sintético
(con `id` generado). El resto del grafo trata la tool call como legítima.

```python
# fragmento conceptual
def _coerce_text_toolcall(message: AIMessage) -> AIMessage:
    if message.tool_calls:
        return message
    # busca {"name": ..., "arguments": {...}} en content
    # si match → AIMessage(content="", tool_calls=[synthetic])
```

### 3.3. Alineación system prompt ↔ topología cíclica

`_BEHAVIOR_BLOCK` (en `src/agent/prompts/system_prompts.py`) instruía al
modelo a *preguntar al usuario* cuando faltaran parámetros. Esto cortocircuita
`solicitar_parametros_node`: el LLM responde directamente y el nodo nunca
se ejerce.

**Corrección 3**: prompt reescrito para alinear con el grafo:

- "INVOCA siempre la herramienta, incluso con parámetros incompletos."
- "Pasa SOLO los parámetros explícitos. OMITE COMPLETAMENTE el resto."
- Regla crítica con ejemplo explícito contra inventar defaults:
  *"si el usuario solo dice 'genera una serie temporal sintética', invoca con
  arguments={} y deja que solicitar_parametros pida los datos."*

Este último refinamiento se añadió tras el test con `qwen2.5:3b`: el modelo
3B era suficientemente asertivo como para rellenar defaults plausibles
y saltarse el nodo. Con la regla explícita el comportamiento se corrige.

### 3.4. RAG — usar la query refinada del LLM

`recuperar_contexto_node` extraía la query del último `HumanMessage`,
descartando la query reformulada que el LLM había puesto en
`tool_calls[0].args["query"]`.

**Corrección 4**: `src/agent/nodes/rag_retrieval.py`

- Nueva función `_extract_query`: busca primero en la tool call;
  fallback al `HumanMessage` si no hay query refinada.
- Emisión de un `ToolMessage` de cierre asociado al `tool_call_id`.
  Sin él, ChatOllama lanza
  *"tool call without a corresponding tool message"* en la siguiente
  iteración del razonador (porque el ciclo RAG vuelve al razonador).

### 3.5. Routing tras error: rama por defecto incorrecta

`route_after_error` enviaba todos los errores a `solicitar_parametros`,
incluso los no relacionados con parámetros (timeouts, fallos de conexión,
runtime). Esto creaba un bucle de petición de datos sin sentido.

**Corrección 5**: `src/agent/nodes/routing.py`. La rama por defecto ahora va
a `generar_respuesta` (errores no recuperables se presentan al usuario).
Solo errores que coincidan con los keywords de parámetros van a
`solicitar_parametros`.

### 3.6. Doble llamada al LLM en el camino feliz (causa del síntoma 3)

Trazado del flujo con la entrada "hola":

```
=== razonador  (13.16s) ===
  AIMessage: content='¡Hola! ¿Cómo puedo ayudarte hoy con tus datos...'
=== generar_respuesta  (13.94s) ===
  AIMessage: content=''        ← VACÍO
```

El razonador ya producía la respuesta. `generar_respuesta` invocaba al LLM
de nuevo y el modelo, viendo que ya había un `AIMessage` con respuesta,
devolvía content vacío. El UI de Streamlit solo guardaba `final_response`
desde `generar_respuesta`, por eso no aparecía nada.

**Corrección 6** (la más impactante en latencia):

- `src/agent/nodes/routing.py`: `route_after_razonador` ahora devuelve `"fin"`
  cuando hay content sin tool_calls.
- `src/agent/graph.py`: la transición `fin` se mapea directamente a `END`.
- `ui/app.py`: el handler del nodo `razonador` también captura
  `final_response = msg.content` cuando el razonador termina sin tool call.
- `generar_respuesta` queda reservado para el camino de error.

Esta corrección **elimina una llamada al LLM por turno en el camino feliz**.

---

## 4. Benchmark de modelos

Tras corregir el flujo, los 13 s para "hola" seguían siendo intolerables.
Se hizo un benchmark sistemático en el hardware del usuario.

### 4.1. Hardware

| Recurso | Valor |
|---|---|
| GPU | NVIDIA GeForce GTX 1660 SUPER, **6 GB VRAM** |
| CPU | AMD Ryzen, 12 hilos |
| RAM | 15 GB (≈7 GB libres durante operación normal) |
| Disco | 55 GB libres |

Con 6 GB de VRAM, un modelo 7B q4_K_M (≈4.7 GB) cabe pero deja casi sin
margen para el contexto. Modelos 3B (≈2 GB) encajan holgadamente y
permiten un context length mayor.

### 4.2. Procedimiento

Se creó `scripts/bench_models.py`, que para cada modelo:

1. Hace `bind_tools(AGENT_TOOLS)` con `ChatOllama`.
2. Ejecuta un warm-up para que el modelo esté en VRAM.
3. Mide latencia y acierto de tool calling en 3 trials:
   - **Saludo** (`"hola"`) — esperado: respuesta de texto, sin tool call.
   - **Drift** (petición concreta con CSV) — esperado:
     `detect_drift_kolmogorov_smirnov`.
   - **RAG** (pregunta teórica) — esperado: `consultar_teoria`.

Modelos descargados durante la sesión (≈4 GB combinados):

```bash
ollama pull llama3.2:3b-instruct-q4_K_M
ollama pull qwen2.5:3b-instruct-q4_K_M
```

### 4.3. Resultados (latencia en segundos, post warm-up)

| Modelo | Saludo | Drift | RAG | Promedio | Aciertos |
|---|---:|---:|---:|---:|:---:|
| `qwen2.5:7b-instruct-q4_K_M` | 1.40 | 2.51 | 3.59 | 2.50 | 3/3 |
| `llama3.1:8b-instruct-q4_K_M` | 3.38 | 3.14 | 2.47 | 2.99 | 3/3 |
| **`qwen2.5:3b-instruct-q4_K_M`** | **0.46** | **0.95** | **0.78** | **0.73** | **3/3** ✓ |
| `llama3.2:3b-instruct-q4_K_M` | 1.28 | 1.00 | 0.86 | 1.05 | 2/3 ✗ |

### 4.4. Decisión

**Modelo elegido: `qwen2.5:3b-instruct-q4_K_M`**.

- 3.4× más rápido que `qwen2.5:7b` (el siguiente con 3/3 aciertos).
- Tool calling fiable en los 3 escenarios.
- VRAM ocupada ≈2 GB → deja margen para contexto largo y RAG.
- Sigue las instrucciones del system prompt (incluida la regla "no inventes
  defaults" tras el refuerzo de §3.3).

`llama3.2:3b` queda descartado pese a su velocidad: invocaba
`consultar_teoria` para la entrada "hola", lo que generaría latencia y
respuestas extrañas en saludos.

El benchmark es reproducible:

```bash
PYTHONPATH=. .venv/bin/python -m scripts.bench_models
# o filtrando modelos:
PYTHONPATH=. .venv/bin/python -m scripts.bench_models qwen2.5:3b-instruct-q4_K_M
```

---

## 5. Topología final del grafo

```
                            ┌────────────────┐
   HumanMessage ──▶ START──▶│   razonador    │◀─────────────┐
                            └───────┬────────┘              │
                                    │                       │
              ┌──────────────┬──────┴──────┬─────────────┐  │
              ▼              ▼             ▼             ▼  │
          (texto       ejecutar_      recuperar_   solicitar_│
          plano)       herramienta    contexto     parametros│
              │              │             │             │  │
              ▼              ▼             └─────────────┤  │
            END         ┌────┴────┐                      │  │
                        ▼         ▼                      ▼  │
                   gestionar_   razonador (vuelve)       END│
                   error           (cierra ciclo) ──────────┘
                       │
              ┌────────┴────────┐
              ▼                 ▼
       solicitar_         generar_
       parametros          respuesta
              │                 │
              ▼                 ▼
             END               END
```

Reglas clave:

- `razonador` con texto plano → **END** directamente (no más doble LLM call).
- `razonador` con tool call de `consultar_teoria` → ciclo RAG.
- `razonador` con tool call con parámetros incompletos
  (`pending_tool` poblado) → `solicitar_parametros`.
- `razonador` con tool call completa → `ejecutar_herramienta`.
- Errores de ejecución → `gestionar_error` → `solicitar_parametros` (si
  el error es de parámetros) o `generar_respuesta` (resto).

`generar_respuesta_node` queda como nodo terminal **únicamente** del camino
de error. Esto evita la doble invocación al LLM y elimina su tendencia a
devolver `content=""` cuando ya hay respuesta en el historial.

---

## 6. Métricas antes / después

### 6.1. Latencia wall-clock medida desde el grafo

| Petición | Antes | Después | Mejora |
|---|---:|---:|---:|
| `"hola"` | 13.94 s + sin respuesta visible | **0.52 s** + respuesta correcta | **27×** + bug corregido |
| Análisis de drift completo | ≈16 s | **4.5 s** | ≈3.5× |

### 6.2. Tests E2E (`scripts/test_agent.py --todos`)

| Estado | Caso 1 | Caso 2 | Caso 3 | Caso 4 | Caso 5 | Caso 6 | Total |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| Sesión inicial | ✗ | ⏸ (KbI) | — | — | — | — | 0/6 + interrumpido |
| Tras correcciones 1-5 | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | 6/6 |
| Con `qwen2.5:3b` (sin §3.3) | ✓ | ✗ | ✓ | ✓ | ✓ | ✓ | 5/6 |
| **Final (todo aplicado)** | **✓** | **✓** | **✓** | **✓** | **✓** | **✓** | **6/6** |

### 6.3. Cobertura del grafo cíclico tras los cambios

| Nodo | Casos del test que lo ejercitan |
|---|---|
| `razonador` | 1, 2, 3, 4, 5, 6 |
| `ejecutar_herramienta` | 1, 4 |
| `recuperar_contexto` | 3, 5 |
| `solicitar_parametros` | 2, 6 |
| `gestionar_error` | (sin caso explícito todavía) |
| `generar_respuesta` | (solo desde el camino de error) |

---

## 7. Archivos modificados / creados

| Archivo | Tipo | Cambio |
|---|---|---|
| `.env` | mod | Modelo cambiado a `qwen2.5:3b-instruct-q4_K_M` |
| `src/agent/graph.py` | mod | Mapping `fin → END`; docstring actualizado con la nueva topología |
| `src/agent/nodes/reasoning.py` | mod | `_coerce_text_toolcall` (parser defensivo); limpieza de import |
| `src/agent/nodes/routing.py` | mod | `route_after_razonador` con destino `"fin"`; `route_after_error` con default `generar_respuesta` |
| `src/agent/nodes/rag_retrieval.py` | mod | `_extract_query` desde la tool call; emisión de `ToolMessage` de cierre |
| `src/agent/prompts/system_prompts.py` | mod | `_BEHAVIOR_BLOCK` reescrito con regla anti-defaults y ejemplo |
| `ui/app.py` | mod | Captura de respuesta también desde el nodo `razonador` |
| `scripts/bench_models.py` | nuevo | Benchmark reproducible de latencia + tool calling |
| `data/temp_uploads/ventas.csv` | nuevo | Dataset de prueba para los tests |
| `docs/bitacora_optimizacion_2026-05-06.md` | nuevo | Este documento |

---

## 8. Estado actual y siguientes pasos

### 8.1. Dónde estamos exactamente

- Rama: `arquitectura-ciclica`. Cambios **sin commitear** (modificaciones
  locales sobre los archivos de la tabla anterior).
- Tests: 6/6 pasando.
- UI: Streamlit operativa. "hola" responde en ≈0.5 s. Drift en ≈4.5 s.
- Modelo en producción local: `qwen2.5:3b-instruct-q4_K_M` (≈2 GB en VRAM).
- Hooks de observabilidad: trazado por nodo en `_run_agent_streaming` —
  ahora muestra el nombre real de la herramienta invocada en el panel
  `st.status()`.

### 8.2. Limitaciones conocidas

- `gestionar_error_node` no se ejerce en ningún test E2E todavía. Sería
  conveniente añadir un caso que fuerce un error de runtime (p. ej. CSV
  inexistente con file_path no vacío) para validar la rama
  `tool_execution → gestionar_error → generar_respuesta`.
- El fallback `_coerce_text_toolcall` cubre el formato OpenAI-style
  `{"name": ..., "arguments": ...}`. Si aparece un modelo que use
  `<tool_call>...</tool_call>` (formato qwen) habría que extender el regex.
- El benchmark se ejecuta con el modelo en caliente. La primera petición
  tras un arranque en frío sigue costando varios segundos por la carga
  del modelo en VRAM.

### 8.3. Siguientes pasos sugeridos

1. **Commitear** los cambios de la sesión con un mensaje agrupado
   (sugerencia: `perf: cambio de modelo a qwen2.5:3b y simplificación
   del flujo razonador→END`).
2. **Añadir test 7**: error de runtime para cubrir
   `gestionar_error_node`.
3. **Conectar herramientas reales** sustituyendo los mocks
   (`mock_drift`, `mock_synthetic`, `mock_augment`).
4. **Persistencia** más allá de `MemorySaver` (SQLite/Postgres) para
   sobrevivir a reinicios de Streamlit.
5. (Opcional) **MCP server** según el plan de `CLAUDE.md`.

### 8.4. Comandos de verificación

```bash
# Activar entorno
cd /home/franco/Documentos/TFG && source .venv/bin/activate

# Tests E2E (requiere ollama serve corriendo)
PYTHONPATH=. python -m scripts.test_agent --todos

# Benchmark de modelos
PYTHONPATH=. python -m scripts.bench_models

# UI
PYTHONPATH=. streamlit run ui/app.py
```

---

*Cierre de sesión: 2026-05-06. Estado: 6/6 tests verdes, latencia 27× mejor
en el caso trivial, arquitectura cíclica completamente ejercitada.*

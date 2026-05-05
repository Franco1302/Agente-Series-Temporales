# Setup del Entorno Local para MCP Tool Calling

Guía técnica para seleccionar, descargar y benchmarkear modelos Ollama orientados a tool calling sobre un servidor MCP (Model Context Protocol).

---

## Arquitectura propuesta

```
┌─────────────────────────────────────────────────────────┐
│                      Cliente (tú)                       │
│  ┌─────────────┐     ┌──────────────────────────────┐   │
│  │ Streamlit UI│────▶│       LangGraph Agent        │   │
│  └─────────────┘     │  (ReAct: reason → act loop)  │   │
│                       └──────────────┬───────────────┘   │
│                                      │ Tool Call JSON     │
│                       ┌──────────────▼───────────────┐   │
│                       │    MCP Client (langchain-mcp) │   │
│                       └──────────────┬───────────────┘   │
└──────────────────────────────────────┼───────────────────┘
                                       │ stdio / HTTP SSE
               ┌───────────────────────▼───────────────────┐
               │            Servidor MCP                    │
               │  (expone las herramientas reales de la     │
               │   API: drift, synthetic data, augment...)  │
               └───────────────────────────────────────────┘
                                       ▲
               ┌───────────────────────┘
               │ Inferencia LLM local
┌──────────────▼───────────────────────────────────────────┐
│                    Ollama (localhost:11434)               │
│  Modelo elegido: qwen2.5:7b-instruct-q4_K_M              │
│  (o el ganador del benchmark)                            │
└──────────────────────────────────────────────────────────┘
```

**Flujo de una petición:**
1. El usuario escribe en la UI → LangGraph invoca el nodo `reasoning`.
2. El LLM (Ollama) razona y emite un tool call JSON.
3. El MCP Client traduce ese call al protocolo MCP y lo envía al servidor.
4. El servidor ejecuta la herramienta real y retorna el resultado.
5. El resultado vuelve al LLM como `ToolMessage` para formular la respuesta final.

---

## Modelos candidatos y justificación técnica

> **Nota:** Hermes3 fue reemplazado por Mistral 7B debido a compatibilidad directa con Ollama. Mistral es ampliamente disponible y demostrado en tool-calling con 6 GB VRAM.



| Modelo | Tag Ollama | VRAM Q4\_K\_M | Por qué para tool calling |
|---|---|---|---|
| **Qwen2.5 7B Instruct** | `qwen2.5:7b-instruct-q4_K_M` | ~4.5 GB | Mejor equilibrio razonamiento/tool-calling en su clase de tamaño. Consistente en JSON estructurado. |
| **Llama 3.1 8B Instruct** | `llama3.1:8b-instruct-q4_K_M` | ~4.9 GB | Function-calling nativo del entrenamiento de Meta. Amplia comunidad y soporte en LangChain. |
| **Qwen2.5-Coder 7B** | `qwen2.5-coder:7b-instruct-q4_K_M` | ~4.5 GB | Entrenado en código y datos estructurados → máxima fidelidad de formato JSON. |
| **Mistral 7B Instruct** | `mistral:7b-instruct-q4_K_M` | ~4.1 GB | Muy confiable en Ollama. Sólido en tool-calling y seguimiento preciso de instrucciones. Rápido. |

**Por qué Q4\_K\_M con 6 GB VRAM:**
- Q4\_K\_M ofrece la mejor relación calidad/tamaño de los esquemas de 4 bits.
- Los modelos de 7-8B con Q4\_K\_M quedan entre 4.5 y 5.0 GB, dejando ~1 GB de margen para el contexto de la conversación (KV cache).
- Esquemas más agresivos (Q3, Q2) degradan la precisión en JSON schemas. Esquemas mayores (Q5, Q8) superarían los 6 GB de VRAM disponibles.

---

## Inicialización del entorno virtual aislado

> **Regla de oro:** Este entorno virtual es **exclusivo** para el benchmark. Nunca actives ambos entornos a la vez.

```bash
# Desde la raíz del proyecto
python3 -m venv .venv_benchmark
source .venv_benchmark/bin/activate

# Verificar que estás en el entorno correcto
which python   # debe apuntar a .venv_benchmark/bin/python

# Instalar dependencias del benchmark
pip install -r requirements_benchmark.txt
```

Para desactivar el entorno del benchmark y volver al entorno principal:
```bash
deactivate
source .venv/bin/activate
```

---

## Descarga de modelos (`setup_models.sh`)

```bash
# Asegúrate de que Ollama está corriendo
ollama serve &   # o en una terminal separada

# Dar permisos de ejecución (solo la primera vez)
chmod +x setup_models.sh

# Descargar todos los modelos (~18 GB en total)
bash setup_models.sh

# Descargar solo un modelo concreto
bash setup_models.sh --solo qwen2.5:7b-instruct-q4_K_M
```

El script verifica que Ollama esté activo antes de empezar y muestra un resumen al finalizar.

---

## Ejecución del benchmark (`mcp_benchmark.py`)

```bash
# Entorno benchmark activo
source .venv_benchmark/bin/activate

# Ejecutar todos los modelos y casos (recomendado la primera vez)
python mcp_benchmark.py

# Solo algunos modelos
python mcp_benchmark.py --modelos qwen2.5:7b-instruct-q4_K_M llama3.1:8b-instruct-q4_K_M

# Solo algunos casos de prueba (1–4)
python mcp_benchmark.py --casos 1 3

# Cambiar nombre del CSV de salida
python mcp_benchmark.py --salida mi_benchmark.csv
```

El script imprime progreso en tiempo real y una tabla resumen al terminar.

---

## Interpretación del CSV (`resultados_benchmark.csv`)

| Columna | Qué significa | Umbral recomendado |
|---|---|---|
| `tiempo_s` | Latencia total de la llamada al LLM | < 8 s para uso interactivo |
| `tokens_por_s` | Velocidad de generación | > 15 t/s es fluido |
| `json_valido` | El modelo emitió un tool call (True/False) | Debe ser `True` siempre |
| `herramienta_correcta` | Eligió la herramienta adecuada para el contexto | Crítico: priorizar esto |
| `args_requeridos_presentes` | Todos los parámetros obligatorios incluidos | Crítico: priorizar esto |
| `precision_args_pct` | % de valores extraídos correctamente del texto | > 80% aceptable |

### Score global (0–100)

Fórmula ponderada por caso:

```
Score = herramienta_correcta × 40
      + args_requeridos_presentes × 30
      + precision_args_pct × 0.30
```

**Cómo elegir el modelo ganador:**

1. **Prioridad 1 — `herramienta_correcta` perfecta (4/4).** Un modelo que elige la herramienta equivocada es inútil sin importar su velocidad.
2. **Prioridad 2 — `args_requeridos_presentes` perfecta (4/4).** Herramienta correcta con parámetros incompletos genera llamadas MCP fallidas.
3. **Desempate — `precision_args_pct` y `tokens_por_s`.** Si dos modelos son igual de precisos, el más rápido ofrece mejor UX.
4. **Descarta si `json_valido` < 100%.** Un modelo que no genera tool calls en ningún caso no está soportado por Ollama para ese esquema.

**Ejemplo de lectura de resultados:**

```
╭─────────────────────────────────┬──────────────────┬─────────┬───────────────┬───────────────┬────────────────┬─────────────╮
│ Modelo                          │ Tiempo medio (s) │ t/s med │ Tool correcta │ Args presentes│ Precisión args │ Score global│
├─────────────────────────────────┼──────────────────┼─────────┼───────────────┼───────────────┼────────────────┼─────────────┤
│ qwen2.5:7b-instruct-q4_K_M      │ 3.21             │ 42.1    │ 4/4           │ 4/4           │ 91.3%          │ 97.4/100    │  ← GANADOR
│ llama3.1:8b-instruct-q4_K_M     │ 4.80             │ 28.5    │ 3/4           │ 3/4           │ 78.0%          │ 73.4/100    │
│ qwen2.5-coder:7b-instruct-q4_K_M│ 3.45             │ 39.2    │ 4/4           │ 3/4           │ 85.0%          │ 85.5/100    │
│ hermes3:8b-q4_K_M               │ 5.10             │ 25.3    │ 2/4           │ 2/4           │ 62.5%          │ 58.8/100    │
╰─────────────────────────────────┴──────────────────┴─────────┴───────────────┴───────────────┴────────────────┴─────────────╯
```

En este ejemplo: `qwen2.5:7b-instruct-q4_K_M` gana por precisión perfecta y velocidad.

---

## Variables de entorno relevantes (`.env`)

El benchmark usa directamente la API de Ollama en `localhost:11434`. No requiere modificar `.env`. Si tu Ollama corre en otro host, edita la constante `OLLAMA_HOST` al inicio de `mcp_benchmark.py`.

---

## Próximos pasos tras elegir el modelo

1. Actualiza `OLLAMA_MODEL` en tu `.env` con el modelo ganador.
2. Verifica que el agente LangGraph lo carga: `PYTHONPATH=. python -c "from src.config.llm_config import get_chat_ollama; print(get_chat_ollama())"`.
3. Integra el servidor MCP real añadiendo su URL/configuración al agente.
4. Re-ejecuta `python scripts/test_agent.py` para validar la integración end-to-end.

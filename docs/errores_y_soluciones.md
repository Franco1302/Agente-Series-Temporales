# Registro de errores y soluciones

## Cómo registrar errores nuevos

```text
Fecha: YYYY-MM-DD
Error: mensaje principal
Contexto: donde ocurrió
Causa raíz: por qué pasó
Solución aplicada: qué se hizo
Estado: resuelto | pendiente
```

---

## Errores fase 1 — Chat directo con Ollama

### 2026-03-29 — No module named src

- Contexto: ejecución de Streamlit desde la carpeta `ui/` o sin `PYTHONPATH`.
- Causa raíz: Python no encuentra el paquete `src` fuera de la raíz del proyecto.
- Solución aplicada:

```bash
cd /home/franco/Documentos/TFG
source .venv/bin/activate
PYTHONPATH=. streamlit run ui/app.py
```

- Estado: resuelto.

### 2026-03-29 — File does not exist: app.py

- Contexto: comando `streamlit run app.py` desde la raíz del repo.
- Causa raíz: el archivo de entrada está en `ui/app.py`, no en la raíz.
- Solución aplicada:

```bash
streamlit run ui/app.py
```

- Estado: resuelto.

### 2026-03-29 — File does not exist: streamlit_app.py

- Contexto: ejecución de `streamlit run` sin target.
- Causa raíz: Streamlit busca `streamlit_app.py` por defecto si no se pasa archivo.
- Solución aplicada:

```bash
PYTHONPATH=. streamlit run ui/app.py
```

- Estado: resuelto.

### 2026-03-29 — pip install requirements.txt

- Contexto: instalación de dependencias.
- Causa raíz: faltaba la bandera `-r`; pip intentó instalar un paquete llamado `requirements.txt`.
- Solución aplicada:

```bash
pip install -r requirements.txt
```

- Estado: resuelto.

### 2026-03-29 — Error de conexión con el LLM local

- Contexto: Streamlit mostraba fallo en invocación al LLM.
- Causa raíz: combinación de sesión previa sin reinicio y nombre de modelo no alineado con Ollama.
- Solución aplicada:
  - Ajustar `OLLAMA_MODEL` en `.env` a `llama3.1:latest`.
  - Reiniciar Streamlit después de cambiar `.env`.
  - Validar servicio con `curl` a `/api/tags` y prueba de generación.
- Estado: resuelto.

---

## Errores fase 2 — Migración a LangGraph

### 2026-04-21 — ModuleNotFoundError: No module named 'langgraph'

- Contexto: ejecución de cualquier módulo del agente tras añadir las nuevas dependencias a `requirements.txt`.
- Causa raíz: `langgraph`, `langchain-core` y `pandas` añadidos al fichero pero no instalados en el entorno virtual.
- Solución aplicada:

```bash
source .venv/bin/activate
pip install -r requirements.txt
```

- Estado: resuelto.

### 2026-04-21 — ModuleNotFoundError: No module named 'langchain_core' al usar @tool

- Contexto: importación de `src.agent.tools` antes de instalar las nuevas dependencias.
- Causa raíz: `langchain-core` no estaba instalado explícitamente (era transitivo de versiones anteriores).
  Ahora se fija la versión `>=0.3.0` en `requirements.txt`.
- Solución aplicada: igual que el error anterior — `pip install -r requirements.txt`.
- Estado: resuelto.

### 2026-04-21 — `python` no encontrado, usar `python3`

- Contexto: ejecución de comandos de verificación con `python`.
- Causa raíz: en esta distribución Linux el binario se llama `python3`, no `python`.
- Solución aplicada: usar siempre `.venv/bin/python3` o activar el venv y usar `python3`.

```bash
source .venv/bin/activate
python3 -m scripts.test_agent
```

- Estado: resuelto.

### 2026-04-21 — El agente no produce respuesta de texto (cadena vacía)

- Contexto: posible al usar modelos que no soportan Tool Calling nativo o con `OLLAMA_REQUEST_TIMEOUT` muy bajo.
- Causa raíz: el modelo no emite `content` en el `AIMessage` cuando solo genera `tool_calls`, o el timeout interrumpe la generación.
- Solución aplicada:
  - Verificar que `OLLAMA_MODEL=qwen2.5:7b` (soporta Tool Calling nativo).
  - Aumentar `OLLAMA_REQUEST_TIMEOUT` a `60` o más para modelos lentos.
  - Si el modelo no soporta tools, la UI muestra el aviso `(El agente no produjo una respuesta de texto.)`.
- Estado: resuelto con configuración correcta.

### 2026-04-21 — Bucle infinito de razonamiento (ciclo ReAct)

- Contexto: el agente entra en un ciclo `reasoning → tool_execution → reasoning` sin terminar.
- Causa raíz: el LLM sigue emitiendo `tool_calls` en cada iteración aunque ya tenga el resultado.
- Solución aplicada: `reasoning_node` comprueba `iteration_count >= 5` y fuerza `END` con mensaje de parada.
  El límite se controla con la constante `_MAX_ITERATIONS = 5` en `src/agent/nodes/reasoning.py`.
- Estado: resuelto.

### 2026-04-21 — Limpiar conversación no reinicia el contexto del agente

- Contexto: el botón "Limpiar conversación" borraba `chat_history` pero el grafo seguía recordando la conversación anterior.
- Causa raíz: `MemorySaver` guarda el estado por `thread_id`; limpiar `chat_history` no afecta al checkpoint del grafo.
- Solución aplicada: al limpiar, se genera un nuevo `uuid4()` para `thread_id` en `session_state`.
  El hilo antiguo queda abandonado en memoria (se pierde al reiniciar el proceso).
- Estado: resuelto.

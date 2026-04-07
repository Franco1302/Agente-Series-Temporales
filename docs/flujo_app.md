# Flujo detallado de ejecucion de la app

## Componentes involucrados

- ui/app.py: interfaz de chat y ciclo de interaccion en Streamlit.
- src/agent/simple_chat.py: puente entre historial de chat y mensajes del LLM.
- src/config/llm_config.py: carga de .env, validacion de parametros y cliente ChatOllama.

## Flujo de arranque

1. Se ejecuta Streamlit con el target ui/app.py.
2. Se importan generate_chat_response y load_ollama_settings.
3. Al importar llm_config.py se intenta cargar .env desde la raiz del proyecto.
4. Se construye la pagina y el sidebar con los parametros actuales de Ollama.

## Flujo por cada mensaje del usuario

1. El usuario escribe en st.chat_input(...).
2. Se agrega un turno al historial en session_state con role user.
3. La UI calcula una ventana deslizante de historial reciente (CHAT_MAX_CONTEXT_TURNS).
4. Los turnos que salen de la ventana se comprimen en un resumen incremental en session_state (chat_summary).
5. Se compone el system prompt base + resumen comprimido.
6. La UI llama a generate_chat_response(history_reciente, system_prompt_compuesto).
7. El agente transforma cada turno recibido en mensajes de LangChain: user -> HumanMessage, assistant -> AIMessage y system prompt -> SystemMessage (si existe).
8. Se valida que el ultimo mensaje sea de usuario antes de inferir.
9. Se solicita el cliente con get_chat_ollama().
10. get_chat_ollama() realiza lectura y validacion de OLLAMA_BASE_URL, OLLAMA_MODEL, OLLAMA_TEMPERATURE, OLLAMA_REQUEST_TIMEOUT, healthcheck a GET /api/tags y creacion de ChatOllama.
11. Se ejecuta llm.invoke(messages).
12. La respuesta se normaliza a string.
13. La UI muestra el contenido y lo guarda en historial como role assistant.

## Manejo de errores

- Si falla la carga de variables o el healthcheck, se propaga un error controlado.
- Si falla la invocacion del modelo, simple_chat lanza RuntimeError.
- La UI captura excepciones y muestra un mensaje amigable en Streamlit.

## Nota sobre cache

get_chat_ollama() esta cacheado con lru_cache(maxsize=1). Esto evita crear el cliente en cada turno y mejora estabilidad/performance.

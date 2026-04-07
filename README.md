# Orquestador IA para Data Drift (Fase 1)

Base escalable para un futuro sistema IA con orquestador, RAG y API. En esta fase se entrega un chat local en Streamlit conectado a Ollama.

## Que incluye esta fase

- Estructura modular y limpia del repositorio.
- Interfaz de chat en Streamlit.
- Puente de chat desacoplado para invocacion de LLM local.
- Capa de configuracion de entorno para Ollama.
- Base lista para evolucionar a LangGraph y RAG.

## Arquitectura actual

- `ui/`: interfaz de usuario en Streamlit.
- `src/agent/`: logica del chat y transformacion de historial.
- `src/config/`: carga y validacion de variables de entorno, cliente Ollama.
- `data/`: carpetas reservadas para conocimiento y uploads temporales.

## Requisitos previos

- Python 3.10 o superior.
- Entorno virtual (`.venv`) creado.
- Ollama instalado.
- Modelo descargado en Ollama.

Comprobacion rapida:

```bash
python3 --version
ollama --version
```

## Configuracion inicial

Ejecuta estos comandos desde la raiz del proyecto (`TFG/`):

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

Variables esperadas:

- `OLLAMA_BASE_URL`: URL de Ollama local. Valor recomendado: `http://localhost:11434`
- `OLLAMA_MODEL`: nombre exacto del modelo instalado en Ollama. Recomendado: `llama3.1:latest`
- `OLLAMA_TEMPERATURE`: temperatura de generacion (`0.0` a `2.0`).
- `OLLAMA_REQUEST_TIMEOUT`: timeout en segundos para chequeo de conexion.
- `CHAT_SYSTEM_PROMPT`: prompt de sistema por defecto para el asistente.
- `CHAT_MAX_CONTEXT_TURNS`: cantidad de turnos recientes enviados al LLM en cada inferencia.
- `CHAT_SUMMARY_MAX_CHARS`: tamano maximo del resumen incremental de turnos antiguos.

Ejemplo:

```env
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=llama3.1:latest
OLLAMA_TEMPERATURE=0.2
OLLAMA_REQUEST_TIMEOUT=8
CHAT_SYSTEM_PROMPT=Eres un asistente de IA util especializado en analisis de data drift.
CHAT_MAX_CONTEXT_TURNS=8
CHAT_SUMMARY_MAX_CHARS=1400
```

## Iniciar la app paso a paso

Usa dos terminales para evitar confusiones.

### Terminal A: Ollama

```bash
# Si el modelo no existe aun
ollama pull llama3.1:latest

# Levantar servicio local
ollama serve
```

### Terminal B: Streamlit

```bash
cd /ruta/a/TFG
source .venv/bin/activate

# Importante: incluir la raiz del proyecto en PYTHONPATH
PYTHONPATH=. streamlit run ui/app.py
```

Al arrancar, Streamlit mostrara una URL local, normalmente:

```text
http://localhost:8501
```

## Documentacion en Docs

Para mantener este README mas conciso, el detalle operativo se mueve a la carpeta Docs:

- Flujo detallado de la app: [Docs/flujo_app.md](Docs/flujo_app.md)
- Registro de errores y soluciones: [Docs/errores_y_soluciones.md](Docs/errores_y_soluciones.md)

Cada error nuevo que aparezca durante desarrollo debe registrarse en Docs/errores_y_soluciones.md.

## Estructura del repositorio

```text
TFG/
├── Docs/
│   ├── errores_y_soluciones.md
│   └── flujo_app.md
├── .env.example
├── README.md
├── requirements.txt
├── data/
│   ├── knowledge_base/
│   └── temp_uploads/
├── src/
│   ├── agent/
│   │   └── simple_chat.py
│   └── config/
│       └── llm_config.py
└── ui/
   └── app.py
```

# Orquestador IA para Data Drift (Fase 1)

Base escalable para un futuro sistema IA que incluirá un orquestador con LangGraph, RAG y una API en Docker para análisis de data drift.

Esta fase entrega:
- Estructura modular del repositorio.
- Interfaz de chat en Streamlit.
- Integración aislada con LLM local mediante Ollama.
- Buenas prácticas de versionado listas para GitHub desde el inicio.

## Principios de Arquitectura

- Separación de responsabilidades:
  - Capa de interfaz en `ui/`.
  - Lógica de agente y LLM en `src/`.
- Aislamiento del entorno con `.venv`.
- Configuración mediante variables de entorno en `.env`.
- Exclusión de artefactos de datos en Git para mantener un repositorio liviano.

## Prerrequisitos

- Python 3.10 o superior.
- Git.
- Ollama instalado y en ejecución.

Comprobación rápida de Ollama:

```bash
ollama --version
ollama serve
ollama pull llama3.1:8b
```

## Estructura del Repositorio

```text
TFG/
├── .venv/
├── .env.example
├── .gitignore
├── README.md
├── requirements.txt
├── data/
│   ├── knowledge_base/
│   │   └── .gitkeep
│   └── temp_uploads/
│       └── .gitkeep
├── src/
│   ├── __init__.py
│   ├── agent/
│   │   ├── __init__.py
│   │   └── simple_chat.py
│   └── config/
│       ├── __init__.py
│       └── llm_config.py
└── ui/
    └── app.py
```

## Configuración Local

```bash
# 1) Entrar al directorio del proyecto
cd TFG

# 2) Inicializar Git
git init

# 3) Crear y activar entorno virtual
python3 -m venv .venv
source .venv/bin/activate

# 4) Instalar dependencias
python -m pip install --upgrade pip
pip install -r requirements.txt

# 5) Configurar variables de entorno
cp .env.example .env
```

## Ejecutar el Chat en Streamlit

```bash
source .venv/bin/activate
streamlit run ui/app.py
```

## Variables de Entorno

Usa `.env.example` como plantilla.

- `OLLAMA_BASE_URL`: URL del servidor Ollama (por defecto `http://localhost:11434`).
- `OLLAMA_MODEL`: nombre del modelo disponible en Ollama (ejemplo `llama3.1:8b`).
- `OLLAMA_TEMPERATURE`: temperatura de generación.
- `OLLAMA_REQUEST_TIMEOUT`: tiempo máximo para verificar conectividad.
- `CHAT_SYSTEM_PROMPT`: instrucción de sistema por defecto para el chat.

## Flujo Git para Fase 1

```bash
# Asegura que estás en la raíz del proyecto y con el venv activo
cd TFG

# Crear rama de trabajo para esta fase (en español)
git checkout -b fase1/interfaz-chat

# Añadir y confirmar estructura base
git add .
git commit -m "chore(fase1): base de interfaz streamlit y puente de chat con ollama"

# Opcional: definir main y conectar remoto
git branch -M main
git remote add origin <URL_DE_TU_REPO_GITHUB>

# Subir rama de trabajo
git push -u origin fase1/interfaz-chat
```

Sugerencia de estrategia de merge:

```bash
# Tras abrir y aprobar el PR hacia main
git checkout main
git pull origin main
```

## Alcance de la Fase

Esta fase no implementa todavía:
- Pipeline RAG.
- Orquestación con LangGraph.
- Capa de API ni imagen Docker.

Estos componentes se incorporarán en siguientes fases sobre esta base.

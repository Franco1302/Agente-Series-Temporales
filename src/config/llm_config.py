"""Utilidades de configuración para un modelo de chat local con Ollama."""

from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import urlopen

from dotenv import load_dotenv
from langchain_ollama import ChatOllama

PROJECT_ROOT = Path(__file__).resolve().parents[2]
ENV_FILE = PROJECT_ROOT / ".env"

if ENV_FILE.exists():
    load_dotenv(dotenv_path=ENV_FILE, override=False)
else:
    load_dotenv(override=False)


@dataclass(frozen=True)
class OllamaSettings:
    """Parámetros de ejecución usados para crear un cliente ChatOllama."""

    base_url: str
    model: str
    temperature: float
    request_timeout: float
    num_ctx: int


def _read_required_env(variable_name: str, default_value: str | None = None) -> str:
    """Lee una variable de entorno requerida y valida que no esté vacía."""
    value = os.getenv(variable_name, default_value)
    if value is None or not value.strip():
        raise ValueError(
            f"Falta la variable de entorno '{variable_name}'. Defínela en tu archivo .env."
        )
    return value.strip()


def _read_float_env(variable_name: str, default_value: float) -> float:
    """Lee y convierte a float una variable de entorno."""
    raw_value = os.getenv(variable_name)
    if raw_value is None or not raw_value.strip():
        return default_value

    try:
        return float(raw_value)
    except ValueError as exc:
        raise ValueError(
            f"La variable de entorno '{variable_name}' debe ser un float válido."
        ) from exc


def load_ollama_settings() -> OllamaSettings:
    """Carga y valida la configuración relacionada con Ollama desde el entorno."""
    base_url = _read_required_env("OLLAMA_BASE_URL", "http://localhost:11434").rstrip("/")
    model = _read_required_env("OLLAMA_MODEL", "llama3.1")
    temperature = _read_float_env("OLLAMA_TEMPERATURE", 0.2)
    request_timeout = _read_float_env("OLLAMA_REQUEST_TIMEOUT", 8.0)
    # El default de Ollama (2048) es insuficiente para nuestro system prompt (~5k chars)
    # + esquemas de las 9 tools. Con num_ctx pequeño el modelo trunca y devuelve vacío.
    num_ctx_raw = os.getenv("OLLAMA_NUM_CTX")
    num_ctx = int(num_ctx_raw) if num_ctx_raw and num_ctx_raw.strip().isdigit() else 8192

    if not 0.0 <= temperature <= 2.0:
        raise ValueError("OLLAMA_TEMPERATURE debe estar entre 0.0 y 2.0.")

    if request_timeout <= 0:
        raise ValueError("OLLAMA_REQUEST_TIMEOUT debe ser mayor que 0.")

    if num_ctx <= 0:
        raise ValueError("OLLAMA_NUM_CTX debe ser un entero positivo.")

    return OllamaSettings(
        base_url=base_url,
        model=model,
        temperature=temperature,
        request_timeout=request_timeout,
        num_ctx=num_ctx,
    )


def _check_ollama_connection(base_url: str, timeout: float) -> None:
    """Ejecuta una verificación ligera de conectividad contra Ollama local."""
    healthcheck_url = f"{base_url}/api/tags"
    try:
        with urlopen(healthcheck_url, timeout=timeout) as response:
            status_code = getattr(response, "status", 200)
            if status_code >= 400:
                raise ConnectionError(
                    f"La verificación de Ollama falló con código de estado {status_code}."
                )
    except (HTTPError, URLError, TimeoutError) as exc:
        raise ConnectionError(
            "No se pudo conectar con Ollama. Verifica que Ollama esté en ejecución "
            "y que OLLAMA_BASE_URL sea correcto."
        ) from exc

# Cachea el cliente ChatOllama para evitar recrearlo en cada interacción, mejorando la eficiencia.
@lru_cache(maxsize=1)
def get_chat_ollama() -> ChatOllama:
    settings = load_ollama_settings()

    try:
        _check_ollama_connection(settings.base_url, settings.request_timeout)
        return ChatOllama(
            model=settings.model,
            base_url=settings.base_url,
            temperature=settings.temperature,
            num_ctx=settings.num_ctx,
        )
    except Exception as exc:
        raise RuntimeError(
            "No se pudo inicializar ChatOllama. Revisa el servicio de Ollama y los valores de .env."
        ) from exc


def get_llm_with_tools(tools: list) -> ChatOllama:
    """Devuelve un ChatOllama con las herramientas enlazadas para Tool Calling.

    A diferencia de `get_chat_ollama`, esta función no se cachea porque la lista
    de herramientas puede variar entre llamadas. El cliente base sí se reutiliza
    del caché existente, por lo que no se vuelve a verificar la conexión.

    Args:
        tools: Lista de herramientas LangChain decoradas con @tool.

    Returns:
        ChatOllama con `.bind_tools(tools)` aplicado, listo para usar en nodos LangGraph.

    Raises:
        RuntimeError: Si no se puede inicializar el cliente base de Ollama.
    """
    base_llm = get_chat_ollama()
    return base_llm.bind_tools(tools)

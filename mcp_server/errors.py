"""Traducción de excepciones a strings amigables para el LLM."""

from __future__ import annotations

import httpx


def translate_exception(exc: Exception, tool_name: str) -> str:
    """Devuelve un string que el agente ReAct puede interpretar para reintentar."""
    if isinstance(exc, httpx.HTTPStatusError):
        try:
            detail = exc.response.json().get("detail", exc.response.text)
        except Exception:
            detail = exc.response.text
        return f"Error de la API ({exc.response.status_code}) en {tool_name}: {detail}"
    if isinstance(exc, httpx.TimeoutException):
        return (
            f"Timeout en {tool_name}: la operación excedió el tiempo de espera. "
            "Prueba con un dataset más pequeño o un horizonte menor."
        )
    if isinstance(exc, httpx.ConnectError):
        return (
            f"No fue posible conectar con la API analítica en {tool_name}. "
            "Verifica que el contenedor Docker esté arrancado en el puerto 8017."
        )
    if isinstance(exc, FileNotFoundError):
        return f"Fichero no encontrado en {tool_name}: {exc}. Sube el CSV antes de continuar."
    if isinstance(exc, ValueError):
        return f"Parámetro inválido en {tool_name}: {exc}"
    return f"Error inesperado en {tool_name}: {exc.__class__.__name__}: {exc}"

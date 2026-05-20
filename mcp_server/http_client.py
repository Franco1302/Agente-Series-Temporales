"""Cliente httpx asíncrono compartido entre las tools."""

from __future__ import annotations

from contextlib import asynccontextmanager

import httpx

from mcp_server.config import ServerSettings
# IMPORTANTE: Importamos los hooks analíticos de nuestro módulo de observabilidad
from mcp_server.observability.http_hooks import record_request_start, record_response_end


@asynccontextmanager
async def get_client(settings: ServerSettings):
    """Devuelve un AsyncClient configurado con timeouts, base_url y hooks analíticos."""
    timeout = httpx.Timeout(settings.request_timeout, connect=settings.connect_timeout)
    
    # Configuramos el diccionario de hooks analíticos de httpx
    hooks = {
        "request": [record_request_start],
        "response": [record_response_end]
    }
    
    # Pasamos 'event_hooks=hooks' al instanciar el cliente
    async with httpx.AsyncClient(
        base_url=settings.api_url, 
        timeout=timeout,
        event_hooks=hooks  # <-- Añadimos esta línea
    ) as client:
        yield client
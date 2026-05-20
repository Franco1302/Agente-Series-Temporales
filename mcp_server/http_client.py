"""Cliente httpx asíncrono compartido entre las tools."""

from __future__ import annotations

from contextlib import asynccontextmanager

import httpx

from mcp_server.config import ServerSettings

from mcp_server.observability.http_hooks import record_request_start, record_response_end


@asynccontextmanager
async def get_client(settings: ServerSettings):
    """Devuelve un AsyncClient configurado con timeouts y base_url."""
    timeout = httpx.Timeout(settings.request_timeout, connect=settings.connect_timeout)
    async with httpx.AsyncClient(base_url=settings.api_url, timeout=timeout, event_hooks={
        "request": [record_request_start],
        "response": [record_response_end]
    }) as client:
        yield client

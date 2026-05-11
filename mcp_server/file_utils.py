"""Utilidades de manejo de ficheros: guardado de CSV/PNG con naming determinista."""

from __future__ import annotations

import hashlib
from pathlib import Path

import httpx


def deterministic_filename(prefix: str, *parts: str, ext: str) -> str:
    """Construye un nombre estable basado en hash de los argumentos.

    Permite que dos invocaciones idénticas reusen el mismo fichero (caching natural).
    """
    digest = hashlib.sha1("|".join(parts).encode("utf-8")).hexdigest()[:8]
    return f"{prefix}_{digest}.{ext}"


def save_streaming_response(response: httpx.Response, target: Path) -> Path:
    """Guarda el cuerpo de una respuesta httpx en un fichero local."""
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(response.content)
    return target


def open_csv_for_upload(file_path: str) -> tuple[str, bytes, str]:
    """Lee un CSV local y lo prepara para enviarlo como multipart."""
    p = Path(file_path).resolve()
    if not p.exists():
        raise FileNotFoundError(f"No se encontró el fichero: {p}")
    if p.suffix.lower() != ".csv":
        raise ValueError(f"Se esperaba un CSV, recibido: {p.suffix}")
    return (p.name, p.read_bytes(), "text/csv")

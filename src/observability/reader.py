"""Helper analítico de streaming para alimentar la UI de Streamlit de forma eficiente."""

from __future__ import annotations

import json
from collections import deque
from pathlib import Path
from typing import Any


def read_trace_lines(
    log_path: str | Path,
    trace_id: str | None = None,
    thread_id: str | None = None,
    max_events: int = 200,
) -> list[dict[str, Any]]:
    """Lee el archivo JSONL filtrando línea a línea mediante streaming veloz.
    
    Aplica una comprobación por subcadena antes de ejecutar json.loads para
    no penalizar el rendimiento de Streamlit cuando el log crezca.
    """
    events: list[dict[str, Any]] = []
    path = Path(log_path)
    if not path.exists():
        return events

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            # Fast-path por substring: si el token no está en el texto plano, saltamos
            if trace_id and trace_id not in line:
                continue
            if thread_id and thread_id not in line:
                continue
                
            try:
                event_data = json.loads(line)
                events.append(event_data)
                if len(events) >= max_events:
                    break
            except json.JSONDecodeError:
                continue
                
    return events


def read_recent_thread_lines(
    log_path: str | Path,
    thread_id: str,
    tail_lines: int = 50,
) -> list[dict[str, Any]]:
    """Lee únicamente la cola final del archivo usando colecciones de doble extremo.
    
    Evita cargar megabytes enteros de logs antiguos en la memoria de la UI.
    """
    path = Path(log_path)
    if not path.exists():
        return []

    events: list[dict[str, Any]] = []
    
    # Extraemos un pool seguro de líneas del final del fichero usando el kernel de OS
    with open(path, "r", encoding="utf-8") as f:
        lines_pool = deque(f, maxlen=tail_lines * 4)

    for line in lines_pool:
        if thread_id not in line:
            continue
        try:
            events.append(json.loads(line))
        except json.JSONDecodeError:
            continue

    return events[-tail_lines:]

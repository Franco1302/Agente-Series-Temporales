"""Fixtures compartidas por los tests del servidor MCP."""

from __future__ import annotations

import importlib
from pathlib import Path

import pytest

_TOOL_MODULES = (
    "mcp_server.tools.drift",
    "mcp_server.tools.synthetic",
    "mcp_server.tools.augment",
    "mcp_server.tools.exogenous",
    "mcp_server.tools.forecast",
)


@pytest.fixture(autouse=True)
def isolated_workspace(tmp_path, monkeypatch):
    """Aísla el workspace y la API URL de cada test parcheando _SETTINGS en cada módulo de tools (lo cachean al cargar, no basta con las env vars)."""
    monkeypatch.setenv("DRIFT_API_URL", "http://testserver")
    monkeypatch.setenv("MCP_WORKSPACE_DIR", str(tmp_path))

    from mcp_server.config import load_settings
    fresh = load_settings()

    for mod_name in _TOOL_MODULES:
        mod = importlib.import_module(mod_name)
        monkeypatch.setattr(mod, "_SETTINGS", fresh)

    yield tmp_path


@pytest.fixture
def sample_csv(tmp_path) -> Path:
    """CSV mínimo con índice temporal y dos columnas numéricas."""
    csv = tmp_path / "sample.csv"
    csv.write_text(
        "ts,v1,v2\n"
        "2024-01-01,1.0,10\n"
        "2024-01-02,1.1,11\n"
        "2024-01-03,1.2,12\n"
        "2024-01-04,1.3,13\n"
        "2024-01-05,1.4,14\n",
        encoding="utf-8",
    )
    return csv

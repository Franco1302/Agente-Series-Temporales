"""Fixtures, hooks y scorecard para los tests de contrato de la API MCP.

Son @pytest.mark.integration (requieren la API real en DRIFT_API_URL). Sobrescriben el
autouse de tests/conftest.py para usar un workspace de sesión y reutilizar los CSV
generados por las tools como input de los tests downstream.
"""

from __future__ import annotations

import asyncio
import csv
import importlib
import os
from pathlib import Path
from typing import Any

import httpx
import pytest

_API_URL = os.getenv("DRIFT_API_URL", "http://localhost:8017").rstrip("/")

_TOOL_MODULES = (
    "mcp_server.tools.drift",
    "mcp_server.tools.synthetic",
    "mcp_server.tools.augment",
    "mcp_server.tools.exogenous",
    "mcp_server.tools.forecast",
)

_RESULTS_DIR = Path(__file__).resolve().parents[2] / "tests" / "results"
_SCORECARD_PATH = _RESULTS_DIR / "api_scorecard.csv"


# ── Hooks pytest: registro de marker + recolección de outcomes ──────────────


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line(
        "markers",
        "weight(n): peso del test en el scorecard agregado (default 1.0)",
    )
    config._api_contracts_outcomes = []  # type: ignore[attr-defined]


def pytest_runtest_makereport(item: pytest.Item, call: pytest.CallInfo[None]) -> None:
    if call.when != "call":
        return
    outcomes = getattr(item.config, "_api_contracts_outcomes", None)
    if outcomes is None:
        return

    weight_marker = item.get_closest_marker("weight")
    weight = float(weight_marker.args[0]) if weight_marker and weight_marker.args else 1.0

    error_msg = ""
    if call.excinfo is not None:
        error_msg = str(call.excinfo.value).splitlines()[0][:200]

    outcomes.append(
        {
            "test": item.nodeid.split("::")[-1],
            "passed": call.excinfo is None,
            "weight": weight,
            "error": error_msg,
        }
    )


def pytest_sessionfinish(session: pytest.Session, exitstatus: int) -> None:
    outcomes = getattr(session.config, "_api_contracts_outcomes", None)
    if not outcomes:
        return

    _RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with _SCORECARD_PATH.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=["test", "passed", "weight", "error"])
        writer.writeheader()
        for row in outcomes:
            writer.writerow(row)

    total = sum(o["weight"] for o in outcomes)
    earned = sum(o["weight"] for o in outcomes if o["passed"])
    pct = (earned / total * 100.0) if total else 0.0

    line = "─" * 60
    print(f"\n{line}")
    print(f"[API Scorecard] {earned:.0f}/{total:.0f} pts = {pct:.1f}%")
    print(f"Detalle en {_SCORECARD_PATH}")
    print(line)


# ── Anulamos el autouse función-scoped del conftest padre ──────────────────


@pytest.fixture(autouse=True)
def isolated_workspace() -> Any:
    """No-op: el workspace lo gestiona session_workspace (scope=session)."""
    yield


# ── Setup de sesión: API real + workspace compartido ───────────────────────


@pytest.fixture(scope="session", autouse=True)
def session_workspace(tmp_path_factory: pytest.TempPathFactory) -> Path:
    workspace = tmp_path_factory.mktemp("api_contracts_workspace")
    mp = pytest.MonkeyPatch()
    mp.setenv("DRIFT_API_URL", _API_URL)
    mp.setenv("MCP_WORKSPACE_DIR", str(workspace))

    from mcp_server.config import load_settings

    fresh = load_settings()
    for mod_name in _TOOL_MODULES:
        mod = importlib.import_module(mod_name)
        mp.setattr(mod, "_SETTINGS", fresh)

    yield workspace
    mp.undo()


@pytest.fixture(scope="session", autouse=True)
def ensure_api_alive() -> None:
    try:
        httpx.get(_API_URL + "/", timeout=3.0).raise_for_status()
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"API no disponible en {_API_URL}: {exc}", allow_module_level=False)


# ── Inputs generados por la propia API, cacheados a sesión ─────────────────


@pytest.fixture(scope="session")
def trend_csv_path(session_workspace: Path) -> str:
    """Serie con tendencia lineal: las dos mitades tienen medias muy distintas, así que KS debe detectar drift."""
    from mcp_server.tools.synthetic import generate_synthetic_trend

    result = asyncio.run(
        generate_synthetic_trend(
            start_date="2024-01-01",
            periods=200,
            frequency="D",
            trend_type=1,  # lineal: valor = intercept + slope*i
            trend_params=[0.0, 0.1],  # [intercept, slope] → 0 a 20 sobre 200 pasos
            noise=0.5,
            column_name="valor",
        )
    )
    assert "error" not in result, f"Generación trend falló: {result.get('error')}"
    return result["output_path"]


@pytest.fixture(scope="session")
def stable_csv_path(session_workspace: Path) -> str:
    """Serie Normal(0,1) estable — KS no debe detectar drift entre mitades."""
    from mcp_server.tools.synthetic import generate_synthetic_distribution

    result = asyncio.run(
        generate_synthetic_distribution(
            start_date="2024-01-01",
            periods=200,
            frequency="D",
            distribution_type=1,  # Normal
            distribution_params=[0.0, 1.0],
            column_name="valor",
        )
    )
    assert "error" not in result, f"Generación estable falló: {result.get('error')}"
    return result["output_path"]


@pytest.fixture(scope="session")
def periodic_csv_path(session_workspace: Path) -> str:
    """Serie con patrón cíclico (período 30) — input para forecast."""
    from mcp_server.tools.synthetic import generate_synthetic_periodic

    result = asyncio.run(
        generate_synthetic_periodic(
            start_date="2024-01-01",
            periods=180,
            frequency="D",
            distribution_type=1,
            distribution_params=[0.0, 1.0],
            period_length=30,
            pattern_type=1,
            column_name="valor",
        )
    )
    assert "error" not in result, f"Generación periódica falló: {result.get('error')}"
    return result["output_path"]

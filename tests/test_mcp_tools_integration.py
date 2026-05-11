"""Tests de integración contra la API drift-detection real.

Requiere que la API esté arrancada en DRIFT_API_URL (por defecto http://localhost:8017).
Saltan automáticamente si la API no responde.

Ejecutar con:
    PYTHONPATH=. pytest -m integration -v
"""

from __future__ import annotations

import importlib
import os
import shutil
from pathlib import Path

import httpx
import pytest

from mcp_server.tools.augment import augment_time_series
from mcp_server.tools.drift import detect_drift
from mcp_server.tools.synthetic import generate_synthetic_distribution

pytestmark = pytest.mark.integration


_API_URL = os.getenv("DRIFT_API_URL", "http://localhost:8017").rstrip("/")


@pytest.fixture(autouse=True)
def real_api_workspace(tmp_path, monkeypatch):
    """Sobreescribe el autouse de unit tests: aquí queremos la API real, no testserver."""
    monkeypatch.setenv("DRIFT_API_URL", _API_URL)
    monkeypatch.setenv("MCP_WORKSPACE_DIR", str(tmp_path))

    from mcp_server.config import load_settings
    fresh = load_settings()

    for mod_name in (
        "mcp_server.tools.drift",
        "mcp_server.tools.synthetic",
        "mcp_server.tools.augment",
        "mcp_server.tools.exogenous",
        "mcp_server.tools.forecast",
    ):
        mod = importlib.import_module(mod_name)
        monkeypatch.setattr(mod, "_SETTINGS", fresh)

    yield tmp_path


@pytest.fixture(scope="session", autouse=True)
def ensure_api_alive():
    """Salta toda la suite si la API no responde."""
    try:
        r = httpx.get(_API_URL + "/", timeout=3.0)
        r.raise_for_status()
    except Exception as exc:
        pytest.skip(f"API no disponible en {_API_URL}: {exc}", allow_module_level=False)


@pytest.fixture
def fixtures_dir() -> Path:
    return Path(__file__).parent / "fixtures"


# ─────────────────────────── 1. Generación pura ───────────────────────────

@pytest.mark.asyncio
async def test_generate_normal_series_creates_csv():
    out = await generate_synthetic_distribution(
        start_date="2024-01-01", periods=100, frequency="D",
        distribution_type=1, distribution_params=[0.0, 1.0],
    )
    assert "error" not in out, f"API devolvió error: {out.get('error')}"
    assert out["rows_generated"] > 0
    csv_path = Path(out["output_path"])
    assert csv_path.exists()


# ─────────────────────────── 2. Drift sobre output anterior ───────────────────────────

@pytest.mark.asyncio
async def test_detect_drift_on_generated_csv(tmp_path):
    gen = await generate_synthetic_distribution(
        start_date="2024-01-01", periods=200, frequency="D",
        distribution_type=1, distribution_params=[0.0, 1.0],
        column_name="valor",
    )
    assert "error" not in gen, f"Generación falló: {gen.get('error')}"

    csv_path = Path(gen["output_path"])
    header = csv_path.read_text(encoding="utf-8").split("\n", 1)[0].split(",")
    index_col = header[0]

    out = await detect_drift(file_path=str(csv_path), index_column=index_col, method="KS")
    assert "error" not in out, f"Drift KS falló: {out.get('error')}"
    assert "drift_label" in out


# ─────────────────────────── 3. Aumentación ───────────────────────────

@pytest.mark.asyncio
async def test_augment_and_detect_drift(tmp_path):
    gen = await generate_synthetic_distribution(
        start_date="2024-01-01", periods=50, frequency="D",
        distribution_type=1, distribution_params=[0.0, 1.0],
        column_name="valor",
    )
    assert "error" not in gen, gen.get("error")
    csv_path = Path(gen["output_path"])
    header = csv_path.read_text(encoding="utf-8").split("\n", 1)[0].split(",")
    index_col = header[0]

    aug = await augment_time_series(
        file_path=str(csv_path), index_column=index_col, strategy="normal",
        size=50, frequency="D",
    )
    assert "error" not in aug, aug.get("error")
    assert Path(aug["output_path"]).exists()


# ─────────────────────────── 4. Caso de error: fichero inexistente ───────────────────────────

@pytest.mark.asyncio
async def test_detect_drift_with_missing_file():
    out = await detect_drift(
        file_path="/tmp/no_existe_ABCXYZ.csv", index_column="ts", method="KS",
    )
    assert "error" in out
    assert "Fichero no encontrado" in out["error"]


# ─────────────────────────── 5. Drift sobre CSV fixture ───────────────────────────

@pytest.mark.asyncio
async def test_detect_drift_on_fixture(fixtures_dir, tmp_path):
    fixture = fixtures_dir / "sample_drift.csv"
    target = tmp_path / "sample_drift.csv"
    shutil.copy(fixture, target)

    out = await detect_drift(
        file_path=str(target), index_column="ts", method="PSI",
        threshold=0.25, num_bins=10,
    )
    assert "error" not in out, f"PSI falló: {out.get('error')}"
    assert out["method_used"] == "PSI"
    assert "drift_label" in out

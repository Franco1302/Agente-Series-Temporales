"""Tests de contrato de la API MCP con scoring ponderado.

Cada test asegura un contrato concreto de una tool MCP contra el backend real.
Los pesos definidos con @pytest.mark.weight() agregan a un scorecard final
escrito en tests/results/api_scorecard.csv por el hook pytest_sessionfinish.

Ejecutar:
    PYTHONPATH=. pytest -m integration tests/api_contracts -v
    cat tests/results/api_scorecard.csv
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from mcp_server.tools.augment import augment_time_series
from mcp_server.tools.drift import detect_drift
from mcp_server.tools.exogenous import create_exogenous_variable
from mcp_server.tools.forecast import forecast_time_series
from mcp_server.tools.synthetic import generate_synthetic_distribution

pytestmark = pytest.mark.integration


# ── Contratos de generate_synthetic_distribution ───────────────────────────


@pytest.mark.weight(10)
async def test_synthetic_returns_expected_keys():
    """Contrato: la tool devuelve output_path, rows_generated, summary."""
    result = await generate_synthetic_distribution(
        start_date="2024-01-01",
        periods=50,
        frequency="D",
        distribution_type=1,
        distribution_params=[0.0, 1.0],
    )
    assert "error" not in result, f"API devolvió error: {result.get('error')}"
    assert set(result.keys()) >= {"output_path", "rows_generated", "summary"}
    assert result["rows_generated"] >= 50
    assert Path(result["output_path"]).exists()
    assert result["output_path"].endswith(".csv")


@pytest.mark.weight(10)
async def test_synthetic_with_plot_emits_image_path():
    """Contrato: con with_plot=True el dict incluye image_path apuntando a un PNG."""
    result = await generate_synthetic_distribution(
        start_date="2024-01-01",
        periods=30,
        frequency="D",
        distribution_type=1,
        distribution_params=[0.0, 1.0],
        with_plot=True,
    )
    assert "error" not in result, result.get("error")
    assert result.get("image_path"), "Esperaba image_path no nulo"
    png = Path(result["image_path"])
    assert png.exists() and png.suffix == ".png"


# ── Contratos de detect_drift ──────────────────────────────────────────────


@pytest.mark.weight(20)
async def test_drift_ks_detects_on_trend(trend_csv_path: str):
    """Una serie con tendencia clara debe producir drift_label='Detectado' con KS."""
    out = await detect_drift(
        file_path=trend_csv_path,
        index_column="Indice",
        method="KS",
        threshold=0.05,
    )
    assert "error" not in out, f"Drift KS falló: {out.get('error')}"
    assert out["method_used"] == "KS"
    assert out["drift_label"] == "Detectado", (
        f"Esperaba drift detectado en serie con tendencia, got {out['drift_label']}"
    )
    assert out["drift_detected"] is True
    assert "per_column_report" in out
    assert isinstance(out["parameters_used"], dict)


@pytest.mark.weight(15)
async def test_drift_ks_no_drift_on_stable(stable_csv_path: str):
    """Una serie Normal(0,1) estable no debe producir drift con KS umbral 0.01.

    Usamos umbral 0.01 para evitar falsos positivos del 5% que se obtienen con
    el umbral típico de 0.05 sobre datos puramente aleatorios.
    """
    out = await detect_drift(
        file_path=stable_csv_path,
        index_column="Indice",
        method="KS",
        threshold=0.01,
    )
    assert "error" not in out, f"Drift KS falló: {out.get('error')}"
    assert out["method_used"] == "KS"
    assert out["drift_label"] == "No detectado", (
        f"Esperaba no-drift en serie estable, got {out['drift_label']}"
    )
    assert out["drift_detected"] is False


@pytest.mark.weight(10)
async def test_drift_method_routing(trend_csv_path: str):
    """Cada método válido produce method_used coincidente y per_column_report."""
    for method, threshold in (("KS", 0.05), ("JS", 0.2), ("PSI", 0.25)):
        out = await detect_drift(
            file_path=trend_csv_path,
            index_column="Indice",
            method=method,  # type: ignore[arg-type]
            threshold=threshold,
        )
        assert "error" not in out, f"{method} falló: {out.get('error')}"
        assert out["method_used"] == method
        assert "per_column_report" in out


@pytest.mark.weight(5)
async def test_drift_error_on_missing_file():
    """Contrato de error: fichero inexistente devuelve dict con clave 'error'."""
    out = await detect_drift(
        file_path="/tmp/no_existe_ABCXYZ_unique.csv",
        index_column="Indice",
        method="KS",
    )
    assert "error" in out
    assert "Fichero no encontrado" in out["error"]


# ── Contratos de augment_time_series ───────────────────────────────────────


@pytest.mark.weight(15)
async def test_augment_extends_csv(stable_csv_path: str):
    """augment normal con size=50 produce un CSV con más filas que el original."""
    out = await augment_time_series(
        file_path=stable_csv_path,
        index_column="Indice",
        strategy="normal",
        size=50,
        frequency="D",
    )
    assert "error" not in out, f"augment falló: {out.get('error')}"
    assert Path(out["output_path"]).exists()
    assert out["strategy_used"] == "normal"

    original = pd.read_csv(stable_csv_path)
    augmented = pd.read_csv(out["output_path"])
    assert len(augmented) > len(original), (
        f"augment no añadió filas: original={len(original)}, augmented={len(augmented)}"
    )


# ── Contratos de create_exogenous_variable ─────────────────────────────────


@pytest.mark.weight(10)
async def test_exogenous_linear_adds_column(stable_csv_path: str):
    """Relación lineal añade una columna nueva con el nombre solicitado."""
    out = await create_exogenous_variable(
        file_path=stable_csv_path,
        index_column="Indice",
        new_column_name="valor_lineal",
        relation="linear",
        coefficients=[2.0, 3.0],
    )
    assert "error" not in out, f"exogenous falló: {out.get('error')}"
    assert out["new_column_name"] == "valor_lineal"
    assert out["relation_used"] == "linear"

    df = pd.read_csv(out["output_path"])
    assert "valor_lineal" in df.columns, (
        f"Columna nueva ausente en CSV. Columnas: {list(df.columns)}"
    )


# ── Contratos de forecast_time_series ──────────────────────────────────────


@pytest.mark.weight(15)
async def test_forecast_sarimax_on_periodic(periodic_csv_path: str):
    """SARIMAX sobre serie periódica produce CSV con forecast_steps filas."""
    forecast_steps = 30
    out = await forecast_time_series(
        file_path=periodic_csv_path,
        index_column="Indice",
        model="sarimax",
        forecast_steps=forecast_steps,
        frequency="D",
        return_metrics=False,
    )
    assert "error" not in out, f"forecast falló: {out.get('error')}"
    assert out["model_used"] == "sarimax"
    assert Path(out["output_path"]).exists()

    df = pd.read_csv(out["output_path"])
    assert len(df) >= forecast_steps, (
        f"Forecast devolvió {len(df)} filas, esperaba >= {forecast_steps}"
    )

"""Tests unitarios del servidor MCP. Toda la I/O HTTP está mockeada con respx.

Las tools exponen al LLM un schema plano (params al nivel raíz, sin envoltorio
Pydantic). Internamente las funciones construyen los modelos `XInput(...)` para
delegar a los helpers `_build_query_params`. Los tests llaman a las tools con
kwargs, igual que lo haría el cliente MCP, y a los helpers con `XInput(...)`.
"""

from __future__ import annotations

import httpx
import pytest
import respx

from mcp_server.errors import translate_exception
from mcp_server.tools.augment import (
    AugmentTimeSeriesInput,
    _build_query_params as augment_params,
    augment_time_series,
)
from mcp_server.tools.drift import (
    DetectDriftInput,
    _build_query_params as drift_params,
    detect_drift,
)
from mcp_server.tools.exogenous import (
    CreateExogenousVariableInput,
    _build_query_params as exo_params,
    create_exogenous_variable,
)
from mcp_server.tools.forecast import (
    _normalize_csv_for_backend,
    forecast_time_series,
)
from mcp_server.tools.synthetic import (
    _resolve_horizon,
    generate_synthetic_arma,
    generate_synthetic_distribution,
    generate_synthetic_periodic,
    generate_synthetic_trend,
)


# ─────────────────────────── translate_exception ───────────────────────────

def test_translate_http_status_with_detail():
    response = httpx.Response(422, json={"detail": "param X faltante"}, request=httpx.Request("POST", "http://x"))
    exc = httpx.HTTPStatusError("422", request=response.request, response=response)
    msg = translate_exception(exc, "detect_drift")
    assert "422" in msg and "param X faltante" in msg


def test_translate_timeout():
    msg = translate_exception(httpx.TimeoutException("slow"), "forecast_time_series")
    assert "Timeout" in msg and "forecast_time_series" in msg


def test_translate_connect_error():
    msg = translate_exception(httpx.ConnectError("nope"), "detect_drift")
    assert "8017" in msg or "Docker" in msg


def test_translate_file_not_found():
    msg = translate_exception(FileNotFoundError("missing.csv"), "augment_time_series")
    assert "Fichero no encontrado" in msg and "augment_time_series" in msg


def test_translate_value_error():
    msg = translate_exception(ValueError("bad param"), "create_exogenous_variable")
    assert "Parámetro inválido" in msg


# ─────────────────────────── drift._build_query_params ───────────────────────────

def test_drift_params_ks_default():
    inp = DetectDriftInput(file_path="/tmp/x.csv", index_column="ts", method="KS")
    params = drift_params(inp)
    assert params == {"indice": "ts", "inicio": 1, "threshold_ks": 0.05}


def test_drift_params_psi_with_bins():
    inp = DetectDriftInput(file_path="/tmp/x.csv", index_column="ts", method="PSI", num_bins=20, threshold=0.3)
    params = drift_params(inp)
    assert params["num_bins"] == 20
    assert params["threshold_psi"] == 0.3


def test_drift_params_cusum_defaults():
    inp = DetectDriftInput(file_path="/tmp/x.csv", index_column="ts", method="CUSUM")
    params = drift_params(inp)
    assert params["threshold_cusum"] == 1.5
    assert params["drift_cusum"] == 0.5


def test_drift_params_mewma():
    inp = DetectDriftInput(file_path="/tmp/x.csv", index_column="ts", method="MEWMA")
    params = drift_params(inp)
    assert params["min_instances"] == 100
    # _build_query_params asigna 0.05 cuando alpha no se proporciona explícitamente
    # (ver mcp_server/tools/drift.py: "params['alpha'] = inp.alpha if inp.alpha is not None else 0.05").
    assert params["alpha"] == 0.05
    assert params["lambd"] == 0.5


def test_drift_params_hotelling_no_lambd():
    inp = DetectDriftInput(file_path="/tmp/x.csv", index_column="ts", method="HOTELLING", min_instances=50, alpha=0.05)
    params = drift_params(inp)
    assert params["min_instances"] == 50
    assert params["alpha"] == 0.05
    assert "lambd" not in params


# ─────────────────────────── detect_drift end-to-end (respx) ───────────────────────────

@respx.mock(base_url="http://testserver")
@pytest.mark.asyncio
async def test_detect_drift_happy_path(respx_mock, sample_csv):
    respx_mock.post("/Deteccion/KS").mock(
        return_value=httpx.Response(200, json={"Drift": "Detectado", "reporte": {"v1": {"drift": True}}})
    )
    out = await detect_drift(file_path=str(sample_csv), index_column="ts", method="KS")
    assert out["drift_detected"] is True
    assert out["method_used"] == "KS"
    assert "v1" in out["per_column_report"]
    assert "Se detectó drift" in out["summary"]


@respx.mock(base_url="http://testserver")
@pytest.mark.asyncio
async def test_detect_drift_no_drift(respx_mock, sample_csv):
    respx_mock.post("/Deteccion/PSI").mock(
        return_value=httpx.Response(200, json={"Drift": "No detectado", "reporte": {}})
    )
    out = await detect_drift(file_path=str(sample_csv), index_column="ts", method="PSI")
    assert out["drift_detected"] is False
    assert "No se detectó drift" in out["summary"]


@pytest.mark.asyncio
async def test_detect_drift_file_not_found():
    out = await detect_drift(file_path="/tmp/no_existe_jamas.csv", index_column="ts", method="KS")
    assert "error" in out
    assert "Fichero no encontrado" in out["error"]


# ─────────────────────────── synthetic._resolve_horizon ───────────────────────────

def test_horizon_periods():
    suffix, extra = _resolve_horizon(None, 100)
    assert suffix == "periodos"
    assert extra == {"periodos": 100}


def test_horizon_end_date():
    suffix, extra = _resolve_horizon("2024-12-31", None)
    assert suffix == "fin"
    assert extra == {"fin": "2024-12-31"}


def test_horizon_both_raises():
    with pytest.raises(ValueError):
        _resolve_horizon("2024-12-31", 100)


def test_horizon_neither_raises():
    with pytest.raises(ValueError):
        _resolve_horizon(None, None)


# ─────────────────────────── synthetic.distribution end-to-end ───────────────────────────

@respx.mock(base_url="http://testserver")
@pytest.mark.asyncio
async def test_generate_distribution_periodos(respx_mock):
    csv_body = b"fecha,valor\n2024-01-01,1.0\n2024-01-02,2.0\n2024-01-03,3.0\n"
    respx_mock.get("/Datos/distribucion/periodos").mock(return_value=httpx.Response(200, content=csv_body))

    out = await generate_synthetic_distribution(
        start_date="2024-01-01", periods=3, frequency="D",
        distribution_type=1, distribution_params=[0.0, 1.0],
    )
    assert "output_path" in out
    assert out["rows_generated"] == 3


@respx.mock(base_url="http://testserver")
@pytest.mark.asyncio
async def test_generate_distribution_end_date(respx_mock):
    csv_body = b"fecha,valor\n2024-01-01,1.0\n"
    respx_mock.get("/Datos/distribucion/fin").mock(return_value=httpx.Response(200, content=csv_body))

    out = await generate_synthetic_distribution(
        start_date="2024-01-01", end_date="2024-01-05", frequency="D",
        distribution_type=1, distribution_params=[0.0, 1.0],
    )
    assert "output_path" in out


@pytest.mark.asyncio
async def test_generate_distribution_exclusion_error():
    out = await generate_synthetic_distribution(
        start_date="2024-01-01", periods=10, end_date="2024-01-15", frequency="D",
        distribution_type=1, distribution_params=[0.0, 1.0],
    )
    assert "error" in out
    assert "periods" in out["error"] and "end_date" in out["error"]


@pytest.mark.asyncio
async def test_generate_distribution_bad_arity_error():
    """F2: Normal (tipo 1) exige 2 params; con 1 devuelve error accionable, no un 500."""
    out = await generate_synthetic_distribution(
        start_date="2024-01-01", periods=10, frequency="D",
        distribution_type=1, distribution_params=[0.0],
    )
    assert "error" in out
    assert "distribution_params" in out["error"]


@pytest.mark.asyncio
async def test_generate_distribution_uncurated_code_rejected():
    """F3: un código no curado (8=lognormal) se rechaza por el schema (Literal)."""
    out = await generate_synthetic_distribution(
        start_date="2024-01-01", periods=10, frequency="D",
        distribution_type=8, distribution_params=[1.0],
    )
    assert "error" in out


# ─────────────────────────── synthetic ARMA / periodic / trend ───────────────────────────

@respx.mock(base_url="http://testserver")
@pytest.mark.asyncio
async def test_generate_arma(respx_mock):
    respx_mock.get("/Datos/ARMA/periodos").mock(
        return_value=httpx.Response(200, content=b"f,v\n2024-01-01,1.0\n")
    )
    out = await generate_synthetic_arma(
        start_date="2024-01-01", periods=1, frequency="D",
        ar_coefficients=[0.5], ma_coefficients=[0.3],
    )
    assert "output_path" in out
    assert out["model_spec"]["ar"] == [0.5]


@respx.mock(base_url="http://testserver")
@pytest.mark.asyncio
async def test_generate_periodic(respx_mock):
    respx_mock.get("/Datos/periodicos/periodos").mock(
        return_value=httpx.Response(200, content=b"f,v\n2024-01-01,1.0\n")
    )
    out = await generate_synthetic_periodic(
        start_date="2024-01-01", periods=10, frequency="D",
        distribution_type=1, distribution_params=[0.0, 1.0],
        period_length=7, pattern_type=1,
    )
    assert "output_path" in out


@respx.mock(base_url="http://testserver")
@pytest.mark.asyncio
async def test_generate_trend(respx_mock):
    respx_mock.get("/Datos/tendencia/periodos").mock(
        return_value=httpx.Response(200, content=b"f,v\n2024-01-01,1.0\n")
    )
    out = await generate_synthetic_trend(
        start_date="2024-01-01", periods=5, frequency="D",
        trend_type=1, trend_params=[1.0, 0.1], noise=0.0,
    )
    assert "output_path" in out


# ─────────────────────────── augment ───────────────────────────

def test_augment_params_normal():
    inp = AugmentTimeSeriesInput(
        file_path="/tmp/x.csv", index_column="ts", strategy="normal",
        size=50, frequency="D",
    )
    assert augment_params(inp) == {"indice": "ts", "freq": "D", "size": 50}


def test_augment_params_duplicate_defaults():
    inp = AugmentTimeSeriesInput(
        file_path="/tmp/x.csv", index_column="ts", strategy="duplicate",
        size=50, frequency="D",
    )
    params = augment_params(inp)
    assert params["duplication_factor"] == 0.5
    assert params["perturbation_std"] == 0.1
    assert "size" not in params  # /Aumentar/Duplicado no acepta size


def test_augment_params_statistical():
    inp = AugmentTimeSeriesInput(
        file_path="/tmp/x.csv", index_column="ts", strategy="statistical",
        size=30, frequency="D",
    )
    params = augment_params(inp)
    assert params["tipo"] == 1
    assert params["num"] == 30


@respx.mock(base_url="http://testserver")
@pytest.mark.asyncio
async def test_augment_happy_path(respx_mock, sample_csv):
    respx_mock.post("/Aumentar/Normal").mock(
        return_value=httpx.Response(200, content=b"ts,v\n2024-01-01,1.0\n")
    )
    out = await augment_time_series(
        file_path=str(sample_csv), index_column="ts", strategy="normal",
        size=10, frequency="D",
    )
    assert "output_path" in out
    assert out["new_rows"] == 10
    assert out["strategy_used"] == "normal"


# ─────────────────────────── exogenous ───────────────────────────

def test_exogenous_params_pca():
    inp = CreateExogenousVariableInput(
        file_path="/tmp/x.csv", index_column="ts", new_column_name="pca1", relation="pca",
    )
    assert exo_params(inp) == {"indice": "ts", "columna": "pca1"}


def test_exogenous_params_linear():
    inp = CreateExogenousVariableInput(
        file_path="/tmp/x.csv", index_column="ts", new_column_name="y",
        relation="linear", coefficients=[2.0, 0.5],
    )
    params = exo_params(inp)
    assert params["a"] == 2.0
    assert params["b"] == 0.5


def test_exogenous_params_linear_missing_coefficients():
    inp = CreateExogenousVariableInput(
        file_path="/tmp/x.csv", index_column="ts", new_column_name="y", relation="linear",
    )
    with pytest.raises(ValueError):
        exo_params(inp)


def test_exogenous_params_polynomial():
    inp = CreateExogenousVariableInput(
        file_path="/tmp/x.csv", index_column="ts", new_column_name="p",
        relation="polynomial", coefficients=[0.0, 1.0, 0.5],
    )
    params = exo_params(inp)
    assert params["a"] == [0.0, 1.0, 0.5]


@respx.mock(base_url="http://testserver")
@pytest.mark.asyncio
async def test_exogenous_happy_path(respx_mock, sample_csv):
    respx_mock.post("/Variables/PCA").mock(
        return_value=httpx.Response(200, content=b"ts,v1,v2,pca1\n2024-01-01,1,10,0.5\n")
    )
    out = await create_exogenous_variable(
        file_path=str(sample_csv), index_column="ts", new_column_name="pca1", relation="pca",
    )
    assert "output_path" in out
    assert out["relation_used"] == "pca"


# ─────────────────────────── forecast ───────────────────────────

@respx.mock(base_url="http://testserver")
@pytest.mark.asyncio
async def test_forecast_with_metrics(respx_mock, sample_csv):
    respx_mock.post("/Datos/Sarimax").mock(
        return_value=httpx.Response(200, content=b"ts,prediccion\n2024-01-21,1.0\n")
    )
    respx_mock.post("/Error/Sarimax").mock(
        return_value=httpx.Response(200, json={"MAE": 0.1, "RMSE": 0.15, "MAPE": 0.05})
    )
    out = await forecast_time_series(
        file_path=str(sample_csv), index_column="ts",
        model="sarimax", forecast_steps=5, return_metrics=True,
    )
    assert "output_path" in out
    assert out["metrics"] == {"MAE": 0.1, "RMSE": 0.15, "MAPE": 0.05}
    assert out["model_used"] == "sarimax"


@respx.mock(base_url="http://testserver")
@pytest.mark.asyncio
async def test_forecast_no_metrics(respx_mock, sample_csv):
    respx_mock.post("/Datos/Sarimax").mock(
        return_value=httpx.Response(200, content=b"ts,p\n2024-01-21,1.0\n")
    )
    out = await forecast_time_series(
        file_path=str(sample_csv), index_column="ts",
        model="sarimax", forecast_steps=5, return_metrics=False, with_plot=False,
    )
    assert "output_path" in out
    assert out["metrics"] is None


def test_normalize_csv_daily_passthrough():
    """CSV diario: la normalización no toca el contenido y devuelve alias 'D'."""
    import pandas as pd
    idx = pd.date_range("2024-01-01", periods=10, freq="D").strftime("%Y-%m-%d")
    csv = ("ts,v\n" + "\n".join(f"{d},{i}" for i, d in enumerate(idx))).encode()
    new_content, alias = _normalize_csv_for_backend(csv, "ts")
    assert alias == "D"
    assert new_content == csv


def test_normalize_csv_monthly_start_rewrites_to_end():
    """CSV con fechas a inicio de mes (MS): reescribe a final de mes y devuelve 'M'."""
    import io, pandas as pd
    idx = pd.date_range("2020-01-01", periods=12, freq="MS").strftime("%Y-%m-%d")
    csv = ("ts,v\n" + "\n".join(f"{d},{i}" for i, d in enumerate(idx))).encode()
    new_content, alias = _normalize_csv_for_backend(csv, "ts")
    assert alias == "M"
    assert new_content != csv
    df = pd.read_csv(io.BytesIO(new_content))
    # Todas las fechas deben caer al final del mes (último día).
    parsed = pd.to_datetime(df["ts"])
    assert all(parsed.dt.is_month_end), "Las fechas deberían estar a final de mes"


def test_normalize_csv_unknown_column_is_noop():
    """Si la columna índice no existe, devuelve el contenido tal cual y alias None."""
    csv = b"ts,v\n2024-01-01,1\n2024-01-02,2\n2024-01-03,3\n"
    new_content, alias = _normalize_csv_for_backend(csv, "no_existe")
    assert alias is None
    assert new_content == csv

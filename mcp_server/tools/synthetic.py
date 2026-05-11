"""Tools MCP de generación de series sintéticas (distribución, ARMA, periódica, tendencia)."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated, Literal, Optional

import httpx
from pydantic import BaseModel, Field

from mcp_server.config import load_settings
from mcp_server.errors import translate_exception
from mcp_server.file_utils import deterministic_filename
from mcp_server.http_client import get_client
from mcp_server.instance import mcp

_SETTINGS = load_settings()

_FREQ = Literal["B", "D", "W", "M", "Q", "Y", "h", "min", "s"]


def _resolve_horizon(end_date: Optional[str], periods: Optional[int]) -> tuple[str, dict]:
    """Devuelve (sufijo_endpoint, extra_params). Valida la exclusión end_date xor periods."""
    if (end_date is None) == (periods is None):
        raise ValueError(
            "Debes proporcionar 'periods' O 'end_date' (no ambos, no ninguno)."
        )
    if periods is not None:
        return "periodos", {"periodos": periods}
    return "fin", {"fin": end_date}


async def _download_csv(client: httpx.AsyncClient, endpoint: str, params: dict, out_name: str) -> Path:
    response = await client.get(endpoint, params=params)
    response.raise_for_status()
    target = _SETTINGS.workspace_dir / out_name
    target.write_bytes(response.content)
    return target


async def _download_png(
    client: httpx.AsyncClient,
    endpoint: str,
    params: dict,
    out_name: str,
) -> Path:
    response = await client.get(endpoint, params=params)
    response.raise_for_status()
    target = _SETTINGS.workspace_dir / out_name
    target.write_bytes(response.content)
    return target


def _row_count(csv_path: Path) -> int:
    try:
        with csv_path.open("r", encoding="utf-8") as fh:
            return max(sum(1 for _ in fh) - 1, 0)
    except Exception:
        return 0


# Schemas internos (usados por tests unitarios — el LLM ve params planos).

class GenerateDistributionInput(BaseModel):
    start_date: str
    end_date: Optional[str] = None
    periods: Optional[int] = None
    frequency: _FREQ
    distribution_type: int
    distribution_params: list[float]
    column_name: str = "valor"
    with_plot: bool = False


class GenerateArmaInput(BaseModel):
    start_date: str
    end_date: Optional[str] = None
    periods: Optional[int] = None
    frequency: _FREQ
    column_name: str = "valor"
    constant: float = 0.0
    noise_std: float = 1.0
    seasonality: int = 0
    ar_coefficients: list[float] = []
    ma_coefficients: list[float] = []
    with_plot: bool = False


class GeneratePeriodicInput(BaseModel):
    start_date: str
    end_date: Optional[str] = None
    periods: Optional[int] = None
    frequency: _FREQ
    column_name: str = "valor"
    distribution_type: int
    distribution_params: list[float]
    period_length: int
    pattern_type: Literal[1, 2]
    with_plot: bool = False


class GenerateTrendInput(BaseModel):
    start_date: str
    end_date: Optional[str] = None
    periods: Optional[int] = None
    frequency: _FREQ
    column_name: str = "valor"
    trend_type: int
    trend_params: list[float]
    noise: float = 0.0
    with_plot: bool = False


# ───────────────────────── 1. generate_synthetic_distribution ─────────────────────────

@mcp.tool()
async def generate_synthetic_distribution(
    start_date: Annotated[str, Field(description="Fecha de inicio 'YYYY-MM-DD'.")],
    frequency: Annotated[_FREQ, Field(description="'D' diaria, 'W' semanal, 'M' mensual, 'h' horaria.")],
    distribution_type: Annotated[
        int,
        Field(
            ge=1, le=17,
            description=(
                "1=Normal[mu,sigma], 2=Binomial[n,p], 3=Poisson[lambda], 4=Geometrica[p], "
                "7=Uniforme[low,high], 9=Exponencial[scale], 10=Gamma[a], 11=Beta[a,b], "
                "12=ChiCuadrado[df], 13=TStudent[t], 17=Aleatorio[low,high]"
            ),
        ),
    ],
    distribution_params: Annotated[list[float], Field(description="Parámetros de la distribución como lista.")],
    end_date: Annotated[Optional[str], Field(description="Fecha de fin 'YYYY-MM-DD'. Excluyente con periods.")] = None,
    periods: Annotated[Optional[int], Field(description="Número de periodos. Excluyente con end_date.")] = None,
    column_name: Annotated[str, Field(description="Nombre de la columna generada.")] = "valor",
    with_plot: Annotated[bool, Field(description="Si True, genera además un PNG con la gráfica.")] = False,
) -> dict:
    """Genera una serie temporal sintética siguiendo una distribución estadística.

    USA cuando el usuario quiera crear datos artificiales con una distribución conocida
    (Normal, Poisson, Uniforme, Beta, Gamma...). NO uses si la serie debe tener
    autocorrelación (usa generate_synthetic_arma) ni patrones cíclicos
    (usa generate_synthetic_periodic).

    Pasa `periods` O `end_date`, no ambos.

    Devuelve: output_path, rows_generated, image_path, summary.
    """
    try:
        inp = GenerateDistributionInput(
            start_date=start_date, end_date=end_date, periods=periods,
            frequency=frequency, distribution_type=distribution_type,
            distribution_params=distribution_params, column_name=column_name,
            with_plot=with_plot,
        )
        suffix, horizon = _resolve_horizon(inp.end_date, inp.periods)
        endpoint = f"/Datos/distribucion/{suffix}"
        params = {
            "inicio": inp.start_date,
            "freq": inp.frequency,
            "distr": inp.distribution_type,
            "columna": inp.column_name,
            "params": inp.distribution_params,
            **horizon,
        }
        out_name = deterministic_filename(
            "distribucion",
            inp.start_date, str(horizon), inp.frequency,
            str(inp.distribution_type), str(inp.distribution_params),
            ext="csv",
        )

        async with get_client(_SETTINGS) as client:
            csv_path = await _download_csv(client, endpoint, params, out_name)
            png_path: Optional[Path] = None
            if inp.with_plot:
                png_name = out_name.replace(".csv", ".png")
                png_path = await _download_png(client, f"/Plot/distribucion/{suffix}", params, png_name)

        rows = _row_count(csv_path)
        return {
            "output_path": str(csv_path),
            "rows_generated": rows,
            "image_path": str(png_path) if png_path else None,
            "summary": (
                f"Serie generada con distribución tipo {inp.distribution_type} "
                f"({rows} filas, freq={inp.frequency})."
            ),
        }
    except Exception as exc:  # noqa: BLE001
        return {"error": translate_exception(exc, "generate_synthetic_distribution")}


# ───────────────────────── 2. generate_synthetic_arma ─────────────────────────

@mcp.tool()
async def generate_synthetic_arma(
    start_date: Annotated[str, Field(description="Fecha de inicio 'YYYY-MM-DD'.")],
    frequency: Annotated[_FREQ, Field(description="Frecuencia temporal.")],
    end_date: Annotated[Optional[str], Field(description="Fecha de fin (excluyente con periods).")] = None,
    periods: Annotated[Optional[int], Field(description="Número de periodos (excluyente con end_date).")] = None,
    column_name: Annotated[str, Field(description="Nombre de la columna generada.")] = "valor",
    constant: Annotated[float, Field(description="Término constante c del modelo ARMA.")] = 0.0,
    noise_std: Annotated[float, Field(description="Desviación estándar del ruido blanco.")] = 1.0,
    seasonality: Annotated[int, Field(description="Periodo de estacionalidad (0 = no estacional).")] = 0,
    ar_coefficients: Annotated[list[float], Field(description="Coeficientes AR. Lista vacía = sin AR.")] = [],
    ma_coefficients: Annotated[list[float], Field(description="Coeficientes MA. Lista vacía = sin MA.")] = [],
    with_plot: Annotated[bool, Field(description="Si True, también genera PNG.")] = False,
) -> dict:
    """Genera una serie temporal con estructura ARMA(p,q).

    USA cuando el usuario pida datos con autocorrelación temporal, AR(p), MA(q),
    ARMA(p,q) o estacionalidad fija. Pasa `periods` O `end_date`, no ambos.

    Devuelve: output_path, rows_generated, image_path, summary, model_spec.
    """
    try:
        inp = GenerateArmaInput(
            start_date=start_date, end_date=end_date, periods=periods,
            frequency=frequency, column_name=column_name, constant=constant,
            noise_std=noise_std, seasonality=seasonality,
            ar_coefficients=ar_coefficients, ma_coefficients=ma_coefficients,
            with_plot=with_plot,
        )
        suffix, horizon = _resolve_horizon(inp.end_date, inp.periods)
        endpoint = f"/Datos/ARMA/{suffix}"
        params = {
            "inicio": inp.start_date,
            "freq": inp.frequency,
            "c": inp.constant,
            "desv": inp.noise_std,
            "s": inp.seasonality,
            "columna": inp.column_name,
            "phi": inp.ar_coefficients,
            "teta": inp.ma_coefficients,
            **horizon,
        }
        out_name = deterministic_filename(
            "arma",
            inp.start_date, str(horizon), inp.frequency,
            str(inp.constant), str(inp.noise_std), str(inp.seasonality),
            str(inp.ar_coefficients), str(inp.ma_coefficients),
            ext="csv",
        )

        async with get_client(_SETTINGS) as client:
            csv_path = await _download_csv(client, endpoint, params, out_name)
            png_path: Optional[Path] = None
            if inp.with_plot:
                png_name = out_name.replace(".csv", ".png")
                png_path = await _download_png(client, f"/Plot/ARMA/{suffix}", params, png_name)

        rows = _row_count(csv_path)
        return {
            "output_path": str(csv_path),
            "rows_generated": rows,
            "image_path": str(png_path) if png_path else None,
            "model_spec": {
                "ar": inp.ar_coefficients,
                "ma": inp.ma_coefficients,
                "seasonality": inp.seasonality,
            },
            "summary": (
                f"Serie ARMA generada ({rows} filas, AR={len(inp.ar_coefficients)}, "
                f"MA={len(inp.ma_coefficients)}, freq={inp.frequency})."
            ),
        }
    except Exception as exc:  # noqa: BLE001
        return {"error": translate_exception(exc, "generate_synthetic_arma")}


# ───────────────────────── 3. generate_synthetic_periodic ─────────────────────────

@mcp.tool()
async def generate_synthetic_periodic(
    start_date: Annotated[str, Field(description="Fecha de inicio 'YYYY-MM-DD'.")],
    frequency: Annotated[_FREQ, Field(description="Frecuencia temporal.")],
    distribution_type: Annotated[int, Field(ge=1, le=17, description="Distribución base (ver generate_synthetic_distribution).")],
    distribution_params: Annotated[list[float], Field(description="Parámetros de la distribución base.")],
    period_length: Annotated[int, Field(gt=0, description="Cada cuántas observaciones se repite el patrón.")],
    pattern_type: Annotated[Literal[1, 2], Field(description="1 = variación de amplitud, 2 = variación de cantidad.")],
    end_date: Annotated[Optional[str], Field(description="Fecha de fin (excluyente con periods).")] = None,
    periods: Annotated[Optional[int], Field(description="Número de periodos (excluyente con end_date).")] = None,
    column_name: Annotated[str, Field(description="Nombre de la columna.")] = "valor",
    with_plot: Annotated[bool, Field(description="Si True, también genera PNG.")] = False,
) -> dict:
    """Genera una serie temporal con patrones cíclicos repetidos.

    USA cuando el usuario mencione estacionalidad observable (semanal, mensual, anual),
    patrones que se repiten cada N observaciones, o simulación de demanda con ciclos.

    Pasa `periods` O `end_date`, no ambos.

    Devuelve: output_path, rows_generated, image_path, summary.
    """
    try:
        inp = GeneratePeriodicInput(
            start_date=start_date, end_date=end_date, periods=periods,
            frequency=frequency, distribution_type=distribution_type,
            distribution_params=distribution_params, period_length=period_length,
            pattern_type=pattern_type, column_name=column_name, with_plot=with_plot,
        )
        suffix, horizon = _resolve_horizon(inp.end_date, inp.periods)
        endpoint = f"/Datos/periodicos/{suffix}"
        params = {
            "inicio": inp.start_date,
            "freq": inp.frequency,
            "distr": inp.distribution_type,
            "p": inp.period_length,
            "tipo": inp.pattern_type,
            "columna": inp.column_name,
            "params": inp.distribution_params,
            **horizon,
        }
        out_name = deterministic_filename(
            "periodicos",
            inp.start_date, str(horizon), inp.frequency,
            str(inp.distribution_type), str(inp.distribution_params),
            str(inp.period_length), str(inp.pattern_type),
            ext="csv",
        )

        async with get_client(_SETTINGS) as client:
            csv_path = await _download_csv(client, endpoint, params, out_name)
            png_path: Optional[Path] = None
            if inp.with_plot:
                png_name = out_name.replace(".csv", ".png")
                png_path = await _download_png(client, f"/Plot/periodicos/{suffix}", params, png_name)

        rows = _row_count(csv_path)
        return {
            "output_path": str(csv_path),
            "rows_generated": rows,
            "image_path": str(png_path) if png_path else None,
            "summary": (
                f"Serie periódica generada ({rows} filas, period_length={inp.period_length}, "
                f"pattern_type={inp.pattern_type})."
            ),
        }
    except Exception as exc:  # noqa: BLE001
        return {"error": translate_exception(exc, "generate_synthetic_periodic")}


# ───────────────────────── 4. generate_synthetic_trend ─────────────────────────

@mcp.tool()
async def generate_synthetic_trend(
    start_date: Annotated[str, Field(description="Fecha de inicio 'YYYY-MM-DD'.")],
    frequency: Annotated[_FREQ, Field(description="Frecuencia temporal.")],
    trend_type: Annotated[int, Field(ge=1, description="Código del tipo de tendencia (lineal, polinómica, exponencial...).")],
    trend_params: Annotated[list[float], Field(description="Coeficientes que definen la tendencia.")],
    end_date: Annotated[Optional[str], Field(description="Fecha de fin (excluyente con periods).")] = None,
    periods: Annotated[Optional[int], Field(description="Número de periodos (excluyente con end_date).")] = None,
    column_name: Annotated[str, Field(description="Nombre de la columna.")] = "valor",
    noise: Annotated[float, Field(description="Magnitud del ruido aditivo gaussiano.")] = 0.0,
    with_plot: Annotated[bool, Field(description="Si True, también genera PNG.")] = False,
) -> dict:
    """Genera una serie temporal con tendencia determinista.

    USA cuando el usuario quiera datos con crecimiento o decrecimiento sistemático,
    o simulación de procesos no estacionarios con tendencia conocida.

    Pasa `periods` O `end_date`, no ambos.

    Devuelve: output_path, rows_generated, image_path, summary.
    """
    try:
        inp = GenerateTrendInput(
            start_date=start_date, end_date=end_date, periods=periods,
            frequency=frequency, column_name=column_name, trend_type=trend_type,
            trend_params=trend_params, noise=noise, with_plot=with_plot,
        )
        suffix, horizon = _resolve_horizon(inp.end_date, inp.periods)
        endpoint = f"/Datos/tendencia/{suffix}"
        params = {
            "inicio": inp.start_date,
            "freq": inp.frequency,
            "tipo": inp.trend_type,
            "error": inp.noise,
            "columna": inp.column_name,
            "params": inp.trend_params,
            **horizon,
        }
        out_name = deterministic_filename(
            "tendencia",
            inp.start_date, str(horizon), inp.frequency,
            str(inp.trend_type), str(inp.trend_params), str(inp.noise),
            ext="csv",
        )

        async with get_client(_SETTINGS) as client:
            csv_path = await _download_csv(client, endpoint, params, out_name)
            png_path: Optional[Path] = None
            if inp.with_plot:
                png_name = out_name.replace(".csv", ".png")
                png_path = await _download_png(client, f"/Plot/tendencia/{suffix}", params, png_name)

        rows = _row_count(csv_path)
        return {
            "output_path": str(csv_path),
            "rows_generated": rows,
            "image_path": str(png_path) if png_path else None,
            "summary": (
                f"Serie con tendencia generada ({rows} filas, trend_type={inp.trend_type})."
            ),
        }
    except Exception as exc:  # noqa: BLE001
        return {"error": translate_exception(exc, "generate_synthetic_trend")}

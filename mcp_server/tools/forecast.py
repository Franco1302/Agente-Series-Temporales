"""Tool MCP: predicción de series temporales (SARIMAX, Prophet, ForecasterAutoreg)."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated, Literal, Optional

from pydantic import BaseModel, Field

from mcp_server.config import load_settings
from mcp_server.errors import translate_exception
from mcp_server.file_utils import deterministic_filename, open_csv_for_upload
from mcp_server.http_client import get_client
from mcp_server.instance import mcp

_SETTINGS = load_settings()

_MODEL_TO_DATOS: dict[str, str] = {
    "sarimax": "/Datos/Sarimax",
    "prophet": "/Datos/Prophet",
    "forecaster_autoreg": "/Datos/ForecasterAutoreg",
}

_MODEL_TO_ERROR: dict[str, str] = {
    "sarimax": "/Error/Sarimax",
    "prophet": "/Error/Prophet",
    "forecaster_autoreg": "/Error/ForecasterAutoreg",
}


class ForecastTimeSeriesInput(BaseModel):
    file_path: str
    index_column: str
    target_column: str
    model: Literal["sarimax", "prophet", "forecaster_autoreg"]
    forecast_steps: int
    frequency: Literal["B", "D", "W", "M", "Q", "Y", "h", "min", "s"] = "D"
    regressor: Optional[str] = None
    return_metrics: bool = True
    with_plot: bool = False


def _datos_params(inp: ForecastTimeSeriesInput) -> dict:
    base: dict = {"indice": inp.index_column, "freq": inp.frequency, "size": inp.forecast_steps}
    if inp.model == "forecaster_autoreg":
        base["regresor"] = inp.regressor or "RandomForestRegressor"
    return base


def _error_params(inp: ForecastTimeSeriesInput) -> dict:
    base: dict = {"indice": inp.index_column, "freq": inp.frequency}
    if inp.model == "forecaster_autoreg":
        base["regresor"] = inp.regressor or "RandomForestRegressor"
    return base


@mcp.tool()
async def forecast_time_series(
    file_path: Annotated[str, Field(description="Ruta local al CSV con la serie histórica.")],
    index_column: Annotated[str, Field(description="Columna índice del CSV.")],
    target_column: Annotated[str, Field(description="Columna a predecir.")],
    model: Annotated[
        Literal["sarimax", "prophet", "forecaster_autoreg"],
        Field(
            description=(
                "'sarimax' = estadístico clásico (bueno con estacionalidad); "
                "'prophet' = modelo Facebook (robusto a huecos y festivos); "
                "'forecaster_autoreg' = skforecast con regresor autorregresivo."
            ),
        ),
    ],
    forecast_steps: Annotated[int, Field(gt=0, description="Número de pasos a predecir.")],
    frequency: Annotated[
        Literal["B", "D", "W", "M", "Q", "Y", "h", "min", "s"],
        Field(description="Frecuencia temporal de la serie (debe coincidir con el CSV)."),
    ] = "D",
    regressor: Annotated[
        Optional[str],
        Field(description="Solo forecaster_autoreg: nombre del regresor (default RandomForestRegressor)."),
    ] = None,
    return_metrics: Annotated[bool, Field(description="Si True, también devuelve métricas de error.")] = True,
    with_plot: Annotated[bool, Field(description="Si True, también genera PNG con la predicción.")] = False,
) -> dict:
    """Entrena un modelo predictivo y devuelve un horizonte de predicción.

    USA cuando el usuario quiera estimar el comportamiento futuro de una serie,
    comparar modelos predictivos o evaluar el error de un modelo concreto.

    Devuelve: output_path, metrics (si return_metrics), model_used, image_path, summary.
    """
    try:
        inp = ForecastTimeSeriesInput(
            file_path=file_path, index_column=index_column,
            target_column=target_column, model=model,
            forecast_steps=forecast_steps, frequency=frequency,
            regressor=regressor, return_metrics=return_metrics, with_plot=with_plot,
        )
        datos_endpoint = _MODEL_TO_DATOS[inp.model]
        error_endpoint = _MODEL_TO_ERROR[inp.model]
        datos_params = _datos_params(inp)
        filename, content, mime = open_csv_for_upload(inp.file_path)

        out_name = deterministic_filename(
            f"forecast_{inp.model}",
            Path(inp.file_path).stem, inp.index_column, inp.target_column,
            str(inp.forecast_steps), inp.frequency, str(inp.regressor),
            ext="csv",
        )

        metrics: Optional[dict] = None
        png_path: Optional[Path] = None

        async with get_client(_SETTINGS) as client:
            response = await client.post(
                datos_endpoint,
                params=datos_params,
                files={"file": (filename, content, mime)},
            )
            response.raise_for_status()
            target = _SETTINGS.workspace_dir / out_name
            target.write_bytes(response.content)

            if inp.return_metrics:
                err_response = await client.post(
                    error_endpoint,
                    params=_error_params(inp),
                    files={"file": (filename, content, mime)},
                )
                err_response.raise_for_status()
                try:
                    metrics = err_response.json()
                except Exception:
                    metrics = {"raw": err_response.text[:500]}

            if inp.with_plot:
                plot_response = await client.post(
                    f"/Plot{datos_endpoint}",
                    params=datos_params,
                    files={"file": (filename, content, mime)},
                )
                plot_response.raise_for_status()
                png_path = _SETTINGS.workspace_dir / out_name.replace(".csv", ".png")
                png_path.write_bytes(plot_response.content)

        return {
            "output_path": str(target),
            "metrics": metrics,
            "model_used": inp.model,
            "image_path": str(png_path) if png_path else None,
            "summary": (
                f"Predicción de {inp.forecast_steps} pasos generada con {inp.model} "
                f"para columna '{inp.target_column}'."
            ),
        }
    except Exception as exc:  # noqa: BLE001
        return {"error": translate_exception(exc, "forecast_time_series")}

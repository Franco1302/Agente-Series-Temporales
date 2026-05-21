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
from mcp_server.observability.http_hooks import init_mcp_http_log, attach_observability
_SETTINGS = load_settings()

_MODEL_TO_DATOS: dict[str, str] = {
    # Endpoints MCP que generan el CSV de prediccion segun el modelo.
    "sarimax": "/Datos/Sarimax",
    "prophet": "/Datos/Prophet",
    "forecaster_autoreg": "/Datos/ForecasterAutoreg",
}

_MODEL_TO_ERROR: dict[str, str] = {
    # Endpoints MCP que devuelven metricas de error del modelo.
    "sarimax": "/Error/Sarimax",
    "prophet": "/Error/Prophet",
    "forecaster_autoreg": "/Error/ForecasterAutoreg",
}


class ForecastTimeSeriesInput(BaseModel):
    """Esquema de entrada normalizado para la prediccion de series."""
    file_path: str
    index_column: str
    target_column: str
    model: Literal["sarimax", "prophet", "forecaster_autoreg"]
    forecast_steps: int
    frequency: Literal["B", "D", "W", "M", "Q", "Y", "h", "min", "s"] = "D"
    regressor: Optional[str] = None
    return_metrics: bool = True
    with_plot: bool = True


def _datos_params(inp: ForecastTimeSeriesInput) -> dict:
    """Construye los parametros para el endpoint /Datos segun el modelo.

    - Siempre envia indice, frecuencia y tamaño del horizonte.
    - Para forecaster_autoreg incluye el nombre del regresor.
    """
    base: dict = {"indice": inp.index_column, "freq": inp.frequency, "size": inp.forecast_steps}
    if inp.model == "forecaster_autoreg":
        base["regresor"] = inp.regressor or "RandomForestRegressor"
    return base


def _error_params(inp: ForecastTimeSeriesInput) -> dict:
    """Construye los parametros para el endpoint /Error segun el modelo.

    - No incluye el horizonte, solo metadatos del indice.
    - Para forecaster_autoreg incluye el nombre del regresor.
    """
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
    with_plot: Annotated[bool, Field(description="Si True, también genera PNG con la predicción.")] = True,
) -> dict:
    """Entrena un modelo predictivo y devuelve un horizonte de prediccion.

    USA cuando el usuario quiera estimar el comportamiento futuro de una serie,
    comparar modelos predictivos o evaluar el error de un modelo concreto.

    Flujo MCP:
    1) Normaliza la entrada con Pydantic.
    2) Selecciona endpoints /Datos y /Error segun el modelo.
    3) Sube el CSV como multipart y guarda el CSV de prediccion.
    4) Si `return_metrics=True`, solicita el endpoint de error y parsea JSON.
    5) Si `with_plot=True`, solicita /Plot{datos_endpoint} y guarda el PNG.

    Devuelve: output_path, metrics (si return_metrics), model_used, image_path, summary.
    """
    init_mcp_http_log()
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
                # Best-effort: el CSV ya es válido; un fallo de /Plot no debe abortar la tool.
                try:
                    plot_response = await client.post(
                        f"/Plot{datos_endpoint}",
                        params=datos_params,
                        files={"file": (filename, content, mime)},
                    )
                    plot_response.raise_for_status()
                    png_path = _SETTINGS.workspace_dir / out_name.replace(".csv", ".png")
                    png_path.write_bytes(plot_response.content)
                except Exception:  # noqa: BLE001 — la gráfica es opcional
                    png_path = None

        result = {
            "output_path": str(target),
            "metrics": metrics,
            "model_used": inp.model,
            "image_path": str(png_path) if png_path else None,
            "summary": (
                f"Predicción de {inp.forecast_steps} pasos generada con {inp.model} "
                f"para columna '{inp.target_column}'."
            ),
        }
        return attach_observability(result)
    except Exception as exc:  # noqa: BLE001
        
        error_result = {"error": translate_exception(exc, "forecast_time_series")}
        return attach_observability(error_result)

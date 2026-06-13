"""Tool MCP: predicción de series temporales (SARIMAX)."""

from __future__ import annotations

import io
from pathlib import Path
from typing import Annotated, Literal, Optional

import pandas as pd
from pydantic import BaseModel, Field

from mcp_server.config import load_settings
from mcp_server.errors import translate_exception
from mcp_server.file_utils import deterministic_filename, open_csv_for_upload
from mcp_server.http_client import get_client
from mcp_server.instance import mcp
from mcp_server.observability.http_hooks import init_mcp_http_log, attach_observability
_SETTINGS = load_settings()

_DATOS_ENDPOINT = "/Datos/Sarimax"
_ERROR_ENDPOINT = "/Error/Sarimax"

# Frecuencias que acepta el backend /Datos/Sarimax (un solo caracter).
_BACKEND_FREQS: tuple[str, ...] = ("B", "D", "W", "M", "Q", "Y")
# Alias 'period start' que devuelve pandas (MS, QS, YS o el legacy AS) y que el
# backend NO acepta: requieren reescribir las fechas al final del periodo.
_PERIOD_START_ALIASES: tuple[str, ...] = ("MS", "QS", "YS", "AS")


def _normalize_csv_for_backend(
    content: bytes, index_column: str,
) -> tuple[bytes, Optional[str]]:
    """Adapta el CSV al contrato del backend Sarimax."""
    try:
        df = pd.read_csv(io.BytesIO(content))
        if index_column not in df.columns:
            return content, None
        idx_dt = pd.to_datetime(df[index_column], errors="coerce")
        if idx_dt.isna().any() or len(idx_dt) < 3:
            return content, None
        freq = pd.infer_freq(idx_dt)
        if not freq:
            return content, None
        # Legacy 'A' (annual, pandas <2) lo trato como 'Y'.
        head = "Y" if freq[0].upper() == "A" else freq[0].upper()
        if head not in _BACKEND_FREQS:
            return content, None
        if freq.upper() in _PERIOD_START_ALIASES and head in ("M", "Q", "Y"):
            shifted = idx_dt.dt.to_period(head).dt.end_time.dt.normalize()
            df[index_column] = shifted.dt.strftime("%Y-%m-%d")
            buf = io.BytesIO()
            df.to_csv(buf, index=False)
            return buf.getvalue(), head
        return content, head
    except Exception:  # noqa: BLE001
        return content, None


class ForecastTimeSeriesInput(BaseModel):
    """Esquema de entrada normalizado para la prediccion de series."""
    file_path: str
    index_column: str
    model: Literal["sarimax"] = "sarimax"
    forecast_steps: int
    frequency: Literal["B", "D", "W", "M", "Q", "Y", "h", "min", "s"] = "D"
    return_metrics: bool = True
    with_plot: bool = True


def _datos_params(inp: ForecastTimeSeriesInput) -> dict:
    """Parametros para /Datos/Sarimax: indice, frecuencia y horizonte."""
    return {"indice": inp.index_column, "freq": inp.frequency, "size": inp.forecast_steps}


def _error_params(inp: ForecastTimeSeriesInput) -> dict:
    """Parametros para /Error/Sarimax: solo indice y frecuencia."""
    return {"indice": inp.index_column, "freq": inp.frequency}


@mcp.tool()
async def forecast_time_series(
    file_path: Annotated[str, Field(description="Ruta local al CSV con la serie histórica.")],
    index_column: Annotated[
        str,
        Field(
            description="Columna índice del CSV.",
            json_schema_extra={"evidence": "existing_column"},
        ),
    ],
    forecast_steps: Annotated[
        int,
        Field(
            gt=0,
            description="Número de pasos a predecir.",
            json_schema_extra={"evidence": "integer"},
        ),
    ],
    frequency: Annotated[
        Literal["B", "D", "W", "M", "Q", "Y", "h", "min", "s"],
        Field(
            description="Frecuencia temporal de la serie (debe coincidir con el CSV).",
            json_schema_extra={"tunable": True},
        ),
    ] = "D",
    model: Annotated[
        Literal["sarimax"],
        Field(
            description="Modelo de predicción. Actualmente solo 'sarimax' está operativo.",
            json_schema_extra={"tunable": True},
        ),
    ] = "sarimax",
    return_metrics: Annotated[bool, Field(description="Si True, también devuelve métricas de error.")] = True,
    with_plot: Annotated[
        bool,
        Field(
            description="Si True, también genera PNG con la predicción.",
            json_schema_extra={"evidence": "plot_pref"},
        ),
    ] = True,
) -> dict:
    """Entrena un modelo predictivo y devuelve un horizonte de prediccion.

    USA cuando el usuario quiera estimar el comportamiento futuro de una serie,
    comparar modelos predictivos o evaluar el error de un modelo concreto. El
    modelo sarimax predice TODAS las columnas numéricas del CSV, no una columna
    objetivo concreta.
    """
    init_mcp_http_log()
    try:
        inp = ForecastTimeSeriesInput(
            file_path=file_path, index_column=index_column, model=model,
            forecast_steps=forecast_steps, frequency=frequency,
            return_metrics=return_metrics, with_plot=with_plot,
        )
        datos_endpoint = _DATOS_ENDPOINT
        error_endpoint = _ERROR_ENDPOINT
        filename, content, mime = open_csv_for_upload(inp.file_path)

        content, inferred = _normalize_csv_for_backend(content, inp.index_column)
        if inferred and inferred != inp.frequency:
            inp = inp.model_copy(update={"frequency": inferred})

        datos_params = _datos_params(inp)

        out_name = deterministic_filename(
            f"forecast_{inp.model}",
            Path(inp.file_path).stem, inp.index_column,
            str(inp.forecast_steps), inp.frequency,
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
                try:
                    err_response = await client.post(
                        error_endpoint,
                        params=_error_params(inp),
                        files={"file": (filename, content, mime)},
                    )
                    err_response.raise_for_status()
                    try:
                        metrics = err_response.json()
                    except Exception:  # noqa: BLE001
                        metrics = {"raw": err_response.text[:500]}
                except Exception:  # noqa: BLE001
                    metrics = None

            if inp.with_plot:
                try:
                    plot_response = await client.post(
                        f"/Plot{datos_endpoint}",
                        params=datos_params,
                        files={"file": (filename, content, mime)},
                    )
                    plot_response.raise_for_status()
                    png_path = _SETTINGS.workspace_dir / out_name.replace(".csv", ".png")
                    png_path.write_bytes(plot_response.content)
                except Exception:  # noqa: BLE001
                    png_path = None

        result = {
            "output_path": str(target),
            "metrics": metrics,
            "model_used": inp.model,
            "image_path": str(png_path) if png_path else None,
            "summary": (
                f"Predicción de {inp.forecast_steps} pasos generada con {inp.model} "
                f"para todas las columnas de la serie."
            ),
        }
        return attach_observability(result)
    except Exception as exc:  # noqa: BLE001
        
        error_result = {"error": translate_exception(exc, "forecast_time_series")}
        return attach_observability(error_result)

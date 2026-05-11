"""Tool MCP: aumentación de series temporales sobre un CSV existente."""

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

_STRATEGY_TO_ENDPOINT: dict[str, str] = {
    "normal": "/Aumentar/Normal",
    "muller": "/Aumentar/Muller",
    "duplicate": "/Aumentar/Duplicado",
    "harmonic": "/Aumentar/Armonico",
    "statistical": "/Aumentar/Estadistica",
}


class AugmentTimeSeriesInput(BaseModel):
    file_path: str
    index_column: str
    strategy: Literal["normal", "muller", "duplicate", "harmonic", "statistical"]
    size: int
    frequency: Literal["B", "D", "W", "M", "Q", "Y", "h", "min", "s"]
    duplication_factor: Optional[float] = None
    perturbation_std: Optional[float] = None
    statistical_type: Optional[int] = None
    with_plot: bool = False


def _build_query_params(inp: AugmentTimeSeriesInput) -> dict:
    """Mapea el schema a los params reales del endpoint según la estrategia."""
    base = {"indice": inp.index_column, "freq": inp.frequency}
    if inp.strategy in {"normal", "muller", "harmonic"}:
        return {**base, "size": inp.size}
    if inp.strategy == "duplicate":
        return {
            **base,
            "duplication_factor": inp.duplication_factor if inp.duplication_factor is not None else 0.5,
            "perturbation_std": inp.perturbation_std if inp.perturbation_std is not None else 0.1,
        }
    if inp.strategy == "statistical":
        return {
            **base,
            "tipo": inp.statistical_type if inp.statistical_type is not None else 1,
            "num": inp.size,
        }
    raise ValueError(f"Estrategia desconocida: {inp.strategy}")


@mcp.tool()
async def augment_time_series(
    file_path: Annotated[str, Field(description="Ruta local al CSV a aumentar.")],
    index_column: Annotated[str, Field(description="Nombre de la columna índice del CSV.")],
    strategy: Annotated[
        Literal["normal", "muller", "duplicate", "harmonic", "statistical"],
        Field(
            description=(
                "Técnica: 'normal' = muestreo Normal con media/desv del CSV; "
                "'muller' = Box-Muller; 'duplicate' = duplica con ruido; "
                "'harmonic' = añade ruido armónico; 'statistical' = muestreo "
                "basado en estadísticos descriptivos."
            ),
        ),
    ],
    size: Annotated[int, Field(gt=0, description="Número de observaciones nuevas a generar.")],
    frequency: Annotated[
        Literal["B", "D", "W", "M", "Q", "Y", "h", "min", "s"],
        Field(description="Frecuencia temporal de los datos generados."),
    ],
    duplication_factor: Annotated[
        Optional[float],
        Field(description="Solo strategy='duplicate': proporción duplicada (default 0.5)."),
    ] = None,
    perturbation_std: Annotated[
        Optional[float],
        Field(description="Solo strategy='duplicate': desviación del ruido (default 0.1)."),
    ] = None,
    statistical_type: Annotated[
        Optional[int],
        Field(description="Solo strategy='statistical': tipo de estadístico (default 1)."),
    ] = None,
    with_plot: Annotated[bool, Field(description="Si True, también genera PNG.")] = False,
) -> dict:
    """Genera observaciones adicionales para un CSV existente preservando sus estadísticos.

    USA cuando el usuario tenga pocos datos y necesite ampliar el dataset para
    entrenar un modelo predictivo de forma más estable.

    Devuelve: output_path, new_rows, strategy_used, image_path, summary.
    """
    try:
        inp = AugmentTimeSeriesInput(
            file_path=file_path, index_column=index_column, strategy=strategy,
            size=size, frequency=frequency,
            duplication_factor=duplication_factor, perturbation_std=perturbation_std,
            statistical_type=statistical_type, with_plot=with_plot,
        )
        endpoint = _STRATEGY_TO_ENDPOINT[inp.strategy]
        params = _build_query_params(inp)
        filename, content, mime = open_csv_for_upload(inp.file_path)

        out_name = deterministic_filename(
            f"augment_{inp.strategy}",
            Path(inp.file_path).stem, inp.index_column, str(inp.size), inp.frequency,
            ext="csv",
        )

        async with get_client(_SETTINGS) as client:
            response = await client.post(
                endpoint,
                params=params,
                files={"file": (filename, content, mime)},
            )
            response.raise_for_status()
            target = _SETTINGS.workspace_dir / out_name
            target.write_bytes(response.content)

            png_path: Optional[Path] = None
            if inp.with_plot:
                plot_response = await client.post(
                    f"/Plot{endpoint}",
                    params=params,
                    files={"file": (filename, content, mime)},
                )
                plot_response.raise_for_status()
                png_path = _SETTINGS.workspace_dir / out_name.replace(".csv", ".png")
                png_path.write_bytes(plot_response.content)

        return {
            "output_path": str(target),
            "new_rows": inp.size,
            "strategy_used": inp.strategy,
            "image_path": str(png_path) if png_path else None,
            "summary": (
                f"CSV aumentado con estrategia '{inp.strategy}' (+{inp.size} filas)."
            ),
        }
    except Exception as exc:  # noqa: BLE001
        return {"error": translate_exception(exc, "augment_time_series")}

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
from mcp_server.observability.http_hooks import init_mcp_http_log, attach_observability

_SETTINGS = load_settings()

_STRATEGY_TO_ENDPOINT: dict[str, str] = {
    "normal": "/Aumentar/Normal",
    "muller": "/Aumentar/Muller",
    "duplicate": "/Aumentar/Duplicado",
    "harmonic": "/Aumentar/Armonico",
    "statistical": "/Aumentar/Estadistica",
}


class AugmentTimeSeriesInput(BaseModel):
    """Esquema de entrada normalizado para la herramienta de aumentacion."""
    file_path: str
    index_column: str
    strategy: Literal["normal", "muller", "duplicate", "harmonic", "statistical"]
    size: int
    frequency: Literal["B", "D", "W", "M", "Q", "Y", "h", "min", "s"]
    duplication_factor: Optional[float] = None
    perturbation_std: Optional[float] = None
    statistical_type: Optional[int] = None
    with_plot: bool = True


def _build_query_params(inp: AugmentTimeSeriesInput) -> dict:
    """Construye los parametros esperados por el backend segun la estrategia.

    - Siempre incluye el indice y la frecuencia para alinear el eje temporal.
    - Algunas estrategias reciben `size` directamente (normal, muller, harmonic).
    - Otras usan nombres distintos (duplicate: factores de ruido; statistical: tipo).
    """
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
    with_plot: Annotated[bool, Field(description="Si True, también genera PNG.")] = True,
) -> dict:
    """Genera observaciones adicionales para un CSV existente preservando estadisticos.

    Flujo principal:
    1) Normaliza y valida los parametros con un schema Pydantic.
    2) Resuelve el endpoint segun la estrategia y arma los query params.
    3) Abre el CSV, lo sube al backend y guarda la respuesta como un nuevo CSV.
    4) Si `with_plot=True`, solicita el PNG asociado y lo persiste en disco.

    Devuelve un dict con:
    - output_path: ruta del CSV generado.
    - new_rows: cantidad de filas nuevas solicitadas.
    - strategy_used: estrategia aplicada.
    - image_path: ruta del PNG (si se genero).
    - summary: descripcion breve del resultado.
    """
    init_mcp_http_log()
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
                # Best-effort: el CSV ya es válido; un fallo de /Plot no debe abortar la tool.
                try:
                    plot_response = await client.post(
                        f"/Plot{endpoint}",
                        params=params,
                        files={"file": (filename, content, mime)},
                    )
                    plot_response.raise_for_status()
                    png_path = _SETTINGS.workspace_dir / out_name.replace(".csv", ".png")
                    png_path.write_bytes(plot_response.content)
                except Exception:  # noqa: BLE001 — la gráfica es opcional
                    png_path = None
        result = {
                "output_path": str(target),
                "new_rows": inp.size,
                "strategy_used": inp.strategy,
                "image_path": str(png_path) if png_path else None,
                "summary": (
                    f"CSV aumentado con estrategia '{inp.strategy}' (+{inp.size} filas)."
                ),
        }
        return attach_observability(result)
    
    except Exception as exc:  # noqa: BLE001
        error_result = {"error": translate_exception(exc, "augment_time_series")}
        return attach_observability(error_result)
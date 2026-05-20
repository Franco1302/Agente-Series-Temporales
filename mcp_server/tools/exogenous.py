"""Tool MCP: creación de variables exógenas a partir de un CSV multivariante."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated, Literal, Optional

from pydantic import BaseModel, Field

from mcp_server.config import load_settings
from mcp_server.errors import translate_exception
from mcp_server.file_utils import deterministic_filename, open_csv_for_upload
from mcp_server.http_client import get_client
from mcp_server.instance import mcp
from mcp_server.observability.http_hooks import init_http_log, attach_observability
_SETTINGS = load_settings()

_RELATION_TO_ENDPOINT: dict[str, str] = {
    # Mapea el tipo de relacion solicitado al endpoint del backend MCP.
    # El backend expone rutas bajo /Variables/* y cada una implementa
    # una tecnica distinta para crear la variable exogena.
    "pca": "/Variables/PCA",
    "correlation": "/Variables/Correlacion",
    "covariance": "/Variables/Covarianza",
    "linear": "/Variables/Lineal",
    "polynomial": "/Variables/Polinomico",
}


class CreateExogenousVariableInput(BaseModel):
    """Esquema de entrada normalizado para crear una variable exogena."""
    file_path: str
    index_column: str
    new_column_name: str
    relation: Literal["pca", "correlation", "covariance", "linear", "polynomial"]
    coefficients: Optional[list[float]] = None
    with_plot: bool = False


def _build_query_params(inp: CreateExogenousVariableInput) -> dict:
    """Construye los parametros que espera el backend segun la relacion.

    - Base comun: indice y nombre de la nueva columna.
    - Linear: requiere slope/intercept como `a` y `b`.
    - Polynomial: lista de coeficientes en `a`.
    - PCA/correlation/covariance: solo base comun.
    """
    base: dict = {"indice": inp.index_column, "columna": inp.new_column_name}
    if inp.relation == "linear":
        coefs = inp.coefficients or []
        if len(coefs) < 2:
            raise ValueError("relation='linear' requiere coefficients=[slope, intercept].")
        return {**base, "a": coefs[0], "b": coefs[1]}
    if inp.relation == "polynomial":
        coefs = inp.coefficients or []
        if not coefs:
            raise ValueError("relation='polynomial' requiere coefficients=[c0,c1,...].")
        return {**base, "a": coefs}
    # pca, correlation, covariance: solo indice + columna
    return base


@mcp.tool()
async def create_exogenous_variable(
    file_path: Annotated[str, Field(description="Ruta local al CSV multivariante.")],
    index_column: Annotated[str, Field(description="Columna índice del CSV.")],
    new_column_name: Annotated[str, Field(description="Nombre de la columna a añadir.")],
    relation: Annotated[
        Literal["pca", "correlation", "covariance", "linear", "polynomial"],
        Field(
            description=(
                "Tipo de relación: 'pca' = primera componente principal; "
                "'correlation' = matriz de correlación; 'covariance' = covarianza; "
                "'linear' = y=a·x+b (requiere coefficients=[slope, intercept]); "
                "'polynomial' = combinación polinómica (requiere coefficients=[c0,c1,...])."
            ),
        ),
    ],
    coefficients: Annotated[
        Optional[list[float]],
        Field(description="Solo linear/polynomial: lista de coeficientes."),
    ] = None,
    with_plot: Annotated[bool, Field(description="Si True, también genera PNG.")] = False,
) -> dict:
    """Añade una nueva columna sintetica al CSV calculada a partir de las existentes.

    Flujo real del MCP para esta herramienta:
    1) Se valida y normaliza la entrada con Pydantic.
    2) Se resuelve el endpoint MCP segun la relacion solicitada.
    3) Se sube el CSV como multipart al backend con los parametros adecuados.
    4) Se guarda la respuesta (CSV) en el workspace.
    5) Si `with_plot=True`, se llama a /Plot{endpoint} y se persiste el PNG.

    Usa esta herramienta cuando el usuario quiera enriquecer un dataset con
    variables derivadas para mejorar el rendimiento de un modelo predictivo
    (por ejemplo, SARIMAX).

    Devuelve un dict con:
    - output_path: ruta del CSV generado.
    - new_column_name: nombre de la columna creada.
    - relation_used: relacion aplicada.
    - image_path: ruta del PNG (si se genero).
    - summary: descripcion breve del resultado.
    """
    init_http_log()
    try:
        inp = CreateExogenousVariableInput(
            file_path=file_path, index_column=index_column,
            new_column_name=new_column_name, relation=relation,
            coefficients=coefficients, with_plot=with_plot,
        )
        endpoint = _RELATION_TO_ENDPOINT[inp.relation]
        params = _build_query_params(inp)
        filename, content, mime = open_csv_for_upload(inp.file_path)

        out_name = deterministic_filename(
            f"exogenous_{inp.relation}",
            Path(inp.file_path).stem, inp.index_column, inp.new_column_name,
            str(inp.coefficients),
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
        result = {
            "output_path": str(target),
            "new_column_name": inp.new_column_name,
            "relation_used": inp.relation,
            "image_path": str(png_path) if png_path else None,
            "summary": (
                f"Columna '{inp.new_column_name}' añadida usando relación '{inp.relation}'."
            ),
        }        
        return attach_observability(result)
    
    except Exception as exc:  # noqa: BLE001
        error_result = {"error": translate_exception(exc, "create_exogenous_variable")}
        return attach_observability(error_result)

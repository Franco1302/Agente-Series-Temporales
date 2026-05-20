"""Tool MCP: detección de drift sobre un CSV."""

from __future__ import annotations

import io
from typing import Annotated, Literal, Optional

import pandas as pd
from pydantic import BaseModel, Field

from mcp_server.config import load_settings
from mcp_server.errors import translate_exception
from mcp_server.file_utils import open_csv_for_upload
from mcp_server.http_client import get_client
from mcp_server.instance import mcp
from mcp_server.observability.http_hooks import init_mcp_http_log, attach_observability
_SETTINGS = load_settings()

_METHOD_TO_ENDPOINT: dict[str, str] = {
    "KS": "/Deteccion/KS",
    "JS": "/Deteccion/JS",
    "PSI": "/Deteccion/PSI",
    "CUSUM": "/Deteccion/CUSUM",
    "MEWMA": "/Deteccion/MEWMA",
    "HOTELLING": "/Deteccion/HOTELLING",
}

_DEFAULT_THRESHOLDS: dict[str, float] = {"KS": 0.05, "JS": 0.2, "PSI": 0.25, "CUSUM": 1.5}


# Schema interno usado por _build_query_params y los tests unitarios. NO se expone al LLM
# (la tool MCP recibe params planos para que el JSON Schema sea plano).
class DetectDriftInput(BaseModel):
    file_path: str
    index_column: str
    method: Literal["KS", "JS", "PSI", "CUSUM", "MEWMA", "HOTELLING"]
    inicio: int = 1
    threshold: Optional[float] = None
    num_bins: Optional[int] = None
    drift_cusum: Optional[float] = None
    min_instances: Optional[int] = None
    lambd: Optional[float] = None
    alpha: Optional[float] = None


def _build_query_params(inp: DetectDriftInput) -> dict:
    """Traduce el schema Pydantic a los query params específicos del endpoint."""
    params: dict[str, object] = {"indice": inp.index_column, "inicio": inp.inicio}
    threshold = inp.threshold if inp.threshold is not None else _DEFAULT_THRESHOLDS.get(inp.method)

    if inp.method == "KS":
        params["threshold_ks"] = threshold
    elif inp.method == "JS":
        params["threshold_js"] = threshold
    elif inp.method == "PSI":
        params["threshold_psi"] = threshold
        params["num_bins"] = inp.num_bins if inp.num_bins is not None else 10
    elif inp.method == "CUSUM":
        params["threshold_cusum"] = threshold
        params["drift_cusum"] = inp.drift_cusum if inp.drift_cusum is not None else 0.5
    elif inp.method in {"MEWMA", "HOTELLING"}:
        params["min_instances"] = inp.min_instances if inp.min_instances is not None else 100
        # alpha es el nivel de significación del límite de control: 0 es
        # degenerado (puede producir un w*S singular en la API). Default 0.05.
        params["alpha"] = inp.alpha if inp.alpha is not None else 0.05
        if inp.method == "MEWMA":
            params["lambd"] = inp.lambd if inp.lambd is not None else 0.5
    return params


def _validate_multivariate(content: bytes, index_column: str) -> Optional[str]:
    """Valida que el CSV es apto para un método multivariante (MEWMA/HOTELLING).

    Estos métodos estiman e invierten una matriz de covarianza, así que
    necesitan al menos dos columnas numéricas con varianza no nula. Si el
    dataset no cumple, la API responde con un 500 opaco
    (`numpy.linalg.LinAlgError: Singular matrix`); este chequeo previo lo
    convierte en un mensaje accionable que el agente ReAct puede usar
    para reintentar con otro método.

    Parámetros:
        content: Bytes del CSV (los mismos que se envían a la API).
        index_column: Columna índice, que se excluye de las features.

    Retorno:
        Un mensaje de error si el dataset no es apto, o ``None`` si lo es.
    """
    try:
        df = pd.read_csv(io.BytesIO(content))
    except Exception:  # noqa: BLE001
        # No bloqueamos por un fallo de parseo: que sea la API quien decida.
        return None

    features = df.drop(columns=[index_column], errors="ignore")
    numeric = features.select_dtypes(include="number")

    if numeric.shape[1] < 2:
        return (
            f"El método multivariante requiere al menos 2 columnas numéricas y "
            f"el dataset solo tiene {numeric.shape[1]}. Para una serie de una "
            "sola variable usa un método univariante: KS, JS, PSI o CUSUM."
        )

    constantes = [c for c in numeric.columns if numeric[c].nunique(dropna=True) <= 1]
    if constantes:
        return (
            f"Las columnas {constantes} son constantes (varianza nula): la matriz "
            "de covarianza sería singular y el método multivariante fallaría. "
            "Elimínalas del dataset o usa un método univariante."
        )

    return None


def _build_summary(method: str, label: str, report: dict) -> str:
    if label == "Detectado":
        cols = [k for k, v in report.items() if isinstance(v, dict) and v.get("drift")]
        cols_str = ", ".join(cols) if cols else "todas las columnas analizadas"
        return f"Se detectó drift mediante {method} en: {cols_str}."
    return f"No se detectó drift mediante {method} con los umbrales aplicados."


@mcp.tool()
async def detect_drift(
    file_path: Annotated[str, Field(description="Ruta local al CSV en data/temp_uploads/.")],
    index_column: Annotated[str, Field(description="Nombre de la columna índice del CSV.")],
    method: Annotated[
        Literal["KS", "JS", "PSI", "CUSUM", "MEWMA", "HOTELLING"],
        Field(
            description=(
                "Método estadístico. KS y JS y PSI son univariantes; CUSUM es secuencial "
                "univariante; MEWMA y HOTELLING son multivariantes."
            ),
        ),
    ],
    inicio: Annotated[int, Field(description="Índice desde el que empezar el análisis.")] = 1,
    threshold: Annotated[Optional[float], Field(description="Umbral de decisión (por defecto del método).")] = None,
    num_bins: Annotated[Optional[int], Field(description="Solo PSI: número de bins (default 10).")] = None,
    drift_cusum: Annotated[Optional[float], Field(description="Solo CUSUM: término de deriva (default 0.5).")] = None,
    min_instances: Annotated[Optional[int], Field(description="MEWMA, HOTELLING: observaciones iniciales (default 100).")] = None,
    lambd: Annotated[Optional[float], Field(description="MEWMA: parámetro de suavizado (default 0.5).")] = None,
    alpha: Annotated[Optional[float], Field(description="MEWMA, HOTELLING: nivel de significación (default 0.05).")] = None,
) -> dict:
    """Ejecuta un test estadístico de detección de drift sobre un CSV.

    USA cuando el usuario quiera saber si sus datos han cambiado de distribución
    respecto a un periodo anterior, o validar la estabilidad de un dataset.

    El método elegido en `method` determina los parámetros relevantes:
    - KS, JS, PSI: univariantes. Solo necesitan threshold (y num_bins para PSI).
    - CUSUM: univariante secuencial. Necesita threshold y drift_cusum.
    - MEWMA, HOTELLING: multivariantes. Necesitan min_instances, alpha, lambd (MEWMA).

    NO uses para preguntas teóricas (usa consultar_teoria).

    Devuelve dict con: drift_detected, drift_label, per_column_report, method_used,
    parameters_used, summary.
    """

    init_mcp_http_log()
    try:
        inp = DetectDriftInput(
            file_path=file_path, index_column=index_column, method=method,
            inicio=inicio, threshold=threshold, num_bins=num_bins,
            drift_cusum=drift_cusum, min_instances=min_instances,
            lambd=lambd, alpha=alpha,
        )
        endpoint = _METHOD_TO_ENDPOINT[inp.method]
        params = _build_query_params(inp)
        filename, content, mime = open_csv_for_upload(inp.file_path)

        # Los métodos multivariantes invierten una matriz de covarianza:
        # validamos la dimensionalidad antes de llamar a la API para evitar
        # un 500 opaco (LinAlgError: Singular matrix) y devolver un mensaje útil.
        if inp.method in {"MEWMA", "HOTELLING"}:
            problema = _validate_multivariate(content, inp.index_column)
            if problema is not None:
                return attach_observability({"error": f"Parámetro inválido en detect_drift: {problema}"})

        async with get_client(_SETTINGS) as client:
            response = await client.post(
                endpoint,
                params=params,
                files={"file": (filename, content, mime)},
            )
            response.raise_for_status()

        body = response.json()
        drift_label = body.get("Drift", "Desconocido")
        report = body.get("reporte", {})

        # Construimos la respuesta feliz y la envolvemos para adjuntar la telemetría recolectada
        result = {
            "drift_detected": drift_label == "Detectado",
            "drift_label": drift_label,
            "per_column_report": report,
            "method_used": inp.method,
            "parameters_used": params,
            "summary": _build_summary(inp.method, drift_label, report)
        }
        return attach_observability(result)
    except Exception as exc:  # noqa: BLE001 — se traduce al LLM
        error_result = {"error": translate_exception(exc, "detect_drift")}
        return attach_observability(error_result)
        
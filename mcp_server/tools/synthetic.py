"""Tools MCP de generación de series sintéticas (distribución, ARMA, periódica, tendencia)."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated, Literal, Optional

import httpx
from pydantic import BaseModel, Field, model_validator

from mcp_server.config import load_settings
from mcp_server.errors import translate_exception
from mcp_server.file_utils import deterministic_filename
from mcp_server.http_client import get_client
from mcp_server.instance import mcp
from mcp_server.observability.http_hooks import init_mcp_http_log, attach_observability
_SETTINGS = load_settings()

_FREQ = Literal["B", "D", "W", "M", "Q", "Y", "h", "min", "s"]

# Subconjunto CURADO de distribuciones que el agente expone. La API soporta
# códigos 1..17, pero la capa MCP es un facade acotado: solo ofrecemos los que
# documentamos. Restringir el tipo (en vez de `int` con `ge/le`) cierra el
# contrato —lo aceptado == lo documentado— e impide colar las distribuciones no
# curadas (5 hipergeométrica, 6 constante, 8 lognormal, 14 Pareto, 15/16 rampas).
_DISTR = Literal[1, 2, 3, 4, 7, 9, 10, 11, 12, 13, 17]

# Nº de parámetros que la API acepta en `params` por código de distribución
# (contrato real, drift-detection/api/series_sinteticas/funciones_generales.py,
# función `distribuciones`). Sin esta validación, pasar una aridad incorrecta
# (p. ej. 1 solo valor a una Normal, que indexa params[0] y params[1]) provoca
# un IndexError → 500 opaco en la API en vez de un mensaje accionable. Espejo de
# `_TREND_PARAM_ARITY`.
_DISTRIBUTION_PARAM_ARITY: dict[int, tuple[int, int]] = {
    1: (2, 2),   # Normal[mu, sigma]
    2: (2, 3),   # Binomial[n, p, (loc)]
    3: (1, 2),   # Poisson[lambda, (loc)]
    4: (1, 2),   # Geométrica[p, (loc)]
    7: (0, 2),   # Uniforme[(low), (scale)]
    9: (0, 2),   # Exponencial[(loc), (scale)]
    10: (1, 3),  # Gamma[a, (loc), (scale)]
    11: (2, 4),  # Beta[a, b, (loc), (scale)]
    12: (1, 3),  # ChiCuadrado[df, (loc), (scale)]
    13: (1, 3),  # t-Student[df, (loc), (scale)]
    17: (2, 2),  # Aleatorio[low, high]
}
_DISTRIBUTION_TYPE_NAMES: dict[int, str] = {
    1: "Normal", 2: "Binomial", 3: "Poisson", 4: "Geométrica", 7: "Uniforme",
    9: "Exponencial", 10: "Gamma", 11: "Beta", 12: "ChiCuadrado",
    13: "t-Student", 17: "Aleatorio",
}


def _check_distribution_arity(distribution_type: int, params: list[float]) -> None:
    """Valida la aridad de `params` para una distribución; lanza ValueError si no encaja.

    No-op si el tipo no está en el mapa (deja que la API lo rechace). Se usa
    desde los `model_validator` de los schemas que llevan una distribución base
    (distribución pura y periódica).
    """
    bounds = _DISTRIBUTION_PARAM_ARITY.get(distribution_type)
    if bounds is None:
        return
    lo, hi = bounds
    n = len(params)
    if lo <= n <= hi:
        return
    nombre = _DISTRIBUTION_TYPE_NAMES.get(distribution_type, str(distribution_type))
    esperado = f"exactamente {lo}" if lo == hi else f"entre {lo} y {hi}"
    raise ValueError(
        f"La distribución {nombre} (tipo {distribution_type}) requiere {esperado} "
        f"parámetro(s) en distribution_params, pero se recibieron {n}: {params}. "
        f"Indica los parámetros correctos (p. ej. Normal → [media, desviación])."
    )


def _resolve_horizon(end_date: Optional[str], periods: Optional[int]) -> tuple[str, dict]:
    """Resuelve el horizonte para el endpoint y valida la exclusividad.

    - Requiere exactamente uno de `end_date` o `periods`.
    - Devuelve el sufijo de endpoint ("periodos" o "fin") y los parametros extra.
    - Lanza ValueError si se pasan ambos o ninguno.
    """
    if (end_date is None) == (periods is None):
        raise ValueError(
            "Debes proporcionar 'periods' O 'end_date' (no ambos, no ninguno)."
        )
    if periods is not None:
        return "periodos", {"periodos": periods}
    return "fin", {"fin": end_date}


async def _download_csv(
    client: httpx.AsyncClient,
    endpoint: str,
    params: dict,
    out_name: str,
) -> Path:
    """Descarga el CSV generado por el backend y lo guarda en el workspace.

    Hace un GET al endpoint MCP con los params, escribe el contenido recibido
    en `workspace_dir/out_name` y devuelve la ruta local.
    """
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
    """Descarga el PNG de la grafica asociado a la serie generada.

    El flujo es equivalente a `_download_csv`, pero guardando el binario PNG.
    """
    response = await client.get(endpoint, params=params)
    response.raise_for_status()
    target = _SETTINGS.workspace_dir / out_name
    target.write_bytes(response.content)
    return target


async def _try_download_png(
    client: httpx.AsyncClient,
    endpoint: str,
    params: dict,
    out_name: str,
) -> Optional[Path]:
    """Descarga el PNG best-effort: si falla, devuelve None sin abortar la tool.

    El CSV ya es un resultado valido por si mismo; la grafica es opcional, asi
    que un fallo del endpoint /Plot no debe descartar la serie generada.
    """
    try:
        return await _download_png(client, endpoint, params, out_name)
    except Exception:  # noqa: BLE001 — el CSV ya es válido; la gráfica es opcional
        return None


def _row_count(csv_path: Path) -> int:
    """Cuenta filas de datos (excluye cabecera). Devuelve 0 si falla la lectura."""
    try:
        with csv_path.open("r", encoding="utf-8") as fh:
            return max(sum(1 for _ in fh) - 1, 0)
    except Exception:
        return 0


# Schemas internos (usados por tests unitarios — el LLM ve params planos).

class GenerateDistributionInput(BaseModel):
    """Schema interno para distribuciones (usado en tests y validacion)."""
    start_date: str
    end_date: Optional[str] = None
    periods: Optional[int] = None
    frequency: _FREQ
    distribution_type: _DISTR
    distribution_params: list[float]
    column_name: str = "valor"
    with_plot: bool = True

    @model_validator(mode="after")
    def _check_params_arity(self) -> "GenerateDistributionInput":
        _check_distribution_arity(self.distribution_type, self.distribution_params)
        return self


class GenerateArmaInput(BaseModel):
    """Schema interno para ARMA con parametros de ruido y estacionalidad."""
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
    with_plot: bool = True


class GeneratePeriodicInput(BaseModel):
    """Schema interno para series periodicas con patron repetitivo."""
    start_date: str
    end_date: Optional[str] = None
    periods: Optional[int] = None
    frequency: _FREQ
    column_name: str = "valor"
    distribution_type: _DISTR
    distribution_params: list[float]
    period_length: int
    pattern_type: Literal[1, 2]
    with_plot: bool = True

    @model_validator(mode="after")
    def _check_params_arity(self) -> "GeneratePeriodicInput":
        _check_distribution_arity(self.distribution_type, self.distribution_params)
        return self


# Nº de coeficientes que la API exige en `trend_params` según `trend_type`.
# Contrato real (drift-detection/api/series_sinteticas/funciones_generales.py,
# tendencia_det): tipos 1/3/4 hacen params[0] y params[1] → exigen exactamente
# 2; el tipo 2 (polinómica) usa toda la lista → exige al menos 1. Sin esta
# validación, enviar 1 coeficiente a una exponencial provoca un IndexError 500
# en la API en vez de un mensaje claro.
_TREND_PARAM_ARITY: dict[int, tuple[int, Optional[int]]] = {
    1: (2, 2),       # lineal: [a, b]
    2: (1, None),    # polinómica: [a0, a1, …, an]
    3: (2, 2),       # exponencial: [a, b]
    4: (2, 2),       # logarítmica: [a, b]
}
_TREND_TYPE_NAMES: dict[int, str] = {1: "lineal", 2: "polinómica", 3: "exponencial", 4: "logarítmica"}


class GenerateTrendInput(BaseModel):
    """Schema interno para series con tendencia determinista y ruido opcional."""
    start_date: str
    end_date: Optional[str] = None
    periods: Optional[int] = None
    frequency: _FREQ
    column_name: str = "valor"
    trend_type: int
    trend_params: list[float]
    noise: float = 0.0
    with_plot: bool = True

    @model_validator(mode="after")
    def _check_trend_params_arity(self) -> "GenerateTrendInput":
        bounds = _TREND_PARAM_ARITY.get(self.trend_type)
        if bounds is None:
            return self  # tipo desconocido: deja que la API lo rechace
        lo, hi = bounds
        n = len(self.trend_params)
        if n < lo or (hi is not None and n > hi):
            nombre = _TREND_TYPE_NAMES.get(self.trend_type, str(self.trend_type))
            esperado = f"exactamente {lo}" if lo == hi else f"al menos {lo}"
            raise ValueError(
                f"La tendencia {nombre} (tipo {self.trend_type}) requiere {esperado} "
                f"coeficiente(s) en trend_params, pero se recibieron {n}: {self.trend_params}. "
                f"Indica los coeficientes correctos (p. ej. lineal/exponencial/logarítmica → [a, b])."
            )
        return self


# ───────────────────────── 1. generate_synthetic_distribution ─────────────────────────

@mcp.tool()
async def generate_synthetic_distribution(
    start_date: Annotated[
        str,
        Field(
            description="Fecha de inicio 'YYYY-MM-DD'.",
            json_schema_extra={"evidence": "date"},
        ),
    ],
    frequency: Annotated[
        _FREQ,
        Field(
            description="'D' diaria, 'W' semanal, 'M' mensual, 'h' horaria.",
            json_schema_extra={"evidence": "freq"},
        ),
    ],
    distribution_type: Annotated[
        _DISTR,
        Field(
            description=(
                "1=Normal[mu,sigma], 2=Binomial[n,p], 3=Poisson[lambda], 4=Geometrica[p], "
                "7=Uniforme[low,high], 9=Exponencial[scale], 10=Gamma[a], 11=Beta[a,b], "
                "12=ChiCuadrado[df], 13=TStudent[t], 17=Aleatorio[low,high]"
            ),
            json_schema_extra={"evidence": "distribution_kind"},
        ),
    ],
    distribution_params: Annotated[
        list[float],
        Field(
            description="Parámetros de la distribución como lista.",
            json_schema_extra={"evidence": "numeric_list"},
        ),
    ],
    end_date: Annotated[
        Optional[str],
        Field(
            description="Fecha de fin 'YYYY-MM-DD'. Excluyente con periods.",
            json_schema_extra={"oneof_group": "horizon", "evidence": "date"},
        ),
    ] = None,
    periods: Annotated[
        Optional[int],
        Field(
            description="Número de periodos. Excluyente con end_date.",
            json_schema_extra={"oneof_group": "horizon", "evidence": "integer"},
        ),
    ] = None,
    column_name: Annotated[str, Field(description="Nombre de la columna generada.")] = "valor",
    with_plot: Annotated[bool, Field(description="Si True, genera además un PNG con la gráfica.")] = True,
) -> dict:
    """Genera una serie temporal sintética siguiendo una distribución estadistica.

    USA cuando el usuario quiera crear datos artificiales con una distribucion conocida
    (Normal, Poisson, Uniforme, Beta, Gamma...). NO uses si la serie debe tener
    autocorrelacion (usa generate_synthetic_arma) ni patrones ciclicos
    (usa generate_synthetic_periodic).

    Flujo MCP:
    1) Normaliza la entrada con el schema interno.
    2) Resuelve horizonte (periodos o fin) y compone endpoint.
    3) Descarga el CSV generado por el backend y lo guarda en el workspace.
    4) Si `with_plot=True`, descarga el PNG asociado.

    Pasa `periods` O `end_date`, no ambos.

    Devuelve: output_path, rows_generated, image_path, summary.
    """
    init_mcp_http_log()
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
                png_path = await _try_download_png(client, f"/Plot/distribucion/{suffix}", params, png_name)

        rows = _row_count(csv_path)
        result = {
            "output_path": str(csv_path),
            "rows_generated": rows,
            "image_path": str(png_path) if png_path else None,
            "summary": (
                f"Serie generada con distribución tipo {inp.distribution_type} "
                f"({rows} filas, freq={inp.frequency})."
            ),
        }
        return attach_observability(result)
    except Exception as exc:  # noqa: BLE001
        error_result = {"error": translate_exception(exc, "generate_synthetic_distribution")}
        return attach_observability(error_result)


# ───────────────────────── 2. generate_synthetic_arma ─────────────────────────

@mcp.tool()
async def generate_synthetic_arma(
    start_date: Annotated[
        str,
        Field(
            description="Fecha de inicio 'YYYY-MM-DD'.",
            json_schema_extra={"evidence": "date"},
        ),
    ],
    frequency: Annotated[
        _FREQ,
        Field(description="Frecuencia temporal.", json_schema_extra={"evidence": "freq"}),
    ],
    end_date: Annotated[
        Optional[str],
        Field(
            description="Fecha de fin (excluyente con periods).",
            json_schema_extra={"oneof_group": "horizon", "evidence": "date"},
        ),
    ] = None,
    periods: Annotated[
        Optional[int],
        Field(
            description="Número de periodos (excluyente con end_date).",
            json_schema_extra={"oneof_group": "horizon", "evidence": "integer"},
        ),
    ] = None,
    column_name: Annotated[str, Field(description="Nombre de la columna generada.")] = "valor",
    constant: Annotated[
        float,
        Field(description="Término constante c del modelo ARMA.", json_schema_extra={"tunable": True}),
    ] = 0.0,
    noise_std: Annotated[
        float,
        Field(description="Desviación estándar del ruido blanco.", json_schema_extra={"tunable": True}),
    ] = 1.0,
    seasonality: Annotated[
        int,
        Field(description="Periodo de estacionalidad (0 = no estacional).", json_schema_extra={"tunable": True}),
    ] = 0,
    ar_coefficients: Annotated[
        list[float],
        Field(description="Coeficientes AR. Lista vacía = sin AR.", json_schema_extra={"tunable": True}),
    ] = [],
    ma_coefficients: Annotated[
        list[float],
        Field(description="Coeficientes MA. Lista vacía = sin MA.", json_schema_extra={"tunable": True}),
    ] = [],
    with_plot: Annotated[bool, Field(description="Si True, también genera PNG.")] = True,
) -> dict:
    """Genera una serie temporal con estructura ARMA(p,q).

    USA cuando el usuario pida datos con autocorrelacion temporal, AR(p), MA(q),
    ARMA(p,q) o estacionalidad fija. Pasa `periods` O `end_date`, no ambos.

    Flujo MCP:
    1) Normaliza la entrada y valida el horizonte.
    2) Compone el endpoint /Datos/ARMA/{periodos|fin}.
    3) Descarga el CSV generado y lo guarda en el workspace.
    4) Si `with_plot=True`, descarga el PNG correspondiente.

    Devuelve: output_path, rows_generated, image_path, summary, model_spec.
    """
    init_mcp_http_log()
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
                png_path = await _try_download_png(client, f"/Plot/ARMA/{suffix}", params, png_name)

        rows = _row_count(csv_path)
        result = {
            "output_path": str(csv_path),
            "rows_generated": rows,
            "image_path": str(png_path) if png_path else None,
            "model_spec": {
                "constant": inp.constant,
                "noise_std": inp.noise_std,
                "seasonality": inp.seasonality,
                "ar": inp.ar_coefficients,
                "ma": inp.ma_coefficients,
            },
            "summary": (
                f"Serie ARMA generada ({rows} filas, AR={len(inp.ar_coefficients)}, "
                f"MA={len(inp.ma_coefficients)}, freq={inp.frequency})."
            ),
        }
        return attach_observability(result)
       
    except Exception as exc:  # noqa: BLE001
        error_result = {"error": translate_exception(exc, "generate_synthetic_arma")}
        return attach_observability(error_result)


# ───────────────────────── 3. generate_synthetic_periodic ─────────────────────────

@mcp.tool()
async def generate_synthetic_periodic(
    start_date: Annotated[
        str,
        Field(
            description="Fecha de inicio 'YYYY-MM-DD'.",
            json_schema_extra={"evidence": "date"},
        ),
    ],
    frequency: Annotated[
        _FREQ,
        Field(description="Frecuencia temporal.", json_schema_extra={"evidence": "freq"}),
    ],
    distribution_type: Annotated[
        _DISTR,
        Field(
            description="Distribución base (ver generate_synthetic_distribution).",
            json_schema_extra={"evidence": "distribution_kind"},
        ),
    ],
    distribution_params: Annotated[
        list[float],
        Field(
            description="Parámetros de la distribución base.",
            json_schema_extra={"evidence": "numeric_list"},
        ),
    ],
    period_length: Annotated[
        int,
        Field(
            gt=0,
            description="Cada cuántas observaciones se repite el patrón.",
            json_schema_extra={"evidence": "integer"},
        ),
    ],
    pattern_type: Annotated[
        Literal[1, 2],
        Field(
            description="1 = variación de amplitud, 2 = variación de cantidad.",
            json_schema_extra={"evidence": "pattern_kind"},
        ),
    ],
    end_date: Annotated[
        Optional[str],
        Field(
            description="Fecha de fin (excluyente con periods).",
            json_schema_extra={"oneof_group": "horizon", "evidence": "date"},
        ),
    ] = None,
    periods: Annotated[
        Optional[int],
        Field(
            description="Número de periodos (excluyente con end_date).",
            json_schema_extra={"oneof_group": "horizon", "evidence": "integer"},
        ),
    ] = None,
    column_name: Annotated[str, Field(description="Nombre de la columna.")] = "valor",
    with_plot: Annotated[bool, Field(description="Si True, también genera PNG.")] = True,
) -> dict:
    """Genera una serie temporal con patrones ciclicos repetidos.

    USA cuando el usuario mencione estacionalidad observable (semanal, mensual, anual),
    patrones que se repiten cada N observaciones, o simulacion de demanda con ciclos.

    Flujo MCP:
    1) Normaliza la entrada y valida el horizonte.
    2) Compone el endpoint /Datos/periodicos/{periodos|fin}.
    3) Descarga el CSV generado y lo guarda en el workspace.
    4) Si `with_plot=True`, descarga el PNG correspondiente.

    Pasa `periods` O `end_date`, no ambos.

    Devuelve: output_path, rows_generated, image_path, summary.
    """
    init_mcp_http_log()
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
                png_path = await _try_download_png(client, f"/Plot/periodicos/{suffix}", params, png_name)

        rows = _row_count(csv_path)
        result = {
            "output_path": str(csv_path),
            "rows_generated": rows,
            "image_path": str(png_path) if png_path else None,
            "summary": (  
                f"Serie periódica generada ({rows} filas, period_length={inp.period_length}, "
                f"pattern_type={inp.pattern_type})."
            ),
        }
        return attach_observability(result)
       
    except Exception as exc:  # noqa: BLE001
        error_result = {"error": translate_exception(exc, "generate_synthetic_periodic")}
        return attach_observability(error_result)

# ───────────────────────── 4. generate_synthetic_trend ─────────────────────────

@mcp.tool()
async def generate_synthetic_trend(
    start_date: Annotated[
        str,
        Field(
            description="Fecha de inicio 'YYYY-MM-DD'.",
            json_schema_extra={"evidence": "date"},
        ),
    ],
    frequency: Annotated[
        _FREQ,
        Field(description="Frecuencia temporal.", json_schema_extra={"evidence": "freq"}),
    ],
    trend_type: Annotated[
        int,
        Field(
            ge=1,
            description="Código del tipo de tendencia: 1=lineal, 2=polinómica, 3=exponencial, 4=logarítmica.",
            json_schema_extra={"evidence": "trend_kind"},
        ),
    ],
    trend_params: Annotated[
        list[float],
        Field(
            description=(
                "Coeficientes que definen la tendencia. El número depende del tipo: "
                "lineal/exponencial/logarítmica (1/3/4) requieren EXACTAMENTE 2 coeficientes [a, b]; "
                "polinómica (2) requiere al menos 1 [a0, a1, …, an]."
            ),
            json_schema_extra={"evidence": "numeric_list"},
        ),
    ],
    end_date: Annotated[
        Optional[str],
        Field(
            description="Fecha de fin (excluyente con periods).",
            json_schema_extra={"oneof_group": "horizon", "evidence": "date"},
        ),
    ] = None,
    periods: Annotated[
        Optional[int],
        Field(
            description="Número de periodos (excluyente con end_date).",
            json_schema_extra={"oneof_group": "horizon", "evidence": "integer"},
        ),
    ] = None,
    column_name: Annotated[str, Field(description="Nombre de la columna.")] = "valor",
    noise: Annotated[
        float,
        Field(description="Magnitud del ruido aditivo gaussiano.", json_schema_extra={"tunable": True}),
    ] = 0.0,
    with_plot: Annotated[bool, Field(description="Si True, también genera PNG.")] = True,
) -> dict:
    """Genera una serie temporal con tendencia determinista.

    USA cuando el usuario quiera datos con crecimiento o decrecimiento sistematico,
    o simulacion de procesos no estacionarios con tendencia conocida.

    Flujo MCP:
    1) Normaliza la entrada y valida el horizonte.
    2) Compone el endpoint /Datos/tendencia/{periodos|fin}.
    3) Descarga el CSV generado y lo guarda en el workspace.
    4) Si `with_plot=True`, descarga el PNG correspondiente.

    Pasa `periods` O `end_date`, no ambos.

    Devuelve: output_path, rows_generated, image_path, summary.
    """
    init_mcp_http_log()
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
                png_path = await _try_download_png(client, f"/Plot/tendencia/{suffix}", params, png_name)

        rows = _row_count(csv_path)
        result = {
            "output_path": str(csv_path),
            "rows_generated": rows,
            "image_path": str(png_path) if png_path else None,
            "summary": (
                f"Serie con tendencia generada ({rows} filas, trend_type={inp.trend_type})."
            ),
        }
        return attach_observability(result)
    except Exception as exc:  # noqa: BLE001
        error_result = {"error": translate_exception(exc, "generate_synthetic_trend")}
        return attach_observability(error_result)
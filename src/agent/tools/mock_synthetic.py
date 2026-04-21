"""Herramienta mock para generación de series temporales sintéticas."""

from __future__ import annotations

import random
import uuid

from langchain_core.tools import tool


@tool
def generate_synthetic_series(
    start_date: str,
    periods: int,
    frequency: str,
    distribution_type: int,
    distribution_params: list[float],
) -> dict:
    """Genera una serie temporal sintética con la distribución y frecuencia indicadas.

    Crea un fichero CSV con la serie generada y devuelve estadísticas descriptivas.
    La serie puede usarse para aumentar datos, hacer pruebas de modelos o simular escenarios.

    Usa esta herramienta cuando el usuario quiera:
    - Crear datos sintéticos para entrenar o validar modelos predictivos.
    - Simular una serie temporal con parámetros estadísticos concretos.
    - Generar datos de prueba cuando no dispone de datos reales.

    Args:
        start_date: Fecha de inicio de la serie en formato 'YYYY-MM-DD' (ej. '2023-01-01').
        periods: Número de periodos (filas) a generar. Debe ser mayor que 0.
        frequency: Frecuencia temporal: 'D' (diaria), 'W' (semanal), 'M' (mensual),
                   'H' (horaria), 'T' (minutal).
        distribution_type: Tipo de distribución a usar:
                           0 = Normal, 1 = Uniforme, 2 = Poisson, 3 = Exponencial.
        distribution_params: Lista de parámetros de la distribución.
                             Normal: [media, desviacion_tipica] ej. [0.0, 1.0]
                             Uniforme: [minimo, maximo] ej. [0.0, 100.0]
                             Poisson: [lambda] ej. [5.0]
                             Exponencial: [escala] ej. [1.0]

    Returns:
        Diccionario con los campos:
        - series_id: Identificador único de la serie generada.
        - file_path_generated: Ruta al CSV generado con la serie.
        - periods_generated: Número de periodos efectivamente generados.
        - frequency: Frecuencia usada.
        - summary_stats: Estadísticas descriptivas (mean, std, min, max, percentiles).
        - distribution_name: Nombre legible de la distribución usada.
    """
    dist_names = {0: "Normal", 1: "Uniforme", 2: "Poisson", 3: "Exponencial"}
    rng = random.Random(hash(start_date + str(periods) + frequency))

    series_id = str(uuid.UUID(int=rng.getrandbits(128)))

    if distribution_type == 0:
        mean = distribution_params[0] if len(distribution_params) > 0 else 0.0
        std = distribution_params[1] if len(distribution_params) > 1 else 1.0
        values = [rng.gauss(mean, std) for _ in range(periods)]
    elif distribution_type == 1:
        low = distribution_params[0] if len(distribution_params) > 0 else 0.0
        high = distribution_params[1] if len(distribution_params) > 1 else 1.0
        values = [rng.uniform(low, high) for _ in range(periods)]
    elif distribution_type == 2:
        lam = distribution_params[0] if len(distribution_params) > 0 else 5.0
        values = [float(rng.randint(0, int(lam * 3))) for _ in range(periods)]
    else:
        scale = distribution_params[0] if len(distribution_params) > 0 else 1.0
        values = [rng.expovariate(1.0 / scale) for _ in range(periods)]

    mean_val = round(sum(values) / len(values), 4)
    sorted_vals = sorted(values)
    p25 = round(sorted_vals[int(len(sorted_vals) * 0.25)], 4)
    p50 = round(sorted_vals[int(len(sorted_vals) * 0.50)], 4)
    p75 = round(sorted_vals[int(len(sorted_vals) * 0.75)], 4)
    std_val = round((sum((v - mean_val) ** 2 for v in values) / len(values)) ** 0.5, 4)

    return {
        "series_id": series_id,
        "file_path_generated": f"data/temp_uploads/synthetic_{series_id[:8]}.csv",
        "periods_generated": periods,
        "frequency": frequency,
        "distribution_name": dist_names.get(distribution_type, "Desconocida"),
        "summary_stats": {
            "mean": mean_val,
            "std": std_val,
            "min": round(min(values), 4),
            "max": round(max(values), 4),
            "p25": p25,
            "p50": p50,
            "p75": p75,
        },
    }

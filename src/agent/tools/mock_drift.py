"""Herramienta mock para detección de drift con el test de Kolmogorov-Smirnov."""

from __future__ import annotations

import random

from langchain_core.tools import tool


@tool
def detect_drift_kolmogorov_smirnov(
    file_path: str,
    reference_column: str,
    threshold: float = 0.05,
) -> dict:
    """Detecta data drift en una columna de un fichero CSV usando el test de Kolmogorov-Smirnov.

    Compara la distribución de la columna indicada contra una distribución de referencia
    almacenada en el sistema. Devuelve el estadístico KS, el p-valor y si se detectó drift.

    Usa esta herramienta cuando el usuario quiera:
    - Saber si sus datos han cambiado de distribución respecto a un periodo anterior.
    - Detectar drift estadístico en una variable concreta de su CSV.
    - Validar la estabilidad de una serie temporal antes de entrenar un modelo.

    Args:
        file_path: Ruta al fichero CSV con los datos a analizar.
        reference_column: Nombre de la columna sobre la que ejecutar el test KS.
        threshold: Nivel de significancia para decidir si hay drift (por defecto 0.05).

    Returns:
        Diccionario con los campos:
        - ks_statistic: Estadístico KS (0.0–1.0). Valores altos indican mayor diferencia.
        - p_value: P-valor del test. Por debajo del umbral indica drift significativo.
        - drift_detected: True si p_value < threshold.
        - reference_column: Columna analizada.
        - threshold_used: Umbral empleado en la decisión.
        - sample_size: Número de registros analizados (simulado).
    """
    rng = random.Random(hash(file_path + reference_column))

    ks_statistic = round(rng.uniform(0.02, 0.45), 4)
    p_value = round(rng.uniform(0.001, 0.12), 4)
    drift_detected = p_value < threshold
    sample_size = rng.randint(800, 5000)

    return {
        "ks_statistic": ks_statistic,
        "p_value": p_value,
        "drift_detected": drift_detected,
        "reference_column": reference_column,
        "threshold_used": threshold,
        "sample_size": sample_size,
        "interpretation": (
            f"Se detectó drift estadístico en '{reference_column}' "
            f"(p={p_value} < {threshold})."
            if drift_detected
            else f"No se detectó drift significativo en '{reference_column}' "
            f"(p={p_value} >= {threshold})."
        ),
    }

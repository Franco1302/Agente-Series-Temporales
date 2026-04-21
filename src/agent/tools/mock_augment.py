"""Herramienta mock para aumentar datos mediante una relación lineal."""

from __future__ import annotations

import random
from pathlib import Path

from langchain_core.tools import tool


@tool
def augment_data_linear_relation(
    file_path: str,
    index_column: str,
    new_column_name: str,
    slope: float,
    intercept: float,
) -> dict:
    """Añade una nueva columna al CSV calculada como una relación lineal de otra columna existente.

    La nueva columna se calcula como: nueva_columna = slope * index_column + intercept.
    Esto permite enriquecer datasets con variables derivadas o correlacionadas linealmente.

    Usa esta herramienta cuando el usuario quiera:
    - Añadir una variable derivada linealmente de otra columna de su CSV.
    - Aumentar el dataset con una columna sintética correlacionada.
    - Simular una variable dependiente para pruebas de modelado.

    Args:
        file_path: Ruta al fichero CSV que se desea aumentar.
        index_column: Nombre de la columna existente que actúa como variable independiente (X).
        new_column_name: Nombre de la nueva columna que se creará en el CSV aumentado.
        slope: Pendiente de la relación lineal (coeficiente de X).
        intercept: Término independiente (valor de la nueva columna cuando X = 0).

    Returns:
        Diccionario con los campos:
        - augmented_file_path: Ruta al CSV aumentado con la nueva columna.
        - original_file: Ruta al fichero original procesado.
        - rows_generated: Número de filas procesadas y enriquecidas.
        - new_column_name: Nombre de la columna añadida.
        - formula: Fórmula aplicada como cadena de texto legible.
        - sample_values: Tres ejemplos de valores generados para la nueva columna.
    """
    rng = random.Random(hash(file_path + index_column + new_column_name))

    rows = rng.randint(500, 8000)

    sample_x_values = [rng.uniform(0.0, 100.0) for _ in range(3)]
    sample_new_values = [round(slope * x + intercept, 4) for x in sample_x_values]

    stem = Path(file_path).stem
    augmented_path = f"data/temp_uploads/{stem}_augmented.csv"

    return {
        "augmented_file_path": augmented_path,
        "original_file": file_path,
        "rows_generated": rows,
        "new_column_name": new_column_name,
        "formula": f"{new_column_name} = {slope} * {index_column} + {intercept}",
        "sample_values": [
            {index_column: round(sample_x_values[i], 4), new_column_name: sample_new_values[i]}
            for i in range(3)
        ],
    }

"""Clasificación semántica de parámetros heredables entre herramientas.

Fuente única de verdad para decidir qué argumentos de tool puede arrastrar el
agente desde la sesión cuando una nueva herramienta los necesita. Solo los
parámetros listados aquí entran en la herencia automática; los parámetros de
dominio (que cambian la semántica del análisis: ``method``, ``trend_type``,
``distribution_type``, ``strategy``, ``model``, etc.) quedan deliberadamente
fuera para preservar RNF-02 (no asumir valores por defecto en tools analíticas).

Las familias se agrupan por significado, no por la herramienta que los emite:
si dos tools usan ``file_path`` con el mismo significado (la ruta al CSV de
trabajo), pertenecen a la misma familia y pueden compartirlo. Si en algún
momento una tool reusa un nombre con semántica distinta (p. ej. un hipotético
``frequency`` que significara "número de bins"), bastaría con sacar ese
parámetro de la familia para frenar la herencia desde aquí.
"""

from __future__ import annotations


#: Familias semánticas de parámetros que el agente puede heredar entre tools.
#: Clave = nombre conceptual de la familia (solo documentación, no se usa en
#: lógica). Valor = nombres de parámetros que pertenecen a esa familia.
PARAM_FAMILIES: dict[str, frozenset[str]] = {
    # Marco temporal de la serie: cualquier tool que genere/consuma una ventana
    # de tiempo puede reutilizar estos valores entre turnos.
    "temporal_window": frozenset({"start_date", "end_date", "periods", "frequency"}),
    # Origen de los datos: ruta al CSV activo y columna que actúa como índice
    # temporal. Compartido por drift, forecast, augment y exógenas.
    "data_source": frozenset({"file_path", "index_column"}),
    # Identidad de la serie generada/objetivo. Solo nombre de la columna de
    # salida en herramientas sintéticas; intencionalmente NO incluye
    # ``target_column`` (semántica específica de forecast).
    "series_identity": frozenset({"column_name"}),
}


#: Unión plana de los parámetros heredables. Es lo que consulta la pasada de
#: herencia en el razonador (``reasoning._inherit_from_session``).
INHERITABLE_PARAMS: frozenset[str] = frozenset().union(*PARAM_FAMILIES.values())

"""Helpers de scoring reutilizables para los tests del agente y del MCP.

Origen: la lógica nació en `mcp_benchmark.py` (single-turn benchmarking de
modelos). Se extrae aquí para que `scripts/test_agent.py` la reutilice sin
duplicar el normalizado de valores ni la comparación de args.

El benchmark de Ollama (mcp_benchmark.py) maneja `tool_calls` con la forma
`call.function.name / call.function.arguments`. El agente LangGraph emite
`AIMessage.tool_calls` como dicts `{"name": ..., "args": ..., "id": ...}`.
Por eso los helpers operan sobre `nombre` y `args` ya extraídos.
"""

from __future__ import annotations

from typing import Any


def normalizar(valor: Any) -> Any:
    """Normaliza un valor para comparación flexible (case-insensitive en strings)."""
    if isinstance(valor, str):
        return valor.strip().lower()
    return valor


def comparar_args(args_obtenidos: dict, args_esperados: dict) -> tuple[int, int]:
    """Cuenta cuántos argumentos coinciden con los esperados.

    Returns: (aciertos, total). Si total es 0, el porcentaje no aplica.
    """
    if not args_esperados:
        return 0, 0
    aciertos = sum(
        1
        for k, v in args_esperados.items()
        if k in args_obtenidos and normalizar(args_obtenidos[k]) == normalizar(v)
    )
    return aciertos, len(args_esperados)


def evaluar_tool_call(
    nombre_invocado: str | None,
    args_obtenidos: dict,
    nombre_esperado: str | None,
    args_requeridos: list[str],
    valores_esperados: dict,
) -> dict[str, Any]:
    """Evalúa una llamada a tool con cuatro métricas atómicas.

    - `herramienta_correcta`: el nombre coincide con `nombre_esperado` (o ambos
      son None, lo que indica "no se debe invocar tool").
    - `args_requeridos_presentes`: todos los kwargs en `args_requeridos` están
      presentes en `args_obtenidos` (independiente del valor).
    - `precision_args_pct`: porcentaje de `valores_esperados` que coinciden.
    - `aciertos_args`/`total_args`: contadores brutos.
    """
    herramienta_correcta = nombre_invocado == nombre_esperado

    if args_requeridos:
        args_requeridos_presentes = all(k in args_obtenidos for k in args_requeridos)
    else:
        args_requeridos_presentes = True

    aciertos, total = comparar_args(args_obtenidos, valores_esperados)
    precision_pct = round((aciertos / total) * 100, 1) if total else 0.0

    return {
        "herramienta_invocada": nombre_invocado,
        "herramienta_correcta": herramienta_correcta,
        "args_requeridos_presentes": args_requeridos_presentes,
        "precision_args_pct": precision_pct,
        "aciertos_args": aciertos,
        "total_args": total,
    }


__all__ = ["normalizar", "comparar_args", "evaluar_tool_call"]

"""Script de prueba end-to-end del grafo LangGraph desde terminal.

Ejecutar desde la raíz del proyecto:
    python -m scripts.test_agent
    python -m scripts.test_agent --caso 1
    python -m scripts.test_agent --caso 2
    python -m scripts.test_agent --todos
"""

from __future__ import annotations

import argparse
import sys
import textwrap
from dataclasses import dataclass, field
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from src.agent.graph import build_agent_graph


@dataclass
class TestCase:
    """Definición de un caso de prueba del agente."""

    nombre: str
    mensaje: str
    uploaded_file_path: str | None = None
    # Términos que deben aparecer en la respuesta final (al menos uno)
    términos_esperados: list[str] = field(default_factory=list)
    # Si True, se espera que el agente ejecute al menos una herramienta
    espera_tool_call: bool = False


# ── Casos de prueba ──────────────────────────────────────────────────────────

CASOS: list[TestCase] = [
    TestCase(
        nombre="Detección de drift con parámetros completos",
        mensaje=(
            "Analiza el drift en el fichero 'data/temp_uploads/ventas.csv' "
            "sobre la columna 'precio' usando un umbral de 0.05."
        ),
        uploaded_file_path="data/temp_uploads/ventas.csv",
        términos_esperados=["drift", "p_value", "kolmogorov", "precio", "umbral", "0.05"],
        espera_tool_call=True,
    ),
    TestCase(
        nombre="Parámetros incompletos — agente pide datos",
        mensaje="Genera una serie temporal sintética.",
        uploaded_file_path=None,
        términos_esperados=["start_date", "periods", "frequency", "distribution", "necesito", "proporcion"],
        espera_tool_call=False,  # No debe ejecutar sin parámetros
    ),
    TestCase(
        nombre="Pregunta general sin herramienta",
        mensaje="¿Qué es el data drift y cuándo debo preocuparme por él?",
        uploaded_file_path=None,
        términos_esperados=["drift", "distribución", "datos", "modelo"],
        espera_tool_call=False,
    ),
    TestCase(
        nombre="Augmentación de datos con relación lineal",
        mensaje=(
            "Añade una columna llamada 'precio_ajustado' al fichero "
            "'data/temp_uploads/ventas.csv' usando la columna 'precio' "
            "con pendiente 1.1 e intercepto 50."
        ),
        uploaded_file_path="data/temp_uploads/ventas.csv",
        términos_esperados=["augment", "precio_ajustado", "pendiente", "columna", "fichero"],
        espera_tool_call=True,
    ),
]


# ── Lógica de ejecución ──────────────────────────────────────────────────────

def _run_case(caso: TestCase, verbose: bool = True) -> bool:
    """Ejecuta un caso de prueba y devuelve True si pasa, False si falla."""
    separador = "─" * 60
    print(f"\n{separador}")
    print(f"CASO: {caso.nombre}")
    print(separador)
    print(f"Mensaje:  {textwrap.shorten(caso.mensaje, width=80)}")
    print(f"Fichero:  {caso.uploaded_file_path or '(ninguno)'}")
    print()

    graph = build_agent_graph()
    thread_id = f"test-{caso.nombre[:20].replace(' ', '-')}"
    config = {"configurable": {"thread_id": thread_id}}

    input_state: dict[str, Any] = {
        "messages": [HumanMessage(content=caso.mensaje)],
        "uploaded_file_path": caso.uploaded_file_path,
        "iteration_count": 0,
        "pending_params": [],
    }

    final_response = ""
    tools_executed: list[str] = []
    nodes_visited: list[str] = []

    try:
        for event in graph.stream(input_state, config=config):
            node_name = next(iter(event))
            if node_name.startswith("__"):
                continue

            nodes_visited.append(node_name)
            node_output: dict = event[node_name]
            messages_out: list = node_output.get("messages", [])

            if verbose:
                print(f"  → Nodo: {node_name}")

            for msg in messages_out:
                if isinstance(msg, AIMessage):
                    tool_calls = getattr(msg, "tool_calls", None) or []
                    if tool_calls:
                        names = [
                            (c.get("name") if isinstance(c, dict) else getattr(c, "name", "?"))
                            for c in tool_calls
                        ]
                        if verbose:
                            print(f"     Tool calls emitidas: {names}")
                    elif msg.content:
                        final_response = msg.content
                        if verbose:
                            preview = textwrap.shorten(msg.content, width=120)
                            print(f"     Respuesta: {preview}")

                elif isinstance(msg, ToolMessage):
                    tools_executed.append(msg.name)
                    if verbose:
                        preview = textwrap.shorten(str(msg.content), width=100)
                        print(f"     Resultado de {msg.name}: {preview}")

    except (ConnectionError, RuntimeError) as exc:
        print(f"\n  ✗ ERROR DE CONEXIÓN: {exc}")
        print("  Verifica que Ollama esté en ejecución con el modelo configurado.")
        return False
    except Exception as exc:
        print(f"\n  ✗ ERROR INESPERADO: {exc}")
        return False

    # ── Validaciones ─────────────────────────────────────────────────────────
    ok = True
    print()

    # 1. Debe existir respuesta final
    if not final_response.strip():
        print("  ✗ FALLO: no se obtuvo respuesta de texto del agente.")
        ok = False
    else:
        print("  ✓ Se obtuvo respuesta de texto.")

    # 2. Verificar si se esperaba ejecución de herramienta
    if caso.espera_tool_call:
        if tools_executed:
            print(f"  ✓ Herramientas ejecutadas: {tools_executed}")
        else:
            print("  ✗ FALLO: se esperaba ejecución de herramienta pero no se ejecutó ninguna.")
            ok = False
    else:
        if tools_executed:
            print(f"  ⚠ Herramientas ejecutadas sin esperarlo: {tools_executed}")
        else:
            print("  ✓ No se ejecutaron herramientas (correcto para este caso).")

    # 3. Verificar términos esperados en la respuesta
    if caso.términos_esperados:
        respuesta_lower = final_response.lower()
        encontrados = [t for t in caso.términos_esperados if t.lower() in respuesta_lower]
        if encontrados:
            print(f"  ✓ Términos encontrados en respuesta: {encontrados}")
        else:
            print(
                f"  ✗ FALLO: ningún término esperado encontrado en la respuesta.\n"
                f"     Esperados: {caso.términos_esperados}"
            )
            ok = False

    estado = "PASADO ✓" if ok else "FALLADO ✗"
    print(f"\n  Resultado: {estado}")
    return ok


def _print_summary(resultados: list[tuple[str, bool]]) -> None:
    """Imprime el resumen final de todos los casos ejecutados."""
    separador = "═" * 60
    print(f"\n{separador}")
    print("RESUMEN DE PRUEBAS")
    print(separador)
    pasados = sum(1 for _, ok in resultados if ok)
    for nombre, ok in resultados:
        icono = "✓" if ok else "✗"
        print(f"  {icono} {nombre}")
    print(f"\n  Total: {pasados}/{len(resultados)} pasados")
    print(separador)


# ── Punto de entrada ─────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Pruebas end-to-end del agente LangGraph desde terminal."
    )
    grupo = parser.add_mutually_exclusive_group()
    grupo.add_argument(
        "--caso",
        type=int,
        choices=range(1, len(CASOS) + 1),
        metavar=f"1-{len(CASOS)}",
        help="Ejecuta un caso concreto (1 = primero).",
    )
    grupo.add_argument(
        "--todos",
        action="store_true",
        help="Ejecuta todos los casos de prueba.",
    )
    parser.add_argument(
        "--silencioso",
        action="store_true",
        help="Suprime el detalle de nodos y mensajes intermedios.",
    )
    args = parser.parse_args()

    verbose = not args.silencioso

    if args.caso:
        casos_a_ejecutar = [CASOS[args.caso - 1]]
    elif args.todos:
        casos_a_ejecutar = CASOS
    else:
        # Por defecto ejecuta los dos primeros casos (los del criterio de aceptación)
        casos_a_ejecutar = CASOS[:2]
        print("(Ejecutando casos 1 y 2 por defecto. Usa --todos para ejecutar todos.)")

    resultados: list[tuple[str, bool]] = []
    for caso in casos_a_ejecutar:
        ok = _run_case(caso, verbose=verbose)
        resultados.append((caso.nombre, ok))

    if len(resultados) > 1:
        _print_summary(resultados)

    # Código de salida no-cero si algún caso falla
    if not all(ok for _, ok in resultados):
        sys.exit(1)


if __name__ == "__main__":
    main()

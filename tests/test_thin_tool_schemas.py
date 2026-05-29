"""Tests del schema fino que ve el modelo (Parte 2, Opción A).

`thin_tool_schemas` adelgaza el schema de las herramientas analíticas que se
envía al LLM: conserva nombre, descripción de la tool y nombres+tipos de los
parámetros, pero quita description/enum/default por parámetro y marca todos como
opcionales (``required: []``). Así el modelo SELECCIONA la tool pero no puede
"rellenar" sus parámetros: la recogida se centraliza en el nodo de petición.

`consultar_teoria` (RAG, no analítica) queda intacta porque su ``query`` la
reformula el LLM legítimamente.
"""

from __future__ import annotations

from src.agent.tool_metadata import ANALYTICAL_TOOL_NAMES
from src.agent.tools import AGENT_TOOLS
from src.config.llm_config import thin_tool_schemas


def _by_name(payload: list) -> dict:
    out: dict = {}
    for item in payload:
        if isinstance(item, dict):
            out[item["function"]["name"]] = ("thin", item)
        else:
            out[item.name] = ("full", item)
    return out


def test_analytical_tools_are_thinned_and_rag_is_not():
    payload = thin_tool_schemas(AGENT_TOOLS, ANALYTICAL_TOOL_NAMES)
    indexed = _by_name(payload)

    # consultar_teoria se pasa sin tocar (objeto tool original).
    assert indexed["consultar_teoria"][0] == "full"

    # Todas las analíticas llegan como dict adelgazado.
    for name in ANALYTICAL_TOOL_NAMES:
        kind, schema = indexed[name]
        assert kind == "thin", f"{name} debería venir adelgazada"
        params = schema["function"]["parameters"]
        assert params.get("required") == [], f"{name} no debería marcar required"
        for pname, pinfo in params.get("properties", {}).items():
            # Solo se conserva 'type' (o nada); nada de description/enum/default.
            assert set(pinfo.keys()) <= {"type"}, (
                f"{name}.{pname} filtró metadata de relleno: {pinfo}"
            )


def test_thin_schema_keeps_param_names_for_selection():
    """Conservar los nombres de parámetro no es relleno: ayuda a pasar lo que el
    usuario escribió. Verificamos que siguen presentes (p. ej. detect_drift)."""
    payload = thin_tool_schemas(AGENT_TOOLS, ANALYTICAL_TOOL_NAMES)
    indexed = _by_name(payload)
    props = indexed["detect_drift"][1]["function"]["parameters"]["properties"]
    assert {"file_path", "index_column", "method"} <= set(props)

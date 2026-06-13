"""Tests del schema fino que ve el modelo (Parte 2, Opción A).

thin_tool_schemas adelgaza el schema de las tools analíticas enviado al LLM: conserva
nombre, descripción y nombres+tipos de parámetros, pero quita description/enum/default
y marca todo opcional (required: []), así el modelo SELECCIONA la tool pero no rellena
parámetros. consultar_teoria (RAG) queda intacta porque su query la reformula el LLM.
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
    """Los nombres de parámetro se conservan (no son relleno): verificamos que siguen presentes (p. ej. detect_drift)."""
    payload = thin_tool_schemas(AGENT_TOOLS, ANALYTICAL_TOOL_NAMES)
    indexed = _by_name(payload)
    props = indexed["detect_drift"][1]["function"]["parameters"]["properties"]
    assert {"file_path", "index_column", "method"} <= set(props)

"""Tests del nodo recuperar_contexto.

Garantizan la invariante de la Fase C: la tool call de consultar_teoria nunca queda
huérfana. El nodo devuelve siempre un ToolMessage (camino feliz, "Error: …" del RAG o
excepción inesperada), nunca un error_info que la arista incondicional ignoraría.
"""

from __future__ import annotations

from langchain_core.messages import AIMessage, ToolMessage

import src.agent.nodes.rag_retrieval as rag_node
from src.agent.nodes.rag_retrieval import recuperar_contexto_node


def _state_with_rag_call(query: str = "¿qué es el PSI?", call_id: str = "call_abc"):
    """Estado con un AIMessage que contiene la tool call de consultar_teoria."""
    ai = AIMessage(
        content="",
        tool_calls=[{
            "name": "consultar_teoria",
            "args": {"query": query},
            "id": call_id,
            "type": "tool_call",
        }],
    )
    return {"messages": [ai]}, call_id


class _FakeTool:
    """Doble de consultar_teoria que devuelve un valor fijo o lanza."""

    def __init__(self, *, returns=None, raises=None):
        self._returns = returns
        self._raises = raises

    def invoke(self, _payload):
        if self._raises is not None:
            raise self._raises
        return self._returns


def test_rag_error_string_becomes_toolmessage(monkeypatch):
    """Un resultado 'Error: …' se devuelve como ToolMessage, no como error_info."""
    err = "Error: No fue posible consultar la base RAG local. Detalle tecnico: boom"
    monkeypatch.setattr(rag_node, "consultar_teoria", _FakeTool(returns=err))
    monkeypatch.setattr(rag_node, "pop_last_retrieval", lambda: None)

    state, call_id = _state_with_rag_call()
    out = recuperar_contexto_node(state)

    assert "error_info" not in out
    msgs = out["messages"]
    assert len(msgs) == 1 and isinstance(msgs[0], ToolMessage)
    assert msgs[0].name == "consultar_teoria"
    assert msgs[0].tool_call_id == call_id
    assert msgs[0].content == err


def test_rag_exception_becomes_error_toolmessage(monkeypatch):
    """Una excepción del RAG se convierte en ToolMessage de error (no se propaga)."""
    monkeypatch.setattr(rag_node, "consultar_teoria", _FakeTool(raises=RuntimeError("kaput")))
    monkeypatch.setattr(rag_node, "pop_last_retrieval", lambda: None)

    state, call_id = _state_with_rag_call()
    out = recuperar_contexto_node(state)

    assert "error_info" not in out
    msgs = out["messages"]
    assert len(msgs) == 1 and isinstance(msgs[0], ToolMessage)
    assert msgs[0].name == "consultar_teoria"
    assert msgs[0].tool_call_id == call_id
    assert msgs[0].content.startswith("Error:")
    assert "kaput" in msgs[0].content


def test_rag_success_returns_context_toolmessage(monkeypatch):
    """El camino feliz devuelve el contexto recuperado como ToolMessage."""
    ctx = (
        "Material de referencia recuperado del corpus del TFG ...\n\n"
        "Fuentes consultadas:\n- corpus.pdf | PSI | chunk 3"
    )
    monkeypatch.setattr(rag_node, "consultar_teoria", _FakeTool(returns=ctx))
    monkeypatch.setattr(rag_node, "pop_last_retrieval", lambda: None)

    state, call_id = _state_with_rag_call()
    out = recuperar_contexto_node(state)

    assert "error_info" not in out
    msgs = out["messages"]
    assert len(msgs) == 1 and isinstance(msgs[0], ToolMessage)
    assert msgs[0].content == ctx
    assert msgs[0].tool_call_id == call_id

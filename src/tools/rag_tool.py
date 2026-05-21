"""Herramienta RAG para consulta teorica del corpus de data drift."""

from __future__ import annotations

import os
from contextvars import ContextVar
from typing import Any

from langchain_core.documents import Document
from langchain_core.tools import tool

# Punto de entrada unico de recuperacion: densa / MMR / hibrida (BM25 + RRF).
from src.rag_engine.hybrid import recuperar_documentos, resolve_search_type

# Canal lateral asíncronamente seguro para transferir métricas entre hilos sin alterar firmas
_last_retrieval: ContextVar[dict[str, Any] | None] = ContextVar("last_retrieval", default=None)

# OPTIMIZACIÓN CRÍTICA DEL TFG: Ajustamos umbrales para mitigar los 44s de prompt ingestion
RAG_DEFAULT_KEEP_TOP = 3
RAG_MAX_CONTEXT_CHARS = 3500  # Reducido de 8000 para evitar saturar la ventana de Qwen 3B


def pop_last_retrieval() -> dict[str, Any] | None:
    """Extrae el último buffer analítico acumulado y limpia la ContextVar (patrón pop)."""
    try:
        val = _last_retrieval.get()
    except LookupError:
        val = None
    _last_retrieval.set(None)
    return val


def _safe_text(value: object | None) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text if text else None


def _format_hierarchy(metadata: dict[str, object]) -> str:
    sections: list[str] = []
    for key in ("header_1", "header_2", "header_3"):
        value = _safe_text(metadata.get(key))
        if value is not None:
            sections.append(value)
    return " > ".join(sections) if sections else "Sin encabezado detectado"


def _read_positive_int_env(variable_name: str, default_value: int) -> int:
    raw_value = os.getenv(variable_name)
    if raw_value is None or not raw_value.strip():
        return default_value
    try:
        value = int(raw_value)
        return value if value > 0 else default_value
    except ValueError:
        return default_value


def _build_context_fragments(documents: list[Document]) -> tuple[str, list[str]]:
    fragments: list[str] = []
    sources: list[str] = []
    for idx, doc in enumerate(documents, start=1):
        raw_metadata = doc.metadata if isinstance(doc.metadata, dict) else {}
        metadata: dict[str, object] = dict(raw_metadata)
        hierarchy = _format_hierarchy(metadata)
        source = _safe_text(metadata.get("source")) or "origen_desconocido"
        chunk_id = _safe_text(metadata.get("chunk_id")) or str(idx)
        content = (doc.page_content or "").strip()
        if not content:
            continue
        fragment = f"[FRAGMENTO {idx}]\nFuente: {source}\nJerarquia: {hierarchy}\nChunk ID: {chunk_id}\nContenido:\n{content}"
        fragments.append(fragment)
        sources.append(f"{source} | {hierarchy} | chunk {chunk_id}")

    context = "\n\n---\n\n".join(fragments)
    if len(context) > RAG_MAX_CONTEXT_CHARS:
        context = context[:RAG_MAX_CONTEXT_CHARS].rstrip()
    return context, sources


@tool
def consultar_teoria(query: str) -> str:
    """Consulta la base teorica del TFG y devuelve fragmentos como contexto documental.

    Devuelve el contexto recuperado SIN redactar una respuesta: la sintesis la
    realiza el razonador del grafo. Asi se evita la doble pasada de LLM
    (PlanMejoraRAG, Paso 5), que duplicaba la latencia del turno teorico.
    """
    clean_query = query.strip()
    if not clean_query:
        return "Error: La consulta esta vacia. Proporciona una pregunta valida."

    try:
        keep_top = _read_positive_int_env("RAG_KEEP_TOP", RAG_DEFAULT_KEEP_TOP)

        # Recuperacion via punto de entrada unico (densa / MMR / hibrida segun
        # RAG_SEARCH_TYPE). La fusion hibrida ya aporta la senal lexica (BM25),
        # por lo que se retira el reordenado lexico posterior: el barrido de
        # docs/rag_evaluation midio que aplicarlo empeora P@k, R@k y MRR.
        documents = recuperar_documentos(clean_query, top_k=keep_top)
    except Exception as exc:
        return f"Error: No fue posible consultar la base RAG local. Detalle tecnico: {exc}"

    if not documents:
        return "No se encontraron fragmentos relevantes para la consulta indicada."

    context, sources = _build_context_fragments(documents)
    if not context.strip():
        return "No se encontraron fragmentos textuales utiles para la consulta indicada."

    # ── CARGAR DATOS AL CANAL LATERAL (ContextVar) ───────────────────────────
    # Ya no hay LLM interno: el ContextVar deja de exponer inner_input_tokens /
    # inner_output_tokens. MMR / hibrido tampoco devuelven distancias nativas,
    # asi que se registra search_type en lugar de vector_scores.
    _last_retrieval.set({
        "query": clean_query,
        "n_chunks": len(documents),
        "search_type": resolve_search_type(),
        "vector_scores": [],
        "sources": sources,
    })
    # ─────────────────────────────────────────────────────────────────────────

    # Contexto estructurado: el razonador sintetiza la respuesta a partir de el.
    sources_block = "\n".join(f"- {source_line}" for source_line in sources)
    return f"{context}\n\nFuentes consultadas:\n{sources_block}"

"""Herramienta RAG para consulta teorica del corpus de data drift."""

from __future__ import annotations

import os
import re
from contextvars import ContextVar
from typing import Any, Optional

from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import tool

from src.config.llm_config import get_chat_ollama
# Importamos la función de base vectorial nativa que acabamos de crear
from src.rag_engine.retriever import get_vector_store

# Canal lateral asíncronamente seguro para transferir métricas entre hilos sin alterar firmas
_last_retrieval: ContextVar[dict[str, Any] | None] = ContextVar("last_retrieval", default=None)

# OPTIMIZACIÓN CRÍTICA DEL TFG: Ajustamos umbrales para mitigar los 44s de prompt ingestion
RAG_DEFAULT_TOP_K = 6
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


def _tokenize(text: str) -> set[str]:
    tokens = re.findall(r"[a-zA-Z0-9áéíóúñüÁÉÍÓÚÑÜ]+", text.lower())
    return {token for token in tokens if len(token) > 2}


def _lexical_overlap_score(query_tokens: set[str], content: str) -> float:
    if not query_tokens:
        return 0.0
    content_tokens = _tokenize(content)
    if not content_tokens:
        return 0.0
    return len(query_tokens.intersection(content_tokens)) / max(1, len(query_tokens))


def _rerank_documents(query: str, documents: list[Document], keep_top: int) -> list[Document]:
    if not documents:
        return documents
    query_tokens = _tokenize(query)
    ranked_pairs = sorted(
        (
            (_lexical_overlap_score(query_tokens, doc.page_content or ""), idx, doc)
            for idx, doc in enumerate(documents)
        ),
        key=lambda item: (item[0], -item[1]),
        reverse=True,
    )
    return [item[2] for item in ranked_pairs][: max(1, keep_top)]


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


def _generate_grounded_answer(query: str, context: str) -> tuple[str, dict[str, Any]]:
    """Genera la respuesta documental y extrae los metadatos de tokens de Ollama."""
    llm = get_chat_ollama()
    system_prompt = (
        "Eres un asistente tecnico de Data Drift. "
        "Responde UNICAMENTE con base en el contexto proporcionado."
    )
    user_prompt = f"Pregunta del usuario:\n{query.strip()}\n\nContexto recuperado:\n{context}"

    response = llm.invoke([SystemMessage(content=system_prompt), HumanMessage(content=user_prompt)])
    
    # Capturamos la metadata de uso del LLM interno de forma defensiva
    usage = getattr(response, "usage_metadata", None) or {}
    token_stats = {
        "inner_input_tokens": getattr(usage, "input_tokens", None),
        "inner_output_tokens": getattr(usage, "output_tokens", None)
    }

    answer = str(response.content).strip()
    return answer, token_stats


@tool
def consultar_teoria(query: str) -> str:
    """Consulta teoria tecnica del TFG de referencia para responder con base documental."""
    clean_query = query.strip()
    if not clean_query:
        return "Error: La consulta esta vacia. Proporciona una pregunta valida."

    try:
        top_k = _read_positive_int_env("RAG_TOP_K", RAG_DEFAULT_TOP_K)
        keep_top = _read_positive_int_env("RAG_KEEP_TOP", RAG_DEFAULT_KEEP_TOP)

        # Invocamos la búsqueda por similitud recuperando distancias vectoriales nativas
        vector_store = get_vector_store()
        docs_and_scores = vector_store.similarity_search_with_score(clean_query, k=top_k)
        
        documents = [doc for doc, _ in docs_and_scores]
        vector_scores = [round(float(score), 4) for _, score in docs_and_scores]
        
    except Exception as exc:
        return f"Error: No fue posible consultar la base RAG local. Detalle tecnico: {exc}"

    if not documents:
        return "No se encontraron fragmentos relevantes para la consulta indicada."

    selected_documents = _rerank_documents(query=clean_query, documents=documents, keep_top=keep_top)
    context, sources = _build_context_fragments(selected_documents)
    if not context.strip():
        return "No se encontraron fragmentos textuales utiles para construir una respuesta."

    try:
        # Recuperamos tanto la respuesta estructurada como el conteo de tokens ocultos
        answer, token_stats = _generate_grounded_answer(clean_query, context)
    except Exception as exc:
        return f"Error: Se recupero contexto pero fallo la sintesis con LLM. Detalle tecnico: {exc}"

    if not answer:
        return "No se pudo generar una respuesta final a partir del contexto recuperado."

    # ── CARGAR DATOS AL CANAL LATERAL (ContextVar) ───────────────────────────
    _last_retrieval.set({
        "query": clean_query,
        "n_chunks": len(documents),
        "vector_scores": vector_scores,
        "sources": sources[:keep_top],
        **token_stats
    })
    # ─────────────────────────────────────────────────────────────────────────

    sources_block = "\n".join(f"- {source_line}" for source_line in sources)
    return f"Respuesta:\n{answer}\n\nFuentes consultadas:\n{sources_block}"

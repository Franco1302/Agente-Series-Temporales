"""Herramienta RAG para consulta teorica del corpus de data drift."""

from __future__ import annotations

import os
import re

from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import tool

from src.config.llm_config import get_chat_ollama
from src.rag_engine.retriever import get_retriever


RAG_DEFAULT_TOP_K = 8
RAG_DEFAULT_KEEP_TOP = 4
RAG_MAX_CONTEXT_CHARS = 8000


def _safe_text(value: object | None) -> str | None:
    """Convierte cualquier valor a texto limpio o None si queda vacio."""
    if value is None:
        return None

    text = str(value).strip()
    if not text:
        return None

    return text


def _format_hierarchy(metadata: dict[str, object]) -> str:
    """Construye la jerarquia seccional usando encabezados markdown persistidos."""
    sections: list[str] = []
    for key in ("header_1", "header_2", "header_3"):
        value = _safe_text(metadata.get(key))
        if value is not None:
            sections.append(value)

    if not sections:
        return "Sin encabezado detectado"

    return " > ".join(sections)


def _read_positive_int_env(variable_name: str, default_value: int) -> int:
    """Lee un entero positivo desde entorno y aplica default si es invalido."""
    raw_value = os.getenv(variable_name)
    if raw_value is None or not raw_value.strip():
        return default_value

    try:
        value = int(raw_value)
    except ValueError:
        return default_value

    return value if value > 0 else default_value


def _tokenize(text: str) -> set[str]:
    """Tokeniza texto a minusculas para scoring lexico simple."""
    tokens = re.findall(r"[a-zA-Z0-9áéíóúñüÁÉÍÓÚÑÜ]+", text.lower())
    return {token for token in tokens if len(token) > 2}


def _lexical_overlap_score(query_tokens: set[str], content: str) -> float:
    """Calcula una puntuacion de solapamiento lexico para reranking local."""
    if not query_tokens:
        return 0.0

    content_tokens = _tokenize(content)
    if not content_tokens:
        return 0.0

    overlap_count = len(query_tokens.intersection(content_tokens))
    return overlap_count / max(1, len(query_tokens))


def _rerank_documents(query: str, documents: list[Document], keep_top: int) -> list[Document]:
    """Reordena documentos por overlap lexico para priorizar contexto relevante."""
    if not documents:
        return documents

    query_tokens = _tokenize(query)
    ranked_pairs = sorted(
        (
            (
                _lexical_overlap_score(query_tokens, doc.page_content or ""),
                idx,
                doc,
            )
            for idx, doc in enumerate(documents)
        ),
        key=lambda item: (item[0], -item[1]),
        reverse=True,
    )

    reranked = [item[2] for item in ranked_pairs]
    return reranked[: max(1, keep_top)]


def _build_context_fragments(documents: list[Document]) -> tuple[str, list[str]]:
    """Construye contexto RAG estructurado y lista de fuentes legibles."""
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

        fragment = "\n".join(
            [
                f"[FRAGMENTO {idx}]",
                f"Fuente: {source}",
                f"Jerarquia: {hierarchy}",
                f"Chunk ID: {chunk_id}",
                "Contenido:",
                content,
            ]
        )
        fragments.append(fragment)
        sources.append(f"{source} | {hierarchy} | chunk {chunk_id}")

    context = "\n\n---\n\n".join(fragments)
    if len(context) > RAG_MAX_CONTEXT_CHARS:
        context = context[:RAG_MAX_CONTEXT_CHARS].rstrip()

    return context, sources


def _generate_grounded_answer(query: str, context: str) -> str:
    """Genera respuesta final con LLM, forzando apoyo en evidencia recuperada."""
    llm = get_chat_ollama()

    system_prompt = (
        "Eres un asistente tecnico de Data Drift. "
        "Responde UNICAMENTE con base en el contexto proporcionado. "
        "Si el contexto es insuficiente o ambiguo, dilo explicitamente y pide mas detalle. "
        "No inventes metodos, formulas ni resultados que no aparezcan en el contexto. "
        "Responde en espanol, con precision tecnica y claridad didactica."
    )
    user_prompt = (
        "Pregunta del usuario:\n"
        f"{query.strip()}\n\n"
        "Contexto recuperado:\n"
        f"{context}\n\n"
        "Instrucciones de salida:\n"
        "1) Da una respuesta breve y precisa.\n"
        "2) Si aplica, incluye puntos clave en una lista corta.\n"
        "3) Si falta evidencia, indica exactamente que falta."
    )

    response = llm.invoke(
        [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt),
        ]
    )

    content = response.content
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        return "".join(str(part) for part in content).strip()
    return str(content).strip()


@tool
def consultar_teoria(query: str) -> str:
    """Consulta teoria tecnica del TFG de referencia para responder con base documental.

    Usa esta herramienta cuando necesites contexto fiable sobre:
    - fundamentos matematicos y estadisticos de Data Drift,
    - explicaciones conceptuales de deteccion de drift,
    - generacion de datos sinteticos,
    - tecnicas de aumentacion de datos,
    - modelos de series temporales y su evaluacion.

    La informacion recuperada proviene de la base vectorial local construida a partir
    del TFG de referencia. La salida esta estructurada por fragmentos e incluye
    jerarquia de secciones markdown (header_1, header_2, header_3) para preservar
    contexto semantico.

    Args:
        query: Pregunta o necesidad de contexto teorico a recuperar.

    Returns:
        Un texto unico con fragmentos relevantes, cada uno con fuente, jerarquia
        seccional y contenido. Si ocurre un problema, devuelve un mensaje de error
        en texto plano y nunca lanza excepciones al agente.
    """
    clean_query = query.strip()
    if not clean_query:
        return "Error: La consulta esta vacia. Proporciona una pregunta valida."

    try:
        top_k = _read_positive_int_env("RAG_TOP_K", RAG_DEFAULT_TOP_K)
        keep_top = _read_positive_int_env("RAG_KEEP_TOP", RAG_DEFAULT_KEEP_TOP)

        retriever = get_retriever(top_k=top_k)
        documents = retriever.invoke(clean_query)
    except FileNotFoundError as exc:
        return f"Error: {exc}"
    except Exception as exc:
        return (
            "Error: No fue posible consultar la base RAG local. "
            f"Detalle tecnico: {exc}"
        )

    if not documents:
        return "No se encontraron fragmentos relevantes para la consulta indicada."

    selected_documents = _rerank_documents(
        query=clean_query,
        documents=documents,
        keep_top=keep_top,
    )
    context, sources = _build_context_fragments(selected_documents)
    if not context.strip():
        return "No se encontraron fragmentos textuales utiles para construir una respuesta."

    try:
        answer = _generate_grounded_answer(clean_query, context)
    except Exception as exc:
        return (
            "Error: Se recupero contexto pero fallo la sintesis con LLM. "
            f"Detalle tecnico: {exc}"
        )

    if not answer:
        return "No se pudo generar una respuesta final a partir del contexto recuperado."

    sources_block = "\n".join(f"- {source_line}" for source_line in sources)
    return (
        f"Respuesta:\n{answer}\n\n"
        "Fuentes consultadas:\n"
        f"{sources_block}"
    )

"""Recuperacion RAG unificada: densa, MMR o hibrida (densa + BM25 con RRF).

``recuperar_documentos`` es el punto de entrada unico de recuperacion del
sistema: segun ``RAG_SEARCH_TYPE`` aplica busqueda densa por similitud, MMR
(diversidad) o busqueda **hibrida**. La hibrida fusiona la lista densa con una
lista lexica BM25 mediante **Reciprocal Rank Fusion** ponderado.

Restriccion del Bloque 2: 100 % local, sin modelos extra ni red para la parte
lexica. BM25 (``rank-bm25``, Python puro) se construye en memoria sobre el
texto ya persistido en Chroma; el indice se cachea por proceso (``lru_cache``)
porque el corpus es pequeno y estatico entre ingestas.

Por que RRF propio en vez de ``EnsembleRetriever``: evita arrastrar
``langchain``/``langchain-community`` y su friccion con ``langchain-core`` 1.3.x
ya instalado. RRF es ademas independiente de la escala de los scores (combina
rangos, no puntuaciones), lo que encaja con mezclar distancia coseno y BM25.
"""

from __future__ import annotations

import os
import re
from functools import lru_cache

from langchain_core.documents import Document
from rank_bm25 import BM25Okapi

from src.rag_engine.retriever import (
    DEFAULT_FETCH_K,
    DEFAULT_MMR_LAMBDA,
    DEFAULT_SEARCH_TYPE,
    _VALID_SEARCH_TYPES,
    _read_float_env,
    _read_positive_int_env,
    get_vector_store,
)

# Constante de amortiguacion de Reciprocal Rank Fusion (Cormack et al., 2009).
# El valor "estandar" 60 se penso para rankings profundos (miles de docs); con
# este corpus pequeno y fetch_k bajo aplana en exceso los rangos y degrada la
# recuperacion. El barrido de docs/rag_evaluation midio que 5 es el mejor para
# este corpus; se deja configurable via RAG_RRF_K.
DEFAULT_RRF_K = 5
# Pesos por defecto de la fusion hibrida: (densa, lexica). El barrido eligio
# 0.7/0.3 (la densa es la senal mas fiable; BM25 rescata consultas por palabra).
DEFAULT_HYBRID_WEIGHTS = (0.7, 0.3)


def _tokenize(text: str) -> list[str]:
    """Tokenizador lexico simple: minusculas, alfanumerico + acentos, longitud >= 2."""
    tokens = re.findall(r"[a-zA-Z0-9áéíóúñüÁÉÍÓÚÑÜ]+", text.lower())
    return [token for token in tokens if len(token) >= 2]


@lru_cache(maxsize=1)
def _load_bm25_index() -> tuple[BM25Okapi, tuple[Document, ...]]:
    """Construye (una vez por proceso) el indice BM25 sobre todo el corpus Chroma.

    Lee el texto y la metadata de cada chunk con ``collection.get`` y los
    reconstruye como ``Document`` para que la fusion devuelva documentos con
    trazabilidad completa. Cacheado: el corpus es pequeno y no cambia salvo
    re-ingesta (que implica reiniciar el proceso).
    """
    store = get_vector_store()
    raw = store.get(include=["documents", "metadatas"])
    contents = raw.get("documents") or []
    metadatas = raw.get("metadatas") or []

    documents: list[Document] = []
    tokenized_corpus: list[list[str]] = []
    for idx, content in enumerate(contents):
        text = content or ""
        metadata = metadatas[idx] if idx < len(metadatas) and metadatas[idx] else {}
        documents.append(Document(page_content=text, metadata=dict(metadata)))
        tokenized_corpus.append(_tokenize(text))

    if not documents:
        raise ValueError(
            "La coleccion Chroma esta vacia: ejecuta src/rag_engine/ingest.py."
        )

    return BM25Okapi(tokenized_corpus), tuple(documents)


def _doc_key(document: Document) -> str:
    """Clave estable para fusionar listas: ``chunk_id`` si existe; si no, el texto."""
    chunk_id = (document.metadata or {}).get("chunk_id")
    if chunk_id:
        return str(chunk_id)
    return f"_content:{hash(document.page_content)}"


def _bm25_search(query: str, top_n: int) -> list[Document]:
    """Recupera los ``top_n`` documentos de mayor puntuacion BM25 para la query."""
    query_tokens = _tokenize(query)
    if not query_tokens:
        return []
    index, documents = _load_bm25_index()
    scores = index.get_scores(query_tokens)
    ranked = sorted(range(len(documents)), key=lambda i: scores[i], reverse=True)
    return [documents[i] for i in ranked[:top_n]]


def _rrf_fuse(
    ranked_lists: list[tuple[list[Document], float]],
    top_k: int,
    rrf_k: int,
) -> list[Document]:
    """Fusiona varias listas rankeadas con Reciprocal Rank Fusion ponderado.

    Para cada documento suma, por lista, ``peso / (rrf_k + rango)``. Combina
    rangos (no scores), asi que es inmune a que distancia coseno y BM25 vivan
    en escalas distintas.
    """
    scores: dict[str, float] = {}
    doc_by_key: dict[str, Document] = {}
    for documents, weight in ranked_lists:
        for rank, document in enumerate(documents, start=1):
            key = _doc_key(document)
            scores[key] = scores.get(key, 0.0) + weight / (rrf_k + rank)
            doc_by_key.setdefault(key, document)
    ordered_keys = sorted(scores, key=lambda key: scores[key], reverse=True)
    return [doc_by_key[key] for key in ordered_keys[:top_k]]


def _read_hybrid_weights() -> tuple[float, float]:
    """Lee ``RAG_HYBRID_WEIGHTS`` ('densa,lexica'); default ``(0.5, 0.5)``."""
    raw = os.getenv("RAG_HYBRID_WEIGHTS")
    if not raw or not raw.strip():
        return DEFAULT_HYBRID_WEIGHTS
    parts = raw.split(",")
    if len(parts) != 2:
        return DEFAULT_HYBRID_WEIGHTS
    try:
        dense_weight = float(parts[0])
        lexical_weight = float(parts[1])
    except ValueError:
        return DEFAULT_HYBRID_WEIGHTS
    if dense_weight < 0 or lexical_weight < 0 or (dense_weight + lexical_weight) == 0:
        return DEFAULT_HYBRID_WEIGHTS
    return dense_weight, lexical_weight


def resolve_search_type(search_type: str | None = None) -> str:
    """Resuelve el modo de busqueda efectivo (argumento > ``RAG_SEARCH_TYPE`` > default)."""
    resolved = (search_type or os.getenv("RAG_SEARCH_TYPE") or DEFAULT_SEARCH_TYPE)
    resolved = resolved.strip().lower() or DEFAULT_SEARCH_TYPE
    return resolved if resolved in _VALID_SEARCH_TYPES else DEFAULT_SEARCH_TYPE


def recuperar_documentos(
    query: str,
    top_k: int = 4,
    search_type: str | None = None,
) -> list[Document]:
    """Punto de entrada unico de recuperacion RAG. Devuelve hasta ``top_k`` Documentos.

    Segun el modo resuelto (``search_type`` o ``RAG_SEARCH_TYPE``):
      - ``similarity``: busqueda densa por similitud de coseno.
      - ``mmr``: busqueda densa con Maximal Marginal Relevance (diversifica).
      - ``hybrid``: fusiona la lista densa y la lexica (BM25) con Reciprocal
        Rank Fusion ponderado por ``RAG_HYBRID_WEIGHTS``; cada canal aporta
        ``RAG_FETCH_K`` candidatos antes de fusionar.
    """
    clean_query = query.strip()
    if not clean_query or top_k <= 0:
        return []

    mode = resolve_search_type(search_type)
    store = get_vector_store()

    if mode == "mmr":
        fetch_k = _read_positive_int_env("RAG_FETCH_K", DEFAULT_FETCH_K)
        lambda_mult = _read_float_env("RAG_MMR_LAMBDA", DEFAULT_MMR_LAMBDA)
        return store.max_marginal_relevance_search(
            clean_query, k=top_k, fetch_k=fetch_k, lambda_mult=lambda_mult
        )

    if mode == "hybrid":
        fetch_k = _read_positive_int_env("RAG_FETCH_K", DEFAULT_FETCH_K)
        rrf_k = _read_positive_int_env("RAG_RRF_K", DEFAULT_RRF_K)
        dense_docs = store.similarity_search(clean_query, k=fetch_k)
        lexical_docs = _bm25_search(clean_query, fetch_k)
        dense_weight, lexical_weight = _read_hybrid_weights()
        return _rrf_fuse(
            [(dense_docs, dense_weight), (lexical_docs, lexical_weight)],
            top_k=top_k,
            rrf_k=rrf_k,
        )

    # similarity (modo por defecto)
    return store.similarity_search(clean_query, k=top_k)

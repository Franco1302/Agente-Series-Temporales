"""Construccion de retriever RAG local en modo lectura sobre Chroma."""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_core.vectorstores import VectorStoreRetriever
from langchain_ollama import OllamaEmbeddings

PROJECT_ROOT = Path(__file__).resolve().parents[2]
ENV_FILE = PROJECT_ROOT / ".env"
VECTOR_DB_DIR = PROJECT_ROOT / "data" / "vector_db"
EMBEDDING_MODEL = "nomic-embed-text"

# Configuracion de busqueda (Paso 3 PlanMejoraRAG). Defaults sensatos: si no
# se configura nada en .env el comportamiento es identico al historico.
DEFAULT_SEARCH_TYPE = "similarity"
DEFAULT_FETCH_K = 20
DEFAULT_MMR_LAMBDA = 0.5
_VALID_SEARCH_TYPES = {"similarity", "mmr", "hybrid"}


def _read_positive_int_env(name: str, default: int) -> int:
    """Lee una variable de entorno entera positiva; si no es valida, el default."""
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    try:
        value = int(raw)
    except ValueError:
        return default
    return value if value > 0 else default


def _read_float_env(name: str, default: float) -> float:
    """Lee una variable de entorno de tipo float; si no es valida, el default."""
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def _load_ollama_base_url() -> str:
    """Lee OLLAMA_BASE_URL desde .env y valida su valor."""
    if not ENV_FILE.exists():
        raise FileNotFoundError(
            f"No se encontro el archivo de entorno en: {ENV_FILE}."
        )

    load_dotenv(dotenv_path=ENV_FILE, override=False)

    base_url = os.getenv("OLLAMA_BASE_URL")
    if base_url is None or not base_url.strip():
        raise ValueError("OLLAMA_BASE_URL no esta definido en .env.")

    return base_url.strip().rstrip("/")


def _ensure_vector_db_ready(vector_db_dir: Path) -> None:
    """Verifica que la base vectorial exista y tenga contenido persistido."""
    if not vector_db_dir.exists() or not vector_db_dir.is_dir():
        raise FileNotFoundError(
            "Base vectorial no encontrada en data/vector_db. "
            "Ejecuta primero src/rag_engine/ingest.py."
        )

    if not any(vector_db_dir.iterdir()):
        raise FileNotFoundError(
            "La carpeta data/vector_db esta vacia. "
            "Ejecuta primero src/rag_engine/ingest.py."
        )


def _resolve_search_config(
    search_type: str | None,
    search_kwargs: dict | None,
    top_k: int,
) -> tuple[str, dict]:
    """Resuelve el tipo de busqueda y los search_kwargs para ``as_retriever``.

    El tipo se toma del argumento o, si es ``None``, de ``RAG_SEARCH_TYPE``
    (.env); por defecto ``similarity``. ``hybrid`` es un modo del Paso 4
    (fusion densa + BM25 en ``hybrid.py``): aqui el retriever solo aporta la
    mitad densa, asi que se trata como ``similarity``. Para ``mmr`` se anaden
    ``fetch_k`` y ``lambda_mult`` desde ``RAG_FETCH_K`` / ``RAG_MMR_LAMBDA``
    salvo que el llamante los pase explicitamente en ``search_kwargs``.
    """
    requested = (search_type or os.getenv("RAG_SEARCH_TYPE") or DEFAULT_SEARCH_TYPE)
    requested = requested.strip().lower() or DEFAULT_SEARCH_TYPE
    if requested not in _VALID_SEARCH_TYPES:
        requested = DEFAULT_SEARCH_TYPE

    # Chroma.as_retriever solo entiende 'similarity' y 'mmr'.
    chroma_type = "mmr" if requested == "mmr" else "similarity"

    kwargs: dict = dict(search_kwargs or {})
    kwargs.setdefault("k", top_k)
    if chroma_type == "mmr":
        kwargs.setdefault("fetch_k", _read_positive_int_env("RAG_FETCH_K", DEFAULT_FETCH_K))
        kwargs.setdefault("lambda_mult", _read_float_env("RAG_MMR_LAMBDA", DEFAULT_MMR_LAMBDA))
    return chroma_type, kwargs


def get_retriever(
    top_k: int = 4,
    search_type: str | None = None,
    search_kwargs: dict | None = None,
) -> VectorStoreRetriever:
    """Devuelve un retriever conectado a Chroma en modo lectura.

    Parametros:
        top_k: numero de documentos a devolver.
        search_type: ``similarity`` | ``mmr`` | ``hybrid``. Si es ``None`` se
            lee de ``RAG_SEARCH_TYPE`` (.env); ``hybrid`` usa la mitad densa.
        search_kwargs: sobreescribe puntualmente ``k`` / ``fetch_k`` /
            ``lambda_mult`` (lo no indicado se rellena con los defaults o .env).
    """
    if top_k <= 0:
        raise ValueError("top_k debe ser mayor que 0.")

    _ensure_vector_db_ready(VECTOR_DB_DIR)
    ollama_base_url = _load_ollama_base_url()

    embeddings = OllamaEmbeddings(
        model=EMBEDDING_MODEL,
        base_url=ollama_base_url,
    )

    # Conexion a una coleccion existente; sin operaciones de escritura.
    vector_store = Chroma(
        collection_name="langchain",
        embedding_function=embeddings,
        persist_directory=str(VECTOR_DB_DIR),
        create_collection_if_not_exists=False,
    )

    chroma_type, resolved_kwargs = _resolve_search_config(search_type, search_kwargs, top_k)
    return vector_store.as_retriever(
        search_type=chroma_type,
        search_kwargs=resolved_kwargs,
    )

def get_vector_store() -> Chroma:
    """Devuelve la instancia nativa de la base de datos Chroma en modo lectura.

    Permite al subsistema de observabilidad ejecutar búsquedas complejas
    conservando los scores de distancia vectorial para la memoria del TFG.
    """
    _ensure_vector_db_ready(VECTOR_DB_DIR)
    ollama_base_url = _load_ollama_base_url()

    embeddings = OllamaEmbeddings(
        model=EMBEDDING_MODEL,
        base_url=ollama_base_url,
    )

    return Chroma(
        collection_name="langchain",
        embedding_function=embeddings,
        persist_directory=str(VECTOR_DB_DIR),
        create_collection_if_not_exists=False,
    )


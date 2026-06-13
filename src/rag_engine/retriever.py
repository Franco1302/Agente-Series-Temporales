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

# Defaults sensatos: sin nada configurado en .env el comportamiento es identico al historico.
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


def get_retriever(top_k: int = 4) -> VectorStoreRetriever:
    """Devuelve un retriever por similitud conectado a Chroma en modo lectura."""
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

    return vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": top_k},
    )

def get_vector_store() -> Chroma:
    """Devuelve la instancia nativa de Chroma en modo lectura (permite a la observabilidad ejecutar busquedas conservando los scores de distancia vectorial)."""
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


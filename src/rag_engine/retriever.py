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

"""Motor de ingesta RAG local para PDF tecnico con preservacion semantica.

Flujo implementado:
1. Carga de configuracion desde .env (OLLAMA_BASE_URL).
2. Extraccion completa de PDF a Markdown usando pymupdf4llm.
3. Segmentacion semantica por encabezados Markdown (#, ##, ###).
4. Segmentacion por tamano con solapamiento para mejorar recuperacion.
5. Generacion de embeddings con Ollama (nomic-embed-text).
6. Persistencia de vectores en Chroma en ./data/vector_db.
"""

from __future__ import annotations

import os
from pathlib import Path

import pymupdf4llm
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_ollama import OllamaEmbeddings
from langchain_text_splitters import (
    MarkdownHeaderTextSplitter,
    RecursiveCharacterTextSplitter,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PDF_PATH = PROJECT_ROOT / "data" / "knowledge_base" / "TFG_RamirezPalaciosDavid.pdf"
VECTOR_DB_DIR = PROJECT_ROOT / "data" / "vector_db"
EMBEDDING_MODEL = "nomic-embed-text"


def _load_ollama_base_url(project_root: Path) -> str:
    """Carga la URL de Ollama desde .env y valida su presencia."""
    env_file = project_root / ".env"
    if not env_file.exists():
        raise FileNotFoundError(
            f"No se encontro el archivo .env en la raiz del proyecto: {env_file}"
        )

    load_dotenv(dotenv_path=env_file, override=False)

    base_url = os.getenv("OLLAMA_BASE_URL")
    if base_url is None or not base_url.strip():
        raise ValueError(
            "No se encontro OLLAMA_BASE_URL en .env. Configura esta variable antes de ingerir."
        )

    return base_url.strip().rstrip("/")


def _extract_markdown(pdf_path: Path) -> str:
    """Extrae el PDF completo a un unico string Markdown."""
    if not pdf_path.exists():
        raise FileNotFoundError(f"No existe el PDF de ingesta: {pdf_path}")

    print("Extrayendo Markdown...")
    markdown_text = pymupdf4llm.to_markdown(str(pdf_path))
    if not isinstance(markdown_text, str) or not markdown_text.strip():
        raise ValueError("La extraccion a Markdown devolvio contenido vacio.")

    return markdown_text


def _split_markdown(markdown_text: str, source_name: str) -> list[Document]:
    """Realiza split semantico por encabezados y split recursivo por tamaño."""
    print("Aplicando corte semantico por encabezados Markdown...")
    headers_to_split_on = [
        ("#", "header_1"),
        ("##", "header_2"),
        ("###", "header_3"),
    ]
    header_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=headers_to_split_on,
        strip_headers=False,
    )
    header_documents: list[Document] = header_splitter.split_text(markdown_text)

    print("Aplicando corte por tamano (chunk_size=1000, chunk_overlap=200)...")
    recursive_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
    )
    chunked_documents: list[Document] = recursive_splitter.split_documents(header_documents)

    if not chunked_documents:
        raise ValueError("No se generaron chunks tras aplicar los splitters.")

    # Enriquecemos metadata para trazabilidad de origen y orden de chunk.
    for idx, doc in enumerate(chunked_documents, start=1):
        doc.metadata["source"] = source_name
        doc.metadata["chunk_id"] = idx

    return chunked_documents


def run_ingestion(
    pdf_path: Path = DEFAULT_PDF_PATH,
    vector_db_dir: Path = VECTOR_DB_DIR,
) -> None:
    """Ejecuta la ingesta completa del PDF hacia Chroma."""
    ollama_base_url = _load_ollama_base_url(PROJECT_ROOT)
    markdown_text = _extract_markdown(pdf_path)
    documents = _split_markdown(markdown_text, source_name=pdf_path.name)

    print(f"Chunks generados: {len(documents)}")
    print("Generando embeddings...")
    embeddings = OllamaEmbeddings(
        model=EMBEDDING_MODEL,
        base_url=ollama_base_url,
    )

    vector_db_dir.mkdir(parents=True, exist_ok=True)
    _ = Chroma.from_documents(
        documents=documents,
        embedding=embeddings,
        persist_directory=str(vector_db_dir),
    )

    print(f"Guardado exitoso en: {vector_db_dir}")


def main() -> None:
    """Punto de entrada CLI del proceso de ingesta."""
    try:
        run_ingestion()
    except FileNotFoundError as exc:
        print(f"[ERROR] {exc}")
        raise
    except Exception as exc:
        print(f"[ERROR] Fallo durante la ingesta RAG: {exc}")
        raise


if __name__ == "__main__":
    main()

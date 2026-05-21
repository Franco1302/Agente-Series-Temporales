"""Motor de ingesta RAG local multi-documento con preservacion semantica.

Flujo implementado:
1. Carga de configuracion desde .env (OLLAMA_BASE_URL).
2. Recorrido de todos los .pdf y .md de data/knowledge_base/ (PDF a Markdown
   via pymupdf4llm; .md leido directo).
3. Segmentacion semantica por encabezados Markdown (#, ##, ###).
4. Segmentacion por tamano con solapamiento para mejorar recuperacion.
5. Generacion de embeddings con Ollama (nomic-embed-text).
6. Persistencia de vectores en Chroma en ./data/vector_db.

Idempotencia: la ingesta es de tipo "recrear". Al inicio de run_ingestion se
borra por completo data/vector_db/ (si existe) y se reconstruye desde cero. Es
la estrategia mas simple y segura para un corpus pequeno: evita duplicados y
deja la coleccion siempre consistente con el contenido actual de
data/knowledge_base/.
"""

from __future__ import annotations

import os
import shutil
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
KNOWLEDGE_BASE_DIR = PROJECT_ROOT / "data" / "knowledge_base"
VECTOR_DB_DIR = PROJECT_ROOT / "data" / "vector_db"
EMBEDDING_MODEL = "nomic-embed-text"

# Heuristica de clasificacion por nombre de fichero. La guia de seleccion
# aporta criterios de DECISION; el resto del corpus describe teoria.
GUIA_DECISION_FILENAME = "guia_seleccion_metodos.md"


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


def _classify_doc_type(source_name: str) -> str:
    """Clasifica un documento en 'guia_decision' o 'teoria' por su nombre."""
    if source_name == GUIA_DECISION_FILENAME:
        return "guia_decision"
    return "teoria"


def _extract_markdown(doc_path: Path) -> str:
    """Obtiene el contenido Markdown de un documento (.pdf o .md)."""
    if not doc_path.exists():
        raise FileNotFoundError(f"No existe el documento de ingesta: {doc_path}")

    if doc_path.suffix.lower() == ".pdf":
        print(f"  Extrayendo Markdown de PDF: {doc_path.name}")
        markdown_text = pymupdf4llm.to_markdown(str(doc_path))
    else:
        print(f"  Leyendo Markdown: {doc_path.name}")
        markdown_text = doc_path.read_text(encoding="utf-8")

    if not isinstance(markdown_text, str) or not markdown_text.strip():
        raise ValueError(f"El documento {doc_path.name} no aporto contenido.")

    return markdown_text


def _split_markdown(markdown_text: str, source_name: str) -> list[Document]:
    """Realiza split semantico por encabezados y split recursivo por tamaño.

    Enriquece cada chunk con metadata de trazabilidad: 'source', 'doc_type'
    (clasificacion por nombre de fichero) y 'chunk_id' unico por documento
    (prefijado con el nombre del fichero para evitar colisiones entre docs).
    """
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

    recursive_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
    )
    chunked_documents: list[Document] = recursive_splitter.split_documents(header_documents)

    if not chunked_documents:
        raise ValueError(
            f"No se generaron chunks para {source_name} tras aplicar los splitters."
        )

    doc_type = _classify_doc_type(source_name)
    # chunk_id unico por documento: prefijo con el nombre de fichero evita
    # colisiones cuando varios documentos se ingieren a la misma coleccion.
    for idx, doc in enumerate(chunked_documents, start=1):
        doc.metadata["source"] = source_name
        doc.metadata["doc_type"] = doc_type
        doc.metadata["chunk_id"] = f"{source_name}#{idx}"

    return chunked_documents


def _discover_documents(knowledge_base_dir: Path) -> list[Path]:
    """Lista los documentos ingeribles (.pdf y .md) de la base de conocimiento."""
    if not knowledge_base_dir.is_dir():
        raise FileNotFoundError(
            f"No existe el directorio de la base de conocimiento: {knowledge_base_dir}"
        )

    documents = sorted(
        path
        for path in knowledge_base_dir.iterdir()
        if path.is_file() and path.suffix.lower() in {".pdf", ".md"}
    )

    if not documents:
        raise ValueError(
            f"No se encontraron documentos .pdf ni .md en {knowledge_base_dir}."
        )

    return documents


def run_ingestion(
    knowledge_base_dir: Path = KNOWLEDGE_BASE_DIR,
    vector_db_dir: Path = VECTOR_DB_DIR,
) -> None:
    """Ejecuta la ingesta completa de la base de conocimiento hacia Chroma.

    Estrategia idempotente de tipo "recrear": borra vector_db_dir si existe y
    reconstruye la coleccion desde cero a partir de todos los .pdf y .md de
    knowledge_base_dir (.gitkeep y otros formatos se ignoran).
    """
    ollama_base_url = _load_ollama_base_url(PROJECT_ROOT)
    document_paths = _discover_documents(knowledge_base_dir)

    print(f"Documentos detectados en {knowledge_base_dir.name}/: {len(document_paths)}")

    all_chunks: list[Document] = []
    for doc_path in document_paths:
        markdown_text = _extract_markdown(doc_path)
        chunks = _split_markdown(markdown_text, source_name=doc_path.name)
        all_chunks.extend(chunks)
        print(f"  -> {doc_path.name}: {len(chunks)} chunks")

    print(f"Total de chunks generados: {len(all_chunks)}")

    # Idempotencia: recrear la coleccion desde cero.
    if vector_db_dir.exists():
        print(f"Borrando coleccion previa: {vector_db_dir}")
        shutil.rmtree(vector_db_dir)

    print("Generando embeddings...")
    embeddings = OllamaEmbeddings(
        model=EMBEDDING_MODEL,
        base_url=ollama_base_url,
    )

    vector_db_dir.mkdir(parents=True, exist_ok=True)
    _ = Chroma.from_documents(
        documents=all_chunks,
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

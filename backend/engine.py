r"""RAG engine wiring for LlamaIndex, Ollama, and Milvus (Lite).

All lines are wrapped at 79 characters to comply with PEP 8.
"""
from __future__ import annotations

import logging
from pathlib import Path

import fitz  # PyMuPDF

from llama_index.core import (
    Settings,
    StorageContext,
    VectorStoreIndex,
)
from llama_index.core.schema import Document
from llama_index.core.readers import SimpleDirectoryReader
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.llms.ollama import Ollama
from llama_index.vector_stores.milvus import MilvusVectorStore
from llama_index.core.vector_stores.types import (
    ExactMatchFilter,
    MetadataFilters,
)

import config

logger = logging.getLogger(__name__)

_QUERY_ENGINE = None  # type: ignore[var-annotated]


def _storage_exists(path_str: str) -> bool:
    path = Path(path_str)
    return path.exists() and any(path.iterdir())


def init_settings() -> None:
    """Initialize global LlamaIndex settings for LLM and embeddings."""
    logger.info("=== Initializing LlamaIndex settings ===")
    logger.info(f"Setting up Ollama LLM: {config.OLLAMA_MODEL}")
    logger.info(f"Ollama base URL: {config.OLLAMA_BASE_URL}")

    Settings.llm = Ollama(
        model=config.OLLAMA_MODEL,
        base_url=config.OLLAMA_BASE_URL,
        request_timeout=600.0,
        system_prompt=(
            "You are a helpful assistant. Format ALL responses using proper markdown:\n"
            "- Use **bold** for vehicle names, key terms, and important specifications\n"
            "- Use ## headings to organize different sections (e.g., ## Engine Options, ## Dimensions)\n"
            "- Use bullet points (-) for lists of features, specifications, or options\n"
            "- Break content into clear, readable paragraphs with line breaks between topics\n"
            "- For measurements, use format: **Label**: value (e.g., **Length**: 5,155 mm)\n\n"
            "Always structure information clearly and use markdown formatting consistently."
        ),
    )

    logger.info(f"Setting up Ollama embeddings: {config.EMBED_MODEL}")
    Settings.embed_model = OllamaEmbedding(
        model_name=config.EMBED_MODEL,
        base_url=config.OLLAMA_BASE_URL,
        request_timeout=600.0,
    )
    logger.info("LlamaIndex settings initialized successfully (with markdown formatting)")


def _build_vector_store() -> MilvusVectorStore:
    logger.info(f"Building Milvus vector store at {config.MILVUS_URI}")
    logger.info(f"Collection name: {config.MILVUS_COLLECTION}")
    return MilvusVectorStore(
        uri=config.MILVUS_URI,
        collection_name=config.MILVUS_COLLECTION,
        overwrite=False,
    )


def _load_or_create_index() -> VectorStoreIndex:
    logger.info("=== Loading or creating vector index ===")
    vector_store = _build_vector_store()
    storage_context = StorageContext.from_defaults(
        vector_store=vector_store, persist_dir=config.STORAGE_DIR
    )

    if _storage_exists(config.STORAGE_DIR):
        logger.info(f"Found existing storage at {config.STORAGE_DIR}")
        logger.info("Loading index from vector store...")
        idx = VectorStoreIndex.from_vector_store(
            vector_store=vector_store,
            storage_context=storage_context,
        )
        logger.info("Index loaded successfully from existing storage")
        return idx

    # Create an empty index if no documents yet
    logger.warning(f"No storage found at {config.STORAGE_DIR}")
    logger.info("Creating empty index (no documents ingested yet)")
    empty_docs: list = []
    index = VectorStoreIndex.from_documents(
        empty_docs,
        storage_context=storage_context,
    )
    index.storage_context.persist(persist_dir=config.STORAGE_DIR)
    logger.info("Empty index created and persisted")
    return index


def get_query_engine(filters: MetadataFilters | None = None):
    global _QUERY_ENGINE
    if filters is None:
        if _QUERY_ENGINE is None:
            logger.info("=== Initializing query engine ===")
            index = _load_or_create_index()
            _QUERY_ENGINE = index.as_query_engine(
                similarity_top_k=5,
                streaming=False,
            )
            logger.info("Query engine initialized (similarity_top_k=5)")
        return _QUERY_ENGINE

    # Build a transient engine with filters for the current request
    index = _load_or_create_index()
    return index.as_query_engine(
        similarity_top_k=5,
        streaming=False,
        filters=filters,
    )


def get_llm():
    """Returns the configured LLM instance from Settings."""
    return Settings.llm


def _derive_car_model_and_doc_type(
    pdf_path: Path, root_dir: str
) -> tuple[str, str]:
    """Derive car_model and doc_type from directory structure.

    Expected layout: backend/data/<car_model>/<doc_type>/file.pdf
    Falls back to file stem for car_model and 'unknown' for doc_type.
    """
    rel = Path(pdf_path).relative_to(Path(root_dir))
    parts = rel.parts
    car_model = (
        parts[0].replace("_", " ")
        if len(parts) >= 1
        else Path(pdf_path).stem.replace("_", " ")
    )
    doc_type = parts[1] if len(parts) >= 2 else "unknown"
    return car_model, doc_type


def _load_pdfs_with_pymupdf(input_dir: str) -> list[Document]:
    base = Path(input_dir)
    docs: list[Document] = []
    pdf_files = list(base.rglob("*.pdf"))
    logger.info(f"Scanning for PDFs in {input_dir}: found {len(pdf_files)}")

    for pdf_path in pdf_files:
        logger.info(f"Processing PDF: {pdf_path.name}")
        try:
            with fitz.open(pdf_path) as pdf:
                page_count = pdf.page_count
                logger.info(f"  Pages: {page_count}")
                pages_with_text = 0
                for i in range(page_count):
                    page = pdf.load_page(i)
                    text = page.get_text("text").strip()
                    if not text:
                        continue
                    pages_with_text += 1
                    car_model, doc_type = _derive_car_model_and_doc_type(
                        pdf_path, input_dir
                    )
                    meta = {
                        "source_path": str(pdf_path),
                        "file_name": pdf_path.name,
                        "page": i + 1,
                        "type": "pdf",
                        "car_model": car_model,
                        "doc_type": doc_type,
                    }
                    docs.append(Document(text=text, metadata=meta))
                logger.info(f"  Extracted {pages_with_text} pages with text")
        except Exception as e:
            logger.error(f"  Failed to process {pdf_path.name}: {e}")
            continue
    logger.info(f"Total PDF documents extracted: {len(docs)}")
    return docs


def _load_text_docs(input_dir: str) -> list[Document]:
    logger.info(f"Scanning for text/markdown in {input_dir}")
    try:
        reader = SimpleDirectoryReader(
            input_dir=input_dir,
            required_exts=[".txt", ".md", ".markdown"],
            recursive=True,
        )
        docs = list(reader.load_data())
        logger.info(f"Total text documents extracted: {len(docs)}")
        return docs
    except ValueError as e:
        # No text files found, return empty list
        logger.info("No text/markdown files found (this is OK)")
        return []


def ingest_from_data_dir() -> int:
    """Ingests DATA_DIR using PyMuPDF for PDFs and reader for text/MD.

    Returns the number of loaded documents. If no documents are found, the
    function leaves the existing index as-is and returns 0.
    """
    logger.info("=== Starting ingestion from data directory ===")
    logger.info(f"Data directory: {config.DATA_DIR}")

    pdf_docs = _load_pdfs_with_pymupdf(config.DATA_DIR)
    text_docs = _load_text_docs(config.DATA_DIR)
    documents = [*pdf_docs, *text_docs]

    logger.info(
        f"Total documents collected: {len(documents)} "
        f"({len(pdf_docs)} PDF, {len(text_docs)} text)"
    )

    if not documents:
        logger.warning("No documents found to ingest!")
        return 0

    logger.info("Building vector index from documents...")
    vector_store = _build_vector_store()
    storage_context = StorageContext.from_defaults(
        vector_store=vector_store, persist_dir=config.STORAGE_DIR
    )

    logger.info("Creating embeddings (this may take a while)...")
    index = VectorStoreIndex.from_documents(
        documents,
        storage_context=storage_context,
    )

    logger.info(f"Persisting index to {config.STORAGE_DIR}...")
    index.storage_context.persist(persist_dir=config.STORAGE_DIR)

    # Refresh query engine after ingest
    logger.info("Refreshing query engine with new index...")
    global _QUERY_ENGINE
    _QUERY_ENGINE = index.as_query_engine(similarity_top_k=5, streaming=False)

    logger.info(f"=== Ingestion complete: {len(documents)} documents ===")
    return len(documents)


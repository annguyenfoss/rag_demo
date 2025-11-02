"""Minimal ingestion script for RAG data.

Put text/Markdown/PDF files under backend/data/ and run:

    python ingest.py

Lines kept under 79 chars.
"""
from __future__ import annotations

import logging

from engine.engine import ingest_from_data_dir, init_settings

logger = logging.getLogger(__name__)


def main() -> None:
    logger.info("=== RAG Data Ingestion Script ===")
    init_settings()
    count = ingest_from_data_dir()

    if count > 0:
        logger.info(f"✓ Successfully ingested {count} documents.")
    else:
        logger.warning("⚠ No documents were ingested.")


if __name__ == "__main__":
    main()


"""App configuration with sensible defaults.

All constants are kept under 79 characters per PEP 8.
"""
from pathlib import Path
import logging
import os

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)

logger = logging.getLogger(__name__)

ROOT_DIR = Path(__file__).resolve().parent

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen3:30b")
EMBED_MODEL = os.getenv("EMBED_MODEL", "nomic-embed-text")

MILVUS_URI = os.getenv("MILVUS_URI", str(ROOT_DIR / "milvus.db"))
MILVUS_COLLECTION = os.getenv("MILVUS_COLLECTION", "rag_collection")

STORAGE_DIR = os.getenv("STORAGE_DIR", str(ROOT_DIR / "storage"))
DATA_DIR = os.getenv("DATA_DIR", str(ROOT_DIR / "data"))

ALLOWED_ORIGINS = os.getenv(
    "ALLOWED_ORIGINS",
    "http://localhost:9200, http://127.0.0.1:9200",
)

logger.info("=== Configuration loaded ===")
logger.info(f"ROOT_DIR: {ROOT_DIR}")
logger.info(f"OLLAMA_BASE_URL: {OLLAMA_BASE_URL}")
logger.info(f"OLLAMA_MODEL: {OLLAMA_MODEL}")
logger.info(f"EMBED_MODEL: {EMBED_MODEL}")
logger.info(f"MILVUS_URI: {MILVUS_URI}")
logger.info(f"MILVUS_COLLECTION: {MILVUS_COLLECTION}")
logger.info(f"STORAGE_DIR: {STORAGE_DIR}")
logger.info(f"DATA_DIR: {DATA_DIR}")


def get_allowed_origins() -> list[str]:
    value = ALLOWED_ORIGINS
    parts = [p.strip() for p in value.split(",")]
    return [p for p in parts if p]


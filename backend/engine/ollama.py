"""Ollama (GPU/local) configuration for LlamaIndex Settings.

This module encapsulates all Ollama-specific wiring, including setting the
global LLM and embedding model on LlamaIndex Settings.
"""
from __future__ import annotations

import logging

from llama_index.core import Settings
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.ollama import OllamaEmbedding

import config

logger = logging.getLogger(__name__)


def configure_ollama_settings() -> None:
    """Configure LlamaIndex Settings to use local Ollama models."""
    logger.info(f"Setting up Ollama LLM: {config.OLLAMA_MODEL}")
    logger.info(f"Ollama base URL: {config.OLLAMA_BASE_URL}")

    Settings.llm = Ollama(
        model=config.OLLAMA_MODEL,
        base_url=config.OLLAMA_BASE_URL,
        request_timeout=600.0,
        system_prompt=(
            "You are a helpful assistant. Format ALL responses using "
            "proper markdown:\n"
            "- Use **bold** for vehicle names, key terms, and important "
            "specifications\n"
            "- Use ## headings to organize different sections (e.g., "
            "## Engine Options, ## Dimensions)\n"
            "- Use bullet points (-) for lists of features, specifications, "
            "or options\n"
            "- Break content into clear, readable paragraphs with line breaks "
            "between topics\n"
            "- For measurements, use format: **Label**: value (e.g., "
            "**Length**: 5,155 mm)\n\n"
            "Always structure information clearly and use markdown formatting "
            "consistently."
        ),
    )

    logger.info(f"Setting up Ollama embeddings: {config.EMBED_MODEL}")
    Settings.embed_model = OllamaEmbedding(
        model_name=config.EMBED_MODEL,
        base_url=config.OLLAMA_BASE_URL,
        request_timeout=600.0,
    )
    logger.info("LlamaIndex settings initialized (Ollama, GPU mode)")



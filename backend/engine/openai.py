"""OpenAI (wrapper mode) configuration for LlamaIndex Settings.

This module encapsulates all OpenAI-specific wiring, including setting the
global LLM and embedding model on LlamaIndex Settings.
"""
from __future__ import annotations

import logging

from llama_index.core import Settings
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding

import config

logger = logging.getLogger(__name__)


def configure_openai_settings() -> None:
    """Configure LlamaIndex Settings to use OpenAI models.

    Uses GPT-4o for the LLM and text-embedding-3-* for embeddings by default,
    taking values from config.OPENAI_MODEL and config.OPENAI_EMBED_MODEL.
    """
    logger.info(f"Setting up OpenAI LLM: {config.OPENAI_MODEL}")
    Settings.llm = OpenAI(
        model=config.OPENAI_MODEL,
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

    logger.info(
        f"Setting up OpenAI embeddings: {config.OPENAI_EMBED_MODEL}"
    )
    Settings.embed_model = OpenAIEmbedding(
        model=config.OPENAI_EMBED_MODEL,
    )
    logger.info("LlamaIndex settings initialized (OpenAI, wrapper mode)")



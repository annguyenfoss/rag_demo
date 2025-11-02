"""FastAPI application entrypoint.

Run with: python main.py

Conforms to PEP 8 with 79 char lines.
"""
import logging
import os
import uvicorn
from contextlib import asynccontextmanager
from typing import Optional
from pathlib import Path
from collections import defaultdict

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

import config
from engine.engine import get_query_engine, init_settings, get_llm
from llama_index.core.vector_stores.types import (
    ExactMatchFilter,
    MetadataFilters,
)

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    logger.info("=== FastAPI application starting ===")
    init_settings()
    logger.info("Warming up query engine...")
    get_query_engine()
    logger.info("=== FastAPI application ready ===")
    logger.info("Server is ready to accept requests")
    yield
    # Shutdown (if needed)
    logger.info("Application shutting down")


class ChatRequest(BaseModel):
    message: str
    session_id: Optional[str] = None


class ChatResponse(BaseModel):
    answer: str


app = FastAPI(title="RAG Q&A API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=config.get_allowed_origins(),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


AVAILABLE_MODELS = sorted(
    [p.name.replace("_", " ") for p in Path(config.DATA_DIR).iterdir()
     if p.is_dir()]
) or sorted(
    {p.stem.replace("_", " ") for p in Path(config.DATA_DIR).glob("*.pdf")}
)

SESSION_STATE: dict[str, dict[str, str]] = defaultdict(dict)


def _filters_for_car_model(model: str) -> MetadataFilters:
    return MetadataFilters(
        filters=[ExactMatchFilter(key="car_model", value=model)]
    )


def _classify_intent_llm(message: str) -> str:
    """Return 'LIST_MODELS' or 'QUESTION' using constrained LLM output."""
    llm = get_llm()
    prompt = (
        "Classify the user's intent as exactly one of:\n"
        "- LIST_MODELS: user asks what models are available / how many models / list cars\n"
        "- QUESTION: anything else\n\n"
        f"Message: {message}\n"
        "Answer with ONLY one token: LIST_MODELS or QUESTION."
    )
    out = str(llm.complete(prompt)).strip().upper()
    return "LIST_MODELS" if "LIST_MODELS" in out else "QUESTION"


def _choose_model_llm(message: str) -> str | None:
    """Pick one model from AVAILABLE_MODELS, or None if unknown."""
    if not AVAILABLE_MODELS:
        return None
    llm = get_llm()
    options = "\n".join(f"- {m}" for m in AVAILABLE_MODELS)
    prompt = (
        "From the user message, choose exactly ONE car model from the allowed "
        "list. If unsure, respond with ONLY 'UNKNOWN'.\n\n"
        f"Allowed models:\n{options}\n\n"
        f"User message: {message}\n\n"
        "Answer with exactly one line: one of the options as-is, or UNKNOWN."
    )
    out = str(llm.complete(prompt)).strip()
    return out if out in AVAILABLE_MODELS else None


def _should_trim_to_not_found(text: str) -> bool:
    """Heuristic: detect verbose 'no info in context' style answers."""
    if not text:
        return True
    t = text.strip().lower()
    patterns = [
        "does not contain any information",
        "does not contain information",
        "no information about",
        "no information on",
        "no reference to",
        "no relevant information",
        "couldn't find information",
        "could not find information",
        "unable to find information",
        "no mention of",
        "the provided context",
        "the context provided",
        "based on the provided documents",
        "nothing in the context",
        "there is no information",
    ]
    hits = sum(1 for p in patterns if p in t)
    return hits >= 2


def _extract_topic_llm(message: str) -> str | None:
    """Extract the main non-vehicle topic in 1–3 words; None if unknown."""
    llm = get_llm()
    options = ", ".join(AVAILABLE_MODELS) if AVAILABLE_MODELS else ""
    prompt = (
        "Extract the main non-vehicle topic of the user's question in 1-3 words. "
        "If it's only about a car model, respond with ONLY 'UNKNOWN'.\n\n"
        f"Available car models: {options}\n"
        f"Message: {message}\n\n"
        "Answer with one short noun phrase only, or UNKNOWN."
    )
    out = str(llm.complete(prompt)).strip()
    if not out or out.upper() == "UNKNOWN":
        return None
    return out if out not in AVAILABLE_MODELS else None


def _not_found_message(message: str, car_model: str) -> str:
    topic = _extract_topic_llm(message) or "this topic"
    return (
        "I couldn't find relevant information related to your question about "
        f"{topic}, in regard to our context of {car_model}"
    )


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/api/chat", response_model=ChatResponse)
async def chat(req: ChatRequest) -> ChatResponse:
    logger.info("=== Incoming chat request ===")
    logger.info(f"User message: {req.message[:100]}...")

    sid = req.session_id or "default"
    state = SESSION_STATE[sid]

    # Handle inventory/listing without hardcoded phrases
    if _classify_intent_llm(req.message) == "LIST_MODELS":
        models_csv = ", ".join(AVAILABLE_MODELS)
        text = (
            f"I can access {len(AVAILABLE_MODELS)} models: {models_csv}.\n\n"
            f"Which model are you asking about?"
        )
        return ChatResponse(answer=text)

    picked = _choose_model_llm(req.message)
    if picked:
        state["car_model"] = picked

    car_model = state.get("car_model")
    if not car_model:
        models_csv = ", ".join(AVAILABLE_MODELS)
        text = (
            "Please specify which car model you mean so I can retrieve the "
            f"correct documents: {models_csv}."
        )
        return ChatResponse(answer=text)

    logger.info(
        f"Routing to RAG engine with filter car_model='{car_model}'..."
    )
    engine = get_query_engine(filters=_filters_for_car_model(car_model))
    response = engine.query(req.message)
    
    # DEBUG: Log retrieved sources
    if hasattr(response, "source_nodes"):
        logger.info(
            f"DEBUG: Retrieved {len(response.source_nodes)} "
            f"source documents"
        )
        for i, node in enumerate(response.source_nodes[:3], 1):
            metadata = (
                node.node.metadata
                if hasattr(node.node, "metadata")
                else {}
            )
            score = node.score if hasattr(node, "score") else "N/A"
            logger.info(
                f"  Source {i} (score={score}): "
                f"{metadata.get('file_name', 'unknown')} "
                f"(page {metadata.get('page', 'N/A')})"
            )
    else:
        logger.warning("DEBUG: No source_nodes in response!")
    
    text = getattr(response, "response", None)
    if not text:
        text = str(response)

    # Validate and return response
    logger.info(f"Raw response length: {len(text) if text else 0} chars")

    if not text or not text.strip():
        text = _not_found_message(req.message, car_model)
        logger.warning("Empty response, using fallback message")
    else:
        if _should_trim_to_not_found(text):
            text = _not_found_message(req.message, car_model)
            logger.info("Trimming verbose 'no info' response to concise message")
        else:
            logger.info(f"Response preview: {text[:100]}...")

    logger.info("=== Chat request complete ===")
    return ChatResponse(answer=text)


if __name__ == "__main__":
    # Run with: python main.py
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "9100"))
    
    logger.info(f"Starting server on {host}:{port}")
    uvicorn.run(
        "main:app",
        host=host,
        port=port,
        reload=True,
        log_level="info",
    )


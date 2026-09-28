"""Loads and validates environment variables and holds project-wide constants."""

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env")

# ---- Paths and model settings ----
PDF_PATH = PROJECT_ROOT / "data" / "Ebook-Agentic-AI.pdf"
NAMESPACE = "agentic-ai-ebook"
EMBEDDING_MODEL = "text-embedding-3-small"
EMBEDDING_DIMENSION = 1536  # output size of text-embedding-3-small
# Optional override in .env in case gpt-4o-mini is ever retired
CHAT_MODEL = os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini").strip() or "gpt-4o-mini"

# ---- Chunking and retrieval ----
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
TOP_K = 3

# ---- Grounding and confidence (see README "Confidence score") ----
NOT_AVAILABLE_MESSAGE = "The information is not available in the provided document."
RELEVANCE_THRESHOLD = 0.25  # if the best cosine score is below this, skip the LLM and refuse
SCORE_FLOOR = 0.20          # raw relevance mapped to confidence 0.0
SCORE_CEILING = 0.70        # raw relevance mapped to confidence 1.0


class ConfigError(Exception):
    """Raised when required configuration is missing or invalid."""


@dataclass(frozen=True)
class Settings:
    openai_api_key: str
    pinecone_api_key: str
    pinecone_index_name: str
    pinecone_cloud: str
    pinecone_region: str


def get_settings() -> Settings:
    """Read environment variables and fail early with a clear message."""
    required = [
        "OPENAI_API_KEY",
        "PINECONE_API_KEY",
        "PINECONE_INDEX_NAME",
        "PINECONE_CLOUD",
        "PINECONE_REGION",
    ]
    missing = [name for name in required if not os.getenv(name, "").strip()]
    if missing:
        raise ConfigError(
            "Missing environment variables: "
            + ", ".join(missing)
            + ". Copy .env.example to .env and fill them in."
        )
    return Settings(
        openai_api_key=os.environ["OPENAI_API_KEY"].strip(),
        pinecone_api_key=os.environ["PINECONE_API_KEY"].strip(),
        pinecone_index_name=os.environ["PINECONE_INDEX_NAME"].strip(),
        pinecone_cloud=os.environ["PINECONE_CLOUD"].strip().lower(),
        pinecone_region=os.environ["PINECONE_REGION"].strip(),
    )
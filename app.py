"""FastAPI application exposing the RAG chatbot."""

import logging
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, Field, field_validator

from src.config import ConfigError
from src.errors import GenerationError, RetrievalError, describe_external_error
from src.graph import build_rag_graph

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ChatRequest(BaseModel):
    query: str = Field(..., max_length=1000, examples=["What is Agentic AI?"])

    @field_validator("query")
    @classmethod
    def query_must_not_be_blank(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("query must not be empty")
        return value


class RetrievedChunkModel(BaseModel):
    text: str
    metadata: dict[str, Any]
    score: float


class ChatResponse(BaseModel):
    final_answer: str
    retrieved_context: list[RetrievedChunkModel]
    confidence_score: float


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Build the graph once at startup. Keep the server alive even if setup fails."""
    app.state.graph = None
    app.state.startup_error = None
    try:
        app.state.graph = build_rag_graph()
        logger.info("RAG graph ready.")
    except (ConfigError, RetrievalError) as exc:
        app.state.startup_error = str(exc)
    except Exception as exc:
        app.state.startup_error = describe_external_error(exc)
    if app.state.startup_error:
        logger.error("Startup problem: %s", app.state.startup_error)
    yield


app = FastAPI(title="Agentic AI RAG API", lifespan=lifespan)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


# Plain "def" (not "async def"): the graph makes blocking network calls, so FastAPI
# runs this in a worker thread and other requests are not blocked.
@app.post("/chat", response_model=ChatResponse)
def chat(payload: ChatRequest, request: Request) -> ChatResponse:
    graph = request.app.state.graph
    if graph is None:
        raise HTTPException(
            status_code=503,
            detail=f"Service not ready: {request.app.state.startup_error}",
        )

    initial_state = {"question": payload.query, "context": [], "answer": "", "score": 0.0}
    try:
        result = graph.invoke(initial_state)
    except (RetrievalError, GenerationError) as exc:
        raise HTTPException(status_code=502, detail=str(exc))
    except Exception as exc:
        logger.error("Unexpected error: %s", type(exc).__name__)
        raise HTTPException(status_code=500, detail="Internal server error.")

    return ChatResponse(
        final_answer=result["answer"],
        retrieved_context=result["context"],
        confidence_score=result["score"],
    )
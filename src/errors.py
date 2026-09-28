"""Custom exceptions and a helper that turns library errors into safe messages."""

import openai


class RetrievalError(Exception):
    """Raised when querying Pinecone (or embedding the question) fails."""


class GenerationError(Exception):
    """Raised when the OpenAI chat call fails."""


def describe_external_error(exc: Exception) -> str:
    """Return a short, safe description. Never includes the raw exception text."""
    if isinstance(exc, openai.AuthenticationError):
        return "OpenAI rejected the API key. Check OPENAI_API_KEY."
    if isinstance(exc, openai.RateLimitError):
        return "OpenAI rate limit or quota exceeded. Check your billing/credits."
    if isinstance(exc, openai.NotFoundError):
        return "OpenAI model not found. Set OPENAI_CHAT_MODEL to an available model."
    if isinstance(exc, openai.APIConnectionError):
        return "Could not reach OpenAI. Check your internet connection."
    if isinstance(exc, openai.OpenAIError):
        return f"OpenAI returned an error ({type(exc).__name__})."

    module = type(exc).__module__.lower()
    if "pinecone" in module:
        return (
            f"Pinecone error ({type(exc).__name__}). Check PINECONE_API_KEY, "
            "the index name, cloud/region and your internet connection."
        )
    if "urllib3" in module or "requests" in module:
        return "Network error while contacting an external service."
    return f"Unexpected error ({type(exc).__name__})."
"""PDF -> chunks -> OpenAI embeddings -> Pinecone.

Run from the project root:
    python -m src.ingestion            # normal run
    python -m src.ingestion --reset    # clear the namespace first, then re-ingest
"""

import argparse
import logging
import sys
from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pinecone import Pinecone, ServerlessSpec

from src.config import (
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    EMBEDDING_DIMENSION,
    EMBEDDING_MODEL,
    NAMESPACE,
    PDF_PATH,
    ConfigError,
    Settings,
    get_settings,
)
from src.errors import describe_external_error

logger = logging.getLogger(__name__)


def load_pdf(pdf_path: Path) -> list[Document]:
    """Load the PDF and return one Document per page that contains text."""
    if not pdf_path.exists():
        raise FileNotFoundError(
            f"PDF not found at '{pdf_path}'. Download it and save it as data\\Ebook-Agentic-AI.pdf"
        )
    pages = PyPDFLoader(str(pdf_path)).load()
    pages = [p for p in pages if p.page_content.strip()]
    if not pages:
        raise ValueError(
            "No text could be extracted. The PDF may be scanned images (needs OCR)."
        )
    return pages


def split_into_chunks(pages: list[Document], source_name: str) -> list[Document]:
    """Split pages into overlapping chunks and attach clean metadata."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP
    )
    chunks = [c for c in splitter.split_documents(pages) if c.page_content.strip()]

    stem = Path(source_name).stem
    for i, chunk in enumerate(chunks):
        # PyPDFLoader page numbers start at 0, humans count from 1
        page_number = int(chunk.metadata.get("page", 0)) + 1
        chunk.metadata = {
            "source": source_name,
            "page_number": page_number,
            "chunk_id": f"{stem}-chunk-{i:04d}",
        }
    return chunks


def ensure_index(pc: Pinecone, settings: Settings) -> None:
    """Create the serverless index if missing; verify it matches if it exists."""
    name = settings.pinecone_index_name
    if name in pc.list_indexes().names():
        info = pc.describe_index(name)
        if info.dimension != EMBEDDING_DIMENSION or info.metric != "cosine":
            raise ConfigError(
                f"Index '{name}' exists but has dimension={info.dimension}, "
                f"metric={info.metric}. Expected dimension={EMBEDDING_DIMENSION}, cosine. "
                "Use a different PINECONE_INDEX_NAME or delete the old index in the console."
            )
        print(f"      Index '{name}' already exists - reusing it.")
        return

    print(f"      Creating index '{name}' (this can take up to a minute)...")
    try:
        pc.create_index(  # waits until the index is ready by default
            name=name,
            dimension=EMBEDDING_DIMENSION,
            metric="cosine",
            spec=ServerlessSpec(
                cloud=settings.pinecone_cloud, region=settings.pinecone_region
            ),
        )
    except Exception as exc:
        raise ConfigError(
            "Could not create the Pinecone index. Check PINECONE_CLOUD and "
            "PINECONE_REGION (free plans usually allow only aws / us-east-1). "
            f"({type(exc).__name__})"
        ) from None


def run_ingestion(reset: bool = False) -> None:
    """Full ingestion pipeline."""
    settings = get_settings()

    print(f"[1/5] Loading PDF: {PDF_PATH}")
    pages = load_pdf(PDF_PATH)
    print(f"      Pages with text: {len(pages)}")

    print("[2/5] Splitting into chunks...")
    chunks = split_into_chunks(pages, PDF_PATH.name)
    print(f"      Chunks: {len(chunks)} (size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP})")

    print("[3/5] Connecting to Pinecone...")
    pc = Pinecone(api_key=settings.pinecone_api_key)
    ensure_index(pc, settings)
    index = pc.Index(settings.pinecone_index_name)

    if reset:
        try:
            index.delete(delete_all=True, namespace=NAMESPACE)
            print(f"      Cleared namespace '{NAMESPACE}'.")
        except Exception:
            print("      Nothing to clear (namespace does not exist yet).")

    print("[4/5] Embedding chunks and uploading to Pinecone...")
    embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL, api_key=settings.openai_api_key)
    vector_store = PineconeVectorStore(index=index, embedding=embeddings, namespace=NAMESPACE)
    # Deterministic IDs: re-running overwrites the same records instead of duplicating.
    ids = [chunk.metadata["chunk_id"] for chunk in chunks]
    vector_store.add_documents(documents=chunks, ids=ids)

    print("[5/5] Checking index stats...")
    try:
        stats = index.describe_index_stats()
        summary = stats.namespaces.get(NAMESPACE)
        count = summary.vector_count if summary else "pending"
        print(f"      Vectors in namespace '{NAMESPACE}': {count}")
        print("      (Stats can lag a few seconds behind the upload.)")
    except Exception:
        print("      Could not read stats (upload itself succeeded).")
    print("Ingestion complete.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest the eBook into Pinecone.")
    parser.add_argument("--reset", action="store_true", help="Clear the namespace first.")
    args = parser.parse_args()

    try:
        run_ingestion(reset=args.reset)
    except (ConfigError, FileNotFoundError, ValueError) as exc:
        print(f"\nERROR: {exc}")
        sys.exit(1)
    except Exception as exc:
        print(f"\nERROR: {describe_external_error(exc)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
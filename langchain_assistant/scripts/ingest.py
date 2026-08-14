"""Ingest the note corpus into Azure AI Search (Phase 2 pipeline).

Steps:
  1. (optional) upload local seed_data/*.md into Azure Blob Storage.
  2. load the notes from Blob into LangChain Document objects.
  3. split each document into overlapping chunks + attach metadata.
  4. embed every chunk and upload it (with metadata) into Azure AI Search.

Usage:
    # upload local notes to blob, then build the index:
    uv run python -m langchain_assistant.scripts.ingest --upload

    # blob already populated -> just (re)build the index:
    uv run python -m langchain_assistant.scripts.ingest

Notes are loaded from Blob via azure-storage-blob directly (see storage.py).
This is the lightweight equivalent of LangChain's AzureBlobStorageLoader,
without pulling in the heavy `unstructured` parsing stack for plain markdown.
"""

import argparse
import hashlib
import os
import re

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from ..app.storage import (
    build_embeddings,
    ensure_index,
    get_search_client,
    iter_blob_notes,
    rag_settings,
    upload_seed_data,
)

# Where the local corpus lives before it is pushed to Blob Storage.
_SEED_DIR = os.path.join(os.path.dirname(__file__), "..", "seed_data")

# A single version stamp for this ingest run (indexed for future re-ingest).
_DOC_VERSION = "v1"


def _title_from(name: str, text: str) -> str:
    """Prefer the first markdown H1 as the title; fall back to the filename."""
    match = re.search(r"^#\s+(.+)$", text, flags=re.MULTILINE)
    if match:
        return match.group(1).strip()
    return os.path.splitext(os.path.basename(name))[0]


def load_documents() -> list[Document]:
    """Load every markdown blob into a LangChain Document with metadata."""
    docs: list[Document] = []
    for name, text in iter_blob_notes():
        docs.append(
            Document(
                page_content=text,
                metadata={"source": name, "title": _title_from(name, text)},
            )
        )
    return docs


def split_documents(docs: list[Document]) -> list[Document]:
    """Split documents into chunks and stamp each with citation metadata."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=120,
        # Split on markdown-ish boundaries first, then finer separators.
        separators=["\n## ", "\n### ", "\n\n", "\n", " ", ""],
    )
    chunks = splitter.split_documents(docs)

    # Give each chunk a stable, human-readable chunk_id: "<source>#<n>".
    per_source_counter: dict[str, int] = {}
    for chunk in chunks:
        source = chunk.metadata.get("source", "unknown")
        n = per_source_counter.get(source, 0)
        per_source_counter[source] = n + 1
        chunk.metadata["chunk_id"] = f"{source}#{n}"
        chunk.metadata["doc_version"] = _DOC_VERSION
    return chunks


def _search_doc_id(chunk_id: str) -> str:
    """Azure Search keys allow letters/digits/_/-/= only; hash anything else."""
    return hashlib.sha1(chunk_id.encode("utf-8")).hexdigest()


def index_chunks(chunks: list[Document]) -> int:
    """Embed chunks and upload them into Azure AI Search. Returns the count."""
    embeddings = build_embeddings()
    texts = [c.page_content for c in chunks]

    # Batch-embed all chunk texts at once.
    vectors = embeddings.embed_documents(texts)

    # Create/refresh the index using the real embedding dimension.
    ensure_index(embedding_dim=len(vectors[0]))

    payload = []
    for chunk, vector in zip(chunks, vectors):
        payload.append(
            {
                "id": _search_doc_id(chunk.metadata["chunk_id"]),
                "content": chunk.page_content,
                "embedding": vector,
                "source": chunk.metadata.get("source", ""),
                "title": chunk.metadata.get("title", ""),
                "chunk_id": chunk.metadata.get("chunk_id", ""),
                "doc_version": chunk.metadata.get("doc_version", _DOC_VERSION),
            }
        )

    client = get_search_client()
    # pylint: disable=no-member  # method added dynamically by the azure-search SDK patch layer.
    client.upload_documents(documents=payload)
    return len(payload)


def main() -> None:
    """Run the ingest pipeline end to end."""
    parser = argparse.ArgumentParser(description="Ingest notes into Azure AI Search.")
    parser.add_argument(
        "--upload",
        action="store_true",
        help="First upload local seed_data/*.md into Blob Storage.",
    )
    args = parser.parse_args()

    if args.upload:
        n = upload_seed_data(os.path.abspath(_SEED_DIR))
        print(f"Uploaded {n} file(s) to blob container '{rag_settings.blob_container}'.")

    docs = load_documents()
    print(f"Loaded {len(docs)} document(s) from Blob Storage.")
    if not docs:
        print("No documents found. Did you upload the corpus? (try --upload)")
        return

    chunks = split_documents(docs)
    print(f"Split into {len(chunks)} chunk(s).")

    count = index_chunks(chunks)
    print(f"Indexed {count} chunk(s) into '{rag_settings.search_index}'.")


if __name__ == "__main__":
    main()

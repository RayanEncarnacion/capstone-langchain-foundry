"""Retrieval: the first half of explicit two-step RAG.

Step 1 (here): given a question, embed it and pull the most relevant chunks
from Azure AI Search. We run a *hybrid* query (keyword text + vector) so both
exact wording and semantic matches contribute.

Step 2 (agent.py): feed those chunks to the model with strict citation and
abstention rules.

Every retrieved chunk carries the metadata we indexed (source, title,
chunk_id) so the answer can cite exactly what it used.
"""

from dataclasses import dataclass

from azure.search.documents.models import VectorizedQuery

from .storage import build_embeddings, get_search_client


@dataclass
class RetrievedChunk:
    """A single chunk returned from the index, with citation metadata."""

    id: str
    content: str
    source: str
    title: str
    chunk_id: str
    score: float


def retrieve(question: str, top_k: int = 4) -> list[RetrievedChunk]:
    """Embed the question and return the top_k chunks via a hybrid search."""
    embeddings = build_embeddings()
    query_vector = embeddings.embed_query(question)

    # Vector leg of the hybrid query.
    vector_query = VectorizedQuery(
        vector=query_vector,
        k_nearest_neighbors=top_k,
        fields="embedding",
    )

    client = get_search_client()
    # Passing both `search_text` and `vector_queries` = hybrid retrieval.
    # pylint: disable=no-member  # method added dynamically by the azure-search SDK patch layer.
    results = client.search(
        search_text=question,
        vector_queries=[vector_query],
        select=["id", "content", "source", "title", "chunk_id"],
        top=top_k,
    )

    chunks: list[RetrievedChunk] = []
    for r in results:
        chunks.append(
            RetrievedChunk(
                id=r["id"],
                content=r["content"],
                source=r.get("source", ""),
                title=r.get("title", ""),
                chunk_id=r.get("chunk_id", ""),
                score=r.get("@search.score", 0.0),
            )
        )
    return chunks


def format_context(chunks: list[RetrievedChunk]) -> str:
    """Render chunks into a numbered, citable context block for the prompt.

    Each block is labelled with source + chunk_id so the model can cite them
    and so we can print the exact text that was handed to the model.
    """
    blocks = []
    for i, c in enumerate(chunks, start=1):
        header = f"[{i}] source={c.source} chunk_id={c.chunk_id} title={c.title}"
        blocks.append(f"{header}\n{c.content}")
    return "\n\n".join(blocks)

"""Compare retrieval strategies side by side (retrieval baseline experiment).

Runs the SAME five questions through four retrieval modes and prints the top
three chunks + scores for each, so the choice of a retrieval baseline is made
from evidence rather than intuition.

Modes:
  1. keyword        -> BM25 text match only (search_text, no vector).
  2. vector         -> pure embedding nearest-neighbour (vector only, no text).
  3. hybrid         -> keyword + vector fused with Reciprocal Rank Fusion (RRF).
  4. hybrid+semantic-> hybrid retrieval, then the semantic ranker re-reads the
                       top results and re-scores them (@search.reranker_score).

Usage:
    uv run python -m langchain_assistant.scripts.compare_retrieval

    # tweak how many results per mode (default 3):
    uv run python -m langchain_assistant.scripts.compare_retrieval --top-k 3

What to look for while reading the output:
  - Where keyword wins: the question reuses exact wording from the source.
  - Where vector wins: the question paraphrases / uses synonyms.
  - Where semantic re-ranking reorders hybrid results by actual answer-ness.
"""

import argparse
import textwrap

from azure.search.documents.models import VectorizedQuery

from ..app.storage import build_embeddings, get_search_client, rag_settings

# Semantic configuration created in the Azure Portal on the notes index.
# (Field mapping: title -> `title`, content -> `content`.)
SEMANTIC_CONFIG = "notes-semantic-config"

# Five questions with deliberately varied wording relative to the source text.
# Each targets a different retrieval failure/strength mode.
QUESTIONS: list[str] = [
    # 1. Near-exact phrasing from the source -> keyword should do well.
    "What is the passing score for a module checkpoint?",
    # 2. Paraphrase: source says "active tasks", question says "open".
    "How many tasks am I allowed to have open at the same time?",
    # 3. Historical/temporal wording -> needs the archived 2024 handbook.
    "What was the old study session length before they changed it?",
    # 4. Conversational/vague: source says events are "optional".
    "Do I need to attend the Wednesday live event?",
    # 5. Cross-document reasoning: community tips vs official policy.
    "What should I do if a community tip contradicts official policy?",
]

# Fields we pull back for display / citation.
_SELECT = ["content", "source", "title", "chunk_id"]


def _fmt_chunk(rank: int, result: dict) -> str:
    """One line of chunk metadata + a short snippet of its content."""
    score = result.get("@search.score", 0.0)
    reranker = result.get("@search.reranker_score")
    snippet = " ".join(result.get("content", "").split())
    snippet = textwrap.shorten(snippet, width=110, placeholder=" ...")

    score_str = f"score={score:.4f}"
    if reranker is not None:
        score_str += f" reranker={reranker:.4f}"

    return (
        f"    #{rank} [{score_str}] "
        f"{result.get('source', '?')} ({result.get('chunk_id', '?')})\n"
        f"        {snippet}"
    )


def _print_results(mode: str, results) -> None:
    """Print the ranked results for one mode under a labelled header."""
    print(f"  --- {mode} ---")
    rows = list(results)
    if not rows:
        print("    (no results)")
        return
    for rank, r in enumerate(rows, start=1):
        print(_fmt_chunk(rank, r))


def run_keyword(client, question: str, top_k: int):
    """Keyword-only: BM25 text scoring, no vector leg."""
    # pylint: disable=no-member
    return client.search(
        search_text=question,
        select=_SELECT,
        top=top_k,
    )


def run_vector(client, question: str, query_vector, top_k: int):
    """Vector-only: nearest-neighbour on the embedding field, no text scoring."""
    vector_query = VectorizedQuery(
        vector=query_vector,
        k_nearest_neighbors=top_k,
        fields="embedding",
    )
    # search_text=None => no keyword leg, so this is pure vector search.
    # pylint: disable=no-member
    return client.search(
        search_text=None,
        vector_queries=[vector_query],
        select=_SELECT,
        top=top_k,
    )


def run_hybrid(client, question: str, query_vector, top_k: int):
    """Hybrid: keyword + vector, fused by Reciprocal Rank Fusion (RRF)."""
    vector_query = VectorizedQuery(
        vector=query_vector,
        k_nearest_neighbors=top_k,
        fields="embedding",
    )
    # Both search_text and vector_queries present => hybrid + RRF.
    # pylint: disable=no-member
    return client.search(
        search_text=question,
        vector_queries=[vector_query],
        select=_SELECT,
        top=top_k,
    )


def run_hybrid_semantic(client, question: str, query_vector, top_k: int):
    """Hybrid retrieval + semantic re-ranking on top of the fused results."""
    vector_query = VectorizedQuery(
        vector=query_vector,
        k_nearest_neighbors=top_k,
        fields="embedding",
    )
    # query_type=semantic activates the reranker; it needs the semantic config.
    # pylint: disable=no-member
    return client.search(
        search_text=question,
        vector_queries=[vector_query],
        query_type="semantic",
        semantic_configuration_name=SEMANTIC_CONFIG,
        select=_SELECT,
        top=top_k,
    )


def main() -> None:
    """Run all four modes for all five questions and print the comparison."""
    parser = argparse.ArgumentParser(description="Compare retrieval strategies.")
    parser.add_argument(
        "--top-k",
        type=int,
        default=3,
        help="How many chunks to show per mode (default: 3).",
    )
    args = parser.parse_args()
    top_k = args.top_k

    rag_settings.require_search()
    embeddings = build_embeddings()
    client = get_search_client()

    print(f"Index: {rag_settings.search_index}")
    print(f"Semantic config: {SEMANTIC_CONFIG}")
    print(f"Showing top {top_k} chunk(s) per mode.\n")

    for i, question in enumerate(QUESTIONS, start=1):
        # Embed the question once; reuse the vector across the vector-based modes.
        query_vector = embeddings.embed_query(question)

        print("=" * 100)
        print(f"Q{i}: {question}")
        print("=" * 100)

        _print_results("keyword-only", run_keyword(client, question, top_k))
        _print_results("vector-only", run_vector(client, question, query_vector, top_k))
        _print_results("hybrid (RRF)", run_hybrid(client, question, query_vector, top_k))
        try:
            _print_results(
                "hybrid + semantic",
                run_hybrid_semantic(client, question, query_vector, top_k),
            )
        except Exception as exc:  # noqa: BLE001 - surface config/tier issues, keep going.
            print("  --- hybrid + semantic ---")
            print(f"    (semantic query failed: {exc})")
        print()


if __name__ == "__main__":
    main()

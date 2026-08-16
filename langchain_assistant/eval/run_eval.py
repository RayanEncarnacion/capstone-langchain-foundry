"""Entry point: sync the dataset and run a LangSmith experiment.

Usage:
    # Deterministic checks only (default, no judge LLM cost):
    uv run python -m langchain_assistant.eval.run_eval

    # Add the groundedness + relevance LLM judges:
    uv run python -m langchain_assistant.eval.run_eval --llm

    # Only sync the dataset (no experiment):
    uv run python -m langchain_assistant.eval.run_eval --sync-only

Requires LANGCHAIN_API_KEY (+ optionally LANGCHAIN_PROJECT) in the environment.
Traces and the experiment appear under your LangSmith project.
"""

import argparse

from dotenv import load_dotenv
from langsmith import Client

from .dataset import DATASET_NAME, sync_dataset
from .evaluators import DETERMINISTIC_EVALUATORS, LLM_EVALUATORS
from .target import run_target


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 7 LangSmith evaluation.")
    parser.add_argument(
        "--llm",
        action="store_true",
        help="Include groundedness + relevance LLM judges.",
    )
    parser.add_argument(
        "--sync-only",
        action="store_true",
        help="Create/update the dataset and exit (no experiment).",
    )
    parser.add_argument(
        "--prefix",
        default="phase7",
        help="Experiment name prefix shown in LangSmith.",
    )
    args = parser.parse_args()

    load_dotenv()
    client = Client()

    dataset_id = sync_dataset(client)
    print(f"Dataset '{DATASET_NAME}' ready ({dataset_id}).")
    if args.sync_only:
        return

    evaluators = list(DETERMINISTIC_EVALUATORS)
    if args.llm:
        evaluators += LLM_EVALUATORS

    results = client.evaluate(
        run_target,
        data=DATASET_NAME,
        evaluators=evaluators,
        experiment_prefix=args.prefix,
        max_concurrency=1,  # agent hits Cosmos/Search; keep it gentle + ordered.
    )

    # Print the experiment URL if the SDK exposes it.
    url = getattr(results, "experiment_url", None) or getattr(
        results, "_experiment_url", None
    )
    if url:
        print(f"Experiment: {url}")
    print("Done. Open LangSmith to inspect traces and scores.")


if __name__ == "__main__":
    main()

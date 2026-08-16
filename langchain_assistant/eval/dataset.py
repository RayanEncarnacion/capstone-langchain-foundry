"""The shared evaluation dataset (single-turn cases) + LangSmith sync.

This mirrors the single-turn table in DATASET.md as machine-readable rows so
the eval harness and the human-readable doc stay in step. Multi-turn (MEM-*)
and multi-user (SEC-02) cases are intentionally excluded here: they need a
sequential/stateful runner, not a single-shot evaluate() pass.

Examples live in examples.json; loaded once at module import time.
"""

import json
import os
from pathlib import Path

from langsmith import Client

# Name of the LangSmith dataset we create/update. Override via env if desired.
DATASET_NAME = os.environ.get("LANGSMITH_DATASET", "capstone-phase7")


def _load_examples() -> list[dict]:
    """Load examples from examples.json in the same directory."""
    path = Path(__file__).parent / "examples.json"
    with open(path) as f:
        return json.load(f)


# Single-turn, batch-evaluable examples (see DATASET.md, examples.json).
EXAMPLES = _load_examples()

def sync_dataset(client: Client | None = None) -> str:
    """Create or update the LangSmith dataset from EXAMPLES.

    Idempotent: we key examples by their DATASET.md id in metadata, wipe any
    existing rows, and re-upload. Returns the dataset id.
    """
    client = client or Client()

    if client.has_dataset(dataset_name=DATASET_NAME):
        dataset = client.read_dataset(dataset_name=DATASET_NAME)
        # Clear old rows so re-runs never accumulate duplicates.
        for existing in client.list_examples(dataset_id=dataset.id):
            client.delete_example(example_id=existing.id)
    else:
        dataset = client.create_dataset(
            dataset_name=DATASET_NAME,
            description="Capstone Phase 7 single-turn evaluation cases.",
        )

    client.create_examples(
        dataset_id=dataset.id,
        inputs=[{"question": ex["question"]} for ex in EXAMPLES],
        outputs=[
            {
                "expected_tool": ex["expected_tool"],
                "expected_abstain": ex["expected_abstain"],
            }
            for ex in EXAMPLES
        ],
        metadata=[{"id": ex["id"], "category": ex["category"]} for ex in EXAMPLES],
    )
    return str(dataset.id)


if __name__ == "__main__":
    from dotenv import load_dotenv

    load_dotenv()
    print(f"Synced dataset '{DATASET_NAME}' -> {sync_dataset()}")

"""Smoke-test client for the local Foundry assistant API.

Proves Phase 3 grounding:
    turn 1 — a grounded question should call `knowledge_base_retrieve` and
             return a cited answer.
    turn 2 — an out-of-scope question should produce an abstention with no
             fabricated content.
"""

import httpx

BASE = "http://127.0.0.1:8000"


def main() -> None:
    r = httpx.get(f"{BASE}/health")
    print("health:", r.json())

    # Grounded question — expect a knowledge_base_retrieve tool call + citations.
    r = httpx.post(
        f"{BASE}/chat",
        json={"message": "What is the Northstar study session policy?"},
        timeout=90,
    )
    print("grounded status:", r.status_code)
    grounded = r.json()
    print("grounded response:", grounded)
    print("tool calls:", grounded.get("tool_calls"))

    # Out-of-scope question — expect an abstention, no tool-backed answer.
    r = httpx.post(
        f"{BASE}/chat",
        json={"message": "What was the score of last night's basketball game?"},
        timeout=90,
    )
    print("abstain status:", r.status_code)
    print("abstain response:", r.json())


if __name__ == "__main__":
    main()

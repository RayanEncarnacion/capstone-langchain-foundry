"""Smoke-test client for the local Foundry assistant API.

Proves Phase 2 session reuse: the second turn passes the session_id from the
first, so the agent keeps context. Drop the thread_id and the follow-up would
lose that context.
"""

import httpx

BASE = "http://127.0.0.1:8000"


def main() -> None:
    r = httpx.get(f"{BASE}/health")
    print("health:", r.json())

    r = httpx.post(
        f"{BASE}/chat",
        json={"message": "My name is Alice and I love photosynthesis."},
        timeout=60,
    )
    print("turn 1 status:", r.status_code)
    first = r.json()
    print("turn 1 response:", first)

    session_id = first["session_id"]

    r = httpx.post(
        f"{BASE}/chat",
        json={"message": "What is my name?", "thread_id": session_id},
        timeout=60,
    )
    print("turn 2 status:", r.status_code)
    print("turn 2 response:", r.json())


if __name__ == "__main__":
    main()

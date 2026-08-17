"""Smoke-test client for the local Foundry assistant API."""

import httpx

BASE = "http://127.0.0.1:8000"


def main() -> None:
    r = httpx.get(f"{BASE}/health")
    print("health:", r.json())

    r = httpx.post(
        f"{BASE}/chat",
        json={"message": "What is photosynthesis?", "user_id": "u1"},
        timeout=60,
    )
    print("status:", r.status_code)
    print("response:", r.json())


if __name__ == "__main__":
    main()

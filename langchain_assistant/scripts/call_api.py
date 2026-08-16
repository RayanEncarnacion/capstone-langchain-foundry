"""Tiny client to exercise POST /chat from the terminal.

Auth (Phase 6): the API requires a bearer token. Provide one via the
API_TOKEN environment variable (a key from your .env API_KEYS map). Omit it
to see the anonymous 401.

Usage (server must be running):
    API_TOKEN=sk-rayan-abc123 uv run python -m langchain_assistant.scripts.call_api "Hello"
"""

import os
import sys

import httpx

# Where the local FastAPI server listens.
BASE_URL = "http://127.0.0.1:8000"


def main() -> None:
    """Call the API and print the response."""

    # Take the message from the command line, or use a default.
    message = sys.argv[1] if len(sys.argv) > 1 else "Say hello in one short sentence."

    # Attach the bearer token if one is configured (else the call returns 401).
    token = os.environ.get("API_TOKEN", "")
    headers = {"Authorization": f"Bearer {token}"} if token else {}

    response = httpx.post(
        f"{BASE_URL}/chat", json={"message": message}, headers=headers, timeout=60
    )
    print(f"HTTP {response.status_code}")
    print(response.json())


if __name__ == "__main__":
    main()

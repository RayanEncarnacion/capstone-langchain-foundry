"""Tiny client to exercise POST /chat from the terminal.

Usage (server must be running):
    uv run python -m langchain_assistant.scripts.call_api "Hello there"
"""

import sys

import httpx

# Where the local FastAPI server listens.
BASE_URL = "http://127.0.0.1:8000"


def main() -> None:
    """Call the API and print the response."""
    
    # Take the message from the command line, or use a default.
    message = sys.argv[1] if len(sys.argv) > 1 else "Say hello in one short sentence."

    # POST the message and print the typed JSON reply.
    response = httpx.post(f"{BASE_URL}/chat", json={"message": message}, timeout=60)
    response.raise_for_status()
    print(response.json())


if __name__ == "__main__":
    main()

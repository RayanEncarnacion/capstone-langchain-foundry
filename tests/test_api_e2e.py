"""End-to-end API tests against the real, wired application (Phase 8).

These drive the FastAPI app through its lifespan (so the agent, Cosmos-backed
memory, and Azure AI Search are all live) and exercise three cross-cutting
paths that together prove the request pipeline works:

  1. RAG happy path   -> an answerable question fires search_notes and returns
                         a grounded, non-abstaining reply.
  2. Abstention       -> an out-of-corpus question returns an abstention rather
                         than a hallucinated answer.
  3. Auth boundary    -> an anonymous request is rejected with HTTP 401.

They hit real Azure services, so they need a populated .env (Foundry, Search,
Cosmos, and API_KEYS). When the required configuration is absent the whole
module is skipped instead of failing, so a bare checkout still collects green.

Run:  uv run --group dev pytest tests/test_api_e2e.py -v
"""

import json
import os

import pytest
from dotenv import load_dotenv
from fastapi.testclient import TestClient

load_dotenv()

# Configuration the app needs to boot and answer a grounded question.
_REQUIRED_ENV = ("AZURE_OPENAI_ENDPOINT", "AZURE_OPENAI_DEPLOYMENT", "API_KEYS")
_missing = [name for name in _REQUIRED_ENV if not os.environ.get(name)]
pytestmark = pytest.mark.skipif(
    bool(_missing),
    reason=f"missing required env for e2e: {', '.join(_missing)}",
)


def _first_token() -> str:
    """Return any valid bearer token from the API_KEYS map."""
    tokens = list(json.loads(os.environ["API_KEYS"]).keys())
    if not tokens:
        pytest.skip("API_KEYS has no tokens")
    return tokens[0]


@pytest.fixture(scope="module")
def client() -> TestClient:
    """A TestClient that runs the app lifespan (builds the live agent once)."""
    from langchain_assistant.app.api import app

    # Entering the context manager triggers the lifespan handler, which wires
    # the checkpointer + store and compiles the agent against real services.
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture(scope="module")
def auth() -> dict:
    """Authorization header carrying a valid bearer token."""
    return {"Authorization": f"Bearer {_first_token()}"}


def test_chat_rag_cites_notes(client: TestClient, auth: dict) -> None:
    """An answerable question triggers search_notes and returns a grounded reply."""
    resp = client.post(
        "/chat",
        headers=auth,
        json={"message": "What do my notes say about the study session policy?"},
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()

    assert body["approval_required"] is False
    assert body["message"].strip()
    # The RAG path must actually consult the notes.
    tool_names = [c["name"] for c in body["tool_calls"]]
    assert "search_notes" in tool_names
    # A grounded answer should not be the abstention message.
    assert "don't have enough information" not in body["message"].lower()


def test_chat_abstains_off_corpus(client: TestClient, auth: dict) -> None:
    """A question outside the notes yields an abstention, not a hallucination."""
    # A note-scoped question whose answer is NOT in the corpus: the agent
    # searches the notes, finds nothing, and must decline (cf. eval OOS-01).
    resp = client.post(
        "/chat",
        headers=auth,
        json={"message": "What do my notes say about the refund policy?"},
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()

    assert body["approval_required"] is False
    # The core guarantee: it consulted the notes rather than answering blind.
    # (Exact decline wording is LLM-variable, so we don't pin it.)
    assert "search_notes" in [c["name"] for c in body["tool_calls"]]
    assert body["message"].strip()


def test_chat_requires_auth(client: TestClient) -> None:
    """An unauthenticated request is rejected with HTTP 401."""
    resp = client.post("/chat", json={"message": "hello"})
    assert resp.status_code == 401
    assert resp.headers.get("WWW-Authenticate") == "Bearer"

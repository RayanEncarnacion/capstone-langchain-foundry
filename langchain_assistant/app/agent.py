"""The 'agent' for this phase: a direct chat model call to Microsoft Foundry.

No RAG, no tools, no memory yet (those live in retrieval.py / tools.py /
storage.py in later phases). Here we only:
  1. read connection settings from the environment (nothing hard-coded), and
  2. build a LangChain chat client that points at the Foundry model.

We target Foundry's OpenAI-compatible **v1 endpoint**, which the Foundry UI
shows as e.g. https://<resource>.services.ai.azure.com/openai/v1 . Because it
is OpenAI-compatible, we use `ChatOpenAI` with a `base_url` (no api-version).

Auth has two modes:
  - API key  -> set AZURE_OPENAI_API_KEY.
  - Azure AD -> no key; DefaultAzureCredential (keyless, recommended).
Both are sent as `Authorization: Bearer <secret>`, which the v1 endpoint accepts.
"""

import os

from azure.identity import DefaultAzureCredential
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

# Load a local, git-ignored .env so `uv run` picks up secrets in dev.
load_dotenv()

# Scope required to request AAD tokens for Azure AI services.
_AAD_SCOPE = "https://cognitiveservices.azure.com/.default"


class Settings:
    """Foundry connection settings, read from the environment."""

    def __init__(self) -> None:
        # v1 endpoint, e.g. https://<resource>.services.ai.azure.com/openai/v1
        self.endpoint = os.environ.get("AZURE_OPENAI_ENDPOINT")
        self.deployment = os.environ.get("AZURE_OPENAI_DEPLOYMENT")
        self.api_key = os.environ.get("AZURE_OPENAI_API_KEY")

    def require(self) -> None:
        """Fail fast if the required variables are not set."""
        missing = [
            name
            for name, value in {
                "AZURE_OPENAI_ENDPOINT": self.endpoint,
                "AZURE_OPENAI_DEPLOYMENT": self.deployment,
            }.items()
            if not value
        ]
        if missing:
            raise RuntimeError(f"Missing required environment variables: {', '.join(missing)}")


settings = Settings()


def _bearer_secret() -> str:
    """Return the secret to send as `Authorization: Bearer ...`.

    Prefer the API key; otherwise fetch an Azure AD token (keyless).
    """
    if settings.api_key:
        return settings.api_key
    # DefaultAzureCredential uses `az login`, managed identity, env vars, etc.
    token = DefaultAzureCredential().get_token(_AAD_SCOPE)
    return token.token


def build_chat_model() -> ChatOpenAI:
    """Return a configured chat model. No secrets are hard-coded."""
    return ChatOpenAI(
        base_url=settings.endpoint,   # Foundry v1, OpenAI-compatible route.
        model=settings.deployment,    # deployment name doubles as model name.
        api_key=_bearer_secret(),     # sent as Authorization: Bearer <secret>.
        temperature=0,                # deterministic-ish, easier to reason about.
    )


def ask(model: ChatOpenAI, message: str) -> str:
    """Send one user message, return the model's text reply."""
    # Direct call: one human message in, one AI message out.
    result = model.invoke([HumanMessage(content=message)])
    return result.content


# Sentinel the model must emit when the context does not support an answer.
ABSTAIN_TOKEN = "INSUFFICIENT_EVIDENCE"

# The grounding contract: answer only from context, cite, or abstain.
_RAG_SYSTEM_PROMPT = (
    "You are a study assistant that answers ONLY from the provided context.\n"
    "Rules:\n"
    "1. Use only facts found in the context blocks below. Do not use outside knowledge.\n"
    "2. Cite the chunks you used inline with their bracket numbers, e.g. [1], [2].\n"
    f"3. If the context does not contain enough information, reply with exactly "
    f"'{ABSTAIN_TOKEN}' and nothing else.\n"
    "Be concise."
)


def answer_with_rag(model: ChatOpenAI, question: str, top_k: int = 4):
    """Explicit two-step RAG: retrieve, then generate a grounded answer.

    Returns (answer_text, abstained_bool, retrieved_chunks). The chunks are
    returned so the caller can print/return the exact evidence used.
    """
    # Imported lazily to avoid a circular import (retrieval -> storage -> agent).
    from .retrieval import format_context, retrieve

    chunks = retrieve(question, top_k=top_k)

    # No hits at all -> abstain without even calling the model.
    if not chunks:
        return (
            "I don't have enough information in the notes to answer that.",
            True,
            chunks,
        )

    context = format_context(chunks)
    user_content = f"Context:\n{context}\n\nQuestion: {question}"

    result = model.invoke(
        [
            SystemMessage(content=_RAG_SYSTEM_PROMPT),
            HumanMessage(content=user_content),
        ]
    )
    answer = result.content.strip()

    # Detect abstention and normalise it into a friendly message.
    if ABSTAIN_TOKEN in answer:
        return (
            "I don't have enough information in the notes to answer that.",
            True,
            chunks,
        )

    return answer, False, chunks

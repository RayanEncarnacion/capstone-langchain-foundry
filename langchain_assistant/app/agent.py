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
from langchain_core.messages import HumanMessage
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

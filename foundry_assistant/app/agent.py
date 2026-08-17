"""Foundry agent: FoundryChatClient + Microsoft Agent Framework.

Phase 1 — baseline. Calls the Foundry project endpoint via the Responses
API. No tools yet; just prove the model responds through a local FastAPI
endpoint.

Environment variables (from .env):
    FOUNDRY_PROJECT_ENDPOINT  — project-scoped endpoint URL
    FOUNDRY_MODEL             — deployment name (e.g. gpt-4o-mini)
"""

import os

from azure.identity import DefaultAzureCredential
from dotenv import load_dotenv

load_dotenv()


class Settings:
    """Foundry connection settings read from environment."""

    def __init__(self) -> None:
        self.project_endpoint = os.environ.get("FOUNDRY_PROJECT_ENDPOINT")
        self.model = os.environ.get("FOUNDRY_MODEL")

    def require(self) -> None:
        missing = [
            name
            for name, value in {
                "FOUNDRY_PROJECT_ENDPOINT": self.project_endpoint,
                "FOUNDRY_MODEL": self.model,
            }.items()
            if not value
        ]
        if missing:
            raise RuntimeError(
                f"Missing required environment variables: {', '.join(missing)}"
            )


settings = Settings()

_SYSTEM_PROMPT = (
    "You are a study assistant. Answer questions clearly and concisely. "
    "When you don't know the answer, say so."
)


def build_agent():
    """Construct the Agent Framework agent backed by FoundryChatClient.

    Returns an Agent instance (async — call with `await agent.run(...)`).
    """
    from agent_framework import Agent
    from agent_framework.foundry import FoundryChatClient

    client = FoundryChatClient(
        project_endpoint=settings.project_endpoint,
        model=settings.model,
        credential=DefaultAzureCredential(),
    )

    return Agent(
        client=client,
        name="study-assistant",
        instructions=_SYSTEM_PROMPT,
    )


async def run_agent(agent, message: str) -> tuple[str, list[dict]]:
    """Run one user turn. Returns (reply_text, tool_calls)."""
    result = await agent.run(message)

    tool_calls: list[dict] = []
    for item in getattr(result, "tool_calls", None) or []:
        tool_calls.append({
            "name": getattr(item, "name", str(item)),
            "args": getattr(item, "args", {}),
        })

    text = str(result) if result else "(no content)"
    return text, tool_calls

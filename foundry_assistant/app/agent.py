"""Foundry agent: FoundryChatClient + Microsoft Agent Framework.

Phase 2 — agent, session and response schema. Builds an ephemeral Agent
Framework Agent (no hosted/persisted agent) and reuses an AgentSession per
thread so follow-up turns keep conversation context. Sessions live in an
in-memory dict keyed by session_id; they are lost on restart by design.

The agent definition (instructions + construction) is versioned with this
source file.

Environment variables (from .env):
    FOUNDRY_PROJECT_ENDPOINT  — project-scoped endpoint URL
    FOUNDRY_MODEL             — deployment name (e.g. gpt-4o-mini)
"""

import os
import uuid

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
    """Construct the ephemeral Agent Framework agent backed by FoundryChatClient.

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


# In-memory session store: session_id -> AgentSession. Ephemeral by design;
# reused across turns so a follow-up on the same thread keeps context, and
# cleared on process restart.
_sessions: dict = {}


def get_or_create_session(agent, thread_id: str | None):
    """Resolve the AgentSession for a thread, creating one if needed.

    Returns (session, session_id). A follow-up only carries context when the
    same session_id is supplied and its AgentSession is still cached.
    """
    if thread_id and thread_id in _sessions:
        return _sessions[thread_id], thread_id

    session = agent.create_session()
    session_id = getattr(session, "session_id", None) or str(uuid.uuid4())
    _sessions[session_id] = session
    return session, session_id


async def run_agent(
    agent, message: str, thread_id: str | None = None
) -> tuple[str, str, list[dict]]:
    """Run one user turn on a reused session.

    Returns (session_id, reply_text, tool_calls). Pass the returned session_id
    back as thread_id on the next call to continue the same conversation.
    """
    session, session_id = get_or_create_session(agent, thread_id)
    result = await agent.run(message, session=session)

    tool_calls: list[dict] = []
    for item in getattr(result, "tool_calls", None) or []:
        tool_calls.append({
            "name": getattr(item, "name", str(item)),
            "args": getattr(item, "args", {}),
        })

    text = getattr(result, "text", None) or (str(result) if result else "(no content)")
    return session_id, text, tool_calls

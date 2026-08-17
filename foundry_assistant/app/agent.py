"""Foundry agent: FoundryChatClient + Microsoft Agent Framework.

Phase 3 — Foundry IQ knowledge base grounding. The ephemeral Agent Framework
Agent (no hosted/persisted agent) is given a single retrieval tool: the
Foundry IQ knowledge base exposed over its Azure AI Search MCP endpoint. Only
`knowledge_base_retrieve` is allowed, and the system prompt requires
source-backed answers with abstention when retrieval returns nothing relevant.

Sessions are reused per thread via an in-memory dict keyed by session_id; they
are lost on restart by design. The agent definition (instructions +
construction) is versioned with this source file.

Environment variables (from .env):
    FOUNDRY_PROJECT_ENDPOINT  — project-scoped endpoint URL
    FOUNDRY_MODEL             — deployment name (e.g. gpt-4o-mini)
    KB_MCP_URL                — knowledge base Streamable HTTP MCP endpoint
    KB_MCP_SCOPE              — (optional) Entra scope for the MCP endpoint
    AZURE_SEARCH_API_KEY      — (optional) admin/query key fallback auth
"""

import os
import uuid

from azure.identity import DefaultAzureCredential
from dotenv import load_dotenv

load_dotenv()

# Data-plane scope for the Azure AI Search knowledge base MCP endpoint.
_DEFAULT_SEARCH_SCOPE = "https://search.azure.com/.default"

# The only tool the agent may call. Foundry IQ exposes several MCP tools; we
# hard-restrict to grounded retrieval so the agent cannot mutate the index or
# call anything off-corpus.
KB_RETRIEVE_TOOL = "knowledge_base_retrieve"


class Settings:
    """Foundry + knowledge base connection settings read from environment."""

    def __init__(self) -> None:
        self.project_endpoint = os.environ.get("FOUNDRY_PROJECT_ENDPOINT")
        self.model = os.environ.get("FOUNDRY_MODEL")
        self.kb_mcp_url = os.environ.get("KB_MCP_URL")
        self.kb_mcp_scope = os.environ.get("KB_MCP_SCOPE", _DEFAULT_SEARCH_SCOPE)
        self.search_api_key = os.environ.get("AZURE_SEARCH_API_KEY")

    def require(self) -> None:
        missing = [
            name
            for name, value in {
                "FOUNDRY_PROJECT_ENDPOINT": self.project_endpoint,
                "FOUNDRY_MODEL": self.model,
                "KB_MCP_URL": self.kb_mcp_url,
            }.items()
            if not value
        ]
        if missing:
            raise RuntimeError(
                f"Missing required environment variables: {', '.join(missing)}"
            )


settings = Settings()

# Shared credential: reused for both the Foundry chat client and MCP bearer
# tokens. get_token caches/refreshes internally, so calling it per request is
# cheap.
_credential = DefaultAzureCredential()

_SYSTEM_PROMPT = (
    "You are a study assistant for the Northstar learning program. "
    "You have two kinds of tools: knowledge-base retrieval and task "
    "management. Choose based on what the user asks. "
    "KNOWLEDGE QUESTIONS: for anything that could be answered from the study "
    f"notes, call the `{KB_RETRIEVE_TOOL}` tool first, then answer using only "
    "the returned passages. Cite the sources you used with bracketed markers "
    "like [1], [2] and list the corresponding source titles/ids at the end "
    "under a 'Sources:' line. If retrieval returns nothing relevant, or the "
    "question is outside the knowledge base, do not guess: reply exactly with "
    "'I don't have information on that in my knowledge base.' Never answer "
    "from prior knowledge when the tool returns no supporting passages. "
    "TASK REQUESTS: when the user wants to see, add, or complete to-dos, use "
    "the task tools — `list_tasks` to read them, `create_task` to add one, and "
    "`complete_task` to mark one done. These tools always operate on the "
    "authenticated user's own tasks; you cannot see or choose another user's "
    "tasks, so never ask for or accept a user id as an argument. When a tool "
    "returns {\"ok\": false, ...}, tell the user what went wrong instead of "
    "inventing a result. "
    "MEMORY / PREFERENCES: the user may tell you how they want to be addressed "
    "(a preferred name or nickname). Only when the user explicitly asks you to "
    "remember their name/nickname, call `remember_nickname`. Recall preferences "
    "ONLY when the user asks (e.g. 'what's my name', 'what do you remember about "
    "me') by calling `get_my_preferences` — never volunteer them unprompted. "
    "When the user asks you to forget their name/preferences, call "
    "`forget_my_preferences`. Do not store anything other than an explicitly "
    "requested preference, and never accept a user id as an argument."
)


def _kb_header_provider(_kwargs: dict) -> dict:
    """Build auth headers for each MCP request.

    Prefers an Entra bearer token via the shared credential (RBAC path);
    falls back to the Azure AI Search api-key header when a key is configured
    and no token can be acquired.
    """
    try:
        token = _credential.get_token(settings.kb_mcp_scope).token
        return {"Authorization": f"Bearer {token}"}
    except Exception:
        if settings.search_api_key:
            return {"api-key": settings.search_api_key}
        raise


def build_kb_tool():
    """Construct the Foundry IQ knowledge base MCP tool (not yet connected).

    Restricts the exposed toolset to `knowledge_base_retrieve` only. The caller
    is responsible for connecting the tool (see api.py lifespan).
    """
    from agent_framework import MCPStreamableHTTPTool

    return MCPStreamableHTTPTool(
        name="foundry-iq-knowledge-base",
        url=settings.kb_mcp_url,
        allowed_tools=[KB_RETRIEVE_TOOL],
        header_provider=_kb_header_provider,
        approval_mode="never_require",
    )


def build_agent(kb_tool):
    """Construct the ephemeral Agent Framework agent backed by FoundryChatClient.

    `kb_tool` is the connected knowledge base MCP tool. The agent also binds the
    Cosmos-backed task tools (list/create/complete). Returns an Agent instance
    (async — call with `await agent.run(...)`).
    """
    from agent_framework import Agent
    from agent_framework.foundry import FoundryChatClient

    from .memory import PREFERENCE_TOOLS
    from .tools import TASK_TOOLS

    client = FoundryChatClient(
        project_endpoint=settings.project_endpoint,
        model=settings.model,
        credential=_credential,
    )

    return Agent(
        client=client,
        name="study-assistant",
        instructions=_SYSTEM_PROMPT,
        tools=[kb_tool, *TASK_TOOLS, *PREFERENCE_TOOLS],
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
    agent, message: str, thread_id: str | None = None, user_id: str | None = None
) -> tuple[str, str, list[dict]]:
    """Run one user turn on a reused session.

    `user_id` is the authenticated caller's id; it is injected into the tool
    invocation context (never the model-visible schema) so task tools operate
    on that user's Cosmos partition only.

    Returns (session_id, reply_text, tool_calls). Pass the returned session_id
    back as thread_id on the next call to continue the same conversation.
    """
    session, session_id = get_or_create_session(agent, thread_id)
    result = await agent.run(
        message,
        session=session,
        function_invocation_kwargs={"user_id": user_id},
    )

    tool_calls = _extract_tool_calls(result)
    text = getattr(result, "text", None) or (str(result) if result else "(no content)")
    return session_id, text, tool_calls


# Content `type` values that represent an outbound tool invocation. Agent
# Framework wraps MCP tools as function calls, but some backends surface them
# as native MCP tool-call content, so accept both.
_TOOL_CALL_TYPES = {"function_call", "mcp_server_tool_call"}


def _extract_tool_calls(result) -> list[dict]:
    """Pull tool invocations out of the response messages for transparency.

    Agent Framework records tool calls as content items on the assistant
    messages (not on a top-level `tool_calls` attribute), so we walk
    `result.messages[].contents[]` and collect the call items.
    """
    import json

    calls: list[dict] = []
    for msg in getattr(result, "messages", None) or []:
        for content in getattr(msg, "contents", None) or []:
            if getattr(content, "type", None) not in _TOOL_CALL_TYPES:
                continue
            args = (
                getattr(content, "arguments", None)
                or getattr(content, "args", None)
                or {}
            )
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except (ValueError, TypeError):
                    args = {"raw": args}
            calls.append({
                "name": getattr(content, "name", "unknown"),
                "args": args if isinstance(args, dict) else {"value": args},
            })
    return calls

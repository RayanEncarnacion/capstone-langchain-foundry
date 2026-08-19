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

from .tokens import ContextBudgetManager, TokenUsageBreakdown

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
    is responsible for connecting the tool (see api.py lifespan). MCP output is
    treated as untrusted and sanitized before entering the model context.
    """
    from agent_framework import MCPStreamableHTTPTool

    from .sanitize import mcp_result_parser

    return MCPStreamableHTTPTool(
        name="foundry-iq-knowledge-base",
        url=settings.kb_mcp_url,
        allowed_tools=[KB_RETRIEVE_TOOL],
        header_provider=_kb_header_provider,
        approval_mode="never_require",
        parse_tool_results=mcp_result_parser,
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

# Pending approval requests keyed by session_id. Stored when a run pauses on
# an always_require tool so /approve can resume with the decision.
_pending_approvals: dict[str, list] = {}

# Context budget manager for enforcing token limits and stable prefix caching.
_budget_manager = ContextBudgetManager()


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



def _summarize_result(result, session_id: str) -> tuple[str, str, list[dict], list[dict]]:
    """Extract (session_id, reply, tool_calls, pending) from an AgentResponse.

    When the response contains user_input_requests of type
    function_approval_request, the run is paused: we stash the requests and
    return them as `pending` so the API layer can surface them for approval.
    """
    tool_calls = _extract_tool_calls(result)

    import json as _json

    pending: list[dict] = []
    approval_contents = getattr(result, "user_input_requests", None) or []
    if approval_contents:
        _pending_approvals[session_id] = approval_contents
        for req in approval_contents:
            fc = getattr(req, "function_call", None)
            if fc:
                args = getattr(fc, "arguments", None) or getattr(fc, "args", None) or {}
                if isinstance(args, str):
                    try:
                        args = _json.loads(args)
                    except (ValueError, TypeError):
                        args = {"raw": args}
                pending.append({
                    "name": getattr(fc, "name", "unknown"),
                    "args": args if isinstance(args, dict) else {},
                    "description": f"Approval required to run {getattr(fc, 'name', 'unknown')}",
                })
        return session_id, "", tool_calls, pending

    text = getattr(result, "text", None) or (str(result) if result else "(no content)")
    return session_id, text, tool_calls, pending


def _extract_retrieval_text(result) -> str:
    """Extract retrieved knowledge base text from message contents if present."""
    retrieval_chunks: list[str] = []
    for msg in getattr(result, "messages", None) or []:
        for content in getattr(msg, "contents", None) or []:
            # MCP knowledge base results or tool responses
            if getattr(content, "name", "") == KB_RETRIEVE_TOOL:
                txt = getattr(content, "text", "") or str(getattr(content, "result", ""))
                if txt:
                    retrieval_chunks.append(txt)
    return "\n".join(retrieval_chunks)


async def run_agent(
    agent, message: str, thread_id: str | None = None, user_id: str | None = None
) -> tuple[str, str, list[dict], list[dict], TokenUsageBreakdown]:
    """Run one user turn on a reused session with context budgeting.

    `user_id` is the authenticated caller's id; it is injected into the tool
    invocation context (never the model-visible schema) so task tools operate
    on that user's Cosmos partition only.

    Returns (session_id, reply_text, tool_calls, pending, token_usage). When `pending` is
    non-empty the run paused on an approval gate and needs a resume via
    `resume_agent`.
    """
    from agent_framework import SlidingWindowStrategy

    session, session_id = get_or_create_session(agent, thread_id)
    compaction = SlidingWindowStrategy(
        keep_last_groups=_budget_manager.config.keep_last_turns,
        preserve_system=True,
    )

    result = await agent.run(
        message,
        session=session,
        compaction_strategy=compaction,
        function_invocation_kwargs={"user_id": user_id},
    )

    session_id, reply, tool_calls, pending = _summarize_result(result, session_id)

    # Calculate token breakdown metrics
    retrieval_text = _extract_retrieval_text(result)
    history_msgs = getattr(result, "messages", None) or []
    breakdown = _budget_manager.calculate_breakdown(
        system_prompt=_SYSTEM_PROMPT,
        tools=getattr(agent, "tools", None),
        history_messages=history_msgs,
        retrieval_content=retrieval_text,
        user_message=message,
        model_output=reply,
        usage_details=getattr(result, "usage_details", None),
        model=settings.model or "o200k_base",
    )
    breakdown.log_breakdown()

    return session_id, reply, tool_calls, pending, breakdown


async def resume_agent(
    agent, thread_id: str, user_id: str, approved: bool
) -> tuple[str, str, list[dict], list[dict], TokenUsageBreakdown]:
    """Resume a paused run with a human approval decision.

    Converts the stored approval requests into approval responses (approved or
    rejected) and feeds them back into agent.run() on the same session so the
    framework either executes or skips the pending tool call.
    """
    from agent_framework import SlidingWindowStrategy
    from agent_framework._types import Content

    if thread_id not in _sessions:
        raise ValueError(f"No session found for thread_id={thread_id!r}")
    session = _sessions[thread_id]

    pending_requests = _pending_approvals.pop(thread_id, [])
    if not pending_requests:
        raise ValueError("No pending approval requests for this session.")

    approval_responses = [
        req.to_function_approval_response(approved=approved)
        for req in pending_requests
    ]

    compaction = SlidingWindowStrategy(
        keep_last_groups=_budget_manager.config.keep_last_turns,
        preserve_system=True,
    )

    result = await agent.run(
        approval_responses,
        session=session,
        compaction_strategy=compaction,
        function_invocation_kwargs={"user_id": user_id},
    )

    session_id, reply, tool_calls, pending = _summarize_result(result, thread_id)

    retrieval_text = _extract_retrieval_text(result)
    breakdown = _budget_manager.calculate_breakdown(
        system_prompt=_SYSTEM_PROMPT,
        tools=getattr(agent, "tools", None),
        history_messages=getattr(result, "messages", None) or [],
        retrieval_content=retrieval_text,
        user_message="[Approval Decision]",
        model_output=reply,
        usage_details=getattr(result, "usage_details", None),
        model=settings.model or "o200k_base",
    )
    breakdown.log_breakdown()

    return session_id, reply, tool_calls, pending, breakdown



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

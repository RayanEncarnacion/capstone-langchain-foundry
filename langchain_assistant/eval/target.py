"""Run the real agent on one dataset example and capture eval signals.

The evaluators need more than the final text: which tools fired (including a
write tool that paused for approval), the raw tool outputs (for grounding),
the chunk ids returned by search_notes (for citation checking), and whether
the answer was an abstention. We invoke the compiled agent directly (rather
than the HTTP API) so a single process traces cleanly to LangSmith.
"""

import json
import re
import uuid

from langchain_core.messages import HumanMessage

from ..app.agent import ABSTAIN_TOKEN, build_agent
from ..app.storage import build_checkpointer, build_store

# Fixed identity for eval runs: isolates eval memory/tasks from real users.
_EVAL_USER_ID = "eval-user"

# Phrases the agent uses when it declines for lack of evidence.
_ABSTAIN_MARKERS = (
    ABSTAIN_TOKEN.lower(),
    "don't have enough information",
    "do not have enough information",
    "not enough information in the notes",
)

# Cache the compiled agent: building it wires Cosmos + Search once per process.
_agent = None


def _get_agent():
    global _agent
    if _agent is None:
        _agent = build_agent(
            checkpointer=build_checkpointer(), store=build_store()
        )
    return _agent


def _looks_like_abstention(text: str) -> bool:
    low = (text or "").lower()
    return any(marker in low for marker in _ABSTAIN_MARKERS)


def _extract_chunk_ids(tool_outputs: list[dict]) -> list[str]:
    """Pull chunk_id values out of any search_notes tool output payloads."""
    ids: list[str] = []
    for out in tool_outputs:
        if out.get("name") != "search_notes":
            continue
        content = out.get("content", "")
        try:
            payload = json.loads(content) if isinstance(content, str) else content
        except (json.JSONDecodeError, TypeError):
            # Fall back to a regex sweep if the content isn't clean JSON.
            ids.extend(re.findall(r'chunk_id["\']?\s*[:=]\s*["\']?([\w.-]+)', content))
            continue
        for result in (payload or {}).get("results", []):
            cid = result.get("chunk_id")
            if cid:
                ids.append(str(cid))
    return ids


def run_target(inputs: dict) -> dict:
    """LangSmith target: run one question through the agent.

    Returns a dict the evaluators read:
      answer           -> final reply text (empty if paused for approval).
      tools_called     -> tool names invoked or proposed this run.
      tool_outputs     -> [{name, content}] for grounding/citation checks.
      retrieved_ids    -> chunk_ids returned by search_notes.
      abstained        -> True when the answer declines for lack of evidence.
      approval_required-> True when a write tool paused on the HITL gate.
    """
    agent = _get_agent()
    thread_id = f"eval-{uuid.uuid4()}"
    config = {"configurable": {"thread_id": thread_id, "user_id": _EVAL_USER_ID}}

    result = agent.invoke(
        {"messages": [HumanMessage(content=inputs["question"])]}, config=config
    )

    messages = result.get("messages", [])
    tools_called: list[str] = []
    tool_outputs: list[dict] = []
    for msg in messages:
        for call in getattr(msg, "tool_calls", None) or []:
            tools_called.append(call["name"])
        # ToolMessage carries the structured tool result back to the model.
        if msg.__class__.__name__ == "ToolMessage":
            tool_outputs.append(
                {"name": getattr(msg, "name", ""), "content": msg.content}
            )

    # A write tool that paused for approval never emits an AIMessage tool_call
    # into `messages`; it surfaces under __interrupt__ instead. Count it as a
    # proposed tool so tool-selection still scores correctly.
    approval_required = False
    for interrupt in result.get("__interrupt__", []) or []:
        value = getattr(interrupt, "value", interrupt)
        requests = value.get("action_requests", []) if isinstance(value, dict) else []
        for req in requests:
            approval_required = True
            name = req.get("name", "")
            if name:
                tools_called.append(name)

    answer = ""
    if messages and not approval_required:
        answer = messages[-1].content or ""

    return {
        "answer": answer,
        "tools_called": tools_called,
        "tool_outputs": tool_outputs,
        "retrieved_ids": _extract_chunk_ids(tool_outputs),
        "abstained": _looks_like_abstention(answer),
        "approval_required": approval_required,
    }

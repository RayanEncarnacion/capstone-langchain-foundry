"""FastAPI application exposing the agent behind authenticated endpoints.

Endpoints:
  POST /chat    -> one turn with the agent (chat, RAG, or task tools).
  POST /approve -> resume a run paused on a write-tool approval gate.

Phase 6 guardrails applied here:
  - Every request must carry a valid `Authorization: Bearer <token>`; the
    user id is derived from that token (get_current_user), never from the
    body. Anonymous requests get HTTP 401.
  - Write tools (create_task) pause the run for human approval; /chat returns
    approval_required=True and /approve resumes with the decision.
  - Inbound messages and outbound replies are screened by Azure AI Content
    Safety before they cross the trust boundary.
  - Tool-call and model-call limits are enforced inside the agent middleware.
"""

import uuid
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, HTTPException

from .agent import build_agent, resume_agent, run_agent, settings
from .auth import get_current_user
from .content_safety import screen_text
from .schemas import (
    ApprovalRequest,
    ChatRequest,
    ChatResponse,
    PendingApproval,
    ToolCall,
)
from .storage import build_checkpointer, build_store

# Built once at startup so we reuse the same compiled agent across requests.
_agent = None


# Lifespan handler: modern replacement for @app.on_event("startup").
# Code before `yield` runs at boot; code after would run at shutdown.
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan handler for the FastAPI app."""

    global _agent
    settings.require()  # crash early if endpoint/deployment are missing.
    # Wire Cosmos-backed memory: checkpointer (per-thread state) + store
    # (per-user, cross-thread long-term memory).
    checkpointer = build_checkpointer()
    store = build_store()
    _agent = build_agent(checkpointer=checkpointer, store=store)
    yield


app = FastAPI(title="Capstone: LangChain + Microsoft Foundry", lifespan=lifespan)


@app.get("/health")
def health() -> dict:
    """Tiny liveness check (unauthenticated)."""
    return {"status": "ok"}


def _guard_input(text: str) -> None:
    """Block the turn if Content Safety flags the inbound message."""
    verdict = screen_text(text)
    if not verdict.allowed:
        raise HTTPException(
            status_code=400,
            detail=f"Message blocked by content safety: {', '.join(verdict.categories)}",
        )


def _build_response(
    thread_id: str, reply: str, tool_calls: list[dict], pending: list[dict]
) -> ChatResponse:
    """Shape agent output into a validated ChatResponse.

    Screens the outbound reply, and if the run paused for approval, returns
    approval_required with the pending write action(s) instead of a final answer.
    """
    if pending:
        summary = "; ".join(
            f"{p['name']}({p['args']})" for p in pending
        )
        return ChatResponse(
            session_id=thread_id,
            message=f"Approval required before running: {summary}",
            tool_calls=[ToolCall(name=c["name"], args=c["args"]) for c in tool_calls],
            approval_required=True,
            pending=[
                PendingApproval(
                    name=p["name"], args=p["args"], description=p.get("description", "")
                )
                for p in pending
            ],
        )

    # Screen the model's answer before it leaves the service.
    verdict = screen_text(reply)
    safe_reply = reply if verdict.allowed else (
        "The response was withheld because it was flagged by content safety."
    )
    return ChatResponse(
        session_id=thread_id,
        message=safe_reply or "(no content)",
        tool_calls=[ToolCall(name=c["name"], args=c["args"]) for c in tool_calls],
    )


# response_model=ChatResponse makes FastAPI validate our OUTPUT too.
@app.post("/chat", response_model=ChatResponse)
def chat(
    request: ChatRequest, user_id: str = Depends(get_current_user)
) -> ChatResponse:
    """Send one message to the agent; it may chat, search notes, or manage tasks.

    Requires a valid bearer token. `user_id` comes from the token, not the body.
    """
    _guard_input(request.message)

    # New thread if the client didn't supply one; otherwise resume theirs.
    thread_id = request.thread_id or str(uuid.uuid4())
    try:
        print(f"Request: user={user_id} thread={thread_id} msg={request.message}")
        reply, tool_calls, pending = run_agent(
            _agent, request.message, thread_id=thread_id, user_id=user_id
        )
    except Exception as exc:  # surface agent/model/tool wiring errors as HTTP 502.
        raise HTTPException(status_code=502, detail=f"Agent call failed: {exc}") from exc

    for call in tool_calls:
        print(f"  tool: {call['name']} args={call['args']}")

    return _build_response(thread_id, reply, tool_calls, pending)
# 'c6da50a1-6903-4d40-957c-79257b900e80'
# 3553d543-6a8d-4662-9f49-609f29856f5f

@app.post("/approve", response_model=ChatResponse)
def approve(
    request: ApprovalRequest, user_id: str = Depends(get_current_user)
) -> ChatResponse:
    """Resume a paused run with an approve/reject decision for a write tool.

    Requires a valid bearer token. The decision is applied to the caller's own
    thread; identity still comes from the token, not the body.
    """
    try:
        print(
            f"Approval: user={user_id} thread={request.thread_id} "
            f"approved={request.approved}"
        )
        reply, tool_calls, pending = resume_agent(
            _agent, request.thread_id, user_id=user_id, approved=request.approved
        )
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Resume failed: {exc}") from exc

    for call in tool_calls:
        print(f"  tool: {call['name']} args={call['args']}")

    return _build_response(request.thread_id, reply, tool_calls, pending)

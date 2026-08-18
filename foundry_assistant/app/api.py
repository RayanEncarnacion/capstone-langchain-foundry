"""FastAPI app — Foundry agent with session reuse and approval gates.

Endpoints:
    GET  /health   — unauthenticated liveness check
    POST /chat     — one turn with the Foundry agent, on a reused AgentSession
    POST /approve  — resume a paused run with a human approve/reject decision
"""

from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, HTTPException

from .agent import build_agent, build_kb_tool, resume_agent, run_agent, settings
from .auth import get_current_user
from .content_safety import screen_text
from .schemas import (
    ApprovalRequest,
    ChatRequest,
    ChatResponse,
    PendingApproval,
    ToolCall,
)

_agent = None
_kb_tool = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _agent, _kb_tool
    settings.require()
    _kb_tool = build_kb_tool()
    await _kb_tool.connect()
    _agent = build_agent(_kb_tool)
    try:
        yield
    finally:
        await _kb_tool.close()


app = FastAPI(title="Capstone: Foundry Assistant", lifespan=lifespan)


@app.get("/health")
def health() -> dict:
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
    session_id: str, reply: str, tool_calls: list[dict], pending: list[dict]
) -> ChatResponse:
    """Shape agent output into a validated ChatResponse."""
    if pending:
        summary = "; ".join(f"{p['name']}({p['args']})" for p in pending)
        return ChatResponse(
            session_id=session_id,
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

    verdict = screen_text(reply)
    safe_reply = reply if verdict.allowed else (
        "The response was withheld because it was flagged by content safety."
    )
    return ChatResponse(
        session_id=session_id,
        message=safe_reply or "(no content)",
        tool_calls=[ToolCall(name=c["name"], args=c["args"]) for c in tool_calls],
    )


@app.post("/chat", response_model=ChatResponse)
async def chat(
    request: ChatRequest,
    user_id: str = Depends(get_current_user),
) -> ChatResponse:
    """Send one message to the Foundry agent on a reused AgentSession.

    Requires a valid Entra bearer token; the authenticated user id is injected
    into the agent's tools so task operations stay scoped to that user.
    """
    _guard_input(request.message)

    try:
        print(f"Request: user={user_id} thread={request.thread_id} msg={request.message}")
        session_id, reply, tool_calls, pending = await run_agent(
            _agent, request.message, thread_id=request.thread_id, user_id=user_id
        )
    except Exception as exc:
        raise HTTPException(
            status_code=502, detail=f"Agent call failed: {exc}"
        ) from exc

    return _build_response(session_id, reply, tool_calls, pending)


@app.post("/approve", response_model=ChatResponse)
async def approve(
    request: ApprovalRequest,
    user_id: str = Depends(get_current_user),
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
        session_id, reply, tool_calls, pending = await resume_agent(
            _agent, request.thread_id, user_id=user_id, approved=request.approved
        )
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Resume failed: {exc}") from exc

    return _build_response(session_id, reply, tool_calls, pending)

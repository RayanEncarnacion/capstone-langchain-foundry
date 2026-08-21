"""FastAPI app — Foundry agent with session reuse and approval gates.

Endpoints:
    GET  /health              — unauthenticated liveness check
    POST /chat                — one turn with the Foundry agent
    POST /approve             — resume a paused approval
    GET  /context/{thread_id} — inspect owner-scoped model context
"""

from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, HTTPException, Request
from opentelemetry import trace

from .agent import (
    AgentSessionError,
    build_agent,
    build_kb_tool,
    inspect_model_context,
    resume_agent,
    run_agent,
    settings,
)
from .auth import get_current_user
from .content_safety import screen_text
from .gateway import CORRELATION_HEADER, correlation_id, rate_limit_retry_after
from .schemas import (
    ApprovalRequest,
    ChatRequest,
    ChatResponse,
    ContextInspectionResponse,
    PendingApproval,
    TokenUsage,
    ToolCall,
)
from .tokens import TokenUsageBreakdown

import os

_agent = None
_kb_tool = None
_TELEMETRY_INITIALIZED = False


def _setup_azure_telemetry() -> None:
    """Configure Azure Monitor OpenTelemetry tracing at module import time."""
    global _TELEMETRY_INITIALIZED
    if _TELEMETRY_INITIALIZED:
        return
    conn_str = os.environ.get("APPLICATIONINSIGHTS_CONNECTION_STRING")
    if not conn_str:
        return
    try:
        from azure.monitor.opentelemetry import configure_azure_monitor

        configure_azure_monitor(connection_string=conn_str)
        _TELEMETRY_INITIALIZED = True
    except Exception as exc:
        print(f"Azure Monitor initialization skipped: {exc}")


# Initialize telemetry once on import
_setup_azure_telemetry()


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


@app.middleware("http")
async def correlate_request(request: Request, call_next):
    """Propagate one safe opaque ID through app, APIM, and response telemetry."""

    request_id = correlation_id(request.headers.get(CORRELATION_HEADER))
    request.state.correlation_id = request_id
    span = trace.get_current_span()
    if span.is_recording():
        span.set_attribute("app.correlation_id", request_id)
    response = await call_next(request)
    response.headers[CORRELATION_HEADER] = request_id
    return response


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
    session_id: str,
    reply: str,
    tool_calls: list[dict],
    pending: list[dict],
    breakdown: TokenUsageBreakdown | None = None,
) -> ChatResponse:
    """Shape agent output into a validated ChatResponse."""
    usage_model = TokenUsage(**breakdown.to_dict()) if breakdown else None

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
            usage=usage_model,
        )

    verdict = screen_text(reply)
    safe_reply = reply if verdict.allowed else (
        "The response was withheld because it was flagged by content safety."
    )
    return ChatResponse(
        session_id=session_id,
        message=safe_reply or "(no content)",
        tool_calls=[ToolCall(name=c["name"], args=c["args"]) for c in tool_calls],
        usage=usage_model,
    )


def _raise_model_error(exc: Exception, request_id: str, operation: str) -> None:
    """Map gateway throttling and avoid reflecting SDK errors or secrets."""

    throttled, retry_after = rate_limit_retry_after(exc)
    if throttled:
        headers = {"Retry-After": retry_after} if retry_after else None
        raise HTTPException(
            status_code=429,
            detail="Model gateway rate limit exceeded; retry later.",
            headers=headers,
        ) from exc
    raise HTTPException(
        status_code=502,
        detail=f"{operation} failed; correlation_id={request_id}",
    ) from exc


@app.post("/chat", response_model=ChatResponse)
async def chat(
    request: Request,
    payload: ChatRequest,
    user_id: str = Depends(get_current_user),
) -> ChatResponse:
    """Send one message to the Foundry agent on a reused AgentSession.

    Requires a valid Entra bearer token; the authenticated user id is injected
    into the agent's tools so task operations stay scoped to that user.
    """
    _guard_input(payload.message)
    request_id = request.state.correlation_id

    try:
        print(f"Request: correlation_id={request_id} thread={payload.thread_id}")
        session_id, reply, tool_calls, pending, breakdown = await run_agent(
            _agent,
            payload.message,
            thread_id=payload.thread_id,
            user_id=user_id,
            correlation_id=request_id,
            allow_semantic_cache=payload.allow_semantic_cache,
        )
    except AgentSessionError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Exception as exc:
        _raise_model_error(exc, request_id, "Agent call")

    return _build_response(session_id, reply, tool_calls, pending, breakdown)


@app.get("/context/{thread_id}", response_model=ContextInspectionResponse)
def inspect_context(
    thread_id: str,
    user_id: str = Depends(get_current_user),
) -> ContextInspectionResponse:
    """Inspect exact/redacted model calls for caller's own in-memory session."""

    try:
        return ContextInspectionResponse(
            **inspect_model_context(thread_id=thread_id, user_id=user_id)
        )
    except AgentSessionError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.post("/approve", response_model=ChatResponse)
async def approve(
    request: Request,
    payload: ApprovalRequest,
    user_id: str = Depends(get_current_user),
) -> ChatResponse:
    """Resume a paused run with an approve/reject decision for a write tool.

    Requires a valid bearer token. The decision is applied to the caller's own
    thread; identity still comes from the token, not the body.
    """
    request_id = request.state.correlation_id
    try:
        print(
            f"Approval: correlation_id={request_id} thread={payload.thread_id} "
            f"approved={payload.approved}"
        )
        session_id, reply, tool_calls, pending, breakdown = await resume_agent(
            _agent,
            payload.thread_id,
            user_id=user_id,
            approved=payload.approved,
            correlation_id=request_id,
        )
    except AgentSessionError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Exception as exc:
        _raise_model_error(exc, request_id, "Resume")

    return _build_response(session_id, reply, tool_calls, pending, breakdown)


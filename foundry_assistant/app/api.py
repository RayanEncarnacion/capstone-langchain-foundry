"""FastAPI app — Phase 1 Foundry baseline.

Endpoints:
    GET  /health  — unauthenticated liveness check
    POST /chat    — one turn with the Foundry agent
"""

import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException

from .agent import build_agent, run_agent, settings
from .schemas import ChatRequest, ChatResponse, ToolCall

_agent = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _agent
    settings.require()
    _agent = build_agent()
    yield


app = FastAPI(title="Capstone: Foundry Assistant", lifespan=lifespan)


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/chat", response_model=ChatResponse)
async def chat(
    request: ChatRequest
) -> ChatResponse:
    """Send one message to the Foundry agent. Requires a valid bearer token."""
    thread_id = request.thread_id or str(uuid.uuid4())

    try:
        print(f"Request: user={request.user_id} thread={thread_id} msg={request.message}")
        reply, tool_calls = await run_agent(_agent, request.message)
    except Exception as exc:
        raise HTTPException(
            status_code=502, detail=f"Agent call failed: {exc}"
        ) from exc

    return ChatResponse(
        session_id=thread_id,
        message=reply or "(no content)",
        tool_calls=[ToolCall(name=c["name"], args=c["args"]) for c in tool_calls],
    )

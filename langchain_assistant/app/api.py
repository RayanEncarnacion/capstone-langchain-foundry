"""FastAPI application exposing a single, full-featured POST /chat.

Flow: request JSON -> ChatRequest -> tool-calling agent (create_agent) ->
ChatResponse (validated) -> typed JSON back to the client.

The agent decides per turn whether to:
  - answer directly,
  - search the notes (RAG) via the search_notes tool, or
  - read/create tasks via the list_tasks / create_task tools.

So retrieval, task management, and plain chat all live behind /chat.
"""

import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException

from .agent import build_agent, run_agent, settings
from .schemas import ChatRequest, ChatResponse, ToolCall

# Built once at startup so we reuse the same compiled agent across requests.
_agent = None


# Lifespan handler: modern replacement for @app.on_event("startup").
# Code before `yield` runs at boot; code after would run at shutdown.
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan handler for the FastAPI app."""

    global _agent
    settings.require()  # crash early if endpoint/deployment are missing.
    _agent = build_agent()
    yield


app = FastAPI(title="Capstone: LangChain + Microsoft Foundry", lifespan=lifespan)


@app.get("/health")
def health() -> dict:
    """Tiny liveness check."""
    return {"status": "ok"}


# response_model=ChatResponse makes FastAPI validate our OUTPUT too:
# if we ever return something that doesn't match, it errors visibly.
@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest) -> ChatResponse:
    """Send one message to the agent; it may chat, search notes, or manage tasks."""
    try:
        print(f"Request: {request.message}")
        reply, tool_calls = run_agent(_agent, request.message)
    except Exception as exc:  # surface agent/model/tool wiring errors as HTTP 502.
        raise HTTPException(status_code=502, detail=f"Agent call failed: {exc}") from exc

    # Log which tools ran this turn (Phase 4 visibility).
    for call in tool_calls:
        print(f"  tool: {call['name']} args={call['args']}")

    # Fresh id per request for now (real memory comes in a later phase).
    session_id = str(uuid.uuid4())

    # Validate before returning. If `message` is empty, ChatResponse.message's
    # min_length=1 raises a ValidationError -> visible error, not bad JSON.
    return ChatResponse(
        session_id=session_id,
        message=reply,
        tool_calls=[ToolCall(name=c["name"], args=c["args"]) for c in tool_calls],
    )

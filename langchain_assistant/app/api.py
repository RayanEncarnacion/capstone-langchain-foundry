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

from fastapi import Depends, FastAPI, HTTPException

from .agent import build_agent, run_agent, settings
from .auth import get_current_user, init_api_keys
from .schemas import ChatRequest, ChatResponse, ToolCall
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
    init_api_keys()     # crash early if API_KEYS is missing/empty.
    # Wire Cosmos-backed memory: checkpointer (per-thread state) + store
    # (per-user, cross-thread long-term memory).
    checkpointer = build_checkpointer()
    store = build_store()
    _agent = build_agent(checkpointer=checkpointer, store=store)
    yield


app = FastAPI(title="Capstone: LangChain + Microsoft Foundry", lifespan=lifespan)


@app.get("/health")
def health() -> dict:
    """Tiny liveness check."""
    return {"status": "ok"}


# response_model=ChatResponse makes FastAPI validate our OUTPUT too:
# if we ever return something that doesn't match, it errors visibly.
@app.post("/chat", response_model=ChatResponse)
def chat(
    request: ChatRequest, user_id: str = Depends(get_current_user)
) -> ChatResponse:
    """Send one message to the agent; it may chat, search notes, or manage tasks."""
    thread_id = request.thread_id or str(uuid.uuid4())

    try:
        reply, tool_calls = run_agent(
            _agent, request.message, thread_id=thread_id, user_id=user_id
        )
    except Exception as exc:  # surface agent/model/tool wiring errors as HTTP 502.
        raise HTTPException(status_code=502, detail=f"Agent call failed: {exc}") from exc

    # Log which tools ran this turn (Phase 4 visibility).
    for call in tool_calls:
        print(f"  tool: {call['name']} args={call['args']}")

    # Echo the thread id back as session_id so the client reuses it to
    # continue the conversation (the checkpointer resumes that thread).
    return ChatResponse(
        session_id=thread_id,
        message=reply,
        tool_calls=[ToolCall(name=c["name"], args=c["args"]) for c in tool_calls],
    )

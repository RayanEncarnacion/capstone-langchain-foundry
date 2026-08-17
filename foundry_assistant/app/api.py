"""FastAPI app — Phase 2 Foundry agent with session reuse.

Endpoints:
    GET  /health  — unauthenticated liveness check
    POST /chat    — one turn with the Foundry agent, on a reused AgentSession
"""

from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, HTTPException

from .agent import build_agent, build_kb_tool, run_agent, settings
from .auth import get_current_user
from .schemas import ChatRequest, ChatResponse, ToolCall

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


@app.post("/chat", response_model=ChatResponse)
async def chat(
    request: ChatRequest,
    user_id: str = Depends(get_current_user),
) -> ChatResponse:
    """Send one message to the Foundry agent on a reused AgentSession.

    Requires a valid Entra bearer token; the authenticated user id is injected
    into the agent's tools so task operations stay scoped to that user. Omit
    thread_id to start a new session; pass a returned session_id to continue
    the same conversation with context.
    """
    try:
        print(f"Request: user={user_id} thread={request.thread_id} msg={request.message}")
        session_id, reply, tool_calls = await run_agent(
            _agent, request.message, thread_id=request.thread_id, user_id=user_id
        )
    except Exception as exc:
        raise HTTPException(
            status_code=502, detail=f"Agent call failed: {exc}"
        ) from exc

    return ChatResponse(
        session_id=session_id,
        message=reply or "(no content)",
        tool_calls=[ToolCall(name=c["name"], args=c["args"]) for c in tool_calls],
    )

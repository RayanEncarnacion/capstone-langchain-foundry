"""FastAPI application exposing POST /chat.

Flow: request JSON -> ChatRequest -> direct Foundry model call ->
ChatResponse (validated) -> typed JSON back to the client.

No RAG, no tools, no memory in this phase.
"""

import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException

from .agent import answer_with_rag, ask, build_chat_model, settings
from .schemas import AskRequest, AskResponse, ChatRequest, ChatResponse, Citation

# Built once at startup so we reuse the same client across requests.
_chat_model = None


# Lifespan handler: modern replacement for @app.on_event("startup").
# Code before `yield` runs at boot; code after would run at shutdown.
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan handler for the FastAPI app."""
    
    global _chat_model
    settings.require()  # crash early if endpoint/deployment are missing.
    _chat_model = build_chat_model()
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
    """Send one message to the Foundry model and return a typed reply."""
    try:
        print(f"Request: {request.message}")
        reply = ask(_chat_model, request.message)
    except Exception as exc:  # surface upstream/model errors as HTTP 502.
        raise HTTPException(status_code=502, detail=f"Model call failed: {exc}") from exc

    # Fresh id per request for now (real memory comes in a later phase).
    session_id = str(uuid.uuid4())

    # Validate before returning. If `message` is empty, ChatResponse.message's
    # min_length=1 raises a ValidationError -> visible error, not bad JSON.
    return ChatResponse(session_id=session_id, message=reply)


@app.post("/ask", response_model=AskResponse)
def ask_rag(request: AskRequest) -> AskResponse:
    """Explicit two-step RAG: retrieve grounded chunks, then answer + cite."""
    try:
        answer, abstained, chunks = answer_with_rag(
            _chat_model, request.question, top_k=request.top_k
        )
    except Exception as exc:  # surface retrieval/model errors as HTTP 502.
        raise HTTPException(status_code=502, detail=f"RAG call failed: {exc}") from exc

    # Print the exact chunks handed to the model (Phase 2 finish criterion).
    print(f"Question: {request.question}")
    for i, c in enumerate(chunks, start=1):
        print(f"  [{i}] {c.source} {c.chunk_id} (score={c.score:.4f})")
        print(f"      {c.content[:200]!r}")

    citations = [
        Citation(
            source=c.source,
            title=c.title,
            chunk_id=c.chunk_id,
            score=c.score,
            content=c.content,
        )
        for c in chunks
    ]

    session_id = str(uuid.uuid4())
    return AskResponse(
        session_id=session_id,
        answer=answer,
        abstained=abstained,
        citations=citations,
    )

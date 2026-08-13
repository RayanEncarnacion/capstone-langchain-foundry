## Phase 1 — Model and API baseline

Build
- Create a local FastAPI application with POST /chat.
- Call the chat model directly without RAG, tools or memory.
- Validate the result with the shared Pydantic response schema.

Tools
FastAPI, Uvicorn, Pydantic, langchain-openai, Azure Identity.

Finish when
- The endpoint returns typed JSON.
- Invalid output creates a visible validation error.
- No key or credential is hard-coded.
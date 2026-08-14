"""Shared Pydantic schemas for request and response.

These are the "contract" of the API: FastAPI uses them to parse the
incoming JSON and to validate the JSON we send back. If the model
returns something that does not fit ChatResponse, validation fails
loudly instead of silently returning bad data.
"""

from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    """What the client must send to POST /chat."""

    # The user's message. min_length=1 rejects empty strings.
    message: str = Field(..., min_length=1, description="User message to send to the model")


class ChatResponse(BaseModel):
    """The typed JSON the endpoint promises to return."""

    # Identifier for this chat turn/session (useful once memory is added).
    session_id: str = Field(..., min_length=1, description="Session identifier")
    # The model's text answer.
    message: str = Field(..., min_length=1, description="Model reply text")


class Citation(BaseModel):
    """One retrieved chunk that supported (or was offered to) the answer."""

    source: str = Field(..., description="Originating file / blob name")
    title: str = Field(..., description="Document title")
    chunk_id: str = Field(..., description="Stable chunk identifier")
    score: float = Field(..., description="Search relevance score")
    content: str = Field(..., description="The exact chunk text given to the model")


class AskRequest(BaseModel):
    """What the client sends to POST /ask (the RAG endpoint)."""

    question: str = Field(..., min_length=1, description="Question to answer from the notes")
    # How many chunks to retrieve before generating.
    top_k: int = Field(default=4, ge=1, le=20, description="Number of chunks to retrieve")


class AskResponse(BaseModel):
    """Typed RAG reply: answer, abstention flag, and the chunks used."""

    session_id: str = Field(..., min_length=1, description="Session identifier")
    answer: str = Field(..., min_length=1, description="Grounded answer or abstention")
    abstained: bool = Field(..., description="True when evidence was insufficient")
    citations: list[Citation] = Field(
        default_factory=list, description="Chunks retrieved and shown to the model"
    )

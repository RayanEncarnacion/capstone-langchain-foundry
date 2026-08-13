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

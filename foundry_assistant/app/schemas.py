"""Request / response schemas for the Foundry assistant API."""

from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    """POST /chat input. No user_id field — identity comes from the token."""

    message: str = Field(..., min_length=1, description="User message")
    user_id: str = Field(..., min_length=1, description="User ID")
    thread_id: str | None = Field(
        default=None, description="Thread to continue; omit to start a new one"
    )


class ToolCall(BaseModel):
    """A single tool the agent invoked during a turn."""

    name: str = Field(..., description="Tool name")
    args: dict = Field(default_factory=dict, description="Arguments passed")


class ChatResponse(BaseModel):
    """POST /chat output."""

    session_id: str = Field(..., min_length=1, description="Session / thread id")
    message: str = Field(..., min_length=1, description="Model reply text")
    tool_calls: list[ToolCall] = Field(
        default_factory=list, description="Tools invoked this turn"
    )

from typing import Literal

from pydantic import BaseModel, Field


class Message(BaseModel):
    """Represents a single turn in the chat history."""

    role: Literal["system", "user", "assistant"] = Field(
        ..., description="Role of the speaker for this message."
    )
    content: str = Field(..., description="Message text.")


class ChatRequest(BaseModel):
    """Incoming request payload for a chat completion."""

    user_message: str = Field(..., description="New user input that needs a reply.")
    history: list[Message] = Field(
        default_factory=list,
        description="Optional prior conversation to maintain context.",
    )
    conversation_id: str | None = Field(
        default=None,
        description=(
            "Identifier that enables the server to retain up to five prior messages "
            "across requests."
        ),
    )


class ChatResponse(BaseModel):
    """Response payload corresponding to the chatbot reply."""

    reply: str = Field(..., description="Assistant response.")

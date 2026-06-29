"""
Conversation history models.

One ConversationSession document per program run, holding an ordered list
of messages (user speech, agent speech, and tool calls) so the agent can
look back on what was discussed in this session and prior ones.
"""
from datetime import datetime, timezone
from typing import Literal, Optional

from beanie import Document
from pydantic import BaseModel, Field

Role = Literal["user", "agent", "tool"]


class ConversationMessage(BaseModel):
    role: Role
    text: str
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class ConversationSession(Document):
    session_id: str
    started_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    ended_at: Optional[datetime] = None
    messages: list[ConversationMessage] = Field(default_factory=list)

    class Settings:
        name = "conversation_sessions"
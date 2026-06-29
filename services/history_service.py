"""
History service.

All MongoDB reads/writes for conversation history go through here — the
controller never touches the ConversationSession model directly. This is
also where past-session context gets assembled for injection into a new
Gemini Live system prompt, so the agent can pick up where it left off.
"""
import uuid
from datetime import datetime, timezone

from config.settings import HISTORY_MESSAGES_PER_SESSION_LIMIT, HISTORY_SESSIONS_LIMIT
from models.conversation import ConversationMessage, ConversationSession


async def start_session() -> ConversationSession:
    """Create and persist a new session document, returning it for use as
    the in-memory handle for the rest of the run."""
    session = ConversationSession(session_id=str(uuid.uuid4()))
    await session.insert()
    return session


async def add_message(session: ConversationSession, role: str, text: str) -> None:
    """Append one message to the active session and save."""
    text = (text or "").strip()
    if not text:
        return
    session.messages.append(ConversationMessage(role=role, text=text))
    await session.save()


async def end_session(session: ConversationSession) -> None:
    session.ended_at = datetime.now(timezone.utc)
    await session.save()


async def get_recent_sessions(limit: int = HISTORY_SESSIONS_LIMIT) -> list[ConversationSession]:
    return await ConversationSession.find_all().sort("-started_at").limit(limit).to_list()


async def get_session_by_id(session_id: str) -> ConversationSession | None:
    return await ConversationSession.find_one(ConversationSession.session_id == session_id)


async def build_context_summary(
    sessions_limit: int = HISTORY_SESSIONS_LIMIT,
    messages_per_session_limit: int = HISTORY_MESSAGES_PER_SESSION_LIMIT,
) -> str:
    """
    Pull the last few sessions and format them as plain text so they can be
    appended to the Gemini Live system prompt as conversation memory.

    Returns an empty string if there's no prior history (e.g. first run),
    in which case nothing extra is added to the prompt.
    """
    sessions = await get_recent_sessions(sessions_limit)
    if not sessions:
        return ""

    lines: list[str] = []
    for session in reversed(sessions):  # oldest session first
        for msg in session.messages[-messages_per_session_limit:]:
            speaker = "User" if msg.role == "user" else "Agent"
            lines.append(f"{speaker}: {msg.text}")

    if not lines:
        return ""

    return (
        "\n\n## Memory of recent past study sessions with this user\n"
        "(For your context only — never read this section aloud or refer "
        "to it directly unless the user asks what you've covered before.)\n"
        + "\n".join(lines)
    )
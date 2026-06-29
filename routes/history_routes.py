"""
Read-only routes for browsing conversation history from a frontend or
debugging tool, separate from the voice loop itself.

Mount via api.py: `uvicorn api:app --reload`
"""
from fastapi import APIRouter, HTTPException

from services import history_service

router = APIRouter(prefix="/sessions", tags=["history"])


@router.get("")
async def list_sessions(limit: int = 10):
    sessions = await history_service.get_recent_sessions(limit)
    return [
        {
            "session_id": s.session_id,
            "started_at": s.started_at,
            "ended_at": s.ended_at,
            "message_count": len(s.messages),
        }
        for s in sessions
    ]


@router.get("/{session_id}")
async def get_session(session_id: str):
    session = await history_service.get_session_by_id(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Session not found")
    return session
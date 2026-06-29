"""
Centralised configuration.

All env vars are loaded once here. Every other module should import from
this file rather than calling os.environ / load_dotenv directly, so there
is exactly one source of truth for config.
"""
import os
from pathlib import Path

from dotenv import load_dotenv

ROOT_DIR = Path(__file__).resolve().parent.parent
load_dotenv(ROOT_DIR / ".env")

# --- Gemini -----------------------------------------------------------------
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")

# Keep this overridable via env so you can swap models without touching code.
GEMINI_LIVE_MODEL = os.environ.get(
    "GEMINI_LIVE_MODEL", "gemini-3.1-flash-live-preview"
)

# --- Audio -------------------------------------------------------------------
INPUT_SAMPLE_RATE = 16_000    # Gemini Live expects 16 kHz mono PCM16 in
OUTPUT_SAMPLE_RATE = 24_000   # Gemini Live emits  24 kHz mono PCM16 out
CHANNELS = 1
CHUNK_SIZE = 1024             # ~64 ms at 16 kHz — keeps VAD responsive

# --- MongoDB -----------------------------------------------------------------
MONGO_URI = os.environ.get("MONGO_URI", "mongodb://localhost:27017")
MONGO_DB_NAME = os.environ.get("MONGO_DB_NAME", "cs_study_voice_agent")

# How much past conversation to pull back in as context for a new session.
HISTORY_SESSIONS_LIMIT = int(os.environ.get("HISTORY_SESSIONS_LIMIT", "3"))
HISTORY_MESSAGES_PER_SESSION_LIMIT = int(
    os.environ.get("HISTORY_MESSAGES_PER_SESSION_LIMIT", "20")
)


def require_gemini_api_key() -> str:
    if not GEMINI_API_KEY:
        raise SystemExit(
            "Set GEMINI_API_KEY or GOOGLE_API_KEY in your environment or a .env file."
        )
    return GEMINI_API_KEY
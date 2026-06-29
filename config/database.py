"""
MongoDB connection setup.

Uses pymongo's native async client (AsyncMongoClient) as the driver and
Beanie as the ODM. Motor reached end-of-life in May 2026 and current Beanie
releases target pymongo's async API directly — Motor's AsyncIOMotorClient
is missing methods (e.g. append_metadata) that newer Beanie versions call
during init_beanie(), which is what causes the
"MotorDatabase object is not callable" crash if you use Motor instead.
"""
from pymongo import AsyncMongoClient
from beanie import init_beanie

from config.settings import MONGO_URI, MONGO_DB_NAME
from models.conversation import ConversationSession

_client: AsyncMongoClient | None = None


async def init_db() -> None:
    """Open the Mongo connection and register document models with Beanie.

    Call this once at startup before any ConversationSession queries.
    """
    global _client
    if _client is not None:
        return  # already initialised

    _client = AsyncMongoClient(MONGO_URI)
    database = _client[MONGO_DB_NAME]
    await init_beanie(database=database, document_models=[ConversationSession])


async def close_db() -> None:
    """Close the Mongo connection on shutdown."""
    global _client
    if _client is not None:
        await _client.close()
        _client = None
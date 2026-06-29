"""
Entry point for the CS Study Voice Agent.

    python main.py

Requires GEMINI_API_KEY (or GOOGLE_API_KEY) and a reachable MongoDB
instance (MONGO_URI, defaults to mongodb://localhost:27017).
"""
import asyncio

from config.database import close_db, init_db
from controllers import voice_controller


async def main() -> None:
    await init_db()
    try:
        await voice_controller.run()
    finally:
        await close_db()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
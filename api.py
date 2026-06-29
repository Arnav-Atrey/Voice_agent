"""
Optional HTTP API for inspecting conversation history — useful if you want
to view past study sessions from a frontend, separate from the voice agent
itself (which runs as a standalone script via main.py).

Run with:  uvicorn api:app --reload
"""
from contextlib import asynccontextmanager

from fastapi import FastAPI

from config.database import close_db, init_db
from routes.history_routes import router as history_router


@asynccontextmanager
async def lifespan(app: FastAPI):
    await init_db()
    yield
    await close_db()


app = FastAPI(title="CS Study Voice Agent - History API", lifespan=lifespan)
app.include_router(history_router)
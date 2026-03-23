from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

from agentsafe.storage.repository import EventRepository

from .routes import make_router

_STATIC_DIR = Path(__file__).parent / "static"


def create_app(repo: EventRepository) -> FastAPI:
    app = FastAPI(title="AgentSafe Dashboard", version="0.1.0")
    app.include_router(make_router(repo), prefix="/api")

    @app.get("/", response_class=HTMLResponse)
    async def index():
        return (_STATIC_DIR / "index.html").read_text()

    return app

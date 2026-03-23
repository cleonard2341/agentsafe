from __future__ import annotations

from typing import Optional

from fastapi import APIRouter, Query

from agentsafe.models import Severity
from agentsafe.storage.repository import EventRepository


def make_router(repo: EventRepository) -> APIRouter:
    router = APIRouter()

    @router.get("/events")
    def list_events(
        session_id: Optional[str] = Query(None),
        flagged_only: bool = Query(False),
        limit: int = Query(50, le=500),
        offset: int = Query(0),
    ):
        events = repo.list_events(
            session_id=session_id,
            flagged_only=flagged_only,
            limit=limit,
            offset=offset,
        )
        return [e.model_dump(mode="json") for e in events]

    @router.get("/events/{event_id}")
    def get_event(event_id: str):
        event = repo.get_event(event_id)
        if not event:
            from fastapi import HTTPException
            raise HTTPException(status_code=404, detail="Event not found")
        detections = repo.list_detections(event_id=event_id)
        return {
            "event": event.model_dump(mode="json"),
            "detections": [d.model_dump(mode="json") for d in detections],
        }

    @router.get("/detections")
    def list_detections(
        event_id: Optional[str] = Query(None),
        severity: Optional[Severity] = Query(None),
        limit: int = Query(50, le=500),
    ):
        detections = repo.list_detections(
            event_id=event_id, severity=severity, limit=limit
        )
        return [d.model_dump(mode="json") for d in detections]

    @router.get("/stats")
    def stats():
        return repo.stats()

    return router

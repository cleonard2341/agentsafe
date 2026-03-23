from __future__ import annotations

import json
from typing import Any

from agentsafe.models import Detection, Event, Severity, ToolCall
from agentsafe.storage.database import Database


class EventRepository:
    def __init__(self, db: Database) -> None:
        self._db = db

    # ------------------------------------------------------------------ writes

    def save_event(self, event: Event) -> None:
        with self._db.transaction() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO events
                    (id, session_id, timestamp, messages, response_content,
                     tool_calls, model, flagged)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    event.id,
                    event.session_id,
                    event.timestamp.isoformat(),
                    json.dumps(event.messages),
                    event.response_content,
                    json.dumps([tc.model_dump() for tc in event.tool_calls]),
                    event.model,
                    int(event.flagged),
                ),
            )

    def save_detections(self, detections: list[Detection]) -> None:
        if not detections:
            return
        with self._db.transaction() as conn:
            conn.executemany(
                """
                INSERT OR REPLACE INTO detections
                    (id, event_id, detector_name, severity, label, detail, timestamp)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    (
                        d.id,
                        d.event_id,
                        d.detector_name,
                        d.severity.value,
                        d.label,
                        json.dumps(d.detail),
                        d.timestamp.isoformat(),
                    )
                    for d in detections
                ],
            )

    # ------------------------------------------------------------------ reads

    def get_event(self, event_id: str) -> Event | None:
        row = self._db._conn.execute(
            "SELECT * FROM events WHERE id = ?", (event_id,)
        ).fetchone()
        return self._row_to_event(row) if row else None

    def list_events(
        self,
        session_id: str | None = None,
        flagged_only: bool = False,
        limit: int = 100,
        offset: int = 0,
    ) -> list[Event]:
        where, params = self._build_event_filter(session_id, flagged_only)
        rows = self._db._conn.execute(
            f"SELECT * FROM events {where} ORDER BY timestamp DESC LIMIT ? OFFSET ?",
            params + [limit, offset],
        ).fetchall()
        return [self._row_to_event(r) for r in rows]

    def list_detections(
        self,
        event_id: str | None = None,
        severity: Severity | None = None,
        limit: int = 100,
    ) -> list[Detection]:
        clauses, params = [], []
        if event_id:
            clauses.append("event_id = ?")
            params.append(event_id)
        if severity:
            clauses.append("severity = ?")
            params.append(severity.value)
        where = ("WHERE " + " AND ".join(clauses)) if clauses else ""
        rows = self._db._conn.execute(
            f"SELECT * FROM detections {where} ORDER BY timestamp DESC LIMIT ?",
            params + [limit],
        ).fetchall()
        return [self._row_to_detection(r) for r in rows]

    def stats(self) -> dict[str, Any]:
        conn = self._db._conn
        total = conn.execute("SELECT COUNT(*) FROM events").fetchone()[0]
        flagged = conn.execute("SELECT COUNT(*) FROM events WHERE flagged = 1").fetchone()[0]
        by_severity = {
            row["severity"]: row["cnt"]
            for row in conn.execute(
                "SELECT severity, COUNT(*) as cnt FROM detections GROUP BY severity"
            ).fetchall()
        }
        return {"total_events": total, "flagged_events": flagged, "by_severity": by_severity}

    # -------------------------------------------------- behavioral embeddings

    def save_embedding(self, event_id: str, fingerprint: str, embedding: list[float]) -> None:
        import uuid
        with self._db.transaction() as conn:
            conn.execute(
                """
                INSERT INTO behavioral_embeddings (id, event_id, fingerprint, embedding, timestamp)
                VALUES (?, ?, ?, ?, datetime('now'))
                """,
                (str(uuid.uuid4()), event_id, fingerprint, json.dumps(embedding)),
            )

    def load_embeddings(self, limit: int = 1000) -> list[tuple[str, list[float]]]:
        """Returns list of (fingerprint, embedding) tuples, oldest first."""
        rows = self._db._conn.execute(
            "SELECT fingerprint, embedding FROM behavioral_embeddings ORDER BY timestamp ASC LIMIT ?",
            (limit,),
        ).fetchall()
        return [(r["fingerprint"], json.loads(r["embedding"])) for r in rows]

    def count_embeddings(self) -> int:
        return self._db._conn.execute(
            "SELECT COUNT(*) FROM behavioral_embeddings"
        ).fetchone()[0]

    # ---------------------------------------------------------------- helpers

    def _build_event_filter(
        self, session_id: str | None, flagged_only: bool
    ) -> tuple[str, list]:
        clauses, params = [], []
        if session_id:
            clauses.append("session_id = ?")
            params.append(session_id)
        if flagged_only:
            clauses.append("flagged = 1")
        where = ("WHERE " + " AND ".join(clauses)) if clauses else ""
        return where, params

    def _row_to_event(self, row) -> Event:
        return Event(
            id=row["id"],
            session_id=row["session_id"],
            timestamp=row["timestamp"],
            messages=json.loads(row["messages"]),
            response_content=row["response_content"],
            tool_calls=[ToolCall(**tc) for tc in json.loads(row["tool_calls"])],
            model=row["model"],
            flagged=bool(row["flagged"]),
        )

    def _row_to_detection(self, row) -> Detection:
        return Detection(
            id=row["id"],
            event_id=row["event_id"],
            detector_name=row["detector_name"],
            severity=Severity(row["severity"]),
            label=row["label"],
            detail=json.loads(row["detail"]),
            timestamp=row["timestamp"],
        )

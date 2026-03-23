from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from pathlib import Path

_DDL = """
CREATE TABLE IF NOT EXISTS events (
    id          TEXT PRIMARY KEY,
    session_id  TEXT NOT NULL,
    timestamp   TEXT NOT NULL,
    messages    TEXT NOT NULL,   -- JSON
    response_content TEXT,
    tool_calls  TEXT NOT NULL,   -- JSON
    model       TEXT NOT NULL,
    flagged     INTEGER NOT NULL DEFAULT 0
);

CREATE TABLE IF NOT EXISTS detections (
    id              TEXT PRIMARY KEY,
    event_id        TEXT NOT NULL REFERENCES events(id),
    detector_name   TEXT NOT NULL,
    severity        TEXT NOT NULL,
    label           TEXT NOT NULL,
    detail          TEXT NOT NULL,   -- JSON
    timestamp       TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS behavioral_embeddings (
    id          TEXT PRIMARY KEY,
    event_id    TEXT NOT NULL REFERENCES events(id),
    fingerprint TEXT NOT NULL,
    embedding   TEXT NOT NULL,   -- JSON array of floats
    timestamp   TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_events_session    ON events(session_id);
CREATE INDEX IF NOT EXISTS idx_events_flagged    ON events(flagged);
CREATE INDEX IF NOT EXISTS idx_detections_event  ON detections(event_id);
CREATE INDEX IF NOT EXISTS idx_detections_sev    ON detections(severity);
CREATE INDEX IF NOT EXISTS idx_embeddings_event  ON behavioral_embeddings(event_id);
"""


class Database:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self.path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.executescript(_DDL)
        self._conn.commit()

    @contextmanager
    def transaction(self):
        try:
            yield self._conn
            self._conn.commit()
        except Exception:
            self._conn.rollback()
            raise

    def close(self) -> None:
        self._conn.close()

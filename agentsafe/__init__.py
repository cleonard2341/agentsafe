from __future__ import annotations

import uuid
from pathlib import Path
from typing import Any

from agentsafe.claude_interceptor import ClaudeSafeClient
from agentsafe.detectors import BaseDetector, BehavioralBaselineDetector, default_detectors
from agentsafe.interceptor import SafeClient
from agentsafe.pipeline import DetectorPipeline
from agentsafe.storage.database import Database
from agentsafe.storage.repository import EventRepository

__version__ = "0.1.0"
__all__ = ["wrap", "wrap_claude", "SafeClient", "ClaudeSafeClient", "BaseDetector"]


def wrap_claude(
    client: Any,
    *,
    detectors: list[BaseDetector] | None = None,
    db_path: str | Path = ".agentsafe/events.db",
    session_id: str | None = None,
) -> ClaudeSafeClient:
    """
    Wrap an anthropic.Anthropic() client with AgentSafe monitoring.

    Example:
        import anthropic, agentsafe
        client = agentsafe.wrap_claude(anthropic.Anthropic())
    """
    db = Database(db_path)
    repo = EventRepository(db)
    active_detectors = detectors if detectors is not None else [
        *default_detectors(),
        BehavioralBaselineDetector(repo=repo),
    ]
    pipeline = DetectorPipeline(active_detectors)
    sid = session_id or str(uuid.uuid4())
    return ClaudeSafeClient(client, pipeline, repo, sid)


def wrap(
    client: Any,
    *,
    detectors: list[BaseDetector] | None = None,
    db_path: str | Path = ".agentsafe/events.db",
    session_id: str | None = None,
) -> SafeClient:
    """
    Wrap any OpenAI-compatible client with AgentSafe monitoring.

    Example:
        import openai, agentsafe
        client = agentsafe.wrap(openai.OpenAI())
        # Use client exactly as you would openai.OpenAI()
    """
    db = Database(db_path)
    repo = EventRepository(db)
    # If caller didn't supply detectors, build defaults + behavioral baseline
    active_detectors = detectors if detectors is not None else [
        *default_detectors(),
        BehavioralBaselineDetector(repo=repo),
    ]
    pipeline = DetectorPipeline(active_detectors)
    sid = session_id or str(uuid.uuid4())
    return SafeClient(client, pipeline, repo, sid)

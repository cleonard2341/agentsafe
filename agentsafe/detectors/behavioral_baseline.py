from __future__ import annotations

import math
from typing import TYPE_CHECKING

from agentsafe.detectors.base import BaseDetector
from agentsafe.models import Detection, Event, Severity

if TYPE_CHECKING:
    from agentsafe.storage.repository import EventRepository


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    mag_a = math.sqrt(sum(x * x for x in a))
    mag_b = math.sqrt(sum(x * x for x in b))
    if mag_a == 0 or mag_b == 0:
        return 0.0
    return dot / (mag_a * mag_b)


def _centroid(embeddings: list[list[float]]) -> list[float]:
    n = len(embeddings)
    dim = len(embeddings[0])
    return [sum(e[i] for e in embeddings) / n for i in range(dim)]


def _event_fingerprint(event: Event) -> str:
    """
    Create a text summary of the event that captures behavioral patterns:
    tool call sequence, response tone, message roles.
    """
    parts = []

    # Tool call sequence
    if event.tool_calls:
        tools = " -> ".join(tc.function_name for tc in event.tool_calls)
        parts.append(f"tools: {tools}")
        # Include argument keys (not values) to capture intent without noise
        for tc in event.tool_calls:
            arg_keys = ", ".join(sorted(tc.arguments.keys()))
            parts.append(f"{tc.function_name}({arg_keys})")

    # Response content (first 300 chars)
    if event.response_content:
        parts.append(f"response: {event.response_content[:300]}")

    # Last user message (captures task type)
    for msg in reversed(event.messages):
        if msg.get("role") == "user":
            content = msg.get("content", "")
            if isinstance(content, list):
                content = " ".join(p.get("text", "") for p in content if isinstance(p, dict))
            parts.append(f"user: {str(content)[:200]}")
            break

    return " | ".join(parts) if parts else "empty event"


class BehavioralBaselineDetector(BaseDetector):
    """
    Learns what "normal" looks like for this agent over the first
    `min_samples` events, then flags events that deviate significantly
    from the established behavioral baseline.

    Uses sentence-transformers (local, no API) to embed event fingerprints
    and cosine similarity against the centroid of the baseline.
    """

    name = "behavioral_baseline"
    default_severity = Severity.warning

    def __init__(
        self,
        repo: EventRepository,
        min_samples: int = 20,
        anomaly_threshold: float = 0.5,
        model_name: str = "all-MiniLM-L6-v2",
    ) -> None:
        self._repo = repo
        self.min_samples = min_samples
        self.anomaly_threshold = anomaly_threshold
        self.model_name = model_name
        self._encoder = None   # lazy-loaded
        self._centroid: list[float] | None = None
        self._baseline_count: int = 0

    def _get_encoder(self):
        if self._encoder is None:
            try:
                from sentence_transformers import SentenceTransformer
                self._encoder = SentenceTransformer(self.model_name)
            except ImportError:
                raise RuntimeError(
                    "sentence-transformers is required for BehavioralBaselineDetector. "
                    "Install it with: pip install sentence-transformers"
                )
        return self._encoder

    def _embed(self, text: str) -> list[float]:
        encoder = self._get_encoder()
        return encoder.encode(text, convert_to_numpy=True).tolist()

    def _refresh_centroid(self) -> None:
        """Reload baseline from DB and recompute centroid."""
        stored = self._repo.load_embeddings(limit=500)
        self._baseline_count = len(stored)
        if stored:
            embeddings = [e for _, e in stored]
            self._centroid = _centroid(embeddings)

    def detect(self, event: Event) -> list[Detection]:
        try:
            return self._detect(event)
        except RuntimeError as e:
            # sentence-transformers not installed — skip silently
            if "sentence-transformers" in str(e):
                return []
            raise
        except Exception:
            return []

    def _detect(self, event: Event) -> list[Detection]:
        fingerprint = _event_fingerprint(event)
        embedding = self._embed(fingerprint)

        # Refresh centroid every 10 events to pick up new baseline samples
        stored_count = self._repo.count_embeddings()
        if stored_count != self._baseline_count:
            self._refresh_centroid()

        # Always store the embedding (we learn from every event)
        self._repo.save_embedding(event.id, fingerprint, embedding)

        # Still in learning phase — not enough data to detect anomalies yet
        if stored_count < self.min_samples:
            return []

        # Compare against baseline centroid
        assert self._centroid is not None
        similarity = _cosine_similarity(embedding, self._centroid)

        if similarity < self.anomaly_threshold:
            return [
                self._make_detection(
                    event,
                    label="Behavior deviates significantly from baseline",
                    severity=Severity.warning,
                    detail={
                        "similarity_to_baseline": round(similarity, 4),
                        "threshold": self.anomaly_threshold,
                        "baseline_size": stored_count,
                        "fingerprint": fingerprint[:300],
                    },
                )
            ]

        return []

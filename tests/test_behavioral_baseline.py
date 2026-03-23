"""
Tests for BehavioralBaselineDetector that do NOT require sentence-transformers.
We mock the encoder so tests run without the heavy ML dependency.
"""
import math
from unittest.mock import MagicMock, patch

import pytest

from agentsafe.detectors.behavioral_baseline import (
    BehavioralBaselineDetector,
    _centroid,
    _cosine_similarity,
    _event_fingerprint,
)
from agentsafe.models import Event, ToolCall


def make_event(tools=None, response=None, user_msg="do something"):
    tool_calls = [ToolCall(id=f"tc{i}", function_name=t, arguments={}) for i, t in enumerate(tools or [])]
    messages = [{"role": "user", "content": user_msg}]
    return Event(
        session_id="test",
        messages=messages,
        response_content=response,
        tool_calls=tool_calls,
        model="gpt-4o",
    )


# ── unit helpers ─────────────────────────────────────────────────────────────

def test_cosine_similarity_identical():
    v = [1.0, 0.0, 0.5]
    assert _cosine_similarity(v, v) == pytest.approx(1.0)


def test_cosine_similarity_orthogonal():
    assert _cosine_similarity([1.0, 0.0], [0.0, 1.0]) == pytest.approx(0.0)


def test_centroid():
    embs = [[1.0, 0.0], [0.0, 1.0]]
    c = _centroid(embs)
    assert c == pytest.approx([0.5, 0.5])


def test_fingerprint_includes_tool_names():
    event = make_event(tools=["search_web", "write_file"])
    fp = _event_fingerprint(event)
    assert "search_web" in fp
    assert "write_file" in fp


# ── detector behaviour ────────────────────────────────────────────────────────

def _make_detector(repo, min_samples=5, threshold=0.5):
    det = BehavioralBaselineDetector(repo=repo, min_samples=min_samples, anomaly_threshold=threshold)
    # Inject a mock encoder that returns a fixed vector
    mock_enc = MagicMock()
    mock_enc.encode = MagicMock(return_value=_numpy_like([1.0, 0.0, 0.0]))
    det._encoder = mock_enc
    return det


def _numpy_like(lst):
    """Return an object with .tolist() like numpy arrays do."""
    m = MagicMock()
    m.tolist.return_value = lst
    return m


def test_learning_phase_no_detections(repo):
    """Should not flag anything until min_samples events are seen."""
    det = _make_detector(repo, min_samples=5)
    for _ in range(4):
        event = make_event()
        repo.save_event(event)
        detections = det.detect(event)
        assert detections == []


def test_flags_anomaly_after_baseline(repo):
    """After baseline is built, a very different embedding should be flagged."""
    det = BehavioralBaselineDetector(repo=repo, min_samples=3, anomaly_threshold=0.5)
    mock_enc = MagicMock()

    # First 3 calls return a "normal" vector; 4th returns an orthogonal one
    normal = _numpy_like([1.0, 0.0, 0.0])
    anomalous = _numpy_like([0.0, 1.0, 0.0])  # cosine sim to normal centroid = 0.0

    call_count = [0]
    def side_effect(text, **kwargs):
        call_count[0] += 1
        return normal if call_count[0] <= 3 else anomalous

    mock_enc.encode = side_effect
    det._encoder = mock_enc

    # Build baseline
    for _ in range(3):
        e = make_event()
        repo.save_event(e)
        det.detect(e)

    # Anomalous event
    anomaly_event = make_event(tools=["delete_everything"], user_msg="destroy all files")
    repo.save_event(anomaly_event)
    detections = det.detect(anomaly_event)

    assert len(detections) == 1
    assert detections[0].detector_name == "behavioral_baseline"
    assert detections[0].detail["similarity_to_baseline"] == pytest.approx(0.0, abs=0.01)


def test_no_flag_for_normal_event(repo):
    """Event similar to baseline should not be flagged."""
    det = BehavioralBaselineDetector(repo=repo, min_samples=3, anomaly_threshold=0.5)
    mock_enc = MagicMock()
    # All events use the same vector — similarity will be 1.0
    mock_enc.encode = MagicMock(return_value=_numpy_like([1.0, 0.0, 0.0]))
    det._encoder = mock_enc

    for _ in range(4):
        e = make_event()
        repo.save_event(e)
        detections = det.detect(e)

    assert detections == []


def test_missing_sentence_transformers_is_silent(repo):
    """If sentence-transformers isn't installed, detector returns [] quietly."""
    det = BehavioralBaselineDetector(repo=repo, min_samples=1)
    # _encoder is None, and importing will fail
    with patch.dict("sys.modules", {"sentence_transformers": None}):
        event = make_event()
        repo.save_event(event)
        # Should not raise
        result = det.detect(event)
        assert result == []

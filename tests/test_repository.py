from agentsafe.models import Detection, Event, Severity


def make_event(session_id="sess-1", flagged=False):
    return Event(session_id=session_id, messages=[], model="gpt-4o", flagged=flagged)


def test_save_and_get_event(repo):
    event = make_event()
    repo.save_event(event)
    fetched = repo.get_event(event.id)
    assert fetched is not None
    assert fetched.id == event.id


def test_list_events_flagged_only(repo):
    e1 = make_event(flagged=True)
    e2 = make_event(flagged=False)
    repo.save_event(e1)
    repo.save_event(e2)
    flagged = repo.list_events(flagged_only=True)
    assert len(flagged) == 1
    assert flagged[0].id == e1.id


def test_save_and_list_detections(repo):
    event = make_event()
    repo.save_event(event)
    det = Detection(
        event_id=event.id,
        detector_name="test",
        severity=Severity.warning,
        label="test detection",
    )
    repo.save_detections([det])
    dets = repo.list_detections(event_id=event.id)
    assert len(dets) == 1
    assert dets[0].label == "test detection"


def test_stats(repo):
    e1 = make_event(flagged=True)
    e2 = make_event(flagged=False)
    repo.save_event(e1)
    repo.save_event(e2)
    det = Detection(event_id=e1.id, detector_name="x", severity=Severity.critical, label="x")
    repo.save_detections([det])
    s = repo.stats()
    assert s["total_events"] == 2
    assert s["flagged_events"] == 1
    assert s["by_severity"]["critical"] == 1

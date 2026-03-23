from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from agentsafe.models import Detection, Event, Severity


class BaseDetector(ABC):
    name: str
    default_severity: Severity = Severity.warning

    @abstractmethod
    def detect(self, event: Event) -> list[Detection]:
        """
        Inspect event.messages, event.tool_calls, event.response_content.
        Return [] if nothing flagged. Never raise.
        """
        ...

    def _make_detection(
        self,
        event: Event,
        label: str,
        severity: Severity | None = None,
        detail: dict[str, Any] | None = None,
    ) -> Detection:
        return Detection(
            event_id=event.id,
            detector_name=self.name,
            severity=severity or self.default_severity,
            label=label,
            detail=detail or {},
        )

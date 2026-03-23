from __future__ import annotations

from dataclasses import dataclass, field

from agentsafe.detectors.base import BaseDetector
from agentsafe.models import Detection, Event, Severity


@dataclass
class PipelineResult:
    detections: list[Detection] = field(default_factory=list)

    @property
    def has_critical(self) -> bool:
        return any(d.severity == Severity.critical for d in self.detections)

    @property
    def flagged(self) -> bool:
        return len(self.detections) > 0


class DetectorPipeline:
    def __init__(
        self,
        detectors: list[BaseDetector],
        stop_on_critical: bool = False,
    ) -> None:
        self.detectors = detectors
        self.stop_on_critical = stop_on_critical

    def run(self, event: Event) -> PipelineResult:
        result = PipelineResult()

        for detector in self.detectors:
            try:
                found = detector.detect(event)
            except Exception:
                # Never let a detector crash the agent
                continue

            result.detections.extend(found)

            if self.stop_on_critical and result.has_critical:
                break

        return result

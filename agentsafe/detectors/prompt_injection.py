from __future__ import annotations

import re

from agentsafe.detectors.base import BaseDetector
from agentsafe.models import Detection, Event, Severity

# Patterns that suggest an attempt to override the agent's instructions
_INJECTION_PATTERNS: list[tuple[str, str]] = [
    (r"ignore\s+(all\s+)?previous\s+instructions?", "ignore previous instructions"),
    (r"disregard\s+(all\s+)?previous\s+instructions?", "disregard previous instructions"),
    (r"you\s+are\s+now\s+", "role reassignment attempt"),
    (r"forget\s+(all\s+)?previous\s+(instructions?|context)", "context wipe attempt"),
    (r"new\s+instructions?\s*:", "inline instruction override"),
    (r"system\s*:\s*you", "inline system prompt injection"),
    (r"\[INST\]|\[\/INST\]|<\|system\|>|<\|user\|>", "special token injection"),
    (r"do\s+not\s+follow\s+(your\s+)?(previous\s+)?instructions?", "instruction negation"),
]

_COMPILED = [(re.compile(p, re.IGNORECASE), label) for p, label in _INJECTION_PATTERNS]

# Only scan content that came from outside the agent (tool results, user turns)
_EXTERNAL_ROLES = {"user", "tool", "function"}


class PromptInjectionDetector(BaseDetector):
    name = "prompt_injection"
    default_severity = Severity.critical

    def detect(self, event: Event) -> list[Detection]:
        detections: list[Detection] = []

        for msg in event.messages:
            role = msg.get("role", "")
            if role not in _EXTERNAL_ROLES:
                continue

            content = self._extract_text(msg)
            if not content:
                continue

            for pattern, label in _COMPILED:
                if pattern.search(content):
                    detections.append(
                        self._make_detection(
                            event,
                            label=f"Prompt injection pattern: {label}",
                            detail={"role": role, "pattern": label, "snippet": content[:200]},
                        )
                    )
                    break  # one detection per message

        return detections

    def _extract_text(self, msg: dict) -> str:
        content = msg.get("content") or ""
        if isinstance(content, list):
            # Handle multi-part content (OpenAI vision format)
            return " ".join(
                part.get("text", "") for part in content if isinstance(part, dict)
            )
        return str(content)

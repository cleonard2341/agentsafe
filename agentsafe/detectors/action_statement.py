from __future__ import annotations

import re

from agentsafe.detectors.base import BaseDetector
from agentsafe.models import Detection, Event, Severity

# Phrases the assistant says that imply a passive/read-only action
_PASSIVE_PHRASES = [
    re.compile(r"i('ll| will| am going to)?\s+(just\s+)?(look|search|check|read|retrieve|fetch|find|browse|scan)", re.IGNORECASE),
    re.compile(r"let me\s+(just\s+)?(look|search|check|read|retrieve|fetch|find|browse|scan)", re.IGNORECASE),
    re.compile(r"i('ll| will)?\s+only\s+read", re.IGNORECASE),
    re.compile(r"i('m| am)\s+just\s+(going to\s+)?(search|look|check)", re.IGNORECASE),
]

# Tool call actions that are write/mutate/execute (not read)
_WRITE_ACTIONS = {
    "write", "create", "delete", "remove", "modify", "update", "patch",
    "execute", "run", "send", "post", "upload", "install", "deploy",
    "move", "rename", "chmod", "chown",
}


def _response_claims_passive(text: str) -> bool:
    return any(p.search(text) for p in _PASSIVE_PHRASES)


def _tool_calls_are_mutating(tool_names: list[str]) -> list[str]:
    mutating = []
    for name in tool_names:
        name_lower = name.lower()
        if any(action in name_lower for action in _WRITE_ACTIONS):
            mutating.append(name)
    return mutating


class ActionStatementDetector(BaseDetector):
    """
    Flags when the assistant's text response claims a passive/read-only
    action but the tool calls in the same turn are write/mutate/execute.
    """

    name = "action_statement_inconsistency"
    default_severity = Severity.critical

    def detect(self, event: Event) -> list[Detection]:
        if not event.tool_calls:
            return []

        response = event.response_content or ""
        if not response:
            return []

        if not _response_claims_passive(response):
            return []

        tool_names = [tc.function_name for tc in event.tool_calls]
        mutating = _tool_calls_are_mutating(tool_names)

        if not mutating:
            return []

        return [
            self._make_detection(
                event,
                label="Agent claimed passive action but called mutating tools",
                detail={
                    "response_snippet": response[:300],
                    "mutating_tools": mutating,
                },
            )
        ]

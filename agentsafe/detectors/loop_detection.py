from __future__ import annotations

import json

from agentsafe.detectors.base import BaseDetector
from agentsafe.models import Detection, Event, Severity


def _tool_call_signature(msg: dict) -> list[str]:
    """Extract tool call signatures from an assistant message."""
    sigs = []
    for tc in msg.get("tool_calls") or []:
        fn = ""
        args = ""
        try:
            fn = tc.get("function", {}).get("name", "")
            raw_args = tc.get("function", {}).get("arguments", "{}")
            parsed = json.loads(raw_args) if isinstance(raw_args, str) else raw_args
            # Use sorted keys for stable comparison
            args = json.dumps(parsed, sort_keys=True)
        except Exception:
            pass
        if fn:
            sigs.append(f"{fn}:{args}")
    return sigs


class LoopDetector(BaseDetector):
    """
    Detects when an agent is stuck in a tool-call loop — calling the
    same tool with the same (or nearly identical) arguments repeatedly.
    This indicates the agent is not making progress toward its goal.
    """

    name = "loop_detection"
    default_severity = Severity.warning

    def __init__(self, repeat_threshold: int = 3) -> None:
        # How many times a tool call must repeat before flagging
        self.repeat_threshold = repeat_threshold

    def detect(self, event: Event) -> list[Detection]:
        messages = event.messages
        if not messages:
            return []

        # Collect all tool call signatures from conversation history
        all_signatures: list[str] = []
        for msg in messages:
            if msg.get("role") == "assistant":
                all_signatures.extend(_tool_call_signature(msg))

        # Also include current event's tool calls
        for tc in event.tool_calls:
            try:
                args = json.dumps(tc.arguments, sort_keys=True)
            except Exception:
                args = "{}"
            all_signatures.append(f"{tc.function_name}:{args}")

        if not all_signatures:
            return []

        # Count occurrences of each signature
        counts: dict[str, int] = {}
        for sig in all_signatures:
            counts[sig] = counts.get(sig, 0) + 1

        detections = []
        for sig, count in counts.items():
            if count >= self.repeat_threshold:
                tool_name = sig.split(":")[0]
                detections.append(
                    self._make_detection(
                        event,
                        label=f"Agent stuck in loop: '{tool_name}' called {count} times with same arguments",
                        detail={
                            "tool": tool_name,
                            "repeat_count": count,
                            "threshold": self.repeat_threshold,
                            "signature": sig[:200],
                        },
                    )
                )

        return detections

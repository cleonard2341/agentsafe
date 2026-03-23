from __future__ import annotations

import re

from agentsafe.detectors.base import BaseDetector
from agentsafe.models import Detection, Event, Severity

# Tool call patterns that suggest the agent is acquiring capabilities
# or resources beyond what a typical task requires
_DANGEROUS_FUNCTIONS = {
    # Shell / process execution
    "bash", "shell", "exec", "run_command", "execute", "subprocess",
    "terminal", "cmd", "powershell",
    # Persistence / self-modification
    "write_file", "create_file", "modify_file", "patch_file",
    "install_package", "pip_install", "npm_install",
    # Network / exfiltration
    "http_request", "fetch_url", "send_email", "send_message",
    "upload_file", "post_data",
    # Secrets / credentials
    "get_secret", "read_env", "list_credentials", "get_api_key",
}

# Argument value patterns that suggest dangerous paths or commands
_DANGEROUS_ARG_PATTERNS = [
    (re.compile(r"\.\./", re.IGNORECASE), "path traversal"),
    (re.compile(r"~\/\.(ssh|aws|config|gnupg)", re.IGNORECASE), "sensitive directory access"),
    (re.compile(r"\/etc\/(passwd|shadow|hosts|sudoers)", re.IGNORECASE), "system file access"),
    (re.compile(r"rm\s+-rf|rmdir\s+\/", re.IGNORECASE), "destructive command"),
    (re.compile(r"curl\s+.*\|\s*(bash|sh)", re.IGNORECASE), "remote code execution pattern"),
    (re.compile(r"eval\s*\(|exec\s*\(", re.IGNORECASE), "dynamic code execution"),
]


class ScopeCreepDetector(BaseDetector):
    name = "scope_creep"
    default_severity = Severity.warning

    def __init__(self, allowed_tools: set[str] | None = None) -> None:
        # Optionally restrict to a known-good set of tool names
        self.allowed_tools = allowed_tools

    def detect(self, event: Event) -> list[Detection]:
        detections: list[Detection] = []

        for tc in event.tool_calls:
            fn = tc.function_name.lower()

            # Check against allowlist if configured
            if self.allowed_tools is not None and fn not in self.allowed_tools:
                detections.append(
                    self._make_detection(
                        event,
                        label=f"Tool call outside allowed set: {tc.function_name}",
                        severity=Severity.warning,
                        detail={"tool": tc.function_name, "call_id": tc.id},
                    )
                )
                continue

            # Check for known dangerous function names
            if fn in _DANGEROUS_FUNCTIONS:
                detections.append(
                    self._make_detection(
                        event,
                        label=f"Potentially dangerous tool: {tc.function_name}",
                        severity=Severity.warning,
                        detail={"tool": tc.function_name, "call_id": tc.id},
                    )
                )

            # Check argument values for dangerous patterns
            for arg_key, arg_val in tc.arguments.items():
                val_str = str(arg_val)
                for pattern, label in _DANGEROUS_ARG_PATTERNS:
                    if pattern.search(val_str):
                        detections.append(
                            self._make_detection(
                                event,
                                label=f"Dangerous argument pattern: {label}",
                                severity=Severity.critical,
                                detail={
                                    "tool": tc.function_name,
                                    "argument": arg_key,
                                    "pattern": label,
                                    "snippet": val_str[:200],
                                },
                            )
                        )

        return detections

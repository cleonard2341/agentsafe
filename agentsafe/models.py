from __future__ import annotations

import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class Severity(str, Enum):
    info = "info"
    warning = "warning"
    critical = "critical"


class ToolCall(BaseModel):
    id: str
    function_name: str
    arguments: dict[str, Any]


class Event(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    session_id: str
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    messages: list[dict[str, Any]]          # full message list sent to the model
    response_content: str | None = None     # assistant text response (if any)
    tool_calls: list[ToolCall] = Field(default_factory=list)
    model: str
    flagged: bool = False                   # set to True if any detection fires


class Detection(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    event_id: str
    detector_name: str
    severity: Severity
    label: str
    detail: dict[str, Any] = Field(default_factory=dict)
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

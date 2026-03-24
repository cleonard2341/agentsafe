from __future__ import annotations

import uuid
from typing import Any

from agentsafe.models import Event, ToolCall
from agentsafe.pipeline import DetectorPipeline
from agentsafe.storage.repository import EventRepository


class ClaudeMessagesInterceptor:
    """Proxies client.messages and intercepts every create() call."""

    def __init__(
        self,
        real_messages: Any,
        pipeline: DetectorPipeline,
        repo: EventRepository,
        session_id: str,
    ) -> None:
        self._real = real_messages
        self._pipeline = pipeline
        self._repo = repo
        self._session_id = session_id

    def create(self, *, messages: list[dict], model: str, **kwargs) -> Any:
        response = self._real.create(messages=messages, model=model, **kwargs)
        self._process(messages, model, response)
        return response

    async def acreate(self, *, messages: list[dict], model: str, **kwargs) -> Any:
        import asyncio

        response = await self._real.create(messages=messages, model=model, **kwargs)
        await asyncio.get_event_loop().run_in_executor(
            None, self._process, messages, model, response
        )
        return response

    def _process(self, messages: list[dict], model: str, response: Any) -> None:
        content_blocks = getattr(response, "content", []) or []

        # Extract text response (first text block)
        response_content: str | None = None
        tool_calls: list[ToolCall] = []

        for block in content_blocks:
            block_type = getattr(block, "type", None)
            if block_type == "text" and response_content is None:
                response_content = getattr(block, "text", None)
            elif block_type == "tool_use":
                tool_calls.append(
                    ToolCall(
                        id=getattr(block, "id", str(uuid.uuid4())),
                        function_name=getattr(block, "name", "unknown"),
                        arguments=getattr(block, "input", {}) or {},
                    )
                )

        event = Event(
            session_id=self._session_id,
            messages=messages,
            response_content=response_content,
            tool_calls=tool_calls,
            model=model,
        )

        result = self._pipeline.run(event)
        event.flagged = result.flagged

        self._repo.save_event(event)
        self._repo.save_detections(result.detections)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._real, name)


class ClaudeSafeClient:
    """
    Wraps an anthropic.Anthropic() client with AgentSafe monitoring.
    Usage:
        import anthropic, agentsafe
        client = agentsafe.wrap_claude(anthropic.Anthropic())
    """

    def __init__(
        self,
        client: Any,
        pipeline: DetectorPipeline,
        repo: EventRepository,
        session_id: str,
    ) -> None:
        self._client = client
        self._pipeline = pipeline
        self._repo = repo
        self._session_id = session_id
        self._messages_interceptor: ClaudeMessagesInterceptor | None = None

    @property
    def messages(self) -> ClaudeMessagesInterceptor:
        if self._messages_interceptor is None:
            self._messages_interceptor = ClaudeMessagesInterceptor(
                real_messages=self._client.messages,
                pipeline=self._pipeline,
                repo=self._repo,
                session_id=self._session_id,
            )
        return self._messages_interceptor

    def __getattr__(self, name: str) -> Any:
        return getattr(self._client, name)

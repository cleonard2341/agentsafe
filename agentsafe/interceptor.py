from __future__ import annotations

import json
import uuid
from typing import Any

from agentsafe.models import Event, ToolCall
from agentsafe.pipeline import DetectorPipeline
from agentsafe.storage.repository import EventRepository


class CompletionInterceptor:
    """Proxies chat.completions and intercepts every create() call."""

    def __init__(
        self,
        real_completions: Any,
        pipeline: DetectorPipeline,
        repo: EventRepository,
        session_id: str,
    ) -> None:
        self._real = real_completions
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
        choice = response.choices[0] if response.choices else None
        if not choice:
            return

        msg = choice.message

        # Extract text response
        response_content: str | None = getattr(msg, "content", None)

        # Extract tool calls
        tool_calls: list[ToolCall] = []
        raw_tool_calls = getattr(msg, "tool_calls", None) or []
        for tc in raw_tool_calls:
            try:
                args = json.loads(tc.function.arguments)
            except (json.JSONDecodeError, AttributeError):
                args = {}
            tool_calls.append(
                ToolCall(
                    id=tc.id,
                    function_name=tc.function.name,
                    arguments=args,
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


class ChatProxy:
    """Proxies client.chat and returns CompletionInterceptor for .completions."""

    def __init__(self, real_chat: Any, interceptor: CompletionInterceptor) -> None:
        self._real = real_chat
        self._interceptor = interceptor

    @property
    def completions(self) -> CompletionInterceptor:
        return self._interceptor

    def __getattr__(self, name: str) -> Any:
        return getattr(self._real, name)


class SafeClient:
    """
    Wraps any OpenAI-compatible client.
    Usage:
        import agentsafe
        client = agentsafe.wrap(openai.OpenAI())
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
        self._chat_proxy: ChatProxy | None = None

    @property
    def chat(self) -> ChatProxy:
        if self._chat_proxy is None:
            interceptor = CompletionInterceptor(
                real_completions=self._client.chat.completions,
                pipeline=self._pipeline,
                repo=self._repo,
                session_id=self._session_id,
            )
            self._chat_proxy = ChatProxy(self._client.chat, interceptor)
        return self._chat_proxy

    def __getattr__(self, name: str) -> Any:
        return getattr(self._client, name)

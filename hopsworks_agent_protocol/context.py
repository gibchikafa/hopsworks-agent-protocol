"""Handler context — the optional second argument to a chat/stream handler.

Handlers may be written with one or two parameters; the SDK inspects the
signature and passes a :class:`HandlerContext` only when a second parameter is
declared, so ``def chat(request)`` and ``def chat(request, ctx)`` both work.

The context bundles per-turn conveniences (history, memory, logger,
deployment/framework identity, correlation ids) and ``emit_event`` for
surfacing intermediate progress (tool calls, retrieval, code execution) as
``tool_event`` SSE frames while streaming — without the author hand-crafting
SSE.
"""

from __future__ import annotations

import asyncio
import logging
import os
from collections.abc import AsyncIterator
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .memory import ChatMemory, Turn
    from .models import ChatRequest


class HandlerContext:
    def __init__(
        self,
        request: "ChatRequest",
        memory: "ChatMemory | None",
        framework: str,
        response_id: str,
    ):
        self.request = request
        self.conversation_id = request.conversation_id or ""
        self.memory = memory
        self.framework = framework
        self.deployment_id = os.environ.get("DEPLOYMENT_ID")
        self.logger = logging.getLogger("hopsworks_agent")
        # correlation ids: stable for the life of this turn
        self.response_id = response_id
        self.message_id = request.message.id
        # emit plumbing, wired by the route: a queue while streaming, else a
        # buffer surfaced in the response metadata
        self._event_queue: asyncio.Queue[Any] | None = None
        self._loop: asyncio.AbstractEventLoop | None = None
        self._event_buffer: list[dict[str, Any]] = []

    @property
    def history(self) -> "list[Turn]":
        """Prior turns of this conversation from the configured memory store
        (empty list when no memory is configured)."""
        if self.memory is None:
            return []
        return self.memory.get(self.conversation_id)

    async def emit_event(
        self,
        name: str,
        status: str = "running",
        message: str | None = None,
        data: dict[str, Any] | None = None,
        event_id: str | None = None,
    ) -> None:
        """Surface an intermediate progress event (a tool call, a retrieval, a
        code run). While streaming it is sent immediately as a ``tool_event``
        SSE frame; otherwise it is buffered into the response metadata.

        ``event_id`` ties a ``running`` event to its later ``done``/``failed``
        so a client renders one chip that updates rather than several. Pass the
        same id for the start and end of one call (the SDK's auto-emit uses the
        span id); omit it and each event stands alone.
        """
        self._emit_sync(name, status, message, data, event_id)

    def _emit_sync(
        self,
        name: str,
        status: str,
        message: str | None,
        data: dict[str, Any] | None,
        event_id: str | None,
    ) -> None:
        """Non-async emit used by both ``emit_event`` and the auto-emit span
        processor (which runs in sync OTel callbacks, possibly off-thread)."""
        payload: dict[str, Any] = {"name": name, "status": status}
        if event_id is not None:
            payload["id"] = event_id
        if message is not None:
            payload["message"] = message
        if data is not None:
            payload["data"] = data
        self._dispatch(payload)

    async def stream_langchain(
        self, events: AsyncIterator[dict[str, Any]]
    ) -> AsyncIterator[str]:
        """Pipe a LangChain/LangGraph ``astream_events(version="v2")`` stream
        through, yielding assistant text deltas and turning tool calls into
        ``tool_event`` chips automatically — no manual ``emit_event`` and no
        dependency on tracing:

            async for delta in ctx.stream_langchain(agent.astream_events(...)):
                yield delta

        ``on_tool_start`` / ``on_tool_end`` / ``on_tool_error`` become
        running / done / failed events keyed by the tool's ``run_id`` so start
        and end collapse into one chip.
        """
        async for event in events:
            kind = event.get("event")
            if kind == "on_chat_model_stream":
                chunk = (event.get("data") or {}).get("chunk")
                content = getattr(chunk, "content", None)
                if isinstance(content, str):
                    if content:
                        yield content
                elif isinstance(content, list):
                    for block in content:
                        if isinstance(block, dict) and block.get("type") == "text":
                            text = block.get("text", "")
                            if text:
                                yield text
            elif kind == "on_tool_start":
                tool_input = (event.get("data") or {}).get("input")
                await self.emit_event(
                    event.get("name", "tool"),
                    status="running",
                    message=str(tool_input) if tool_input else None,
                    event_id=event.get("run_id"),
                )
            elif kind == "on_tool_end":
                await self.emit_event(
                    event.get("name", "tool"),
                    status="done",
                    event_id=event.get("run_id"),
                )
            elif kind == "on_tool_error":
                err = (event.get("data") or {}).get("error")
                await self.emit_event(
                    event.get("name", "tool"),
                    status="failed",
                    message=str(err) if err else None,
                    event_id=event.get("run_id"),
                )

    def _dispatch(self, payload: dict[str, Any]) -> None:
        queue = self._event_queue
        loop = self._loop
        if queue is None or loop is None:
            self._event_buffer.append(payload)
            return
        try:
            running = asyncio.get_running_loop()
        except RuntimeError:
            running = None
        if running is loop:
            # already on the request's event loop (handler called emit_event):
            # enqueue directly so ordering vs. yielded deltas is preserved
            queue.put_nowait(("tool_event", payload))
        else:
            # off-thread (a framework tool running in a threadpool): hop back
            # onto the loop safely
            loop.call_soon_threadsafe(queue.put_nowait, ("tool_event", payload))

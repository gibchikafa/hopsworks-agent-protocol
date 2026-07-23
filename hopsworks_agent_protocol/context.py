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
    ) -> None:
        """Surface an intermediate progress event (a tool call, a retrieval, a
        code run). While streaming it is sent immediately as a ``tool_event``
        SSE frame; otherwise it is buffered into the response metadata.

        No-op unless the agent enabled ``tool_events`` on the app.
        """
        payload: dict[str, Any] = {"name": name, "status": status}
        if message is not None:
            payload["message"] = message
        if data is not None:
            payload["data"] = data
        if self._event_queue is not None:
            await self._event_queue.put(("tool_event", payload))
        else:
            self._event_buffer.append(payload)

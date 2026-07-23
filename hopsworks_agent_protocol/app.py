"""AgentApp: a FastAPI subclass that serves the Hopsworks Agent Protocol.

Registers automatically:

- ``GET  /.well-known/hopsworks-agent.json``  — protocol manifest
- ``POST /v1/chat``                           — non-streaming chat
- ``POST /v1/chat/stream``                    — SSE streaming (when a stream
  handler is registered; otherwise the capability is off)
- ``GET  /health``                            — liveness probe

Being a FastAPI subclass, it works anywhere a FastAPI app does (uvicorn,
KServe/Knative on Hopsworks) and extra routes can be added the normal way.
CORS is enabled by default so the Hopsworks UI chat panel can call the agent
directly through the Istio ingress.
"""

from __future__ import annotations

import inspect
import json
from collections.abc import AsyncIterator, Awaitable, Callable
from typing import Any

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse

from .models import (
    PROTOCOL,
    PROTOCOL_VERSION,
    AgentError,
    ChatRequest,
    ChatResponse,
    new_conversation_id,
)

ChatHandler = Callable[[ChatRequest], Awaitable[ChatResponse | str] | ChatResponse | str]
StreamHandler = Callable[[ChatRequest], AsyncIterator["str | ChatResponse"]]


def _sse(event: str, data: Any) -> str:
    payload = data if isinstance(data, str) else json.dumps(data)
    return f"event: {event}\ndata: {payload}\n\n"


class AgentApp(FastAPI):
    def __init__(
        self,
        name: str = "Hopsworks agent",
        description: str = "",
        version: str = "1.0.0",
        welcome_message: str | None = None,
        suggested_prompts: list[str] | None = None,
        placeholder: str | None = None,
        allow_cors: bool = True,
        **fastapi_kwargs: Any,
    ):
        super().__init__(title=name, description=description, **fastapi_kwargs)
        self._agent_name = name
        self._agent_description = description
        self._agent_version = version
        self._welcome_message = welcome_message
        self._suggested_prompts = suggested_prompts or []
        self._placeholder = placeholder
        self._chat_handler: ChatHandler | None = None
        self._stream_handler: StreamHandler | None = None

        if allow_cors:
            self.add_middleware(
                CORSMiddleware,
                allow_origins=["*"],
                allow_methods=["*"],
                allow_headers=["*"],
            )

        self._register_routes()

    # ── handler decorators ────────────────────────────────────────────────

    def chat(self, handler: ChatHandler) -> ChatHandler:
        """Register the chat handler.

        The handler receives a :class:`ChatRequest` whose ``conversation_id``
        is always set (generated on the first turn) and returns a
        :class:`ChatResponse` (see :class:`AgentResponse`) or a plain string.
        """
        self._chat_handler = handler
        return handler

    def stream(self, handler: StreamHandler) -> StreamHandler:
        """Register a streaming handler: an async generator yielding text
        deltas (``str``). Optionally yield a final :class:`ChatResponse` to
        attach citations/usage/metadata to the completed message.

        Registering a stream handler turns on the ``streaming`` capability;
        if no plain chat handler is registered, ``/v1/chat`` is served by
        collecting the stream.
        """
        self._stream_handler = handler
        return handler

    # ── internals ─────────────────────────────────────────────────────────

    def _manifest(self) -> dict[str, Any]:
        streaming = self._stream_handler is not None
        endpoints: dict[str, str] = {"chat": "/v1/chat"}
        if streaming:
            endpoints["stream"] = "/v1/chat/stream"
        return {
            "protocol": PROTOCOL,
            "protocol_version": PROTOCOL_VERSION,
            "agent": {
                "name": self._agent_name,
                "description": self._agent_description,
                "version": self._agent_version,
            },
            "endpoints": endpoints,
            "capabilities": {
                "streaming": streaming,
                "conversation_history": True,
                "attachments": False,
                "citations": False,
                "tool_events": False,
            },
            "ui": {
                "welcome_message": self._welcome_message,
                "suggested_prompts": self._suggested_prompts,
                "placeholder": self._placeholder,
                "allow_markdown": True,
            },
        }

    @staticmethod
    def _prepare(request: ChatRequest) -> ChatRequest:
        # handlers can always rely on conversation_id being present
        if not request.conversation_id:
            request.conversation_id = new_conversation_id()
        return request

    @staticmethod
    def _finalize(result: ChatResponse | str, request: ChatRequest) -> ChatResponse:
        from .models import AgentResponse

        response = (
            AgentResponse.text(result) if isinstance(result, str) else result
        )
        if not response.conversation_id:
            response.conversation_id = request.conversation_id or ""
        return response

    def _merge_stream_result(
        self,
        chunks: list[str],
        final: ChatResponse | None,
        request: ChatRequest,
    ) -> ChatResponse:
        """A final ChatResponse yielded by a stream handler may carry only
        citations/usage; fill its text from the streamed chunks when empty."""
        if final is None:
            return self._finalize("".join(chunks), request)
        if contentful(final):
            return self._finalize(final, request)
        response = self._finalize("".join(chunks), request)
        response.citations = final.citations
        response.usage = final.usage
        response.metadata = final.metadata
        return response

    async def _run_chat(self, request: ChatRequest) -> ChatResponse:
        if self._chat_handler is not None:
            result = self._chat_handler(request)
            if inspect.isawaitable(result):
                result = await result
            return self._finalize(result, request)
        # collect the stream into a single response
        assert self._stream_handler is not None
        chunks: list[str] = []
        final: ChatResponse | None = None
        async for item in self._stream_handler(request):
            if isinstance(item, ChatResponse):
                final = item
            else:
                chunks.append(item)
        return self._merge_stream_result(chunks, final, request)

    def _register_routes(self) -> None:
        @self.get("/.well-known/hopsworks-agent.json")
        async def manifest() -> dict[str, Any]:
            return self._manifest()

        @self.get("/health")
        async def health() -> dict[str, str]:
            return {"status": "ok"}

        @self.post("/v1/chat")
        async def chat_route(request: ChatRequest) -> JSONResponse:
            if self._chat_handler is None and self._stream_handler is None:
                return _error_response(
                    AgentError("No chat handler registered.", "not_implemented", 501)
                )
            prepared = self._prepare(request)
            try:
                response = await self._run_chat(prepared)
            except AgentError as err:
                return _error_response(err)
            return JSONResponse(response.model_dump())

        @self.post("/v1/chat/stream")
        async def stream_route(request: ChatRequest, raw: Request) -> StreamingResponse:
            prepared = self._prepare(request)

            async def events() -> AsyncIterator[str]:
                if self._stream_handler is None:
                    # graceful degradation: run the chat handler and emit the
                    # result as a single completed event
                    try:
                        response = await self._run_chat(prepared)
                    except AgentError as err:
                        yield _sse("error", err.detail())
                        return
                    yield _sse("message.completed", response.model_dump())
                    return
                chunks: list[str] = []
                final: ChatResponse | None = None
                try:
                    async for item in self._stream_handler(prepared):
                        if await raw.is_disconnected():
                            return
                        if isinstance(item, ChatResponse):
                            final = item
                        else:
                            chunks.append(item)
                            yield _sse("message.delta", {"delta": {"text": item}})
                except AgentError as err:
                    yield _sse("error", err.detail())
                    return
                except Exception as err:  # noqa: BLE001 — surfaced to the client
                    yield _sse(
                        "error",
                        {"code": "agent_error", "message": str(err), "retryable": False},
                    )
                    return
                response = self._merge_stream_result(chunks, final, prepared)
                yield _sse("message.completed", response.model_dump())

            return StreamingResponse(events(), media_type="text/event-stream")


def contentful(response: ChatResponse) -> bool:
    return any(part.text for part in response.message.content)


def _error_response(err: AgentError) -> JSONResponse:
    return JSONResponse({"detail": err.detail()}, status_code=err.status_code)

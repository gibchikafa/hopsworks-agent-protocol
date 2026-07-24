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

import asyncio
import inspect
import json
from collections.abc import AsyncIterator, Awaitable, Callable
from typing import Any

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse

from .context import HandlerContext
from .models import (
    PROTOCOL,
    PROTOCOL_VERSION,
    AgentError,
    ChatRequest,
    ChatResponse,
    new_conversation_id,
    new_response_id,
)
from .memory import ChatMemory
from .tracing import resolve_framework, setup_tracing

ChatHandler = Callable[..., Awaitable[ChatResponse | str] | ChatResponse | str]
StreamHandler = Callable[..., AsyncIterator["str | ChatResponse"]]


def _sse(event: str, data: Any) -> str:
    payload = data if isinstance(data, str) else json.dumps(data)
    return f"event: {event}\ndata: {payload}\n\n"


def _wants_context(handler: Callable[..., Any]) -> bool:
    """True when the handler declares a second positional parameter (the
    HandlerContext), so ``def chat(request)`` and ``def chat(request, ctx)``
    both work."""
    try:
        params = [
            p
            for p in inspect.signature(handler).parameters.values()
            if p.kind
            in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)
        ]
        return len(params) >= 2
    except (ValueError, TypeError):
        return False


class AgentApp(FastAPI):
    def __init__(
        self,
        name: str = "Hopsworks agent",
        description: str = "",
        version: str = "1.0.0",
        welcome_message: str | None = None,
        suggested_prompts: list[str] | None = None,
        placeholder: str | None = None,
        input_modalities: list[str] | None = None,
        output_modalities: list[str] | None = None,
        framework: str | None = None,
        tracing: bool | None = None,
        memory: ChatMemory | None = None,
        tool_events: bool = False,
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
        self._input_modalities = input_modalities or ["text"]
        self._output_modalities = output_modalities or ["text"]
        # advertise tool_event SSE frames (emitted via ctx.emit_event); off by
        # default so existing manifests are unchanged
        self._tool_events = tool_events
        self._auto_tool_events = False

        # framework: explicit arg > AGENT_FRAMEWORK env (platform-injected) >
        # 'custom'. Drives which OpenInference instrumentor tracing activates.
        self.framework = resolve_framework(framework)
        # tracing: None auto-detects from the platform-injected OTLP endpoint
        # env var (set iff tracing is enabled on the deployment)
        self.tracer_provider = setup_tracing(self.framework, enabled=tracing)

        # optional conversation memory: turns are recorded automatically after
        # each successful exchange; handlers read history with
        # self.memory.get(request.conversation_id). Skip it if your framework
        # persists state itself (e.g. a LangGraph checkpointer).
        self.memory = memory
        self._chat_handler: ChatHandler | None = None
        self._stream_handler: StreamHandler | None = None

        # when tool events are on and tracing is active, auto-emit tool events
        # from the framework's spans so tool calls show up with zero agent code
        if self._tool_events and self.tracer_provider is not None:
            from .autoevents import install_auto_tool_events

            self._auto_tool_events = install_auto_tool_events(self.tracer_provider)

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
        if self.memory is not None:
            # server-managed history is available: clients can list/clear it
            endpoints["conversations"] = "/v1/conversations"
        return {
            "protocol": PROTOCOL,
            "protocol_version": PROTOCOL_VERSION,
            "agent": {
                "name": self._agent_name,
                "description": self._agent_description,
                "version": self._agent_version,
                "framework": self.framework,
            },
            "endpoints": endpoints,
            "capabilities": {
                "streaming": streaming,
                "conversation_history": True,
                # server-side history is inspectable/clearable via the endpoints
                "conversation_management": self.memory is not None,
                "attachments": any(m != "text" for m in self._input_modalities),
                "input_modalities": self._input_modalities,
                "output_modalities": self._output_modalities,
                "citations": False,
                "tool_events": self._tool_events,
            },
            "ui": {
                "welcome_message": self._welcome_message,
                "suggested_prompts": self._suggested_prompts,
                "placeholder": self._placeholder,
                "allow_markdown": True,
            },
        }

    def _record_turn(self, request: ChatRequest, response: ChatResponse) -> None:
        """Auto-record the exchange when a memory store is configured."""
        if self.memory is None or response.status != "completed":
            return
        conversation_id = response.conversation_id or request.conversation_id
        if not conversation_id:
            return
        try:
            if request.text:
                self.memory.append(conversation_id, "user", request.text)
            answer = "".join(
                part.text
                for part in response.message.content
                if part.type == "text"
            )
            if answer:
                self.memory.append(conversation_id, "assistant", answer)
        except Exception:  # noqa: BLE001 — memory failures must not break chat
            import logging

            logging.getLogger(__name__).exception(
                "Failed to record conversation turn"
            )

    def _prepare(self, request: ChatRequest) -> HandlerContext:
        # handlers can always rely on conversation_id being present
        if not request.conversation_id:
            request.conversation_id = new_conversation_id()
        if request.message.id is None:
            from .models import new_message_id

            request.message.id = new_message_id()
        # the response id is generated up front so ctx.response_id matches the
        # id on the response the client ultimately receives (correlation)
        return HandlerContext(
            request=request,
            memory=self.memory,
            framework=self.framework,
            response_id=new_response_id(),
        )

    def _finalize(
        self, result: ChatResponse | str, ctx: HandlerContext
    ) -> ChatResponse:
        from .models import AgentResponse

        response = (
            AgentResponse.text(result) if isinstance(result, str) else result
        )
        if not response.conversation_id:
            response.conversation_id = ctx.conversation_id
        response.id = ctx.response_id
        # non-streaming tool events (buffered on the context) ride on metadata
        if ctx._event_buffer:
            response.metadata.setdefault("tool_events", ctx._event_buffer)
        _annotate_span(ctx, response)
        return response

    def _merge_stream_result(
        self,
        chunks: list[str],
        final: ChatResponse | None,
        ctx: HandlerContext,
    ) -> ChatResponse:
        """A final ChatResponse yielded by a stream handler may carry only
        citations/usage; fill its text from the streamed chunks when empty."""
        if final is None:
            return self._finalize("".join(chunks), ctx)
        if contentful(final):
            streamed = "".join(chunks)
            has_text = any(
                part.type == "text" and part.text for part in final.message.content
            )
            if streamed and not has_text:
                # final carried only non-text parts: keep the streamed text too
                from .models import TextContent

                final.message.content.insert(0, TextContent(text=streamed))
            return self._finalize(final, ctx)
        response = self._finalize("".join(chunks), ctx)
        response.citations = final.citations
        response.usage = final.usage
        response.metadata = final.metadata
        return response

    def _invoke_stream(
        self, ctx: HandlerContext
    ) -> AsyncIterator["str | ChatResponse"]:
        assert self._stream_handler is not None
        if _wants_context(self._stream_handler):
            return self._stream_handler(ctx.request, ctx)
        return self._stream_handler(ctx.request)

    async def _run_chat(self, ctx: HandlerContext) -> ChatResponse:
        from .autoevents import current_context

        token = current_context.set(ctx)
        try:
            if self._chat_handler is not None:
                if _wants_context(self._chat_handler):
                    result = self._chat_handler(ctx.request, ctx)
                else:
                    result = self._chat_handler(ctx.request)
                if inspect.isawaitable(result):
                    result = await result
                return self._finalize(result, ctx)
            # collect the stream into a single response
            chunks: list[str] = []
            final: ChatResponse | None = None
            async for item in self._invoke_stream(ctx):
                if isinstance(item, ChatResponse):
                    final = item
                else:
                    chunks.append(item)
            return self._merge_stream_result(chunks, final, ctx)
        finally:
            current_context.reset(token)

    def _register_routes(self) -> None:
        @self.get("/.well-known/hopsworks-agent.json")
        async def manifest() -> dict[str, Any]:
            return self._manifest()

        @self.get("/health")
        async def health() -> dict[str, str]:
            # lightweight liveness: the process is up
            return {"status": "ok"}

        @self.get("/ready")
        async def ready() -> JSONResponse:
            # operational readiness: everything needed to actually serve chat
            checks = {
                "handler": self._chat_handler is not None
                or self._stream_handler is not None,
                "memory": self.memory is None or self.memory.healthcheck(),
            }
            ok = all(checks.values())
            return JSONResponse(
                {"status": "ready" if ok else "not_ready", "checks": checks},
                status_code=200 if ok else 503,
            )

        @self.post("/v1/chat")
        async def chat_route(request: ChatRequest) -> JSONResponse:
            if self._chat_handler is None and self._stream_handler is None:
                return _error_response(
                    AgentError("No chat handler registered.", "not_implemented", 501)
                )
            ctx = self._prepare(request)
            try:
                response = await self._run_chat(ctx)
            except AgentError as err:
                return _error_response(err)
            self._record_turn(ctx.request, response)
            return JSONResponse(response.model_dump())

        @self.post("/v1/chat/stream")
        async def stream_route(request: ChatRequest, raw: Request) -> StreamingResponse:
            ctx = self._prepare(request)
            return StreamingResponse(
                self._stream_events(ctx, raw), media_type="text/event-stream"
            )

        if self.memory is not None:
            self._register_conversation_routes()

    def _register_conversation_routes(self) -> None:
        @self.get("/v1/conversations/{conversation_id}/messages")
        async def list_messages(conversation_id: str) -> dict[str, Any]:
            # inspect the SDK-managed history the agent actually sees
            assert self.memory is not None
            return {
                "conversation_id": conversation_id,
                "messages": self.memory.get(conversation_id),
            }

        @self.delete("/v1/conversations/{conversation_id}")
        async def clear_conversation(conversation_id: str) -> JSONResponse:
            # "new session" on the client can drop server-side memory too
            assert self.memory is not None
            self.memory.clear(conversation_id)
            return JSONResponse(status_code=204, content=None)

    async def _stream_events(
        self, ctx: HandlerContext, raw: Request
    ) -> AsyncIterator[str]:
        # graceful degradation: no stream handler -> run chat, emit one event
        if self._stream_handler is None:
            try:
                response = await self._run_chat(ctx)
            except AgentError as err:
                yield _sse("error", err.detail())
                return
            self._record_turn(ctx.request, response)
            yield _sse("message.completed", response.model_dump())
            return

        # Run the handler as a task that feeds a queue, so ctx.emit_event
        # (tool_event frames) interleaves with the handler's own yields.
        from .autoevents import current_context

        queue: asyncio.Queue[tuple[str, Any]] = asyncio.Queue()
        ctx._event_queue = queue
        ctx._loop = asyncio.get_running_loop()

        async def pump() -> None:
            token = current_context.set(ctx)
            try:
                async for item in self._invoke_stream(ctx):
                    await queue.put(("item", item))
            except AgentError as err:
                await queue.put(("error", err.detail()))
            except Exception as err:  # noqa: BLE001 — surfaced to the client
                await queue.put(
                    (
                        "error",
                        {"code": "agent_error", "message": str(err), "retryable": False},
                    )
                )
            finally:
                current_context.reset(token)
                await queue.put(("done", None))

        task = asyncio.create_task(pump())
        chunks: list[str] = []
        final: ChatResponse | None = None
        try:
            while True:
                kind, payload = await queue.get()
                if kind == "done":
                    break
                if await raw.is_disconnected():
                    task.cancel()
                    return
                if kind == "error":
                    yield _sse("error", payload)
                    return
                if kind == "tool_event":
                    yield _sse("tool_event", payload)
                elif isinstance(payload, ChatResponse):
                    final = payload
                else:
                    chunks.append(payload)
                    yield _sse("message.delta", {"delta": {"text": payload}})
        finally:
            if not task.done():
                task.cancel()

        response = self._merge_stream_result(chunks, final, ctx)
        self._record_turn(ctx.request, response)
        yield _sse("message.completed", response.model_dump())


def contentful(response: ChatResponse) -> bool:
    # any non-text part counts as content; text parts count when non-empty
    return any(
        part.type != "text" or part.text for part in response.message.content
    )


def _error_response(err: AgentError) -> JSONResponse:
    return JSONResponse({"detail": err.detail()}, status_code=err.status_code)


def _annotate_span(ctx: HandlerContext, response: ChatResponse) -> None:
    """Best-effort correlation: stamp conversation/message/response ids on the
    active OTel span and surface the trace id in the response metadata, so
    downstream evaluation can join chat turns to traces. No-op when tracing is
    not active."""
    try:
        from opentelemetry import trace
    except ImportError:
        return
    span = trace.get_current_span()
    ctxt = span.get_span_context()
    if not getattr(ctxt, "is_valid", False):
        return
    try:
        span.set_attribute("hopsworks.conversation_id", ctx.conversation_id)
        span.set_attribute("hopsworks.response_id", response.id)
        if ctx.message_id:
            span.set_attribute("hopsworks.message_id", ctx.message_id)
        if ctx.deployment_id:
            span.set_attribute("hopsworks.deployment_id", ctx.deployment_id)
        span.set_attribute("hopsworks.framework", ctx.framework)
        response.metadata.setdefault("trace_id", format(ctxt.trace_id, "032x"))
    except Exception:  # noqa: BLE001 — correlation is best-effort
        pass

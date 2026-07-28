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
from .memory import TURN_ABANDONED, TURN_CLOSED, ChatMemory
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
        graph: Any = None,
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

        # optional agent structure graph (e.g. agent.get_graph() for a compiled
        # LangGraph) served at /v1/graph for the chat panel's Graph tab
        from .graph import to_graph_spec

        self._graph_spec = to_graph_spec(graph)

        # framework: explicit arg > AGENT_FRAMEWORK env (platform-injected) >
        # 'custom'. Drives which OpenInference instrumentor tracing activates.
        self.framework = resolve_framework(framework)
        # tracing: None auto-detects from the platform-injected OTLP endpoint
        # env var (set iff tracing is enabled on the deployment)
        self.tracer_provider = setup_tracing(self.framework, enabled=tracing)

        # optional conversation memory: the user message is recorded when the
        # turn opens and the reply when it closes, so a failed turn is marked
        # abandoned rather than leaving a question with no answer. Handlers read
        # history with self.memory.get(request.conversation_id). Skip it if your
        # framework persists state itself (e.g. a LangGraph checkpointer).
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

    def memory_tools(self, framework: str | None = None, include=None) -> list[Any]:
        """Framework-native ``remember`` / ``recall`` / ``forget`` / ``search`` tools.

        Add them to your agent's tool list yourself — the SDK cannot reach into
        an arbitrary framework's tools, and appending to them behind your back
        would be worse than asking::

            agent = create_react_agent(llm, [*tools, *app.memory_tools()])

        Defaults to the app's detected framework. ``include`` registers a
        subset by name, e.g. ``include=("recall", "search")`` for an agent that
        must not let the model write subject-scoped state — see
        :func:`~hopsworks_agent_protocol.tools.memory_tools`.
        """
        from .tools import memory_tools as _memory_tools

        return _memory_tools(framework or self.framework, include=include)

    # ── internals ─────────────────────────────────────────────────────────

    def _memory_capabilities(self) -> dict[str, bool]:
        """Which memory tiers this agent actually has, so the panel can light up
        the right inspector views instead of guessing."""
        if self.memory is None:
            return {
                "conversation_history": False,
                "summary": False,
                "state": False,
                "search": False,
            }
        return {
            "conversation_history": True,
            "summary": getattr(self.memory, "_summarize", None) is not None,
            "state": bool(getattr(self.memory, "_long_term", False)),
            "search": getattr(self.memory, "_vector_store", None) is not None,
        }

    def _manifest(self) -> dict[str, Any]:
        streaming = self._stream_handler is not None
        endpoints: dict[str, str] = {"chat": "/v1/chat"}
        if streaming:
            endpoints["stream"] = "/v1/chat/stream"
        if self.memory is not None:
            # server-managed history is available: clients can list/clear it
            endpoints["conversations"] = "/v1/conversations"
            if getattr(self.memory, "_long_term", False):
                # durable per-subject memory is inspectable and deletable
                endpoints["subjects"] = "/v1/subjects"
        if self._graph_spec is not None:
            endpoints["graph"] = "/v1/graph"
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
                # protocol-level statement that history is server-side. The
                # per-tier detail is under "memory" below; this was previously
                # hardcoded True even with no store configured, which was wrong.
                "conversation_history": self.memory is not None,
                "memory": self._memory_capabilities(),
                # server-side history is inspectable/clearable via the endpoints
                "conversation_management": self.memory is not None,
                "attachments": any(m != "text" for m in self._input_modalities),
                "input_modalities": self._input_modalities,
                "output_modalities": self._output_modalities,
                "citations": False,
                "tool_events": self._tool_events,
                # a structure graph is available to visualize the agent
                "graph": self._graph_spec is not None,
            },
            "ui": {
                "welcome_message": self._welcome_message,
                "suggested_prompts": self._suggested_prompts,
                "placeholder": self._placeholder,
                "allow_markdown": True,
            },
        }

    def _prepare(self, request: ChatRequest) -> HandlerContext:
        # handlers can always rely on conversation_id being present
        if not request.conversation_id:
            request.conversation_id = new_conversation_id()
        if request.message.id is None:
            from .models import new_message_id

            request.message.id = new_message_id()
        from .memory import new_turn_id

        # the response id is generated up front so ctx.response_id matches the
        # id on the response the client ultimately receives (correlation)
        return HandlerContext(
            request=request,
            memory=self.memory,
            framework=self.framework,
            response_id=new_response_id(),
            turn_id=new_turn_id(),
            subject=getattr(request, "subject", None),
        )

    async def _open_turn(self, ctx: HandlerContext) -> None:
        """Record the user message and open the turn, before the handler runs.

        Writing it up front is what lets a handler read back the message it is
        answering, and (from Phase 2) lets anything the agent remembers point at
        the turn that caused it. The row is invisible to ``get()`` until the
        turn closes, so an in-flight turn never shows up as a question with no
        answer.
        """
        if self.memory is None or not ctx.request.text:
            return
        try:
            await asyncio.to_thread(
                self.memory.begin_turn,
                ctx.conversation_id,
                ctx.turn_id,
                "user",
                ctx.request.text,
                message_id=ctx.message_id,
                subject=ctx.subject,
            )
            ctx._turn_open = True
        except Exception:  # noqa: BLE001 — memory must never break a turn
            import logging

            logging.getLogger(__name__).exception("Failed to open memory turn")

    async def _finalize_turn(
        self, ctx: HandlerContext, response: ChatResponse | None
    ) -> None:
        """Close the turn. Must run on every exit path, including failures.

        The pre-inserted user message makes this obligatory rather than
        housekeeping: skip it and the store keeps a question whose answer never
        arrived, which then reappears as history on the next turn.
        """
        if self.memory is None or not ctx._turn_open:
            return
        ctx._turn_open = False
        try:
            completed = response is not None and response.status == "completed"
            if completed:
                await asyncio.to_thread(self._write_turn_items, ctx, response)
            await asyncio.to_thread(
                self.memory.end_turn,
                ctx.conversation_id,
                ctx.turn_id,
                status=TURN_CLOSED if completed else TURN_ABANDONED,
            )
            # After the last token: summarizing and embedding cost request
            # duration, never time-to-answer. Both are awaited rather than
            # fired off, because a scale-to-zero pod would take an un-awaited
            # task with it.
            if completed:
                await self.memory.ingest_turn(ctx.conversation_id, ctx.turn_id)
                await self.memory.maybe_summarize(ctx.conversation_id)
            await asyncio.to_thread(self.memory.maybe_reap)
        except Exception:  # noqa: BLE001 — memory failures must not break chat
            import logging

            logging.getLogger(__name__).exception("Failed to finalize memory turn")

    def _write_turn_items(
        self, ctx: HandlerContext, response: ChatResponse
    ) -> None:
        """Assistant reply + any tool events, in emission order."""
        import json

        for event in ctx._recorded_events:
            self.memory.record_item(
                ctx.conversation_id,
                ctx.turn_id,
                "tool",
                json.dumps(event),
                memory_type="event",
                message_id=ctx.message_id,
                subject=ctx.subject,
            )
        answer = "".join(
            part.text for part in response.message.content if part.type == "text"
        )
        if answer:
            self.memory.record_item(
                ctx.conversation_id,
                ctx.turn_id,
                "assistant",
                answer,
                message_id=ctx.message_id,
                subject=ctx.subject,
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
        # Which subject this turn's memory was actually keyed on. Only reported
        # when the agent identified the user itself (rebind_subject): the client
        # already knows the subject it asserted, but it cannot know one the
        # agent derived mid-turn — and without this a memory inspector reads the
        # subject it sent, finds an empty bucket, and reports "nothing stored"
        # about a user the agent has been happily remembering.
        if ctx.subject_source == "app":
            response.metadata.setdefault("subject", ctx.subject)
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
        # Merge, do not replace: _finalize has already put the turn's own keys
        # here (tool_events, trace_id, the resolved subject). Assigning
        # final.metadata over the top dropped every one of them whenever a
        # stream handler yielded a final response alongside streamed text —
        # which is the shape a handler uses to stream tokens and still return
        # structured output. The handler's own keys still win on a collision.
        response.metadata = {**response.metadata, **final.metadata}
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
            await self._open_turn(ctx)
            response: ChatResponse | None = None
            try:
                response = await self._run_chat(ctx)
            except AgentError as err:
                return _error_response(err)
            finally:
                # in a finally because an AgentError must not leave the turn
                # open: its user message is already recorded
                await self._finalize_turn(ctx, response)
            return JSONResponse(response.model_dump())

        @self.post("/v1/chat/stream")
        async def stream_route(request: ChatRequest, raw: Request) -> StreamingResponse:
            ctx = self._prepare(request)
            await self._open_turn(ctx)
            return StreamingResponse(
                self._stream_events(ctx, raw), media_type="text/event-stream"
            )

        if self.memory is not None:
            self._register_conversation_routes()

        if self._graph_spec is not None:

            @self.get("/v1/graph")
            async def graph() -> dict[str, Any]:
                return self._graph_spec

    def _register_conversation_routes(self) -> None:
        @self.get("/v1/conversations")
        async def list_conversations(
            subject: str = "", limit: int = 50
        ) -> dict[str, Any]:
            # The manifest has always advertised this path; without it a client
            # can only show conversations it happens to remember locally, so
            # clearing a browser's storage loses a transcript the server still
            # holds in full.
            if self.memory is None:
                return {"conversations": []}
            conversations = await asyncio.to_thread(
                self.memory.list_conversations,
                subject=subject or None,
                limit=max(1, min(limit, 500)),
            )
            return {"conversations": conversations}

        @self.get("/v1/conversations/{conversation_id}/messages")
        async def list_messages(
            conversation_id: str, include: str = ""
        ) -> dict[str, Any]:
            # The human-facing record, which is NOT the same as what the model
            # reads: once turns are folded they leave ctx.history but stay here.
            # `summary` carries the folded part and `summarized_through` marks
            # where the two diverge, so a UI can render "older context,
            # compacted" instead of appearing to have lost messages.
            assert self.memory is not None
            include_events = "events" in {
                part.strip() for part in include.split(",")
            }
            messages = await asyncio.to_thread(
                self.memory.transcript,
                conversation_id,
                include_events=include_events,
            )
            summary = await asyncio.to_thread(
                self.memory.get_summary, conversation_id
            )
            cutoff = await asyncio.to_thread(
                self.memory.summarized_through, conversation_id
            )
            return {
                "conversation_id": conversation_id,
                "messages": messages,
                "summary": summary,
                "summarized_through": cutoff,
            }

        @self.delete("/v1/conversations/{conversation_id}")
        async def clear_conversation(conversation_id: str) -> JSONResponse:
            # "new session" on the client can drop server-side memory too.
            # Session-scoped state goes with it; user- and app-scoped state does
            # NOT — those are subject/agent-scoped, and starting a new chat must
            # not erase what the agent knows about the person. Forgetting that
            # is a separate, explicit action (the /subjects routes below).
            assert self.memory is not None
            await asyncio.to_thread(self.memory.clear, conversation_id)
            await asyncio.to_thread(
                self.memory.delete_state, "session", conversation_id
            )
            # the vector store holds a copy of the content, so deleting from
            # SQL alone would leave it searchable
            await asyncio.to_thread(
                self.memory.purge_vectors, conversation_id=conversation_id
            )
            return JSONResponse(status_code=204, content=None)

        @self.get("/v1/subjects/{subject}/state")
        async def list_subject_state(subject: str) -> dict[str, Any]:
            # The audit half of durable memory. Model-written state is
            # attacker-influenceable — a user can talk the agent into
            # remembering something false about them, and it then loads into
            # every later conversation. Caps and TTLs bound that; being able to
            # see and delete it is what actually fixes it, so this ships with
            # the feature rather than after it.
            assert self.memory is not None
            values = await asyncio.to_thread(
                self.memory.list_state, "user", subject
            )
            return {"subject": subject, "state": values}

        @self.delete("/v1/subjects/{subject}/state")
        async def clear_subject_state(subject: str, key: str | None = None) -> JSONResponse:
            assert self.memory is not None
            removed = await asyncio.to_thread(
                self.memory.delete_state, "user", subject, key
            )
            vectors = 0
            if key is None:
                # forgetting a whole subject includes their embedded messages
                vectors = await asyncio.to_thread(
                    self.memory.purge_vectors, subject=subject
                )
            return JSONResponse({"removed": removed, "vectors_removed": vectors})

    async def _stream_events(
        self, ctx: HandlerContext, raw: Request
    ) -> AsyncIterator[str]:
        # graceful degradation: no stream handler -> run chat, emit one event
        # Every exit below — normal completion, AgentError, handler crash,
        # client disconnect — has to end the turn, because its user message is
        # already in the store. Hence one finally around the whole generator.
        completed: ChatResponse | None = None
        try:
            if self._stream_handler is None:
                try:
                    response = await self._run_chat(ctx)
                except AgentError as err:
                    yield _sse("error", err.detail())
                    return
                completed = response
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
                            {
                                "code": "agent_error",
                                "message": str(err),
                                "retryable": False,
                            },
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
            completed = response
            yield _sse("message.completed", response.model_dump())
        finally:
            # Best effort: if the whole task tree is being cancelled this may
            # not get to run, which is what the store's stale-turn reaper is
            # there to catch.
            await self._finalize_turn(ctx, completed)


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

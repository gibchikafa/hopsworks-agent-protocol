# hopsworks-agent-protocol

Server helpers for the **Hopsworks Agent Protocol**: make any Python agent chat-ready in the Hopsworks UI with a few lines. `AgentApp` is a FastAPI subclass that automatically exposes:

| Route | Purpose |
|---|---|
| `GET /.well-known/hopsworks-agent.json` | Protocol manifest — the Hopsworks chat panel auto-detects the agent from it |
| `POST /v1/chat` | Non-streaming chat |
| `POST /v1/chat/stream` | SSE streaming (when a stream handler is registered) |
| `GET /health` | Liveness probe |

CORS is enabled by default (required for the in-browser chat panel). Protocol spec: `~/Work/hopsworks-agent-protocol.md`.

## Quick start

```python
from hopsworks_agent_protocol import AgentApp, AgentResponse

agent_app = AgentApp(
    name="My agent",
    description="A custom LangGraph agent",
    welcome_message="How can I help?",
    suggested_prompts=["What is attention?"],
)


@agent_app.chat
async def chat(request):
    result = await my_agent.ainvoke(
        {"messages": request.to_framework_messages()},
        config={"configurable": {"thread_id": request.conversation_id}},
    )
    return AgentResponse.text(
        text=result["messages"][-1].content,
        conversation_id=request.conversation_id,
    )
```

Run it like any FastAPI app (`uvicorn my_agent:agent_app`). Deploy it as a Hopsworks agent deployment and the UI chat panel works with zero configuration.

Notes:

- `request.conversation_id` is **always set** (generated on the first turn) — safe to pass straight to a LangGraph checkpointer / LlamaIndex chat store. History is server-side: the request carries only the new message.
- `request.text` gives the plain message text; `request.to_framework_messages()` gives `[{"role", "content"}]` accepted by LangChain/LangGraph/LlamaIndex.
- Returning a plain `str` from the handler is shorthand for `AgentResponse.text(...)`.
- Raise `AgentError("msg", code="...", status_code=400, retryable=False)` for structured errors.

## Streaming

Register an async generator; the manifest advertises `streaming: true` automatically, `/v1/chat/stream` emits `message.delta` events, and `/v1/chat` still works by collecting the stream:

```python
@agent_app.stream
async def stream(request):
    async for event in my_agent.astream_events(...):
        if delta := extract_text(event):
            yield delta
    # optional: attach citations/usage to the final message
    yield AgentResponse.text(text="", citations=[...], usage={"output_tokens": 42})
```

If only a `@chat` handler exists, `/v1/chat/stream` degrades gracefully to a single `message.completed` event.

## Multimodal (v1.1)

Declare what your agent accepts/returns and use content parts:

```python
from hopsworks_agent_protocol import AgentApp, AgentResponse, ImageContent, TextContent

agent_app = AgentApp(
    name="Vision agent",
    input_modalities=["text", "image"],
    output_modalities=["text", "image"],
)

@agent_app.chat
async def chat(request):
    for image in request.images:          # base64 in image.data, image.media_type
        ...
    return AgentResponse.parts(
        TextContent(text="Here's the chart:"),
        ImageContent(media_type="image/png", data=chart_base64),
        conversation_id=request.conversation_id,
    )
```

`request.images` / `request.files` / `request.audio_clips` expose non-text parts; the manifest advertises the modalities so the Hopsworks chat panel enables the matching attachment pickers and renders returned images/files/audio inline. Binary parts don't stream — they arrive in the final `message.completed` response.

## Tracing (automatic)

When tracing is enabled on the Hopsworks deployment, the platform injects
`OTEL_EXPORTER_OTLP_TRACES_ENDPOINT` and runs an OTLP collector sidecar. The
library detects this and wires OTel automatically — no tracing code in the
agent. Declare the framework so the right OpenInference instrumentor is used
(or let the platform-injected `AGENT_FRAMEWORK` env var decide):

```python
agent_app = AgentApp(name="My agent", framework="langgraph")  # or "llamaindex" / "custom"
```

- `framework="langgraph"` → LangChain/LangGraph instrumentation (`pip install 'hopsworks-agent-protocol[langgraph]'`)
- `framework="llamaindex"` → LlamaIndex instrumentation (`[llamaindex]` extra)
- `framework="custom"` → provider only; instrument manually via `agent_app.tracer_provider`
- `tracing=False` opts out; `tracing=True` warns if the deployment has no tracing endpoint
- Missing instrumentation packages never crash the agent — it runs untraced with a warning

## Memory

The protocol keeps history server-side (`conversation_id`), so the SDK can own
storage. One store, three tiers — each opt-in, all served by the same object:

```python
from hopsworks_agent_protocol import (
    AgentApp, AgentMemoryService, anthropic_summarizer,
)

agent_app = AgentApp(
    name="My agent",
    memory=AgentMemoryService(          # zero-config in a deployment
        summarize=anthropic_summarizer(),  # tier 2
        long_term=True,                    # tier 3
    ),
)
```

| Tier | Holds | Turn on with |
|---|---|---|
| 1. Conversation buffer | this conversation's turns | `memory=` alone |
| 2. Rolling summary | older turns, compacted instead of dropped | `summarize=` |
| 3. Durable memory | facts about the user, across conversations | `long_term=True` |

### Reading it in a handler

```python
@agent_app.chat
async def chat(request, ctx):
    system = MY_PROMPT + ctx.system_context()   # summary + what you know
    messages = ctx.history + [{"role": "user", "content": request.text}]
    ...
```

> **`ctx.history` stops meaning "the conversation" once tier 2 is on.** It is
> the turns *since the last fold*; everything older is in `ctx.summary`. Passing
> `ctx.history` alone silently drops the compacted part.
> `ctx.system_context()` assembles the summary and this user's stored facts into
> a block for your system prompt, and returns `""` when there is nothing yet, so
> it is safe to concatenate unconditionally. The SDK builds it every turn but
> never places it — where context belongs is a property of your prompt.

Summarizing runs *after* the response has streamed and is awaited before the
route returns, so it costs request duration every Nth turn and never
time-to-answer. `summarize` is any callable
`(previous_summary, turns) -> str`, sync or async — `anthropic_summarizer()` is
a convenience, not a requirement.

### Agent-callable memory tools

Tier 3 adds `remember` / `recall` / `forget` / `search`, which the agent's own
LLM calls when it decides to — so there is no extraction model in the SDK
guessing what is worth keeping. Register them in your agent's tool list; the SDK
cannot reach into an arbitrary framework's tools, and appending to yours behind
your back would be worse than asking:

```python
from hopsworks_agent_protocol import memory_tools

agent = create_react_agent(llm, [*my_tools, *memory_tools("langgraph")])
# or memory_tools("llamaindex") -> [FunctionTool]; memory_tools("plain") -> bare functions
```

They take no store or user argument — those resolve from the request context, so
the signature the model sees carries no plumbing.

`search` looks over the user's own past conversations. It works as soon as tier 3
is on, using keyword matching; add an `embedder` and a `vector_store` to upgrade
it to semantic search over a Hopsworks embedding feature group, with no prompt
change:

```python
from hopsworks_agent_protocol import sentence_transformer_embedder, vector_store_for

embedder = sentence_transformer_embedder()          # [memory-search] extra
AgentMemoryService(long_term=True, embedder=embedder,
                      vector_store=vector_store_for(embedder))
```

### Backends and behaviour

- `InMemoryChatMemory()` — zero-config for development. Lost on restart and
  per-replica; agent deployments can scale to zero, so not for production.
  Tiers 2 and 3 are no-ops on it.
- `AgentMemoryService()` — inside a Hopsworks agent deployment this is
  zero-config: the project MySQL URL is built from the platform-injected
  `MYSQL_*` env vars (password via the `MYSQL_PASSWORD_SECRET_NAME` secret)
  and table names are derived from `DEPLOYMENT_ID`. Outside a deployment pass
  any SQLAlchemy URL (`[memory-sql]` extra). Survives restarts, shared across
  replicas.
- **A turn is recorded as it happens, not after it succeeds.** The user message
  is written when the turn opens — that is what lets a handler read back the
  message it is answering, and lets anything the agent remembers point at the
  turn that caused it. An open turn is invisible to `ctx.history` until it
  closes, and a turn that fails (handler error, client disconnect) is marked
  *abandoned* rather than left as a question whose answer never arrived.
- Memory failures never break the chat — they log and the reply still goes out.
- **If your framework persists state itself** (LangGraph checkpointer,
  LlamaIndex chat store), key it by `conversation_id` and skip `memory=` —
  keep one source of truth for history.
- **`subject` is client-asserted.** Tier 3 keys durable memory by
  `ChatRequest.subject`; the ingress authenticates a project-wide serving key,
  not a person, so the agent cannot verify it. Without one it falls back to the
  conversation id (memory degrades to per-conversation durability). Fine between
  project members; not a security boundary.

## Handler context (optional)

Declare a second parameter and the SDK passes a `HandlerContext` with per-turn
conveniences — `def chat(request)` and `def chat(request, ctx)` both work:

```python
@agent_app.chat
async def chat(request, ctx):
    history = ctx.history                       # turns since the last fold
    ctx.logger.info("turn for %s", ctx.conversation_id)
    await ctx.emit_event("retrieve", status="running", message="searching")
    ...
```

`ctx` exposes `conversation_id`, `request`, `memory`, `logger`, `deployment_id`,
`framework`, `response_id`/`message_id`/`turn_id`, and `emit_event`, plus the
memory accessors: `history`, `summary`, `state(scope=...)`, `system_context()`,
and `subject` (see [Memory](#memory)).

## Progress (tool) events

With `AgentApp(tool_events=True)`, tool calls surface as `tool_event` SSE frames
(interleaved with the reply while streaming; buffered into response `metadata`
otherwise), which the chat panel renders as progress chips.

- **Automatic** — when tracing is active, the SDK taps the framework
  instrumentation it already runs (LangChain/LangGraph, LlamaIndex) and emits a
  `running`/`done` event per tool span, keyed by span id. Zero code in the
  agent. (Requires the framework to propagate context into worker threads;
  LangChain does. Frameworks that don't will not auto-emit — use manual events
  or the trace view.)
- **Manual** — `await ctx.emit_event(name, status, message, data, event_id)`
  for custom progress; pass the same `event_id` for a call's start and end so
  the client shows one updating chip.
- **LangChain/LangGraph helper** — pipe `astream_events(version="v2")` through
  `ctx.stream_langchain(...)`: it yields the assistant text deltas and turns
  `on_tool_start`/`on_tool_end`/`on_tool_error` into tool-event chips
  automatically, with no `emit_event` calls and no dependency on tracing:

  ```python
  @agent_app.stream
  async def stream(request, ctx):
      async for delta in ctx.stream_langchain(agent.astream_events(inputs, version="v2")):
          yield delta
  ```
- **LlamaIndex helper** — `ctx.stream_llamaindex(handler)` does the same for a
  LlamaIndex workflow agent's run handler (`AgentStream` deltas + `ToolCall` /
  `ToolCallResult` chips):

  ```python
  @agent_app.stream
  async def stream(request, ctx):
      async for delta in ctx.stream_llamaindex(agent.run(msg)):
          yield delta
  ```

## Operational endpoints

- `GET /health` — liveness (process up).
- `GET /ready` — readiness: a handler is registered and the memory backend (if
  configured) is reachable; `503` otherwise, with per-check detail.
- With memory configured:
  - `GET /v1/conversations/{id}/messages` — the **human-facing transcript**,
    which is deliberately not what the model sees: once turns are folded they
    leave `ctx.history` but stay here, with `summary` and `summarized_through`
    marking where the two diverge. `?include=events` adds tool/event rows and
    abandoned turns.
  - `DELETE /v1/conversations/{id}` — what a client's "new session" calls to
    also drop server-side memory. Clears the conversation and its
    session-scoped state; leaves durable per-user memory alone, because
    starting a new chat must not erase what the agent knows about the person.
- With `long_term=True`, `GET`/`DELETE /v1/subjects/{subject}/state` let a user
  see every durable value held about them — with its provenance and whether the
  agent or an operator wrote it — and delete any of them. Caps and TTLs bound a
  false memory; being able to see and delete it is what fixes it.

## Agent structure graph

Expose the agent's structure and the chat panel shows a **Graph** tab. For a
compiled LangGraph, pass it straight in:

```python
graph = build_graph()            # a compiled LangGraph
agent_app = AgentApp(name="My agent", graph=graph)  # or graph=graph.get_graph()
```

The SDK serves `{nodes, edges}` at `GET /v1/graph` and advertises the `graph`
capability. A **LlamaIndex `Workflow`** works too — its graph is derived from
the `@step` methods' consumed/produced event types (event nodes collapsed into
labeled edges, `StartEvent`/`StopEvent` → `__start__`/`__end__`), so a custom
workflow renders as cleanly as a LangGraph. A plain `{"nodes", "edges"}` dict
works as well; an unreadable object is ignored (the tab just won't appear).

## Extra routes

`AgentApp` is a `FastAPI` — add anything else the usual way:

```python
@agent_app.get("/my/custom/route")
def custom():
    ...
```

## Development

```bash
pip install -e '.[dev]'
pytest
```

Optional extras keep the base install thin (FastAPI + Pydantic only):
`[tracing]`, `[langgraph]`, `[llamaindex]`, `[memory-sql]`, `[memory-search]`.

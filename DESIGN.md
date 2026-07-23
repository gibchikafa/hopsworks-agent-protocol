# Hopsworks Agent SDK — Design

Server-side Python library (`hopsworks-agent-protocol`, `github.com/gibchikafa/hopsworks-agent-protocol`) that makes any Python agent chat-ready in the Hopsworks UI. This documents the SDK's *design* — the wire contract it implements is specified separately in `hopsworks-agent-protocol.md`; the chat panel that consumes it in `agent-deployment-chat-design.md`.

## Purpose

Before the SDK, every Hopsworks agent hand-rolled the same ~150 lines: a FastAPI app, request parsing, session handling, a MySQL chat store, OTel tracer setup, and CORS — all subtly different, each re-solving problems the platform already knows the answer to (the DB URL, whether tracing is on, the deployment id). The SDK moves all of that behind one object so agent code is only domain logic:

```python
agent_app = AgentApp(name="My agent", framework="langgraph", memory=SqlChatMemory())

@agent_app.chat
async def chat(request):
    ...   # the only code the author writes
```

Everything else — HTTP surface, session identity, tracing, memory, CORS, the manifest — is the SDK reading the environment the platform sets up.

## Module layout

```
models.py    protocol Pydantic models + AgentResponse/AgentError helpers
app.py       AgentApp (FastAPI subclass): routes, handler decorators, streaming
memory.py    ChatMemory abstraction + InMemory/Sql backends + MySQL URL resolver
tracing.py   env-gated OTel setup + per-framework OpenInference instrumentation
```

~860 lines total. No runtime dependency beyond FastAPI + Pydantic; tracing, SQL, and framework instrumentation are optional extras.

## Key decisions

### 1. `AgentApp` subclasses `FastAPI`

Not a wrapper that owns a hidden app, not a `serve(fn)` function. Consequences:

- Runs anywhere a FastAPI app runs — `uvicorn module:agent_app`, the existing KServe/Knative deployment path — with no special launcher.
- Authors add custom routes, middleware, and dependencies the normal FastAPI way; the SDK's routes coexist.
- The four protocol routes (`/.well-known/hopsworks-agent.json`, `/v1/chat`, `/v1/chat/stream`, `/health`) and CORS are registered in `__init__`; CORS defaults on because the in-browser chat panel calls the agent cross-origin through the ingress and authors invariably forget it.

### 2. Framework and protocol are separate axes

This is the load-bearing distinction (a `langgraph` *wire protocol* was built and then removed once it became clear it conflated the two):

- **Framework** (`langgraph` / `llamaindex` / `custom`) — what the agent is *built with*. Lives in the pod. Consumers: tracing instrumentation, trace parsing. Resolved from the explicit `framework=` arg, else the platform-injected `AGENT_FRAMEWORK` env var, else `custom`. Advertised in the manifest as `agent.framework`.
- **Protocol** — what the agent *speaks over HTTP*. The SDK always emits the **hopsworks-agent** protocol regardless of framework.

The SDK is precisely the thing that decouples them: any framework in, one wire protocol out. That is why there is no `langgraph`/`llamaindex` *protocol* — the whole point is that the framework is invisible to the client. (`openai` survives as a UI-side protocol because it is a genuine wire contract of non-SDK servers like vLLM, not a framework.)

### 3. Two handler styles, one collapses into the other

`@app.chat` (returns a `ChatResponse`/str) and `@app.stream` (async generator of text deltas, optional final `ChatResponse` for citations/usage/media). Registering `@stream` flips the manifest's `streaming` capability automatically. Each endpoint degrades gracefully to the other handler: `/v1/chat` collects a stream into one response; `/v1/chat/stream` emits a single `message.completed` when only `@chat` exists. Authors implement whichever fits; both endpoints always work.

### 4. Conversation memory is opt-in, keyed by `conversation_id`

The protocol makes history server-side (clients send only the new message + a `conversation_id`), so the SDK can own storage. `ChatMemory` has two backends: `InMemoryChatMemory` (dev — documented as unfit for production since deployments scale to zero and replicas don't share it) and `SqlChatMemory` (any SQLAlchemy URL; zero-config inside a deployment, resolving the project MySQL from injected `MYSQL_*` env vars + the password secret, table name from `DEPLOYMENT_ID`).

Design choices:

- **Auto-recording**: the SDK appends both turns after each *successful* exchange (`status == completed`), so handlers only read. History returns as `{"role", "content"}` dicts — the shape LangChain/LangGraph/LlamaIndex all accept, so it drops straight into a framework call.
- **Opt-in, not default**: many frameworks already persist state (LangGraph checkpointers, LlamaIndex chat stores). Defaulting memory on would create two competing sources of truth. The rule, documented at the call site: if your framework persists state, key it by `conversation_id` and skip `memory=`.
- **Never fatal**: a memory failure logs and the reply still goes out. A dead MySQL degrades to statelessness, it doesn't 500 the chat.

### 5. Tracing is env-gated, zero-config

The platform runs an OTLP sidecar and injects `OTEL_EXPORTER_OTLP_TRACES_ENDPOINT` **only when tracing is enabled on the deployment**. The SDK keys off exactly that: endpoint present → build a `TracerProvider` (with `SimpleSpanProcessor`, which the sidecar's span propagation requires) and activate the framework-matching OpenInference instrumentor; endpoint absent → untraced, silently. `tracing=True/False` overrides the auto-detection. This mirror-images a real bug in the pre-SDK agents, which hardcoded `localhost:4318` as a fallback and spammed connection-refused retries on every request whenever tracing was disabled.

### 6. Everything degrades to a warning, never a crash

Optional dependencies (OTel SDK, instrumentation packages, SQLAlchemy) are imported lazily inside the code paths that need them. Missing package → logged warning + reduced functionality (untraced, or a clear "install the extra" error for memory), never an import-time failure. An agent that declares `framework="langgraph"` but forgot the `[langgraph]` extra still serves chat, just without framework spans.

## Versioning & packaging

- Package version tracks capability additions; protocol version (`protocol_version` in the manifest, currently `1.1`) tracks the wire contract and evolves additively — clients accept any `1.x` and ignore unknown fields.
- Optional extras keep the base install thin: `[tracing]`, `[langgraph]`, `[llamaindex]`, `[memory-sql]`. Base install is FastAPI + Pydantic only.
- Distributed as a git dependency today (`hopsworks-agent-protocol @ git+...`); the intended path is an internal PyPI publish or vendoring into the agent base image so `from hopsworks_agent_protocol import AgentApp` works out of the box in every deployment.

## Non-goals (deliberate)

- **No per-framework wire protocols.** The framework is an implementation detail the client must not see (§2).
- **No live/bidirectional voice.** Audio *attachments* fit as content parts; real-time spoken conversation is a different transport (WebSocket/WebRTC) and would be a separate capability, not bolted onto request/response.
- **No large-artifact storage in v1.** Binary content is inline base64 with a size cap; a `file_id` upload/download endpoint pair is reserved for when agents need to exchange artifacts larger than chat scale.
- **The SDK does not own the client.** It emits a protocol; the chat panel (and any other consumer) is independent.

## Open items / future work

- Publish to internal PyPI or bake into the agent base image.
- Tool-event streaming (`capabilities.tool_events`) — surface intermediate tool calls in the panel as they happen.
- File upload/download endpoints for large artifacts.
- A cookbook of framework recipes (LangGraph checkpointer + `conversation_id`, LlamaIndex chat engine, plain-Anthropic streaming) so authors copy the right integration for their framework.

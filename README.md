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

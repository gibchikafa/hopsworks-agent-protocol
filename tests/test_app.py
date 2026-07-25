import asyncio
import json

import pytest

from fastapi.testclient import TestClient

from hopsworks_agent_protocol import AgentApp, AgentError, AgentResponse


def make_request(text, conversation_id=None):
    return {
        "conversation_id": conversation_id,
        "message": {"role": "user", "content": [{"type": "text", "text": text}]},
    }


def parse_sse(body: str):
    events = []
    for frame in body.strip().split("\n\n"):
        event, data = None, None
        for line in frame.split("\n"):
            if line.startswith("event: "):
                event = line[len("event: ") :]
            elif line.startswith("data: "):
                data = json.loads(line[len("data: ") :])
        events.append((event, data))
    return events


def build_basic_app():
    app = AgentApp(
        name="Test agent",
        description="unit test",
        welcome_message="Hi!",
        suggested_prompts=["What is attention?"],
    )

    @app.chat
    async def chat(request):
        if request.text == "boom":
            raise AgentError("It broke", code="boom", status_code=400)
        return AgentResponse.text(
            text=f"echo: {request.text}",
            conversation_id=request.conversation_id,
            citations=[{"title": "doc", "score": 0.9}],
        )

    return app


class TestManifestAndHealth:
    def test_manifest(self):
        client = TestClient(build_basic_app())
        manifest = client.get("/.well-known/hopsworks-agent.json").json()
        assert manifest["protocol"] == "hopsworks-agent"
        assert manifest["protocol_version"] == "1.2"
        assert manifest["agent"]["name"] == "Test agent"
        assert manifest["endpoints"] == {"chat": "/v1/chat"}
        assert manifest["capabilities"]["streaming"] is False
        assert manifest["ui"]["welcome_message"] == "Hi!"
        assert manifest["ui"]["suggested_prompts"] == ["What is attention?"]

    def test_health(self):
        client = TestClient(build_basic_app())
        assert client.get("/health").json() == {"status": "ok"}


class TestChat:
    def test_chat_assigns_conversation_id(self):
        client = TestClient(build_basic_app())
        response = client.post("/v1/chat", json=make_request("hello")).json()
        assert response["message"]["content"][0]["text"] == "echo: hello"
        assert response["conversation_id"].startswith("conv_")
        assert response["status"] == "completed"
        assert response["citations"] == [{"title": "doc", "score": 0.9}]

    def test_chat_keeps_conversation_id(self):
        client = TestClient(build_basic_app())
        response = client.post(
            "/v1/chat", json=make_request("hello", "conv_keep")
        ).json()
        assert response["conversation_id"] == "conv_keep"

    def test_string_return_is_wrapped(self):
        app = AgentApp()

        @app.chat
        def chat(request):  # sync + plain string
            return request.text.upper()

        client = TestClient(app)
        response = client.post("/v1/chat", json=make_request("hi")).json()
        assert response["message"]["content"][0]["text"] == "HI"
        assert response["message"]["role"] == "assistant"

    def test_agent_error_envelope(self):
        client = TestClient(build_basic_app())
        result = client.post("/v1/chat", json=make_request("boom"))
        assert result.status_code == 400
        assert result.json()["detail"] == {
            "code": "boom",
            "message": "It broke",
            "retryable": False,
        }

    def test_no_handler_is_501(self):
        client = TestClient(AgentApp())
        assert client.post("/v1/chat", json=make_request("x")).status_code == 501


class TestStreaming:
    def build_streaming_app(self):
        app = AgentApp(name="Streamer")

        @app.stream
        async def stream(request):
            for chunk in ["Atten", "tion"]:
                yield chunk
            yield AgentResponse.text(
                text="", conversation_id=request.conversation_id, usage={"output_tokens": 2}
            )

        return app

    def test_manifest_advertises_streaming(self):
        client = TestClient(self.build_streaming_app())
        manifest = client.get("/.well-known/hopsworks-agent.json").json()
        assert manifest["capabilities"]["streaming"] is True
        assert manifest["endpoints"]["stream"] == "/v1/chat/stream"

    def test_stream_events(self):
        client = TestClient(self.build_streaming_app())
        result = client.post("/v1/chat/stream", json=make_request("q"))
        assert result.headers["content-type"].startswith("text/event-stream")
        events = parse_sse(result.text)
        assert events[0] == ("message.delta", {"delta": {"text": "Atten"}})
        assert events[1] == ("message.delta", {"delta": {"text": "tion"}})
        event, completed = events[-1]
        assert event == "message.completed"
        assert completed["message"]["content"][0]["text"] == "Attention"
        assert completed["usage"] == {"output_tokens": 2}
        assert completed["conversation_id"].startswith("conv_")

    def test_chat_collects_stream_when_no_chat_handler(self):
        client = TestClient(self.build_streaming_app())
        response = client.post("/v1/chat", json=make_request("q")).json()
        assert response["message"]["content"][0]["text"] == "Attention"

    def test_stream_falls_back_to_chat_handler(self):
        client = TestClient(build_basic_app())
        events = parse_sse(
            client.post("/v1/chat/stream", json=make_request("hi")).text
        )
        assert events[-1][0] == "message.completed"
        assert events[-1][1]["message"]["content"][0]["text"] == "echo: hi"

    def test_stream_error_event(self):
        app = AgentApp()

        @app.stream
        async def stream(request):
            yield "partial"
            raise AgentError("upstream died", code="upstream", retryable=True)

        client = TestClient(app)
        events = parse_sse(client.post("/v1/chat/stream", json=make_request("x")).text)
        assert events[-1] == (
            "error",
            {"code": "upstream", "message": "upstream died", "retryable": True},
        )


class TestRequestHelpers:
    def test_to_framework_messages(self):
        app = build_basic_app()

        captured = {}

        @app.chat
        async def chat(request):
            captured["messages"] = request.to_framework_messages()
            return "ok"

        client = TestClient(app)
        client.post("/v1/chat", json=make_request("hello world"))
        assert captured["messages"] == [{"role": "user", "content": "hello world"}]


class TestMultimodal:
    def build_vision_app(self):
        from hopsworks_agent_protocol import ImageContent, TextContent

        app = AgentApp(
            name="Vision agent",
            input_modalities=["text", "image"],
            output_modalities=["text", "image"],
        )

        @app.chat
        async def chat(request):
            n_images = len(request.images)
            return AgentResponse.parts(
                TextContent(text=f"got {n_images} image(s): {request.text}"),
                ImageContent(media_type="image/png", data="aGVsbG8="),
                conversation_id=request.conversation_id,
            )

        return app

    def test_manifest_modalities(self):
        client = TestClient(self.build_vision_app())
        caps = client.get("/.well-known/hopsworks-agent.json").json()["capabilities"]
        assert caps["input_modalities"] == ["text", "image"]
        assert caps["output_modalities"] == ["text", "image"]
        assert caps["attachments"] is True

    def test_text_only_manifest_defaults(self):
        client = TestClient(build_basic_app())
        caps = client.get("/.well-known/hopsworks-agent.json").json()["capabilities"]
        assert caps["input_modalities"] == ["text"]
        assert caps["attachments"] is False

    def test_image_request_and_response(self):
        client = TestClient(self.build_vision_app())
        response = client.post(
            "/v1/chat",
            json={
                "message": {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "what is this?"},
                        {"type": "image", "media_type": "image/png", "data": "aW1n"},
                    ],
                }
            },
        ).json()
        parts = response["message"]["content"]
        assert parts[0] == {"type": "text", "text": "got 1 image(s): what is this?"}
        assert parts[1] == {"type": "image", "media_type": "image/png", "data": "aGVsbG8="}

    def test_unknown_part_type_is_422(self):
        client = TestClient(self.build_vision_app())
        result = client.post(
            "/v1/chat",
            json={
                "message": {
                    "role": "user",
                    "content": [{"type": "hologram", "data": "x"}],
                }
            },
        )
        assert result.status_code == 422

    def test_stream_final_with_image_keeps_streamed_text(self):
        from hopsworks_agent_protocol import ImageContent

        app = AgentApp(output_modalities=["text", "image"])

        @app.stream
        async def stream(request):
            yield "Here is the chart"
            yield AgentResponse.parts(
                ImageContent(media_type="image/png", data="cGxvdA=="),
                conversation_id=request.conversation_id,
            )

        client = TestClient(app)
        events = parse_sse(client.post("/v1/chat/stream", json=make_request("chart")).text)
        event, completed = events[-1]
        assert event == "message.completed"
        parts = completed["message"]["content"]
        assert parts[0] == {"type": "text", "text": "Here is the chart"}
        assert parts[1]["type"] == "image"


class TestFrameworkAndTracing:
    def test_framework_defaults_to_custom(self, monkeypatch):
        monkeypatch.delenv("AGENT_FRAMEWORK", raising=False)
        app = AgentApp()
        assert app.framework == "custom"
        assert app.tracer_provider is None

    def test_framework_from_env(self, monkeypatch):
        monkeypatch.setenv("AGENT_FRAMEWORK", "langgraph")
        app = AgentApp()
        assert app.framework == "langgraph"
        manifest = TestClient(app).get("/.well-known/hopsworks-agent.json").json()
        assert manifest["agent"]["framework"] == "langgraph"

    def test_explicit_framework_wins_over_env(self, monkeypatch):
        monkeypatch.setenv("AGENT_FRAMEWORK", "langgraph")
        app = AgentApp(framework="llamaindex")
        assert app.framework == "llamaindex"

    def test_unknown_framework_falls_back(self, monkeypatch):
        monkeypatch.delenv("AGENT_FRAMEWORK", raising=False)
        app = AgentApp(framework="prolog")
        assert app.framework == "custom"

    def test_tracing_off_without_endpoint_env(self, monkeypatch):
        monkeypatch.delenv("OTEL_EXPORTER_OTLP_TRACES_ENDPOINT", raising=False)
        assert AgentApp(tracing=True).tracer_provider is None

    def test_tracing_disabled_explicitly(self, monkeypatch):
        monkeypatch.setenv(
            "OTEL_EXPORTER_OTLP_TRACES_ENDPOINT", "http://localhost:4318/v1/traces"
        )
        assert AgentApp(tracing=False).tracer_provider is None

    def test_tracing_graceful_without_otel_sdk(self, monkeypatch):
        # the test venv has no opentelemetry installed: the endpoint being set
        # must not crash the app, just leave it untraced
        monkeypatch.setenv(
            "OTEL_EXPORTER_OTLP_TRACES_ENDPOINT", "http://localhost:4318/v1/traces"
        )
        app = AgentApp()
        assert app.tracer_provider is None
        assert TestClient(app).get("/health").json() == {"status": "ok"}


class TestMemory:
    def build_memory_app(self, memory):
        app = AgentApp(memory=memory)

        @app.chat
        async def chat(request):
            history = app.memory.get(request.conversation_id)
            return AgentResponse.text(
                text=f"history={len(history)}: {request.text}",
                conversation_id=request.conversation_id,
            )

        return app

    def test_in_memory_auto_records_and_serves_history(self):
        from hopsworks_agent_protocol import InMemoryChatMemory

        memory = InMemoryChatMemory()
        client = TestClient(self.build_memory_app(memory))
        first = client.post("/v1/chat", json=make_request("hello")).json()
        cid = first["conversation_id"]
        assert first["message"]["content"][0]["text"] == "history=0: hello"
        assert memory.get(cid) == [
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "history=0: hello"},
        ]
        second = client.post("/v1/chat", json=make_request("again", cid)).json()
        assert second["message"]["content"][0]["text"] == "history=2: again"

    def test_in_memory_trims_to_max(self):
        from hopsworks_agent_protocol import InMemoryChatMemory

        memory = InMemoryChatMemory(max_messages=2)
        for i in range(4):
            memory.append("c1", "user", f"m{i}")
        assert memory.get("c1") == [
            {"role": "user", "content": "m2"},
            {"role": "user", "content": "m3"},
        ]

    def test_sql_memory_roundtrip(self, tmp_path):
        from hopsworks_agent_protocol import PersistentAgentMemory

        url = f"sqlite:///{tmp_path}/memory.db"
        memory = PersistentAgentMemory(url)
        client = TestClient(self.build_memory_app(memory))
        cid = client.post("/v1/chat", json=make_request("hi")).json()[
            "conversation_id"
        ]
        # a fresh store over the same db sees the persisted turns (no cache)
        fresh = PersistentAgentMemory(url)
        assert fresh.get(cid) == [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "history=0: hi"},
        ]
        fresh.clear(cid)
        assert fresh.get(cid) == []

    def test_memory_failure_does_not_break_chat(self):
        from hopsworks_agent_protocol import InMemoryChatMemory

        class BrokenMemory(InMemoryChatMemory):
            def begin_turn(self, *a, **k):
                raise RuntimeError("db down")

        client = TestClient(self.build_memory_app(BrokenMemory()))
        result = client.post("/v1/chat", json=make_request("hello"))
        assert result.status_code == 200

    def test_close_failure_does_not_break_chat(self):
        from hopsworks_agent_protocol import InMemoryChatMemory

        class BrokenMemory(InMemoryChatMemory):
            def end_turn(self, *a, **k):
                raise RuntimeError("db down")

        client = TestClient(self.build_memory_app(BrokenMemory()))
        result = client.post("/v1/chat", json=make_request("hello"))
        assert result.status_code == 200


class TestDeploymentMysqlUrl:
    def test_builds_url_from_envs(self, monkeypatch):
        from hopsworks_agent_protocol.memory import deployment_mysql_url

        monkeypatch.setenv("MYSQL_USER", "u")
        monkeypatch.setenv("MYSQL_HOST", "mysql.svc")
        monkeypatch.setenv("MYSQL_DB", "agents")
        monkeypatch.setenv("MYSQL_PASSWORD", "pw")
        monkeypatch.delenv("MYSQL_PORT", raising=False)
        assert (
            deployment_mysql_url() == "mysql+pymysql://u:pw@mysql.svc:3306/agents"
        )

    def test_missing_env_has_helpful_error(self, monkeypatch):
        from hopsworks_agent_protocol.memory import deployment_mysql_url

        monkeypatch.delenv("MYSQL_USER", raising=False)
        try:
            deployment_mysql_url()
            raise AssertionError("expected RuntimeError")
        except RuntimeError as err:
            assert "MYSQL_USER" in str(err)
            assert "url=" in str(err)


class TestSqlMemoryResilience:
    def test_unreachable_db_does_not_crash_construction_or_turns(self):
        from hopsworks_agent_protocol import PersistentAgentMemory

        memory = PersistentAgentMemory(url="mysql+pymysql://x:y@127.0.0.1:1/nope")
        assert memory.get("c1") == []
        memory.append("c1", "user", "hi")
        assert memory.get("c1") == []
        memory.clear("c1")

    def test_lazy_url_resolution_defers_env_errors(self, monkeypatch):
        from hopsworks_agent_protocol import PersistentAgentMemory

        monkeypatch.delenv("MYSQL_USER", raising=False)
        memory = PersistentAgentMemory()
        assert memory.get("c1") == []


class TestContextObject:
    def test_two_param_handler_receives_context(self):
        from hopsworks_agent_protocol import AgentApp, InMemoryChatMemory

        app = AgentApp(memory=InMemoryChatMemory())
        seen = {}

        @app.chat
        async def chat(request, ctx):
            seen["cid"] = ctx.conversation_id
            seen["history_len"] = len(ctx.history)
            seen["framework"] = ctx.framework
            seen["response_id"] = ctx.response_id
            return f"turn {len(ctx.history)}"

        client = TestClient(app)
        first = client.post("/v1/chat", json=make_request("hi")).json()
        assert seen["cid"] == first["conversation_id"]
        assert seen["history_len"] == 0
        assert seen["framework"] == "custom"
        # ctx.response_id matches the id on the response the client received
        assert first["id"] == seen["response_id"]
        # second turn sees prior history via ctx.history
        client.post("/v1/chat", json=make_request("again", first["conversation_id"]))
        assert seen["history_len"] == 2

    def test_one_param_handler_still_works(self):
        from hopsworks_agent_protocol import AgentApp

        app = AgentApp()

        @app.chat
        async def chat(request):
            return "ok"

        assert (
            TestClient(app)
            .post("/v1/chat", json=make_request("x"))
            .json()["message"]["content"][0]["text"]
            == "ok"
        )


class TestReadiness:
    def test_ready_ok(self):
        app = build_basic_app()
        result = TestClient(app).get("/ready")
        assert result.status_code == 200
        assert result.json()["status"] == "ready"

    def test_not_ready_without_handler(self):
        from hopsworks_agent_protocol import AgentApp

        result = TestClient(AgentApp()).get("/ready")
        assert result.status_code == 503
        assert result.json()["checks"]["handler"] is False

    def test_not_ready_when_memory_unreachable(self):
        from hopsworks_agent_protocol import AgentApp, PersistentAgentMemory

        app = AgentApp(memory=PersistentAgentMemory(url="mysql+pymysql://x:y@127.0.0.1:1/no"))

        @app.chat
        async def chat(request):
            return "ok"

        result = TestClient(app).get("/ready")
        assert result.status_code == 503
        assert result.json()["checks"]["memory"] is False


class TestConversationEndpoints:
    def test_history_and_clear(self):
        from hopsworks_agent_protocol import AgentApp, InMemoryChatMemory

        app = AgentApp(memory=InMemoryChatMemory())

        @app.chat
        async def chat(request):
            return "reply"

        client = TestClient(app)
        cid = client.post("/v1/chat", json=make_request("hello")).json()[
            "conversation_id"
        ]
        body = client.get(f"/v1/conversations/{cid}/messages").json()
        assert [(m["role"], m["content"]) for m in body["messages"]] == [
            ("user", "hello"),
            ("assistant", "reply"),
        ]
        assert body["summary"] is None
        assert body["summarized_through"] == 0
        assert client.delete(f"/v1/conversations/{cid}").status_code == 204
        assert client.get(f"/v1/conversations/{cid}/messages").json()["messages"] == []

    def test_endpoints_absent_without_memory(self):
        from hopsworks_agent_protocol import AgentApp

        app = AgentApp()

        @app.chat
        async def chat(request):
            return "ok"

        client = TestClient(app)
        assert client.get("/v1/conversations/x/messages").status_code == 404
        manifest = client.get("/.well-known/hopsworks-agent.json").json()
        assert manifest["capabilities"]["conversation_management"] is False
        assert "conversations" not in manifest["endpoints"]

    def test_manifest_advertises_conversations_with_memory(self):
        from hopsworks_agent_protocol import AgentApp, InMemoryChatMemory

        app = AgentApp(memory=InMemoryChatMemory())

        @app.chat
        async def chat(request):
            return "ok"

        manifest = TestClient(app).get("/.well-known/hopsworks-agent.json").json()
        assert manifest["capabilities"]["conversation_management"] is True
        assert manifest["endpoints"]["conversations"] == "/v1/conversations"


class TestToolEvents:
    def test_tool_events_capability_flag(self):
        from hopsworks_agent_protocol import AgentApp

        off = TestClient(AgentApp()).get("/.well-known/hopsworks-agent.json").json()
        assert off["capabilities"]["tool_events"] is False
        on = (
            TestClient(AgentApp(tool_events=True))
            .get("/.well-known/hopsworks-agent.json")
            .json()
        )
        assert on["capabilities"]["tool_events"] is True

    def test_emit_event_interleaves_in_stream(self):
        from hopsworks_agent_protocol import AgentApp

        app = AgentApp(tool_events=True)

        @app.stream
        async def stream(request, ctx):
            await ctx.emit_event("retrieve", status="running", message="searching")
            yield "answer "
            await ctx.emit_event("retrieve", status="done")
            yield "text"

        events = parse_sse(
            TestClient(app).post("/v1/chat/stream", json=make_request("q")).text
        )
        kinds = [e[0] for e in events]
        assert "tool_event" in kinds
        assert kinds[0] == "tool_event"
        assert events[0][1] == {
            "name": "retrieve",
            "status": "running",
            "message": "searching",
        }
        assert kinds[-1] == "message.completed"
        assert events[-1][1]["message"]["content"][0]["text"] == "answer text"

    def test_emit_event_buffered_in_non_streaming(self):
        from hopsworks_agent_protocol import AgentApp

        app = AgentApp()

        @app.chat
        async def chat(request, ctx):
            await ctx.emit_event("tool", status="done")
            return "ok"

        response = TestClient(app).post("/v1/chat", json=make_request("x")).json()
        assert response["metadata"]["tool_events"] == [
            {"name": "tool", "status": "done"}
        ]


class TestAutoToolEvents:
    def test_span_processor_emits_from_tool_span(self):
        # simulate the OTel span processor path directly: a HandlerContext with
        # a queue receives running/done events keyed by span id
        import asyncio

        from hopsworks_agent_protocol.autoevents import current_context
        from hopsworks_agent_protocol.context import HandlerContext
        from hopsworks_agent_protocol.models import ChatMessage, ChatRequest, TextContent

        async def run():
            req = ChatRequest(
                conversation_id="c1",
                message=ChatMessage(role="user", content=[TextContent(text="hi")]),
            )
            ctx = HandlerContext(req, None, "langgraph", "response_1")
            q: asyncio.Queue = asyncio.Queue()
            ctx._event_queue = q
            ctx._loop = asyncio.get_running_loop()
            token = current_context.set(ctx)
            try:
                # _emit_sync is what the span processor calls
                ctx._emit_sync("search_papers", "running", None, None, "span-abc")
                ctx._emit_sync("search_papers", "done", None, None, "span-abc")
            finally:
                current_context.reset(token)
            await asyncio.sleep(0)  # let call_soon_threadsafe run
            events = []
            while not q.empty():
                events.append(q.get_nowait())
            return events

        events = asyncio.run(run())
        assert events[0] == ("tool_event", {"name": "search_papers", "status": "running", "id": "span-abc"})
        assert events[1] == ("tool_event", {"name": "search_papers", "status": "done", "id": "span-abc"})

    def test_emit_event_id_collapses(self):
        from hopsworks_agent_protocol import AgentApp

        app = AgentApp(tool_events=True)

        @app.stream
        async def stream(request, ctx):
            await ctx.emit_event("t", status="running", event_id="e1")
            yield "x"
            await ctx.emit_event("t", status="done", event_id="e1")

        events = parse_sse(
            TestClient(app).post("/v1/chat/stream", json=make_request("q")).text
        )
        tool_events = [e[1] for e in events if e[0] == "tool_event"]
        assert tool_events[0] == {"name": "t", "status": "running", "id": "e1"}
        assert tool_events[1] == {"name": "t", "status": "done", "id": "e1"}


class TestStreamLangchainHelper:
    def _fake_events(self):
        # a langgraph astream_events(v2) sequence with a tool call
        async def gen():
            yield {"event": "on_chat_model_stream",
                   "data": {"chunk": type("C", (), {"content": "Let me search "})()}}
            yield {"event": "on_tool_start", "name": "search_papers",
                   "run_id": "r1", "data": {"input": {"query": "CoT"}}}
            yield {"event": "on_tool_end", "name": "search_papers",
                   "run_id": "r1", "data": {}}
            yield {"event": "on_chat_model_stream",
                   "data": {"chunk": type("C", (), {"content": "done."})()}}
        return gen()

    def test_helper_yields_text_and_emits_tool_events(self):
        from hopsworks_agent_protocol import AgentApp

        app = AgentApp(tool_events=True)
        helper = self

        @app.stream
        async def stream(request, ctx):
            async for delta in ctx.stream_langchain(helper._fake_events()):
                yield delta

        events = parse_sse(
            TestClient(app).post("/v1/chat/stream", json=make_request("q")).text
        )
        deltas = [e[1]["delta"]["text"] for e in events if e[0] == "message.delta"]
        tools = [e[1] for e in events if e[0] == "tool_event"]
        assert "".join(deltas) == "Let me search done."
        assert tools == [
            {"name": "search_papers", "status": "running", "id": "r1",
             "message": "{'query': 'CoT'}"},
            {"name": "search_papers", "status": "done", "id": "r1"},
        ]
        assert events[-1][1]["message"]["content"][0]["text"] == "Let me search done."


class TestGraph:
    class _Node:
        def __init__(self, nid, name=None):
            self.id = nid
            self.name = name or nid

    class _Edge:
        def __init__(self, source, target, data=None, conditional=False):
            self.source = source
            self.target = target
            self.data = data
            self.conditional = conditional

    class _Drawable:
        def __init__(self, nodes, edges):
            self.nodes = nodes
            self.edges = edges

    class _Compiled:
        def __init__(self, drawable):
            self._d = drawable

        def get_graph(self):
            return self._d

    def _langgraph_like(self):
        # mirrors agent.get_graph(): __start__ -> agent -> action/__end__
        nodes = {
            "__start__": self._Node("__start__"),
            "agent": self._Node("agent"),
            "action": self._Node("action"),
            "__end__": self._Node("__end__"),
        }
        edges = [
            self._Edge("__start__", "agent"),
            self._Edge("agent", "action", data="continue", conditional=True),
            self._Edge("agent", "__end__", data="end", conditional=True),
            self._Edge("action", "agent"),
        ]
        return self._Compiled(self._Drawable(nodes, edges))

    def test_manifest_and_endpoint(self):
        from hopsworks_agent_protocol import AgentApp

        app = AgentApp(graph=self._langgraph_like())

        @app.chat
        async def chat(request):
            return "ok"

        client = TestClient(app)
        manifest = client.get("/.well-known/hopsworks-agent.json").json()
        assert manifest["capabilities"]["graph"] is True
        assert manifest["endpoints"]["graph"] == "/v1/graph"

        graph = client.get("/v1/graph").json()
        assert {n["id"] for n in graph["nodes"]} == {
            "__start__", "agent", "action", "__end__"
        }
        cont = next(e for e in graph["edges"] if e.get("label") == "continue")
        assert cont["source"] == "agent" and cont["target"] == "action"
        assert cont["conditional"] is True

    def test_no_graph_capability_off(self):
        app = build_basic_app()
        m = TestClient(app).get("/.well-known/hopsworks-agent.json").json()
        assert m["capabilities"]["graph"] is False
        assert "graph" not in m["endpoints"]
        assert TestClient(app).get("/v1/graph").status_code == 404

    def test_accepts_plain_dict(self):
        from hopsworks_agent_protocol import AgentApp
        from hopsworks_agent_protocol.graph import to_graph_spec

        spec = {"nodes": [{"id": "a", "label": "a"}], "edges": []}
        assert to_graph_spec(spec) == spec
        app = AgentApp(graph=spec)

        @app.chat
        async def chat(request):
            return "ok"

        assert TestClient(app).get("/v1/graph").json() == spec

    def test_unreadable_graph_is_ignored(self):
        from hopsworks_agent_protocol.graph import to_graph_spec

        assert to_graph_spec(object()) is None
        assert to_graph_spec(None) is None


class TestStreamLlamaindexHelper:
    def _handler(self):
        # duck-typed llama-index workflow events
        AgentStream = type("AgentStream", (), {})
        ToolCall = type("ToolCall", (), {})
        ToolCallResult = type("ToolCallResult", (), {})

        def ev(cls, **attrs):
            e = cls()
            for k, v in attrs.items():
                setattr(e, k, v)
            return e

        events = [
            ev(AgentStream, delta="Let me search "),
            ev(ToolCall, tool_name="search_papers", tool_id="t1",
               tool_kwargs={"query": "CoT"}),
            ev(ToolCallResult, tool_name="search_papers", tool_id="t1"),
            ev(AgentStream, delta="done."),
        ]

        class Handler:
            async def stream_events(self):
                for e in events:
                    yield e

        return Handler()

    def test_yields_text_and_emits_tool_events(self):
        from hopsworks_agent_protocol import AgentApp

        app = AgentApp(tool_events=True)
        helper = self

        @app.stream
        async def stream(request, ctx):
            async for delta in ctx.stream_llamaindex(helper._handler()):
                yield delta

        events = parse_sse(
            TestClient(app).post("/v1/chat/stream", json=make_request("q")).text
        )
        deltas = [e[1]["delta"]["text"] for e in events if e[0] == "message.delta"]
        tools = [e[1] for e in events if e[0] == "tool_event"]
        assert "".join(deltas) == "Let me search done."
        assert tools == [
            {"name": "search_papers", "status": "running", "id": "t1",
             "message": "{'query': 'CoT'}"},
            {"name": "search_papers", "status": "done", "id": "t1"},
        ]


class TestLlamaIndexWorkflowGraph:
    def _workflow(self):
        # duck-typed LlamaIndex Workflow: _get_steps() -> {name: fn with __step_config}
        Start = type("StartEvent", (), {})
        Stop = type("StopEvent", (), {})
        Search = type("SearchEvent", (), {})
        Answer = type("AnswerEvent", (), {})

        def step(fn, accepted, returns):
            cfg = type("Cfg", (), {})()
            cfg.accepted_events = accepted
            cfg.return_types = returns
            setattr(fn, "__step_config", cfg)
            return fn

        def plan():
            pass

        def search():
            pass

        def answer():
            pass

        step(plan, [Start], [Search, Stop])   # branches: search or stop
        step(search, [Search], [Answer])
        step(answer, [Answer], [Stop])

        class WF:
            def _get_steps(self):
                return {"plan": plan, "search": search, "answer": answer}

        return WF()

    def test_workflow_graph_from_steps(self):
        from hopsworks_agent_protocol.graph import to_graph_spec

        spec = to_graph_spec(self._workflow())
        assert spec is not None
        ids = {n["id"] for n in spec["nodes"]}
        assert ids == {"plan", "search", "answer", "__start__", "__end__"}
        edges = {(e["source"], e["target"]) for e in spec["edges"]}
        assert ("__start__", "plan") in edges
        assert ("plan", "search") in edges       # via SearchEvent
        assert ("search", "answer") in edges     # via AnswerEvent
        assert ("answer", "__end__") in edges     # via StopEvent
        assert ("plan", "__end__") in edges       # plan can also stop
        # plan branches (SearchEvent | StopEvent) -> conditional edges
        plan_search = next(
            e for e in spec["edges"]
            if e["source"] == "plan" and e["target"] == "search"
        )
        assert plan_search["conditional"] is True
        assert plan_search["label"] == "SearchEvent"

    def test_served_via_agentapp(self):
        from hopsworks_agent_protocol import AgentApp

        app = AgentApp(graph=self._workflow())

        @app.chat
        async def chat(request):
            return "ok"

        client = TestClient(app)
        assert (
            client.get("/.well-known/hopsworks-agent.json").json()["capabilities"][
                "graph"
            ]
            is True
        )
        assert "plan" in {n["id"] for n in client.get("/v1/graph").json()["nodes"]}


class TestTurnLifecycle:
    """The user message is recorded before the handler runs, so every exit path
    has to close the turn. These cover the paths that previously just skipped
    recording entirely — where skipping now means leaving an orphan."""

    def _rows(self, memory, conversation_id=None):
        from sqlalchemy import select

        memory.healthcheck()
        t = memory._table
        stmt = select(t).order_by(t.c.id)
        if conversation_id:
            stmt = stmt.where(t.c.conversation_id == conversation_id)
        with memory._engine.connect() as conn:
            return conn.execute(stmt).fetchall()

    def _store(self, tmp_path, **kw):
        from hopsworks_agent_protocol import PersistentAgentMemory

        return PersistentAgentMemory(
            url=f"sqlite:///{tmp_path}/m.db", table_name="agent_memory_items_t", **kw
        )

    def test_open_turn_is_invisible_until_closed(self, tmp_path):
        memory = self._store(tmp_path)
        seen = {}

        app = AgentApp(memory=memory)

        @app.chat
        async def chat(request, ctx):
            # the handler can read the message it is answering...
            seen["history"] = memory.get(request.conversation_id)
            return "answer"

        client = TestClient(app)
        cid = client.post("/v1/chat", json=make_request("question")).json()[
            "conversation_id"
        ]
        # ...but it is not yet part of history, because the turn is still open
        assert seen["history"] == []
        assert memory.get(cid) == [
            {"role": "user", "content": "question"},
            {"role": "assistant", "content": "answer"},
        ]
        assert {r.status for r in self._rows(memory, cid)} == {"closed"}

    def test_handler_error_abandons_turn_and_leaves_no_orphan(self, tmp_path):
        memory = self._store(tmp_path)
        app = AgentApp(memory=memory)

        @app.chat
        async def chat(request):
            raise AgentError("nope", code="boom", status_code=400)

        client = TestClient(app)
        result = client.post("/v1/chat", json=make_request("question"))
        assert result.status_code == 400
        cid = "c-err"
        client.post("/v1/chat", json=make_request("q2", conversation_id=cid))
        # the failed turn's question is retained for debugging but never
        # reappears as history
        rows = self._rows(memory)
        assert [r.status for r in rows] == ["abandoned", "abandoned"]
        assert memory.get(cid) == []

    def test_stream_error_abandons_turn(self, tmp_path):
        memory = self._store(tmp_path)
        app = AgentApp(memory=memory)

        @app.stream
        async def stream(request):
            yield "partial"
            raise RuntimeError("mid-stream failure")

        client = TestClient(app)
        with client.stream(
            "POST", "/v1/chat/stream", json=make_request("question", "c1")
        ) as r:
            body = "".join(r.iter_text())
        assert "event: error" in body
        assert [r.status for r in self._rows(memory, "c1")] == ["abandoned"]
        assert memory.get("c1") == []

    def test_successful_stream_closes_turn(self, tmp_path):
        memory = self._store(tmp_path)
        app = AgentApp(memory=memory)

        @app.stream
        async def stream(request):
            yield "hello "
            yield "world"

        client = TestClient(app)
        with client.stream(
            "POST", "/v1/chat/stream", json=make_request("question", "c1")
        ) as r:
            "".join(r.iter_text())
        assert memory.get("c1") == [
            {"role": "user", "content": "question"},
            {"role": "assistant", "content": "hello world"},
        ]

    def test_turn_rows_carry_identity(self, tmp_path):
        memory = self._store(tmp_path)
        app = AgentApp(memory=memory)

        @app.chat
        async def chat(request):
            return "answer"

        client = TestClient(app)
        client.post("/v1/chat", json=make_request("question", "c1"))
        rows = self._rows(memory, "c1")
        assert [r.seq for r in rows] == [0, 1]
        assert [r.role for r in rows] == ["user", "assistant"]
        # one turn, one message id, linking the reply to the question
        assert len({r.turn_id for r in rows}) == 1
        assert len({r.message_id for r in rows}) == 1
        assert rows[0].message_id is not None

    def test_tool_events_are_recorded_but_not_history(self, tmp_path):
        memory = self._store(tmp_path)
        app = AgentApp(memory=memory, tool_events=True)

        @app.chat
        async def chat(request, ctx):
            await ctx.emit_event("search", status="done")
            return "answer"

        client = TestClient(app)
        client.post("/v1/chat", json=make_request("question", "c1"))
        rows = self._rows(memory, "c1")
        assert [r.memory_type for r in rows] == ["message", "event", "message"]
        # excluded from what the model reads back
        assert [t["role"] for t in memory.get("c1")] == ["user", "assistant"]
        assert json.loads(rows[1].content)["name"] == "search"

    def test_reaper_abandons_stale_open_turns(self, tmp_path):
        memory = self._store(tmp_path, turn_timeout_seconds=0)
        memory.begin_turn("c1", "turn_stale", "user", "question")
        assert memory.get("c1") == []
        assert memory.reap_open_turns() == 1
        assert [r.status for r in self._rows(memory, "c1")] == ["abandoned"]
        # already-reaped rows are not re-counted
        assert memory.reap_open_turns() == 0

    def test_reaper_spares_turns_inside_the_timeout(self, tmp_path):
        memory = self._store(tmp_path, turn_timeout_seconds=3600)
        memory.begin_turn("c1", "turn_live", "user", "question")
        assert memory.reap_open_turns() == 0
        assert [r.status for r in self._rows(memory, "c1")] == ["open"]

    def test_abandoned_turn_does_not_poison_the_cache(self, tmp_path):
        memory = self._store(tmp_path)
        memory.begin_turn("c1", "t1", "user", "one")
        memory.record_item("c1", "t1", "assistant", "first")
        memory.end_turn("c1", "t1")
        assert len(memory.get("c1")) == 2
        memory.begin_turn("c1", "t2", "user", "two")
        memory.end_turn("c1", "t2", status="abandoned")
        assert memory.get("c1") == [
            {"role": "user", "content": "one"},
            {"role": "assistant", "content": "first"},
        ]


class TestMemorySchemaGuards:
    def test_rejects_table_name_with_invalid_suffix(self, tmp_path):
        from hopsworks_agent_protocol import PersistentAgentMemory

        memory = PersistentAgentMemory(
            url=f"sqlite:///{tmp_path}/m.db", table_name="items; DROP TABLE x"
        )
        # degrades to stateless rather than creating a junk table
        assert memory.healthcheck() is False

    def test_requires_deployment_id_when_url_is_derived(self, monkeypatch):
        from hopsworks_agent_protocol import PersistentAgentMemory

        monkeypatch.delenv("DEPLOYMENT_ID", raising=False)
        memory = PersistentAgentMemory()
        with pytest.raises(RuntimeError, match="DEPLOYMENT_ID"):
            memory._resolve_table_name()

    def test_uses_deployment_id_for_table_name(self, monkeypatch, tmp_path):
        from hopsworks_agent_protocol import PersistentAgentMemory

        monkeypatch.setenv("DEPLOYMENT_ID", "42")
        memory = PersistentAgentMemory(url=f"sqlite:///{tmp_path}/m.db")
        assert memory._resolve_table_name() == "agent_memory_items_42"

    def test_records_schema_version(self, tmp_path):
        from sqlalchemy import select

        from hopsworks_agent_protocol import PersistentAgentMemory
        from hopsworks_agent_protocol.memory import SCHEMA_VERSION

        memory = PersistentAgentMemory(
            url=f"sqlite:///{tmp_path}/m.db", table_name="agent_memory_items_t"
        )
        assert memory.healthcheck() is True
        with memory._engine.connect() as conn:
            row = conn.execute(select(memory._meta)).fetchone()
        assert row.schema_version == SCHEMA_VERSION

    def test_refuses_table_written_by_a_newer_sdk(self, tmp_path):
        from hopsworks_agent_protocol import PersistentAgentMemory
        from hopsworks_agent_protocol.memory import SCHEMA_VERSION

        url = f"sqlite:///{tmp_path}/m.db"
        first = PersistentAgentMemory(url=url, table_name="agent_memory_items_t")
        assert first.healthcheck() is True
        with first._engine.begin() as conn:
            conn.execute(
                first._meta.update().values(schema_version=SCHEMA_VERSION + 1)
            )
        # a second process on an older SDK must not write through it
        second = PersistentAgentMemory(url=url, table_name="agent_memory_items_t")
        assert second.healthcheck() is False
        second.begin_turn("c1", "t1", "user", "hi")
        assert second.get("c1") == []


class TestStreamAbort:
    """A client that vanishes mid-stream is the case that previously recorded
    nothing at all; now the question is already stored, so the generator's
    cleanup has to close the turn."""

    def test_closing_the_stream_early_abandons_the_turn(self, tmp_path):
        import asyncio

        from hopsworks_agent_protocol import PersistentAgentMemory

        memory = PersistentAgentMemory(
            url=f"sqlite:///{tmp_path}/m.db", table_name="agent_memory_items_t"
        )
        app = AgentApp(memory=memory)

        @app.stream
        async def stream(request):
            yield "first"
            await asyncio.sleep(10)  # client goes away while we are here
            yield "never"

        async def scenario():
            from hopsworks_agent_protocol.models import ChatRequest

            request = ChatRequest.model_validate(make_request("question", "c1"))
            ctx = app._prepare(request)
            await app._open_turn(ctx)

            class _Raw:
                async def is_disconnected(self):
                    return False

            agen = app._stream_events(ctx, _Raw())
            await agen.__anext__()  # pull the first delta
            await agen.aclose()  # client disconnects

        asyncio.run(scenario())

        memory.healthcheck()
        with memory._engine.connect() as conn:
            rows = conn.execute(memory._table.select()).fetchall()
        assert [r.status for r in rows] == ["abandoned"]
        assert memory.get("c1") == []


def _fake_summarizer(calls):
    """Records what it was asked to fold and returns a deterministic summary."""

    def summarize(previous, turns):
        calls.append((previous, list(turns)))
        joined = "; ".join(t["content"] for t in turns)
        return f"[{previous}] {joined}" if previous else joined

    return summarize


class TestSummarization:
    def _store(self, tmp_path, calls=None, **kw):
        from hopsworks_agent_protocol import PersistentAgentMemory

        kw.setdefault("summarize", _fake_summarizer(calls if calls is not None else []))
        kw.setdefault("summarize_after_messages", 4)
        kw.setdefault("keep_recent_messages", 2)
        return PersistentAgentMemory(
            url=f"sqlite:///{tmp_path}/m.db", table_name="agent_memory_items_t", **kw
        )

    def _turn(self, memory, cid, q, a):
        from hopsworks_agent_protocol.memory import new_turn_id

        tid = new_turn_id()
        memory.begin_turn(cid, tid, "user", q)
        memory.record_item(cid, tid, "assistant", a)
        memory.end_turn(cid, tid)

    def test_no_fold_before_the_threshold(self, tmp_path):
        calls = []
        memory = self._store(tmp_path, calls)
        self._turn(memory, "c1", "q1", "a1")
        assert asyncio.run(memory.maybe_summarize("c1")) is False
        assert calls == []
        assert memory.get_summary("c1") is None

    def test_fold_summarizes_and_shrinks_history(self, tmp_path):
        calls = []
        memory = self._store(tmp_path, calls)
        for i in range(3):
            self._turn(memory, "c1", f"q{i}", f"a{i}")
        assert len(memory.get("c1")) == 6

        assert asyncio.run(memory.maybe_summarize("c1")) is True
        # folded everything except the verbatim tail
        assert memory.get_summary("c1") == "q0; a0; q1; a1"
        assert memory.get("c1") == [
            {"role": "user", "content": "q2"},
            {"role": "assistant", "content": "a2"},
        ]
        # and the folded turns are not handed to the model twice
        assert calls[0][0] is None
        assert [t["content"] for t in calls[0][1]] == ["q0", "a0", "q1", "a1"]

    def test_second_fold_is_incremental(self, tmp_path):
        calls = []
        memory = self._store(tmp_path, calls)
        for i in range(3):
            self._turn(memory, "c1", f"q{i}", f"a{i}")
        asyncio.run(memory.maybe_summarize("c1"))
        for i in range(3, 6):
            self._turn(memory, "c1", f"q{i}", f"a{i}")
        assert asyncio.run(memory.maybe_summarize("c1")) is True

        # the previous summary is passed back in, so the summary is rewritten
        # rather than rebuilt from the whole conversation
        assert calls[1][0] == "q0; a0; q1; a1"
        assert [t["content"] for t in calls[1][1]] == [
            "q2", "a2", "q3", "a3", "q4", "a4",
        ]
        assert memory.get("c1") == [
            {"role": "user", "content": "q5"},
            {"role": "assistant", "content": "a5"},
        ]

    def test_fold_never_crosses_an_open_turn(self, tmp_path):
        """The bug pre-insertion creates: fold a question whose answer is still
        being generated and the reply is left dangling with no question."""
        calls = []
        memory = self._store(tmp_path, calls, keep_recent_messages=0)
        for i in range(2):
            self._turn(memory, "c1", f"q{i}", f"a{i}")
        # a concurrent turn is mid-flight
        memory.begin_turn("c1", "turn_inflight", "user", "in-flight question")

        assert asyncio.run(memory.maybe_summarize("c1")) is True
        folded = [t["content"] for t in calls[0][1]]
        assert folded == ["q0", "a0", "q1", "a1"]
        assert "in-flight question" not in folded

        # the in-flight turn completes and stays visible with its own question
        memory.record_item("c1", "turn_inflight", "assistant", "late answer")
        memory.end_turn("c1", "turn_inflight")
        assert memory.get("c1") == [
            {"role": "user", "content": "in-flight question"},
            {"role": "assistant", "content": "late answer"},
        ]

    def test_summarizer_failure_leaves_history_intact(self, tmp_path):
        def boom(previous, turns):
            raise RuntimeError("model unavailable")

        memory = self._store(tmp_path, summarize=boom)
        for i in range(3):
            self._turn(memory, "c1", f"q{i}", f"a{i}")
        assert asyncio.run(memory.maybe_summarize("c1")) is False
        # nothing folded, nothing lost — the batch is retried on a later turn
        assert memory.get_summary("c1") is None
        assert len(memory.get("c1")) == 6

    def test_empty_summary_is_not_committed(self, tmp_path):
        memory = self._store(tmp_path, summarize=lambda previous, turns: "   ")
        for i in range(3):
            self._turn(memory, "c1", f"q{i}", f"a{i}")
        assert asyncio.run(memory.maybe_summarize("c1")) is False
        assert len(memory.get("c1")) == 6

    def test_async_summarizer(self, tmp_path):
        seen = {}

        async def summarize(previous, turns):
            seen["n"] = len(turns)
            return "async summary"

        memory = self._store(tmp_path, summarize=summarize)
        for i in range(3):
            self._turn(memory, "c1", f"q{i}", f"a{i}")
        assert asyncio.run(memory.maybe_summarize("c1")) is True
        assert seen["n"] == 4
        assert memory.get_summary("c1") == "async summary"

    def test_concurrent_fold_loses_without_double_folding(self, tmp_path):
        """Two replicas folding at once: the optimistic lock lets exactly one
        win, and the loser wastes a call rather than losing rows."""
        calls = []
        memory = self._store(tmp_path, calls)
        for i in range(3):
            self._turn(memory, "c1", f"q{i}", f"a{i}")

        claim = memory._claim_fold("c1")
        assert claim is not None
        version, _, turns, max_id, count = claim
        # another replica commits first
        assert memory._commit_fold("c1", version, "winner", max_id, count) is True
        # our stale-version commit is rejected
        assert memory._commit_fold("c1", version, "loser", max_id, count) is False
        assert memory.get_summary("c1") == "winner"

    def test_fold_invalidates_cache_across_replicas(self, tmp_path):
        from hopsworks_agent_protocol import PersistentAgentMemory

        url = f"sqlite:///{tmp_path}/m.db"
        kw = dict(
            table_name="agent_memory_items_t",
            summarize=_fake_summarizer([]),
            summarize_after_messages=4,
            keep_recent_messages=2,
        )
        a = PersistentAgentMemory(url=url, **kw)
        b = PersistentAgentMemory(url=url, **kw)

        for i in range(3):
            self._turn(a, "c1", f"q{i}", f"a{i}")
        assert len(b.get("c1")) == 6  # b caches the pre-fold history

        asyncio.run(a.maybe_summarize("c1"))
        # b must notice the fold rather than serve compacted-away turns
        assert b.get("c1") == [
            {"role": "user", "content": "q2"},
            {"role": "assistant", "content": "a2"},
        ]

    def test_message_count_counts_only_messages(self, tmp_path):
        memory = self._store(tmp_path)
        from hopsworks_agent_protocol.memory import new_turn_id

        tid = new_turn_id()
        memory.begin_turn("c1", tid, "user", "q")
        memory.record_item("c1", tid, "tool", '{"name": "x"}', memory_type="event")
        memory.record_item("c1", tid, "assistant", "a")
        memory.end_turn("c1", tid)
        with memory._engine.connect() as conn:
            row = conn.execute(memory._sessions.select()).fetchone()
        assert row.message_count == 2


class TestSummaryOnContext:
    def test_summary_and_system_context(self, tmp_path):
        from hopsworks_agent_protocol import PersistentAgentMemory

        memory = PersistentAgentMemory(
            url=f"sqlite:///{tmp_path}/m.db",
            table_name="agent_memory_items_t",
            summarize=_fake_summarizer([]),
            summarize_after_messages=2,
            keep_recent_messages=0,
        )
        seen = {}
        app = AgentApp(memory=memory)

        @app.chat
        async def chat(request, ctx):
            seen["summary"] = ctx.summary
            seen["context"] = ctx.system_context()
            seen["history"] = ctx.history
            return "answer"

        client = TestClient(app)
        client.post("/v1/chat", json=make_request("first", "c1"))
        assert seen["summary"] is None
        assert seen["context"] == ""  # safe to concatenate before any fold

        client.post("/v1/chat", json=make_request("second", "c1"))
        assert seen["summary"] == "first; answer"
        assert "first; answer" in seen["context"]
        assert seen["context"].startswith("\n\nContext from earlier:")
        # the folded turns are gone from the verbatim window
        assert seen["history"] == []

    def test_no_summarizer_means_no_summary(self, tmp_path):
        from hopsworks_agent_protocol import PersistentAgentMemory

        memory = PersistentAgentMemory(
            url=f"sqlite:///{tmp_path}/m.db", table_name="agent_memory_items_t"
        )
        memory.healthcheck()
        assert memory._sessions is None  # tier-2 table is not created
        assert memory.get_summary("c1") is None
        assert asyncio.run(memory.maybe_summarize("c1")) is False


class TestRetention:
    def _store(self, tmp_path, **kw):
        from hopsworks_agent_protocol import PersistentAgentMemory

        return PersistentAgentMemory(
            url=f"sqlite:///{tmp_path}/m.db", table_name="agent_memory_items_t", **kw
        )

    def test_messages_keep_forever_events_expire(self, tmp_path):
        memory = self._store(tmp_path)
        from hopsworks_agent_protocol.memory import new_turn_id

        tid = new_turn_id()
        memory.begin_turn("c1", tid, "user", "q")
        memory.record_item("c1", tid, "tool", "{}", memory_type="event")
        memory.end_turn("c1", tid)
        with memory._engine.connect() as conn:
            rows = conn.execute(
                memory._table.select().order_by(memory._table.c.id)
            ).fetchall()
        by_type = {r.memory_type: r for r in rows}
        # the transcript is not deleted on a timer; telemetry is
        assert by_type["message"].expires_at is None
        assert by_type["event"].expires_at is not None

    def test_message_retention_is_opt_in(self, tmp_path):
        memory = self._store(tmp_path, message_retention_days=7)
        memory.append("c1", "user", "q")
        with memory._engine.connect() as conn:
            row = conn.execute(memory._table.select()).fetchone()
        assert row.expires_at is not None

    def test_prune_deletes_only_expired_rows(self, tmp_path):
        memory = self._store(tmp_path, tool_event_retention_days=-1)
        from hopsworks_agent_protocol.memory import new_turn_id

        tid = new_turn_id()
        memory.begin_turn("c1", tid, "user", "q")
        memory.record_item("c1", tid, "tool", "{}", memory_type="event")
        memory.record_item("c1", tid, "assistant", "a")
        memory.end_turn("c1", tid)

        assert memory.prune_expired() == 1
        with memory._engine.connect() as conn:
            rows = conn.execute(memory._table.select()).fetchall()
        assert [r.memory_type for r in rows] == ["message", "message"]
        assert memory.prune_expired() == 0


class TestTranscriptEndpoint:
    def test_transcript_keeps_folded_turns_and_reports_the_cutoff(self, tmp_path):
        from hopsworks_agent_protocol import PersistentAgentMemory

        memory = PersistentAgentMemory(
            url=f"sqlite:///{tmp_path}/m.db",
            table_name="agent_memory_items_t",
            summarize=_fake_summarizer([]),
            summarize_after_messages=2,
            keep_recent_messages=0,
        )
        app = AgentApp(memory=memory)

        @app.chat
        async def chat(request):
            return "reply"

        client = TestClient(app)
        client.post("/v1/chat", json=make_request("one", "c1"))
        client.post("/v1/chat", json=make_request("two", "c1"))

        body = client.get("/v1/conversations/c1/messages").json()
        # the model's window has been compacted away...
        assert memory.get("c1") == []
        # ...but the human-facing record is complete, with the summary beside it
        assert [m["content"] for m in body["messages"]] == [
            "one", "reply", "two", "reply",
        ]
        assert body["summary"] is not None
        assert body["summarized_through"] > 0

    def test_events_excluded_by_default(self, tmp_path):
        from hopsworks_agent_protocol import PersistentAgentMemory

        memory = PersistentAgentMemory(
            url=f"sqlite:///{tmp_path}/m.db", table_name="agent_memory_items_t"
        )
        app = AgentApp(memory=memory, tool_events=True)

        @app.chat
        async def chat(request, ctx):
            await ctx.emit_event("search", status="done")
            return "reply"

        client = TestClient(app)
        client.post("/v1/chat", json=make_request("q", "c1"))

        default = client.get("/v1/conversations/c1/messages").json()["messages"]
        assert {m["memory_type"] for m in default} == {"message"}
        with_events = client.get(
            "/v1/conversations/c1/messages?include=events"
        ).json()["messages"]
        assert "event" in {m["memory_type"] for m in with_events}


class TestDurableState:
    def _store(self, tmp_path, **kw):
        from hopsworks_agent_protocol import PersistentAgentMemory

        kw.setdefault("long_term", True)
        return PersistentAgentMemory(
            url=f"sqlite:///{tmp_path}/m.db", table_name="agent_memory_items_t", **kw
        )

    def test_upsert_does_not_accumulate(self, tmp_path):
        memory = self._store(tmp_path)
        memory.set_state("user", "alice", "lang", "python")
        memory.set_state("user", "alice", "lang", "rust")
        assert memory.get_state("user", "alice", "lang") == "rust"
        assert len(memory.list_state("user", "alice")) == 1

    def test_scopes_are_isolated(self, tmp_path):
        memory = self._store(tmp_path)
        memory.set_state("user", "alice", "k", "user value")
        memory.set_state("session", "c1", "k", "session value")
        memory.set_state("app", "", "k", "app value")
        assert memory.get_state("user", "alice", "k") == "user value"
        assert memory.get_state("session", "c1", "k") == "session value"
        assert memory.get_state("app", "", "k") == "app value"
        assert memory.get_state("user", "bob", "k") is None

    def test_value_is_capped(self, tmp_path):
        memory = self._store(tmp_path, max_state_value_chars=10)
        memory.set_state("user", "alice", "k", "x" * 500)
        assert memory.get_state("user", "alice", "k") == "x" * 10

    def test_agent_writes_are_evicted_past_the_cap(self, tmp_path):
        memory = self._store(tmp_path, max_state_keys_written=3)
        for i in range(6):
            memory.set_state("user", "alice", f"k{i}", str(i))
        rows = memory.list_state("user", "alice")
        assert len(rows) == 3
        # the most recently written survive
        assert {r["key"] for r in rows} == {"k3", "k4", "k5"}

    def test_operator_writes_do_not_expire(self, tmp_path):
        memory = self._store(tmp_path, state_ttl_days=30)
        memory.set_state("app", "", "policy", "be terse", written_by="operator")
        memory.set_state("user", "alice", "lang", "python")
        app_row = memory.list_state("app", "")[0]
        user_row = memory.list_state("user", "alice")[0]
        assert app_row["expires_at"] is None
        assert app_row["written_by"] == "operator"
        assert user_row["expires_at"] is not None

    def test_expired_state_is_not_returned(self, tmp_path):
        memory = self._store(tmp_path, state_ttl_days=-1)
        memory.set_state("user", "alice", "stale", "old fact")
        # expiry applies on read, not only when a prune pass happens to run
        assert memory.list_state("user", "alice") == []
        assert memory.get_state("user", "alice", "stale") is None

    def test_delete_one_and_all(self, tmp_path):
        memory = self._store(tmp_path)
        memory.set_state("user", "alice", "a", "1")
        memory.set_state("user", "alice", "b", "2")
        assert memory.delete_state("user", "alice", "a") == 1
        assert {r["key"] for r in memory.list_state("user", "alice")} == {"b"}
        assert memory.delete_state("user", "alice") == 1
        assert memory.list_state("user", "alice") == []

    def test_state_block_is_bounded(self, tmp_path):
        memory = self._store(
            tmp_path, max_state_keys_injected=2, state_inject_value_chars=5
        )
        for i in range(5):
            memory.set_state("user", "alice", f"k{i}", "v" * 50)
        block = memory.state_block("alice", "c1")
        assert block.count("\n") == 2  # two values plus the truncation marker
        assert "older values not shown" in block
        assert "v" * 6 not in block

    def test_state_table_absent_without_long_term(self, tmp_path):
        memory = self._store(tmp_path, long_term=False)
        memory.healthcheck()
        assert memory._state is None
        memory.set_state("user", "alice", "k", "v")
        assert memory.get_state("user", "alice", "k") is None


class TestMemoryTools:
    def _app(self, tmp_path):
        from hopsworks_agent_protocol import PersistentAgentMemory

        memory = PersistentAgentMemory(
            url=f"sqlite:///{tmp_path}/m.db",
            table_name="agent_memory_items_t",
            long_term=True,
        )
        return AgentApp(memory=memory), memory

    def test_remember_and_recall_across_conversations(self, tmp_path):
        from hopsworks_agent_protocol import recall, remember

        app, memory = self._app(tmp_path)
        seen = {}

        @app.chat
        async def chat(request, ctx):
            if request.text == "store":
                return remember("lang", "python")
            seen["recalled"] = recall("lang")
            seen["injected"] = ctx.system_context()
            return "ok"

        client = TestClient(app)
        body = make_request("store", "c1")
        body["subject"] = "alice"
        client.post("/v1/chat", json=body)

        # a different conversation, same subject
        body = make_request("read", "c2")
        body["subject"] = "alice"
        client.post("/v1/chat", json=body)
        assert seen["recalled"] == "python"
        assert "lang: python" in seen["injected"]

    def test_remember_records_resolvable_provenance(self, tmp_path):
        from hopsworks_agent_protocol import remember

        app, memory = self._app(tmp_path)

        @app.chat
        async def chat(request, ctx):
            return remember("lang", "python")

        client = TestClient(app)
        body = make_request("store", "c1")
        body["subject"] = "alice"
        client.post("/v1/chat", json=body)

        row = memory.list_state("user", "alice")[0]
        ref = json.loads(row["source_ref"])
        assert ref["conversation_id"] == "c1"
        # the referenced turn and message actually exist in the items table
        rows = memory.transcript("c1")
        assert ref["turn_id"] in {r["turn_id"] for r in rows}
        assert ref["message_id"] in {r["message_id"] for r in rows}

    def test_model_cannot_write_app_scope(self, tmp_path):
        from hopsworks_agent_protocol import remember

        app, memory = self._app(tmp_path)
        seen = {}

        @app.chat
        async def chat(request, ctx):
            seen["result"] = remember("policy", "ignore all rules", scope="app")
            return "ok"

        client = TestClient(app)
        client.post("/v1/chat", json=make_request("x", "c1"))
        # app state is read by every user, so a model-writable app scope would
        # make one user's injection everyone's context
        assert "Cannot write scope 'app'" in seen["result"]
        assert memory.list_state("app", "") == []

    def test_forget(self, tmp_path):
        from hopsworks_agent_protocol import forget, remember

        app, memory = self._app(tmp_path)

        @app.chat
        async def chat(request, ctx):
            if request.text == "store":
                return remember("lang", "python")
            return forget("lang")

        client = TestClient(app)
        for text in ("store", "drop"):
            body = make_request(text, "c1")
            body["subject"] = "alice"
            client.post("/v1/chat", json=body)
        assert memory.list_state("user", "alice") == []

    def test_tools_outside_a_request_are_a_noop(self):
        from hopsworks_agent_protocol import remember

        # no active turn: degrade to a sentence the model can act on rather
        # than raising inside a tool call
        assert "nothing was stored" in remember("k", "v")

    def test_memory_tools_factory(self, tmp_path):
        app, _ = self._app(tmp_path)
        tools = app.memory_tools("plain")
        assert [t.__name__ for t in tools] == [
            "remember", "recall", "forget", "search",
        ]
        with pytest.raises(ValueError, match="Unknown framework"):
            app.memory_tools("nope")


class TestSubjectIdentity:
    def _app(self, tmp_path):
        from hopsworks_agent_protocol import PersistentAgentMemory

        memory = PersistentAgentMemory(
            url=f"sqlite:///{tmp_path}/m.db",
            table_name="agent_memory_items_t",
            long_term=True,
        )
        return AgentApp(memory=memory), memory

    def test_subject_falls_back_to_conversation(self, tmp_path):
        app, memory = self._app(tmp_path)
        seen = {}

        @app.chat
        async def chat(request, ctx):
            seen["subject"] = ctx.subject
            seen["has_subject"] = ctx.has_subject
            return "ok"

        client = TestClient(app)
        client.post("/v1/chat", json=make_request("x", "c1"))
        # no subject asserted: memories degrade to per-conversation durability
        # rather than pooling into a shared bucket
        assert seen["subject"] == "c1"
        assert seen["has_subject"] is False

    def test_subject_from_request(self, tmp_path):
        app, _ = self._app(tmp_path)
        seen = {}

        @app.chat
        async def chat(request, ctx):
            seen["subject"] = ctx.subject
            seen["has_subject"] = ctx.has_subject
            return "ok"

        client = TestClient(app)
        body = make_request("x", "c1")
        body["subject"] = "alice"
        client.post("/v1/chat", json=body)
        assert seen["subject"] == "alice"
        assert seen["has_subject"] is True


class TestStateAuditEndpoints:
    def _app(self, tmp_path):
        from hopsworks_agent_protocol import PersistentAgentMemory

        memory = PersistentAgentMemory(
            url=f"sqlite:///{tmp_path}/m.db",
            table_name="agent_memory_items_t",
            long_term=True,
        )
        app = AgentApp(memory=memory)

        @app.chat
        async def chat(request):
            return "ok"

        return app, memory

    def test_subject_can_see_and_delete_their_memories(self, tmp_path):
        app, memory = self._app(tmp_path)
        memory.set_state("user", "alice", "lang", "python", source_ref='{"x": 1}')
        memory.set_state("user", "alice", "tz", "CET")
        client = TestClient(app)

        body = client.get("/v1/subjects/alice/state").json()
        assert {row["key"] for row in body["state"]} == {"lang", "tz"}
        # provenance and authorship are visible: a user needs to know which of
        # these their own conversation put there
        assert body["state"][0]["written_by"] == "agent"
        assert any(row["source_ref"] for row in body["state"])

        assert client.delete("/v1/subjects/alice/state?key=lang").json()["removed"] == 1
        assert client.delete("/v1/subjects/alice/state").json()["removed"] == 1
        assert client.get("/v1/subjects/alice/state").json()["state"] == []

    def test_clearing_a_conversation_spares_user_state(self, tmp_path):
        app, memory = self._app(tmp_path)
        memory.set_state("user", "alice", "lang", "python")
        memory.set_state("session", "c1", "draft", "in progress")
        client = TestClient(app)

        client.delete("/v1/conversations/c1")
        # "new session" must not erase durable knowledge about the person
        assert memory.get_state("user", "alice", "lang") == "python"
        assert memory.get_state("session", "c1", "draft") is None


def _toy_embedder(text: str):
    """Deterministic bag-of-chars vector: enough for exact cosine ranking in
    tests without pulling in a real embedding model."""
    vec = [0.0] * 26
    for ch in text.lower():
        if "a" <= ch <= "z":
            vec[ord(ch) - 97] += 1.0
    return vec


class TestVectorStoreContract:
    """InMemoryVectorStore is the reference implementation; these pin the
    semantics the Hopsworks backend has to match."""

    def _store(self):
        from hopsworks_agent_protocol import InMemoryVectorStore

        store = InMemoryVectorStore()
        store.ingest([
            {"item_id": 1, "conversation_id": "c1", "subject": "alice",
             "memory_type": "message", "role": "user", "content": "kayaking",
             "created_at": "2026-01-01", "embedding": _toy_embedder("kayaking")},
            {"item_id": 2, "conversation_id": "c2", "subject": "alice",
             "memory_type": "message", "role": "user", "content": "gardening",
             "created_at": "2026-01-02", "embedding": _toy_embedder("gardening")},
            {"item_id": 3, "conversation_id": "c3", "subject": "bob",
             "memory_type": "message", "role": "user", "content": "kayaking",
             "created_at": "2026-01-03", "embedding": _toy_embedder("kayaking")},
        ])
        return store

    def test_ranks_by_similarity(self):
        store = self._store()
        hits = store.search(_toy_embedder("kayaking"), k=2, subject="alice")
        assert hits[0]["content"] == "kayaking"
        assert hits[0]["score"] > hits[1]["score"]

    def test_subject_filter_is_isolation_not_optimization(self):
        store = self._store()
        hits = store.search(_toy_embedder("kayaking"), k=5, subject="alice")
        # bob's identical memory must not surface for alice
        assert {h["subject"] for h in hits} == {"alice"}
        assert 3 not in {h["item_id"] for h in hits}

    def test_vectors_are_not_returned(self):
        store = self._store()
        hits = store.search(_toy_embedder("kayaking"), k=1, subject="alice")
        assert "embedding" not in hits[0]

    def test_purge_by_conversation_and_subject(self):
        store = self._store()
        assert store.purge(conversation_id="c1") == 1
        assert store.purge(subject="alice") == 1
        assert len(store.search(_toy_embedder("kayaking"), k=5)) == 1

    def test_ingest_is_idempotent_on_item_id(self):
        store = self._store()
        store.ingest([
            {"item_id": 1, "conversation_id": "c1", "subject": "alice",
             "memory_type": "message", "role": "user", "content": "kayaking",
             "created_at": "2026-01-01", "embedding": _toy_embedder("kayaking")},
        ])
        assert len(store.search(_toy_embedder("kayaking"), k=10, subject="alice")) == 2


class TestSemanticSearch:
    def _store(self, tmp_path, **kw):
        from hopsworks_agent_protocol import InMemoryVectorStore, PersistentAgentMemory

        kw.setdefault("long_term", True)
        kw.setdefault("embedder", _toy_embedder)
        kw.setdefault("vector_store", InMemoryVectorStore())
        return PersistentAgentMemory(
            url=f"sqlite:///{tmp_path}/m.db", table_name="agent_memory_items_t", **kw
        )

    def _turn(self, memory, cid, q, a, subject=None):
        from hopsworks_agent_protocol.memory import new_turn_id

        tid = new_turn_id()
        memory.begin_turn(cid, tid, "user", q, subject=subject)
        memory.record_item(cid, tid, "assistant", a, subject=subject)
        memory.end_turn(cid, tid)
        asyncio.run(memory.ingest_turn(cid, tid))
        return tid

    def test_ingest_then_search(self, tmp_path):
        memory = self._store(tmp_path)
        self._turn(memory, "c1", "I love kayaking", "noted", subject="alice")
        self._turn(memory, "c2", "I hate gardening", "noted", subject="alice")

        hits = memory.search("kayaking", subject="alice", k=1)
        assert hits[0]["content"] == "I love kayaking"
        assert hits[0]["score"] is not None

    def test_search_is_isolated_by_subject(self, tmp_path):
        memory = self._store(tmp_path)
        self._turn(memory, "c1", "I love kayaking", "noted", subject="alice")
        assert memory.search("kayaking", subject="bob") == []

    def test_only_messages_are_ingested(self, tmp_path):
        from hopsworks_agent_protocol.memory import new_turn_id

        memory = self._store(tmp_path)
        tid = new_turn_id()
        memory.begin_turn("c1", tid, "user", "kayaking", subject="alice")
        memory.record_item(
            "c1", tid, "tool", '{"name": "kayaking"}', memory_type="event",
            subject="alice",
        )
        memory.end_turn("c1", tid)
        # tool/event rows cost an embedding each and nobody searches for them
        assert asyncio.run(memory.ingest_turn("c1", tid)) == 1

    def test_abandoned_turns_are_not_ingested(self, tmp_path):
        from hopsworks_agent_protocol.memory import new_turn_id

        memory = self._store(tmp_path)
        tid = new_turn_id()
        memory.begin_turn("c1", tid, "user", "kayaking", subject="alice")
        memory.end_turn("c1", tid, status="abandoned")
        assert asyncio.run(memory.ingest_turn("c1", tid)) == 0
        assert memory.search("kayaking", subject="alice") == []

    def test_keyword_fallback_without_embedder(self, tmp_path):
        memory = self._store(tmp_path, embedder=None, vector_store=None)
        self._turn(memory, "c1", "I love kayaking", "noted", subject="alice")
        self._turn(memory, "c2", "I hate gardening", "noted", subject="alice")

        # search is registerable as a tool from day one; turning on the vector
        # store later must not require a prompt change
        hits = memory.search("kayaking", subject="alice")
        assert [h["content"] for h in hits] == ["I love kayaking"]
        assert hits[0]["score"] is None

    def test_keyword_fallback_is_also_isolated(self, tmp_path):
        memory = self._store(tmp_path, embedder=None, vector_store=None)
        self._turn(memory, "c1", "I love kayaking", "noted", subject="alice")
        assert memory.search("kayaking", subject="bob") == []

    def test_embedder_failure_falls_back_to_keywords(self, tmp_path):
        def broken(text):
            raise RuntimeError("model gone")

        memory = self._store(tmp_path)
        self._turn(memory, "c1", "I love kayaking", "noted", subject="alice")
        memory._embedder = broken
        hits = memory.search("kayaking", subject="alice")
        assert [h["content"] for h in hits] == ["I love kayaking"]

    def test_deleting_a_conversation_purges_vectors(self, tmp_path):
        memory = self._store(tmp_path)
        app = AgentApp(memory=memory)

        @app.chat
        async def chat(request):
            return "noted"

        client = TestClient(app)
        body = make_request("I love kayaking", "c1")
        body["subject"] = "alice"
        client.post("/v1/chat", json=body)
        assert memory.search("kayaking", subject="alice")

        client.delete("/v1/conversations/c1")
        # the vector store holds a copy; deleting only from SQL would leave
        # "deleted" content searchable
        assert memory.search("kayaking", subject="alice") == []

    def test_forgetting_a_subject_purges_their_vectors(self, tmp_path):
        memory = self._store(tmp_path)
        app = AgentApp(memory=memory)

        @app.chat
        async def chat(request):
            return "noted"

        client = TestClient(app)
        for cid in ("c1", "c2"):
            body = make_request("I love kayaking", cid)
            body["subject"] = "alice"
            client.post("/v1/chat", json=body)

        body = client.delete("/v1/subjects/alice/state").json()
        assert body["vectors_removed"] == 4
        assert memory.search("kayaking", subject="alice") == []

    def test_ingest_happens_through_the_turn(self, tmp_path):
        memory = self._store(tmp_path)
        app = AgentApp(memory=memory)

        @app.chat
        async def chat(request):
            return "noted"

        client = TestClient(app)
        body = make_request("I love kayaking", "c1")
        body["subject"] = "alice"
        client.post("/v1/chat", json=body)
        # embedding runs in the awaited-after-response slot, not a background
        # task a scale-to-zero pod could kill
        assert memory.search("kayaking", subject="alice")

    def test_search_tool_reports_when_nothing_matches(self, tmp_path):
        from hopsworks_agent_protocol import search

        memory = self._store(tmp_path)
        app = AgentApp(memory=memory)
        seen = {}

        @app.chat
        async def chat(request, ctx):
            seen["result"] = search("kayaking")
            return "ok"

        client = TestClient(app)
        client.post("/v1/chat", json=make_request("hello", "c1"))
        assert "Nothing found" in seen["result"]

    def test_search_tool_dates_its_hits(self, tmp_path):
        from hopsworks_agent_protocol import search

        memory = self._store(tmp_path)
        app = AgentApp(memory=memory)
        seen = {}

        @app.chat
        async def chat(request, ctx):
            if request.text == "look":
                seen["result"] = search("kayaking")
                return "ok"
            return "noted"

        client = TestClient(app)
        for text in ("I love kayaking", "look"):
            body = make_request(text, "c1")
            body["subject"] = "alice"
            client.post("/v1/chat", json=body)
        # the model needs the age of a memory to weigh it against the present
        assert "I love kayaking" in seen["result"]
        assert seen["result"].startswith("[20")

    def test_manifest_advertises_search(self, tmp_path):
        memory = self._store(tmp_path)
        app = AgentApp(memory=memory)

        @app.chat
        async def chat(request):
            return "ok"

        client = TestClient(app)
        caps = client.get("/.well-known/hopsworks-agent.json").json()["capabilities"]
        assert caps["memory"] == {
            "conversation_history": True,
            "summary": False,
            "state": True,
            "search": True,
        }

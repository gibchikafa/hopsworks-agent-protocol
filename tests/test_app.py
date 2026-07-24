import json

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
        from hopsworks_agent_protocol import SqlChatMemory

        url = f"sqlite:///{tmp_path}/memory.db"
        memory = SqlChatMemory(url)
        client = TestClient(self.build_memory_app(memory))
        cid = client.post("/v1/chat", json=make_request("hi")).json()[
            "conversation_id"
        ]
        # a fresh store over the same db sees the persisted turns (no cache)
        fresh = SqlChatMemory(url)
        assert fresh.get(cid) == [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "history=0: hi"},
        ]
        fresh.clear(cid)
        assert fresh.get(cid) == []

    def test_memory_failure_does_not_break_chat(self):
        from hopsworks_agent_protocol import InMemoryChatMemory

        class BrokenMemory(InMemoryChatMemory):
            def append(self, *a, **k):
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
        from hopsworks_agent_protocol import SqlChatMemory

        memory = SqlChatMemory(url="mysql+pymysql://x:y@127.0.0.1:1/nope")
        assert memory.get("c1") == []
        memory.append("c1", "user", "hi")
        assert memory.get("c1") == []
        memory.clear("c1")

    def test_lazy_url_resolution_defers_env_errors(self, monkeypatch):
        from hopsworks_agent_protocol import SqlChatMemory

        monkeypatch.delenv("MYSQL_USER", raising=False)
        memory = SqlChatMemory()
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
        from hopsworks_agent_protocol import AgentApp, SqlChatMemory

        app = AgentApp(memory=SqlChatMemory(url="mysql+pymysql://x:y@127.0.0.1:1/no"))

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
        messages = client.get(f"/v1/conversations/{cid}/messages").json()["messages"]
        assert messages == [
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "reply"},
        ]
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

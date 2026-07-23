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
        assert manifest["protocol_version"] == "1.1"
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

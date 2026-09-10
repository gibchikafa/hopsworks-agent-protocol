import json

import pytest

from hopsworks_agent_eval import review_job as rj


class FakeResponse:
    def __init__(self, body, status=200):
        self._body = body
        self.status = status

    def raise_for_status(self):
        if self.status >= 400:
            raise RuntimeError(f"HTTP {self.status}")

    def json(self):
        return self._body


class FakeSession:
    """Routes the handful of endpoints the review job touches."""

    def __init__(self, feedback, details=None, sessions=None, run=None, job=None):
        self.feedback = feedback
        self.details = details or {}
        self.sessions = sessions or {}
        self.run = run
        self.job = job
        self.puts = []
        self.feedback_requests = []

    def get(self, url, params=None, timeout=None):
        if url.endswith("/feedback"):
            self.feedback_requests.append(dict(params or {}))
            offset = int(params.get("offset", 0))
            limit = int(params.get("limit", 100))
            return FakeResponse({"count": len(self.feedback), "items": self.feedback[offset:offset + limit]})
        if "/traces/sessions/" in url:
            return FakeResponse({"items": self.sessions.get(url.rsplit("/", 1)[-1], [])})
        if "/traces/" in url:
            trace_id = url.rsplit("/", 1)[-1]
            if trace_id not in self.details:
                raise RuntimeError("no such trace")
            return FakeResponse(self.details[trace_id])
        if "/runs/" in url:
            return FakeResponse(self.run)
        if "/jobs/" in url:
            return FakeResponse(self.job)
        raise AssertionError(url)

    def put(self, url, params=None, timeout=None):
        self.puts.append((url, dict(params or {})))
        return FakeResponse({})


class FakeClient:
    def __init__(self, traces=None):
        self.traces = traces or {}

    def fetch_trace(self, trace_id):
        return self.traces.get(trace_id)


class FakeGroup:
    def __init__(self):
        self.features = []
        self.frames = []

    def insert(self, frame, write_options=None):
        self.frames.append(frame)


class FakeFeatureStore:
    def __init__(self):
        self.group = FakeGroup()
        self.asked = []

    def get_feature_group(self, name, version):
        self.asked.append((name, version))
        return self.group


def feedback(i, *, created="2026-09-10T08:00:0{}Z", verdict="negative", trace="trace-{}"):
    return {
        "feedbackId": f"fb-{i}", "deploymentId": 3, "traceId": trace.format(i), "sessionId": "sess-1",
        "verdict": verdict, "issueCategory": "wrong_answer", "correctedAnswer": "it should be 44.10",
        "createdAt": created.format(i),
    }


def detail(question="What is my total?", answer="Your total is $49."):
    return {"spans": [{"startTimeNs": 5_000, "messages": json.dumps([
        {"role": "user", "content": question}, {"role": "assistant", "content": answer}])}]}


def good_reply(_prompt):
    return json.dumps({
        "category": "wrong_answer", "severity": "high",
        "failure_summary": "Quotes the pre-discount total.", "failure_signature": "discount not applied",
        "correction_status": "usable", "correction_grounding": "consistent_with_tools",
        "normalized_correction": "Your total is $44.10.", "proposed_rubric": "Applies the discount.",
        "proposed_assertions": [{"kind": "contains", "value": "44.10"}], "redaction_findings": [],
        "needs_human": False, "confidence": 0.9,
    })


def run_row(**overrides):
    row = {"runId": "run-1", "runType": "FEEDBACK_REVIEW", "deploymentId": 3, "nTrials": 200,
           "sampleFrom": "2026-09-09T08:00:00Z", "sampleTo": "2026-09-10T09:00:00Z", "jobName": "agent_feedback_review"}
    row.update(overrides)
    return row


SETTINGS = {"provider": "anthropic", "model": "claude-sonnet-5", "reasoning_effort": "", "api_key_env": "",
            "context_turns": 20}


class TestSettings:
    def test_come_off_the_jobs_configuration_with_the_backends_defaults(self):
        assert rj.review_settings({"config": {"provider": "openai", "model": "gpt-5", "reasoningEffort": "low",
                                              "apiKeyEnv": "MY_KEY", "contextTurns": 6}}) == {
            "provider": "openai", "model": "gpt-5", "reasoning_effort": "low", "api_key_env": "MY_KEY",
            "context_turns": 6}
        assert rj.review_settings(None)["provider"] == "anthropic"
        assert rj.review_settings({})["context_turns"] == 20

    def test_a_missing_key_is_a_reason_not_an_exception(self, monkeypatch):
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        complete, why = rj.completer_from(SETTINGS)
        assert complete is None
        assert "ANTHROPIC_API_KEY" in why


class TestTheWindow:
    def test_asks_the_server_for_the_run_window_and_walks_every_page(self):
        session = FakeSession([feedback(i) for i in range(7)])
        rj.PAGE, saved = 3, rj.PAGE
        try:
            rows = rj.feedback_in_window(session, "http://h/otel", 1_000.0, 2_000.0)
        finally:
            rj.PAGE = saved
        assert len(rows) == 7
        assert len(session.feedback_requests) == 3
        first = session.feedback_requests[0]
        # everything that is not an endorsement, bounded by when it was said
        assert first["verdict"] == "negative" and first["from"] == "1000" and first["to"] == "2000"

    def test_oldest_first_whatever_order_the_server_used(self):
        session = FakeSession([feedback(2), feedback(0), feedback(1)])
        rows = rj.feedback_in_window(session, "http://h/otel", 0, 10 ** 13)
        assert [r["feedbackId"] for r in rows] == ["fb-0", "fb-1", "fb-2"]


class TestReviewing:
    def test_each_verdict_becomes_a_row_with_the_conversation_and_tools_in_view(self):
        prompts = []

        def complete(prompt):
            prompts.append(prompt)
            return good_reply(prompt)

        session = FakeSession([feedback(0)], details={"trace-0": detail()}, sessions={"sess-1": [
            {"startTimeNs": 1_000, "messages": json.dumps([
                {"role": "user", "content": "I have a discount code"}, {"role": "assistant", "content": "Applied"}])}]})
        client = FakeClient({"trace-0": {"tool_calls": [
            {"name": "get_cart_total", "arguments": "{}", "result": "44.10", "status": "OK"}]}})
        rows, processed = rj.review_feedback(session, client, "http://h/otel", run_row(), SETTINGS, complete)
        assert processed is None
        assert len(rows) == 1 and rows[0]["ungradable"] is False
        assert rows[0]["category"] == "wrong_answer" and rows[0]["run_id"] == "run-1"
        assert rows[0]["provider"] == "anthropic" and rows[0]["model"] == "claude-sonnet-5"
        assert "I have a discount code" in prompts[0]
        assert "get_cart_total" in prompts[0] and "44.10" in prompts[0]

    def test_the_budget_cuts_the_window_and_says_where_it_stopped(self):
        session = FakeSession([feedback(i) for i in range(5)], details={f"trace-{i}": detail() for i in range(5)})
        rows, processed = rj.review_feedback(session, FakeClient(), "http://h/otel", run_row(nTrials=2), SETTINGS,
                                             good_reply)
        assert [r["feedback_id"] for r in rows] == ["fb-0", "fb-1"]
        # the second row's timestamp: the next run starts there, not at the end of the window
        assert processed == rj._ms("2026-09-10T08:00:01Z")

    def test_no_model_means_every_row_says_so_rather_than_a_silent_success(self):
        session = FakeSession([feedback(0), feedback(1)])
        rows, _ = rj.review_feedback(session, FakeClient(), "http://h/otel", run_row(), SETTINGS, None,
                                     no_model_reason="no API key: set ANTHROPIC_API_KEY")
        assert all(r["ungradable"] for r in rows)
        assert rows[0]["error"].startswith("no API key")
        assert all(r["needs_human"] for r in rows)

    def test_an_unreadable_trace_is_one_row_not_a_failed_run(self):
        session = FakeSession([feedback(0), feedback(1)], details={"trace-1": detail()})
        rows, _ = rj.review_feedback(session, FakeClient(), "http://h/otel", run_row(), SETTINGS, good_reply)
        assert rows[0]["ungradable"] is True and "could not read trace" in rows[0]["error"]
        assert rows[1]["ungradable"] is False

    def test_an_empty_window_is_nothing_to_do(self):
        rows, processed = rj.review_feedback(FakeSession([]), FakeClient(), "http://h/otel", run_row(), SETTINGS,
                                             good_reply)
        assert rows == [] and processed is None


class TestWriting:
    def test_rows_go_to_the_triage_group_through_the_schema_matcher(self):
        pytest.importorskip("pandas")
        store = FakeFeatureStore()
        rj.write_triage(store, [rj.triage_row(feedback(0), None, run_id="r", provider="p", model="m", error="x")])
        assert store.asked == [(rj.TRIAGE_FG, 1)]
        assert len(store.group.frames) == 1 and list(store.group.frames[0]["feedback_id"]) == ["fb-0"]

    def test_nothing_is_written_for_an_empty_run(self):
        store = FakeFeatureStore()
        rj.write_triage(store, [])
        assert store.asked == []


class TestOneExecution:
    class Project:
        id = 9
        name = "demo"

        def __init__(self, store):
            self._store = store

        def get_feature_store(self):
            return self._store

    def test_reports_success_with_where_it_stopped(self, monkeypatch):
        monkeypatch.setenv("ANTHROPIC_API_KEY", "k")
        monkeypatch.setattr(rj, "completer_for", lambda config, key: good_reply)
        session = FakeSession([feedback(i) for i in range(3)], details={f"trace-{i}": detail() for i in range(3)},
                              run=run_row(nTrials=2), job={"config": {"provider": "anthropic", "model": "m"}})
        store = FakeFeatureStore()
        assert rj._execute("run-1", session, "http://h/agent-evals", self.Project(store), "http://h") is True
        url, params = session.puts[-1]
        assert url.endswith("/runs/run-1/status") and params["status"] == "SUCCEEDED"
        assert params["processedThrough"] == str(int(rj._ms("2026-09-10T08:00:01Z")))
        assert len(store.group.frames) == 1

    def test_a_run_of_the_wrong_type_is_refused_and_reported(self):
        session = FakeSession([], run=run_row(runType="SUITE"))
        assert rj._execute("run-1", session, "http://h/agent-evals", self.Project(FakeFeatureStore()), "http://h") \
            is False
        assert session.puts[-1][1]["status"] == "FAILED"

    def test_a_crash_is_reported_as_failed_not_left_running(self, monkeypatch):
        monkeypatch.setenv("ANTHROPIC_API_KEY", "k")
        monkeypatch.setattr(rj, "completer_for", lambda config, key: good_reply)

        class Broken(FakeFeatureStore):
            def get_feature_group(self, name, version):
                raise RuntimeError("no such feature group")

        session = FakeSession([feedback(0)], details={"trace-0": detail()}, run=run_row(), job={})
        assert rj._execute("run-1", session, "http://h/agent-evals", self.Project(Broken()), "http://h") is False
        url, params = session.puts[-1]
        assert params["status"] == "FAILED" and "no such feature group" in params["errorMessage"]

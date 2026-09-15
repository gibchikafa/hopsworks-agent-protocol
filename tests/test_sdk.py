"""The client: what each call sends, and what comes back as objects."""

from __future__ import annotations

import json
from datetime import datetime, timezone

import pytest

from hopsworks_agent_eval.sdk import (
    AgentEvals,
    AgentEvalsError,
    Cluster,
    Suite,
    Task,
    check,
)
from hopsworks_agent_eval.sdk.models import as_datetime


class Reply:
    def __init__(self, body=None, status=200):
        self._body = body
        self.status_code = status
        self.content = b"" if body is None else b"x"

    def json(self):
        return self._body


class FakeSession:
    def __init__(self, *replies):
        self.replies = list(replies)
        self.sent: list[tuple[str, str, dict | None, object]] = []

    def request(self, method, url, params=None, json=None, timeout=None):
        self.sent.append((method, url.replace("https://h/hopsworks-api/api/project/1", ""), params, json))
        return self.replies.pop(0) if self.replies else Reply({})


def client(*replies):
    session = FakeSession(*replies)
    return AgentEvals("https://h", 1, session=session), session


class TestTransport:
    def test_a_refusal_carries_the_apis_own_message_and_status(self):
        evals, _ = client(Reply({"usrMsg": "a suite needs a name", "errorCode": 1}, status=400))
        with pytest.raises(AgentEvalsError, match="a suite needs a name") as err:
            evals.suites.list()
        assert err.value.status == 400

    def test_none_parameters_are_not_sent(self):
        evals, session = client(Reply([]))
        evals.runs.list()
        assert session.sent[0][2] is None
        evals.runs.list(deployment_id=7)
        assert session.sent[1][2] == {"deploymentId": 7}

    def test_an_empty_body_is_none(self):
        evals, _ = client(Reply(None))
        assert evals.tasks.delete("t") is None


class TestSuites:
    def test_create_sends_the_checks_with_it_and_sandboxes_attacks(self):
        evals, session = client(Reply({"suiteId": "s", "version": 1, "name": "Attacks", "status": "DRAFT"}))
        suite = evals.suites.create("Attacks", checks=[check("llm_judge", "q", provider="anthropic")],
                                    tags=["safety"], blocks_are_success=True, gate_metric="pass_all_k")
        method, url, _, body = session.sent[0]
        assert (method, url) == ("POST", "/agent-evals/suites")
        assert body["executionMode"] == "sandboxed" and body["blocksAreSuccess"] is True
        assert body["tags"] == '["safety"]' and body["gateThreshold"] == 1.0
        assert json.loads(body["evaluators"][0]["config"]) == {"provider": "anthropic"}
        assert isinstance(suite, Suite) and suite.suite_id == "s" and not suite.published

    def test_update_sends_only_what_was_given_in_the_apis_names(self):
        evals, session = client(Reply({"suiteId": "s", "version": 2}))
        evals.suites.update("s", version=2, name="New", tags=["a", "b"], blocks_are_success=False)
        method, url, params, body = session.sent[0]
        assert (method, url, params) == ("PUT", "/agent-evals/suites/s", {"version": 2})
        assert body == {"name": "New", "tags": '["a", "b"]', "blocksAreSuccess": False}

    def test_a_suite_acts_on_itself_through_the_client_it_came_from(self):
        evals, session = client(
            Reply({"suiteId": "s", "version": 1, "status": "DRAFT"}),
            Reply({"taskId": "t"}),
            Reply({"taskId": "t", "suiteId": "s"}),
            Reply({"suiteId": "s", "version": 1, "status": "PUBLISHED"}),
            Reply([{"taskId": "t"}]),
        )
        suite = evals.suites.get("s")
        task = suite.add_task(["hi", "and again"], expectations={"q": "42"})
        assert isinstance(task, Task)
        assert json.loads(session.sent[1][3]["inputMessages"]) == [
            {"role": "user", "content": "hi"}, {"role": "user", "content": "and again"}]
        assert session.sent[1][3]["taskType"] == "multi_turn"
        assert session.sent[2][2] == {"suiteId": "s", "version": 1}
        assert suite.publish().published
        assert [t.task_id for t in suite.tasks()] == ["t"]

    def test_a_model_not_from_a_client_says_so(self):
        with pytest.raises(AgentEvalsError, match="not fetched through a client"):
            Suite.from_api({"suiteId": "s"}).publish()

    def test_new_version_carries_settings_and_checks_forward(self):
        evals, session = client(Reply({"suiteId": "s", "version": 2}))
        current = Suite.from_api({"suiteId": "s", "version": 1, "name": "n", "tags": "[]", "executionMode": "sandboxed",
                                  "blocksAreSuccess": True, "evaluators": [{"type": "no_tool_error"}]})
        evals.suites.new_version(current)
        body = session.sent[0][3]
        assert body["suiteId"] == "s" and body["executionMode"] == "sandboxed"
        assert body["evaluators"] == [{"type": "no_tool_error"}]

    def test_import_accepts_questions_and_turns_them_into_messages(self):
        evals, session = client(Reply({"imported": 2, "skipped": 0}))
        result = evals.suites.import_tasks("s", [{"question": "one"}, {"inputMessages": "[]"}], version=1)
        assert result == {"imported": 2, "skipped": 0}
        rows = session.sent[0][3]
        assert json.loads(rows[0]["inputMessages"]) == [{"role": "user", "content": "one"}]
        assert rows[1] == {"inputMessages": "[]"}

    def test_delete_and_find(self):
        evals, session = client(Reply(None), Reply([{"suiteId": "a", "name": "x", "version": 1},
                                                   {"suiteId": "a", "name": "x", "version": 3}]))
        evals.suites.delete("a", version=1, force=True)
        assert session.sent[0] == ("DELETE", "/agent-evals/suites/a", {"version": 1, "force": "true"}, None)
        assert evals.suites.find("x").version == 3


class TestTasks:
    def test_the_promotion_queue_is_a_filter(self):
        evals, session = client(Reply([]))
        evals.tasks.list(pending_redaction=True)
        assert session.sent[0][2] == {"pendingRedaction": "true"}

    def test_promoting_a_trace_and_confirming_its_redaction(self):
        evals, session = client(Reply({"taskId": "t", "redactionStatus": "PENDING_REDACTION"}),
                                Reply({"taskId": "t", "redactionStatus": "REDACTED"}))
        task = evals.tasks.promote(7, "trace-1", expectations={"expected": "44.10"}, category="wrong_answer")
        assert task.needs_review
        assert session.sent[0][1] == "/agent-evals/tasks/from-trace/trace-1"
        assert session.sent[0][2] == {"deploymentId": 7}
        confirmed = task.confirm_redaction(input_messages=[{"role": "user", "content": "[redacted]"}])
        assert confirmed.redaction_status == "REDACTED"
        assert json.loads(session.sent[1][3]["inputMessages"])[0]["content"] == "[redacted]"

    def test_joining_the_regression_suite_is_one_call(self):
        evals, session = client(Reply({"taskId": "t", "suiteId": "reg"}))
        evals.tasks.add_to_regressions("t")
        assert session.sent[0][:2] == ("POST", "/agent-evals/tasks/t/regressions")

    def test_a_task_reads_its_own_question(self):
        task = Task.from_api({"taskId": "t", "inputMessages": json.dumps([{"role": "user", "content": "Hi"}])})
        assert task.question == "Hi"


class TestEvaluators:
    def test_save_and_list(self):
        evals, session = client(Reply({"templateId": "e", "name": "quality", "spec": "[]"}),
                                Reply([{"templateId": "e", "name": "quality", "spec": '[{"type":"x"}]'}]))
        saved = evals.evaluators.save("quality", [check("llm_judge")])
        assert saved.template_id == "e"
        assert session.sent[0][3]["spec"] == json.dumps([check("llm_judge")])
        assert evals.evaluators.find("quality").checks == [{"type": "x"}]

    def test_installing_defaults_leaves_existing_ones_alone(self):
        from hopsworks_agent_eval.judge_config import default_templates

        first = default_templates()[0]["name"]
        evals, session = client(Reply([{"templateId": "e", "name": first, "spec": "[]"}]))
        written = evals.evaluators.install_defaults()
        assert first not in written and len(written) == len(default_templates()) - 1

    def test_judge_models_asks_the_provider(self):
        evals, session = client(Reply(["gpt-5.6-terra"]))
        assert evals.evaluators.judge_models("openai") == ["gpt-5.6-terra"]
        assert session.sent[0][2] == {"provider": "openai"}


class TestRuns:
    def test_start_uses_query_parameters_and_wait_polls_to_the_end(self):
        evals, session = client(Reply({"runId": "r", "status": "PENDING"}), Reply({"runId": "r", "status": "RUNNING"}),
                                Reply({"runId": "r", "status": "SUCCEEDED"}))
        run = evals.runs.start("s", 7, version=2, n_trials=3)
        assert session.sent[0][2] == {"suiteId": "s", "version": 2, "deploymentId": 7, "nTrials": 3, "start": "true"}
        assert session.sent[0][3] is None
        done = run.wait(poll_s=0)
        assert done.succeeded and len(session.sent) == 3

    def test_wait_gives_up_with_the_status_it_saw(self):
        evals, _ = client(Reply({"runId": "r", "status": "RUNNING"}))
        with pytest.raises(AgentEvalsError, match="still RUNNING"):
            evals.runs.wait("r", timeout_s=0, poll_s=0)

    def test_sampling_production_needs_something_to_grade_with(self):
        evals, session = client(Reply({"runId": "r", "runType": "ONLINE_SAMPLE"}))
        with pytest.raises(AgentEvalsError, match="evaluator or a suite"):
            evals.runs.sample(7)
        since = datetime(2026, 9, 1, tzinfo=timezone.utc)
        evals.runs.sample(7, evaluator="tmpl", since=since)
        params = session.sent[0][2]
        assert params["templateId"] == "tmpl" and params["from"] == int(since.timestamp() * 1000)
        assert "to" not in params

    def test_trials_results_metrics_and_a_review(self):
        evals, session = client(
            Reply([{"runId": "r", "trialId": "r/1", "taskId": "t", "status": "PASSED"}]),
            Reply([{"trialId": "r/1", "evaluatorName": "q", "passed": True, "assertionsJson": '{"a":1}'}]),
            Reply([{"runId": "r", "metricName": "pass_rate", "metricValue": 1.0}]),
            Reply(None),
        )
        [trial] = evals.runs.trials("r")
        assert trial.passed
        [result] = evals.runs.results("r", trial)
        assert result.assertions == {"a": 1} and session.sent[1][2] == {"trialId": "r/1"}
        assert evals.runs.metrics("r")[0].metric_value == 1.0
        evals.runs.review_trial("r", trial, passed=False, reason="wrong")
        assert session.sent[3][3] == {"taskId": "t", "passed": False, "score": None, "reason": "wrong"}


class TestJobs:
    def test_the_eval_job_is_configured_with_suite_and_evaluator_references(self):
        evals, session = client(Reply({"name": "agent_eval", "created": True}))
        suite = Suite.from_api({"suiteId": "s", "version": 2})
        job = evals.jobs.ensure_eval_job(7, suites=[suite, "x:1"], evaluators=["tmpl"], monitor=True, cores=2)
        assert job.created
        assert session.sent[0][2] == {"deploymentId": 7}
        assert session.sent[0][3] == {"suites": ["s:2", "x:1"], "evaluators": ["tmpl"], "monitor": True, "cores": 2}

    def test_the_review_job_settings_are_named_as_the_model_names_them(self):
        evals, session = client(Reply({"name": "agent_feedback_review", "sources": "feedback,errors"}))
        job = evals.jobs.ensure_review_job(7, provider="custom", base_url="https://gw/v1",
                                           headers={"X-Tenant": "acme"}, sources=["feedback", "errors"],
                                           read_source_code=False, budget_calls=50)
        body = session.sent[0][3]
        assert body == {"provider": "custom", "baseUrl": "https://gw/v1", "headers": '{"X-Tenant": "acme"}',
                        "sources": "feedback,errors", "readSourceCode": False, "budgetCalls": 50}
        assert job.source_list == ["feedback", "errors"]

    def test_analysing_since_the_last_run_a_window_or_one_trace(self):
        evals, session = client(Reply({"runId": "a"}), Reply({"runId": "b"}), Reply({"runId": "c"}))
        evals.jobs.analyse("job")
        evals.jobs.analyse("job", since=1000, until=2000)
        evals.jobs.analyse("job", trace_id="305b")
        assert session.sent[0][2] is None
        assert session.sent[1][2] == {"from": 1000, "to": 2000}
        assert session.sent[2][2] == {"traceId": "305b"}
        assert session.sent[0][:2] == ("POST", "/agent-evals/review-jobs/job/run")

    def test_regressions(self):
        evals, session = client(Reply({"exists": True, "name": "a_feedback_regressions", "taskCount": 3}),
                                Reply({"runId": "r"}))
        assert evals.jobs.regressions(7).task_count == 3
        assert evals.jobs.run_regressions(7).run_id == "r"
        assert session.sent[1] == ("POST", "/agent-evals/regressions/run", {"deploymentId": 7}, None)


def trace_row(trace_id, start_ms, messages):
    return {"traceId": trace_id, "spanId": "s", "startTimeNs": start_ms * 1_000_000, "sessionId": "conv",
            "messages": json.dumps(messages)}


class TestTracing:
    def test_traces_are_listed_and_searched(self):
        evals, session = client(Reply({"items": [trace_row("t1", 10, [{"role": "user", "content": "hi"}])], "count": 1}))
        [trace] = evals.deployment(7).traces(search="hi", search_field="messages", limit=5)
        assert session.sent[0][1] == "/otel/servings/7/traces"
        assert session.sent[0][2] == {"limit": 5, "search": "hi", "searchField": "messages"}
        assert trace.trace_id == "t1" and trace.conversation[0]["content"] == "hi"
        assert trace.started_at == datetime.fromtimestamp(0.01, tz=timezone.utc)

    def test_a_session_becomes_the_conversation_the_user_had(self):
        evals, _ = client(Reply({"items": [
            trace_row("later", 20, [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "hello"},
                                    {"role": "user", "content": "my order?"}, {"role": "assistant", "content": "shipped"}]),
            trace_row("first", 10, [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "hello"}]),
        ]}))
        turns = evals.deployment(7).conversation("conv")
        assert [t["trace_id"] for t in turns] == ["first", "later"]
        assert turns[1] == {"trace_id": "later", "user": "my order?", "assistant": "shipped"}

    def test_a_whole_trace_reads_its_tool_calls_off_the_attributes(self):
        evals, _ = client(Reply({"spans": [
            {"traceId": "t", "spanId": "root", "startTimeNs": 1},
            {"traceId": "t", "spanId": "tool", "parentSpanId": "root", "name": "lookup", "startTimeNs": 2,
             "statusCode": "STATUS_CODE_ERROR"},
        ], "spanAttributes": [
            {"spanId": "tool", "attrKey": "openinference.span.kind", "attrValue": "TOOL"},
            {"spanId": "tool", "attrKey": "tool.name", "attrValue": "lookup_customer"},
            {"spanId": "tool", "attrKey": "input.value", "attrValue": '{"key": "629e"}'},
        ], "totalInputTokens": 12}))
        trace = evals.deployment(7).trace("t")
        assert trace.trace_id == "t" and trace.root["spanId"] == "root"
        [call] = trace.tool_calls
        assert call["name"] == "lookup_customer" and call["status"] == "STATUS_CODE_ERROR"
        assert call["arguments"] == '{"key": "629e"}'

    def test_feedback_is_given_read_and_taken_back(self):
        evals, session = client(
            Reply({"feedbackId": "f", "traceId": "t", "verdict": "negative", "reviewer": "me@x"}),
            Reply({"count": 1, "items": [{"feedbackId": "f", "traceId": "t", "verdict": "negative",
                                          "reviewer": "detector:tool_error"}]}),
            Reply(None),
        )
        deployment = evals.deployment(7)
        given = deployment.give_feedback("t", "negative", issue_category="wrong_tool", corrected_answer="use the key")
        assert given.source == "human"
        assert session.sent[0][3] == {"verdict": "negative", "issueCategory": "wrong_tool",
                                      "correctedAnswer": "use the key"}
        page = deployment.feedback(verdict="negative", limit=10)
        assert page.count == 1 and page.feedback[0].source == "detector"
        assert session.sent[1][2] == {"limit": 10, "offset": 0, "verdict": "negative"}
        deployment.retract_feedback("t")
        assert session.sent[2][:2] == ("DELETE", "/otel/servings/7/traces/t/feedback")
        with pytest.raises(ValueError, match="verdict must be"):
            deployment.give_feedback("t", "meh")

    def test_all_feedback_walks_the_pages(self):
        evals, session = client(
            Reply({"count": 3, "items": [{"feedbackId": "1"}, {"feedbackId": "2"}]}),
            Reply({"count": 3, "items": [{"feedbackId": "3"}]}),
        )
        rows = evals.deployment(7).all_feedback(limit=2, verdict="negative")
        assert [r.feedback_id for r in rows] == ["1", "2", "3"]
        assert session.sent[1][2]["offset"] == 2

    def test_the_summary_is_about_people_unless_asked_otherwise(self):
        evals, session = client(Reply({"from": 1, "to": 2, "positive": 3, "negative": 1, "issueCategories": {"x": 1}}))
        summary = evals.deployment(7).feedback_summary(since=1, until=2)
        assert summary.positive == 3 and summary.from_ms == 1 and summary.issue_categories == {"x": 1}
        assert session.sent[0][2] == {"from": 1, "to": 2, "source": "human"}

    def test_triage_and_the_decision_that_calibrates_it(self):
        evals, session = client(
            Reply([{"triageId": "f/run", "feedbackId": "f", "suspectedCodeBug": True,
                    "codeFindings": json.dumps([{"file": "a.py", "line": 3, "finding": "x", "verified": True}])}]),
            Reply(None),
            Reply({"decisions": 4, "accepted": 3, "acceptanceRate": 0.75}),
        )
        deployment = evals.deployment(7)
        [triage] = deployment.triage(["f"])
        assert session.sent[0][2] == {"feedbackId": ["f"]}
        assert triage.suspected_code_bug and triage.findings[0]["file"] == "a.py" and not triage.decided
        deployment.decide_triage(triage, "accepted")
        assert session.sent[1] == ("PUT", "/otel/servings/7/feedback/triage/f%2Frun/decision".replace("%2F", "/"),
                                   {"decision": "accepted"}, None)
        assert deployment.calibration().acceptance_rate == 0.75
        assert deployment.triage([]) == []

    def test_clusters_are_promoted_from_their_representative(self):
        evals, session = client(
            Reply([{"clusterId": "c", "size": 2, "representativeTraceId": "t", "representativeFeedbackId": "f"}]),
            Reply([{"triageId": "x", "feedbackId": "f", "normalizedCorrection": "Use the key."}]),
            Reply({"taskId": "task"}),
            Reply(None),
        )
        deployment = evals.deployment(7)
        [cluster] = deployment.clusters()
        assert session.sent[0][2] == {"status": "open"}
        task = deployment.promote_cluster(cluster)
        assert task.task_id == "task"
        assert session.sent[2][1] == "/agent-evals/tasks/from-trace/t"
        assert session.sent[2][3]["expectations"] == {"expected": "Use the key."}
        assert session.sent[3] == ("PUT", "/otel/servings/7/feedback/clusters/c",
                                   {"status": "promoted", "promotedTaskId": "task"}, None)

    def test_dismiss_rename_and_reopen(self):
        evals, session = client(Reply(None), Reply(None), Reply(None))
        deployment = evals.deployment(7)
        deployment.dismiss_cluster("c", "working_as_intended")
        deployment.rename_cluster(Cluster.from_api({"clusterId": "c"}), "Ignores the key")
        deployment.reopen_cluster("c")
        assert session.sent[0][2] == {"status": "dismissed", "dismissReason": "working_as_intended"}
        assert session.sent[1][2] == {"label": "Ignores the key"}
        assert session.sent[2][2] == {"status": "open"}

    def test_metrics_gates_and_canary(self):
        evals, session = client(
            Reply([{"windowStart": 1, "windowEnd": 2, "traceCount": 5, "traceErrorCount": 1}]),
            Reply([{"toolName": "lookup", "toolCallCount": 3}]),
            Reply({"passed": False, "hasEvidence": True, "checks": [{"suiteName": "s", "passed": False}]}),
            Reply([{"runId": "r"}]),
        )
        deployment = evals.deployment(7)
        assert deployment.trace_metrics(since=1, until=2)[0].trace_error_count == 1
        assert session.sent[0][2] == {"from": 1, "to": 2}
        assert deployment.tool_metrics()[0].tool_name == "lookup"
        gates = deployment.gates()
        assert not gates.passed and gates.gate_checks[0].suite_name == "s"
        assert deployment.canary()[0].run_id == "r"
        assert session.sent[3] == ("POST", "/agent-evals/canary", {"deploymentId": 7}, None)

    def test_analyse_creates_the_job_when_the_deployment_has_none(self):
        evals, session = client(Reply([]), Reply({"name": "agent_feedback_review"}), Reply({"runId": "r"}))
        run = evals.deployment(7).analyse()
        assert run.run_id == "r"
        assert session.sent[1][:2] == ("POST", "/agent-evals/review-jobs")
        assert session.sent[2][1] == "/agent-evals/review-jobs/agent_feedback_review/run"


class TestModels:
    def test_timestamps_read_as_the_api_sends_them(self):
        assert as_datetime("2026-09-10T08:00:00Z") == datetime(2026, 9, 10, 8, tzinfo=timezone.utc)
        assert as_datetime(1_000) == datetime.fromtimestamp(1, tz=timezone.utc)
        assert as_datetime("") is None and as_datetime("nope") is None

    def test_unknown_fields_stay_reachable_in_raw(self):
        suite = Suite.from_api({"suiteId": "s", "somethingNew": 1})
        assert suite.raw["somethingNew"] == 1

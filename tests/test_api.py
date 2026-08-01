"""The scripting surface: what it sends, and what it does with a refusal."""

from __future__ import annotations

import json

import pytest

from hopsworks_agent_eval.api import (
    EvalApi,
    EvalApiError,
    evaluator,
    tool_expectation,
)


class FakeResponse:
    def __init__(self, status=200, body=None):
        self.status_code = status
        self._body = {} if body is None else body
        self.content = b"{}"

    def json(self):
        return self._body


class FakeSession:
    """Records what was sent, and answers with whatever was queued."""

    def __init__(self, replies=None):
        self.headers = {}
        self.verify = True
        self.sent = []
        self._replies = list(replies or [])

    def request(self, method, url, json=None, timeout=None):
        self.sent.append((method, url, json))
        return self._replies.pop(0) if self._replies else FakeResponse()


def api(replies=None):
    client = EvalApi.__new__(EvalApi)
    client._base = "https://h/hopsworks-api/api/project/1/agent-evals"
    client._session = FakeSession(replies)
    return client


class TestWritingAChecksConfig:
    def test_a_checks_own_configuration_is_held_apart_from_its_type(self):
        # the row shape: everything but type and name goes in config, because what a
        # check takes differs by type
        one = evaluator("llm_judge", name="quality", provider="anthropic",
                        temperature=0)
        assert one["type"] == "llm_judge"
        assert one["name"] == "quality"
        assert json.loads(one["config"]) == {"provider": "anthropic", "temperature": 0}

    def test_a_check_defaults_to_being_named_after_its_type(self):
        assert evaluator("tool_call")["name"] == "tool_call"

    def test_a_check_with_no_configuration_still_sends_an_object(self):
        assert json.loads(evaluator("no_tool_error")["config"]) == {}

    def test_a_tool_expectation_carries_both_directions(self):
        assert json.loads(tool_expectation(required=["a"], forbidden=["b"])) == {
            "required": ["a"], "forbidden": ["b"]
        }

    def test_either_direction_alone_is_enough(self):
        # a call check judges both, so a forbidden list on its own is a real expectation
        assert json.loads(tool_expectation(forbidden=["place_order"])) == {
            "required": [], "forbidden": ["place_order"]
        }


class TestWhatItSends:
    def test_a_suite_is_created_with_its_checks_in_the_same_call(self):
        # a suite with no checks grades by nothing and every trial comes back ungradable
        client = api()
        client.create_suite("s", evaluators=[evaluator("contains")], tags=["safety"])
        method, url, body = client._session.sent[0]
        assert (method, url.endswith("/suites")) == ("POST", True)
        assert body["evaluators"][0]["type"] == "contains"
        assert json.loads(body["tags"]) == ["safety"]

    def test_adding_a_task_authors_it_then_joins_it_with_its_expectations(self):
        client = api([FakeResponse(body={"taskId": "t1"}), FakeResponse()])
        client.add_task({"suiteId": "s1", "version": 2}, "hello",
                        {"contains": "UB40"})
        author, join = client._session.sent
        assert json.loads(author[2]["inputMessages"])[0]["content"] == "hello"
        assert "suiteId=s1&version=2" in join[1]
        assert join[2] == {"expectations": {"contains": "UB40"}}

    def test_a_task_with_no_expectations_still_sends_the_key(self):
        # an omitted map and an empty one mean different things to the server
        client = api([FakeResponse(body={"taskId": "t1"}), FakeResponse()])
        client.add_task({"suiteId": "s1", "version": 1}, "hello")
        assert client._session.sent[1][2] == {"expectations": {}}

    def test_starting_a_run_records_it_then_starts_it(self):
        client = api([FakeResponse(body={"runId": "r1"}), FakeResponse()])
        client.start_run({"suiteId": "s1", "version": 1}, deployment_id=7, n_trials=3)
        record, start = client._session.sent
        assert record[2]["deploymentId"] == 7
        assert record[2]["nTrials"] == 3
        assert start[1].endswith("/runs/r1/start")

    def test_the_runner_job_can_be_sized_before_it_exists(self):
        client = api()
        client.ensure_runner_job(environment_name="mine", cores=2, memory=4096)
        method, url, body = client._session.sent[0]
        assert (method, url.endswith("/runner-job")) == ("POST", True)
        assert (body["environmentName"], body["cores"]) == ("mine", 2)


class TestWhenTheApiRefuses:
    def test_the_reason_the_api_gave_is_what_is_raised(self):
        # the rules refuse with a sentence saying which one and why; a bare status
        # sends people to read the source instead
        client = api([FakeResponse(400, {"usrMsg": "publish the suite before running it"})])
        with pytest.raises(EvalApiError, match="publish the suite"):
            client.suites()

    def test_a_refusal_with_no_message_still_says_something(self):
        client = api([FakeResponse(503, {})])
        with pytest.raises(EvalApiError, match="503"):
            client.suites()

    def test_an_unparseable_refusal_does_not_mask_itself(self):
        response = FakeResponse(502)
        response.json = lambda: (_ for _ in ()).throw(ValueError("not json"))
        with pytest.raises(EvalApiError, match="502"):
            api([response]).suites()

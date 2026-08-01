"""What the API sends, as the runner's models.

This seam had no tests, and a whole model change went through it unnoticed: the
runner kept building Tasks from expectedOutput, requiredTools, forbiddenTools and
rubric long after none of those existed on either side. It failed on the first
real run, after the four connection problems ahead of it had been fixed one at a
time.
"""

from __future__ import annotations

import json

from hopsworks_agent_eval.evaluator_spec import evaluators_for_suite
from hopsworks_agent_eval.models import ExecutionMode, PassPolicy
from hopsworks_agent_eval.run_job import _tags, _to_suite


def run(**overrides):
    base = {
        "suiteId": "s1",
        "suiteVersion": 2,
        "executionMode": "read_only",
        "passPolicy": "all",
        "passThreshold": 0.7,
        "evaluators": [
            {"type": "contains", "name": "names_the_record", "position": 0,
             "config": "{}"},
        ],
    }
    base.update(overrides)
    return base


def task(**overrides):
    base = {
        "taskId": "t1",
        "version": 1,
        "inputMessages": json.dumps([{"role": "user", "content": "hi"}]),
        "taskType": "single_turn",
    }
    base.update(overrides)
    return base


class TestTasks:
    def test_expectations_are_copied_under_the_check_that_reads_them(self):
        # the same key results come back under, so this is a copy not a translation
        suite = _to_suite(run(), [task(expectations={"names_the_record": "UB40"})])
        assert suite.tasks[0].expects_text("names_the_record") == "UB40"

    def test_a_blank_expectation_is_dropped_rather_than_stored(self):
        # an empty value reads as a check that has been answered, which is the
        # opposite of the truth
        suite = _to_suite(run(), [task(expectations={"names_the_record": ""})])
        assert suite.tasks[0].expectations == {}

    def test_a_task_with_no_expectations_is_still_a_task(self):
        # a suite whose checks judge the run itself expects nothing of its tasks
        suite = _to_suite(run(), [task()])
        assert suite.tasks[0].task_id == "t1"
        assert suite.tasks[0].expectations == {}

    def test_the_question_survives(self):
        suite = _to_suite(run(), [task()])
        assert suite.tasks[0].prompt == "hi"

    def test_tool_expectations_keep_both_directions(self):
        stored = json.dumps({"required": ["lookup"], "forbidden": ["refund"]})
        suite = _to_suite(run(), [task(expectations={"tool_call": stored})])
        assert suite.tasks[0].expects_tools("tool_call") == (["lookup"], ["refund"])


class TestSuite:
    def test_the_checks_arrive_as_rows_and_still_build(self):
        # the end of this path: rows in, evaluators out
        [evaluator] = evaluators_for_suite(_to_suite(run(), [task()]))
        assert evaluator.type == "contains"
        assert evaluator.name == "names_the_record"

    def test_the_version_that_ran_is_the_one_recorded(self):
        # a result has to be able to say which version of the suite produced it
        assert _to_suite(run(), []).suite_version == 2

    def test_the_pass_policy_and_threshold_come_from_the_run(self):
        suite = _to_suite(run(passPolicy="threshold", passThreshold=0.9), [])
        assert suite.pass_policy is PassPolicy.THRESHOLD
        assert suite.pass_threshold == 0.9

    def test_the_execution_mode_comes_from_the_run(self):
        # what the runner refuses on depends on this, so a default is not harmless
        assert _to_suite(run(executionMode="sandboxed"), []).execution_mode is (
            ExecutionMode.SANDBOXED
        )

    def test_blocks_counting_as_success_survives(self):
        assert _to_suite(run(blocksAreSuccess=True), []).blocks_are_success


class TestTags:
    def test_a_json_array_becomes_a_list(self):
        # the column holds JSON; a raw string would make every character a tag
        assert _tags('["safety","regression"]') == ["safety", "regression"]

    def test_an_unreadable_value_is_no_tags(self):
        assert _tags("not json") == []

    def test_a_list_is_left_alone(self):
        assert _tags(["safety"]) == ["safety"]

    def test_nothing_is_no_tags(self):
        assert _tags(None) == []


class TestWritingResultsMatchesTheSchema:
    """Python has one integer type and pandas reads it as int64, so a column the
    feature group declares `int` arrives as `bigint` and the insert is refused —
    after the whole suite has run, which is the most expensive moment to find out.
    """

    def frame(self, **columns):
        import pandas as pd

        return pd.DataFrame(columns)

    def group(self, **types):
        features = [
            type("F", (), {"name": name, "type": kind})() for name, kind in types.items()
        ]
        return type("G", (), {"features": features})()

    def test_an_int_column_is_narrowed_to_what_the_group_declares(self):
        from hopsworks_agent_eval.run_job import _match_schema

        frame = _match_schema(
            self.group(trial_index="int"), self.frame(trial_index=[0, 1])
        )
        assert str(frame["trial_index"].dtype) == "int32"

    def test_a_bigint_column_is_left_wide(self):
        from hopsworks_agent_eval.run_job import _match_schema

        frame = _match_schema(
            self.group(deployment_id="bigint"), self.frame(deployment_id=[1, 2])
        )
        assert str(frame["deployment_id"].dtype) == "int64"

    def test_a_column_with_nulls_is_left_alone(self):
        # pandas cannot hold a null in a plain integer type, and refusing the whole
        # write over one absent value would be worse
        from hopsworks_agent_eval.run_job import _match_schema

        frame = _match_schema(
            self.group(input_tokens="bigint"), self.frame(input_tokens=[1, None])
        )
        assert frame["input_tokens"].isna().any()

    def test_a_column_the_frame_does_not_have_is_not_invented(self):
        from hopsworks_agent_eval.run_job import _match_schema

        frame = _match_schema(self.group(absent="int"), self.frame(present=[1]))
        assert list(frame.columns) == ["present"]

    def test_a_type_it_does_not_know_is_left_alone(self):
        from hopsworks_agent_eval.run_job import _match_schema

        frame = _match_schema(self.group(reason="string"), self.frame(reason=["ok"]))
        assert frame["reason"].tolist() == ["ok"]

    def test_a_group_that_reports_no_schema_is_not_an_error(self):
        # rather than failing the write over an introspection detail
        from hopsworks_agent_eval.run_job import _match_schema

        frame = _match_schema(type("G", (), {})(), self.frame(a=[1]))
        assert frame["a"].tolist() == [1]


class Secrets:
    def __init__(self, **values):
        self._values = values

    def get_secret(self, name):
        if name not in self._values:
            raise PermissionError(f"no such secret {name}")
        return type("S", (), {"value": self._values[name]})()


def project(**secrets):
    return type(
        "P", (), {"id": 1, "name": "g1",
                  "get_secrets_api": staticmethod(lambda: Secrets(**secrets))},
    )()


class TestFindingTheDefaultJudgesKey:
    """A key already set for anything else in the project should be found without
    being configured a second time — every provider's SDK reads its own variable."""

    def test_the_named_secret_wins(self, monkeypatch):
        # the explicit choice: a release gate naming its own key means that one
        from hopsworks_agent_eval import run_job

        monkeypatch.setenv("ANTHROPIC_API_KEY", "from-env")
        judge = run_job._judge_for(project(EVAL_JUDGE_API_KEY="from-secret"))
        assert judge is not None

    def test_the_providers_environment_variable_is_used_when_no_secret_names_one(
        self, monkeypatch
    ):
        # what silently skipped every judge before: the key was there, under the
        # name its own SDK reads, and nothing looked
        from hopsworks_agent_eval import run_job

        monkeypatch.setenv("ANTHROPIC_API_KEY", "from-env")
        assert run_job._judge_for(project()) is not None

    def test_a_project_secret_of_that_name_is_the_last_place_looked(self, monkeypatch):
        from hopsworks_agent_eval import run_job

        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        assert run_job._judge_for(project(ANTHROPIC_API_KEY="from-secret")) is not None

    def test_with_no_key_anywhere_it_says_where_it_looked(self, monkeypatch, caplog):
        # not "no secret X": the reason a suite reported 100% while its only real
        # check never ran
        import logging

        from hopsworks_agent_eval import run_job

        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        with caplog.at_level(logging.INFO):
            assert run_job._judge_for(project()) is None
        assert "EVAL_JUDGE_API_KEY" in caplog.text
        assert "ANTHROPIC_API_KEY" in caplog.text
        assert "report nothing" in caplog.text

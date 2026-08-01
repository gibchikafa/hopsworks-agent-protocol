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

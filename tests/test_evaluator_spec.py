"""The checks a suite names, and what it refuses to accept as one.

The spec lives on the suite: a pass rate only means something if every task in
it was measured the same way. Inference was removed with the move — checks
derived from whatever each task happened to declare are exactly what makes two
tasks in one suite incomparable.
"""

from __future__ import annotations

import json

import pytest

from hopsworks_agent_eval.evaluator_spec import (
    SpecError,
    evaluators_for_suite,
    evaluators_from_spec,
    validate_spec,
)
from hopsworks_agent_eval.models import Suite, Task


def task(**kwargs) -> Task:
    return Task(
        task_id="t1",
        input_messages=json.dumps([{"role": "user", "content": "hi"}]),
        **kwargs,
    )


def completer(_prompt: str) -> str:
    return '{"score": 1.0, "passed": true, "reason": "ok"}'


class TestBuildingFromSpec:
    def test_every_deterministic_type_can_be_named(self):
        spec = [
            {"type": "exact_match"},
            {"type": "contains", "expected": "yes"},
            {"type": "regex", "pattern": "^a"},
            {"type": "json_schema", "required_keys": ["id"]},
            {"type": "tool_call"},
            {"type": "tool_order"},
            {"type": "no_tool_error"},
            {"type": "human_review"},
            {"type": "sql_state", "sql": "SELECT 1", "expect": 1},
        ]
        evaluators = evaluators_from_spec(json.dumps(spec))
        assert [g.type for g in evaluators] == [
            "exact_match", "contains", "regex", "json_schema", "tool_call",
            "tool_order", "no_tool_error", "human_review", "sql_state",
        ]

    def test_a_named_evaluator_keeps_its_name(self):
        # two regex checks in one suite are only distinguishable by name
        evaluators = evaluators_from_spec(
            [{"type": "regex", "pattern": "a", "name": "has_order_id"}]
        )
        assert evaluators[0].name == "has_order_id"

    def test_judge_evaluators_need_a_completer(self):
        evaluators = evaluators_from_spec([{"type": "llm_judge"}], judge_completer=completer)
        assert [g.type for g in evaluators] == ["llm_judge"]

    def test_judge_evaluators_are_skipped_when_no_judge_is_configured(self):
        # not an error: the deterministic checks in the same suite still run,
        # and pretending a judgement happened would be worse than skipping it
        evaluators = evaluators_from_spec(
            [{"type": "llm_judge"}, {"type": "regex", "pattern": "a"}]
        )
        assert [g.type for g in evaluators] == ["regex"]

    def test_pairwise_can_be_named(self):
        evaluators = evaluators_from_spec(
            [{"type": "pairwise", "reference": "the good answer"}],
            judge_completer=completer,
        )
        assert evaluators[0].type == "pairwise"
        assert evaluators[0].reference == "the good answer"

    def test_a_sql_evaluator_is_handed_the_query_function(self):
        called = []
        evaluators = evaluators_from_spec(
            [{"type": "sql_state", "sql": "SELECT status FROM o", "expect": "x"}],
            query=lambda sql: called.append(sql) or "x",
        )
        assert evaluators[0].grade(task(), _trial(), None).passed
        assert called == ["SELECT status FROM o"]

    def test_an_empty_spec_is_no_evaluators_not_an_error(self):
        assert evaluators_from_spec("") == []
        assert evaluators_from_spec(None) == []
        assert evaluators_from_spec([]) == []


class TestRefusals:
    def test_unparseable_json(self):
        with pytest.raises(SpecError, match="not valid JSON"):
            evaluators_from_spec("{not json")

    def test_an_object_where_an_array_belongs(self):
        with pytest.raises(SpecError, match="array of objects"):
            evaluators_from_spec('{"type": "regex"}')

    def test_an_entry_that_is_not_an_object(self):
        with pytest.raises(SpecError, match="entry 2"):
            evaluators_from_spec('[{"type": "tool_call"}, "regex"]')

    def test_an_unknown_type_lists_what_is_allowed(self):
        # a typo should not silently produce a suite graded by nothing
        with pytest.raises(SpecError, match="unknown evaluator type"):
            evaluators_from_spec('[{"type": "vibes"}]')

    def test_a_regex_evaluator_without_a_pattern(self):
        with pytest.raises(SpecError, match="entry 1: a regex evaluator needs a pattern"):
            evaluators_from_spec('[{"type": "regex"}]')

    def test_a_schema_evaluator_without_keys(self):
        with pytest.raises(SpecError, match="required_keys"):
            evaluators_from_spec('[{"type": "json_schema"}]')

    def test_a_sql_evaluator_without_a_query(self):
        with pytest.raises(SpecError, match="needs a sql query"):
            evaluators_from_spec('[{"type": "sql_state", "expect": 1}]')

    def test_validate_accepts_judge_and_sql_evaluators_without_runtime_config(self):
        # authoring must not depend on a provider key or a database session
        validate_spec('[{"type": "llm_judge"}, {"type": "sql_state", "sql": "SELECT 1"}]')

    def test_validate_rejects_what_the_runner_could_not_build(self):
        with pytest.raises(SpecError):
            validate_spec('[{"type": "regex"}]')


class TestTheSuiteOwnsTheChecks:
    def test_a_suite_names_what_every_task_is_graded_by(self):
        suite = Suite(
            suite_id="s1",
            evaluators='[{"type": "regex", "pattern": "^ORD"}]',
        )
        assert [g.type for g in evaluators_for_suite(suite)] == ["regex"]

    def test_a_suite_with_no_checks_grades_by_nothing(self):
        # there is no inference: checks derived from whatever each task happened
        # to declare would make two tasks in one suite incomparable, which is
        # the thing a suite exists to prevent
        assert evaluators_for_suite(Suite(suite_id="s1")) == []

    def test_the_checks_do_not_depend_on_the_tasks(self):
        # the same suite grades every task the same way; what varies is what
        # each task supplies to those checks
        suite = Suite(
            suite_id="s1",
            evaluators='[{"type": "contains"}, {"type": "tool_call"}]',
            tasks=[task(expected_output="a"), task(required_tools=["x"])],
        )
        assert [g.type for g in evaluators_for_suite(suite)] == [
            "contains", "tool_call",
        ]


def _trial():
    from hopsworks_agent_eval.models import Trial

    return Trial(
        trial_id="x", run_id="r", task_id="t1", task_version=1,
        trial_index=0, deployment_id=1, final_output="anything",
    )

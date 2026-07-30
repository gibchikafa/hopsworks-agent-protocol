"""Graders a task asks for by name, rather than ones inferred for it.

Inference covers the ordinary cases and reaches none of the others: before a
spec existed, four implemented graders could never be selected by anything.
"""

from __future__ import annotations

import json

import pytest

from hopsworks_agent_eval.grader_spec import (
    SpecError,
    graders_for_task,
    graders_from_spec,
    validate_spec,
)
from hopsworks_agent_eval.models import Task


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
        graders = graders_from_spec(json.dumps(spec))
        assert [g.type for g in graders] == [
            "exact_match", "contains", "regex", "json_schema", "tool_call",
            "tool_order", "no_tool_error", "human_review", "sql_state",
        ]

    def test_a_named_grader_keeps_its_name(self):
        # two regex graders on one task are only distinguishable by name
        graders = graders_from_spec(
            [{"type": "regex", "pattern": "a", "name": "has_order_id"}]
        )
        assert graders[0].name == "has_order_id"

    def test_judge_graders_need_a_completer(self):
        graders = graders_from_spec([{"type": "llm_judge"}], judge_completer=completer)
        assert [g.type for g in graders] == ["llm_judge"]

    def test_judge_graders_are_skipped_when_no_judge_is_configured(self):
        # not an error: the deterministic graders on the same task still run, and
        # pretending a judgement happened would be worse than skipping it
        graders = graders_from_spec(
            [{"type": "llm_judge"}, {"type": "regex", "pattern": "a"}]
        )
        assert [g.type for g in graders] == ["regex"]

    def test_pairwise_can_be_named(self):
        graders = graders_from_spec(
            [{"type": "pairwise", "reference": "the good answer"}],
            judge_completer=completer,
        )
        assert graders[0].type == "pairwise"
        assert graders[0].reference == "the good answer"

    def test_a_sql_grader_is_handed_the_query_function(self):
        called = []
        graders = graders_from_spec(
            [{"type": "sql_state", "sql": "SELECT status FROM o", "expect": "x"}],
            query=lambda sql: called.append(sql) or "x",
        )
        assert graders[0].grade(task(), _trial(), None).passed
        assert called == ["SELECT status FROM o"]

    def test_an_empty_spec_is_no_graders_not_an_error(self):
        assert graders_from_spec("") == []
        assert graders_from_spec(None) == []
        assert graders_from_spec([]) == []


class TestRefusals:
    def test_unparseable_json(self):
        with pytest.raises(SpecError, match="not valid JSON"):
            graders_from_spec("{not json")

    def test_an_object_where_an_array_belongs(self):
        with pytest.raises(SpecError, match="array of objects"):
            graders_from_spec('{"type": "regex"}')

    def test_an_entry_that_is_not_an_object(self):
        with pytest.raises(SpecError, match="entry 2"):
            graders_from_spec('[{"type": "tool_call"}, "regex"]')

    def test_an_unknown_type_lists_what_is_allowed(self):
        # a typo should not silently produce a task graded by nothing
        with pytest.raises(SpecError, match="unknown grader type"):
            graders_from_spec('[{"type": "vibes"}]')

    def test_a_regex_grader_without_a_pattern(self):
        with pytest.raises(SpecError, match="entry 1: a regex grader needs a pattern"):
            graders_from_spec('[{"type": "regex"}]')

    def test_a_schema_grader_without_keys(self):
        with pytest.raises(SpecError, match="required_keys"):
            graders_from_spec('[{"type": "json_schema"}]')

    def test_a_sql_grader_without_a_query(self):
        with pytest.raises(SpecError, match="needs a sql query"):
            graders_from_spec('[{"type": "sql_state", "expect": 1}]')

    def test_validate_accepts_judge_and_sql_graders_without_runtime_config(self):
        # authoring must not depend on a provider key or a database session
        validate_spec('[{"type": "llm_judge"}, {"type": "sql_state", "sql": "SELECT 1"}]')

    def test_validate_rejects_what_the_runner_could_not_build(self):
        with pytest.raises(SpecError):
            validate_spec('[{"type": "regex"}]')


class TestSpecVersusInference:
    def test_a_task_with_no_spec_is_inferred_as_before(self):
        graders = graders_for_task(
            task(expected_output="yes", required_tools=["a", "b"])
        )
        assert [g.type for g in graders] == [
            "contains", "tool_call", "tool_order", "no_tool_error",
        ]

    def test_a_spec_replaces_inference_rather_than_adding_to_it(self):
        # a task asking for one regex must not also fail a substring check it
        # never asked for
        graders = graders_for_task(
            task(
                expected_output="yes",
                required_tools=["a"],
                graders='[{"type": "regex", "pattern": "^ORD"}]',
            )
        )
        assert [g.type for g in graders] == ["regex"]

    def test_inference_still_adds_a_judge_when_a_rubric_is_present(self):
        graders = graders_for_task(
            task(rubric="cites the policy"), judge_completer=completer
        )
        assert [g.type for g in graders] == ["llm_judge"]

    def test_a_task_declaring_nothing_gets_nothing(self):
        # its trials come back ungradable, which is honest; inventing a grader
        # here would be a silent free pass
        assert graders_for_task(task()) == []


def _trial():
    from hopsworks_agent_eval.models import Trial

    return Trial(
        trial_id="x", run_id="r", task_id="t1", task_version=1,
        trial_index=0, deployment_id=1, final_output="anything",
    )

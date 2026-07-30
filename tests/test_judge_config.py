"""A judge configured with named criteria, weights and floors.

The two behaviours worth protecting: a weighted total cannot rescue a
catastrophic score on a criterion someone marked critical, and a judge that
answers only half the criteria is ungradable rather than averaged over whatever
came back.
"""

from __future__ import annotations

import json

import pytest

from hopsworks_agent_eval.grader_spec import SpecError, graders_from_spec
from hopsworks_agent_eval.judge_config import (
    FAILURE_CATEGORIES,
    JudgeConfigError,
    default_templates,
    parse_judge_config,
    render_prompt,
)
from hopsworks_agent_eval.judges import MultiCriteriaJudge
from hopsworks_agent_eval.models import Task, Trial

CRITERIA = {
    "task_completion": {"weight": 0.5, "description": "Did it finish the job?"},
    "correctness": {"weight": 0.3, "description": "Is it factually right?"},
    "safety": {"weight": 0.2, "description": "Did it respect the limits?"},
}


def entry(**overrides):
    base = {
        "type": "llm_judge",
        "score_range": [1, 5],
        "criteria": CRITERIA,
        "thresholds": {"pass_score": 4.0, "critical_dimensions": {"safety": 4}},
    }
    base.update(overrides)
    return base


def task(**kwargs) -> Task:
    return Task(
        task_id="t1",
        input_messages=json.dumps([{"role": "user", "content": "cancel 4471"}]),
        **kwargs,
    )


def trial(output: str = "Done — order 4471 is cancelled.") -> Trial:
    return Trial(
        trial_id="x", run_id="r", task_id="t1", task_version=1,
        trial_index=0, deployment_id=1, final_output=output,
    )


def judge(reply: dict | str, **overrides) -> MultiCriteriaJudge:
    text = reply if isinstance(reply, str) else json.dumps(reply)
    return MultiCriteriaJudge(lambda _p: text, parse_judge_config(entry(**overrides)))


class TestParsing:
    def test_criteria_as_an_object_of_settings(self):
        config = parse_judge_config(entry())
        assert [c.name for c in config.criteria] == [
            "task_completion", "correctness", "safety",
        ]
        assert config.criteria[2].critical_min == 4

    def test_criteria_as_a_plain_list_of_names(self):
        config = parse_judge_config(
            {"type": "llm_judge", "criteria": ["accuracy", "tone"]}
        )
        assert [c.weight for c in config.criteria] == [1.0, 1.0]

    def test_pass_score_defaults_to_three_quarters_of_the_scale(self):
        config = parse_judge_config({"type": "llm_judge", "score_range": [1, 5],
                                     "criteria": ["a"]})
        assert config.pass_score == 4.0

    def test_a_critical_floor_naming_no_criterion_is_refused(self):
        # silently ignoring it would leave someone believing a floor is enforced
        with pytest.raises(JudgeConfigError, match="no such criterion"):
            parse_judge_config(entry(
                thresholds={"critical_dimensions": {"tone": 3}},
            ))

    def test_a_pass_score_outside_the_range_is_refused(self):
        with pytest.raises(JudgeConfigError, match="inside score_range"):
            parse_judge_config(entry(thresholds={"pass_score": 9}))

    def test_an_unknown_provider_is_refused(self):
        with pytest.raises(JudgeConfigError, match="provider must be"):
            parse_judge_config(entry(provider="mystery"))

    def test_an_unknown_input_is_refused(self):
        with pytest.raises(JudgeConfigError, match="unknown input"):
            parse_judge_config(entry(inputs=["user_request", "the_weather"]))

    def test_zero_weights_are_refused(self):
        with pytest.raises(JudgeConfigError, match="weights cannot all be zero"):
            parse_judge_config(entry(criteria={"a": {"weight": 0}}))

    def test_a_custom_template_must_have_the_answer_in_it(self):
        with pytest.raises(JudgeConfigError, match="prompt_template"):
            parse_judge_config(entry(prompt_template="grade {question} please"))

    def test_temperature_is_bounded(self):
        with pytest.raises(JudgeConfigError, match="temperature"):
            parse_judge_config(entry(temperature=5))

    def test_normalisation_maps_the_scale_onto_zero_to_one(self):
        config = parse_judge_config(entry())
        assert config.normalise(1) == 0.0
        assert config.normalise(5) == 1.0
        assert config.normalise(3) == 0.5

    def test_normalisation_clamps_a_judge_that_left_the_scale(self):
        # models told to answer 1-5 occasionally answer 0 or 6
        config = parse_judge_config(entry())
        assert config.normalise(0) == 0.0
        assert config.normalise(9) == 1.0


class TestPrompt:
    def test_it_names_every_criterion_with_its_description(self):
        prompt = render_prompt(parse_judge_config(entry()), question="q", answer="a")
        assert "task_completion: Did it finish the job?" in prompt
        assert "safety: Did it respect the limits?" in prompt

    def test_it_asks_for_the_configured_scale(self):
        prompt = render_prompt(parse_judge_config(entry()), question="q", answer="a")
        assert "1 to 5" in prompt

    def test_inputs_control_what_the_judge_is_shown(self):
        # a judge asked whether the agent could get there alone must not be
        # shown the answer
        config = parse_judge_config(entry(inputs=["user_request", "agent_response"]))
        prompt = render_prompt(config, question="q", answer="a", expected="the answer")
        assert "the answer" not in prompt

        seeing = parse_judge_config(entry(inputs=["expected_result"]))
        assert "the answer" in render_prompt(
            seeing, question="q", answer="a", expected="the answer"
        )

    def test_tool_context_appears_only_when_asked_for(self):
        config = parse_judge_config(
            entry(inputs=["user_request", "agent_response", "tool_results"])
        )
        prompt = render_prompt(
            config, question="q", answer="a",
            tool_calls="lookup()", tool_results="status=open",
        )
        assert "status=open" in prompt
        assert "lookup()" not in prompt

    def test_the_failure_taxonomy_is_enumerated_in_the_prompt(self):
        prompt = render_prompt(parse_judge_config(entry()), question="q", answer="a")
        for category in FAILURE_CATEGORIES:
            assert category in prompt


class TestGrading:
    def test_a_weighted_total_above_the_bar_passes(self):
        result = judge({"scores": {"task_completion": 5, "correctness": 4, "safety": 5}}).grade(
            task(), trial(), None
        )
        assert result.passed
        assert result.assertions["weighted_score"] == pytest.approx(4.7)
        assert result.score == pytest.approx((4.7 - 1) / 4)

    def test_a_weighted_total_below_the_bar_fails_and_says_so(self):
        result = judge({"scores": {"task_completion": 3, "correctness": 3, "safety": 4}}).grade(
            task(), trial(), None
        )
        assert result.passed is False
        assert "needs 4" in result.reason

    def test_a_critical_floor_overrides_a_good_total(self):
        # the whole point: six good scores must not hide one catastrophic result
        result = judge({"scores": {"task_completion": 5, "correctness": 5, "safety": 1}}).grade(
            task(), trial(), None
        )
        assert result.passed is False
        assert result.assertions["critical_breached"] == ["safety"]
        assert "below the floor on safety" in result.reason

    def test_the_breakdown_is_kept_in_both_scales(self):
        result = judge({"scores": {"task_completion": 5, "correctness": 4, "safety": 5}}).grade(
            task(), trial(), None
        )
        assert result.assertions["criteria"]["correctness"] == 4
        assert result.assertions["criteria_normalised"]["correctness"] == 0.75

    def test_a_judge_scoring_only_some_criteria_is_ungradable(self):
        # a weighted total over whichever criteria came back is a different
        # measurement every time
        result = judge({"scores": {"task_completion": 5}}).grade(task(), trial(), None)
        assert result.ungradable
        assert "did not score correctness, safety" in result.reason

    def test_prose_instead_of_json_is_ungradable_not_a_failure(self):
        result = judge("Looks good to me!").grade(task(), trial(), None)
        assert result.ungradable and result.passed is False

    def test_a_judge_that_raises_is_ungradable(self):
        grader = MultiCriteriaJudge(
            lambda _p: (_ for _ in ()).throw(RuntimeError("rate limited")),
            parse_judge_config(entry()),
        )
        result = grader.grade(task(), trial(), None)
        assert result.ungradable
        assert "rate limited" in result.reason

    def test_no_answer_is_ungradable(self):
        result = judge({"scores": {}}).grade(task(), trial(""), None)
        assert result.ungradable

    def test_an_unknown_failure_category_becomes_other(self):
        result = judge({
            "scores": {"task_completion": 5, "correctness": 4, "safety": 5},
            "failure_category": "vibes",
        }).grade(task(), trial(), None)
        assert result.assertions["failure_category"] == "other"

    def test_reasoning_reaches_the_reason_when_it_failed(self):
        result = judge({
            "scores": {"task_completion": 2, "correctness": 2, "safety": 5},
            "reasoning": {"correctness": "the order id was invented"},
        }).grade(task(), trial(), None)
        assert "the order id was invented" in result.reason

    def test_a_judge_reading_tool_calls_needs_a_trace(self):
        grader = judge({"scores": {}}, inputs=["user_request", "agent_response", "tool_calls"])
        assert grader.needs_trace is True
        assert grader.grade(task(), trial(), None).ungradable

    def test_a_judge_not_reading_tool_calls_does_not_need_one(self):
        # otherwise it would go ungradable for want of a trace it never reads
        grader = judge({"scores": {"task_completion": 5, "correctness": 5, "safety": 5}})
        assert grader.needs_trace is False
        assert grader.grade(task(), trial(), None).passed


class TestSpecIntegration:
    def test_criteria_turn_an_llm_judge_into_a_multi_criteria_one(self):
        graders = graders_from_spec([entry()], judge_completer=lambda _p: "{}")
        assert isinstance(graders[0], MultiCriteriaJudge)

    def test_without_criteria_it_stays_the_single_score_judge(self):
        graders = graders_from_spec(
            [{"type": "llm_judge"}], judge_completer=lambda _p: "{}"
        )
        assert not isinstance(graders[0], MultiCriteriaJudge)

    def test_a_bad_judge_config_is_a_spec_error(self):
        # so it is refused at authoring time like every other malformed entry
        with pytest.raises(SpecError, match="provider must be"):
            graders_from_spec([entry(provider="mystery")], judge_completer=lambda _p: "")

    def test_a_judge_naming_a_missing_secret_is_skipped_not_failed(self):
        graders = graders_from_spec(
            [entry(provider="openai", model="gpt-4o", api_key_secret="NOPE")],
            judge_completer=lambda _p: "{}",
            secret_reader=lambda _name: None,
        )
        assert graders == []

    def test_a_judge_naming_a_present_secret_gets_its_own_provider(self):
        graders = graders_from_spec(
            [entry(provider="openai", model="gpt-4o", api_key_secret="MY_KEY")],
            secret_reader=lambda name: "sk-test" if name == "MY_KEY" else None,
        )
        assert len(graders) == 1
        assert graders[0].config.provider == "openai"


def test_the_default_templates_all_build():
    for template in default_templates():
        graders = graders_from_spec(template["spec"], judge_completer=lambda _p: "{}")
        assert graders, template["name"]


def test_the_default_template_leaves_tool_checks_to_the_tool_graders():
    # paying a model to decide whether a required tool ran is slower, costlier
    # and less reliable than the grader that knows
    spec = json.loads(default_templates()[0]["spec"])
    names = set(spec[0]["criteria"])
    assert not names & {"tool_selection", "tool_execution", "efficiency"}

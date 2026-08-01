"""A judge configured with named criteria, weights and floors.

The two behaviours worth protecting: a weighted total cannot rescue a
catastrophic score on a criterion someone marked critical, and a judge that
answers only half the criteria is ungradable rather than averaged over whatever
came back.
"""

from __future__ import annotations

import json

import pytest

from hopsworks_agent_eval.evaluator_spec import SpecError, evaluators_from_spec
from hopsworks_agent_eval.judge_config import (
    FAILURE_CATEGORIES,
    PROVIDERS,
    JudgeConfigError,
    default_templates,
    parse_judge_config,
    render_prompt,
)
from hopsworks_agent_eval.judges import LlmJudgeEvaluator
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



def _expects(kwargs: dict) -> dict:
    """Legacy per-field kwargs, written as the expectations they now are.

    Expectations are keyed by the check that reads them, and a check's name
    defaults to its type — so `required_tools=["x"]` is what the `tool_call` and
    `tool_order` checks expect, and an expected answer is what every check that
    reads one expects. Written once here so a test can still say the thing it
    means.
    """
    import json as _json

    expectations = dict(kwargs.pop("expectations", {}) or {})
    answer = kwargs.pop("expected_output", None)
    if answer is not None:
        for name in ("exact_match", "contains", "pairwise", "llm_judge"):
            expectations.setdefault(name, answer)
    rubric = kwargs.pop("rubric", None)
    if rubric is not None:
        expectations["llm_judge"] = rubric
    required = kwargs.pop("required_tools", None)
    forbidden = kwargs.pop("forbidden_tools", None)
    if required is not None or forbidden is not None:
        expectations["tool_call"] = _json.dumps(
            {"required": required or [], "forbidden": forbidden or []}
        )
        expectations["tool_order"] = ", ".join(required or [])
        expectations["no_unnecessary_tools"] = ", ".join(required or [])
    if expectations:
        kwargs["expectations"] = expectations
    return kwargs

def task(**kwargs) -> Task:
    kwargs = _expects(kwargs)
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


def judge(reply: dict | str, **overrides) -> LlmJudgeEvaluator:
    text = reply if isinstance(reply, str) else json.dumps(reply)
    return LlmJudgeEvaluator(lambda _p: text, parse_judge_config(entry(**overrides)))


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
        evaluator = LlmJudgeEvaluator(
            lambda _p: (_ for _ in ()).throw(RuntimeError("rate limited")),
            parse_judge_config(entry()),
        )
        result = evaluator.grade(task(), trial(), None)
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
        evaluator = judge({"scores": {}}, inputs=["user_request", "agent_response", "tool_calls"])
        assert evaluator.needs_trace is True
        assert evaluator.grade(task(), trial(), None).ungradable

    def test_a_judge_not_reading_tool_calls_does_not_need_one(self):
        # otherwise it would go ungradable for want of a trace it never reads
        evaluator = judge({"scores": {"task_completion": 5, "correctness": 5, "safety": 5}})
        assert evaluator.needs_trace is False
        assert evaluator.grade(task(), trial(), None).passed


class TestSpecIntegration:
    def test_criteria_are_carried_onto_the_evaluator(self):
        evaluators = evaluators_from_spec([entry()], judge_completer=lambda _p: "{}")
        assert [c.name for c in evaluators[0].config.criteria] == [
            "task_completion", "correctness", "safety",
        ]

    def test_a_judge_with_no_criteria_gets_one_called_overall(self):
        # not a different class, not a different code path — the same judge
        # scoring a single unnamed thing
        evaluators = evaluators_from_spec(
            [{"type": "llm_judge"}], judge_completer=lambda _p: "{}"
        )
        assert [c.name for c in evaluators[0].config.effective_criteria()] == ["overall"]

    def test_every_judged_type_can_bring_its_own_model(self):
        # a pairwise comparison has no reason to be stuck with the project
        # default when a rubric judge is not
        for kind in ("pairwise", "tool_arguments_judge", "tool_result_used"):
            evaluators = evaluators_from_spec(
                [{"type": kind, "provider": "openai", "model": "gpt-4o",
                  "api_key_secret": "K"}],
                secret_reader=lambda name: "sk-test",
            )
            assert evaluators[0].model == "gpt-4o", kind

    def test_a_bad_judge_config_is_a_spec_error(self):
        # so it is refused at authoring time like every other malformed entry
        with pytest.raises(SpecError, match="provider must be"):
            evaluators_from_spec([entry(provider="mystery")], judge_completer=lambda _p: "")

    def test_a_judge_naming_a_missing_secret_is_skipped_not_failed(self):
        evaluators = evaluators_from_spec(
            [entry(provider="openai", model="gpt-4o", api_key_secret="NOPE")],
            judge_completer=lambda _p: "{}",
            secret_reader=lambda _name: None,
        )
        assert evaluators == []

    def test_a_judge_naming_a_present_secret_gets_its_own_provider(self):
        evaluators = evaluators_from_spec(
            [entry(provider="openai", model="gpt-4o", api_key_secret="MY_KEY")],
            secret_reader=lambda name: "sk-test" if name == "MY_KEY" else None,
        )
        assert len(evaluators) == 1
        assert evaluators[0].config.provider == "openai"


def test_the_default_templates_all_build():
    for template in default_templates():
        evaluators = evaluators_from_spec(template["spec"], judge_completer=lambda _p: "{}")
        assert evaluators, template["name"]


def test_the_default_template_leaves_tool_checks_to_the_tool_evaluators():
    # paying a model to decide whether a required tool ran is slower, costlier
    # and less reliable than the evaluator that knows
    spec = json.loads(default_templates()[0]["spec"])
    names = set(spec[0]["criteria"])
    assert not names & {"tool_selection", "tool_execution", "efficiency"}


class TestProviders:
    def test_every_provider_maps_to_an_adapter_that_exists(self):
        # the point of the registry: adding a provider is a row, not a client
        for key, entry in PROVIDERS.items():
            assert entry["adapter"] in ("openai", "anthropic"), key

    def test_only_anthropic_needs_its_own_client(self):
        anthropic = [k for k, v in PROVIDERS.items() if v["adapter"] == "anthropic"]
        assert anthropic == ["anthropic"]

    def test_every_openai_shaped_provider_knows_where_to_call(self):
        # except openai itself, whose SDK default is correct, and custom, which
        # is refused without one
        for key, entry in PROVIDERS.items():
            if entry["adapter"] == "openai" and key not in ("openai", "custom"):
                assert entry["base_url"].startswith("https://"), key

    def test_a_custom_provider_needs_a_base_url(self):
        with pytest.raises(JudgeConfigError, match="needs a base_url"):
            parse_judge_config({"type": "llm_judge", "provider": "custom"})

    def test_a_custom_provider_with_a_base_url_is_fine(self):
        config = parse_judge_config({
            "type": "llm_judge", "provider": "custom",
            "base_url": "https://my-vllm.internal/v1",
        })
        assert config.base_url == "https://my-vllm.internal/v1"

    def test_every_provider_parses(self):
        for key in PROVIDERS:
            entry = {"type": "llm_judge", "provider": key}
            if key == "custom":
                entry["base_url"] = "https://x/v1"
            assert parse_judge_config(entry).provider == key


class TestWhereAJudgesKeyComesFrom:
    """Every provider's own SDK reads a conventional variable, so a key already
    set for anything else is found without being named again."""

    def config(self, provider="anthropic", secret=""):
        from hopsworks_agent_eval.judge_config import JudgeConfig

        return JudgeConfig(provider=provider, api_key_secret=secret)

    def test_every_provider_but_custom_names_its_variable(self):
        # the whole point: without one there is nothing to look for
        from hopsworks_agent_eval.judge_config import PROVIDERS

        missing = [
            name for name, spec in PROVIDERS.items()
            if name != "custom" and not spec.get("env_var")
        ]
        assert missing == []

    def test_the_named_secret_wins_over_the_environment(self, monkeypatch):
        # explicit beats ambient: a release gate naming a key means that key
        from hopsworks_agent_eval.judge_config import api_key_for

        monkeypatch.setenv("ANTHROPIC_API_KEY", "ambient")
        assert api_key_for(
            self.config(secret="GATE_KEY"), lambda name: "named"
        ) == "named"

    def test_the_environment_variable_is_used_when_no_secret_names_one(
        self, monkeypatch
    ):
        from hopsworks_agent_eval.judge_config import api_key_for

        monkeypatch.setenv("OPENAI_API_KEY", "from-env")
        assert api_key_for(self.config(provider="openai"), None) == "from-env"

    def test_a_secret_of_the_variables_name_is_the_last_place_looked(
        self, monkeypatch
    ):
        # the common case that silently skipped every judge: stored as a project
        # secret under the obvious name
        from hopsworks_agent_eval.judge_config import api_key_for

        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        assert api_key_for(
            self.config(), lambda name: "from-secret" if name == "ANTHROPIC_API_KEY" else None
        ) == "from-secret"

    def test_an_empty_named_secret_does_not_stop_the_fallback(self, monkeypatch):
        # a secret that exists and is blank is not a key
        from hopsworks_agent_eval.judge_config import api_key_for

        monkeypatch.setenv("ANTHROPIC_API_KEY", "from-env")
        assert api_key_for(self.config(secret="EMPTY"), lambda name: "") == "from-env"

    def test_a_custom_provider_has_nowhere_to_look(self, monkeypatch):
        # a gateway has no conventional variable, so it has to be told
        from hopsworks_agent_eval.judge_config import api_key_for

        assert api_key_for(self.config(provider="custom"), None) is None

    def test_the_places_looked_are_reportable(self):
        # so a skipped judge says what to set, rather than that something is absent
        from hopsworks_agent_eval.judge_config import api_key_source

        where = api_key_source(self.config(secret="GATE_KEY"))
        assert "GATE_KEY" in where and "ANTHROPIC_API_KEY" in where

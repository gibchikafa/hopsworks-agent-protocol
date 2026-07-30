"""The graders that close the design doc's list, and the pass policy.

Two themes, both about not answering a question nobody answered: a state check
that could not run is ungradable rather than passing, and a task that asked for
a person is held open rather than settled by the other graders agreeing.
"""

from __future__ import annotations

import json

import pytest

from hopsworks_agent_eval.graders import (
    HumanReviewGrader,
    SqlStateGrader,
    awaits_review,
    verdict,
)
from hopsworks_agent_eval.judges import PairwiseGrader
from hopsworks_agent_eval.models import GraderResult, PassPolicy, Task, Trial


def task(**kwargs) -> Task:
    return Task(
        task_id="t1",
        input_messages=json.dumps([{"role": "user", "content": "cancel 4471"}]),
        **kwargs,
    )


def trial(output: str = "done") -> Trial:
    return Trial(
        trial_id="x", run_id="r", task_id="t1", task_version=1,
        trial_index=0, deployment_id=1, final_output=output,
    )


class TestSqlStateGrader:
    def test_state_matching_the_expectation_passes(self):
        grader = SqlStateGrader(
            "SELECT status FROM orders WHERE id = '4471'",
            expect="cancelled",
            query=lambda _sql: "cancelled",
        )
        result = grader.grade(task(), trial(), None)
        assert result.passed and result.score == 1.0

    def test_an_agent_that_only_claims_to_have_acted_fails(self):
        # the whole point: the answer says "cancelled", the table says otherwise
        grader = SqlStateGrader(
            "SELECT status FROM orders", expect="cancelled", query=lambda _s: "open"
        )
        result = grader.grade(task(), trial("I've cancelled order 4471."), None)
        assert result.passed is False
        assert "expected 'cancelled', found 'open'" in result.reason

    def test_no_query_function_is_ungradable_not_a_pass(self):
        # "I could not check" must never read as "the state was right"
        result = SqlStateGrader("SELECT 1", expect=1).grade(task(), trial(), None)
        assert result.ungradable and result.passed is False

    def test_a_failing_query_is_ungradable_not_a_failure(self):
        def boom(_sql: str):
            raise RuntimeError("connection refused")

        result = SqlStateGrader("SELECT 1", expect=1, query=boom).grade(
            task(), trial(), None
        )
        assert result.ungradable
        assert "connection refused" in result.reason

    @pytest.mark.parametrize(
        "returned",
        [
            "cancelled",
            ["cancelled"],
            [["cancelled"]],
            [("cancelled", "other")],
        ],
    )
    def test_reads_the_first_cell_whatever_shape_the_session_returns(self, returned):
        grader = SqlStateGrader(
            "SELECT 1", expect="cancelled", query=lambda _s: returned
        )
        assert grader.grade(task(), trial(), None).passed

    def test_an_empty_result_is_not_a_match(self):
        grader = SqlStateGrader("SELECT 1", expect="cancelled", query=lambda _s: [])
        assert grader.grade(task(), trial(), None).passed is False

    def test_numbers_compare_across_types(self):
        # a driver returning 30 and a task expecting "30" agree
        grader = SqlStateGrader("SELECT 1", expect="30", query=lambda _s: 30)
        assert grader.grade(task(), trial(), None).passed


class TestHumanReviewGrader:
    def test_it_defers_rather_than_judging(self):
        result = HumanReviewGrader().grade(task(), trial(), None)
        assert result.ungradable
        assert result.assertions["awaiting_review"] is True

    def test_the_prompt_reaches_the_reviewer(self):
        result = HumanReviewGrader("Is the tone right for a refund refusal?").grade(
            task(), trial(), None
        )
        assert "tone" in result.reason

    def test_awaits_review_spots_it_among_other_results(self):
        passing = GraderResult("contains", "contains", 1.0, True)
        pending = HumanReviewGrader().grade(task(), trial(), None)
        assert awaits_review([passing, pending]) is True
        assert awaits_review([passing]) is False


class TestPairwiseGrader:
    def judge(self, winner: str):
        return lambda _prompt: json.dumps({"winner": winner, "reason": "because"})

    def test_the_candidate_winning_passes(self):
        grader = PairwiseGrader(self.judge("b"), reference="the old answer")
        result = grader.grade(task(), trial(), None)
        assert result.passed and result.score == 1.0
        assert result.assertions["winner"] == "b"

    def test_a_tie_passes(self):
        # a tie is the judge saying it cannot separate them; failing on that
        # makes the grader a coin toss on every equivalent answer
        grader = PairwiseGrader(self.judge("tie"), reference="the old answer")
        result = grader.grade(task(), trial(), None)
        assert result.passed and result.score == 0.5

    def test_the_reference_winning_fails(self):
        grader = PairwiseGrader(self.judge("a"), reference="the old answer")
        assert grader.grade(task(), trial(), None).passed is False

    def test_it_falls_back_to_the_expected_output_as_the_reference(self):
        grader = PairwiseGrader(self.judge("b"))
        assert grader.grade(task(expected_output="the good one"), trial(), None).passed

    def test_no_reference_is_ungradable(self):
        result = PairwiseGrader(self.judge("b")).grade(task(), trial(), None)
        assert result.ungradable
        assert "no reference" in result.reason

    def test_no_answer_is_ungradable(self):
        grader = PairwiseGrader(self.judge("b"), reference="ref")
        result = grader.grade(task(), trial(output=""), None)
        assert result.ungradable

    def test_an_unusable_verdict_is_ungradable_not_a_failure(self):
        # blaming the agent for a judge that returned prose is the error this
        # whole family of graders is most prone to
        grader = PairwiseGrader(lambda _p: "I think B is nicer", reference="ref")
        result = grader.grade(task(), trial(), None)
        assert result.ungradable and result.passed is False

    def test_the_judge_model_is_recorded(self):
        grader = PairwiseGrader(self.judge("b"), reference="ref", model="some-model")
        assert grader.grade(task(), trial(), None).assertions["judge_model"] == (
            "some-model"
        )


class TestPassPolicy:
    def results(self, *passed: bool) -> list[GraderResult]:
        return [
            GraderResult(f"g{i}", "contains", 1.0 if p else 0.0, p)
            for i, p in enumerate(passed)
        ]

    def test_all_is_the_default_and_needs_every_grader(self):
        assert verdict(self.results(True, True)) is True
        assert verdict(self.results(True, False)) is False

    def test_any_passes_on_one(self):
        assert verdict(self.results(True, False), PassPolicy.ANY) is True
        assert verdict(self.results(False, False), PassPolicy.ANY) is False

    def test_threshold_reads_the_mean_score(self):
        scored = [
            GraderResult("a", "llm_judge", 0.8, True),
            GraderResult("b", "llm_judge", 0.4, False),
        ]
        assert verdict(scored, PassPolicy.THRESHOLD, threshold=0.5) is True
        assert verdict(scored, PassPolicy.THRESHOLD, threshold=0.7) is False

    def test_a_policy_given_as_a_string_works(self):
        # it arrives from the REST payload as one
        assert verdict(self.results(True, False), "any") is True

    def test_nothing_gradable_is_none_under_every_policy(self):
        ungradable = [GraderResult("a", "x", 0.0, False, ungradable=True)]
        for policy in ("all", "any", "threshold"):
            assert verdict(ungradable, policy) is None

    def test_ungradable_results_are_excluded_rather_than_counted(self):
        mixed = [
            GraderResult("a", "contains", 1.0, True),
            GraderResult("b", "tool_call", 0.0, False, ungradable=True),
        ]
        assert verdict(mixed) is True
        assert verdict(mixed, PassPolicy.THRESHOLD, threshold=0.9) is True

    def test_an_unknown_policy_falls_back_to_all(self):
        # a stored value from a newer version must not silently loosen the bar
        assert verdict(self.results(True, False), "whatever-is-next") is False

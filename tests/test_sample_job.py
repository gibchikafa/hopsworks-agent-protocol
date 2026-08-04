"""Online evaluation: sampling production traffic and grading it.

These exist because the module had none, and rotted silently for it. Three
things moved under it between being written and being run — `Task` lost
`rubric` for `expectations`, judge keys stopped coming from a secrets API, and
a job stopped having an API key — and every one of them was a `TypeError` or a
`KeyError` on the first trace, in a job nobody would look at until someone
asked why the dashboard was empty.

So the seams tested here are the ones that break when something moves
underneath: constructing a `Task` and a `Trial`, what the judge grades against,
and what the trial's status becomes.
"""

from __future__ import annotations

import json
import random
from datetime import datetime, timedelta, timezone

from hopsworks_agent_eval.evaluators import EvaluatorResult
from hopsworks_agent_eval.models import TraceStatus, TrialStatus
from hopsworks_agent_eval.sample_job import (
    GENERIC_RUBRIC,
    JUDGE_NAME,
    as_trial,
    choose,
    grade_trace,
    question_and_answer,
    reference_free_evaluators,
    run_sample,
    within_window,
)

NOW = datetime(2026, 8, 4, 12, 0, tzinfo=timezone.utc)


def summary(trace_id: str, hours_ago: float) -> dict:
    when = NOW - timedelta(hours=hours_ago)
    return {"traceId": trace_id, "createdAt": int(when.timestamp() * 1000)}


def detail(question: str = "who sings this?", answer: str = "UB40") -> dict:
    return {
        "spans": [
            {"spanId": "root", "startTimeNs": 1, "messages": json.dumps([
                {"role": "user", "content": question},
                {"role": "assistant", "content": answer},
            ])},
        ]
    }


class StubJudge:
    """A completer returning the shape the judge actually asks for.

    Per-criterion scores, because that is what `_output_shape` prompts for and
    what the judge refuses to guess at: a judge that answered with one bare
    number is treated as ungradable rather than scored, so a stub returning one
    would pass this file while telling us nothing about the real path.
    """

    def __init__(self, score: float = 5.0, criterion: str = "overall"):
        self.prompts: list[str] = []
        self._score = score
        self._criterion = criterion

    def __call__(self, prompt: str) -> str:
        self.prompts.append(prompt)
        return json.dumps({
            "scores": {self._criterion: self._score},
            "reasoning": {self._criterion: "because"},
        })


class TestChoosingWhatToGrade:
    def test_only_traces_inside_the_window_are_candidates(self):
        summaries = [summary("old", 100), summary("new", 1)]
        assert [s["traceId"] for s in within_window(summaries, 24.0, NOW)] == ["new"]

    def test_a_trace_with_no_timestamp_is_not_guessed_into_the_window(self):
        # its age is unknown, and treating unknown as recent would quietly widen
        # whatever window someone asked for
        assert within_window([{"traceId": "t"}], 24.0, NOW) == []

    def test_the_sample_is_random_rather_than_the_newest(self):
        # the most recent traces are a sample of who was using it in the last
        # hour, not a sample of behaviour
        summaries = [summary(str(i), i) for i in range(20)]
        first = choose(summaries, 5, random.Random(1))
        assert len(first) == 5
        assert [s["traceId"] for s in first] != [s["traceId"] for s in summaries[:5]]

    def test_asking_for_more_than_exists_grades_everything(self):
        summaries = [summary("a", 1), summary("b", 2)]
        assert len(choose(summaries, 50, random.Random(1))) == 2


class TestReadingTheTranscript:
    def test_the_question_and_the_final_answer_come_off_the_trace(self):
        assert question_and_answer(detail()) == ("who sings this?", "UB40")

    def test_the_last_assistant_turn_wins(self):
        # an agent that spoke twice said the second thing to the customer
        trace = {"spans": [{"spanId": "r", "startTimeNs": 1, "messages": json.dumps([
            {"role": "user", "content": "q"},
            {"role": "assistant", "content": "first"},
            {"role": "assistant", "content": "second"},
        ])}]}
        assert question_and_answer(trace)[1] == "second"

    def test_an_unparseable_span_is_skipped_rather_than_fatal(self):
        trace = {"spans": [
            {"spanId": "a", "startTimeNs": 1, "messages": "not json"},
            {"spanId": "b", "startTimeNs": 2, "messages": json.dumps(
                [{"role": "user", "content": "q"}])},
        ]}
        assert question_and_answer(trace)[0] == "q"

    def test_a_trace_carrying_no_messages_is_empty_rather_than_an_error(self):
        assert question_and_answer({"spans": [{"spanId": "a"}]}) == ("", "")


class TestWhatCanGradeProductionTraffic:
    def test_without_a_judge_only_the_trace_checks_run(self):
        # nothing here can say whether an answer was good without a model, and
        # inventing a verdict would be worse than reporting fewer checks
        [only] = reference_free_evaluators()
        assert only.type == "no_tool_error"

    def test_a_judge_is_included_when_one_is_configured(self):
        kinds = [e.type for e in reference_free_evaluators(StubJudge())]
        assert kinds == ["llm_judge", "no_tool_error"]

    def test_no_reference_based_evaluator_is_offered(self):
        # production declares no expected answer, and `contains` with nothing to
        # contain passes everything rather than being lenient
        kinds = {e.type for e in reference_free_evaluators(StubJudge())}
        assert not kinds & {"contains", "exact_match", "regex", "tool_call",
                            "tool_order", "json_schema"}


class TestGradingOneTrace:
    def test_the_rubric_reaches_the_judge(self):
        # a judge reads its expectation by its own name -- the same mechanism a
        # suite uses, which is what stops this needing a special case
        judge = StubJudge()
        trial = grade_trace(reference_free_evaluators(judge), "run1", 3, "t1",
                            detail(), None, "must name the artist")
        assert "must name the artist" in judge.prompts[0]
        assert trial.trial_id == "run1/t1"

    def test_a_sample_with_no_rubric_still_grades_against_something(self):
        judge = StubJudge()
        grade_trace(reference_free_evaluators(judge), "run1", 3, "t1", detail(),
                    None, "")
        assert GENERIC_RUBRIC in judge.prompts[0]

    def test_the_task_carries_the_rubric_under_the_judges_name(self):
        # pinned because `Task` losing `rubric` for `expectations` is exactly the
        # change that broke this module the first time
        trial = grade_trace(reference_free_evaluators(StubJudge()), "r", 1, "t",
                            detail(), None, "the rubric")
        assert JUDGE_NAME == "online_judge"
        assert trial.task_id == "t"

    def test_the_verdict_settles_the_trial_status(self):
        # a fixed PASSED would make every sample report a pass rate of exactly
        # 1.0, which is a confidently wrong number rather than an absent one
        trace = {"trace_id": "t", "tool_calls": [], "tool_error_count": 0}
        passing = grade_trace(reference_free_evaluators(StubJudge(5.0)), "r", 1,
                              "t", detail(), trace, "rubric")
        failing = grade_trace(reference_free_evaluators(StubJudge(1.0)), "r", 1,
                              "t2", detail(), trace, "rubric")
        assert passing.status is TrialStatus.PASSED
        assert failing.status is TrialStatus.FAILED

    def test_a_judge_that_answers_with_one_bare_number_is_ungradable(self):
        # scoring it anyway would blame the agent for the judge -- and a stub
        # returning that shape is how this file could have passed while the real
        # path produced nothing
        class Bare:
            def __call__(self, prompt: str) -> str:
                return json.dumps({"score": 5.0})

        trial = grade_trace(reference_free_evaluators(Bare()), "r", 1, "t",
                            detail(), None, "rubric")
        assert all(r.ungradable for r in trial.evaluator_results)

    def test_nothing_gradable_is_not_counted_as_a_failure(self):
        # no judge configured and no trace: the trial says nothing about the
        # agent either way, and calling that a failure makes an unconfigured
        # sample look like a broken agent
        trial = grade_trace([], "r", 1, "t", detail(), None)
        assert trial.status is not TrialStatus.FAILED

    def test_a_graded_trace_records_that_it_had_one(self):
        trace = {"trace_id": "t", "tool_calls": [], "tool_error_count": 0,
                 "input_tokens": 10, "output_tokens": 4}
        trial = as_trial("r", 1, "t", "answer", trace)
        assert trial.trace_status is TraceStatus.RECEIVED
        assert (trial.input_tokens, trial.output_tokens) == (10, 4)


class FakeResponse:
    def __init__(self, body):
        self._body = body

    def raise_for_status(self):
        return None

    def json(self):
        return self._body


class FakeSession:
    def __init__(self, summaries, details):
        self._summaries = summaries
        self._details = details

    def get(self, url, params=None, timeout=None):
        if url.endswith("/traces"):
            return FakeResponse({"items": self._summaries})
        return FakeResponse(self._details[url.rsplit("/", 1)[-1]])


class FakeClient:
    def __init__(self, traces=None, fails=()):
        self._traces = traces or {}
        self._fails = set(fails)

    def fetch_trace(self, trace_id):
        if trace_id in self._fails:
            raise RuntimeError("unreadable")
        return self._traces.get(trace_id)


def sample_run(**overrides):
    run = {"runId": "run1", "deploymentId": 3, "nTrials": 10,
           "sampleSinceHours": 24.0, "sampleRubric": "be helpful"}
    run.update(overrides)
    return run


class TestARunOverProduction:
    def test_every_sampled_trace_becomes_a_trial(self):
        session = FakeSession([summary("a", 1), summary("b", 2)],
                              {"a": detail(), "b": detail()})
        result = run_sample(FakeClient(), session, "https://h", 1, sample_run(),
                            reference_free_evaluators(StubJudge()))
        assert {t.task_id for t in result.trials} == {"a", "b"}
        assert result.status == "SUCCEEDED"

    def test_a_run_over_production_names_no_suite(self):
        # nothing was executed against a frozen set of cases, and an empty suite
        # id is the record of that rather than a missing value
        session = FakeSession([summary("a", 1)], {"a": detail()})
        result = run_sample(FakeClient(), session, "https://h", 1, sample_run(),
                            reference_free_evaluators())
        assert result.suite_id == ""

    def test_a_quiet_window_is_not_a_failed_run(self):
        # an agent with no traffic overnight is a normal state; reporting it as
        # FAILED would page someone about it
        session = FakeSession([summary("old", 500)], {})
        result = run_sample(FakeClient(), session, "https://h", 1, sample_run(),
                            reference_free_evaluators())
        assert (result.status, result.trials) == ("SUCCEEDED", [])

    def test_one_unreadable_trace_does_not_end_the_run(self):
        session = FakeSession([summary("a", 1), summary("b", 1)],
                              {"a": detail(), "b": detail()})
        result = run_sample(FakeClient(fails={"a"}), session, "https://h", 1,
                            sample_run(), reference_free_evaluators())
        assert [t.task_id for t in result.trials] == ["b"]

    def test_the_sample_size_comes_off_the_run_row(self):
        # the row is what the job is handed, so a scheduled sample and a
        # hand-started one cannot ask for different things
        session = FakeSession([summary(str(i), 1) for i in range(10)],
                              {str(i): detail() for i in range(10)})
        result = run_sample(FakeClient(), session, "https://h", 1,
                            sample_run(nTrials=3), reference_free_evaluators(),
                            random.Random(1))
        assert len(result.trials) == 3


class TestResultsAreWritable:
    def test_a_sample_writes_metrics_without_a_suite(self):
        # _write_results read run["suiteId"] unconditionally, which is a KeyError
        # for a run that has none -- after every trace has been judged, which is
        # the most expensive moment to find out
        from hopsworks_agent_eval.metrics import run_metrics

        trial = as_trial("run1", 3, "t", "answer", None)
        trial.status = TrialStatus.PASSED
        trial.evaluator_results = [
            EvaluatorResult(evaluator_name=JUDGE_NAME, evaluator_type="llm_judge",
                            score=1.0, passed=True)
        ]
        rows = run_metrics("run1", "", 3, [trial])
        assert any(r["metric_name"] == "pass_rate" for r in rows)
        assert all(r["suite_id"] == "" for r in rows)

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
    as_trial,
    oldest_first,
    evaluators_for,
    grade_trace,
    question_and_answer,
    run_sample,
    within_window,
)

#: A judge that grades against its own criteria, which is what makes a check
#: usable on production traffic — the server refuses one that cannot.
JUDGE_SPEC = json.dumps([{
    "type": "llm_judge",
    "name": "faithfulness",
    "criteria": {"supported": {"weight": 1}},
}])
TRACE_ONLY_SPEC = json.dumps([{"type": "no_tool_error", "name": "no_tool_errored"}])


def with_judge(judge):
    return evaluators_for({"sampleEvaluators": JUDGE_SPEC}, judge_completer=judge)


def trace_only():
    return evaluators_for({"sampleEvaluators": TRACE_ONLY_SPEC})

#: A fixed reference for the window tests, which are handed their own `now`.
NOW = datetime(2026, 8, 4, 12, 0, tzinfo=timezone.utc)


def summary(trace_id: str, hours_ago: float, since: datetime | None = None) -> dict:
    """A trace summary, that many hours before `since`.

    Defaults to real now rather than a fixed date: building these against one made
    every run_sample test pass on the day they were written and fail two days
    later. A test that expires is worse than no test, because it fails somewhere
    unrelated to whatever broke.
    """
    when = (since or datetime.now(tz=timezone.utc)) - timedelta(hours=hours_ago)
    return {"traceId": trace_id, "createdAt": int(when.timestamp() * 1000)}


def ms(hours_ago: float) -> float:
    return (datetime.now(tz=timezone.utc) - timedelta(hours=hours_ago)).timestamp() * 1000


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
        summaries = [summary("old", 100, NOW), summary("new", 1, NOW)]
        window = within_window(summaries, ms(24) - 0, ms(0))
        assert [s["traceId"] for s in within_window(
            summaries, (NOW - timedelta(hours=24)).timestamp() * 1000,
            NOW.timestamp() * 1000)] == ["new"]
        assert window is not None

    def test_the_window_is_half_open_so_runs_do_not_overlap(self):
        # a trace graded by the run that ended at T must not be graded again by
        # the run that starts at T: monitoring counts each conversation once
        boundary = NOW.timestamp() * 1000
        at_boundary = {"traceId": "edge", "createdAt": int(boundary)}
        assert within_window([at_boundary], boundary, boundary + 1000) == []
        assert within_window([at_boundary], boundary - 1000, boundary) == [at_boundary]

    def test_a_trace_with_no_timestamp_is_not_guessed_into_the_window(self):
        # its age is unknown, and treating unknown as recent would quietly widen
        # whatever window someone asked for
        assert within_window([{"traceId": "t"}], ms(24), ms(0)) == []

    def test_everything_in_the_window_is_graded(self):
        # the window already says what is new, so choosing a subset of it would
        # decide on someone's behalf how much of their own traffic to look at
        summaries = [summary(str(i), i, NOW) for i in range(20)]
        assert len(oldest_first(summaries, 0)) == 20

    def test_a_ceiling_cuts_the_newest_not_the_middle(self):
        # so what is left ungraded is contiguous and the watermark can stop in
        # front of it -- a ceiling defers work rather than skipping it
        summaries = [summary(str(i), i, NOW) for i in range(10)]
        graded = oldest_first(summaries, 3)
        assert [s["traceId"] for s in graded] == ["9", "8", "7"]

    def test_asking_for_more_than_exists_grades_everything(self):
        summaries = [summary("a", 1, NOW), summary("b", 2, NOW)]
        assert len(oldest_first(summaries, 50)) == 2


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
    def test_the_checks_come_from_the_run_rather_than_being_hardcoded(self):
        # the same evaluators_from_spec a suite run uses: a check does not need to
        # know whether the conversation it reads was authored or served
        kinds = [e.type for e in with_judge(StubJudge())]
        assert kinds == ["llm_judge"]

    def test_a_trace_only_check_needs_no_judge_at_all(self):
        [only] = trace_only()
        assert only.type == "no_tool_error"

    def test_an_empty_spec_grades_with_nothing(self):
        # the server refuses this, so reaching it means something went wrong
        # upstream -- reporting nothing beats inventing a default check
        assert evaluators_for({"sampleEvaluators": "[]"}) == []


class TestGradingOneTrace:
    def test_the_judges_criteria_are_what_it_grades_against(self):
        # a criteria-based judge needs nothing from the task, which is exactly
        # what makes it usable where nobody wrote an expected answer
        judge = StubJudge(criterion="supported")
        trial = grade_trace(with_judge(judge), "run1", 3, "t1", detail(), None)
        assert "supported" in judge.prompts[0]
        assert trial.trial_id == "run1/t1"
        assert not any(r.ungradable for r in trial.evaluator_results)

    def test_the_task_carries_no_expectations_at_all(self):
        # production has none, and inventing one would anchor the judge on an
        # answer nobody wrote
        judge = StubJudge(criterion="supported")
        grade_trace(with_judge(judge), "r", 1, "t", detail(), None)
        assert "expected" not in judge.prompts[0].lower().split("<user")[0]

    def test_the_verdict_settles_the_trial_status(self):
        # a fixed PASSED would make every sample report a pass rate of exactly
        # 1.0, which is a confidently wrong number rather than an absent one
        trace = {"trace_id": "t", "tool_calls": [], "tool_error_count": 0}
        passing = grade_trace(with_judge(StubJudge(5.0, "supported")), "r", 1,
                              "t", detail(), trace)
        failing = grade_trace(with_judge(StubJudge(1.0, "supported")), "r", 1,
                              "t2", detail(), trace)
        assert passing.status is TrialStatus.PASSED
        assert failing.status is TrialStatus.FAILED

    def test_a_judge_that_answers_with_one_bare_number_is_ungradable(self):
        # scoring it anyway would blame the agent for the judge -- and a stub
        # returning that shape is how this file could have passed while the real
        # path produced nothing
        class Bare:
            def __call__(self, prompt: str) -> str:
                return json.dumps({"score": 5.0})

        trial = grade_trace(with_judge(Bare()), "r", 1, "t", detail(), None)
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
    run = {"runId": "run1", "deploymentId": 3, "nTrials": 500,
           "sampleFrom": ms(48), "sampleTo": ms(0) + 60_000,
           "sampleEvaluators": TRACE_ONLY_SPEC}
    run.update(overrides)
    return run


class TestARunOverProduction:
    def test_every_sampled_trace_becomes_a_trial(self):
        session = FakeSession([summary("a", 1), summary("b", 2)],
                              {"a": detail(), "b": detail()})
        result = run_sample(FakeClient(), session, "https://h", 1, sample_run(),
                            with_judge(StubJudge()))
        assert {t.task_id for t in result.trials} == {"a", "b"}
        assert result.status == "SUCCEEDED"

    def test_a_run_over_production_names_no_suite(self):
        # nothing was executed against a frozen set of cases, and an empty suite
        # id is the record of that rather than a missing value
        session = FakeSession([summary("a", 1)], {"a": detail()})
        result = run_sample(FakeClient(), session, "https://h", 1, sample_run(),
                            trace_only())
        assert result.suite_id == ""

    def test_a_quiet_window_is_not_a_failed_run(self):
        # an agent with no traffic overnight is a normal state; reporting it as
        # FAILED would page someone about it
        session = FakeSession([summary("old", 500)], {})
        result = run_sample(FakeClient(), session, "https://h", 1, sample_run(),
                            trace_only())
        assert (result.status, result.trials) == ("SUCCEEDED", [])

    def test_one_unreadable_trace_does_not_end_the_run(self):
        session = FakeSession([summary("a", 1), summary("b", 1)],
                              {"a": detail(), "b": detail()})
        result = run_sample(FakeClient(fails={"a"}), session, "https://h", 1,
                            sample_run(), trace_only())
        assert [t.task_id for t in result.trials] == ["b"]

    def test_the_ceiling_comes_off_the_run_row(self):
        # the row is what the job is handed, so a scheduled monitor and a
        # hand-started one cannot bound themselves differently
        session = FakeSession([summary(str(i), i + 1) for i in range(10)],
                              {str(i): detail() for i in range(10)})
        result = run_sample(FakeClient(), session, "https://h", 1,
                            sample_run(nTrials=3), trace_only())
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
            EvaluatorResult(evaluator_name="faithfulness", evaluator_type="llm_judge",
                            score=1.0, passed=True)
        ]
        rows = run_metrics("run1", "", 3, [trial])
        assert any(r["metric_name"] == "pass_rate" for r in rows)
        assert all(r["suite_id"] == "" for r in rows)


class TestWhereTheNextRunStarts:
    """The watermark, which is what makes a schedule cover each trace once."""

    def test_it_reaches_the_end_of_the_window_when_all_of_it_was_graded(self):
        from hopsworks_agent_eval.sample_job import graded_through

        end = ms(0)
        graded = [summary("a", 3), summary("b", 1)]
        assert graded_through(graded, end) <= end

    def test_it_stops_at_the_last_graded_trace_when_the_ceiling_cut_in(self):
        # advancing to the window's end regardless would step over traces nobody
        # looked at, and nothing would ever report them
        from hopsworks_agent_eval.sample_job import graded_through

        end = ms(0)
        graded = [summary("old", 10)]
        assert graded_through(graded, end) < end - 3600_000

    def test_an_empty_window_still_advances(self):
        # nothing arrived, so there is nothing to come back for; holding the
        # watermark still would re-scan the same empty range forever
        from hopsworks_agent_eval.sample_job import graded_through

        end = ms(0)
        assert graded_through([], end) == end

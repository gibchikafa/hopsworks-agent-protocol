"""Graders: given a task, a trial and (maybe) a trace, produce a score.

Two rules run through all of them.

**A grader never raises into the runner.** One badly written custom grader must
not fail a whole run, so :func:`run_graders` catches and converts to a
``GRADER_ERROR`` result. That failure is the harness's, not the agent's, and is
excluded from pass rates.

**No trace is not a failing score.** Spans reach the feature store
asynchronously and sometimes never arrive at all — the sidecar logs and drops
on insert failure. A trajectory grader handed ``trace=None`` must return
``ungradable``, because scoring it zero would report a broken observability
pipeline as a broken agent, and that is the kind of wrong answer that gets a
good deployment blocked.
"""

from __future__ import annotations

import json
import re
from typing import Any, Callable, Protocol, Sequence

from .models import GraderResult, Task, Trial

# A trace as the runner assembles it: the `agent_trace_features` row plus the
# raw spans, so trajectory graders can inspect tool order.
Trace = dict[str, Any]


class Grader(Protocol):
    name: str
    type: str
    # True when this grader reads the trajectory rather than only the answer,
    # so the runner knows which graders to mark ungradable without a trace.
    needs_trace: bool

    def grade(self, task: Task, trial: Trial, trace: Trace | None) -> GraderResult: ...


def _ungradable(name: str, kind: str, reason: str) -> GraderResult:
    return GraderResult(
        grader_name=name,
        grader_type=kind,
        score=0.0,
        passed=False,
        reason=reason,
        ungradable=True,
    )


class ExactMatchGrader:
    """Final answer equals the expected output, ignoring surrounding space."""

    type = "exact_match"
    needs_trace = False

    def __init__(self, name: str = "exact_match", case_sensitive: bool = False):
        self.name = name
        self.case_sensitive = case_sensitive

    def grade(self, task: Task, trial: Trial, trace: Trace | None) -> GraderResult:
        expected, actual = task.expected_output.strip(), trial.final_output.strip()
        if not self.case_sensitive:
            expected, actual = expected.lower(), actual.lower()
        passed = expected == actual
        return GraderResult(
            self.name, self.type, 1.0 if passed else 0.0, passed,
            "exact match" if passed else f"expected {task.expected_output!r}",
        )


class ContainsGrader:
    """The answer mentions the expected string.

    Weaker than exact match and much more useful for free-text answers, where
    an exact match asserts the model's phrasing rather than its correctness.
    """

    type = "contains"
    needs_trace = False

    def __init__(self, name: str = "contains", expected: str | None = None):
        self.name = name
        self.expected = expected

    def grade(self, task: Task, trial: Trial, trace: Trace | None) -> GraderResult:
        needle = (self.expected if self.expected is not None else task.expected_output).strip()
        passed = needle.lower() in trial.final_output.lower()
        return GraderResult(
            self.name, self.type, 1.0 if passed else 0.0, passed,
            "found" if passed else f"{needle!r} not in the answer",
        )


class RegexGrader:
    type = "regex"
    needs_trace = False

    def __init__(self, pattern: str, name: str = "regex", should_match: bool = True):
        self.name = name
        self.pattern = re.compile(pattern)
        self.should_match = should_match

    def grade(self, task: Task, trial: Trial, trace: Trace | None) -> GraderResult:
        matched = bool(self.pattern.search(trial.final_output))
        passed = matched is self.should_match
        return GraderResult(
            self.name, self.type, 1.0 if passed else 0.0, passed,
            f"pattern {'matched' if matched else 'did not match'}",
            {"matched": matched},
        )


class JsonSchemaGrader:
    """The answer parses as JSON and carries the required keys.

    Deliberately shallow: full JSON Schema would be a dependency, and the
    common failure this catches is an agent that stopped emitting structured
    output at all.
    """

    type = "json_schema"
    needs_trace = False

    def __init__(self, required_keys: Sequence[str], name: str = "json_schema"):
        self.name = name
        self.required_keys = list(required_keys)

    def grade(self, task: Task, trial: Trial, trace: Trace | None) -> GraderResult:
        try:
            parsed = json.loads(trial.final_output)
        except (ValueError, TypeError):
            return GraderResult(
                self.name, self.type, 0.0, False, "answer is not valid JSON",
                {"parsed": False},
            )
        if not isinstance(parsed, dict):
            return GraderResult(
                self.name, self.type, 0.0, False, "answer is not a JSON object",
                {"parsed": True},
            )
        missing = [k for k in self.required_keys if k not in parsed]
        return GraderResult(
            self.name, self.type, 0.0 if missing else 1.0, not missing,
            f"missing keys: {missing}" if missing else "all required keys present",
            {"missing": missing},
        )


class ToolCallGrader:
    """Required tools were called and forbidden ones were not.

    Reads the trajectory, so it is ungradable without a trace.
    """

    type = "tool_call"
    needs_trace = True

    def __init__(self, name: str = "tool_call"):
        self.name = name

    def grade(self, task: Task, trial: Trial, trace: Trace | None) -> GraderResult:
        if trace is None:
            return _ungradable(
                self.name, self.type,
                "no trace: tool use cannot be judged from the answer alone",
            )
        called = _tool_names(trace)
        missing = [t for t in task.required_tools if t not in called]
        forbidden = [t for t in task.forbidden_tools if t in called]
        passed = not missing and not forbidden
        reasons = []
        if missing:
            reasons.append(f"never called {missing}")
        if forbidden:
            reasons.append(f"called forbidden {forbidden}")
        return GraderResult(
            self.name, self.type, 1.0 if passed else 0.0, passed,
            "; ".join(reasons) or "tool use as expected",
            {
                "called": called,
                "missing_required": missing,
                "called_forbidden": forbidden,
            },
        )


class ToolOrderGrader:
    """Required tools were called in the order the task lists them.

    Order matters for agents whose steps depend on each other — looking up a
    customer before issuing their refund. Subsequence, not equality: extra
    calls in between are allowed, since an agent that also checked something
    else has not done the wrong thing.
    """

    type = "tool_order"
    needs_trace = True

    def __init__(self, name: str = "tool_order"):
        self.name = name

    def grade(self, task: Task, trial: Trial, trace: Trace | None) -> GraderResult:
        if trace is None:
            return _ungradable(self.name, self.type, "no trace: order cannot be judged")
        called = _tool_names(trace)
        remaining = list(task.required_tools)
        for name in called:
            if remaining and name == remaining[0]:
                remaining.pop(0)
        passed = not remaining
        return GraderResult(
            self.name, self.type, 1.0 if passed else 0.0, passed,
            "order as expected" if passed else f"never reached {remaining}",
            {"called": called, "unmatched": remaining},
        )


class NoToolErrorGrader:
    type = "no_tool_error"
    needs_trace = True

    def __init__(self, name: str = "no_tool_error"):
        self.name = name

    def grade(self, task: Task, trial: Trial, trace: Trace | None) -> GraderResult:
        if trace is None:
            return _ungradable(self.name, self.type, "no trace: tool errors unknown")
        errors = int(trace.get("tool_error_count") or 0)
        return GraderResult(
            self.name, self.type, 1.0 if errors == 0 else 0.0, errors == 0,
            "no tool errors" if errors == 0 else f"{errors} tool call(s) failed",
            {"tool_error_count": errors},
        )


class FunctionGrader:
    """Wraps a plain function, the interface the design documents::

        def grade(task, trial, trace) -> dict
    """

    type = "function"
    needs_trace = True

    def __init__(self, fn: Callable[..., dict], name: str | None = None,
                 needs_trace: bool = True):
        self.fn = fn
        self.name = name or getattr(fn, "__name__", "function")
        self.needs_trace = needs_trace

    def grade(self, task: Task, trial: Trial, trace: Trace | None) -> GraderResult:
        raw = self.fn(task, trial, trace)
        return GraderResult(
            self.name, self.type,
            float(raw.get("score", 0.0)), bool(raw.get("passed", False)),
            str(raw.get("reason", "")), dict(raw.get("assertions", {})),
        )


def _tool_names(trace: Trace) -> list[str]:
    raw = trace.get("tool_names")
    if isinstance(raw, list):
        return [str(t) for t in raw]
    try:
        parsed = json.loads(raw) if raw else []
    except (ValueError, TypeError):
        return []
    return [str(t) for t in parsed] if isinstance(parsed, list) else []


def run_graders(
    graders: Sequence[Grader], task: Task, trial: Trial, trace: Trace | None
) -> list[GraderResult]:
    """Every grader, in order, none of which may break the run."""
    results: list[GraderResult] = []
    for grader in graders:
        try:
            results.append(grader.grade(task, trial, trace))
        except Exception as err:  # noqa: BLE001 — a bad grader is not a bad agent
            results.append(
                GraderResult(
                    getattr(grader, "name", "unknown"),
                    getattr(grader, "type", "unknown"),
                    0.0, False, f"grader raised: {err}", ungradable=True,
                )
            )
    return results


def verdict(
    results: Sequence[GraderResult],
    policy: "PassPolicy | str" = "all",
    threshold: float = 0.7,
) -> bool | None:
    """Whether the trial passed, under the suite's policy.

    Returns ``None`` when nothing was gradable, which is not the same as
    failing — it means the trial says nothing about the agent either way.

    ``all`` is the default and stays the default. The other two can turn a
    failing trial into a passing one, so a suite has to ask for them:

    - ``any`` — one grader passing is enough. For a task with several acceptable
      answers expressed as separate checks.
    - ``threshold`` — the mean score clears a bar. For rubric-led suites where
      partial credit is the measure and a hard pass/fail per grader is not.
    """
    gradable = [r for r in results if not r.ungradable]
    if not gradable:
        return None

    name = getattr(policy, "value", policy)
    if name == "any":
        return any(r.passed for r in gradable)
    if name == "threshold":
        return (sum(r.score for r in gradable) / len(gradable)) >= threshold
    return all(r.passed for r in gradable)


class SqlStateGrader:
    """Assert the world changed, not just that the agent said it did.

    An agent that answers "I've cancelled order 4471" convincingly and calls
    nothing is indistinguishable, on the text alone, from one that did the work.
    This runs a read query and compares one value against what the task expects.

    ``query`` is injected rather than opened here: the grader has no business
    holding credentials, and the job that runs it already has a feature store
    session. Without one the grader is ungradable — never passing, since "I
    could not check" must not read as "the state was right".
    """

    type = "sql_state"
    needs_trace = False

    def __init__(
        self,
        sql: str,
        expect: Any = None,
        *,
        query: Callable[[str], Any] | None = None,
        name: str = "sql_state",
    ):
        self.sql = sql
        self.expect = expect
        self.query = query
        self.name = name

    def grade(self, task: Task, trial: Trial, trace: Trace | None) -> GraderResult:
        if self.query is None:
            return _ungradable(
                self.name, self.type,
                "no query function configured: state could not be checked",
            )
        try:
            actual = self.query(self.sql)
        except Exception as err:  # noqa: BLE001 — a broken query is not a broken agent
            return _ungradable(self.name, self.type, f"query failed: {err}")

        got = _scalar(actual)
        passed = str(got).strip() == str(self.expect).strip()
        return GraderResult(
            grader_name=self.name,
            grader_type=self.type,
            score=1.0 if passed else 0.0,
            passed=passed,
            reason="" if passed else f"expected {self.expect!r}, found {got!r}",
            assertions={"sql": self.sql, "expected": self.expect, "actual": got},
        )


def _scalar(result: Any) -> Any:
    """First cell of whatever the query returned.

    Accepts a DataFrame, a list of rows, or a bare value, because the caller's
    session decides the shape and the grader asserts one value either way.
    """
    if result is None:
        return None
    values = getattr(result, "values", None)
    if values is not None and hasattr(result, "columns"):  # DataFrame
        return values[0][0] if len(values) and len(values[0]) else None
    if isinstance(result, (list, tuple)):
        if not result:
            return None
        first = result[0]
        if isinstance(first, (list, tuple)):
            return first[0] if first else None
        return first
    return result


class HumanReviewGrader:
    """A verdict this run cannot produce, held open until a person gives one.

    Returns ungradable with ``awaiting_review``, which the runner reads to mark
    the trial ``AWAITING_REVIEW`` rather than letting the other graders decide
    it. That distinction is the point: a task that asked for human judgement and
    silently passed on a substring match has not been judged.

    The verdict arrives later through the review endpoint, which writes a real
    grader result alongside this one.
    """

    type = "human_review"
    needs_trace = False

    def __init__(self, prompt: str = "", name: str = "human_review"):
        self.prompt = prompt
        self.name = name

    def grade(self, task: Task, trial: Trial, trace: Trace | None) -> GraderResult:
        result = _ungradable(
            self.name, self.type,
            self.prompt or "waiting for a reviewer to judge this answer",
        )
        result.assertions = {"awaiting_review": True, "prompt": self.prompt}
        return result


AWAITING_REVIEW = "awaiting_review"


def awaits_review(results: Sequence[GraderResult]) -> bool:
    """Whether any grader deferred to a person."""
    return any(r.assertions.get(AWAITING_REVIEW) for r in results)

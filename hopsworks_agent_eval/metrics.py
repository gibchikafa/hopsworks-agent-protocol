"""Aggregate trials into ``agent_eval_run_metrics`` rows.

Computed once, here, and read by everything else. A second implementation in
the UI would eventually disagree with the one the release gate uses, and the
disagreement would surface as "the dashboard says we passed but promotion is
blocked".

``pass^k`` is the metric most incumbents do not surface and the one that
matters for agents: an agent that succeeds four times in five is not 80% good
at a task, it is unreliable at it, and a user hits the fifth case.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Sequence

from .models import TraceStatus, Trial, TrialStatus, gradable_trials


def _passed(trial: Trial) -> bool:
    return trial.status is TrialStatus.PASSED


def percentile(values: Sequence[float], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    k = (len(ordered) - 1) * pct / 100.0
    lo = int(k)
    hi = min(lo + 1, len(ordered) - 1)
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (k - lo)


def pass_at_k(trials: Sequence[Trial]) -> float:
    """Fraction of tasks that succeeded at least once across their trials."""
    by_task = _by_task(trials)
    if not by_task:
        return 0.0
    return sum(
        1 for task_trials in by_task.values() if any(_passed(t) for t in task_trials)
    ) / len(by_task)


def pass_all_k(trials: Sequence[Trial]) -> float:
    """Fraction of tasks that succeeded on *every* trial — ``pass^k``.

    The reliability number. A task that passes 4 of 5 counts here as a failure,
    which is the honest reading: the agent is non-deterministic and a user will
    meet the fifth case.
    """
    by_task = _by_task(trials)
    if not by_task:
        return 0.0
    return sum(
        1 for task_trials in by_task.values() if all(_passed(t) for t in task_trials)
    ) / len(by_task)


def flaky_tasks(trials: Sequence[Trial]) -> list[str]:
    """Tasks that both passed and failed across their trials.

    Worth naming separately: a flaky task is neither a regression to fix nor a
    capability that works, and averaging it into a pass rate hides it.
    """
    flaky = []
    for task_id, task_trials in _by_task(trials).items():
        outcomes = {_passed(t) for t in task_trials}
        if len(outcomes) > 1:
            flaky.append(task_id)
    return sorted(flaky)


def _by_task(trials: Sequence[Trial]) -> dict[str, list[Trial]]:
    grouped: dict[str, list[Trial]] = defaultdict(list)
    for trial in gradable_trials(trials):
        grouped[trial.task_id].append(trial)
    return dict(grouped)


def _tool_error_rate(trials: Sequence[Trial]) -> float:
    """Share of trials with a visible trajectory in which a tool call failed."""
    seen = [
        t for t in trials
        if t.trace_status is not TraceStatus.MISSING and t.tool_error_count is not None
    ]
    if not seen:
        return 0.0
    return sum(1 for t in seen if t.tool_error_count) / len(seen)


def run_metrics(run_id: str, suite_id: str, deployment_id: int,
                trials: Sequence[Trial]) -> list[dict[str, Any]]:
    """One row per metric, scoped to the run."""
    gradable = gradable_trials(trials)
    latencies = [t.latency_ms for t in gradable if t.latency_ms]
    passed = sum(1 for t in gradable if _passed(t))

    values: dict[str, float] = {
        "pass_rate": passed / len(gradable) if gradable else 0.0,
        "pass_at_k": pass_at_k(trials),
        "pass_all_k": pass_all_k(trials),
        "p50_latency_ms": percentile(latencies, 50),
        "p95_latency_ms": percentile(latencies, 95),
        "p99_latency_ms": percentile(latencies, 99),
        # Not a quality metric: a high rate means the observability pipeline is
        # broken, and trajectory numbers from this run cannot be trusted.
        "trace_missing_rate": (
            sum(1 for t in trials if t.trace_status is TraceStatus.MISSING) / len(trials)
            if trials else 0.0
        ),
        "guardrail_block_rate": (
            sum(1 for t in trials if t.status is TrialStatus.BLOCKED_BY_GUARDRAIL)
            / len(trials) if trials else 0.0
        ),
        "flaky_task_count": float(len(flaky_tasks(trials))),
        # Trials in which at least one tool call failed, over trials whose
        # trajectory was actually visible. Scoped that way on purpose: dividing
        # by every trial would report an observability gap as a healthy tool
        # layer, since a trial with no trace contributes no errors.
        "tool_error_rate": _tool_error_rate(trials),
    }

    task_count = len({t.task_id for t in trials})
    return [
        {
            "run_id": run_id,
            "suite_id": suite_id,
            "deployment_id": deployment_id,
            "metric_scope": "run",
            # empty for run scope; without the value in the key two categories
            # reporting the same metric would collide
            "metric_scope_value": "",
            "metric_name": name,
            "metric_value": value,
            "task_count": task_count,
            "trial_count": len(trials),
        }
        for name, value in values.items()
    ]

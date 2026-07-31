"""Execute a suite against a deployed agent.

The runner is a *client* of the agent, which is what makes this design differ
from the SaaS eval tools: they run the task loop in the user's own process, so
they get output capture and trace correlation for free by being the caller.
Here the unit of evaluation is a deployed serving endpoint, so correlation,
trace readiness, and execution safety all have to be built rather than assumed.

Everything that touches the network is injected (:class:`AgentClient`), because
the parts worth testing are the refusals and the failure classification, not
the HTTP.
"""

from __future__ import annotations

import logging
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Protocol, Sequence

from .evaluators import Evaluator, Trace, awaits_review, run_evaluators, verdict
from .models import (
    ExecutionMode,
    Suite,
    SuiteType,
    Task,
    TraceStatus,
    Trial,
    TrialStatus,
    derive_trial_id,
)

log = logging.getLogger(__name__)


class SuiteRefused(Exception):
    """The run was refused before any request was sent.

    Deliberately raised rather than returned: every one of these means the run
    would have produced results that look valid and are not, and a caller that
    ignores a return value would then act on them.
    """


@dataclass
class AgentResponse:
    text: str
    trace_id: str = ""
    blocked_by_guardrail: bool = False
    latency_ms: float = 0.0
    error: str = ""


class AgentClient(Protocol):
    """What the runner needs from the outside world."""

    def manifest(self) -> dict[str, Any]: ...

    def call(self, prompt: str, *, traceparent: str, baggage: str,
             timeout_s: float) -> AgentResponse: ...

    def fetch_trace(self, trace_id: str) -> Trace | None: ...


@dataclass
class RunnerConfig:
    n_trials: int = 1
    timeout_s: float = 120.0
    max_concurrency: int = 4
    # Derived from the Stage 1 probe's trajectory-stable p95, not guessed. A
    # full suite is n_tasks x n_trials requests at a shared endpoint, so this
    # and max_concurrency are the difference between an eval run and a
    # self-inflicted load test on production capacity.
    readiness_timeout_s: float = 120.0
    readiness_poll_s: float = 2.0
    # Taken from the deployment's tracing config so an eval run is costed the
    # same way production traffic is. Absent means unpriced, which is reported
    # as such rather than as free.
    input_token_price_per_million: float | None = None
    output_token_price_per_million: float | None = None


@dataclass
class RunResult:
    run_id: str
    suite_id: str
    deployment_id: int
    trials: list[Trial] = field(default_factory=list)
    status: str = "SUCCEEDED"
    started_at: datetime = field(default_factory=lambda: datetime.now(tz=timezone.utc))
    completed_at: datetime | None = None


def check_deployment_supports(suite: Suite, manifest: dict[str, Any]) -> None:
    """Refuse a run that cannot produce trustworthy results.

    Both checks exist because the alternative is worse than an error: a run
    against an agent that ignores the traceparent writes trial rows pointing at
    traces that were never created, and a sandboxed suite against a live
    deployment executes injection attempts against real tools. Neither announces
    itself in the results.
    """
    capabilities = manifest.get("capabilities", {})

    if suite.execution_mode is ExecutionMode.LIVE:
        raise SuiteRefused(
            "execution_mode 'live' is not supported: it needs a per-suite "
            "allowlist of tools the run may trigger"
        )

    if "trace_correlation" not in capabilities:
        raise SuiteRefused(
            "the deployment's manifest does not report trace_correlation, so "
            "it is running an SDK too old to continue the runner's trace "
            "context; every trial would point at a trace that does not exist"
        )
    if not capabilities["trace_correlation"]:
        raise SuiteRefused(
            "tracing is disabled on this deployment, so no trial can be "
            "correlated to a trace"
        )

    if suite.execution_mode is ExecutionMode.SANDBOXED and not capabilities.get("eval_mode"):
        raise SuiteRefused(
            f"suite {suite.suite_id} is sandboxed but the deployment does not "
            "report eval_mode: its tools may still reach production systems"
        )
    if suite.type is SuiteType.SAFETY and suite.execution_mode is not ExecutionMode.SANDBOXED:
        raise SuiteRefused(
            "safety suites must be sandboxed: they contain injection and "
            "data-exfiltration attempts by construction"
        )


def _traceparent(trace_id: str, span_id: str) -> str:
    return f"00-{trace_id}-{span_id}-01"


def _baggage(run_id: str, suite: Suite, task: Task, trial_index: int, trial_id: str) -> str:
    from hopsworks_agent_protocol import conventions

    return ",".join(
        [
            f"{conventions.EVAL_RUN_ID}={run_id}",
            f"{conventions.EVAL_SUITE_ID}={suite.suite_id}",
            f"{conventions.EVAL_SUITE_VERSION}={suite.suite_version}",
            f"{conventions.EVAL_TASK_ID}={task.task_id}",
            f"{conventions.EVAL_TASK_VERSION}={task.task_version}",
            f"{conventions.EVAL_TRIAL_ID}={trial_id}",
            f"{conventions.EVAL_TRIAL_INDEX}={trial_index}",
        ]
    )


def _new_ids() -> tuple[str, str]:
    import secrets

    return secrets.token_hex(16), secrets.token_hex(8)


def _classify(response: AgentResponse, suite: Suite) -> TrialStatus | None:
    """Turn a response into a failure class, or None when it is gradable."""
    if response.error:
        return TrialStatus.INFRA_ERROR if "timeout" not in response.error.lower() \
            else TrialStatus.TIMEOUT
    if response.blocked_by_guardrail:
        return TrialStatus.BLOCKED_BY_GUARDRAIL
    return None


def _guardrail_outcome(suite: Suite) -> TrialStatus:
    """A guardrail block is an outcome, not inherently a failure.

    In a safety suite the block is the desired result and the trial passes; in
    a capability or regression suite it is an over-refusal and the trial fails.
    Measuring only the first is the classic mistake — guardrails look effective
    while quietly degrading the product.
    """
    if suite.type is SuiteType.SAFETY:
        return TrialStatus.PASSED
    return TrialStatus.BLOCKED_BY_GUARDRAIL


def _await_trace(
    client: AgentClient, trace_id: str, config: RunnerConfig, sleep=time.sleep
) -> tuple[Trace | None, TraceStatus]:
    """Poll until the trace is readable, or give up.

    Sidecar insert failures are logged and dropped, so a missing trace is a
    permanent condition to tolerate, not only a timing one.
    """
    if not trace_id:
        return None, TraceStatus.MISSING
    deadline = time.monotonic() + config.readiness_timeout_s
    while time.monotonic() < deadline:
        try:
            trace = client.fetch_trace(trace_id)
        except Exception:  # noqa: BLE001 — a transient read must not end the poll
            trace = None
        if trace:
            status = (
                TraceStatus.RECEIVED
                if trace.get("root_span_id")
                else TraceStatus.PARTIAL
            )
            return trace, status
        sleep(config.readiness_poll_s)
    return None, TraceStatus.MISSING


def _run_trial(
    client: AgentClient,
    suite: Suite,
    task: Task,
    trial_index: int,
    run_id: str,
    deployment_id: int,
    evaluators: Sequence[Evaluator],
    config: RunnerConfig,
    sleep=time.sleep,
) -> Trial:
    trial_id = derive_trial_id(run_id, task.task_id, task.task_version, trial_index)
    trace_id, span_id = _new_ids()
    trial = Trial(
        trial_id=trial_id,
        run_id=run_id,
        task_id=task.task_id,
        task_version=task.task_version,
        trial_index=trial_index,
        deployment_id=deployment_id,
        # stamped up front: a trial whose request fails is still correlated,
        # which is the reason the runner generates the id rather than reading
        # it back from the response
        trace_id=trace_id,
    )

    try:
        response = client.call(
            task.prompt,
            traceparent=_traceparent(trace_id, span_id),
            baggage=_baggage(run_id, suite, task, trial_index, trial_id),
            timeout_s=config.timeout_s,
        )
    except Exception as err:  # noqa: BLE001
        trial.status = TrialStatus.INFRA_ERROR
        trial.error_type = type(err).__name__
        trial.error_message = str(err)
        trial.completed_at = datetime.now(tz=timezone.utc)
        return trial

    trial.final_output = response.text
    trial.latency_ms = response.latency_ms

    failure = _classify(response, suite)
    if failure is TrialStatus.BLOCKED_BY_GUARDRAIL:
        trial.status = _guardrail_outcome(suite)
        trial.completed_at = datetime.now(tz=timezone.utc)
        return trial
    if failure is not None:
        trial.status = failure
        trial.error_message = response.error
        trial.completed_at = datetime.now(tz=timezone.utc)
        return trial

    trace, trace_status = _await_trace(client, response.trace_id or trace_id, config, sleep)
    trial.trace_status = trace_status

    if trace is not None:
        trial.tool_error_count = int(trace.get("tool_error_count") or 0)
        trial.tool_call_count = len(trace.get("tool_calls") or trace.get("tool_names") or [])
        trial.input_tokens = trace.get("input_tokens")
        trial.output_tokens = trace.get("output_tokens")
        trial.estimated_cost = _cost(trial, config)

    trial.evaluator_results = run_evaluators(evaluators, task, trial, trace)
    outcome = verdict(trial.evaluator_results, suite.pass_policy, suite.pass_threshold)

    if awaits_review(trial.evaluator_results):
        # A task that asked for human judgement has not been judged by the other
        # evaluators agreeing with each other. Held open rather than resolved, so a
        # reviewer's verdict is the thing that settles it.
        trial.status = TrialStatus.AWAITING_REVIEW
        trial.completed_at = datetime.now(tz=timezone.utc)
        return trial

    if trace_status is TraceStatus.MISSING:
        # Final-answer evaluators still ran on the captured response; only the
        # trajectory ones went ungradable. The trial keeps its answer verdict
        # but is marked so trajectory metrics can exclude it, and so a high
        # rate of these fails the run rather than the agent.
        trial.status = TrialStatus.TRACE_MISSING if outcome is None else (
            TrialStatus.PASSED if outcome else TrialStatus.FAILED
        )
        if outcome is None:
            trial.error_type = "TRACE_MISSING"
    else:
        trial.status = (
            TrialStatus.PASSED if outcome else
            TrialStatus.FAILED if outcome is False else TrialStatus.TRACE_MISSING
        )

    trial.completed_at = datetime.now(tz=timezone.utc)
    return trial


def _cost(trial: Trial, config: RunnerConfig) -> float | None:
    """What this trial cost, or None when nobody configured a price.

    None rather than 0.0 on purpose: a run with no pricing set up and a run that
    genuinely cost nothing are different facts, and a dashboard summing zeros
    reports the second when it means the first.
    """
    if config.input_token_price_per_million is None \
            and config.output_token_price_per_million is None:
        return None
    return (
        (trial.input_tokens or 0) * (config.input_token_price_per_million or 0.0)
        + (trial.output_tokens or 0) * (config.output_token_price_per_million or 0.0)
    ) / 1_000_000


def run_suite(
    client: AgentClient,
    suite: Suite,
    *,
    run_id: str,
    deployment_id: int,
    evaluators: Sequence[Evaluator] | Callable[[Task], Sequence[Evaluator]],
    config: RunnerConfig | None = None,
    sleep=time.sleep,
) -> RunResult:
    """Execute every task, ``n_trials`` times each, and grade the results.

    ``evaluators`` may be one list applied to every task, or a function of the
    task. Per-task matters more than it looks: which evaluators apply depends on
    what a task declares, so one fixed list either judges a task on tools it
    never claimed to use, or judges nothing at all and reports a run that says
    nothing while looking complete.
    """
    config = config or RunnerConfig()
    check_deployment_supports(suite, client.manifest())

    unsupported = [t for t in suite.tasks if t.task_type != "single_turn"]
    if unsupported:
        # v1 rejects what it cannot run rather than silently running the first
        # turn of a multi-turn task and reporting a pass rate for it
        raise SuiteRefused(
            f"unsupported task types: {sorted({t.task_type for t in unsupported})}"
        )

    result = RunResult(run_id=run_id, suite_id=suite.suite_id, deployment_id=deployment_id)
    work = [
        (task, index)
        for task in suite.tasks
        for index in range(max(1, config.n_trials))
    ]

    # Bounded: a full suite against a shared endpoint is otherwise a
    # self-inflicted load test on production capacity.
    resolve = evaluators if callable(evaluators) else (lambda _task: evaluators)

    with ThreadPoolExecutor(max_workers=max(1, config.max_concurrency)) as pool:
        futures = [
            pool.submit(
                _run_trial, client, suite, task, index, run_id, deployment_id,
                resolve(task), config, sleep,
            )
            for task, index in work
        ]
        for future in futures:
            result.trials.append(future.result())

    result.trials.sort(key=lambda t: (t.task_id, t.trial_index))
    missing = sum(1 for t in result.trials if t.trace_status is TraceStatus.MISSING)
    if result.trials and missing / len(result.trials) > 0.5:
        # A high rate means the observability pipeline is broken, not the
        # agent. Passing the run would publish a pass rate computed mostly
        # from final answers while claiming to have judged trajectories.
        result.status = "FAILED"
        log.error("%d/%d trials never got a trace", missing, len(result.trials))
    result.completed_at = datetime.now(tz=timezone.utc)
    return result

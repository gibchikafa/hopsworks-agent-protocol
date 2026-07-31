"""The job the backend starts when a run is created.

    python -m hopsworks_agent_eval.run_job --run-id <runId>

The run id is the only argument, deliberately: which suite version, which
deployment and how many trials all live on the row it names, so the job and the
record cannot disagree about what was executed.

What it does, in order: read the run, load the suite's tasks, execute them
against the deployment, write trials, evaluator results and metrics to the feature
store, and report the outcome back. Reporting back matters as much as the work
— a run stuck in RUNNING because the job died is indistinguishable from one
still going, and the UI has no way to tell you which.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from datetime import datetime, timezone
from typing import Any

from .evaluator_spec import SpecError, evaluators_for_suite
from .evaluators import (
    ContainsEvaluator,
    ExactMatchEvaluator,
    Evaluator,
    NoToolErrorEvaluator,
    ToolCallEvaluator,
    ToolOrderEvaluator,
)
from .judges import DEFAULT_MODEL, LlmJudgeEvaluator, anthropic_completer
from .metrics import run_metrics
from .models import ExecutionMode, PassPolicy, Suite, Task
from .runner import RunnerConfig, SuiteRefused, run_suite

log = logging.getLogger(__name__)

TRIALS_FG = "agent_eval_trials"
EVALUATOR_RESULTS_FG = "agent_eval_evaluator_results"
RUN_METRICS_FG = "agent_eval_run_metrics"


def _api(host: str, project_id: int) -> str:
    return f"{host.rstrip('/')}/hopsworks-api/api/project/{project_id}/agent-evals"


def _query_for(project: Any) -> Any:
    """A read-only query function for state evaluators, or None.

    Bound to the feature store the job already authenticated to, so a state
    assertion sees exactly what the project can see and nothing wider. Returned
    lazily: a suite with no sql_state evaluator should not pay for a session.
    """
    def query(sql: str) -> Any:
        return project.get_feature_store().sql(sql)

    return query


def _judge_for(project: Any) -> LlmJudgeEvaluator | None:
    """An LLM judge, if the project has configured one.

    The key comes from a project secret named ``EVAL_JUDGE_API_KEY``: the
    provider is the project's choice, and a run should never carry credentials
    of its own. Absent, tasks with rubrics simply go ungradable — better than a
    run that quietly grades nothing while looking complete.
    """
    try:
        secret = project.get_secrets_api().get_secret("EVAL_JUDGE_API_KEY")
        api_key = secret.value
    except Exception:  # noqa: BLE001 — no judge configured is a normal state
        log.info("no EVAL_JUDGE_API_KEY secret; rubric tasks will be ungradable")
        return None

    model = os.environ.get("EVAL_JUDGE_MODEL", DEFAULT_MODEL)
    log.info("LLM judge enabled (model=%s)", model)
    return LlmJudgeEvaluator(anthropic_completer(api_key, model), model=model)


def _to_suite(run: dict[str, Any], tasks: list[dict[str, Any]]) -> Suite:
    def parse_tools(raw: Any) -> list[str]:
        if not raw:
            return []
        try:
            return json.loads(raw) if isinstance(raw, str) else list(raw)
        except (ValueError, TypeError):
            return []

    return Suite(
        suite_id=run["suiteId"],
        suite_version=run.get("suiteVersion", 1),
        tags=run.get("tags") or [],
        blocks_are_success=bool(run.get("blocksAreSuccess")),
        execution_mode=ExecutionMode(run.get("executionMode", "read_only")),
        evaluators=run.get("evaluators") or "",
        pass_policy=PassPolicy(run.get("passPolicy") or "all"),
        pass_threshold=float(run.get("passThreshold") or 0.7),
        tasks=[
            Task(
                task_id=t["taskId"],
                task_version=t.get("version", 1),
                input_messages=t.get("inputMessages") or "[]",
                task_type=t.get("taskType", "single_turn"),
                expected_output=t.get("expectedOutput") or "",
                required_tools=parse_tools(t.get("requiredTools")),
                forbidden_tools=parse_tools(t.get("forbiddenTools")),
                rubric=t.get("rubric") or "",
                category=t.get("category") or "",
            )
            for t in tasks
        ],
    )


def _write_results(feature_store: Any, result: Any, run: dict[str, Any],
                   tasks: list[dict[str, Any]] | None = None) -> None:
    import pandas as pd

    now = datetime.now(tz=timezone.utc)
    trials = [
        {
            "run_id": t.run_id,
            "trial_id": t.trial_id,
            "task_id": t.task_id,
            "task_version": t.task_version,
            "trial_index": t.trial_index,
            "deployment_id": t.deployment_id,
            "trace_id": t.trace_id,
            "trace_status": t.trace_status.value,
            "session_id": "",
            "status": t.status.value,
            "started_at": t.started_at,
            "completed_at": t.completed_at or now,
            "latency_ms": t.latency_ms,
            "input_tokens": t.input_tokens or 0,
            "output_tokens": t.output_tokens or 0,
            "estimated_cost": t.estimated_cost,
            "final_output": t.final_output,
            "error_type": t.error_type,
            "error_message": t.error_message,
            "created_at": now,
        }
        for t in result.trials
    ]
    evaluator_rows = [
        {
            "run_id": t.run_id,
            "result_id": f"{t.trial_id}/{g.evaluator_name}",
            "trial_id": t.trial_id,
            "task_id": t.task_id,
            "evaluator_name": g.evaluator_name,
            "evaluator_type": g.evaluator_type,
            "score": g.score,
            "passed": g.passed,
            "ungradable": g.ungradable,
            "reason": g.reason,
            "assertions_json": json.dumps(g.assertions),
            "judge_model": str(g.assertions.get("judge_model", "")),
            "evaluator_version": "1",
            "created_at": now,
        }
        for t in result.trials
        for g in t.evaluator_results
    ]
    metric_rows = [
        {**m, "suite_version": run.get("suiteVersion", 1), "created_at": now}
        for m in run_metrics(
            result.run_id,
            run["suiteId"],
            run["deploymentId"],
            result.trials,
            blocks_are_success=bool(run.get("blocksAreSuccess")),
            # so score-by-category has categories to group on; the tasks are
            # already in hand from building the suite
            categories={
                t["taskId"]: t.get("category") or "" for t in (tasks or [])
            },
        )
    ]

    for name, rows in (
        (TRIALS_FG, trials),
        (EVALUATOR_RESULTS_FG, evaluator_rows),
        (RUN_METRICS_FG, metric_rows),
    ):
        if not rows:
            continue
        feature_store.get_feature_group(name, 1).insert(
            pd.DataFrame(rows), write_options={"mode": "append"}
        )
        log.info("wrote %d rows to %s", len(rows), name)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--readiness-timeout-s", type=float, default=120.0,
                        help="derive from the Stage 1 probe's trajectory-stable "
                             "p95 rather than accepting this default")
    parser.add_argument("--max-concurrency", type=int, default=4)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)

    import hopsworks
    import requests

    project = hopsworks.login()
    host = os.environ.get("HOPSWORKS_HOST") or os.environ["REST_ENDPOINT"]
    api_key = os.environ["HOPSWORKS_API_KEY"]
    session = requests.Session()
    session.headers["Authorization"] = f"ApiKey {api_key}"
    base = _api(host, project.id)

    def report(status: str, error: str | None = None) -> None:
        # Reported even on the failure paths: a run left RUNNING because the job
        # died looks exactly like one still going, and nothing can tell you which.
        try:
            session.put(f"{base}/runs/{args.run_id}/status",
                        params={"status": status, **({"errorMessage": error} if error else {})},
                        timeout=30)
        except Exception:  # noqa: BLE001 — never mask the real failure
            log.exception("could not report status %s", status)

    try:
        run = session.get(f"{base}/runs/{args.run_id}", timeout=60).json()
        tasks = session.get(
            f"{base}/suites/{run['suiteId']}/tasks",
            params={"version": run.get("suiteVersion")},
            timeout=60,
        ).json()
        suite = _to_suite(run, tasks)

        from .client import HopsworksAgentClient

        judge = _judge_for(project)
        # The judge is a evaluator; the spec needs the bare completer behind it, so
        # a task can ask for a rubric judge and a pairwise judge independently.
        completer = judge._complete if judge is not None else None  # noqa: SLF001
        query = _query_for(project)

        def secret_reader(name: str) -> str | None:
            """A project secret, for a judge that names its own key.

            Looked up per name rather than passed in, so a suite can mix a
            cheap judge for canaries with an expensive one for release gating
            without either key leaving the project's secret store.
            """
            try:
                return project.get_secrets_api().get_secret(name).value
            except Exception:  # noqa: BLE001 — a missing secret is a normal state
                log.info("no secret %r; judges naming it will be skipped", name)
                return None
        client = HopsworksAgentClient(
            session=session,
            api_base=host,
            project_id=project.id,
            project_name=project.name,
            deployment_id=run["deploymentId"],
        )

        result = run_suite(
            client,
            suite,
            run_id=args.run_id,
            deployment_id=run["deploymentId"],
            # One list for the whole suite: every task is measured the same
            # way, which is what makes the run's pass rate comparable to the
            # next one's.
            evaluators=evaluators_for_suite(
                suite, judge_completer=completer, query=query,
                secret_reader=secret_reader,
            ),
            config=RunnerConfig(
                n_trials=run.get("nTrials", 1),
                readiness_timeout_s=args.readiness_timeout_s,
                max_concurrency=args.max_concurrency,
                input_token_price_per_million=run.get("inputTokenPricePerMillion"),
                output_token_price_per_million=run.get("outputTokenPricePerMillion"),
            ),
        )
        _write_results(project.get_feature_store(), result, run, tasks)
        report(result.status)
        log.info("run %s finished: %s", args.run_id, result.status)
    except SpecError as err:
        # Authoring validates the spec, so reaching here means a task was written
        # before that check existed or around it. Named as a run failure rather
        # than an agent one.
        log.exception("a task has an unusable evaluator spec")
        report("FAILED", f"evaluator spec: {err}")
    except SuiteRefused as err:
        # A refusal is a result, not a crash: the run would have produced
        # numbers that looked valid, and saying so is the point.
        log.error("run refused: %s", err)
        report("FAILED", str(err))
        raise SystemExit(1)
    except Exception as err:  # noqa: BLE001
        log.exception("run %s failed", args.run_id)
        report("FAILED", str(err))
        raise SystemExit(1)


if __name__ == "__main__":
    main()

"""The job the backend starts when a run is created.

    python -m hopsworks_agent_eval.run_job --run-id <runId>

The run id is the only argument, deliberately: which suite version, which
deployment and how many trials all live on the row it names, so the job and the
record cannot disagree about what was executed.

What it does, in order: read the run, load the suite's tasks, execute them
against the deployment, write trials, grader results and metrics to the feature
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

from .graders import (
    ContainsGrader,
    ExactMatchGrader,
    Grader,
    NoToolErrorGrader,
    ToolCallGrader,
    ToolOrderGrader,
)
from .metrics import run_metrics
from .models import ExecutionMode, Suite, SuiteType, Task
from .runner import RunnerConfig, SuiteRefused, run_suite

log = logging.getLogger(__name__)

TRIALS_FG = "agent_eval_trials"
GRADER_RESULTS_FG = "agent_eval_grader_results"
RUN_METRICS_FG = "agent_eval_run_metrics"


def _api(host: str, project_id: int) -> str:
    return f"{host.rstrip('/')}/hopsworks-api/api/project/{project_id}/agent-evals"


def _graders_for(task: Task) -> list[Grader]:
    """Which graders a task gets, inferred from what it declares.

    A task that names no expectation and no tools would otherwise be graded by
    nothing and pass silently, so it gets nothing and its trials come back
    ungradable — which is the honest answer rather than a free pass.
    """
    graders: list[Grader] = []
    if task.expected_output:
        # `contains` rather than exact match: for free text an exact match
        # asserts the model's phrasing rather than its correctness
        graders.append(ContainsGrader())
    if task.required_tools or task.forbidden_tools:
        graders.append(ToolCallGrader())
    if len(task.required_tools) > 1:
        graders.append(ToolOrderGrader())
    if task.required_tools:
        graders.append(NoToolErrorGrader())
    return graders


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
        type=SuiteType(run.get("suiteType", "regression")),
        execution_mode=ExecutionMode(run.get("executionMode", "read_only")),
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


def _write_results(feature_store: Any, result: Any, run: dict[str, Any]) -> None:
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
            "input_tokens": 0,
            "output_tokens": 0,
            "estimated_cost": None,
            "final_output": t.final_output,
            "error_type": t.error_type,
            "error_message": t.error_message,
            "created_at": now,
        }
        for t in result.trials
    ]
    grader_rows = [
        {
            "run_id": t.run_id,
            "result_id": f"{t.trial_id}/{g.grader_name}",
            "trial_id": t.trial_id,
            "task_id": t.task_id,
            "grader_name": g.grader_name,
            "grader_type": g.grader_type,
            "score": g.score,
            "passed": g.passed,
            "ungradable": g.ungradable,
            "reason": g.reason,
            "assertions_json": json.dumps(g.assertions),
            "judge_model": "",
            "grader_version": "1",
            "created_at": now,
        }
        for t in result.trials
        for g in t.grader_results
    ]
    metric_rows = [
        {**m, "suite_version": run.get("suiteVersion", 1), "created_at": now}
        for m in run_metrics(
            result.run_id, run["suiteId"], run["deploymentId"], result.trials
        )
    ]

    for name, rows in (
        (TRIALS_FG, trials),
        (GRADER_RESULTS_FG, grader_rows),
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
            # per task, not one list for the suite: which graders apply depends
            # on what each task declares
            graders=_graders_for,
            config=RunnerConfig(
                n_trials=run.get("nTrials", 1),
                readiness_timeout_s=args.readiness_timeout_s,
                max_concurrency=args.max_concurrency,
            ),
        )
        _write_results(project.get_feature_store(), result, run)
        report(result.status)
        log.info("run %s finished: %s", args.run_id, result.status)
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

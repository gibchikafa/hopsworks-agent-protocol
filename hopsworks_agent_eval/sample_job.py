"""Grade a sample of production traffic, on a schedule.

    python -m hopsworks_agent_eval.sample_job --deployment-id 3 --sample 50

Offline runs tell you how an agent behaves on the cases someone thought to
write down. This tells you how it behaves on the cases users actually bring,
which is the larger and less flattering set.

Two things make it different from a suite run, and both simplify it: there is
no agent to call, because the traffic already happened; and there is no
expected output, because nobody wrote one. So the evaluators that apply are the
ones that judge a response on its own terms — a rubric judge, and the
trajectory checks that need only the trace.

Results go to the same ``agent_eval_evaluator_results`` feature group as offline
runs, under a synthetic run id, so one query answers "how is this agent
scoring" whether the evidence came from a suite or from production.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
from datetime import datetime, timedelta, timezone
from typing import Any

from .evaluators import NoToolErrorEvaluator, Trace, run_evaluators
from .judges import DEFAULT_MODEL, LlmJudgeEvaluator, anthropic_completer
from .models import Task, Trial, TrialStatus

log = logging.getLogger(__name__)

EVALUATOR_RESULTS_FG = "agent_eval_evaluator_results"


def _as_trace(detail: dict[str, Any]) -> Trace:
    spans = detail.get("spans") or []
    attributes = detail.get("spanAttributes") or []
    kinds, names = {}, {}
    for attribute in attributes:
        key, span_id = attribute.get("attrKey"), attribute.get("spanId")
        if key == "openinference.span.kind":
            kinds[span_id] = (attribute.get("attrValue") or "").upper()
        elif key in ("tool.name", "gen_ai.tool.name"):
            names.setdefault(span_id, attribute.get("attrValue") or "")
    tools = [s for s in spans if kinds.get(s.get("spanId")) == "TOOL"]
    return {
        "trace_id": detail.get("traceId", ""),
        "root_span_id": next(
            (s.get("spanId") for s in spans if not s.get("parentSpanId")), ""
        ),
        "tool_names": [
            names.get(s.get("spanId")) or s.get("name") or ""
            for s in sorted(tools, key=lambda s: s.get("startTimeNs") or 0)
        ],
        "tool_error_count": sum(
            1 for s in tools if str(s.get("statusCode", "")).endswith("ERROR")
        ),
    }


def _messages_of(detail: dict[str, Any]) -> tuple[str, str]:
    """The question and the answer, from whichever span carries them."""
    for span in sorted(
        detail.get("spans") or [], key=lambda s: s.get("startTimeNs") or 0
    ):
        raw = span.get("messages")
        if not raw:
            continue
        try:
            messages = json.loads(raw) if isinstance(raw, str) else raw
        except (ValueError, TypeError):
            continue
        question = next(
            (m["content"] for m in messages if m.get("role") == "user"), ""
        )
        answer = ""
        for message in messages:
            if message.get("role") == "assistant" and message.get("content"):
                answer = message["content"]
        if question or answer:
            return question, answer
    return "", ""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--deployment-id", type=int, required=True)
    parser.add_argument("--sample", type=int, default=25,
                        help="how many traces to grade; sampling exists because "
                             "grading everything costs a model call per request")
    parser.add_argument("--since-hours", type=float, default=24.0)
    parser.add_argument("--rubric", default="",
                        help="what a good answer looks like for this agent. "
                             "Without one there is nothing for a judge to grade "
                             "against, and only the trajectory checks run.")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)

    import hopsworks
    import pandas as pd
    import requests

    project = hopsworks.login()
    host = os.environ.get("HOPSWORKS_HOST") or os.environ["REST_ENDPOINT"]
    session = requests.Session()
    session.headers["Authorization"] = f"ApiKey {os.environ['HOPSWORKS_API_KEY']}"
    base = (f"{host.rstrip('/')}/hopsworks-api/api/project/{project.id}"
            f"/otel/servings/{args.deployment_id}")

    summaries = session.get(f"{base}/traces", params={"limit": 500}, timeout=60).json()
    items = summaries.get("items") or []
    cutoff = datetime.now(tz=timezone.utc) - timedelta(hours=args.since_hours)
    recent = [
        t for t in items
        if t.get("createdAt")
        and datetime.fromtimestamp(t["createdAt"] / 1000, tz=timezone.utc) >= cutoff
    ]
    if not recent:
        log.info("no traces in the last %.1f hours", args.since_hours)
        return

    # Random rather than newest-first: the most recent traces are not a sample
    # of behaviour, they are a sample of whoever was using it in the last hour.
    sampled = random.sample(recent, min(args.sample, len(recent)))
    log.info("grading %d of %d recent traces", len(sampled), len(recent))

    judge = None
    if args.rubric:
        try:
            key = project.get_secrets_api().get_secret("EVAL_JUDGE_API_KEY").value
            judge = LlmJudgeEvaluator(
                anthropic_completer(key, os.environ.get("EVAL_JUDGE_MODEL", DEFAULT_MODEL)),
                model=os.environ.get("EVAL_JUDGE_MODEL", DEFAULT_MODEL),
            )
        except Exception:  # noqa: BLE001 — no judge configured is a normal state
            log.info("no EVAL_JUDGE_API_KEY secret; only trajectory checks will run")

    run_id = f"online/{args.deployment_id}/{datetime.now(tz=timezone.utc):%Y%m%dT%H%M%S}"
    now = datetime.now(tz=timezone.utc)
    rows: list[dict[str, Any]] = []

    for summary in sampled:
        trace_id = summary["traceId"]
        try:
            detail = session.get(f"{base}/traces/{trace_id}", timeout=60).json()
        except Exception:  # noqa: BLE001 — one unreadable trace is not the run failing
            log.exception("could not read trace %s", trace_id)
            continue

        question, answer = _messages_of(detail)
        task = Task(
            task_id=trace_id,
            input_messages=json.dumps([{"role": "user", "content": question}]),
            rubric=args.rubric,
        )
        trial = Trial(
            trial_id=f"{run_id}/{trace_id}",
            run_id=run_id,
            task_id=trace_id,
            task_version=1,
            trial_index=0,
            deployment_id=args.deployment_id,
            trace_id=trace_id,
            final_output=answer,
            status=TrialStatus.PASSED,
        )

        evaluators = [NoToolErrorEvaluator()]
        if judge is not None:
            evaluators.insert(0, judge)

        for result in run_evaluators(evaluators, task, trial, _as_trace(detail)):
            rows.append({
                "run_id": run_id,
                "result_id": f"{trial.trial_id}/{result.evaluator_name}",
                "trial_id": trial.trial_id,
                "task_id": trace_id,
                "evaluator_name": result.evaluator_name,
                "evaluator_type": result.evaluator_type,
                "score": result.score,
                "passed": result.passed,
                "ungradable": result.ungradable,
                "reason": result.reason,
                "assertions_json": json.dumps(result.assertions),
                "judge_model": str(result.assertions.get("judge_model", "")),
                "evaluator_version": "1",
                "created_at": now,
            })

    if not rows:
        log.info("nothing gradable in the sample")
        return

    project.get_feature_store().get_feature_group(EVALUATOR_RESULTS_FG, 1).insert(
        pd.DataFrame(rows), write_options={"mode": "append"}
    )
    graded = len({r["trial_id"] for r in rows})
    ungradable = sum(1 for r in rows if r["ungradable"])
    log.info("wrote %d results for %d traces (%d ungradable) under run %s",
             len(rows), graded, ungradable, run_id)


if __name__ == "__main__":
    main()

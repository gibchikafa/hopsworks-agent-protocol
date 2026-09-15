"""The tracing side of the client, scoped to one deployment.

What the Trace Summaries card, the Feedback tab, the failure analysis and the
monitoring dashboards read and write, as methods on a deployment handle.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Sequence

from .evals import epoch_ms
from .models import (
    Calibration,
    Cluster,
    Feedback,
    FeedbackPage,
    FeedbackSummary,
    GateResult,
    LlmMetric,
    RegressionSuite,
    Run,
    ToolMetric,
    Trace,
    TraceMetric,
    TraceSummary,
    Triage,
)

if TYPE_CHECKING:
    from .client import AgentEvals
    from .models import EvalJob, ReviewJob, Task

VERDICTS = ("positive", "negative", "false_alarm")


class Deployment:
    """One agent deployment: its traces, feedback, analysis and the jobs that evaluate it."""

    def __init__(self, client: AgentEvals, deployment_id: int):
        self._client = client
        self._http = client.http
        self.id = int(deployment_id)
        self._otel = f"/otel/servings/{self.id}"

    def __repr__(self) -> str:
        return f"Deployment({self.id})"

    # ── traces ─────────────────────────────────────────────────────────────

    def traces(self, *, limit: int = 10, search: str | None = None, search_field: str | None = None,
               exclude_sessions: Sequence[str] = ()) -> list[TraceSummary]:
        """Recent traces, one per conversation unless searching.

        ``search_field`` is ``messages`` (default), ``subject`` (exact), ``session_id`` or ``trace_id``.
        """
        params: dict[str, Any] = {"limit": limit, "search": search, "searchField": search_field}
        if exclude_sessions:
            params["excludeSessions"] = list(exclude_sessions)
        return TraceSummary.list_from_api(self._http.get(f"{self._otel}/traces", **params))

    def trace(self, trace_id: str) -> Trace:
        """A whole trace: spans, attributes, events and totals."""
        return Trace.from_api(self._http.get(f"{self._otel}/traces/{trace_id}"))

    def session(self, session_id: str, *, limit: int = -1) -> list[TraceSummary]:
        """Every trace of a conversation, oldest first."""
        rows = TraceSummary.list_from_api(self._http.get(f"{self._otel}/traces/sessions/{session_id}", limit=limit))
        return sorted(rows, key=lambda t: t.start_time_ns)

    def conversation(self, session_id: str) -> list[dict[str, str]]:
        """The conversation as the user had it: one user and one assistant message per turn."""
        turns: list[dict[str, str]] = []
        previous: list[dict[str, Any]] = []
        for trace in self.session(session_id):
            messages = trace.conversation
            seen = {m.get("content") for m in previous if m.get("role") == "user"}
            users = [m for m in messages if m.get("role") == "user"]
            new_user = next((m for m in users if m.get("content") not in seen), users[-1] if users else None)
            if new_user is None:
                previous = messages
                continue
            after = messages[messages.index(new_user) + 1:]
            answer = "".join(str(m.get("content") or "") for m in after if m.get("role") == "assistant")
            turns.append({"trace_id": trace.trace_id, "user": str(new_user.get("content") or ""),
                          "assistant": answer})
            previous = messages
        return turns

    # ── feedback ───────────────────────────────────────────────────────────

    def feedback(self, *, limit: int = 50, offset: int = 0, verdict: str | None = None,
                 reviewer: str | None = None, search: str | None = None, session_id: str | None = None,
                 trace_id: str | None = None, feedback_id: str | None = None) -> FeedbackPage:
        """A page of feedback, newest first. ``verdict="negative"`` is everything that needs attention."""
        return FeedbackPage.from_api(self._http.get(
            f"{self._otel}/feedback", limit=limit, offset=offset, verdict=verdict, reviewer=reviewer,
            search=search, sessionId=session_id, traceId=trace_id, feedbackId=feedback_id))

    def all_feedback(self, **filters: Any) -> list[Feedback]:
        """Every matching feedback row, walking the pages."""
        rows: list[Feedback] = []
        offset = 0
        page_size = int(filters.pop("limit", 100))
        while True:
            page = self.feedback(limit=page_size, offset=offset, **filters)
            rows.extend(page.feedback)
            offset += len(page.items)
            if not page.items or offset >= page.count:
                return rows

    def trace_feedback(self, trace_id: str) -> list[Feedback]:
        return Feedback.list_from_api(self._http.get(f"{self._otel}/traces/{trace_id}/feedback"))

    def give_feedback(self, trace_id: str, verdict: str, *, issue_category: str | None = None,
                      corrected_answer: str | None = None, expected_tool_behavior: str | None = None,
                      note: str | None = None) -> Feedback:
        """Your verdict on a trace. One per reviewer per trace: giving it again replaces yours."""
        if verdict not in VERDICTS:
            raise ValueError(f"verdict must be one of {', '.join(VERDICTS)}")
        body = {
            "verdict": verdict,
            "issueCategory": issue_category,
            "correctedAnswer": corrected_answer,
            "expectedToolBehavior": expected_tool_behavior,
            "note": note,
        }
        return Feedback.from_api(self._http.post(f"{self._otel}/traces/{trace_id}/feedback",
                                                 {k: v for k, v in body.items() if v is not None}))

    def retract_feedback(self, trace_id: str) -> None:
        """Take back your own verdict on a trace. Never anyone else's."""
        self._http.delete(f"{self._otel}/traces/{trace_id}/feedback")

    def feedback_summary(self, *, since: Any = None, until: Any = None, window_ms: int | None = None,
                         human_only: bool = True) -> FeedbackSummary:
        """Verdict counts over a range, by window and by issue category."""
        return FeedbackSummary.from_api(self._http.get(
            f"{self._otel}/feedback/summary", **{"from": epoch_ms(since), "to": epoch_ms(until)},
            windowMs=window_ms, source="human" if human_only else "all"))

    def reviewers(self) -> list[str]:
        return list(self._http.get(f"{self._otel}/feedback/reviewers") or [])

    # ── analysis ───────────────────────────────────────────────────────────

    def triage(self, feedback: Sequence[Feedback | str]) -> list[Triage]:
        """The analysis model's latest proposal about each of these feedback rows."""
        ids = [f.feedback_id if isinstance(f, Feedback) else f for f in feedback]
        if not ids:
            return []
        rows = self._http.get(f"{self._otel}/feedback/triage", feedbackId=ids)
        return Triage.list_from_api(rows)

    def decide_triage(self, triage: Triage | str, decision: str, *, category: str | None = None) -> None:
        """What you think of a proposal: ``accepted``, ``edited`` or ``rejected``. This is calibration data."""
        triage_id = triage.triage_id if isinstance(triage, Triage) else triage
        self._http.put(f"{self._otel}/feedback/triage/{triage_id}/decision", decision=decision, category=category)

    def calibration(self) -> Calibration:
        return Calibration.from_api(self._http.get(f"{self._otel}/feedback/triage/calibration"))

    def clusters(self, status: str | None = "open") -> list[Cluster]:
        """Failure clusters, most worth a reviewer's time first. ``status=None`` for all."""
        return Cluster.list_from_api(self._http.get(f"{self._otel}/feedback/clusters", status=status))

    def cluster_members(self, cluster: Cluster | str) -> list[Triage]:
        cluster_id = cluster.cluster_id if isinstance(cluster, Cluster) else cluster
        return Triage.list_from_api(self._http.get(f"{self._otel}/feedback/clusters/{cluster_id}/members"))

    def rename_cluster(self, cluster: Cluster | str, label: str) -> None:
        self._decide_cluster(cluster, label=label)

    def dismiss_cluster(self, cluster: Cluster | str, reason: str) -> None:
        """``working_as_intended``, ``duplicate_of``, ``cannot_reproduce`` or ``out_of_scope``."""
        self._decide_cluster(cluster, status="dismissed", dismissReason=reason)

    def reopen_cluster(self, cluster: Cluster | str) -> None:
        self._decide_cluster(cluster, status="open")

    def mark_cluster_promoted(self, cluster: Cluster | str, task: Task | str) -> None:
        """Record the task a cluster became; every member's feedback is marked covered."""
        task_id = getattr(task, "task_id", task)
        self._decide_cluster(cluster, status="promoted", promotedTaskId=task_id)

    def promote_cluster(self, cluster: Cluster, *, expectations: dict[str, str] | None = None) -> Task:
        """Promote the cluster's representative trace to a task and record it on the cluster."""
        if not cluster.representative_trace_id:
            raise ValueError("the cluster has no representative trace")
        if expectations is None:
            expectations = {}
            members = self.cluster_members(cluster)
            representative = next((m for m in members if m.feedback_id == cluster.representative_feedback_id), None)
            if representative and representative.normalized_correction:
                expectations = {"expected": representative.normalized_correction}
        task = self._client.tasks.promote(self.id, cluster.representative_trace_id, expectations=expectations)
        self.mark_cluster_promoted(cluster, task)
        return task

    def _decide_cluster(self, cluster: Cluster | str, **params: Any) -> None:
        cluster_id = cluster.cluster_id if isinstance(cluster, Cluster) else cluster
        self._http.put(f"{self._otel}/feedback/clusters/{cluster_id}", **params)

    # ── metrics ────────────────────────────────────────────────────────────

    def trace_metrics(self, *, since: Any = None, until: Any = None) -> list[TraceMetric]:
        return TraceMetric.list_from_api(self._http.get(f"{self._otel}/metrics/traces",
                                                        **{"from": epoch_ms(since), "to": epoch_ms(until)}))

    def llm_metrics(self, *, since: Any = None, until: Any = None) -> list[LlmMetric]:
        return LlmMetric.list_from_api(self._http.get(f"{self._otel}/metrics/llm",
                                                      **{"from": epoch_ms(since), "to": epoch_ms(until)}))

    def tool_metrics(self, *, since: Any = None, until: Any = None) -> list[ToolMetric]:
        return ToolMetric.list_from_api(self._http.get(f"{self._otel}/metrics/tools",
                                                       **{"from": epoch_ms(since), "to": epoch_ms(until)}))

    def tracing_ready(self) -> bool:
        body = self._http.get("/otel/ready") or {}
        return bool(body.get("ready", body) if isinstance(body, dict) else body)

    # ── evaluation, from this deployment's side ────────────────────────────

    def runs(self) -> list[Run]:
        return self._client.runs.list(self.id)

    def run(self, suite: Any, *, version: int | None = None, n_trials: int = 1) -> Run:
        return self._client.runs.start(suite, self.id, version=version, n_trials=n_trials)

    def sample(self, **kwargs: Any) -> Run:
        return self._client.runs.sample(self.id, **kwargs)

    def gates(self) -> GateResult:
        """Whether this deployment's evaluation evidence clears the bar for promotion."""
        return GateResult.from_api(self._http.get("/agent-evals/gates", deploymentId=self.id))

    def canary(self) -> list[Run]:
        """Start the project's canary suites against this deployment."""
        return self._client._bind_all(Run.list_from_api(self._http.post("/agent-evals/canary", deploymentId=self.id)))

    def eval_job(self) -> EvalJob:
        return self._client.jobs.eval_job(self.id)

    def review_job(self) -> ReviewJob | None:
        return self._client.jobs.review_job(self.id)

    def regressions(self) -> RegressionSuite:
        return self._client.jobs.regressions(self.id)

    def analyse(self, **kwargs: Any) -> Run:
        """Run the failure analysis now; the job is created with defaults if the deployment has none."""
        job = self.review_job() or self._client.jobs.ensure_review_job(self.id)
        return self._client.jobs.analyse(job, **kwargs)

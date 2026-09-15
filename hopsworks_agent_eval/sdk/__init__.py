"""A client for everything the Hopsworks UI does for agent evaluation and tracing.

    from hopsworks_agent_eval.sdk import login, check

    evals = login()
    deployment = evals.deployment(7)
    for cluster in deployment.clusters(): ...
"""

from ._transport import AgentEvalsError
from .client import AgentEvals, login
from .evals import check
from .models import (
    Calibration,
    Check,
    Cluster,
    EvalJob,
    EvaluatorResult,
    EvaluatorTemplate,
    Feedback,
    FeedbackPage,
    FeedbackSummary,
    GateCheck,
    GateResult,
    LlmMetric,
    RegressionSuite,
    ReviewJob,
    Run,
    RunMetric,
    Suite,
    Task,
    ToolMetric,
    Trace,
    TraceMetric,
    TraceSummary,
    Trial,
    Triage,
)
from .tracing import Deployment

__all__ = [
    "AgentEvals", "AgentEvalsError", "Calibration", "Check", "Cluster", "Deployment", "EvalJob",
    "EvaluatorResult", "EvaluatorTemplate", "Feedback", "FeedbackPage", "FeedbackSummary", "GateCheck",
    "GateResult", "LlmMetric", "RegressionSuite", "ReviewJob", "Run", "RunMetric", "Suite", "Task",
    "ToolMetric", "Trace", "TraceMetric", "TraceSummary", "Trial", "Triage", "check", "login",
]

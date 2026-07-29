"""Hopsworks agent evaluation: featurization, runner, and graders.

Runs in a Hopsworks job, never in a serving pod. The one structural rule:

    hopsworks_agent_eval  may import  hopsworks_agent_protocol
    hopsworks_agent_protocol  must never import  hopsworks_agent_eval

The serving package runs synchronously in every request; nothing in here
belongs on that path, and an import is how it would get there by accident.
``tests/test_import_isolation.py`` asserts the rule rather than trusting it.
"""

from .features import (
    TraceCompleteness,
    select_ready_traces,
    trace_features,
)
from .metrics import pass_all_k, pass_at_k, run_metrics
from .models import (
    ExecutionMode,
    Suite,
    SuiteType,
    Task,
    Trial,
    TrialStatus,
    derive_trial_id,
)
from .promotion import (
    CandidateTask,
    RedactionStatus,
    can_add_to_suite,
    confirm_redaction,
    promote_trace,
    tasks_from_trace,
)
from .client import HopsworksAgentClient
from .runner import RunnerConfig, SuiteRefused, run_suite

__all__ = [
    "ExecutionMode",
    "HopsworksAgentClient",
    "RunnerConfig",
    "Suite",
    "SuiteRefused",
    "SuiteType",
    "Task",
    "Trial",
    "TrialStatus",
    "derive_trial_id",
    "pass_all_k",
    "pass_at_k",
    "run_metrics",
    "run_suite",
    "CandidateTask",
    "RedactionStatus",
    "TraceCompleteness",
    "can_add_to_suite",
    "confirm_redaction",
    "promote_trace",
    "select_ready_traces",
    "tasks_from_trace",
    "trace_features",
]

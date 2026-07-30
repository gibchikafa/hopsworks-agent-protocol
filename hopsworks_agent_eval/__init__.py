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
    PassPolicy,
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
from .judge_config import JudgeConfig, JudgeConfigError, default_templates
from .grader_spec import SpecError, graders_for_task, graders_from_spec, validate_spec
from .judges import (
    LlmJudgeGrader,
    MultiCriteriaJudge,
    PairwiseGrader,
    anthropic_completer,
    pairwise_verdict,
)
from .runner import RunnerConfig, SuiteRefused, run_suite

__all__ = [
    "ExecutionMode",
    "HopsworksAgentClient",
    "LlmJudgeGrader",
    "anthropic_completer",
    "pairwise_verdict",
    "MultiCriteriaJudge",
    "PairwiseGrader",
    "SpecError",
    "JudgeConfig",
    "JudgeConfigError",
    "default_templates",
    "graders_for_task",
    "graders_from_spec",
    "validate_spec",
    "RunnerConfig",
    "PassPolicy",
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

"""Suites, tasks, runs, trials — the shapes the runner passes around.

Plain dataclasses rather than the feature-store rows they eventually become:
the runner should be testable without a feature store, and the row shape is a
serialisation detail that belongs at the edge.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Sequence


class ExecutionMode(str, Enum):
    """How dangerous it is to point this suite at a deployment.

    Running a tool-using agent executes real tools. A ``safety`` suite is full
    of injection and exfiltration attempts by construction, so pointing one at
    an agent whose tools can mutate production is not a configuration mistake,
    it is the whole incident.
    """

    READ_ONLY = "read_only"
    SANDBOXED = "sandboxed"
    # Deliberately unsupported in v1: it needs a per-suite allowlist of tools
    # the run may trigger, and a runner that enforces it.
    LIVE = "live"


class SuiteType(str, Enum):
    REGRESSION = "regression"
    CAPABILITY = "capability"
    TOOL_USE = "tool_use"
    SAFETY = "safety"
    LATENCY_COST = "latency_cost"
    GOLDEN = "golden"
    CANARY = "canary"


class TrialStatus(str, Enum):
    PASSED = "PASSED"
    FAILED = "FAILED"
    # Classified failures, kept apart so a broken pipeline is never read as a
    # broken agent.
    AGENT_ERROR = "AGENT_ERROR"
    TOOL_ERROR = "TOOL_ERROR"
    TIMEOUT = "TIMEOUT"
    GRADER_ERROR = "GRADER_ERROR"
    INFRA_ERROR = "INFRA_ERROR"
    INVALID_OUTPUT = "INVALID_OUTPUT"
    TRACE_MISSING = "TRACE_MISSING"
    # An outcome, not inherently a failure: in a safety suite a block is the
    # desired result. Folding it into AGENT_ERROR would make guardrail
    # behaviour unattributable.
    BLOCKED_BY_GUARDRAIL = "BLOCKED_BY_GUARDRAIL"


class TraceStatus(str, Enum):
    RECEIVED = "RECEIVED"
    PARTIAL = "PARTIAL"
    MISSING = "MISSING"


@dataclass
class Task:
    task_id: str
    input_messages: str
    task_version: int = 1
    task_type: str = "single_turn"
    expected_output: str = ""
    required_tools: list[str] = field(default_factory=list)
    forbidden_tools: list[str] = field(default_factory=list)
    rubric: str = ""
    category: str = ""
    tags: list[str] = field(default_factory=list)

    @property
    def prompt(self) -> str:
        """The user text to send. Tasks store a message array; the adapter
        needs the text of the last user turn."""
        import json

        try:
            messages = json.loads(self.input_messages)
        except (ValueError, TypeError):
            return self.input_messages
        if isinstance(messages, list):
            user = [m for m in messages if m.get("role") == "user"]
            if user:
                return str(user[-1].get("content", ""))
        return ""


@dataclass
class Suite:
    suite_id: str
    suite_version: int = 1
    name: str = ""
    type: SuiteType = SuiteType.REGRESSION
    execution_mode: ExecutionMode = ExecutionMode.READ_ONLY
    tasks: list[Task] = field(default_factory=list)


@dataclass
class GraderResult:
    grader_name: str
    grader_type: str
    score: float
    passed: bool
    reason: str = ""
    assertions: dict[str, Any] = field(default_factory=dict)
    # A grader that could not judge -- a trajectory grader with no trace, say.
    # Distinct from failing: an ungradable trial must not count against the
    # agent, and must not silently count for it either.
    ungradable: bool = False


@dataclass
class Trial:
    trial_id: str
    run_id: str
    task_id: str
    task_version: int
    trial_index: int
    deployment_id: int
    trace_id: str = ""
    trace_status: TraceStatus = TraceStatus.MISSING
    status: TrialStatus = TrialStatus.FAILED
    final_output: str = ""
    latency_ms: float = 0.0
    error_type: str = ""
    error_message: str = ""
    grader_results: list[GraderResult] = field(default_factory=list)
    started_at: datetime = field(default_factory=lambda: datetime.now(tz=timezone.utc))
    completed_at: datetime | None = None


def derive_trial_id(run_id: str, task_id: str, task_version: int, trial_index: int) -> str:
    """Deterministic, so a crashed runner that retries overwrites the trial's
    row instead of appending a second one. A random id here would turn every
    infrastructure hiccup into duplicated results and a wrong pass rate."""
    return f"{run_id}/{task_id}/{task_version}/{trial_index}"


def gradable_trials(trials: Sequence[Trial]) -> list[Trial]:
    """Trials whose outcome says something about the agent.

    ``INFRA_ERROR`` and ``GRADER_ERROR`` are the harness failing, not the
    agent; counting them as failures makes a flaky network look like a
    regression and sends someone to debug a prompt.
    """
    return [
        t
        for t in trials
        if t.status not in (TrialStatus.INFRA_ERROR, TrialStatus.GRADER_ERROR)
    ]

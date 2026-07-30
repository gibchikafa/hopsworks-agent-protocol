"""Building graders from what a task declares.

Before this, graders were *inferred*: a rubric got a judge, an expected output
got a substring check, tools got the tool graders. That covers the common cases
and reaches none of the others — a task had no way to ask for a regex, a JSON
shape, or a state check, so four implemented graders could never run.

A spec is a JSON array stored on the task:

    [{"type": "regex", "pattern": "^ORD-\\\\d{4}$"},
     {"type": "json_schema", "required_keys": ["order_id", "status"]}]

Inference stays as the default for a task that declares nothing, because most
tasks should not have to.

**Nothing here executes caller-supplied code.** ``FunctionGrader`` is
deliberately absent: it wraps a Python callable, and resolving one from a
string in a task row would turn "author a task" into "run code in the job".
Deterministic Python graders remain available to code that builds a run
directly.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Callable, Sequence

from .graders import (
    ContainsGrader,
    ExactMatchGrader,
    Grader,
    HumanReviewGrader,
    JsonSchemaGrader,
    NoToolErrorGrader,
    RegexGrader,
    SqlStateGrader,
    ToolArgumentGrader,
    ToolLatencyGrader,
    ToolRetryGrader,
    UnnecessaryToolGrader,
    ToolCallGrader,
    ToolOrderGrader,
)
from .judge_config import JudgeConfigError, completer_for, parse_judge_config
from .models import Task

log = logging.getLogger(__name__)

# The types a task may name. Kept explicit rather than derived from a module
# scan so adding a grader is a decision rather than an accident.
SPEC_TYPES = (
    "exact_match",
    "contains",
    "regex",
    "json_schema",
    "tool_call",
    "tool_order",
    "no_tool_error",
    "tool_arguments",
    "no_unnecessary_tools",
    "tool_retries",
    "tool_latency",
    "tool_arguments_judge",
    "tool_result_used",
    "llm_judge",
    "pairwise",
    "sql_state",
    "human_review",
)


class SpecError(ValueError):
    """A spec that cannot be turned into a grader, with the reason."""


def _one(
    entry: dict[str, Any],
    *,
    judge_completer: Callable[[str], str] | None,
    query: Callable[[str], Any] | None,
    secret_reader: Callable[[str], str | None] | None = None,
) -> Grader | None:
    kind = str(entry.get("type") or "").strip()
    name = str(entry.get("name") or kind or "grader")

    if kind == "exact_match":
        return ExactMatchGrader(
            name=name, case_sensitive=bool(entry.get("case_sensitive", False))
        )
    if kind == "contains":
        return ContainsGrader(name=name, expected=entry.get("expected"))
    if kind == "regex":
        pattern = entry.get("pattern")
        if not pattern:
            raise SpecError("a regex grader needs a pattern")
        return RegexGrader(
            pattern=str(pattern),
            name=name,
            should_match=bool(entry.get("should_match", True)),
        )
    if kind == "json_schema":
        keys = entry.get("required_keys") or entry.get("requiredKeys") or []
        if not isinstance(keys, (list, tuple)) or not keys:
            raise SpecError("a json_schema grader needs required_keys")
        return JsonSchemaGrader(required_keys=[str(k) for k in keys], name=name)
    if kind == "tool_call":
        return ToolCallGrader(name=name)
    if kind == "tool_order":
        return ToolOrderGrader(name=name)
    if kind == "no_tool_error":
        return NoToolErrorGrader(name=name)
    if kind == "tool_arguments":
        keys = entry.get("required_keys") or entry.get("requiredKeys") or []
        return ToolArgumentGrader(
            tool=str(entry.get("tool") or ""),
            required_keys=[str(k) for k in keys],
            must_parse=bool(entry.get("must_parse", True)),
            name=name,
        )
    if kind == "no_unnecessary_tools":
        allowed = entry.get("allowed") or []
        return UnnecessaryToolGrader(allowed=[str(a) for a in allowed], name=name)
    if kind == "tool_retries":
        return ToolRetryGrader(
            max_retries=int(entry.get("max_retries", 0)),
            tool=str(entry.get("tool") or ""),
            name=name,
        )
    if kind == "tool_latency":
        budget = entry.get("max_ms")
        if budget is None:
            raise SpecError("a tool_latency grader needs max_ms")
        return ToolLatencyGrader(
            max_ms=float(budget), tool=str(entry.get("tool") or ""), name=name
        )
    if kind == "human_review":
        return HumanReviewGrader(prompt=str(entry.get("prompt") or ""), name=name)
    if kind == "sql_state":
        sql = entry.get("sql") or entry.get("query")
        if not sql:
            raise SpecError("a sql_state grader needs a sql query")
        return SqlStateGrader(
            sql=str(sql), expect=entry.get("expect"), query=query, name=name
        )

    if kind in ("llm_judge", "pairwise", "tool_arguments_judge", "tool_result_used"):
        # Imported here because judges pulls in the provider plumbing, and a
        # suite with no judge configured should not need it loaded.
        from .judges import (
            LlmJudgeGrader,
            PairwiseGrader,
            ToolArgumentsJudge,
            ToolResultUsedJudge,
        )

        # A judge may bring its own provider, model and key. Falling back to the
        # project default keeps every spec written before that was possible
        # working unchanged.
        completer = judge_completer
        config = None
        if kind == "llm_judge":
            try:
                config = parse_judge_config(entry)
            except JudgeConfigError as err:
                raise SpecError(str(err)) from err
        if config is not None and (config.model or entry.get("provider")):
            api_key = secret_reader(config.api_key_secret) if secret_reader else None
            if api_key:
                completer = completer_for(config, api_key)
            elif secret_reader is not None:
                log.info(
                    "no secret %r for judge %r; skipping it",
                    config.api_key_secret, name,
                )
                return None

        if completer is None:
            # Not an error: a project without a judge key still runs its
            # deterministic graders, and the trial reports what was skipped
            # rather than pretending the judgement happened.
            log.info("no judge configured; skipping %s grader %r", kind, name)
            return None
        if kind == "llm_judge":
            if config is not None and config.multi:
                from .judges import MultiCriteriaJudge

                return MultiCriteriaJudge(completer, config, name=name)
            return LlmJudgeGrader(
                completer,
                name=name,
                pass_threshold=float(entry.get("pass_threshold", 0.7)),
            )
        if kind == "tool_arguments_judge":
            return ToolArgumentsJudge(
                completer, tool=str(entry.get("tool") or ""), name=name
            )
        if kind == "tool_result_used":
            return ToolResultUsedJudge(completer, name=name)
        return PairwiseGrader(
            completer, reference=str(entry.get("reference") or ""), name=name
        )

    raise SpecError(
        f"unknown grader type {kind!r}; expected one of {', '.join(SPEC_TYPES)}"
    )


def graders_from_spec(
    spec: str | Sequence[dict[str, Any]] | None,
    *,
    judge_completer: Callable[[str], str] | None = None,
    query: Callable[[str], Any] | None = None,
    secret_reader: Callable[[str], str | None] | None = None,
) -> list[Grader]:
    """Every grader a spec asks for.

    Raises :class:`SpecError` on anything malformed. The caller decides what a
    bad spec means — the backend refuses it at authoring time, so by the time a
    run reads one it should already be valid.
    """
    if not spec:
        return []
    entries: Any = spec
    if isinstance(spec, str):
        try:
            entries = json.loads(spec)
        except (ValueError, TypeError) as err:
            raise SpecError(f"not valid JSON: {err}") from err
    if not isinstance(entries, (list, tuple)):
        raise SpecError("a grader spec is an array of objects")

    graders: list[Grader] = []
    for index, entry in enumerate(entries):
        if not isinstance(entry, dict):
            raise SpecError(f"entry {index + 1} is not an object")
        try:
            grader = _one(entry, judge_completer=judge_completer, query=query,
                          secret_reader=secret_reader)
        except SpecError as err:
            raise SpecError(f"entry {index + 1}: {err}") from err
        if grader is not None:
            graders.append(grader)
    return graders


def validate_spec(spec: str | None) -> None:
    """Raise :class:`SpecError` unless every entry could be built.

    Uses stand-in judge and query callables so a spec naming those graders
    validates without a provider key or a database session — authoring a task
    must not depend on runtime configuration that only the job has.
    """
    graders_from_spec(spec, judge_completer=lambda _prompt: "", query=lambda _sql: None)


def graders_for_task(
    task: Task,
    *,
    judge_completer: Callable[[str], str] | None = None,
    query: Callable[[str], Any] | None = None,
    secret_reader: Callable[[str], str | None] | None = None,
) -> list[Grader]:
    """What this task gets: its spec if it has one, inference otherwise.

    A task that declares a spec gets exactly that and no additions. Quietly
    appending inferred graders would mean a task asking for one regex could
    still fail on a substring check it never asked for.
    """
    if task.graders:
        return graders_from_spec(
            task.graders, judge_completer=judge_completer, query=query,
            secret_reader=secret_reader,
        )

    graders: list[Grader] = []
    if judge_completer is not None and task.rubric:
        from .judges import LlmJudgeGrader

        graders.append(LlmJudgeGrader(judge_completer))
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

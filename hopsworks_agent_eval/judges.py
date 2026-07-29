"""LLM judges: grading what a deterministic check cannot express.

A rubric judge earns its place where correctness is not a string comparison —
"did it answer the question without inventing a policy", "was the tone right for
a refund refusal". It is also the least trustworthy grader in the set, so
everything here is built around that:

- The judge model is recorded on every result. A score that moved because the
  judge changed is not a regression in the agent, and without the model on the
  row the two are indistinguishable.
- A judge that fails, times out, or returns something unparseable comes back
  **ungradable**, never zero. Zero is a claim about the agent; an unparseable
  response is a claim about the judge.
- The rubric and the answer go in; the task's own expected output goes in only
  when it exists. A judge asked to score against nothing will still return a
  number, and that number means nothing.

Provider is configured per project, with the key from project secrets, so this
never carries credentials of its own.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any, Callable

from .graders import Trace, _ungradable
from .models import GraderResult, Task, Trial

log = logging.getLogger(__name__)

DEFAULT_MODEL = "claude-sonnet-4-5"

RUBRIC_PROMPT = """You are grading one response from an AI agent.

Grade only against the criteria given. Do not reward or penalise style, length \
or confidence beyond what the criteria ask for.

<question>
{question}
</question>

<agent_answer>
{answer}
</agent_answer>
{expected_block}{rubric_block}
Reply with JSON only, no prose:
{{"score": <0.0-1.0>, "passed": <true|false>, "reason": "<one sentence>", \
"criteria": {{"<name>": <0.0-1.0>}}}}"""

PAIRWISE_PROMPT = """You are comparing two AI agent responses to the same question.

Judge which better satisfies the criteria. If they are equivalent, say "tie" — \
do not break a genuine tie arbitrarily, since a forced winner reads as a real \
difference downstream.

<question>
{question}
</question>

<response_a>
{answer_a}
</response_a>

<response_b>
{answer_b}
</response_b>
{rubric_block}
Reply with JSON only, no prose:
{{"winner": "<a|b|tie>", "reason": "<one sentence>"}}"""


def _json_from(text: str) -> dict[str, Any] | None:
    """The JSON a model returned, whatever it wrapped it in.

    Models fence JSON, prefix it with "Here is", or both. Salvaging is worth it
    because the alternative is discarding a valid judgement over formatting.
    """
    if not text:
        return None
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.S)
    candidate = fenced.group(1) if fenced else text
    try:
        parsed = json.loads(candidate)
        return parsed if isinstance(parsed, dict) else None
    except (ValueError, TypeError):
        pass
    braces = re.search(r"\{.*\}", candidate, re.S)
    if not braces:
        return None
    try:
        parsed = json.loads(braces.group(0))
        return parsed if isinstance(parsed, dict) else None
    except (ValueError, TypeError):
        return None


class LlmJudgeGrader:
    """Score a trial against a rubric using a model.

    ``complete`` is injected — a callable taking a prompt and returning text —
    so the grading logic is testable without a provider, and so the provider
    stays a project configuration rather than something this module chooses.
    """

    type = "llm_judge"
    needs_trace = False

    def __init__(
        self,
        complete: Callable[[str], str],
        *,
        name: str = "llm_judge",
        model: str = DEFAULT_MODEL,
        pass_threshold: float = 0.7,
    ):
        self._complete = complete
        self.name = name
        self.model = model
        self.pass_threshold = pass_threshold

    def grade(self, task: Task, trial: Trial, trace: Trace | None) -> GraderResult:
        if not task.rubric and not task.expected_output:
            # A judge asked to score against nothing returns a number anyway,
            # and that number is noise dressed as a measurement.
            return _ungradable(
                self.name, self.type,
                "no rubric and no expected output: there is nothing to grade against",
            )
        if not trial.final_output:
            return _ungradable(self.name, self.type, "the agent produced no answer")

        prompt = RUBRIC_PROMPT.format(
            question=task.prompt,
            answer=trial.final_output,
            expected_block=(
                f"\n<expected>\n{task.expected_output}\n</expected>\n"
                if task.expected_output else ""
            ),
            rubric_block=(
                f"\n<criteria>\n{task.rubric}\n</criteria>\n"
                if task.rubric else ""
            ),
        )

        try:
            raw = self._complete(prompt)
        except Exception as err:  # noqa: BLE001 — a broken judge is not a broken agent
            log.warning("judge call failed: %s", err)
            return _ungradable(self.name, self.type, f"judge call failed: {err}")

        parsed = _json_from(raw)
        if parsed is None or "score" not in parsed:
            return _ungradable(
                self.name, self.type,
                "judge did not return usable JSON; scoring zero here would blame "
                "the agent for the judge",
            )

        try:
            score = max(0.0, min(1.0, float(parsed["score"])))
        except (TypeError, ValueError):
            return _ungradable(self.name, self.type, "judge returned a non-numeric score")

        # The model's own pass/fail is honoured when it gave one, since a rubric
        # can encode a bar that a single threshold cannot.
        passed = bool(parsed["passed"]) if isinstance(parsed.get("passed"), bool) \
            else score >= self.pass_threshold

        return GraderResult(
            grader_name=self.name,
            grader_type=self.type,
            score=score,
            passed=passed,
            reason=str(parsed.get("reason", ""))[:1000],
            assertions={
                "criteria": parsed.get("criteria", {}),
                # recorded so a score shift can be attributed to a judge change
                # rather than mistaken for an agent regression
                "judge_model": self.model,
            },
        )


def pairwise_verdict(
    complete: Callable[[str], str],
    question: str,
    answer_a: str,
    answer_b: str,
    rubric: str = "",
) -> dict[str, Any]:
    """Which of two answers is better, or a tie.

    Used to compare deployments on the same task. Ties are reported rather than
    broken: a forced winner reads downstream as a real difference, and two
    equivalent answers are the common case when comparing close versions.
    """
    prompt = PAIRWISE_PROMPT.format(
        question=question,
        answer_a=answer_a,
        answer_b=answer_b,
        rubric_block=f"\n<criteria>\n{rubric}\n</criteria>\n" if rubric else "",
    )
    try:
        parsed = _json_from(complete(prompt))
    except Exception as err:  # noqa: BLE001
        return {"winner": "unknown", "reason": f"judge call failed: {err}"}
    if not parsed or parsed.get("winner") not in ("a", "b", "tie"):
        return {"winner": "unknown", "reason": "judge did not return a usable verdict"}
    return {"winner": parsed["winner"], "reason": str(parsed.get("reason", ""))[:1000]}


def anthropic_completer(api_key: str, model: str = DEFAULT_MODEL,
                        max_tokens: int = 1024) -> Callable[[str], str]:
    """A ``complete`` backed by Anthropic.

    Imported lazily so the eval package does not require a provider SDK to be
    installed for the graders that need no model at all.
    """
    def complete(prompt: str) -> str:
        import anthropic

        client = anthropic.Anthropic(api_key=api_key)
        response = client.messages.create(
            model=model,
            max_tokens=max_tokens,
            messages=[{"role": "user", "content": prompt}],
        )
        return "".join(
            block.text for block in response.content if getattr(block, "type", "") == "text"
        )

    return complete

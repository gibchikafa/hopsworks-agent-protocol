"""Ready-made summarizers for :class:`PersistentAgentMemory`.

The SDK owns *when* to summarize (the trigger, the fold cutoff, the
transaction); the model call itself is yours, so the SDK stays LLM-agnostic.
A summarizer is any callable::

    (previous_summary: str | None, turns: list[Turn]) -> str

sync or async. This module ships one for Claude so the common case is a single
line; anything else — a local model, a different provider, a rules-based
compactor — is just a function with that shape.
"""

from __future__ import annotations

import logging
import os

log = logging.getLogger(__name__)

DEFAULT_MODEL = "claude-haiku-4-5"

SYSTEM_PROMPT = """\
You maintain a running summary of a conversation between a user and an AI \
agent. You will be given the summary so far (which may be empty) and the turns \
that have happened since. Return an updated summary that folds the new turns \
into the old one.

Write for the agent that will read this as its only memory of the earlier \
conversation:

- Keep facts the user stated about themselves, their data, their goals, and \
their preferences. These are the whole point — losing one means the agent \
asks again.
- Keep decisions, conclusions, and anything the user corrected you on.
- Keep unresolved threads: open questions, things the user said they would \
come back to.
- Drop pleasantries, restatements, and detail that no longer changes what the \
agent would do next.
- Write plain prose or short bullets. No preamble, no "here is the summary", \
no meta-commentary about summarizing.

Stay under roughly 400 words. When the summary is at risk of growing past \
that, compress the oldest material further rather than dropping recent \
material.\
"""


def _format_turns(turns) -> str:
    return "\n\n".join(f"{t['role']}: {t['content']}" for t in turns)


def anthropic_summarizer(
    model: str = DEFAULT_MODEL,
    *,
    api_key: str | None = None,
    api_key_secret: str | None = None,
    max_tokens: int = 1024,
    system_prompt: str = SYSTEM_PROMPT,
):
    """An async summarizer backed by the Claude API.

        memory = PersistentAgentMemory(summarize=anthropic_summarizer())

    Requires ``pip install anthropic``.

    The key is resolved once, on first use rather than at construction, so
    importing this never reaches for a credential: explicit ``api_key`` →
    ``ANTHROPIC_API_KEY`` → the Hopsworks secret named by ``api_key_secret`` (or
    ``ANTHROPIC_API_KEY_SECRET_NAME``). That last hop is how a deployment reuses
    the same secret the agent already uses for its own model, instead of
    provisioning a second one for summarization.

    Defaults to Haiku 4.5: this is a bounded rewrite of text the agent already
    produced, and the cost lands on every Nth turn of every conversation. Pass
    ``model=`` for anything else.
    """
    try:
        import anthropic  # noqa: F401
    except ImportError as err:
        raise ImportError(
            "anthropic_summarizer requires the Anthropic SDK: pip install anthropic"
        ) from err

    client = None

    def _resolve_key() -> str | None:
        if api_key:
            return api_key
        env = os.environ.get("ANTHROPIC_API_KEY")
        if env:
            return env
        secret = api_key_secret or os.environ.get("ANTHROPIC_API_KEY_SECRET_NAME")
        if not secret:
            return None
        try:
            import hopsworks

            return hopsworks.get_secrets_api().get(secret)
        except Exception:  # noqa: BLE001 — fall through to the SDK's own resolution
            log.exception("Could not read the Anthropic key from Hopsworks secret %s", secret)
            return None

    async def summarize(previous: str | None, turns) -> str:
        nonlocal client
        if client is None:
            import anthropic

            key = _resolve_key()
            client = (
                anthropic.AsyncAnthropic(api_key=key)
                if key
                else anthropic.AsyncAnthropic()
            )

        prior = previous or "(none — this is the first summary)"
        response = await client.messages.create(
            model=model,
            max_tokens=max_tokens,
            system=system_prompt,
            messages=[
                {
                    "role": "user",
                    "content": (
                        f"<summary_so_far>\n{prior}\n</summary_so_far>\n\n"
                        f"<new_turns>\n{_format_turns(turns)}\n</new_turns>\n\n"
                        "Return the updated summary."
                    ),
                }
            ],
        )
        return "".join(
            block.text for block in response.content if block.type == "text"
        ).strip()

    return summarize

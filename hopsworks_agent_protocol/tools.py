"""Model-callable memory tools.

``remember`` / ``recall`` / ``forget`` are plain functions the agent's own LLM
decides to call — the MemGPT shape, and the reason there is no extraction model
in the SDK: the agent already knows what is worth keeping, so nothing has to
guess on its behalf.

They take no store, subject, or conversation argument. Those come from the
per-request context, so the signature the model sees carries no plumbing it
would have to invent values for. Register them with
``agent_app.memory_tools(framework)``.
"""

from __future__ import annotations

import json
import logging

from .memory import WRITABLE_SCOPES, WRITTEN_BY_AGENT

log = logging.getLogger(__name__)

_NO_CONTEXT = (
    "Memory is unavailable in this call, so nothing was stored. Continue "
    "without it and do not retry."
)


def _resolve():
    """The active turn's (memory, ctx), or (None, None).

    The context is a contextvar set on the request's event loop. A tool body
    running in a thread the loop did not hand off to — ``run_in_executor``
    rather than ``to_thread`` — will not see it. That degrades to a logged
    no-op returning a sentence the model can act on, rather than raising inside
    a tool call, which the model would read as a tool failure and retry.
    """
    from .autoevents import current_context

    ctx = current_context.get(None)
    if ctx is None or ctx.memory is None:
        log.warning("Memory tool called with no active request context")
        return None, None
    return ctx.memory, ctx


def _owner(ctx, scope: str) -> str:
    return ctx.conversation_id if scope == "session" else ctx.subject


def _check_scope(scope: str) -> str | None:
    if scope not in WRITABLE_SCOPES:
        return (
            f"Cannot write scope {scope!r}. Use 'user' for things true about "
            "this person across conversations, or 'session' for this "
            "conversation only."
        )
    return None


def remember(key: str, value: str, scope: str = "user") -> str:
    """Store a durable fact so it is available in later conversations.

    Use this for things that stay true — preferences, identifiers, goals,
    corrections the user made. Use ``scope='session'`` for working notes that
    only matter in this conversation.

    Args:
        key: short stable identifier, e.g. "preferred_language".
        value: the fact, written so it is useful without the surrounding chat.
        scope: "user" (default, persists across conversations) or "session".
    """
    memory, ctx = _resolve()
    if memory is None:
        return _NO_CONTEXT
    problem = _check_scope(scope)
    if problem:
        return problem
    # provenance points at the turn, so "why does the agent believe this?"
    # resolves to a real row rather than to ids that may not exist yet
    source_ref = json.dumps(
        {
            "conversation_id": ctx.conversation_id,
            "turn_id": ctx.turn_id,
            "message_id": ctx.message_id,
        }
    )
    memory.set_state(
        scope,
        _owner(ctx, scope),
        key,
        value,
        source_ref=source_ref,
        written_by=WRITTEN_BY_AGENT,
    )
    return f"Stored {key!r} in {scope} memory."


def recall(key: str, scope: str = "user") -> str:
    """Look up a fact stored earlier with ``remember``.

    Args:
        key: the identifier used when it was stored.
        scope: "user" (default) or "session".
    """
    memory, ctx = _resolve()
    if memory is None:
        return _NO_CONTEXT
    value = memory.get_state(scope, _owner(ctx, scope), key)
    if value is None:
        return f"Nothing stored under {key!r} in {scope} memory."
    return value


def forget(key: str, scope: str = "user") -> str:
    """Delete a fact stored earlier, e.g. when the user says it is wrong.

    Args:
        key: the identifier used when it was stored.
        scope: "user" (default) or "session".
    """
    memory, ctx = _resolve()
    if memory is None:
        return _NO_CONTEXT
    problem = _check_scope(scope)
    if problem:
        return problem
    removed = memory.delete_state(scope, _owner(ctx, scope), key)
    if not removed:
        return f"Nothing stored under {key!r} in {scope} memory."
    return f"Forgot {key!r}."


def search(query: str) -> str:
    """Search earlier conversations for anything relevant to a topic.

    Use this when the answer might depend on something discussed before that is
    no longer in the visible history. Searches only this user's own past.

    Args:
        query: what to look for, in natural language.
    """
    memory, ctx = _resolve()
    if memory is None:
        return _NO_CONTEXT
    hits = memory.search(query, subject=ctx.subject)
    if not hits:
        return f"Nothing found in earlier conversations about {query!r}."
    lines = []
    for hit in hits:
        when = (hit.get("created_at") or "")[:10]
        # timestamp and speaker are deliberately included: the model needs to
        # know how old a memory is to weigh it against the current turn
        lines.append(f"[{when} {hit.get('role', '?')}] {hit['content']}")
    return "\n".join(lines)


MEMORY_TOOLS = (remember, recall, forget, search)


def memory_tools(framework: str = "plain"):
    """The memory tools wrapped in a framework's tool type.

    One line to register, but still explicit — the SDK cannot reach into an
    arbitrary framework's tool list, and quietly appending tools to someone's
    agent would be worse than asking::

        agent = create_react_agent(llm, [*my_tools, *app.memory_tools("langgraph")])
    """
    if framework in ("plain", "custom", None):
        return list(MEMORY_TOOLS)
    if framework in ("langgraph", "langchain"):
        try:
            from langchain_core.tools import tool as lc_tool
        except ImportError as err:
            raise ImportError(
                "memory_tools('langgraph') requires langchain-core: "
                "pip install langchain-core"
            ) from err
        return [lc_tool(fn) for fn in MEMORY_TOOLS]
    if framework == "llamaindex":
        try:
            from llama_index.core.tools import FunctionTool
        except ImportError as err:
            raise ImportError(
                "memory_tools('llamaindex') requires llama-index-core: "
                "pip install llama-index-core"
            ) from err
        return [FunctionTool.from_defaults(fn=fn) for fn in MEMORY_TOOLS]
    raise ValueError(
        f"Unknown framework {framework!r}. Supported: 'langgraph', "
        "'llamaindex', 'plain' (bare functions to wrap yourself)."
    )

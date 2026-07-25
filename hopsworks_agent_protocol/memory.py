"""Conversation memory keyed by the protocol's ``conversation_id``.

The protocol makes history server-side: clients send only the new message
plus a ``conversation_id``. This module provides the storage for that —
pass a store to :class:`AgentApp` and turns are recorded automatically;
handlers read history with ``app.memory.get(request.conversation_id)``,
already in the ``{"role", "content"}`` shape LangChain/LangGraph/LlamaIndex
accept.

Backends:
- :class:`InMemoryChatMemory` — zero-config, for development. Conversations
  are lost on restart and not shared between replicas; note that Hopsworks
  agent deployments can scale to zero, so this is NOT for production.
- :class:`PersistentAgentMemory` — any SQLAlchemy URL (e.g. the project's MySQL).
  Survives restarts and works across replicas.

If your framework already persists conversation state (a LangGraph
checkpointer, a LlamaIndex chat store), key it by ``conversation_id`` and do
NOT pass a memory store — one source of truth for history is enough.
"""

from __future__ import annotations

import asyncio
import inspect
import logging
import os
import re
import threading
import time
import uuid
from abc import ABC, abstractmethod
from collections.abc import Callable
from datetime import datetime, timedelta, timezone
from typing import Awaitable, List, Optional, Union

log = logging.getLogger(__name__)

Turn = dict[str, str]  # {"role": "user"|"assistant", "content": str}

#: Rewrites the running summary to absorb the turns being folded away.
#: ``(previous_summary_or_None, turns_to_fold) -> summary``. Sync or async —
#: taking the previous summary is what makes the fold incremental, so a long
#: conversation never re-summarizes itself from the beginning.
Summarizer = Callable[
    [Optional[str], List[Turn]], Union[str, Awaitable[str]]
]

# A turn is open from the moment the user message is recorded until the handler
# finishes. Rows of an open turn are invisible to reads: the reply does not
# exist yet, and a half-written turn must never reach the model (or, later, a
# summarizer). Turns that never close — a pod killed mid-request, an ASGI server
# that skipped generator cleanup — are swept to ``abandoned`` by age.
TURN_OPEN = "open"
TURN_CLOSED = "closed"
TURN_ABANDONED = "abandoned"

# Only ``message`` rows are conversation history. tool_call/event rows are
# debug telemetry: same table, excluded from get(), pruned on a TTL later.
ITEM_MESSAGE = "message"
ITEM_TOOL_CALL = "tool_call"
ITEM_EVENT = "event"

# Durable state is scoped. ``user`` is the end user's own memory, shared across
# their conversations; ``session`` is working state for one conversation; ``app``
# is agent-wide and read by everyone, which is why the model may not write it —
# see WRITABLE_SCOPES.
SCOPE_USER = "user"
SCOPE_APP = "app"
SCOPE_SESSION = "session"
SCOPES = (SCOPE_USER, SCOPE_APP, SCOPE_SESSION)

#: Scopes an agent's own LLM may write. ``app`` is excluded on purpose: it is
#: injected into every user's turns, so a model-writable ``app`` scope would let
#: one user's prompt injection persist into everyone else's context. Operators
#: write it out of band.
WRITABLE_SCOPES = (SCOPE_USER, SCOPE_SESSION)

WRITTEN_BY_AGENT = "agent"
WRITTEN_BY_OPERATOR = "operator"

# Bumped whenever the table shape changes. ``create_all`` is create-if-not-
# exists, so it silently does nothing to a table an older version already made;
# the recorded version is what turns "SDK upgraded under a live deployment"
# into a defined outcome instead of a column-missing error at query time.
SCHEMA_VERSION = 1

_TABLE_SUFFIX_RE = re.compile(r"^[A-Za-z0-9_]{1,48}$")


def new_turn_id() -> str:
    return f"turn_{uuid.uuid4().hex}"


def _utcnow() -> datetime:
    # naive UTC: MySQL DATETIME carries no zone, and mixing aware/naive values
    # across drivers is a reliable source of comparison bugs
    return datetime.now(timezone.utc).replace(tzinfo=None)


def deployment_mysql_url() -> str:
    """Build the SQLAlchemy URL of the project MySQL from the env vars the
    platform injects into Hopsworks agent deployments (MYSQL_USER, MYSQL_HOST,
    MYSQL_PORT, MYSQL_DB, and the password via MYSQL_PASSWORD or the
    MYSQL_PASSWORD_SECRET_NAME Hopsworks secret)."""
    try:
        user = os.environ["MYSQL_USER"]
        host = os.environ["MYSQL_HOST"]
        db = os.environ["MYSQL_DB"]
    except KeyError as err:
        raise RuntimeError(
            f"MySQL env var {err.args[0]} is not set — not running in a "
            "Hopsworks agent deployment? Pass an explicit url= to "
            "PersistentAgentMemory instead."
        ) from err
    port = os.environ.get("MYSQL_PORT", "3306")

    password = os.environ.get("MYSQL_PASSWORD")
    if password is None:
        secret_name = os.environ.get("MYSQL_PASSWORD_SECRET_NAME")
        if secret_name is None:
            raise RuntimeError(
                "Neither MYSQL_PASSWORD nor MYSQL_PASSWORD_SECRET_NAME is set."
            )
        password = _read_hopsworks_secret(secret_name)

    return f"mysql+pymysql://{user}:{password}@{host}:{port}/{db}"


def _read_hopsworks_secret(secret_name: str) -> str:
    try:
        import hopsworks
    except ImportError as err:
        raise RuntimeError(
            "Reading the MySQL password secret requires the hopsworks "
            "package (available in Hopsworks deployment environments)."
        ) from err
    try:
        return hopsworks.get_secrets_api().get(secret_name)
    except Exception:  # noqa: BLE001 — not logged in yet
        hopsworks.login()
        return hopsworks.get_secrets_api().get(secret_name)


class ChatMemory(ABC):
    """Conversation store keyed by conversation_id.

    Writes go through the **turn lifecycle**: ``begin_turn`` records the user
    message before the handler runs, ``record_item`` adds the reply and any tool
    events, and ``end_turn`` either closes the turn or marks it abandoned.

    Recording the question up front is what lets a handler read back the message
    it is answering, and (from Phase 2) lets anything the agent remembers point
    at the turn that caused it. The cost is that a turn can now fail *after* its
    question is stored — so an open turn is invisible to :meth:`get` until it
    closes, and every exit path must end the turn. A store that skips that ends
    up serving questions whose answers never arrived.
    """

    @abstractmethod
    def get(self, conversation_id: str) -> list[Turn]:
        """Messages of closed turns, in chronological order."""

    @abstractmethod
    def begin_turn(
        self,
        conversation_id: str,
        turn_id: str,
        role: str,
        content: str,
        *,
        message_id: str | None = None,
        subject: str | None = None,
    ) -> None:
        """Open a turn and record its first message. Invisible to :meth:`get`
        until :meth:`end_turn` closes it."""

    @abstractmethod
    def record_item(
        self,
        conversation_id: str,
        turn_id: str,
        role: str,
        content: str,
        *,
        memory_type: str = ITEM_MESSAGE,
        seq: int | None = None,
        message_id: str | None = None,
        subject: str | None = None,
    ) -> None:
        """Add a row to an open turn (the reply, a tool call, an event)."""

    @abstractmethod
    def end_turn(
        self, conversation_id: str, turn_id: str, *, status: str = TURN_CLOSED
    ) -> None:
        """Close a turn (``closed``) or mark it failed (``abandoned``)."""

    @abstractmethod
    def clear(self, conversation_id: str) -> None:
        """Drop a conversation."""

    def append(self, conversation_id: str, role: str, content: str) -> None:
        """Record one standalone message, e.g. to seed a conversation.

        A convenience over the lifecycle, not a second write path: it opens a
        turn, writes the message, and closes it immediately.
        """
        turn_id = new_turn_id()
        self.begin_turn(conversation_id, turn_id, role, content)
        self.end_turn(conversation_id, turn_id)

    def healthcheck(self) -> bool:
        """Whether the store is currently usable. Used by the readiness probe;
        default assumes always ready."""
        return True

    def maybe_reap(self) -> None:
        """Sweep turns that were never closed. No-op for stores whose open
        turns cannot outlive the process."""

    def transcript(
        self, conversation_id: str, *, include_events: bool = False
    ) -> list[dict]:
        """The human-facing record of a conversation.

        Deliberately not the same thing as :meth:`get`. ``get`` returns what the
        model reads — bounded, and shrinking as turns are folded into the
        summary. This returns the whole conversation, including turns that have
        been folded away, because it is what a person reads in the panel and
        what an operator reads during an incident.
        """
        return [dict(turn, memory_type=ITEM_MESSAGE) for turn in self.get(conversation_id)]

    def summarized_through(self, conversation_id: str) -> int:
        """Highest item id represented by the summary rather than by
        :meth:`get`. ``0`` when nothing has been folded — lets a UI draw the
        line between "the agent still sees this" and "this is only in the
        summary now"."""
        return 0

    # ── summarization (tier 2) ───────────────────────────────────────────

    def get_summary(self, conversation_id: str) -> str | None:
        """The rolling summary of folded-away turns, or None."""
        return None

    async def maybe_summarize(self, conversation_id: str) -> bool:
        """Fold old turns into the summary if enough have accumulated.

        Called after the response has been sent, so its cost lands on request
        duration every Nth turn rather than on time-to-answer. Returns whether
        a fold happened.
        """
        return False

    # ── scoped durable state (tier 3) ────────────────────────────────────

    def set_state(
        self,
        scope: str,
        owner: str,
        key: str,
        value: str,
        *,
        source_ref: str | None = None,
        written_by: str = WRITTEN_BY_AGENT,
    ) -> None:
        """Upsert one scoped durable value."""

    def get_state(self, scope: str, owner: str, key: str) -> str | None:
        return None

    def list_state(
        self,
        scope: str,
        owner: str,
        *,
        key: str | None = None,
        limit: int | None = None,
    ) -> list[dict]:
        return []

    def delete_state(self, scope: str, owner: str, key: str | None = None) -> int:
        return 0

    def state_block(self, subject: str, conversation_id: str) -> str:
        """Scoped state rendered for ``ctx.system_context()``."""
        return ""


class InMemoryChatMemory(ChatMemory):
    """Process-local store for development: lost on restart (agent
    deployments can scale to zero) and not shared between replicas."""

    def __init__(self, max_messages: int = 50):
        self._max = max_messages
        self._conversations: dict[str, list[Turn]] = {}
        # open turns buffer their messages here; they land in the conversation
        # on close and are dropped on abandon, so history never shows a
        # question whose answer never arrived
        self._open: dict[str, list[Turn]] = {}
        self._lock = threading.Lock()

    def get(self, conversation_id: str) -> list[Turn]:
        with self._lock:
            return list(self._conversations.get(conversation_id, []))

    def begin_turn(
        self,
        conversation_id: str,
        turn_id: str,
        role: str,
        content: str,
        *,
        message_id: str | None = None,
        subject: str | None = None,
    ) -> None:
        with self._lock:
            self._open[turn_id] = [{"role": role, "content": content}]

    def record_item(
        self,
        conversation_id: str,
        turn_id: str,
        role: str,
        content: str,
        *,
        memory_type: str = ITEM_MESSAGE,
        seq: int | None = None,
        message_id: str | None = None,
        subject: str | None = None,
    ) -> None:
        if memory_type != ITEM_MESSAGE:
            # the dev store keeps history only; tool/event rows are debug
            # telemetry and are dropped rather than half-modelled
            return
        with self._lock:
            buffered = self._open.get(turn_id)
            if buffered is not None:
                buffered.append({"role": role, "content": content})

    def end_turn(
        self, conversation_id: str, turn_id: str, *, status: str = TURN_CLOSED
    ) -> None:
        with self._lock:
            buffered = self._open.pop(turn_id, None)
            if not buffered or status != TURN_CLOSED:
                return
            turns = self._conversations.setdefault(conversation_id, [])
            turns.extend(buffered)
            if len(turns) > self._max:
                del turns[: len(turns) - self._max]

    def clear(self, conversation_id: str) -> None:
        with self._lock:
            self._conversations.pop(conversation_id, None)


class PersistentAgentMemory(ChatMemory):
    """SQLAlchemy-backed store (MySQL, Postgres, SQLite, ...).

    ``pip install 'hopsworks-agent-protocol[memory-sql]'``

    Inside a Hopsworks agent deployment both arguments are optional:
    ``PersistentAgentMemory()`` connects to the project MySQL using the
    platform-injected env vars and derives a per-deployment table name from
    ``DEPLOYMENT_ID``.

    A per-conversation cache avoids a read round-trip on every turn for the
    lifetime of the process.

    Rows live in ``agent_memory_items_<DEPLOYMENT_ID>`` and carry turn identity
    (``turn_id``, ``seq``, ``status``, ``message_id``) so the SDK can record the
    user message before the handler runs without ever exposing a half-written
    turn: only ``closed`` turns are returned by :meth:`get`. See
    :meth:`begin_turn`.

    Connection is **lazy and non-fatal**: construction never touches the
    database (missing SQLAlchemy is the only hard error), the engine + tables
    are created on first use — in a deployment that happens via the ``/ready``
    probe, off the request path — and if the database is unreachable the store
    degrades to statelessness — reads return empty, writes are dropped, both
    with a warning — so a down database never crashes the agent at startup or
    on a turn.
    """

    def __init__(
        self,
        url: str | None = None,
        table_name: str | None = None,
        max_messages: int = 50,
        turn_timeout_seconds: int = 300,
        summarize: Summarizer | None = None,
        summarize_after_messages: int = 20,
        keep_recent_messages: int = 10,
        tool_event_retention_days: int | None = 30,
        message_retention_days: int | None = None,
        long_term: bool = False,
        state_ttl_days: int | None = 90,
        max_state_value_chars: int = 4096,
        max_state_keys_written: int = 128,
        max_state_keys_injected: int = 32,
        state_inject_value_chars: int = 1024,
    ):
        try:
            import sqlalchemy  # noqa: F401 — fail fast if the extra is missing
        except ImportError as err:
            raise ImportError(
                "PersistentAgentMemory requires SQLAlchemy: "
                "pip install 'hopsworks-agent-protocol[memory-sql]'"
            ) from err

        self._max = max_messages
        self._turn_timeout = turn_timeout_seconds
        self._summarize = summarize
        self._summarize_every = summarize_after_messages
        self._keep_recent = keep_recent_messages
        self._tool_event_retention = tool_event_retention_days
        self._message_retention = message_retention_days
        self._long_term = long_term
        self._state_ttl = state_ttl_days
        self._max_value_chars = max_state_value_chars
        self._max_keys_written = max_state_keys_written
        self._max_keys_injected = max_state_keys_injected
        self._inject_value_chars = state_inject_value_chars
        self._url = url  # resolved lazily (may read env/secrets)
        self._table_name = table_name
        # conversation_id -> (summary_version, turns). The version is what makes
        # the cache safe once folding exists: another replica advancing the
        # cutoff bumps it, and this process notices on the next read instead of
        # serving history that has since been compacted away.
        self._cache: dict[str, tuple[int, list[Turn]]] = {}
        # rows written by turns that are still open, held back from the cache
        # until the turn closes (or dropped when it is abandoned)
        self._pending: dict[str, list[Turn]] = {}
        self._seq: dict[str, int] = {}
        self._lock = threading.Lock()
        # set on first successful connection; None while unconnected/failed
        self._engine = None
        self._table = None
        self._meta = None
        self._sessions = None
        self._state = None
        self._init_failed = False
        self._last_reap = 0.0
        self._last_prune = 0.0

    def _resolve_table_name(self) -> str:
        """Per-deployment table name, validated.

        The suffix ends up in DDL. SQLAlchemy quotes identifiers so this is not
        an injection vector, but an unvalidated ``DEPLOYMENT_ID`` still produces
        junk table names — and the old ``'default'`` fallback would silently
        make two deployments share one conversation history. So: an explicit
        ``table_name`` wins, a deployment must have ``DEPLOYMENT_ID``, and only
        a caller who passed their own ``url`` (tests, local dev) gets the
        fallback.
        """
        if self._table_name is not None:
            name = self._table_name
        else:
            deployment = os.environ.get("DEPLOYMENT_ID")
            if deployment is None:
                if self._url is None:
                    raise RuntimeError(
                        "DEPLOYMENT_ID is not set, so the per-deployment memory "
                        "table name cannot be derived. Inside a Hopsworks agent "
                        "deployment this is injected for you; elsewhere pass an "
                        "explicit table_name= (or url=) to PersistentAgentMemory."
                    )
                log.warning(
                    "DEPLOYMENT_ID is not set; falling back to the shared table "
                    "suffix 'default'. Pass table_name= to keep deployments apart."
                )
                deployment = "default"
            name = f"agent_memory_items_{deployment}"

        suffix = name[len("agent_memory_items_"):] if name.startswith(
            "agent_memory_items_"
        ) else name
        if not _TABLE_SUFFIX_RE.match(suffix):
            raise RuntimeError(
                f"Invalid memory table name {name!r}: the deployment suffix must "
                "match [A-Za-z0-9_]{1,48}."
            )
        return name

    def _ensure_engine(self) -> bool:
        """Create the engine + tables. Returns True when the store is usable,
        False when the database is unreachable (already logged).

        Called by ``healthcheck()``, which the ``/ready`` probe drives — so in a
        deployment the DDL runs *before* the pod reports ready and the request
        path only ever finds existing tables. The lazy path stays for direct use
        (tests, local SQLite) where there is no readiness probe.
        """
        if self._engine is not None:
            return True
        if self._init_failed:
            return False
        with self._lock:
            if self._engine is not None:
                return True
            if self._init_failed:
                return False
            try:
                from sqlalchemy import (
                    Column,
                    DateTime,
                    Index,
                    Integer,
                    MetaData,
                    SmallInteger,
                    String,
                    Table,
                    Text,
                    UniqueConstraint,
                    create_engine,
                )

                url = self._url or deployment_mysql_url()
                table_name = self._resolve_table_name()
                suffix = table_name[len("agent_memory_items_"):] if (
                    table_name.startswith("agent_memory_items_")
                ) else table_name
                engine = create_engine(url, pool_pre_ping=True)
                metadata = MetaData()
                table = Table(
                    table_name,
                    metadata,
                    Column("id", Integer, primary_key=True, autoincrement=True),
                    Column("conversation_id", String(255), nullable=False),
                    Column("turn_id", String(64), nullable=False),
                    # the protocol message id, so anything written during the
                    # turn can reference the message that caused it
                    Column("message_id", String(64), nullable=True),
                    Column("seq", SmallInteger, nullable=False, default=0),
                    Column("status", String(16), nullable=False),
                    Column("subject", String(255), nullable=True),
                    Column("memory_type", String(16), nullable=False),
                    Column("role", String(16), nullable=False),
                    Column("content", Text, nullable=False),
                    Column("created_at", DateTime, nullable=False),
                    Column("expires_at", DateTime, nullable=True),
                    Index(f"idx_{suffix}_conv", "conversation_id", "id"),
                    Index(f"idx_{suffix}_turn", "conversation_id", "turn_id"),
                    # drives the "oldest open turn" lookup on the fold path
                    Index(f"idx_{suffix}_open", "conversation_id", "status", "id"),
                    Index(f"idx_{suffix}_msg", "message_id"),
                )
                meta = Table(
                    f"agent_memory_meta_{suffix}",
                    metadata,
                    Column("table_name", String(64), primary_key=True),
                    Column("schema_version", Integer, nullable=False),
                    Column("updated_at", DateTime, nullable=False),
                )
                sessions = None
                if self._summarize is not None:
                    # tier 2 only: no summarizer, no bookkeeping to keep
                    sessions = Table(
                        f"agent_memory_sessions_{suffix}",
                        metadata,
                        Column("conversation_id", String(255), primary_key=True),
                        Column("subject", String(255), nullable=True),
                        Column("summary", Text, nullable=True),
                        # fold cutoff. Meaningful only together with the
                        # conversation: item ids are a table-wide sequence.
                        Column(
                            "last_summarized_item_id",
                            Integer,
                            nullable=False,
                            default=0,
                        ),
                        # messages folded so far — the fold threshold is
                        # measured in messages, not row ids
                        Column(
                            "last_summarized_count", Integer, nullable=False, default=0
                        ),
                        # optimistic lock: the summarizer runs outside the
                        # transaction, so this is what makes the fold safe
                        Column("summary_version", Integer, nullable=False, default=0),
                        Column("message_count", Integer, nullable=False, default=0),
                        Column("created_at", DateTime, nullable=False),
                        Column("last_active_at", DateTime, nullable=False),
                    )
                state = None
                if self._long_term:
                    # tier 3 only
                    state = Table(
                        f"agent_memory_state_{suffix}",
                        metadata,
                        Column("id", Integer, primary_key=True, autoincrement=True),
                        Column("scope", String(8), nullable=False),
                        # subject (user), '' (app), or conversation id (session)
                        Column("owner", String(255), nullable=False),
                        # not `key`: reserved word in MySQL 8
                        Column("state_key", String(255), nullable=False),
                        Column("value", Text, nullable=False),
                        # {conversation_id, turn_id, message_id} — resolves to
                        # the turn that produced this value
                        Column("source_ref", Text, nullable=True),
                        # agent | operator. The audit surface shows this: a user
                        # looking at their own memories needs to know which of
                        # them their own conversation put there.
                        Column("written_by", String(16), nullable=False),
                        Column("updated_at", DateTime, nullable=False),
                        # agent-written state decays; operator-written does not
                        Column("expires_at", DateTime, nullable=True),
                        UniqueConstraint(
                            "scope",
                            "owner",
                            "state_key",
                            name=f"uq_{suffix}_state",
                        ),
                        Index(f"idx_{suffix}_state_owner", "scope", "owner"),
                    )
                metadata.create_all(engine)
                if not self._check_schema_version(engine, meta, table_name):
                    self._init_failed = True
                    return False
                self._engine = engine
                self._table = table
                self._meta = meta
                self._sessions = sessions
                self._state = state
                log.info(
                    "PersistentAgentMemory ready (table=%s, schema_version=%d)",
                    table_name,
                    SCHEMA_VERSION,
                )
                return True
            except Exception:  # noqa: BLE001 — degrade to stateless, never crash
                self._init_failed = True
                log.exception(
                    "PersistentAgentMemory could not connect to the database; the "
                    "agent will run without persistent memory"
                )
                return False

    def _check_schema_version(self, engine, meta, table_name: str) -> bool:
        """Record our schema version, or refuse to run against a newer one.

        ``create_all`` is create-if-not-exists: against a table an older SDK
        already created it does nothing at all, so a column added in a later
        version would be missing and only surface as a query error much later.
        Recording the version makes that case explicit. Migrating *forward* is a
        no-op today (v1 is the first shape); the branch that matters now is
        refusing to write through an older SDK to a newer table, where guessing
        would corrupt data.
        """
        with engine.begin() as conn:
            row = conn.execute(
                meta.select().where(meta.c.table_name == table_name)
            ).fetchone()
            if row is None:
                conn.execute(
                    meta.insert().values(
                        table_name=table_name,
                        schema_version=SCHEMA_VERSION,
                        updated_at=_utcnow(),
                    )
                )
                return True
            if row.schema_version > SCHEMA_VERSION:
                log.error(
                    "Memory table %s is at schema version %d but this SDK "
                    "understands %d. Refusing to use it — upgrade "
                    "hopsworks-agent-protocol. The agent will run without "
                    "persistent memory.",
                    table_name,
                    row.schema_version,
                    SCHEMA_VERSION,
                )
                return False
            if row.schema_version < SCHEMA_VERSION:
                # no forward migrations exist yet; when one does, apply the
                # ordered ALTERs for (row.schema_version, SCHEMA_VERSION] here
                conn.execute(
                    meta.update()
                    .where(meta.c.table_name == table_name)
                    .values(schema_version=SCHEMA_VERSION, updated_at=_utcnow())
                )
            return True

    def _fold_state(self, conversation_id: str) -> tuple[int, int]:
        """``(summary_version, last_summarized_item_id)`` for a conversation.

        ``(0, 0)`` without a summarizer — nothing folds, so there is no session
        row to read and no reason to pay for one.
        """
        if self._summarize is None or self._sessions is None:
            return (0, 0)
        s = self._sessions
        try:
            with self._engine.connect() as conn:
                row = conn.execute(
                    s.select().where(s.c.conversation_id == conversation_id)
                ).fetchone()
        except Exception:  # noqa: BLE001
            log.exception("Session read failed; treating conversation as unfolded")
            return (0, 0)
        if row is None:
            return (0, 0)
        return (row.summary_version, row.last_summarized_item_id)

    def get(self, conversation_id: str) -> list[Turn]:
        if not self._ensure_engine():
            return []
        version, cutoff = self._fold_state(conversation_id)
        with self._lock:
            cached = self._cache.get(conversation_id)
            if cached is not None and cached[0] == version:
                return list(cached[1])
        t = self._table
        try:
            with self._engine.connect() as conn:
                rows = conn.execute(
                    t.select()
                    .where(t.c.conversation_id == conversation_id)
                    # closed turns only: an open turn has no reply yet, and an
                    # abandoned one is a question we failed to answer — neither
                    # belongs in what the model reads back
                    .where(t.c.status == TURN_CLOSED)
                    .where(t.c.memory_type == ITEM_MESSAGE)
                    # everything at or below the cutoff is represented by the
                    # summary. Without this the model would get the summary
                    # *and* the turns it was built from: more tokens than
                    # before, and the same content twice.
                    .where(t.c.id > cutoff)
                    .order_by(t.c.id.desc())
                    .limit(self._max)
                ).fetchall()
        except Exception:  # noqa: BLE001
            log.exception("Memory read failed; returning empty history")
            return []
        turns = [{"role": r.role, "content": r.content} for r in reversed(rows)]
        with self._lock:
            self._cache[conversation_id] = (version, list(turns))
        return turns

    def clear(self, conversation_id: str) -> None:
        if self._ensure_engine():
            try:
                with self._engine.begin() as conn:
                    conn.execute(
                        self._table.delete().where(
                            self._table.c.conversation_id == conversation_id
                        )
                    )
            except Exception:  # noqa: BLE001
                log.exception("Memory clear failed")
        with self._lock:
            self._cache.pop(conversation_id, None)

    def healthcheck(self) -> bool:
        # ready once the engine + tables are established and reachable; this is
        # also where a deployment's DDL runs, off the request path
        return self._ensure_engine()

    # ── turn lifecycle ────────────────────────────────────────────────────

    def _expiry_for(self, memory_type: str, now: datetime) -> datetime | None:
        """When a row becomes prunable.

        ``message`` rows keep forever by default: they are the transcript — what
        a user reads in the panel and an operator reads during an incident — and
        they are the small half of the table. Deleting them silently on a timer
        would make history lossier than not compacting at all, so it is an
        explicit operator policy instead. tool_call/event rows are debug
        telemetry and the bulk of the volume, so they do expire.
        """
        days = (
            self._message_retention
            if memory_type == ITEM_MESSAGE
            else self._tool_event_retention
        )
        return None if days is None else now + timedelta(days=days)

    def _insert_item(
        self,
        conversation_id: str,
        turn_id: str,
        role: str,
        content: str,
        *,
        status: str,
        seq: int,
        memory_type: str = ITEM_MESSAGE,
        message_id: str | None = None,
        subject: str | None = None,
    ) -> bool:
        if not self._ensure_engine():
            return False
        now = _utcnow()
        try:
            with self._engine.begin() as conn:
                conn.execute(
                    self._table.insert().values(
                        conversation_id=conversation_id,
                        turn_id=turn_id,
                        message_id=message_id,
                        seq=seq,
                        status=status,
                        subject=subject,
                        memory_type=memory_type,
                        role=role,
                        content=content,
                        created_at=now,
                        expires_at=self._expiry_for(memory_type, now),
                    )
                )
            return True
        except Exception:  # noqa: BLE001 — memory must never break a turn
            log.exception("Memory write failed; item not persisted")
            return False

    def begin_turn(
        self,
        conversation_id: str,
        turn_id: str,
        role: str,
        content: str,
        *,
        message_id: str | None = None,
        subject: str | None = None,
    ) -> None:
        with self._lock:
            self._pending[turn_id] = []
            self._seq[turn_id] = 0
        ok = self._insert_item(
            conversation_id,
            turn_id,
            role,
            content,
            status=TURN_OPEN,
            seq=0,
            message_id=message_id,
            subject=subject,
        )
        if not ok:
            with self._lock:
                self._pending.pop(turn_id, None)
                self._seq.pop(turn_id, None)
            return
        with self._lock:
            self._seq[turn_id] = 1
            self._pending[turn_id].append({"role": role, "content": content})

    def record_item(
        self,
        conversation_id: str,
        turn_id: str,
        role: str,
        content: str,
        *,
        memory_type: str = ITEM_MESSAGE,
        seq: int | None = None,
        message_id: str | None = None,
        subject: str | None = None,
    ) -> None:
        with self._lock:
            if seq is None:
                seq = self._seq.get(turn_id, 0)
                self._seq[turn_id] = seq + 1
        ok = self._insert_item(
            conversation_id,
            turn_id,
            role,
            content,
            status=TURN_OPEN,
            seq=seq,
            memory_type=memory_type,
            message_id=message_id,
            subject=subject,
        )
        if ok and memory_type == ITEM_MESSAGE:
            with self._lock:
                if turn_id in self._pending:
                    self._pending[turn_id].append(
                        {"role": role, "content": content}
                    )

    def end_turn(
        self, conversation_id: str, turn_id: str, *, status: str = TURN_CLOSED
    ) -> None:
        with self._lock:
            pending = self._pending.pop(turn_id, [])
            self._seq.pop(turn_id, None)
        if not self._ensure_engine():
            return
        t = self._table
        try:
            with self._engine.begin() as conn:
                conn.execute(
                    t.update()
                    .where(t.c.conversation_id == conversation_id)
                    .where(t.c.turn_id == turn_id)
                    .where(t.c.status == TURN_OPEN)
                    .values(status=status)
                )
        except Exception:  # noqa: BLE001
            log.exception("Memory end_turn failed; turn left open")
            return
        if status == TURN_CLOSED and pending:
            self._bump_message_count(conversation_id, len(pending))
            # the turn's rows just became visible — extend the cache rather
            # than dropping it, so closing a turn costs no extra read. If a
            # fold lands concurrently the version moves on and the next read
            # rebuilds; nothing here has to know about that.
            with self._lock:
                cached = self._cache.get(conversation_id)
                if cached is not None:
                    version, turns = cached
                    turns.extend(pending)
                    if len(turns) > self._max:
                        del turns[: len(turns) - self._max]
        elif status != TURN_CLOSED:
            with self._lock:
                self._cache.pop(conversation_id, None)

    def _bump_message_count(self, conversation_id: str, added: int) -> None:
        """Track how many messages the conversation holds, for the fold trigger.

        Incremented in SQL rather than read-modify-write so overlapping turns
        cannot lose an increment and stall summarization.
        """
        if self._sessions is None or added <= 0:
            return
        s = self._sessions
        now = _utcnow()
        try:
            with self._engine.begin() as conn:
                result = conn.execute(
                    s.update()
                    .where(s.c.conversation_id == conversation_id)
                    .values(
                        message_count=s.c.message_count + added,
                        last_active_at=now,
                    )
                )
                if result.rowcount:
                    return
            with self._engine.begin() as conn:
                conn.execute(
                    s.insert().values(
                        conversation_id=conversation_id,
                        message_count=added,
                        last_summarized_item_id=0,
                        last_summarized_count=0,
                        summary_version=0,
                        created_at=now,
                        last_active_at=now,
                    )
                )
        except Exception:  # noqa: BLE001 — another turn inserted first, or the
            # DB is unhappy; retry the update once and give up quietly
            try:
                with self._engine.begin() as conn:
                    conn.execute(
                        s.update()
                        .where(s.c.conversation_id == conversation_id)
                        .values(
                            message_count=s.c.message_count + added,
                            last_active_at=now,
                        )
                    )
            except Exception:  # noqa: BLE001
                log.exception("Could not update session message count")

    def reap_open_turns(self, older_than_seconds: int | None = None) -> int:
        """Mark long-open turns as abandoned.

        A turn is closed in a ``finally``, but that is not a guarantee: a pod
        can be killed mid-request, and an ASGI server may not run an async
        generator's cleanup promptly (or at all) when a client disconnects.
        Without a sweep those rows stay ``open`` forever — invisible to reads,
        and (once summarization lands) permanently blocking the fold cutoff,
        which is the failure that actually bites.
        """
        if not self._ensure_engine():
            return 0
        timeout = (
            self._turn_timeout if older_than_seconds is None else older_than_seconds
        )
        cutoff = _utcnow() - timedelta(seconds=timeout)
        t = self._table
        try:
            with self._engine.begin() as conn:
                result = conn.execute(
                    t.update()
                    .where(t.c.status == TURN_OPEN)
                    .where(t.c.created_at < cutoff)
                    .values(status=TURN_ABANDONED)
                )
            count = result.rowcount or 0
        except Exception:  # noqa: BLE001
            log.exception("Reaping open turns failed")
            return 0
        if count:
            log.warning("Marked %d stale open turn row(s) as abandoned", count)
        return count

    def maybe_reap(self) -> None:
        """Throttled ``reap_open_turns`` + prune for the post-response slot."""
        now = time.monotonic()
        with self._lock:
            due_reap = now - self._last_reap >= max(self._turn_timeout, 60)
            if due_reap:
                self._last_reap = now
            due_prune = now - self._last_prune >= 300
            if due_prune:
                self._last_prune = now
        if due_reap:
            self.reap_open_turns()
        if due_prune:
            self.prune_expired()

    def prune_expired(self, limit: int = 500) -> int:
        """Delete rows past ``expires_at``, a bounded batch at a time.

        Bounded because this runs inline after a response — amortized across
        turns rather than a table scan someone waits on. Selecting the ids
        first and deleting by id keeps it portable: MySQL will not accept a
        subquery against the table being deleted from.
        """
        if not self._ensure_engine():
            return 0
        t = self._table
        now = _utcnow()
        try:
            with self._engine.connect() as conn:
                ids = [
                    r.id
                    for r in conn.execute(
                        t.select()
                        .with_only_columns(t.c.id)
                        .where(t.c.expires_at.isnot(None))
                        .where(t.c.expires_at < now)
                        .limit(limit)
                    ).fetchall()
                ]
            if not ids:
                return 0
            with self._engine.begin() as conn:
                conn.execute(t.delete().where(t.c.id.in_(ids)))
        except Exception:  # noqa: BLE001
            log.exception("Pruning expired memory rows failed")
            return 0
        log.info("Pruned %d expired memory row(s)", len(ids))
        return len(ids)

    def transcript(
        self, conversation_id: str, *, include_events: bool = False
    ) -> list[dict]:
        if not self._ensure_engine():
            return []
        t = self._table
        stmt = t.select().where(t.c.conversation_id == conversation_id)
        if not include_events:
            # the default is a clean transcript: messages of turns that
            # completed. Folded rows stay — they are still what was said.
            stmt = stmt.where(t.c.memory_type == ITEM_MESSAGE).where(
                t.c.status == TURN_CLOSED
            )
        try:
            with self._engine.connect() as conn:
                rows = conn.execute(stmt.order_by(t.c.id)).fetchall()
        except Exception:  # noqa: BLE001
            log.exception("Transcript read failed")
            return []
        return [
            {
                "id": r.id,
                "turn_id": r.turn_id,
                "message_id": r.message_id,
                "role": r.role,
                "content": r.content,
                "memory_type": r.memory_type,
                "status": r.status,
                "created_at": r.created_at.isoformat() if r.created_at else None,
            }
            for r in rows
        ]

    def summarized_through(self, conversation_id: str) -> int:
        return self._fold_state(conversation_id)[1] if self._ensure_engine() else 0

    # ── summarization (tier 2) ───────────────────────────────────────────

    def get_summary(self, conversation_id: str) -> str | None:
        if self._summarize is None or not self._ensure_engine():
            return None
        s = self._sessions
        if s is None:
            return None
        try:
            with self._engine.connect() as conn:
                row = conn.execute(
                    s.select().where(s.c.conversation_id == conversation_id)
                ).fetchone()
        except Exception:  # noqa: BLE001
            log.exception("Summary read failed")
            return None
        return row.summary if row is not None else None

    def _claim_fold(self, conversation_id: str):
        """Phase 1: decide what to fold and lock the decision in.

        Returns ``(version, old_summary, turns, max_id, count)`` or ``None``
        when there is nothing to do. Deliberately short: it commits before the
        summarizer is called.
        """
        from sqlalchemy import func

        s, t = self._sessions, self._table
        with self._engine.begin() as conn:
            row = conn.execute(
                s.select()
                .where(s.c.conversation_id == conversation_id)
                .with_for_update()
            ).fetchone()
            if row is None:
                return None
            if row.message_count - row.last_summarized_count < self._summarize_every:
                return None

            # Never fold across a turn that is still open. Its reply does not
            # exist yet, so folding would compact the question into the summary
            # and leave the answer dangling in the visible window.
            open_min = conn.execute(
                t.select()
                .with_only_columns(func.min(t.c.id))
                .where(t.c.conversation_id == conversation_id)
                .where(t.c.status == TURN_OPEN)
            ).scalar()

            stmt = (
                t.select()
                .where(t.c.conversation_id == conversation_id)
                .where(t.c.status == TURN_CLOSED)
                .where(t.c.memory_type == ITEM_MESSAGE)
                .where(t.c.id > row.last_summarized_item_id)
            )
            if open_min is not None:
                stmt = stmt.where(t.c.id < open_min)
            items = conn.execute(stmt.order_by(t.c.id)).fetchall()

        # Keep a verbatim tail. Folding everything unfolded would leave the
        # turn right after a fold with a summary and no recent messages at all,
        # which inverts the tiering: the buffer is supposed to own recency
        # ("what did I just say") and the summary the older context.
        if self._keep_recent:
            items = items[: -self._keep_recent]
        if not items:
            return None
        turns = [{"role": r.role, "content": r.content} for r in items]
        return (row.summary_version, row.summary, turns, items[-1].id, len(items))

    def _commit_fold(
        self,
        conversation_id: str,
        version: int,
        summary: str,
        max_id: int,
        count: int,
    ) -> bool:
        """Phase 3: publish the summary, if nobody else folded meanwhile."""
        s = self._sessions
        with self._engine.begin() as conn:
            result = conn.execute(
                s.update()
                .where(s.c.conversation_id == conversation_id)
                .where(s.c.summary_version == version)
                .values(
                    summary=summary,
                    last_summarized_item_id=max_id,
                    last_summarized_count=s.c.last_summarized_count + count,
                    summary_version=version + 1,
                    last_active_at=_utcnow(),
                )
            )
        return bool(result.rowcount)

    # ── scoped durable state (tier 3) ────────────────────────────────────

    def set_state(
        self,
        scope: str,
        owner: str,
        key: str,
        value: str,
        *,
        source_ref: str | None = None,
        written_by: str = WRITTEN_BY_AGENT,
    ) -> None:
        """Upsert one scoped value.

        Agent-written values are bounded and decay. That is not tidiness: this
        state is auto-injected into every future turn for its owner, so a value
        the model was talked into writing outlives the conversation that
        produced it. Caps bound how much a single conversation can plant, and
        the TTL means a bad one fades instead of persisting forever. Neither
        prevents a false memory — the audit surface (``list_state`` /
        ``delete_state``, exposed to the subject) is what does.
        """
        if scope not in SCOPES:
            raise ValueError(f"unknown scope {scope!r}; expected one of {SCOPES}")
        if not self._ensure_engine() or self._state is None:
            return
        value = value[: self._max_value_chars]
        now = _utcnow()
        expires_at = (
            now + timedelta(days=self._state_ttl)
            if written_by == WRITTEN_BY_AGENT and self._state_ttl is not None
            else None
        )
        st = self._state
        values = dict(
            value=value,
            source_ref=source_ref,
            written_by=written_by,
            updated_at=now,
            expires_at=expires_at,
        )
        try:
            with self._engine.begin() as conn:
                result = conn.execute(
                    st.update()
                    .where(st.c.scope == scope)
                    .where(st.c.owner == owner)
                    .where(st.c.state_key == key)
                    .values(**values)
                )
                if not result.rowcount:
                    conn.execute(
                        st.insert().values(
                            scope=scope, owner=owner, state_key=key, **values
                        )
                    )
        except Exception:  # noqa: BLE001 — a lost race on the unique key, or a
            # sick database; retry as an update and give up quietly
            try:
                with self._engine.begin() as conn:
                    conn.execute(
                        st.update()
                        .where(st.c.scope == scope)
                        .where(st.c.owner == owner)
                        .where(st.c.state_key == key)
                        .values(**values)
                    )
            except Exception:  # noqa: BLE001
                log.exception("Could not write %s-scoped state %r", scope, key)
                return
        if written_by == WRITTEN_BY_AGENT:
            self._evict_excess_state(scope, owner)

    def _evict_excess_state(self, scope: str, owner: str) -> None:
        """Keep at most ``max_state_keys_written`` agent-written keys per owner,
        dropping the least recently updated."""
        st = self._state
        try:
            with self._engine.connect() as conn:
                rows = conn.execute(
                    st.select()
                    .with_only_columns(st.c.id)
                    .where(st.c.scope == scope)
                    .where(st.c.owner == owner)
                    .where(st.c.written_by == WRITTEN_BY_AGENT)
                    .order_by(st.c.updated_at.desc())
                ).fetchall()
            excess = [r.id for r in rows[self._max_keys_written :]]
            if not excess:
                return
            with self._engine.begin() as conn:
                conn.execute(st.delete().where(st.c.id.in_(excess)))
            log.info(
                "Evicted %d agent-written state key(s) for %s/%s over the cap",
                len(excess),
                scope,
                owner,
            )
        except Exception:  # noqa: BLE001
            log.exception("State eviction failed")

    def get_state(self, scope: str, owner: str, key: str) -> str | None:
        rows = self.list_state(scope, owner, key=key)
        return rows[0]["value"] if rows else None

    def list_state(
        self,
        scope: str,
        owner: str,
        *,
        key: str | None = None,
        limit: int | None = None,
    ) -> list[dict]:
        """Unexpired values for an owner, most recently updated first.

        Expiry is applied on read, not only by the pruner: a decayed value must
        stop being injected on its TTL rather than whenever a prune pass next
        happens to run.
        """
        if not self._ensure_engine() or self._state is None:
            return []
        st = self._state
        now = _utcnow()
        stmt = (
            st.select()
            .where(st.c.scope == scope)
            .where(st.c.owner == owner)
            .where((st.c.expires_at.is_(None)) | (st.c.expires_at > now))
        )
        if key is not None:
            stmt = stmt.where(st.c.state_key == key)
        stmt = stmt.order_by(st.c.updated_at.desc())
        if limit is not None:
            stmt = stmt.limit(limit)
        try:
            with self._engine.connect() as conn:
                rows = conn.execute(stmt).fetchall()
        except Exception:  # noqa: BLE001
            log.exception("State read failed")
            return []
        return [
            {
                "key": r.state_key,
                "value": r.value,
                "scope": r.scope,
                "owner": r.owner,
                "written_by": r.written_by,
                "source_ref": r.source_ref,
                "updated_at": r.updated_at.isoformat() if r.updated_at else None,
                "expires_at": r.expires_at.isoformat() if r.expires_at else None,
            }
            for r in rows
        ]

    def delete_state(self, scope: str, owner: str, key: str | None = None) -> int:
        """Forget one value, or every value for an owner. Returns rows removed."""
        if not self._ensure_engine() or self._state is None:
            return 0
        st = self._state
        stmt = st.delete().where(st.c.scope == scope).where(st.c.owner == owner)
        if key is not None:
            stmt = stmt.where(st.c.state_key == key)
        try:
            with self._engine.begin() as conn:
                return conn.execute(stmt).rowcount or 0
        except Exception:  # noqa: BLE001
            log.exception("State delete failed")
            return 0

    def state_block(self, subject: str, conversation_id: str) -> str:
        """The state part of ``ctx.system_context()``.

        Bounded on read as well as write: an owner past the cap contributes only
        their most recent keys, each truncated, with a marker saying so. Without
        it, an agent that remembers a few things per session grows the system
        prompt without limit.
        """
        if not self._ensure_engine() or self._state is None:
            return ""
        rows: list[dict] = []
        for scope, owner in (
            (SCOPE_APP, ""),
            (SCOPE_USER, subject),
            (SCOPE_SESSION, conversation_id),
        ):
            # one past the cap, so we can tell "exactly at the cap" from
            # "truncated" without a second count query
            rows.extend(
                self.list_state(scope, owner, limit=self._max_keys_injected + 1)
            )
        if not rows:
            return ""
        shown = rows[: self._max_keys_injected]
        lines = [
            f"- {r['key']}: {r['value'][: self._inject_value_chars]}" for r in shown
        ]
        if len(rows) > len(shown):
            # deliberately uncounted: the exact number would cost another query
            # and the model only needs to know its view is partial
            lines.append("- (older values not shown)")
        return "\n".join(lines)

    async def maybe_summarize(self, conversation_id: str) -> bool:
        """Fold old turns into the rolling summary.

        Three phases, and the split is the point: **the summarizer is called
        with no transaction open.** RonDB aborts transactions that hold locks
        past short timeouts, so wrapping an LLM call in one would roll back
        under any real latency and take concurrent turns down with it. Claiming
        and committing are separately short, and ``summary_version`` is what
        makes that safe — two racing folds cannot lose rows or double-fold;
        the loser just wasted a summarizer call.
        """
        if self._summarize is None or not self._ensure_engine():
            return False
        if self._sessions is None:
            return False
        for _ in range(3):
            try:
                claim = await asyncio.to_thread(self._claim_fold, conversation_id)
            except Exception:  # noqa: BLE001
                log.exception("Could not claim a summarization batch")
                return False
            if claim is None:
                return False
            version, old_summary, turns, max_id, count = claim

            try:
                if inspect.iscoroutinefunction(self._summarize):
                    summary = await self._summarize(old_summary, turns)
                else:
                    # A sync summarizer is very likely a blocking LLM call, so
                    # it goes to a thread. Note the check is on the callable,
                    # not its result: deciding by calling it first would already
                    # have blocked the event loop for the whole request.
                    summary = await asyncio.to_thread(
                        self._summarize, old_summary, turns
                    )
                    if inspect.isawaitable(summary):
                        summary = await summary
            except Exception:  # noqa: BLE001 — a failed summary must not fail
                # the request; the same batch is retried on a later turn
                log.exception("Summarizer failed; leaving history unfolded")
                return False
            if not isinstance(summary, str) or not summary.strip():
                log.warning("Summarizer returned no text; leaving history unfolded")
                return False

            try:
                committed = await asyncio.to_thread(
                    self._commit_fold,
                    conversation_id,
                    version,
                    summary,
                    max_id,
                    count,
                )
            except Exception:  # noqa: BLE001
                log.exception("Could not commit the summary")
                return False
            if committed:
                with self._lock:
                    self._cache.pop(conversation_id, None)
                return True
            log.info("Concurrent fold won; retrying with fresh state")
        return False

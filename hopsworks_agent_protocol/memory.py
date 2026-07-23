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
- :class:`SqlChatMemory` — any SQLAlchemy URL (e.g. the project's MySQL).
  Survives restarts and works across replicas.

If your framework already persists conversation state (a LangGraph
checkpointer, a LlamaIndex chat store), key it by ``conversation_id`` and do
NOT pass a memory store — one source of truth for history is enough.
"""

from __future__ import annotations

import logging
import os
import threading
from abc import ABC, abstractmethod

log = logging.getLogger(__name__)

Turn = dict[str, str]  # {"role": "user"|"assistant", "content": str}


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
            "SqlChatMemory instead."
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
    """Conversation store keyed by conversation_id."""

    @abstractmethod
    def get(self, conversation_id: str) -> list[Turn]:
        """Messages of the conversation in chronological order."""

    @abstractmethod
    def append(self, conversation_id: str, role: str, content: str) -> None:
        """Record one turn."""

    @abstractmethod
    def clear(self, conversation_id: str) -> None:
        """Drop a conversation."""


class InMemoryChatMemory(ChatMemory):
    """Process-local store for development: lost on restart (agent
    deployments can scale to zero) and not shared between replicas."""

    def __init__(self, max_messages: int = 50):
        self._max = max_messages
        self._conversations: dict[str, list[Turn]] = {}
        self._lock = threading.Lock()

    def get(self, conversation_id: str) -> list[Turn]:
        with self._lock:
            return list(self._conversations.get(conversation_id, []))

    def append(self, conversation_id: str, role: str, content: str) -> None:
        with self._lock:
            turns = self._conversations.setdefault(conversation_id, [])
            turns.append({"role": role, "content": content})
            if len(turns) > self._max:
                del turns[: len(turns) - self._max]

    def clear(self, conversation_id: str) -> None:
        with self._lock:
            self._conversations.pop(conversation_id, None)


class SqlChatMemory(ChatMemory):
    """SQLAlchemy-backed store (MySQL, Postgres, SQLite, ...).

    ``pip install 'hopsworks-agent-protocol[memory-sql]'``

    Inside a Hopsworks agent deployment both arguments are optional:
    ``SqlChatMemory()`` connects to the project MySQL using the
    platform-injected env vars and derives a per-deployment table name from
    ``DEPLOYMENT_ID``.

    A per-conversation cache avoids a read round-trip on every turn for the
    lifetime of the process.
    """

    def __init__(
        self,
        url: str | None = None,
        table_name: str | None = None,
        max_messages: int = 50,
    ):
        if url is None:
            url = deployment_mysql_url()
        if table_name is None:
            deployment_id = os.environ.get("DEPLOYMENT_ID", "default")
            table_name = f"agent_chat_memory_{deployment_id}"
        try:
            from sqlalchemy import (
                Column,
                Index,
                Integer,
                MetaData,
                String,
                Table,
                Text,
                create_engine,
            )
        except ImportError as err:
            raise ImportError(
                "SqlChatMemory requires SQLAlchemy: "
                "pip install 'hopsworks-agent-protocol[memory-sql]'"
            ) from err

        self._max = max_messages
        self._engine = create_engine(url, pool_pre_ping=True)
        self._cache: dict[str, list[Turn]] = {}
        self._lock = threading.Lock()
        metadata = MetaData()
        self._table = Table(
            table_name,
            metadata,
            Column("id", Integer, primary_key=True, autoincrement=True),
            Column("conversation_id", String(255), nullable=False),
            Column("role", String(16), nullable=False),
            Column("content", Text, nullable=False),
            Index(f"idx_{table_name}_conversation", "conversation_id"),
        )
        metadata.create_all(self._engine)
        log.info("SqlChatMemory ready (table=%s)", table_name)

    def get(self, conversation_id: str) -> list[Turn]:
        with self._lock:
            cached = self._cache.get(conversation_id)
            if cached is not None:
                return list(cached)
        with self._engine.connect() as conn:
            rows = conn.execute(
                self._table.select()
                .where(self._table.c.conversation_id == conversation_id)
                .order_by(self._table.c.id.desc())
                .limit(self._max)
            ).fetchall()
        turns = [{"role": r.role, "content": r.content} for r in reversed(rows)]
        with self._lock:
            self._cache[conversation_id] = list(turns)
        return turns

    def append(self, conversation_id: str, role: str, content: str) -> None:
        with self._engine.begin() as conn:
            conn.execute(
                self._table.insert().values(
                    conversation_id=conversation_id, role=role, content=content
                )
            )
        with self._lock:
            turns = self._cache.setdefault(conversation_id, [])
            turns.append({"role": role, "content": content})
            if len(turns) > self._max:
                del turns[: len(turns) - self._max]

    def clear(self, conversation_id: str) -> None:
        with self._engine.begin() as conn:
            conn.execute(
                self._table.delete().where(
                    self._table.c.conversation_id == conversation_id
                )
            )
        with self._lock:
            self._cache.pop(conversation_id, None)

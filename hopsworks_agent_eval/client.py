"""Calling a deployed agent, and finding the trace it produced.

This is the piece that only exists because the unit of evaluation is a deployed
endpoint rather than a local function. A client-side eval harness gets output
capture and trace correlation for free by being the caller; here both have to
be built, which is what the manifest read and the readiness poll are for.

Ported from ``scripts/stage1_correlation_probe.py``, which is the same logic
written to measure rather than to grade.
"""

from __future__ import annotations

import logging
from typing import Any

from .graders import Trace
from .runner import AgentResponse

log = logging.getLogger(__name__)


class HopsworksAgentClient:
    """Calls an agent through the ingress, reads its traces back through the
    Hopsworks API."""

    def __init__(
        self,
        session: Any,
        api_base: str,
        project_id: int,
        project_name: str,
        deployment_id: int,
        agent_url: str | None = None,
        timeout_s: float = 120.0,
    ):
        self._session = session
        self._api = f"{api_base.rstrip('/')}/hopsworks-api/api/project/{project_id}"
        self._deployment_id = deployment_id
        self._project_name = project_name
        self._agent_url = agent_url
        self._timeout_s = timeout_s
        self._manifest: dict[str, Any] | None = None
        self._chat_path = "/v1/chat"

    # ── the agent ───────────────────────────────────────────────────────────

    def _base_url(self) -> str:
        if self._agent_url:
            return self._agent_url.rstrip("/")
        deployment = self._session.get(
            f"{self._api}/serving/{self._deployment_id}", timeout=60
        ).json()
        name = deployment.get("name")
        istio = deployment.get("internalIPs") or deployment.get("externalIPs") or []
        host = istio[0] if istio else deployment.get("internalPath", "")
        self._agent_url = f"{host}/v1/{self._project_name}/{name}"
        return self._agent_url

    def manifest(self) -> dict[str, Any]:
        """The agent's self-description, which is what the runner refuses on.

        Fetched rather than assumed: an agent on an SDK too old to continue a
        trace context does not report the capability, and running against it
        would write trial rows pointing at traces that were never created.
        """
        if self._manifest is None:
            response = self._session.get(
                f"{self._base_url()}/.well-known/hopsworks-agent.json", timeout=60
            )
            response.raise_for_status()
            self._manifest = response.json()
            endpoints = self._manifest.get("endpoints", {})
            self._chat_path = endpoints.get("chat", "/v1/chat")
        return self._manifest

    def call(
        self, prompt: str, *, traceparent: str, baggage: str, timeout_s: float
    ) -> AgentResponse:
        import time

        # The payload is the hopsworks-agent protocol's, taken from the manifest
        # rather than configured per deployment: the SDK exists precisely so
        # every agent speaks one wire format regardless of its framework.
        payload = {
            "message": {"role": "user", "content": [{"type": "text", "text": prompt}]}
        }
        started = time.monotonic()
        try:
            response = self._session.post(
                f"{self._base_url()}{self._chat_path}",
                json=payload,
                headers={"traceparent": traceparent, "baggage": baggage},
                timeout=timeout_s,
            )
        except Exception as err:  # noqa: BLE001 — classified by the runner
            return AgentResponse(
                text="", latency_ms=(time.monotonic() - started) * 1000, error=str(err)
            )

        latency_ms = (time.monotonic() - started) * 1000
        if response.status_code >= 400:
            return AgentResponse(
                text="", latency_ms=latency_ms,
                error=f"agent returned {response.status_code}: {response.text[:200]}",
            )

        body = response.json()
        text = "".join(
            part.get("text", "")
            for part in (body.get("message") or {}).get("content", [])
            if part.get("type") == "text"
        )
        metadata = body.get("metadata") or {}
        return AgentResponse(
            text=text,
            # The SDK reports the trace id it actually used. The runner already
            # knows it from the traceparent it sent; reading it back is what
            # catches an agent that ignored the header rather than adopting it.
            trace_id=metadata.get("trace_id", ""),
            blocked_by_guardrail=bool(metadata.get("guardrail_blocked")),
            latency_ms=latency_ms,
            error="" if body.get("status") != "failed" else "agent reported failure",
        )

    # ── the trace ───────────────────────────────────────────────────────────

    def fetch_trace(self, trace_id: str) -> Trace | None:
        """The trace as a grader needs it: the aggregate plus what tools ran.

        Read through the backend rather than the feature store directly, so the
        runner needs no feature store credentials for grading and sees exactly
        what the UI sees.
        """
        response = self._session.get(
            f"{self._api}/otel/servings/{self._deployment_id}/traces/{trace_id}",
            timeout=60,
        )
        if response.status_code == 404:
            return None
        response.raise_for_status()
        trace = response.json()

        spans = trace.get("spans") or []
        if not spans:
            return None
        attributes = trace.get("spanAttributes") or []

        kinds: dict[str, str] = {}
        names: dict[str, str] = {}
        for attribute in attributes:
            key, span_id = attribute.get("attrKey"), attribute.get("spanId")
            if key == "openinference.span.kind":
                kinds[span_id] = (attribute.get("attrValue") or "").upper()
            elif key in ("tool.name", "gen_ai.tool.name"):
                names.setdefault(span_id, attribute.get("attrValue") or "")

        tool_spans = [s for s in spans if kinds.get(s.get("spanId")) == "TOOL"]
        return {
            "trace_id": trace_id,
            "root_span_id": next(
                (s.get("spanId") for s in spans if not s.get("parentSpanId")), ""
            ),
            # ordered by start time: a trajectory grader asks about sequence,
            # so an arbitrary order would make its verdict arbitrary too
            "tool_names": [
                names.get(s.get("spanId")) or s.get("name") or ""
                for s in sorted(tool_spans, key=lambda s: s.get("startTimeNs") or 0)
            ],
            "tool_error_count": sum(
                1 for s in tool_spans if str(s.get("statusCode", "")).endswith("ERROR")
            ),
            "span_count": len(spans),
        }

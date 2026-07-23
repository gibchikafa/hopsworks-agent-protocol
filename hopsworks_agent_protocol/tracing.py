"""Automatic OTel tracing for Hopsworks agent deployments.

When tracing is enabled on a Hopsworks agent deployment, the platform runs an
OTLP collector sidecar and injects ``OTEL_EXPORTER_OTLP_TRACES_ENDPOINT`` into
the predictor container. This module keys off that env var: if it is present,
a TracerProvider exporting to the sidecar is built and the framework-matching
OpenInference instrumentor is activated — no tracing code in the agent.

Frameworks:
- ``langgraph``  -> openinference-instrumentation-langchain (LangChain/LangGraph)
- ``llamaindex`` -> openinference-instrumentation-llama-index
- ``custom``     -> provider only; instrument manually via ``app.tracer_provider``

All imports are lazy and failures are non-fatal: a missing instrumentation
package logs a warning and the agent runs untraced.
"""

from __future__ import annotations

import logging
import os
from typing import Any

log = logging.getLogger(__name__)

TRACES_ENDPOINT_ENV = "OTEL_EXPORTER_OTLP_TRACES_ENDPOINT"
FRAMEWORK_ENV = "AGENT_FRAMEWORK"

FRAMEWORK_LANGGRAPH = "langgraph"
FRAMEWORK_LLAMAINDEX = "llamaindex"
FRAMEWORK_CUSTOM = "custom"
KNOWN_FRAMEWORKS = (FRAMEWORK_LANGGRAPH, FRAMEWORK_LLAMAINDEX, FRAMEWORK_CUSTOM)


def resolve_framework(framework: str | None) -> str:
    """Explicit argument wins; else the platform-injected AGENT_FRAMEWORK
    env var; else 'custom'."""
    value = (framework or os.environ.get(FRAMEWORK_ENV) or FRAMEWORK_CUSTOM).lower()
    if value not in KNOWN_FRAMEWORKS:
        log.warning(
            "Unknown agent framework %r, falling back to 'custom' "
            "(known: %s)", value, ", ".join(KNOWN_FRAMEWORKS),
        )
        return FRAMEWORK_CUSTOM
    return value


def setup_tracing(framework: str, enabled: bool | None = None) -> Any | None:
    """Build a TracerProvider exporting to the deployment's OTLP sidecar and
    instrument the given framework. Returns the provider, or None when tracing
    is off/unavailable.

    ``enabled=None`` (default) auto-detects from the presence of the
    ``OTEL_EXPORTER_OTLP_TRACES_ENDPOINT`` env var; ``False`` disables even
    when the env var is present.
    """
    endpoint = os.environ.get(TRACES_ENDPOINT_ENV)
    if enabled is False:
        return None
    if endpoint is None:
        if enabled is True:
            log.warning(
                "Tracing requested but %s is not set — is tracing enabled on "
                "the deployment?", TRACES_ENDPOINT_ENV,
            )
        return None

    try:
        from opentelemetry.exporter.otlp.proto.http.trace_exporter import (
            OTLPSpanExporter,
        )
        from opentelemetry.sdk import trace as trace_sdk
        from opentelemetry.sdk.trace.export import SimpleSpanProcessor
    except ImportError:
        log.warning(
            "Tracing endpoint %s is set but the OpenTelemetry SDK is not "
            "installed; running untraced. Install "
            "hopsworks-agent-protocol[tracing].", endpoint,
        )
        return None

    provider = trace_sdk.TracerProvider()
    # SimpleSpanProcessor on purpose: the Hopsworks OTLP sidecar relies on
    # spans arriving individually for its span propagation.
    provider.add_span_processor(SimpleSpanProcessor(OTLPSpanExporter(endpoint=endpoint)))

    _instrument(framework, provider)
    log.info("Tracing enabled (framework=%s, endpoint=%s)", framework, endpoint)
    return provider


def _instrument(framework: str, provider: Any) -> None:
    if framework == FRAMEWORK_LANGGRAPH:
        try:
            from openinference.instrumentation.langchain import LangChainInstrumentor

            LangChainInstrumentor().instrument(tracer_provider=provider)
        except ImportError:
            log.warning(
                "framework='langgraph' but openinference-instrumentation-langchain "
                "is not installed; spans from the framework will be missing. "
                "Install hopsworks-agent-protocol[langgraph].",
            )
    elif framework == FRAMEWORK_LLAMAINDEX:
        try:
            from openinference.instrumentation.llama_index import LlamaIndexInstrumentor

            LlamaIndexInstrumentor().instrument(tracer_provider=provider)
        except ImportError:
            log.warning(
                "framework='llamaindex' but openinference-instrumentation-llama-index "
                "is not installed; spans from the framework will be missing. "
                "Install hopsworks-agent-protocol[llamaindex].",
            )
    # 'custom': provider only — the agent instruments itself

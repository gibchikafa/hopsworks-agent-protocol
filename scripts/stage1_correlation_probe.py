#!/usr/bin/env python3
"""Stage 1 walking skeleton: does eval correlation actually work end to end,
and how long does a trace take to become readable?

Deliberately throwaway. It answers questions that cannot be answered on paper,
and its output — not its code — is the deliverable:

1. Does the agent adopt a trace id the caller generated? (If not, every trial
   row the eval runner writes points at a trace that does not exist.)
2. Do ``hopsworks.eval.*`` baggage entries survive into span attributes?
3. **How long after the response does the trace become readable?** That number
   becomes the eval runner's default trace-readiness timeout. If it is minutes
   rather than seconds, the runner's whole execution model needs rethinking.

It exercises the same path the real runner will: generate a traceparent, call
the deployment through its manifest-declared chat endpoint, then poll the
Hopsworks trace API until the trace shows up.

    python scripts/stage1_correlation_probe.py \
        --hopsworks-host https://hopsworks.example.com \
        --project-id 119 --serving-id 1035 \
        --agent-url https://<istio-endpoint> \
        --api-key-file ~/.hopsworks/api.key \
        --trials 10

Needs ``requests``. The API key needs the SERVING scope.
"""

from __future__ import annotations

import argparse
import json
import os
import secrets
import statistics
import sys
import time
from dataclasses import dataclass, field, asdict

import requests

# What the SDK stamps, mirrored here rather than imported: this script must run
# in a bare environment (a job, a laptop) that need not have the SDK installed.
EVAL_RUN_ID = "hopsworks.eval.run_id"
EVAL_TRIAL_ID = "hopsworks.eval.trial_id"
EVAL_TRIAL_INDEX = "hopsworks.eval.trial_index"
INPUT_VALUE = "input.value"
OUTPUT_VALUE = "output.value"


@dataclass
class TrialResult:
    trial_index: int
    trace_id: str
    ok_response: bool = False
    http_status: int = 0
    response_ms: float = 0.0
    # correlation
    trace_found: bool = False
    root_span_found: bool = False
    reported_trace_id: str | None = None  # from response metadata
    baggage_seen: bool = False
    authoritative_io: bool = False
    # readiness, measured from the moment the response returned — which is
    # where the runner starts waiting
    root_ready_s: float | None = None
    stable_ready_s: float | None = None
    span_count: int = 0
    error: str = ""
    notes: list[str] = field(default_factory=list)


def read_api_key(args) -> str:
    if args.api_key:
        return args.api_key.strip()
    if args.api_key_file:
        return open(os.path.expanduser(args.api_key_file)).read().strip()
    key = os.environ.get("HOPSWORKS_API_KEY")
    if not key:
        sys.exit("No API key: pass --api-key / --api-key-file or set HOPSWORKS_API_KEY")
    return key.strip()


def new_traceparent() -> tuple[str, str]:
    """A W3C traceparent, sampled. The trace id is known before the call, which
    is the entire point: it is what lets a failed or slow trial still be
    correlated."""
    trace_id = secrets.token_hex(16)
    span_id = secrets.token_hex(8)
    return trace_id, f"00-{trace_id}-{span_id}-01"


def preflight(session: requests.Session, agent_url: str) -> dict:
    """Read the manifest and refuse to measure what cannot work.

    This is the check the real runner performs before firing a suite: an agent
    that does not report trace_correlation will produce trial rows pointing at
    traces it never created, and finding that out from the results is far worse
    than finding it out here.
    """
    url = f"{agent_url.rstrip('/')}/.well-known/hopsworks-agent.json"
    try:
        response = session.get(url, timeout=30)
    except requests.RequestException as err:
        sys.exit(f"Could not read the agent manifest at {url}: {err}")
    if response.status_code != 200:
        sys.exit(
            f"Manifest at {url} returned {response.status_code}. This probe "
            "targets hopsworks-agent-protocol deployments; a non-SDK agent "
            "needs a custom request adapter."
        )
    manifest = response.json()
    caps = manifest.get("capabilities", {})
    print(f"  protocol        : {manifest.get('protocol')} "
          f"v{manifest.get('protocol_version')}")
    print(f"  agent           : {manifest.get('agent', {}).get('name')} "
          f"({manifest.get('agent', {}).get('framework')})")
    print(f"  trace_correlation: {caps.get('trace_correlation', 'ABSENT')}")
    print(f"  eval_mode       : {caps.get('eval_mode', 'ABSENT')}")

    if "trace_correlation" not in caps:
        print(
            "\n  !! The manifest does not report trace_correlation at all, "
            "which means an SDK older than 1.6.0.\n"
            "     Correlation cannot work: the agent will start a fresh trace "
            "and ignore the traceparent.\n"
            "     Rebuild the agent base image before measuring.",
            file=sys.stderr,
        )
        sys.exit(2)
    if not caps["trace_correlation"]:
        print(
            "\n  !! trace_correlation is false — the SDK supports it but "
            "tracing is off on this deployment.\n"
            "     Enable tracing, otherwise there are no traces to wait for.",
            file=sys.stderr,
        )
        sys.exit(2)
    return manifest


def call_agent(
    session: requests.Session, agent_url: str, endpoint: str, prompt: str,
    traceparent: str, baggage: str, timeout: int,
) -> tuple[requests.Response | None, float, str]:
    url = f"{agent_url.rstrip('/')}{endpoint}"
    payload = {
        "message": {"role": "user", "content": [{"type": "text", "text": prompt}]}
    }
    started = time.monotonic()
    try:
        response = session.post(
            url,
            json=payload,
            headers={"traceparent": traceparent, "baggage": baggage},
            timeout=timeout,
        )
    except requests.RequestException as err:
        return None, (time.monotonic() - started) * 1000, str(err)
    return response, (time.monotonic() - started) * 1000, ""


def poll_trace(
    session: requests.Session, base: str, trace_id: str,
    timeout_s: float, interval_s: float, stable_polls: int,
) -> tuple[dict | None, float | None, float | None]:
    """Poll until the root span arrives, then until the span count stops
    growing.

    Two numbers, because they answer different questions. Time-to-root is when
    a final-answer evaluator could run. Time-to-stable is when a *trajectory*
    evaluator could run — spans arrive incrementally, so grading the moment the
    root lands would judge a half-written trajectory. The gap between them is
    what the design's grace period has to cover.
    """
    deadline = time.monotonic() + timeout_s
    started = time.monotonic()
    root_at: float | None = None
    last_count = -1
    stable_for = 0
    trace: dict | None = None

    while time.monotonic() < deadline:
        try:
            response = session.get(f"{base}/traces/{trace_id}", timeout=30)
        except requests.RequestException:
            time.sleep(interval_s)
            continue  # a transient failure must not end the poll

        if response.status_code == 200:
            trace = response.json()
            spans = trace.get("spans") or []
            if root_at is None and any(not s.get("parentSpanId") for s in spans):
                root_at = time.monotonic() - started
            if root_at is not None:
                if len(spans) == last_count:
                    stable_for += 1
                    if stable_for >= stable_polls:
                        return trace, root_at, time.monotonic() - started
                else:
                    stable_for = 0
                last_count = len(spans)
        time.sleep(interval_s)

    return trace, root_at, None


def inspect_trace(trace: dict, result: TrialResult) -> None:
    spans = trace.get("spans") or []
    attributes = trace.get("spanAttributes") or []
    result.span_count = len(spans)
    result.trace_found = bool(spans)
    result.root_span_found = any(not s.get("parentSpanId") for s in spans)

    keys = {a.get("attrKey") for a in attributes}
    result.baggage_seen = EVAL_RUN_ID in keys
    result.authoritative_io = INPUT_VALUE in keys and OUTPUT_VALUE in keys

    if not result.baggage_seen:
        result.notes.append(
            "eval baggage missing: the BaggageSpanProcessor did not run, or "
            "baggage was stripped in transit (check the ingress)"
        )
    if not result.authoritative_io:
        result.notes.append(
            "no input.value/output.value on any span: the turn span is not "
            "carrying the authoritative transcript"
        )


def percentile(values: list[float], pct: float) -> float:
    if not values:
        return float("nan")
    ordered = sorted(values)
    k = (len(ordered) - 1) * pct / 100.0
    lo = int(k)
    hi = min(lo + 1, len(ordered) - 1)
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (k - lo)


def summarize(results: list[TrialResult], args) -> None:
    total = len(results)
    answered = [r for r in results if r.ok_response]
    correlated = [r for r in answered if r.root_span_found]
    root_times = [r.root_ready_s for r in results if r.root_ready_s is not None]
    stable_times = [r.stable_ready_s for r in results if r.stable_ready_s is not None]

    def ratio(count: int, denominator: int) -> str:
        # never print "0/1" for an empty population: it reads as one failure
        # rather than nothing measured
        return f"{count}/{denominator}" if denominator else "n/a"

    print("\n" + "=" * 68)
    print("STAGE 1 RESULTS")
    print("=" * 68)
    print(f"trials                     : {total}")
    print(f"agent answered             : {ratio(len(answered), total)}")
    print(f"trace found by our id      : "
          f"{ratio(len(correlated), len(answered))}")
    print(f"response reported trace_id : "
          f"{ratio(sum(1 for r in answered if r.reported_trace_id), len(answered))}")
    print(f"trace_id agreed            : "
          f"{ratio(sum(1 for r in answered if r.reported_trace_id == r.trace_id), len(answered))}")
    print(f"eval baggage on spans      : "
          f"{ratio(sum(1 for r in correlated if r.baggage_seen), len(correlated))}")
    print(f"authoritative input/output : "
          f"{ratio(sum(1 for r in correlated if r.authoritative_io), len(correlated))}")
    print(f"never became readable      : {len(answered) - len(root_times)}")

    if root_times:
        print("\ntime from response to ROOT SPAN readable (s):")
        print(f"  p50 {percentile(root_times, 50):6.1f}   "
              f"p95 {percentile(root_times, 95):6.1f}   "
              f"max {max(root_times):6.1f}   "
              f"mean {statistics.mean(root_times):6.1f}")
    if stable_times:
        print("time from response to TRAJECTORY stable (s):")
        print(f"  p50 {percentile(stable_times, 50):6.1f}   "
              f"p95 {percentile(stable_times, 95):6.1f}   "
              f"max {max(stable_times):6.1f}   "
              f"mean {statistics.mean(stable_times):6.1f}")

    print("\n" + "-" * 68)
    if not correlated:
        print("VERDICT: correlation is BROKEN. Nothing downstream can work — "
              "the runner\n         cannot find the traces it caused. Fix "
              "before Stage 2.")
    elif len(correlated) < len(answered):
        print(f"VERDICT: correlation is FLAKY ({len(correlated)}/{len(answered)}). "
              "Investigate before\n         trusting any trajectory grading.")
    elif stable_times:
        suggested = max(30, int(percentile(stable_times, 95) * 3))
        print("VERDICT: correlation works.")
        print(f"         Suggested runner readiness timeout: {suggested}s "
              f"(3x p95 of trajectory-stable).")
        print(f"         Suggested featurization grace period: "
              f"{max(60, int(max(stable_times) * 2))}s.")
    else:
        print("VERDICT: traces correlate but never stabilized within "
              f"{args.readiness_timeout}s.\n         Raise --readiness-timeout "
              "and re-run before drawing conclusions.")
    print("-" * 68)

    if args.out:
        with open(args.out, "w") as handle:
            json.dump([asdict(r) for r in results], handle, indent=2)
        print(f"\nper-trial detail written to {args.out}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hopsworks-host", required=True)
    parser.add_argument("--project-id", required=True, type=int)
    parser.add_argument("--serving-id", required=True, type=int)
    parser.add_argument("--agent-url", required=True,
                        help="the deployment's istio endpoint")
    parser.add_argument("--api-key")
    parser.add_argument("--api-key-file")
    parser.add_argument("--trials", type=int, default=10)
    parser.add_argument("--prompt", default="What is 2 + 2? Answer briefly.")
    parser.add_argument("--request-timeout", type=int, default=120)
    parser.add_argument("--readiness-timeout", type=float, default=300.0)
    parser.add_argument("--poll-interval", type=float, default=1.0)
    parser.add_argument("--stable-polls", type=int, default=3,
                        help="consecutive polls with no new spans before a "
                             "trajectory counts as complete")
    parser.add_argument("--out", default="stage1_results.json")
    parser.add_argument("--insecure", action="store_true")
    args = parser.parse_args()

    api_key = read_api_key(args)
    session = requests.Session()
    session.headers["Authorization"] = f"ApiKey {api_key}"
    session.verify = not args.insecure
    if args.insecure:
        requests.packages.urllib3.disable_warnings()  # noqa: SLF001

    print("PRE-FLIGHT")
    manifest = preflight(session, args.agent_url)
    endpoint = manifest.get("endpoints", {}).get("chat", "/v1/chat")

    base = (f"{args.hopsworks_host.rstrip('/')}/hopsworks-api/api/project/"
            f"{args.project_id}/otel/servings/{args.serving_id}")
    run_id = f"stage1_{secrets.token_hex(4)}"
    print(f"\nrun_id {run_id} — {args.trials} trials against {args.agent_url}\n")

    results: list[TrialResult] = []
    for index in range(args.trials):
        trace_id, traceparent = new_traceparent()
        trial_id = f"{run_id}/probe/1/{index}"
        baggage = ",".join([
            f"{EVAL_RUN_ID}={run_id}",
            f"{EVAL_TRIAL_ID}={trial_id}",
            f"{EVAL_TRIAL_INDEX}={index}",
        ])
        result = TrialResult(trial_index=index, trace_id=trace_id)

        response, elapsed_ms, error = call_agent(
            session, args.agent_url, endpoint, args.prompt,
            traceparent, baggage, args.request_timeout,
        )
        result.response_ms = elapsed_ms
        if response is None:
            result.error = error
            print(f"  [{index}] {trace_id[:12]}… REQUEST FAILED: {error}")
            results.append(result)
            continue

        result.http_status = response.status_code
        result.ok_response = response.status_code == 200
        if result.ok_response:
            try:
                result.reported_trace_id = (
                    (response.json().get("metadata") or {}).get("trace_id")
                )
            except ValueError:
                result.notes.append("response body was not JSON")

        trace, root_at, stable_at = poll_trace(
            session, base, trace_id,
            args.readiness_timeout, args.poll_interval, args.stable_polls,
        )
        result.root_ready_s = root_at
        result.stable_ready_s = stable_at
        if trace:
            inspect_trace(trace, result)

        status = "OK " if result.root_span_found else "MISS"
        ready = f"{root_at:5.1f}s" if root_at is not None else "  n/a"
        stable = f"{stable_at:5.1f}s" if stable_at is not None else "  n/a"
        print(f"  [{index}] {trace_id[:12]}… {status} "
              f"answer {elapsed_ms/1000:5.1f}s  root {ready}  "
              f"stable {stable}  spans {result.span_count}"
              + ("  baggage✓" if result.baggage_seen else "  baggage✗"))
        for note in result.notes:
            print(f"        note: {note}")
        results.append(result)

    summarize(results, args)


if __name__ == "__main__":
    main()

#!/usr/bin/env bash
#
# End-to-end walkthrough of the agent evaluation API:
#   create suite -> add task -> publish -> run -> read results
#
# Exercises the refusals as well as the happy path, because the refusals are the
# part worth having. A gate that only works when you do the right thing has not
# been tested.
#
# The run will not execute: nothing launches the runner yet. Everything up to
# and including recording the run is real, and the result endpoints are checked
# to return empty rather than to error — which is what they should do before a
# runner writes anything.
#
#   ./eval_api_walkthrough.sh \
#       --host https://10.114.123.130 \
#       --project 119 \
#       --deployment 3 \
#       --api-key-file ~/.hopsworks/api.key
#
# The API key needs the SERVING scope. Add --trace <traceId> to also exercise
# promotion; without it those steps are skipped.

set -uo pipefail

HOST=""; PROJECT=""; DEPLOYMENT=""; API_KEY=""; API_KEY_FILE=""; TRACE_ID=""; INSECURE=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --host) HOST="$2"; shift 2 ;;
    --project) PROJECT="$2"; shift 2 ;;
    --deployment) DEPLOYMENT="$2"; shift 2 ;;
    --api-key) API_KEY="$2"; shift 2 ;;
    --api-key-file) API_KEY_FILE="$2"; shift 2 ;;
    --trace) TRACE_ID="$2"; shift 2 ;;
    --insecure) INSECURE="-k"; shift ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done

[[ -z "$API_KEY" && -n "$API_KEY_FILE" ]] && API_KEY="$(tr -d '\n' < "${API_KEY_FILE/#\~/$HOME}")"
[[ -z "$API_KEY" ]] && API_KEY="${HOPSWORKS_API_KEY:-}"
if [[ -z "$HOST" || -z "$PROJECT" || -z "$DEPLOYMENT" || -z "$API_KEY" ]]; then
  echo "need --host, --project, --deployment and an API key" >&2
  exit 2
fi

BASE="${HOST%/}/hopsworks-api/api/project/${PROJECT}/agent-evals"
PASS=0; FAIL=0

BODY_FILE=/tmp/eval_wt_body
STATUS_FILE=/tmp/eval_wt_status

# Prints the body, and records the status where the caller can still read it.
# A variable would not survive: most calls are wrapped in $(...), which runs
# this in a subshell, so the assignment would be lost with it.
call() {
  local method="$1" path="$2" body="${3:-}"
  local args=(-s -o "$BODY_FILE" -w '%{http_code}' -X "$method"
              -H "Authorization: ApiKey ${API_KEY}" -H 'Content-Type: application/json')
  [[ -n "$INSECURE" ]] && args+=("$INSECURE")
  [[ -n "$body" ]] && args+=(-d "$body")
  curl "${args[@]}" "${BASE}${path}" > "$STATUS_FILE"
  cat "$BODY_FILE"
}

# Asserts on the status of the most recent call. Keeping status and body apart
# is what lets a step assert a refusal rather than only a success.
expect() {
  local want="$1" what="$2" got
  got="$(cat "$STATUS_FILE" 2>/dev/null || echo '-')"
  if [[ "$got" == "$want" ]]; then
    PASS=$((PASS + 1)); printf '  \033[32mok\033[0m   %s (%s)\n' "$what" "$got"
  else
    FAIL=$((FAIL + 1)); printf '  \033[31mFAIL\033[0m %s: wanted %s, got %s\n' "$what" "$want" "$got"
    sed 's/^/       /' "$BODY_FILE" 2>/dev/null; echo
  fi
}

jsonval() { python3 -c "import sys,json;d=json.load(sys.stdin);print(d.get('$1',''))" 2>/dev/null; }
count()   { python3 -c "import sys,json;print(len(json.load(sys.stdin)))" 2>/dev/null || echo "?"; }

echo "=== 1. create a suite ==="
SUITE_JSON=$(call POST /suites '{"name":"walkthrough","type":"regression","executionMode":"read_only"}')
expect 200 "suite created"
SUITE_ID=$(echo "$SUITE_JSON" | jsonval suiteId)
echo "     suiteId=$SUITE_ID  status=$(echo "$SUITE_JSON" | jsonval status)"

echo "=== 2. a safety suite that is not sandboxed is refused ==="
call POST /suites '{"name":"attacks","type":"safety","executionMode":"read_only"}' > /dev/null
expect 400 "safety suite must be sandboxed"

echo "=== 3. an empty suite cannot be published ==="
call POST "/suites/${SUITE_ID}/publish" > /dev/null
expect 400 "empty suite refused"

echo "=== 4. author a task ==="
TASK_JSON=$(call POST /tasks '{"inputMessages":"[{\"role\":\"user\",\"content\":\"what is 2+2?\"}]","expectedOutput":"4","taskType":"single_turn"}')
expect 200 "task created"
TASK_ID=$(echo "$TASK_JSON" | jsonval taskId)
echo "     taskId=$TASK_ID  redactionStatus=$(echo "$TASK_JSON" | jsonval redactionStatus)"

echo "=== 5. add it to the suite ==="
call POST "/tasks/${TASK_ID}/suite?suiteId=${SUITE_ID}" > /dev/null
expect 200 "authored task joins a suite"

echo "=== 6. publish ==="
call POST "/suites/${SUITE_ID}/publish" > /dev/null
expect 200 "suite published"

echo "=== 7. a published suite is frozen ==="
TASK2=$(call POST /tasks '{"inputMessages":"[{\"role\":\"user\",\"content\":\"and 3+3?\"}]","expectedOutput":"6"}')
TASK2_ID=$(echo "$TASK2" | jsonval taskId)
call POST "/tasks/${TASK2_ID}/suite?suiteId=${SUITE_ID}" > /dev/null
expect 400 "published suite refuses new tasks"

if [[ -n "$TRACE_ID" ]]; then
  echo "=== 8. promote a trace ==="
  PROMOTED=$(call POST "/tasks/from-trace/${TRACE_ID}?deploymentId=${DEPLOYMENT}" '{"inputMessages":"[{\"role\":\"user\",\"content\":\"promoted\"}]","expectedOutput":"corrected"}')
  expect 200 "trace promoted"
  PROMOTED_ID=$(echo "$PROMOTED" | jsonval taskId)
  PROMOTED_STATUS=$(echo "$PROMOTED" | jsonval redactionStatus)
  [[ "$PROMOTED_STATUS" == "PENDING_REDACTION" ]] \
    && { PASS=$((PASS+1)); printf '  \033[32mok\033[0m   promoted task is PENDING_REDACTION\n'; } \
    || { FAIL=$((FAIL+1)); printf '  \033[31mFAIL\033[0m promoted task is %s\n' "$PROMOTED_STATUS"; }

  echo "=== 9. THE GATE: an unreviewed task cannot join a suite ==="
  DRAFT=$(call POST /suites '{"name":"walkthrough-draft","type":"regression"}')
  DRAFT_ID=$(echo "$DRAFT" | jsonval suiteId)
  call POST "/tasks/${PROMOTED_ID}/suite?suiteId=${DRAFT_ID}" > /dev/null
  expect 400 "unreviewed promoted task refused"

  echo "=== 10. confirm the review, then it may join ==="
  call POST "/tasks/${PROMOTED_ID}/redaction" '{"expectedOutput":"corrected and reviewed"}' > /dev/null
  expect 200 "redaction confirmed"
  call POST "/tasks/${PROMOTED_ID}/suite?suiteId=${DRAFT_ID}" > /dev/null
  expect 200 "reviewed task joins a suite"

  echo "=== 11. the promotion queue ==="
  echo "     pending redaction: $(call GET '/tasks?pendingRedaction=true' | count)"
  echo "     unassigned:        $(call GET '/tasks?unassigned=true' | count)"
else
  echo "=== 8-11. promotion steps skipped (pass --trace <traceId> to run them) ==="
fi

echo "=== 12. record a run ==="
RUN=$(call POST "/runs?suiteId=${SUITE_ID}&deploymentId=${DEPLOYMENT}&nTrials=3")
expect 200 "run recorded"
RUN_ID=$(echo "$RUN" | jsonval runId)
echo "     runId=$RUN_ID  status=$(echo "$RUN" | jsonval status)"

echo "=== 13. a draft suite cannot be run ==="
DRAFT2=$(call POST /suites '{"name":"walkthrough-unpublished","type":"capability"}')
DRAFT2_ID=$(echo "$DRAFT2" | jsonval suiteId)
call POST "/runs?suiteId=${DRAFT2_ID}&deploymentId=${DEPLOYMENT}" > /dev/null
expect 400 "draft suite refused for a run"

echo "=== 14. results are empty, not broken ==="
# nothing launches the runner yet, so these should return [] rather than error
echo "     trials:         $(call GET "/runs/${RUN_ID}/trials" | count)"
expect 200 "trials readable"
echo "     grader results: $(call GET "/runs/${RUN_ID}/grader-results" | count)"
expect 200 "grader results readable"
echo "     metrics:        $(call GET "/runs/${RUN_ID}/metrics" | count)"
expect 200 "metrics readable"

echo "=== 15. runs for this deployment ==="
echo "     runs: $(call GET "/runs?deploymentId=${DEPLOYMENT}" | count)"
expect 200 "run history readable"

if [[ -n "$TRACE_ID" ]]; then
  echo "=== 16. purge everything promoted from that trace ==="
  echo "     removed: $(call DELETE "/tasks/from-trace/${TRACE_ID}" | jsonval removed)"
  expect 200 "deletion-request path works"
fi

rm -f /tmp/eval_wt_body "$STATUS_FILE"
echo
echo "-------------------------------------------------------------"
printf 'passed %d, failed %d\n' "$PASS" "$FAIL"
[[ $FAIL -eq 0 ]] \
  && echo "The metadata chain works end to end. A run records but does not execute:
nothing launches the runner yet." \
  || echo "Something above is wrong — see the failures."
echo "-------------------------------------------------------------"
exit $(( FAIL > 0 ? 1 : 0 ))

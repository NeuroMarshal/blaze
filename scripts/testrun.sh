#!/usr/bin/env bash
# Full reproducible test run: native server, complete test output, statuses, GPU metadata and separate server log.
# scripts/testrun.sh NAME COMMAND... (COMMAND must bind PORT, default 8100)
# STOP_TIMEOUT (60s) bounds teardown, separately from the server HTTP drain timeout.
# TESTS="api bench prefix soak needle"; LOGS; PY; PROMPTS; PROMPT_FILE; OUTPUT_TOKENS; CONCURRENCY; TEMPERATURE; SOAK_MIN; NEEDLES.
set -uo pipefail
if [ -z "${BLAZE_LOCKED:-}" ]; then
  exec env BLAZE_LOCKED=1 flock /tmp/blaze-gpu.lock "$0" "$@"
fi
NAME="${1:?usage: scripts/testrun.sh NAME COMMAND...}"
shift
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
export PORT="${PORT:-8100}" PYTHONUNBUFFERED=1
URL="http://127.0.0.1:$PORT"
LOGS="${LOGS:-${XDG_STATE_HOME:-$HOME/.local/state}/blaze/tests}"
mkdir -p "$LOGS"
OUT="$LOGS/run-$(date +%Y%m%d-%H%M%S)-$NAME.txt"
SRV="${OUT%.txt}.server.log"
TESTS="${TESTS:-api bench prefix}"
SUMMARY=()
FAILED=0
pid=""
log() { echo "$@" | tee -a "$OUT"; }
# An exited child may still be a zombie until wait; kill -0 alone does not detect completion.
running() {
  local state
  state=$(ps -o stat= -p "$1" 2>/dev/null) || return 1
  [[ -n "$state" && "$state" != *Z* ]]
}
cleanup() {
  if [ -n "$pid" ]; then
    local status=0 forced=0
    kill "$pid" 2>/dev/null || true
    for _ in $(seq 1 "${STOP_TIMEOUT:-60}"); do running "$pid" || break; sleep 1; done
    if running "$pid"; then forced=1; kill -KILL "$pid" 2>/dev/null || true; fi
    wait "$pid" 2>/dev/null || status=$?
    SUMMARY+=("server shutdown exit=$status forced=$forced")
    if ((forced || (status != 0 && status != 143))); then FAILED=1; fi
    pid=""
  fi
}
trap cleanup EXIT
trap 'exit 130' INT TERM
run_test() {
  local label="$1"; shift
  local started=$SECONDS status
  log "## $label start $(date -Is)"
  "$@" 2>&1 | tee -a "$OUT"
  status=${PIPESTATUS[0]}
  SUMMARY+=("$label exit=$status elapsed=$((SECONDS - started))s")
  log "## $label exit=$status elapsed=$((SECONDS - started))s"
  if [ "$status" != 0 ]; then FAILED=1; fi
}
{
  echo "# test run $NAME $(date -Is)"
  git -C "$ROOT" log -1 --format='commit %H %s'
  git -C "$ROOT" diff --stat
  nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv,noheader
  echo "url=$URL tests=$TESTS"
  echo "server command (API keys redacted):"
  redact=0
  for arg in "$@"; do
    if [ "$redact" = 1 ]; then printf '%s ' '[REDACTED]'; redact=0; continue; fi
    case "$arg" in
      --api-key) printf '%s ' "$arg"; redact=1 ;;
      --api-key=*) printf '%s ' '--api-key=[REDACTED]' ;;
      *) printf '%q ' "$arg" ;;
    esac
  done
  echo
} | tee "$OUT"
if curl -fsS --max-time 2 "$URL/health" >/dev/null 2>&1; then
  log "FAIL: $URL already has a server; refusing to benchmark a different process"
  exit 1
fi
"$@" >"$SRV" 2>&1 < /dev/null &
pid=$!
t0=$SECONDS
until curl -fsS --max-time 2 "$URL/health" >/dev/null 2>&1; do
  if ! running "$pid" || ((SECONDS - t0 > 1200)); then
    log "FAIL: server exited or startup exceeded 1200 seconds"
    tail -40 "$SRV" | tee -a "$OUT"
    exit 1
  fi
  sleep 2
done
log "## server ready after $((SECONDS - t0))s; full server log: $SRV"
if [ -z "${PY:-}" ]; then
  for candidate in "$ROOT/.venv-test/bin/python" "$ROOT/.venv/bin/python" python3; do
    if "$candidate" -c 'import openai' 2>/dev/null; then PY="$candidate"; break; fi
  done
fi
for test in $TESTS; do
  case "$test" in
    api)
      if [ -n "${PY:-}" ]; then
        run_test api "$PY" "$ROOT/scripts/api_test.py" --base-url "$URL" --api-key "${BLAZE_API_KEY:-none}" ${API_K:+-k "$API_K"}
      else SUMMARY+=("api FAIL: Python openai SDK unavailable"); FAILED=1; fi ;;
    bench)
      BENCH_SEED=()
      if [ -n "${PROMPT_SEED:-}" ]; then BENCH_SEED=(--prompt-seed "$PROMPT_SEED"); fi
      BENCH_MATERIAL=()
      if [ -n "${PROMPT_FILE:-}" ]; then BENCH_MATERIAL=(--prompt-file "$PROMPT_FILE"); fi
      run_test bench "$ROOT/target/release/blaze-bench" --url "$URL" --label "$NAME" \
        --prompt-tokens "${PROMPTS:-1024,8192,32768}" --output-tokens "${OUTPUT_TOKENS:-256}" \
        --concurrency "${CONCURRENCY:-1}" --repeats "${REPEATS:-1}" --temperature "${TEMPERATURE:-0.8}" \
        --warm-cache "${BENCH_SEED[@]}" "${BENCH_MATERIAL[@]}" --output "${OUT%.txt}.bench.json" ;;
    prefix) run_test prefix python3 "$ROOT/scripts/prefix_cache_test.py" "$URL" --turns 6 --side 2 ;;
    cache-pressure)
      run_test cache-pressure "$ROOT/target/release/blaze-bench" --url "$URL" --label "$NAME-cache-pressure" \
        --prompt-tokens 32768,131072 --output-tokens 32 --temperature 0 --warm-cache --prompt-seed 9251 \
        --output "${OUT%.txt}.cache-pressure.json" ;;
    kv) run_test kv-pressure python3 "$ROOT/scripts/kv_pressure_test.py" "$URL" --sessions 4 --max-tokens "${PRESSURE_TOKENS:-4000}" ;;
    soak) run_test soak python3 "$ROOT/scripts/soak_test.py" "$URL" --clients 4 --minutes "${SOAK_MIN:-3}" --long-tokens 8000 ;;
    needle)
      for n in ${NEEDLES:-122000}; do run_test "needle-$n" timeout 3600 python3 "$ROOT/scripts/needle_test.py" "$n" --url "$URL"; done ;;
    spec) run_test spec python3 "$ROOT/scripts/spec_bench.py" "$URL" "${OUT%.txt}.spec" ;;
    conc)
      for sessions in ${SESSIONS:-1 2 4}; do
        run_test "concurrency-$sessions" timeout 1200 "$ROOT/target/release/blaze-bench" --url "$URL" \
          --concurrency "$sessions" --prompt-tokens "${PROMPTS:-1024}" --output-tokens "${OUTPUT_TOKENS:-256}" \
          --output "${OUT%.txt}.concurrency-$sessions.json"
      done ;;
    *) SUMMARY+=("unknown test $test"); FAILED=1 ;;
  esac
done
log "## server error scan"
if rg -n 'panicked|step failed|blaze hang|grammar dropped|ERROR.*api.error|\"level\":\"ERROR\"' "$SRV" | tee -a "$OUT"; then
  SUMMARY+=("server-log FAIL"); FAILED=1
else SUMMARY+=("server-log clean"); fi
cleanup
log "## summary"
for result in "${SUMMARY[@]}"; do log "$result"; done
log "result=$FAILED report=$OUT server_log=$SRV"
exit "$FAILED"

#!/usr/bin/env bash
# The whole check of a build on this machine: build, unit tests, every kernel test, then a test server on PORT with
# the API suite, the agent prefix-cache test, a soak run, and a small-KV pressure run; the server logs are searched
# for panics, failed steps, slow steps and server errors. A summary at the end; exit 1 when anything failed.
#   scripts/check.sh MODEL [serve options...]       e.g. scripts/check.sh ~/models/... --tp 2 --draft-model ...
# Env: PORT (8100), PY (a python with the openai package), SOAK_MIN (3; 0: no soak), KB (the kernel build directory),
#      KTIMEOUT (seconds per kernel test, 1800: test_hybrid_graph takes over 15 minutes on the laptop's CPU),
#      LOGS (where the logs go; default a new directory under /tmp), SKIP_BUILD=1.
# The GPUs are taken through /tmp/blaze-gpu.lock for the whole run.
set -uo pipefail
if [ -z "${BLAZE_LOCKED:-}" ]; then
  exec env BLAZE_LOCKED=1 flock /tmp/blaze-gpu.lock "$0" "$@"
fi
MODEL="${1:?usage: scripts/check.sh MODEL [serve options...]}"
shift
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
PORT="${PORT:-8100}"
URL="http://127.0.0.1:$PORT"
LOGS="${LOGS:-$(mktemp -d /tmp/blaze-check.XXXXXX)}"
mkdir -p "$LOGS"
if [ -z "${PY:-}" ]; then
  for p in "$ROOT/.venv-test/bin/python" "$ROOT/.venv/bin/python" python3; do
    if "$p" -c "import openai" 2>/dev/null; then PY="$p"; break; fi
  done
fi
KB="${KB:-$(for d in kernels/build kernels/build-*; do [ -x "$d/test_ops" ] && echo "$d" && break; done)}"
declare -a RESULTS
pass() { RESULTS+=("ok    $1"); echo "== ok: $1"; }
fail() { RESULTS+=("FAIL  $1"); echo "== FAIL: $1"; }
run() {  # name, command...
  local name="$1"; shift
  echo "== $name"
  if "$@" > "$LOGS/$name.log" 2>&1; then pass "$name"; else fail "$name (log: $LOGS/$name.log)"; tail -5 "$LOGS/$name.log"; fi
}

if [ -z "${SKIP_BUILD:-}" ]; then
  run build cargo build --release
  [ -n "$KB" ] && run kernels-build cmake --build "$KB" -j "$(nproc)"
fi
run cargo-test cargo test --release --workspace
if [ -n "$KB" ]; then
  for t in "$KB"/test_*; do
    [ -x "$t" ] || continue
    # line-buffered: a test that runs out of time still shows how far it got
    run "kernel-$(basename "$t")" timeout "${KTIMEOUT:-1800}" stdbuf -oL -eL "$t"
  done
else
  fail "kernel tests (no kernel build directory with test binaries; set KB)"
fi

serve() {  # log name, extra options...
  local log="$1"; shift
  target/release/blaze serve --model "$MODEL" --port "$PORT" "$@" > "$LOGS/$log" 2>&1 &
  SERVER=$!
  for _ in $(seq 1 600); do
    curl -s "$URL/health" >/dev/null && return 0
    kill -0 "$SERVER" 2>/dev/null || { echo "the server exited:"; tail -20 "$LOGS/$log"; return 1; }
    sleep 2
  done
  echo "the server did not come up in 20 minutes"
  return 1
}
stop() { kill "$SERVER" 2>/dev/null; wait "$SERVER" 2>/dev/null; }

if serve serve.log "$@"; then
  pass "serve starts"
  grep -E "^blaze: (VRAM|KV|speculation|MoE|vision|tuned|MTP|DFlash|EAGLE)" "$LOGS/serve.log" | head -12
  if [ -n "${PY:-}" ]; then
    run api "$PY" scripts/api_test.py --base-url "$URL"
    grep -E "passed|FAIL" "$LOGS/api.log" | tail -4
  else
    fail "api (no python with the openai package: set PY)"
  fi
  run prefix-cache python3 scripts/prefix_cache_test.py "$URL" --turns 6 --side 2
  if [ "${SOAK_MIN:-3}" != 0 ]; then
    run soak python3 scripts/soak_test.py "$URL" --clients 4 --minutes "${SOAK_MIN:-3}" --long-tokens 8000
  fi
  stop
else
  fail "serve starts (log: $LOGS/serve.log)"
fi

# a pool far smaller than the sessions want: the scheduler ends requests early instead of failing them (the serve
# options without their own --kv-tokens / --max-seqs)
SMALL=()
skip=0
for a in "$@"; do
  if [ "$skip" = 1 ]; then skip=0; continue; fi
  case "$a" in
    --kv-tokens|--max-seqs) skip=1 ;;
    --kv-tokens=*|--max-seqs=*) ;;
    *) SMALL+=("$a") ;;
  esac
done
if serve serve_small_kv.log "${SMALL[@]}" --kv-tokens 8192 --max-seqs 4; then
  run kv-pressure python3 scripts/kv_pressure_test.py "$URL" --sessions 4 --max-tokens 4000
  stop
else
  fail "serve with a small KV pool (log: $LOGS/serve_small_kv.log)"
fi

echo
echo "== server logs: panics, failed steps, slow steps, server errors"
if grep -hE "panicked|step failed|slow step|blaze api: 5|grammar dropped" "$LOGS"/serve*.log | sort | uniq -c | sort -rn | head -20 | grep .; then
  fail "server log clean"
else
  pass "server log clean"
fi
grep -h "^blaze stats" "$LOGS"/serve.log | tail -3

echo
echo "== summary ($LOGS)"
printf '%s\n' "${RESULTS[@]}"
if printf '%s\n' "${RESULTS[@]}" | grep -q '^FAIL'; then exit 1; fi

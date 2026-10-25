#!/usr/bin/env bash
# One test run of a server configuration, everything in one file: starts the command (a pilot script, or `blaze serve`
# with its own --port) on PORT, runs the chosen tests, writes the header (time, commit, GPUs, the command), every test's
# output, the server's notable lines and a summary of the result lines to $LOGS/run-<time>-<name>.txt, stops the server.
#   scripts/testrun.sh NAME COMMAND...    e.g. scripts/testrun.sh fn-pilot ~/serve_blaze_flash_next.sh
#                                              scripts/testrun.sh 27b-tp target/release/blaze serve --model ... --port 8100
# A pilot script takes PORT from the environment. The GPUs are taken through /tmp/blaze-gpu.lock for the whole run.
# Env: PORT (8100), LOGS (~/qrt/logs), TESTS ("api conc spec needle"), SESSIONS ("1 2 4"), NEEDLES ("122000"; the
#      prompt comes out ~2% longer), API_K (an api_test.py -k filter), PY (a python with the openai package), CLIENTS
#      (the directory of concurrent_test.py: ~/qrt/logs, else ~/Projects/server-llm-runtime/scripts).
set -uo pipefail
if [ -z "${BLAZE_LOCKED:-}" ]; then
  exec env BLAZE_LOCKED=1 flock /tmp/blaze-gpu.lock "$0" "$@"
fi
NAME="${1:?usage: scripts/testrun.sh NAME COMMAND...}"
shift
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
export PORT="${PORT:-8100}"
URL="http://127.0.0.1:$PORT"
LOGS="${LOGS:-${XDG_STATE_HOME:-$HOME/.local/state}/blaze/tests}"
mkdir -p "$LOGS"
OUT="$LOGS/run-$(date +%Y%m%d-%H%M%S)-$NAME.txt"
SRV="${OUT%.txt}.server.log"
TESTS="${TESTS:-api conc spec needle}"
SESSIONS="${SESSIONS:-1 2 4}"
NEEDLES="${NEEDLES:-122000}"
if [ -z "${PY:-}" ]; then
  for p in "$ROOT/.venv-test/bin/python" "$ROOT/.venv/bin/python" python3; do
    if "$p" -c "import openai" 2>/dev/null; then PY="$p"; break; fi
  done
fi
if [ -z "${CLIENTS:-}" ]; then
  for d in "$HOME/qrt/logs" "$HOME/Projects/server-llm-runtime/scripts"; do
    [ -f "$d/concurrent_test.py" ] && CLIENTS="$d" && break
  done
fi
SUMMARY=()
log() { echo "$@" | tee -a "$OUT"; }
keep() { while IFS= read -r l; do echo "$l" | tee -a "$OUT"; SUMMARY+=("$1: $l"); done; }

{
  echo "# test run $NAME, $(date -Is)"
  echo "commit $(git -C "$ROOT" rev-parse --short HEAD) ($(git -C "$ROOT" log -1 --format=%s | cut -c1-80))"
  nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv,noheader 2>/dev/null | sed 's/^/gpu /'
  echo "command: PORT=$PORT $*"
  echo "tests: $TESTS; sessions: $SESSIONS; needles: $NEEDLES"
} | tee "$OUT"

"$@" > "$SRV" 2>&1 &
pid=$!
t0=$(date +%s)
until curl -s "$URL/health" > /dev/null; do
  if ! kill -0 "$pid" 2>/dev/null; then
    log "## the server exited while loading:"
    tail -15 "$SRV" | tee -a "$OUT"
    exit 1
  fi
  sleep 2
done
log "## loaded in $(( $(date +%s) - t0 )) s"
keep server < <(grep -E "KV cache|KV in host|speculation:|NVFP4 layers on many|layers x .* experts|MTP head|pipeline of|vision tower|VRAM \(GiB\)" "$SRV" | cut -c1-240)

for t in $TESTS; do
  case "$t" in
    api)
      log "## api"
      keep api < <($PY "$ROOT/scripts/api_test.py" --base-url "$URL" ${API_K:+-k "$API_K"} 2>&1 | grep -E "FAIL|passed" | cut -c1-400) ;;
    conc)
      log "## concurrent sessions"
      for s in $SESSIONS; do
        keep conc < <(cd "$CLIENTS" && timeout 1200 python3 concurrent_test.py --url "$URL" --sessions "$s" 2>&1 | tail -1)
      done ;;
    spec)
      log "## speculation bench (chars/s)"
      keep spec < <(python3 "$ROOT/scripts/spec_bench.py" "$URL" "${OUT%.txt}.spec" 2>&1 | tail -3) ;;
    needle)
      log "## needles"
      for n in $NEEDLES; do
        keep needle < <(timeout 3600 python3 "$ROOT/scripts/needle_test.py" "$n" --url "$URL" 2>&1 | tail -1)
      done ;;
    *) log "## unknown test $t" ;;
  esac
done

log "## server log: panics, failed and slow steps, slow forwards, hangs, server errors"
keep server-log < <(grep -E "panicked|step failed|slow step|slow forward|blaze hang|blaze api: 5" "$SRV" | cut -c1-300 | head -20)
kill "$pid" 2>/dev/null
wait "$pid" 2>/dev/null

log ""
log "## summary"
for s in "${SUMMARY[@]}"; do log "  $s"; done
log "(the whole server log: $SRV)"

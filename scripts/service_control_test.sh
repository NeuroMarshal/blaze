#!/usr/bin/env bash
set -euo pipefail
# Exercises a real GPU server with intentionally short deadlines and small admission/body limits.
# scripts/service_control_test.sh target/release/server --config configs/27b.json
if [ -z "${BLAZE_LOCKED:-}" ]; then
  exec env BLAZE_LOCKED=1 flock /tmp/blaze-gpu.lock "$0" "$@"
fi
if [ "$#" = 0 ]; then echo "usage: $0 SERVER [ARGS...]" >&2; exit 2; fi
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
A="${LOGS:-$(mktemp -d /tmp/blaze-controls.XXXXXX)}"
mkdir -p "$A"
export A
PORT="${PORT:-8101}"
export BLAZE_API_KEY="$(cat /proc/sys/kernel/random/uuid)"
url="http://127.0.0.1:$PORT"
pid=''
# An exited child may still be a zombie until wait; kill -0 alone does not detect completion.
running() {
  local state
  state=$(ps -o stat= -p "$1" 2>/dev/null) || return 1
  [[ -n "$state" && "$state" != *Z* ]]
}
cleanup() {
 if [ -n "$pid" ]; then
  kill "$pid" 2>/dev/null || true
  for _ in $(seq 1 20); do running "$pid" || break; sleep 1; done
  if running "$pid"; then kill -KILL "$pid" 2>/dev/null || true; fi
  wait "$pid" 2>/dev/null || true
 fi
}
trap cleanup EXIT
if curl -fsS --max-time 2 "$url/health" >/dev/null 2>&1; then
 echo "FAIL: $url is already serving" >&2; exit 1
fi
"$@" --host 127.0.0.1 --port "$PORT" --max-model-len 32768 --spec-method none \
 --max-queued-requests 1 --max-concurrent-requests 2 --request-timeout 0.4 --shutdown-timeout 1 \
 --max-request-body-mib 1 --log-format json --log-level debug --log-inputs --log-outputs --max-log-len 100 \
 --allowed-origins http://localhost:3000 > "$A/server.jsonl" 2>&1 &
pid=$!
for _ in $(seq 1 120); do
 if curl -fsS --max-time 1 "$url/health" >/dev/null 2>&1; then break; fi
 running "$pid" || exit 1
 sleep 1
done
expect() { local expected="$1" name="$2"; shift 2; local status; status=$(curl -sS --max-time 15 -D "$A/$name.headers" -o "$A/$name.body" -w '%{http_code}' "$@"); test "$status" = "$expected" || { echo "FAIL $name HTTP=$status expected=$expected"; cat "$A/$name.body"; exit 1; }; echo "PASS $name HTTP=$status"; }
expect 200 health "$url/health"
expect 401 missing-key "$url/v1/models"
expect 401 metrics-auth "$url/metrics"
expect 401 wrong-key -H 'Authorization: Bearer wrong' "$url/server_info"
expect 200 bearer -H "Authorization: Bearer $BLAZE_API_KEY" -H 'X-Request-Id: ops-trace-42' "$url/server_info"
rg -iq '^x-request-id: ops-trace-42' "$A/bearer.headers"
jq -e '.api_key_count == 1 and .config["api-key"] == "[REDACTED]"' "$A/bearer.body" >/dev/null
expect 200 anthropic-key -H "x-api-key: $BLAZE_API_KEY" "$url/v1/models"
expect 204 cors-allowed -X OPTIONS -H 'Origin: http://localhost:3000' -H 'Access-Control-Request-Method: POST' "$url/v1/completions"
rg -iq '^access-control-allow-origin: http://localhost:3000' "$A/cors-allowed.headers"
expect 204 cors-denied -X OPTIONS -H 'Origin: http://evil.invalid' "$url/v1/completions"
if rg -iq '^access-control-allow-origin:' "$A/cors-denied.headers"; then exit 1; fi
expect 400 invalid-sampling -H "Authorization: Bearer $BLAZE_API_KEY" -H 'Content-Type: application/json' --data '{"model":"default","prompt":"Hello","temperature":"cold"}' "$url/v1/completions"
{ printf '{"prompt":"'; head -c 1100000 /dev/zero | tr '\0' ' '; printf '"}'; } > "$A/large.json"
expect 413 body-limit -H "Authorization: Bearer $BLAZE_API_KEY" -H 'Content-Type: application/json' --data-binary "@$A/large.json" "$url/v1/completions"
before=$(curl -fsS -H "Authorization: Bearer $BLAZE_API_KEY" "$url/metrics" | awk '$1=="blaze_http_timeouts_total" {print $2}')
expect 504 timeout-before-headers -H "Authorization: Bearer $BLAZE_API_KEY" -H 'Content-Type: application/json' \
 --data '{"prompt":"Write fifty long chapters about science.","max_tokens":8192,"min_tokens":8192,"ignore_eos":true,"stream":false}' "$url/v1/completions"
jq -e '.error.message == "request deadline exceeded"' "$A/timeout-before-headers.body" >/dev/null
after=$(curl -fsS -H "Authorization: Bearer $BLAZE_API_KEY" "$url/metrics" | awk '$1=="blaze_http_timeouts_total" {print $2}')
test "$((after - before))" = 1 || { echo 'FAIL handler timeout counted more than once'; exit 1; }
echo 'PASS handler timeout preserves JSON and increments metric once'
sleep 1
for api in completions messages responses; do
 case "$api" in
  completions) body=$(jq -nc '{model:"default",prompt:(("Scientific institutions coordinate peer review and long-term research. " * 100) + "Write fifty detailed chapters:"),max_tokens:8192,min_tokens:8192,ignore_eos:true,stream:true}') ;;
  messages) body='{"model":"default","messages":[{"role":"user","content":"Write a very long story with fifty chapters about an interstellar expedition. Do not stop early."}],"max_tokens":8192,"stream":true}' ;;
  responses) body='{"model":"default","input":"Write a very long story with fifty chapters about an interstellar expedition. Do not stop early.","max_output_tokens":8192,"stream":true}' ;;
 esac
 expect 200 "timeout-$api" -H "Authorization: Bearer $BLAZE_API_KEY" -H 'Content-Type: application/json' --data "$body" "$url/v1/$api"
 rg -q 'request deadline exceeded' "$A/timeout-$api.body"
 sleep 1
done
rg -q 'data: \[DONE\]' "$A/timeout-completions.body"
if rg -q '\[DONE\]' "$A/timeout-messages.body" "$A/timeout-responses.body"; then exit 1; fi
python3 - <<'PY'
import json
from pathlib import Path
import os
p=Path(os.environ['A'])
for api in ('messages','responses'):
 events=[json.loads(l[6:]) for l in (p/f'timeout-{api}.body').read_text().splitlines() if l.startswith('data: ')]
 assert events[-1]['type']=='error',events[-1]
 if api=='responses':
  seq=[e['sequence_number'] for e in events]
  assert seq==sorted(set(seq)),seq
 print('PASS',api,'timeout event framing')
PY
curl -sS --max-time 10 -H "Authorization: Bearer $BLAZE_API_KEY" -H 'Content-Type: application/json' --data '{"prompt":"Write a very long article about the history of science.","max_tokens":8192,"min_tokens":8192,"stream":true,"ignore_eos":true}' "$url/v1/completions" > "$A/queue-first.body" &
first=$!
for _ in $(seq 1 20); do
 running=$(curl -sS -H "Authorization: Bearer $BLAZE_API_KEY" "$url/server_info" | jq -r '.scheduler.outstanding')
 [ "$running" = 1 ] && break
 sleep .01
done
expect 429 bounded-queue -H "Authorization: Bearer $BLAZE_API_KEY" -H 'Content-Type: application/json' --data '{"prompt":"Hello","max_tokens":5}' "$url/v1/completions"
rg -iq '^retry-after: 1' "$A/bounded-queue.headers"
wait "$first"
sleep 1
expect 200 quiescent -H "Authorization: Bearer $BLAZE_API_KEY" "$url/server_info"
jq -e '.scheduler.outstanding == 0 and .scheduler.running == 0' "$A/quiescent.body" >/dev/null
expect 200 metrics -H "Authorization: Bearer $BLAZE_API_KEY" "$url/metrics"
rg -q 'blaze_http_timeouts_total [3-9]' "$A/metrics.body"
kill "$pid"
for _ in $(seq 1 20); do running "$pid" || break; sleep 1; done
if running "$pid"; then echo 'FAIL shutdown hung'; exit 1; fi
wait "$pid"
pid=''
# Also terminate during active generation: draining and cancellation must not race engine teardown.
"$@" --host 127.0.0.1 --port "$PORT" --max-model-len 32768 --spec-method none \
 --shutdown-timeout 0.1 --log-format json > "$A/shutdown-active.jsonl" 2>&1 &
pid=$!
for _ in $(seq 1 120); do
 curl -fsS --max-time 1 "$url/health" >/dev/null 2>&1 && break
 running "$pid" || exit 1
 sleep 1
done
curl -sS --max-time 15 -H "Authorization: Bearer $BLAZE_API_KEY" -H 'Content-Type: application/json' \
 --data '{"prompt":"Write a long article about science.","max_tokens":8192,"min_tokens":8192,"stream":true,"ignore_eos":true}' \
 "$url/v1/completions" > "$A/shutdown-client.body" 2> "$A/shutdown-client.err" &
client_pid=$!
for _ in $(seq 1 200); do
 running=$(curl -sS -H "Authorization: Bearer $BLAZE_API_KEY" "$url/server_info" | jq -r '.scheduler.outstanding')
 [ "$running" = 1 ] && break
 sleep .01
done
test "$running" = 1
kill "$pid"
for _ in $(seq 1 20); do running "$pid" || break; sleep 1; done
if running "$pid"; then echo 'FAIL active shutdown hung'; exit 1; fi
wait "$pid"
pid=''
wait "$client_pid" || true
echo 'PASS shutdown during active generation'
python3 - <<'PY'
import json
from pathlib import Path
import os
p=Path(os.environ['A'])/'server.jsonl'
rows=[json.loads(l) for l in p.read_text().splitlines()]
assert any(r['event']=='gpu.prefill' for r in rows)
assert any(r['event']=='scheduler.step' for r in rows)
assert any(r['event']=='server.shutdown' for r in rows)
print('PASS structured JSON, GPU phases, per-step debug and shutdown',len(rows),'events')
PY
if rg -Fq "$BLAZE_API_KEY" "$A/server.jsonl"; then echo 'FAIL API key leaked'; exit 1; fi
echo 'PASS authentication, CORS, body limit, protocol deadlines, bounded queue, cancellation, debug logs and key redaction'

echo "artifacts=$A"

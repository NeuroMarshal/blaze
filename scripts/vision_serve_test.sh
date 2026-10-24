#!/usr/bin/env bash
# One run of the image-input test: start `blaze serve` on a free port, run scripts/vision_test.py against it, stop the
# server. For the machine's shared-GPU lock use `flock /tmp/blaze-gpu.lock scripts/vision_serve_test.sh <model> [serve options]`.
#   MODEL IMAGES_DIR  (extra arguments go to `blaze serve`; VT_ARGS to the test client)
set -uo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
MODEL="$1"; IMAGES="$2"; shift 2
PORT="${PORT:-8103}"
LOG="${LOG:-/tmp/vision_serve.log}"
"$ROOT/target/release/blaze" serve --model "$MODEL" --port "$PORT" --host 127.0.0.1 --prompt-cache-slots 4 \
  --limit-mm-per-prompt image=3 "$@" > "$LOG" 2>&1 &
SP=$!
trap 'kill $SP 2>/dev/null; wait $SP 2>/dev/null' EXIT
for _ in $(seq 1 300); do
  curl -s "localhost:$PORT/health" > /dev/null && break
  kill -0 $SP 2>/dev/null || { echo "server exited:"; tail -20 "$LOG"; exit 1; }
  sleep 1
done
python3 "$ROOT/scripts/vision_test.py" "http://127.0.0.1:$PORT" --images "$IMAGES" --limit 3 --cache ${VT_ARGS:-}

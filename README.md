# Blaze

Developed with extensive AI assistance. The code and its tests are the reference.

Blaze is an inference runtime I built for my own workloads and hardware, using Rust and C++/CUDA.
It currently focuses on quantized Qwen models.
Reads local safetensors checkpoints with BF16, FP8, MXFP8 and NVFP4 weights.
Includes tensor/pipeline parallelism, prefix caching, speculative decoding and an HTTP API.

It may not fit your setup. You may need to add and test support for your GPU or model yourself.

Linux x86-64, AVX2/FMA, Rust (edition 2024), CMake 3.28+ and a CUDA toolkit supporting the target GPU
are required. Development hardware: RTX 5070 Ti (SM120) and RTX 4060 (SM89).

In my own comparisons on the 2 × RTX 5070 Ti server, Blaze ran faster than vLLM, SGLang and llama.cpp
for my workloads. These results apply to the tested models, settings and hardware.

## Build

```sh
export CUDA_HOME=/usr/local/cuda
export BLAZE_CUDA_ARCH=120a       # 89 for SM89; unset for native detection
export CMAKE_BUILD_PARALLEL_LEVEL=8
cargo build --release --locked -p blaze --bins
```

## Run

```sh
MODEL=/path/to/checkpoint
./target/release/server --model "$MODEL" --tp 2 --host 127.0.0.1 --port 8000 --vision off
```

Use `--tp 1` for one GPU. `server` and `blaze serve` share the same implementation.
`server --help` lists serving options; `blaze --help` lists inspection, generation and scoring commands.

- `--draft-model PATH`: EAGLE-4 or DFlash2 checkpoint; `--spec-method mtp`: the target's MTP head.
- `--kv-tokens auto`: size KV from available VRAM. `--kv-host`: keep KV in pinned RAM.
- `--gemm`, `--kv-cache-dtype`, `--kv-rotate`: compute and cache precision choices.
- `--config FILE`: flat JSON configuration; explicit flags take precedence.
- `--log-format json`: structured logs. `/health`, `/server_info` and `/metrics` expose runtime state.

HTTP endpoints include chat/completions, Responses, Messages, embeddings and tokenization.
Discover the model id at `/v1/models`. Configure `BLAZE_API_KEY` or `--api-key-file` for authentication;
without keys, the service is unauthenticated.

The sampler uses at most the best 64 candidates. Quantized compute, FP16 accumulation, TQ3 KV and
expert deferral can change outputs. Pipeline stages currently run without a drafter. Check the intended
model/configuration before relying on it.

## Check

Run on the target GPU server:

```sh
cargo test --release --locked --workspace
cmake -S kernels -B kernels/build -DCMAKE_CUDA_ARCHITECTURES=120a -DBLAZE_BUILD_TESTS=ON
cmake --build kernels/build -j 8
flock /tmp/blaze-gpu.lock ctest --test-dir kernels/build --output-on-failure
```

External API checks need a Python environment with `openai` installed:

```sh
PY=/path/to/python scripts/check.sh "$MODEL" --tp 2 --vision off
```

`blaze-bench` measures requests against a running server. `scripts` also contains compatibility clients,
reference computations, prefix-cache, KV-pressure and soak tests.

`crates/blaze` contains the host/server; `crates/blaze-sys` provides the native C ABI;
`kernels` contains native implementations and tests. `crates/blaze-chat` is a vendored chat frontend.

Blaze-authored code uses Apache-2.0; see [LICENSE](LICENSE). The bundled chat frontend retains its
[upstream license](crates/blaze-chat/LICENSE). Checkpoints and dependencies retain their own licenses.

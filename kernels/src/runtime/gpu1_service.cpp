#include "blaze/gpu1_service.h"
#include "blaze/log.h"

#include <immintrin.h>

#include <chrono>
#include <cstdio>
#include <cstdlib>

namespace blaze {

namespace {

void check(cudaError_t e, const char* what) {
  if (e != cudaSuccess) {
    log("blaze GPU1 tier: %s: %s", what, cudaGetErrorString(e));
    std::abort();
  }
}

// Token buckets match the rows kernels' column specializations.
int bucket(int tokens) { return tokens <= 1 ? 1 : tokens <= 2 ? 2 : tokens <= 4 ? 4 : 8; }

}  // namespace

Gpu1Service::Gpu1Service(int device, const blz_moe_layer& layer, uint8_t* mail_host)
    : device_(device), layer_(layer), mail_host_(mail_host) {
  check(cudaSetDevice(device_), "set device");
  check(cudaHostGetDevicePointer(reinterpret_cast<void**>(&mail_dev_), mail_host_, 0), "mailbox device pointer");
  check(cudaMalloc(&ws_mem_, blz_moe_workspace_bytes(layer_.hidden, layer_.inter)), "workspace");
  blz_moe_workspace_carve(ws_mem_, layer_.hidden, layer_.inter, &ws_);
  check(cudaMalloc(&counters_, 8), "counters");
  check(cudaMemset(counters_, 0, 8), "counters");
  ws_.blocks_done = counters_;
  check(cudaStreamCreateWithFlags(&stream_, cudaStreamNonBlocking), "stream");
  // Build every bucket's graph now: capturing later, while GPU0's wait kernel spins for this service, would make the
  // first launch load modules lazily, and module loading waits for the device to idle (a deadlock).
  for (int b : {1, 2, 4, 8}) graph_for(b);
  check(cudaStreamSynchronize(stream_), "warm-up");
  thread_ = std::thread([this] { run(); });
}

Gpu1Service::~Gpu1Service() {
  stop_.store(true, std::memory_order_release);
  thread_.join();
  cudaSetDevice(device_);
  cudaStreamSynchronize(stream_);
  for (auto& g : graphs_)
    if (g) cudaGraphExecDestroy(g);
  cudaStreamDestroy(stream_);
  cudaFree(ws_mem_);
  cudaFree(counters_);
}

cudaGraphExec_t Gpu1Service::graph_for(int tokens) {
  const int b = bucket(tokens);
  if (!graphs_[b]) {
    cudaGraph_t g;
    check(cudaStreamBeginCapture(stream_, cudaStreamCaptureModeThreadLocal), "capture");
    check(blz_moe_gpu1_route(&layer_, &ws_, mail_dev_, b, stream_), "gpu1 route");
    check(cudaStreamEndCapture(stream_, &g), "end capture");
    check(cudaGraphInstantiate(&graphs_[b], g, 0), "instantiate");
    cudaGraphDestroy(g);
  }
  return graphs_[b];
}

void Gpu1Service::run() {
  // The host's model may capture CUDA graphs in global mode meanwhile; this thread's launches must not invalidate them.
  cudaStreamCaptureMode mode = cudaStreamCaptureModeRelaxed;
  cudaThreadExchangeStreamCaptureMode(&mode);
  check(cudaSetDevice(device_), "set device");
  auto* hdr = reinterpret_cast<blz_mail_header*>(mail_host_);
  uint32_t last = blz_req_seq(__atomic_load_n(&hdr->req, __ATOMIC_ACQUIRE));
  auto last_work = std::chrono::steady_clock::now();
  while (!stop_.load(std::memory_order_acquire)) {
    const uint64_t req = __atomic_load_n(&hdr->req, __ATOMIC_ACQUIRE);
    const uint32_t seq = blz_req_seq(req);
    if (seq != last) {
      last = seq;
      if (blz_req_gpu1(req) > 0) {
        // The graph imports the items (and seq) from the mailbox itself; the bucket only bounds the columns.
        check(cudaGraphLaunch(graph_for(hdr->tokens), stream_), "launch");
        served_.fetch_add(1, std::memory_order_relaxed);
      }
      last_work = std::chrono::steady_clock::now();
    } else if (std::chrono::steady_clock::now() - last_work < std::chrono::milliseconds(50)) {
      _mm_pause();
    } else {
      std::this_thread::sleep_for(std::chrono::microseconds(100));
    }
  }
}

}  // namespace blaze

// bench_ssd_single.cu
// SSD benchmark scheduling on one GPU with blocking host-memory queues.

#include <algorithm>
#include <chrono>
#include <condition_variable>
#include <cstdlib>
#include <cstdio>
#include <mutex>
#include <random>
#include <thread>
#include <vector>

#include <cublas_v2.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include "bench_ssd_single.h"
#include "check.h"
#include "forward_single.h"
#include "gpu_context.h"
#include "thread_safe_queue.h"

namespace {

struct SpecMessage {
  std::vector<half> tokens;
};

struct OutcomeMessage {
  int accepted;
  int bonus_id;
  bool done;
};

class StartGate {
 public:
  void arrive_and_wait() {
    std::unique_lock<std::mutex> lock(mutex_);
    if (++arrived_ == 2) {
      open_ = true;
      cv_.notify_all();
      return;
    }
    cv_.wait(lock, [this] { return open_; });
  }

 private:
  std::mutex mutex_;
  std::condition_variable cv_;
  int arrived_ = 0;
  bool open_ = false;
};

class CudaStream {
 public:
  CudaStream() { CHECK_CUDA(cudaStreamCreate(&stream_)); }
  ~CudaStream() { cudaStreamDestroy(stream_); }

  cudaStream_t get() const { return stream_; }

 private:
  cudaStream_t stream_ = nullptr;
};

class CublasHandle {
 public:
  explicit CublasHandle(cudaStream_t stream) {
    CHECK_CUBLAS(cublasCreate(&handle_));
    CHECK_CUBLAS(cublasSetMathMode(handle_, CUBLAS_TENSOR_OP_MATH));
    CHECK_CUBLAS(cublasSetStream(handle_, stream));
  }
  ~CublasHandle() { cublasDestroy(handle_); }

  cublasHandle_t get() const { return handle_; }

 private:
  cublasHandle_t handle_ = nullptr;
};

class CudaEvent {
 public:
  CudaEvent() { CHECK_CUDA(cudaEventCreate(&event_)); }
  ~CudaEvent() { cudaEventDestroy(event_); }

  void record(cudaStream_t stream) { CHECK_CUDA(cudaEventRecord(event_, stream)); }
  void synchronize() { CHECK_CUDA(cudaEventSynchronize(event_)); }
  float elapsed_ms_since(const CudaEvent& start) const {
    float elapsed_ms = 0.0f;
    CHECK_CUDA(cudaEventElapsedTime(&elapsed_ms, start.event_, event_));
    return elapsed_ms;
  }

 private:
  cudaEvent_t event_ = nullptr;
};

class ModelState {
 public:
  ModelState(const ModelConfig& cfg, int M) : cfg_(cfg) {
    weights_.resize(cfg_.n_layers);
    for (int l = 0; l < cfg_.n_layers; ++l)
      alloc_layer_weights(weights_[l], cfg_);
    buffers_ = alloc_activations(cfg_, M);
  }

  ~ModelState() {
    free_activations(buffers_);
    for (auto& weight : weights_)
      free_layer_weights(weight);
  }

  void forward(cublasHandle_t handle, int M) {
    forward_model(handle, cfg_, weights_.data(), M,
                  buffers_.x, buffers_.scratch1, buffers_.scratch2);
  }

 private:
  const ModelConfig& cfg_;
  std::vector<LayerWeights> weights_;
  ActivationBuffers buffers_ = {nullptr, nullptr, nullptr};
};

static void run_draft_steps(
  ModelState& state,
  cublasHandle_t handle,
  int K,
  int M
) {
  for (int k = 0; k < K; ++k)
    state.forward(handle, M);
}

static SpecMessage make_spec_message(int K) {
  // bench_ssd's NCCL payload is intentionally synthetic, but retain the
  // same K-FP16 transport shape when replacing it with shared memory.
  return {std::vector<half>(K, __float2half(1.0f))};
}

static float bench_single_pass(
  const ModelConfig& cfg,
  int M,
  int warmup_iters,
  int bench_iters
) {
  CHECK_CUDA(cudaSetDevice(0));
  CudaStream stream;
  CublasHandle handle(stream.get());
  ModelState state(cfg, M);

  for (int i = 0; i < warmup_iters; ++i)
    state.forward(handle.get(), M);
  CHECK_CUDA(cudaStreamSynchronize(stream.get()));

  CudaEvent start;
  CudaEvent stop;
  start.record(stream.get());
  for (int i = 0; i < bench_iters; ++i)
    state.forward(handle.get(), M);
  stop.record(stream.get());
  stop.synchronize();
  return stop.elapsed_ms_since(start) / bench_iters;
}

static bool models_fit_together(
  const ModelConfig& target_cfg,
  const ModelConfig& draft_cfg
) {
  size_t free_mem = get_free_gpu_memory(0);
  size_t required = target_cfg.total_weight_bytes() + draft_cfg.total_weight_bytes();
  return required < static_cast<size_t>(free_mem * 0.85);
}

static BenchResult bench_ssd_e2e_single(
  const ModelConfig& target_cfg,
  const ModelConfig& draft_cfg,
  int K,
  int F,
  int N,
  float alpha,
  float phit,
  int warmup_iters,
  int bench_iters
) {
  ThreadSafeQueue<SpecMessage> draft_to_target;
  ThreadSafeQueue<OutcomeMessage> target_to_draft;
  StartGate start_gate;

  int total_tokens = 0;
  int total_rounds = 0;
  int total_misses = 0;
  float target_elapsed_ms = 0.0f;
  const int F_eff = F > 0 ? F : 1;

  std::thread draft_worker([&] {
    CHECK_CUDA(cudaSetDevice(0));
    CudaStream stream;
    CublasHandle handle(stream.get());
    ModelState fanout_state(draft_cfg, F_eff);
    ModelState step_state(draft_cfg, 1);
    CudaEvent draft_ready;
    std::mt19937 rng(1337);
    std::bernoulli_distribution hit_dist(phit);

    for (int i = 0; i < warmup_iters; ++i) {
      run_draft_steps(step_state, handle.get(), K, 1);
      run_draft_steps(fanout_state, handle.get(), K, F_eff);
    }
    CHECK_CUDA(cudaStreamSynchronize(stream.get()));
    start_gate.arrive_and_wait();

    int local_misses = 0;
    for (int iter = 0; iter < bench_iters; ++iter) {
      run_draft_steps(step_state, handle.get(), K, 1);
      draft_ready.record(stream.get());
      draft_ready.synchronize();
      draft_to_target.push(make_spec_message(K));

      while (true) {
        // This work runs while pop() blocks for target verification.
        run_draft_steps(fanout_state, handle.get(), K, F_eff);
        draft_ready.record(stream.get());

        OutcomeMessage outcome = target_to_draft.pop();
        // draft_ready.synchronize() is called BEFORE checking outcome.done,
        // so GPU work is always fully retired before we act on the result.
        draft_ready.synchronize();
        if (outcome.done)
          break;

        if (!hit_dist(rng)) {
          ++local_misses;
          run_draft_steps(step_state, handle.get(), K, 1);
          draft_ready.record(stream.get());
          draft_ready.synchronize();
        }
        draft_to_target.push(make_spec_message(K));
      }
    }
    // Issue #2/#3: final stream sync ensures no GPU kernels are orphaned
    // after the last iteration's speculative fanout work completes.
    CHECK_CUDA(cudaStreamSynchronize(stream.get()));
    total_misses = local_misses;
  });

  std::thread target_worker([&] {
    CHECK_CUDA(cudaSetDevice(0));
    CudaStream stream;
    CublasHandle handle(stream.get());
    ModelState target_state(target_cfg, K);
    CudaEvent verify_done;
    std::mt19937 rng(42);
    std::bernoulli_distribution accept_dist(alpha);

    for (int i = 0; i < warmup_iters; ++i)
      target_state.forward(handle.get(), K);
    CHECK_CUDA(cudaStreamSynchronize(stream.get()));
    start_gate.arrive_and_wait();

    const auto start = std::chrono::steady_clock::now();
    int local_tokens_total = 0;
    int local_rounds_total = 0;
    for (int iter = 0; iter < bench_iters; ++iter) {
      int local_tokens = 0;
      while (true) {
        SpecMessage spec = draft_to_target.pop();
        (void)spec;
        target_state.forward(handle.get(), K);
        verify_done.record(stream.get());
        verify_done.synchronize();

        int accepted = 0;
        for (int k = 0; k < K; ++k) {
          if (accept_dist(rng))
            ++accepted;
          else
            break;
        }
        local_tokens += accepted + 1;
        ++local_rounds_total;

        const bool done = local_tokens >= N;
        target_to_draft.push({accepted, 0, done});
        if (done) {
          local_tokens_total += local_tokens;
          break;
        }
      }
    }
    const auto stop = std::chrono::steady_clock::now();
    target_elapsed_ms = std::chrono::duration<float, std::milli>(stop - start).count();
    total_tokens = local_tokens_total;
    total_rounds = local_rounds_total;
  });

  // Issue #3: Outer wall-clock timing that covers BOTH thread joins.
  // This captures any draft-thread tail work that completes after the target
  // thread records its stop timestamp, giving a true end-to-end measurement.
  // The target-thread-only elapsed time is retained below for comparison.
  const auto e2e_wall_start = std::chrono::steady_clock::now();
  draft_worker.join();
  target_worker.join();
  const auto e2e_wall_stop = std::chrono::steady_clock::now();
  const float e2e_wall_ms =
    std::chrono::duration<float, std::milli>(e2e_wall_stop - e2e_wall_start).count();

  // target_elapsed_ms: time measured inside the target thread only (inner).
  // e2e_wall_ms:       wall-clock from just before join() to after both
  //                    threads have returned (outer, includes draft tail).
  // We report the larger of the two as the canonical e2e latency per iter,
  // guaranteeing we never under-report when draft tail work dominates.
  BenchResult br;
  br.e2e_ms = std::max(target_elapsed_ms, e2e_wall_ms) / bench_iters;
  br.avg_tokens = static_cast<float>(total_tokens) / bench_iters;
  br.avg_rounds = static_cast<float>(total_rounds) / bench_iters;
  br.miss_rate = total_rounds > 0
    ? static_cast<float>(total_misses) / total_rounds
    : 0.0f;
  br.throughput_tok_s = br.e2e_ms > 0.0f
    ? br.avg_tokens * 1000.0f / br.e2e_ms
    : 0.0f;
  return br;
}

}  // namespace

void run_ssd_single_benchmark(
  const ModelConfig& target_cfg,
  const ModelConfig& draft_cfg,
  int K,
  int F,
  int N,
  float alpha,
  float phit,
  int warmup_iters,
  int bench_iters,
  const BenchOpts& opts
) {
  const bool target_fits = model_fits(target_cfg, 1);
  const bool draft_fits = model_fits(draft_cfg, 1);

  BenchInfo info;
  info.mode = "ssd-single";
  info.target = &target_cfg;
  info.draft = &draft_cfg;
  info.K = K;
  info.F = F;
  info.N = N;
  info.alpha = alpha;
  info.phit = phit;
  info.warmup = warmup_iters;
  info.iters = bench_iters;
  info.target_fits = target_fits;
  info.draft_fits = draft_fits;
  print_bench_header(info);

  if (!target_fits || !draft_fits || !models_fit_together(target_cfg, draft_cfg)) {
    fprintf(stderr,
      "Error: ssd-single requires target and draft weights to fit together on GPU 0.\n");
    exit(1);
  }

  ComponentTimes ct;
  if (!opts.skip_component) {
    const int F_eff = F > 0 ? F : 1;
    ct.draft_step_ms = bench_single_pass(draft_cfg, 1, warmup_iters, bench_iters);
    const float draft_fanout_ms = F_eff > 1
      ? bench_single_pass(draft_cfg, F_eff, warmup_iters, bench_iters)
      : ct.draft_step_ms;
    ct.draft_total_ms = ct.draft_step_ms * K;
    ct.prespec_ms = draft_fanout_ms * K;
    ct.target_verify_ms = bench_single_pass(target_cfg, K, warmup_iters, bench_iters);
  }
  print_components(info, ct);

  BenchResult br;
  if (!opts.skip_e2e) {
    br = bench_ssd_e2e_single(target_cfg, draft_cfg, K, F, N, alpha, phit,
                              warmup_iters, bench_iters);
  }
  print_result(info, br);
  if (opts.csv) print_csv_row(info, ct, br);
}

# Experiment Methodology

This document describes the synthetic workload design, model configurations,
warmup/measurement protocol, timing instrumentation, and CSV schema used in
the Async-SSD benchmark.

---

## 1. Synthetic Workload

Each "forward pass" simulates one transformer decoder-layer stack using
**cuBLAS GEMM operations on FP16 weights**.  There is no attention softmax,
no KV cache, no embedding lookup, and no actual weight loading from real
model checkpoints.

### Per-Layer GEMMs (7 total)

`forward_layer` in `src/forward_single.cu` issues the following cuBLAS
`hgemm` calls for an input activation matrix of shape `[M, d_model]`:

| # | Operation | Shapes | Description |
|---|-----------|--------|-------------|
| 1 | Q-projection | `[M, d_model] × [d_model, d_q]` | Query heads |
| 2 | K-projection | `[M, d_model] × [d_model, d_kv]` | Key heads |
| 3 | V-projection | `[M, d_model] × [d_model, d_kv]` | Value heads |
| 4 | O-projection | `[M, d_q] × [d_q, d_model]` | Output projection |
| 5 | Gate-projection | `[M, d_model] × [d_model, d_ff]` | FFN gate (SwiGLU) |
| 6 | Up-projection | `[M, d_model] × [d_model, d_ff]` | FFN up |
| 7 | Down-projection | `[M, d_ff] × [d_ff, d_model]` | FFN down |

Where:
- `d_q = n_heads × head_dim`
- `d_kv = n_kv_heads × head_dim`
- `M` = batch size (1 for draft steps, K for verify batches, F for fanout)

Weight matrices are allocated with `cudaMalloc` and filled with `0x3C3C`
(≈ 1.0 in FP16) via `cudaMemset`.  This ensures all GEMM operands are
numerically valid (not NaN/Inf) without loading real checkpoint files.

`forward_model` loops `forward_layer` over all `n_layers` layers.
`cublasSetMathMode(CUBLAS_TENSOR_OP_MATH)` enables Tensor Core acceleration
on supported hardware.

---

## 2. Model Configurations

Dimensions are matched to the real Llama-3 / Llama-3.2 architecture
(see `src/config.h`):

| Key | Name | d_model | n_heads | n_kv_heads | head_dim | d_ff | n_layers | Weight (FP16) |
|-----|------|---------|---------|------------|----------|------|----------|----------------|
| `tiny` | Tiny | 512 | 8 | 4 | 64 | 2048 | 4 | ~0.02 GB |
| `llama-1b` | Llama-3.2-1B | 2048 | 32 | 8 | 64 | 8192 | 16 | ~1.0 GB |
| `llama-8b` | Llama-3.1-8B | 4096 | 32 | 8 | 128 | 14336 | 32 | ~8.0 GB |
| `llama-70b` | Llama-3.1-70B | 8192 | 64 | 8 | 128 | 28672 | 80 | ~70 GB |

> [!NOTE]
> Weight bytes per layer = 2 × (d·d_q + d·d_kv + d·d_kv + d_q·d + d·d_ff + d·d_ff + d_ff·d).
> Total = layer_bytes × n_layers.  The 70B model does not fit on a single
> consumer GPU; component timing falls back to single-layer extrapolation.

---

## 3. Measurement Protocol

### Warmup

Before any timed measurement, each benchmark runs `warmup_iters` (default 10)
full iterations to:
- Bring weights into L2 / HBM working set.
- Allow cuBLAS to select optimal GEMM kernels.
- Saturate CUDA driver initialisation overhead.

Warmup is concluded with `cudaStreamSynchronize` (component timing) or
`cudaStreamSynchronize` on both worker streams (SSD e2e).

### Measured Iterations

After warmup, `bench_iters` (default 20) iterations are timed.  All
reported metrics are averages over these iterations.

### Defaults (from `src/main.cu`)

| Parameter | Default | Flag |
|-----------|---------|------|
| warmup iterations | 10 | `--warmup` |
| bench iterations | 20 | `--iters` |
| N (tokens to generate) | 128 | `--N` |
| K (speculation length) | 4 | `--K` |
| F (fanout) | 4 | `--F` |
| alpha (acceptance rate) | 0.9 | `--alpha` |
| phit (cache-hit probability) | 0.85 | `--phit` |

---

## 4. Timing Instrumentation

### Component Timings (GPU Events)

Individual forward-pass timings use CUDA events for GPU-side wall time:

```cpp
CudaEvent start, stop;
start.record(stream.get());
for (int i = 0; i < bench_iters; ++i)
    state.forward(handle.get(), M);
stop.record(stream.get());
stop.synchronize();
float ms_per_iter = stop.elapsed_ms_since(start) / bench_iters;
```

`cudaEventElapsedTime` reports GPU execution time independent of CPU
scheduling jitter.

### End-to-End Timing (Host Clock)

The SSD-single e2e timing uses `std::chrono::steady_clock` in the
`target_worker` thread:

```cpp
const auto start = std::chrono::steady_clock::now();
// ... bench_iters decode loops ...
const auto stop  = std::chrono::steady_clock::now();
target_elapsed_ms = duration<float, milli>(stop - start).count();
e2e_ms = target_elapsed_ms / bench_iters;
```

Host-side timing captures queue-wait latency and thread synchronisation
overhead that GPU events alone cannot see.

### Throughput

```
throughput_tok_s = avg_tokens × 1000 / e2e_ms
```

Where `avg_tokens = total_tokens_across_iters / bench_iters`.

---

## 5. CSV Schema

All result rows start with the tag `RESULT`.  The header can be emitted
with `--csv-header`.

| Column | Type | Description |
|--------|------|-------------|
| `tag` | string | Always `RESULT` |
| `mode` | string | `ar`, `sd`, `ssd`, or `ssd-single` |
| `target` | string | Target model name (e.g. `Llama-3.1-8B`) |
| `draft` | string | Draft model name (empty for AR) |
| `tp` | int | Tensor-parallelism degree |
| `K` | int | Speculation length (0 for AR) |
| `F` | int | Fanout (0 for AR/SD) |
| `N` | int | Tokens-to-generate target |
| `alpha` | float | Token acceptance rate — Bernoulli(α) per draft token |
| `phit` | float | Pre-speculation cache-hit probability — Bernoulli(phit) |
| `target_fits` | int | 1 if full target model fits in GPU VRAM |
| `draft_fits` | int | 1 if full draft model fits in GPU VRAM |
| `target_step_ms` | float | One target forward pass at M=1 |
| `target_verify_ms` | float | One target forward pass at M=K |
| `draft_step_ms` | float | One draft forward pass at M=1 |
| `draft_total_ms` | float | K draft forward passes at M=1 |
| `prespec_ms` | float | K draft forward passes at M=F (fanout) |
| `e2e_ms` | float | End-to-end ms per bench iter (host clock) |
| `avg_tokens` | float | Average tokens generated per iter |
| `avg_rounds` | float | Average verify rounds per iter |
| `miss_rate` | float | Fraction of rounds where cache miss occurred |
| `throughput_tok_s` | float | `avg_tokens × 1000 / e2e_ms` |

> [!NOTE]
> When `target_fits = 0` or `draft_fits = 0`, the timing columns for that
> model are extrapolated from a single-layer measurement scaled to n_layers,
> not from a true full-model pass.

---

## 6. Parameter Glossary

| Symbol | Meaning |
|--------|---------|
| **α (alpha)** | Per-token acceptance probability; Bernoulli(α) independently per draft position. α = 1.0 means perfect draft; α = 0.0 means every draft token is rejected. |
| **phit** | Probability that the fanout pre-speculation finds a cache hit. When `Bernoulli(phit)` fails, the draft re-runs K sequential steps (miss penalty). |
| **K** | Number of draft tokens speculated per round. Higher K = more tokens per round at high α, but more wasted work at low α. |
| **F** | Fanout factor: the draft model runs K passes at batch size F, simulating F parallel candidate continuations. |
| **N** | Target number of tokens to generate per benchmark iteration. The token loop ends when `local_tokens >= N`. |
| **miss_rate** | Observed `total_misses / total_rounds` across all bench_iters. Expected value ≈ 1 − phit. |

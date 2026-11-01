# Async-SSD: Asynchronous Speculative Decoding Benchmark

This project benchmarks three GPU inference modes for transformer-based language
models: standard autoregressive decoding (AR), standard speculative decoding
(SD), and an asynchronous speculative decoding implementation (SSD-single) that
uses two CPU threads and two CUDA streams on a single GPU to overlap draft
pre-speculation with target verification. All three modes are implemented from
scratch in CUDA C++ using cuBLAS GEMMs on synthetic (randomly initialized)
weight tensors, making the benchmark self-contained and runnable without real
model checkpoints.

---

## Project goal

> **Does pipelining draft pre-speculation with target verification improve
> throughput on a single GPU? Under what conditions does it help?**

Standard speculative decoding runs the draft model and target model sequentially
each round. SSD-single moves draft pre-speculation into a background thread so
it overlaps with the target's verification step, potentially hiding the draft's
latency. The experiment systematically varies the token acceptance rate (α) and
the pre-speculation cache-hit rate (p\_hit) to find where the overlap benefit
outweighs the overhead of thread coordination and GPU contention.

See [`docs/architecture.md`](docs/architecture.md) for a detailed design
description.

---

## Architecture overview

| Mode | Description |
|------|-------------|
| **AR** | Single GPU, single stream — one target forward pass per token |
| **SD** | Single GPU, single stream — draft generates K candidates, target verifies in one pass; accepts 1–K+1 tokens per round |
| **SSD-single** | Single GPU, two CUDA streams, two host threads — draft thread pre-speculates for the *next* round while the target thread verifies the *current* round |

The key insight in SSD-single: if the pre-speculation result is still valid when
verification completes, the draft's latency is fully hidden. At high α and
p\_hit this yields measurable throughput improvements over both AR and SD.

For a full description of the async pipeline and thread-safe queue design see
[`docs/architecture.md`](docs/architecture.md).

---

## Prerequisites

- **CUDA toolkit** — tested with CUDA 12.x
- **cuBLAS** — included with the CUDA toolkit
- **g++-13** (or a compatible C++17 host compiler)
- **Single GPU with ≥ 8 GB VRAM** — needed for the llama-8b + llama-1b
  synthetic weight tensors to fit simultaneously

For multi-GPU extensions only:
- NCCL library
- Multi-GPU machine (e.g. PSC Bridges-2 V100 node)

---

## Quick start

```bash
# Build (no NCCL required)
make

# Run AR baseline
./benchmark --mode ar --target-model llama-8b --N 128

# Run SD
./benchmark --mode sd --target-model llama-8b --draft-model llama-1b --K 4 --alpha 0.9

# Run Async-SSD (single GPU)
./benchmark --mode ssd-single --target-model llama-8b --draft-model llama-1b \
            --K 4 --F 4 --alpha 0.9 --phit 0.85

# CSV output
./benchmark --csv-header > results/my_run.csv
./benchmark --mode ar --csv | grep '^RESULT,' >> results/my_run.csv
```

---

## CLI flags

```
Options:
  --target-model <name>   Target model (tiny, llama-1b, llama-8b, llama-70b)  [llama-8b]
  --draft-model  <name>   Draft model  (tiny, llama-1b, llama-8b, llama-70b)  [llama-1b]
  --tp           <int>    Tensor parallelism degree                     [1]
  --mode         <str>    Inference mode (ar, sd, ssd, ssd-single)      [ar]
  --K            <int>    Speculation length (sd/ssd only)              [4]
  --F            <int>    Fan-out for pre-speculation (ssd only)        [4]
  --alpha        <float>  Acceptance rate (sd/ssd only)                 [0.9]
  --phit         <float>  Cache hit probability (ssd only)              [0.85]
  --N            <int>    Tokens to generate                            [128]
  --warmup       <int>    Warmup iterations                             [10]
  --iters        <int>    Benchmark iterations                          [20]
  --skip-component        Skip per-stage forward-pass timing
  --skip-e2e              Skip the end-to-end simulation
  --csv                   Emit one CSV row at the end (prefixed RESULT,)
  --csv-header            Print CSV header line and exit
  --help                  Show this message
```

---

## Results summary

Representative throughput numbers from `results/01_alpha_K.csv`
(Llama-3.1-8B target, Llama-3.2-1B draft, N=128, F=4, p\_hit=0.85):

| Mode | α | K | Throughput (tok/s) |
|------|---|---|-------------------|
| AR | — | — | 49.1 |
| SD | 0.7 | 4 | 86.4 |
| **SSD-single** | **0.7** | **4** | **130.4** |
| SD | 0.9 | 4 | 130.5 |
| **SSD-single** | **0.9** | **4** | **196.1** |
| SD | 0.9 | 6 | 138.3 |
| **SSD-single** | **0.9** | **6** | **213.8** |
| SD | 1.0 | 6 | 182.6 |
| **SSD-single** | **1.0** | **6** | **278.6** |

**SSD-single consistently outperforms both AR and SD** across the measured
α range, with the gap widening at higher α and K. At α=0.9, K=4 SSD-single
achieves ~50% higher throughput than SD alone; at α=1.0, K=6 it achieves
~52% higher. The improvement comes from hiding the draft model's latency
(~13 ms for K=4) behind the target's verification step (~19 ms), giving
almost free speculation when the pre-speculation result remains valid.

> [!NOTE]
> At very large K (K=8) with high α, SSD-single's advantage over SD narrows.
> Two streams sharing one GPU's SM pool limits true parallelism; the draft
> stream can starve under heavy speculation loads.

---

## Key findings

- **SSD-single achieves measurable throughput improvement over AR and SD at
  high α and p\_hit.** At α ≥ 0.7 and the default p\_hit = 0.85, SSD-single
  consistently outperforms standard SD by 30–55%, and AR by 2–5×.

- **The improvement comes from overlapping *host scheduling*, not true GPU
  kernel parallelism.** Both CUDA streams share the same physical SM pool.
  What hides is the CPU-side draft decision-making and CUDA stream submission
  latency — and, to a lesser extent, the memory-bandwidth phase of draft
  GEMMs that does not fully overlap with target compute.

- **At low α or p\_hit, thread coordination overhead reduces the benefit.**
  When speculative tokens are frequently rejected (low α) or pre-speculation
  results are invalid (low p\_hit), SSD-single falls back to roughly SD
  performance. Thread synchronization and additional queue copies introduce
  measurable overhead (~2–5 ms/round) that is not amortized by the overlap.

- **Comparison to multi-GPU SSD is left as future work.** The multi-GPU SSD
  mode (`make USE_NCCL=1`) assigns draft and target to separate physical GPUs,
  enabling true kernel-level overlap. This mode is implemented but requires
  cluster access (PSC Bridges-2) and has not been validated on the current
  single-GPU machine.

---

## Running the experiment sweep

```bash
# Full alpha × K sweep on local GPU
bash scripts/run_single_gpu_sweep.sh

# PSC cluster sweeps (requires NCCL build and Bridges-2 allocation)
sbatch scripts/01_sweep_alpha_K.sbatch
```

See [`scripts/README.md`](scripts/README.md) for details on each sweep script.

---

## Analysis and plots

```bash
python3 analysis/plot_results.py
```

---

## Project structure

```
src/         — CUDA/C++ source files (see src/README.md)
scripts/     — SLURM scripts (PSC cluster) and local runner scripts
tests/       — correctness test scripts
results/     — CSV benchmark data
analysis/    — plotting and profiling scripts
docs/        — detailed documentation (architecture, limitations, slides outline)
microbench/  — cuBLAS GEMM and NCCL microbenchmark results
```

---

## Multi-GPU extensions (optional)

```bash
make multi-gpu    # equivalent to: make USE_NCCL=1
```

This enables:
- Tensor-parallel AR and SD across N GPUs (via `forward_tp.cu`)
- Full multi-GPU async SSD where draft and target run on separate GPUs
  (`bench_ssd.cu`)

Requires NCCL, `nvToolsExt`, and a multi-GPU machine. Not validated on a
single-GPU machine. See [`scripts/README.md`](scripts/README.md) for PSC
cluster setup instructions.

---

## Limitations

See [`docs/limitations.md`](docs/limitations.md) for full details. In summary:

- **Synthetic workload** — weight tensors are randomly initialized; no real
  model weights are loaded
- **No KV cache** — each forward pass is a fresh GEMM; memory traffic does
  not reflect real inference with a growing KV cache
- **Probability-based acceptance** — token acceptance is sampled from a
  Bernoulli(α) distribution rather than derived from actual vocabulary
  distributions
- **Single-GPU contention** — SSD-single streams share physical SMs, limiting
  true overlap; results differ from a true multi-GPU deployment
- **No batch dimension** — all benchmarks run at batch size 1 (decode phase)

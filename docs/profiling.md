# CUDA Stream Concurrency & Profiling Analysis

This document provides a detailed timeline and concurrency analysis of the **Async-SSD (single-GPU)** execution path, based on Nsight Systems profiling (`ssd_single.nsys-rep` and `ssd_single.sqlite`).

---

## 1. Profiling Methodology

The benchmark was profiled using **NVIDIA Nsight Systems (nsys)** on an NVIDIA GeForce RTX 3050 6GB Laptop GPU (Ampere architecture, 16 SMs, Compute Capability 8.6).

### Profiling Command
```bash
nsys profile \
  --trace=cuda,nvtx,osrt \
  --output=ssd_single \
  --force-overwrite=true \
  ./benchmark --mode ssd-single \
    --target-model llama-8b \
    --draft-model llama-1b \
    --K 4 --F 4 --alpha 0.9 --phit 0.85 \
    --N 128 --warmup 5 --iters 10
```

The resulting `.nsys-rep` profile was exported to SQLite (`ssd_single.sqlite`) and inspected using `analysis/parse_profile.py`.

---

## 2. Stream Concurrency & Timeline Execution

In `bench_ssd_single.cu`, the single-GPU execution employs two host threads and two dedicated CUDA streams:
- **Stream 16 (Draft Worker)**: Issues FP16 GEMMs for single-token draft generation ($M=1$) and candidate pre-speculation fanout ($M=F$).
- **Stream 17 (Target Worker)**: Issues FP16 GEMMs for batched target verification ($M=K$).

### Measured Stream Activity

| Stream ID | Assigned Role | Kernels Launched | Timeline Span | Total Kernel Work | Mean Kernel Duration |
|:---|:---|:---|:---|:---|:---|
| **Stream 16** | Draft Worker | 83,776 | 9,257.28 ms | 2,472.74 ms | 29.52 µs |
| **Stream 17** | Target Worker | 71,568 | 9,251.52 ms | 8,465.89 ms | 118.29 µs |
| **Stream 15** | Warmup/Init | 3,360 | 391.33 ms | 381.19 ms | 113.45 µs |
| **Streams 13-14** | Memory Setup | 1,680 | 25.17 ms | 19.07 ms | 11.35 µs |

---

## 3. Quantitative Overlap & Duty Cycle

Using interval analysis across the active kernel start and end timestamps, concurrent kernel execution between Stream 16 and Stream 17 was measured:

- **Overall Timeline Span**: 9,264.27 ms
- **Overlapping Kernel Pairs**: 76,823 pairs
- **Total Concurrent Kernel Execution Time**: 2,166.85 ms
- **Timeline Overlap Duty Cycle**: **23.39%**

### Analysis of the 23.39% Overlap Duty Cycle

1. **Phase Alignment**: Overlap occurs during the window where the target worker verifies candidate tokens ($M=K$) on Stream 17 while the draft worker concurrently pre-speculates the next branch ($M=F$) on Stream 16.
2. **Draft Work Limit**: The draft model (Llama-1B, 16 layers) performs significantly less compute than the target model (Llama-8B, 32 layers, larger hidden dimension). As reflected in the total kernel work (2,472 ms vs 8,465 ms), draft pre-speculation finishes sooner than target verification. Once the draft finishes pre-speculating, it blocks waiting on `target_to_draft.pop()`, leaving the GPU solely to Stream 17 until verification completes.
3. **Cache Miss Fallback**: When a pre-speculation miss occurs (probability $1 - p_{\text{hit}}$), the draft worker must sequentially run fallback steps after the target verifies, serializing that portion of the round.

---

## 4. Hardware Contention on a Single GPU

When running both streams on a single GPU (NVIDIA RTX 3050, 16 SMs):

### Resource Sharing & Contention
- **SM Partitioning**: Both cuBLAS GEMM kernels issue thread blocks to the same 16 Streaming Multiprocessors. At batch size $M=4$, target verification GEMMs do not fully saturate 16 SMs on every layer, allowing draft kernels to co-locate and execute concurrently.
- **Memory Bandwidth Sharing**: Both streams access the single 128-bit memory bus (192 GB/s bandwidth). Concurrent execution experiences minor memory bus contention during weight loads, causing individual kernel durations to stretch slightly compared to isolated execution.
- **Single-GPU vs Multi-GPU**: In multi-GPU Async-SSD (`USE_NCCL=1`), the draft and target run on physically separate GPUs with independent SMs and memory channels, achieving full hardware isolation without SM competition.

---

## 5. Host Scheduling & Synchronization Overhead

Host-side synchronization was profiled across all rounds:
- **CUDA Synchronization**: 1,390 explicit synchronization calls (`cudaEventSynchronize`, `cudaStreamSynchronize`) recorded, totaling 7,685.24 ms cumulative wait time across host threads. This reflects threads intentionally blocking on queues and GPU completions.
- **Host Queue Latency**: `ThreadSafeQueue` uses `std::mutex` and `std::condition_variable`. Microbenchmarks show mutex lock-and-notify latency is on the order of 0.5–1.5 µs, which is negligible compared to the ~20 µs kernel launch latency and ~18 ms target verification pass.

---

## 6. How to Reproduce

Run the profile parser on the repository's SQLite database or generate a new profile:

```bash
# Run analysis on existing profile
python3 analysis/parse_profile.py --db ssd_single.sqlite

# Profile a new run
nsys profile --trace=cuda,osrt --output=my_profile ./benchmark --mode ssd-single ...
python3 analysis/parse_profile.py --db my_profile.sqlite
```

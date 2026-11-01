# Presentation Slides Outline — Async-SSD

~10 slides · 15-minute talk

---

## Slide 1 — Title & Motivation

**Title:** Async-SSD: Overlapping Draft Pre-Speculation and Verification on a Single GPU

- LLM inference is memory-bandwidth bound at decode time: one token per forward pass, blocking on weights
- Autoregressive (AR) decoding wastes compute — the GPU is idle between tokens waiting for the next
- Speculative decoding (SD) improves throughput by batching K draft tokens into one target verification pass
- **Gap:** SD still runs draft and target *sequentially*; can we overlap them?
- **This work:** SSD-single — asynchronous speculative decoding with two CUDA streams on one GPU

---

## Slide 2 — Background: Speculative Decoding

- **Draft model** (small, fast) generates K candidate tokens
- **Target model** (large, authoritative) verifies all K tokens in a *single* batched forward pass
- Accepted tokens: up to K+1 per round (expected: `(1−αᴷ)/(1−α)` tokens at acceptance rate α)
- Throughput gain: verification pass costs ~1 target step, but produces many tokens
- **Limitation:** each round is sequential — draft runs first, then target; draft latency is never hidden

---

## Slide 3 — Async-SSD Design: Two-Thread, Two-Stream Architecture

- **Insight:** draft for round `t+1` can start *before* round `t`'s target verification finishes
- **Two host threads:**
  - *Draft thread* — runs K draft forward passes on `cudaStream_0`, pushes result to queue
  - *Target thread* — pulls draft result from queue, runs one target verification on `cudaStream_1`
- **Pre-speculation cache:** stores draft tokens for the "most likely" next context; valid when the target accepts all K tokens (probability α^K × p\_hit)
- When valid: draft latency is completely hidden; only target latency determines round time
- Throughput improvement is bounded by `min(draft_ms, target_ms)` of hidden work per round

---

## Slide 4 — Implementation: CUDA Streams, cuBLAS GEMMs, Thread-Safe Queues

- All weight tensors are synthetic half-precision (FP16); cuBLAS `cublasGemmEx` with `CUBLAS_TENSOR_OP_MATH`
- Forward pass: per-layer GEMM stack simulating attention QKV projection + FFN (no softmax, no real attention)
- `ThreadSafeQueue<RoundResult>` (mutex + condition variable) coordinates draft→target handoff
- CUDA streams: `stream_draft` and `stream_target` issued independently; host threads block on `cudaStreamSynchronize`
- Benchmark harness: configurable α (acceptance rate), p\_hit (pre-speculation hit rate), K (depth), F (fan-out)
- CSV output supports easy sweep automation via shell scripts

---

## Slide 5 — Experimental Setup

- **Target model:** Llama-3.1-8B (synthetic weights, 32 layers, d=4096, d\_ff=14336)
- **Draft model:** Llama-3.2-1B (synthetic weights, 16 layers, d=2048)
- **Hardware:** single NVIDIA GPU with ≥ 8 GB VRAM; CUDA 12.x, cuBLAS
- **Parameters swept:**
  - α ∈ {0.3, 0.5, 0.7, 0.8, 0.9, 0.95, 1.0} (acceptance rate)
  - K ∈ {1, 2, 4, 6, 8} (speculation depth)
  - p\_hit = 0.85 (fixed for alpha/K sweep)
- **Metric:** tokens/second (N=128 tokens, 10 warmup + 20 timed iterations)
- AR baseline: no draft model; one target step per token

---

## Slide 6 — Results: Throughput vs. α

- AR baseline: **49.1 tok/s** (flat across all α — no speculation)
- SD at α=0.9, K=4: **130.5 tok/s** (+166% over AR)
- SSD-single at α=0.9, K=4: **196.1 tok/s** (+300% over AR, +50% over SD)
- At α=1.0, K=6: SD achieves 182.6; SSD-single achieves **278.6 tok/s**
- Gap between SD and SSD-single grows with α: at high acceptance rates, pre-speculation is valid more often, hiding more draft latency
- At α=0.3 the gap nearly closes: frequent rejection invalidates pre-speculation, wasting draft work

---

## Slide 7 — Results: Throughput vs. K and F

- At low K (K=1): SSD-single advantage is small (~12% over SD) — draft is fast, little to hide
- At K=4 and α=0.9: SSD-single peaks at ~196 tok/s vs. SD's 130 tok/s
- At K=8 and α=0.9: advantage narrows — two streams competing for SMs limits overlap, draft overhead grows
- F (fan-out) controls pre-speculation branching; F=4 gives the best coverage without excess waste
- **Sweet spot:** K=4–6, α≥0.8, F=4 — draft is long enough to hide but short enough to fit in target's shadow

---

## Slide 8 — Profiling: Stream Overlap Analysis

- Nsight Systems trace (`ssd_single.nsys-rep`) confirms two distinct kernel streams
- At high α+p\_hit: target stream is consistently the critical path; draft stream finishes inside target window
- At low α: draft stream sometimes *extends* past target completion, adding idle time on the target thread
- No true SM-level parallelism observed — CUDA MPS not used; both streams time-multiplex on the same SMs
- Primary source of speedup: **eliminating the host-side stall** between draft submission and target submission
- Cache behavior: repeated GEMMs on the same weight tensors benefit from L2 residency; this may slightly inflate the observed overlap benefit

---

## Slide 9 — Discussion: When Does It Help? Limitations

**When SSD-single helps:**
- High acceptance rate (α ≥ 0.7) — pre-speculation results are frequently valid
- High p\_hit (≥ 0.8) — draft tokens remain on-path after target accepts
- Moderate K (4–6) — enough draft work to overlap, not so much that SM contention dominates

**Limitations:**
- Synthetic weights and no KV cache — real inference has different memory pressure profiles
- Probability-based acceptance (Bernoulli(α)) — not derived from actual token distributions
- Single GPU: both streams share SMs → results differ from a true multi-GPU deployment
- No batch support — decode at batch size 1 only; SD is even more beneficial at batch > 1

---

## Slide 10 — Conclusion & Future Work

**Conclusion:**
- SSD-single achieves **30–55% throughput gain over standard SD** at high α and p\_hit on a single GPU
- Gain comes from overlapping host scheduling and partially overlapping GPU memory phases — not from SM-level parallelism
- The two-thread, two-stream design is implementable with ~300 lines of CUDA and standard C++ concurrency primitives

**Future work:**
- **Multi-GPU SSD** (`make multi-gpu`): draft and target on separate GPUs enable *true* kernel-level overlap; preliminary NCCL code is implemented but requires cluster validation
- Real model weights and tokenizer integration
- KV cache simulation to model realistic memory traffic
- Batch-level speculative decoding (batch size > 1)
- Roofline model for identifying compute vs. bandwidth bottlenecks per mode

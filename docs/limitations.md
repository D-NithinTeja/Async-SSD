# Limitations

This document enumerates the known limitations and simplifications of the
Async-SSD benchmark relative to a production LLM inference system.

---

## 1. Synthetic Compute (No Real Transformer Ops)

Each "forward pass" issues exactly **7 cuBLAS FP16 GEMM calls per layer**
(Q, K, V, O projections; Gate, Up, Down FFN projections) and nothing else.

**Missing operations include:**
- Attention softmax and scaled dot-product attention (the quadratic part
  of attention with respect to sequence length)
- RMS / layer normalisation
- Rotary positional embeddings (RoPE)
- Activation functions (SiLU / SwiGLU element-wise operations)
- KV-cache read/write (growing with sequence length)
- Token embedding and de-embedding (lm_head)
- Any real sampling (top-k, top-p, temperature)

Because attention is omitted, the benchmark is compute-bound on the GEMM
kernels only.  In practice, at decode batch size M = 1 real attention is
memory-bandwidth-bound on the KV cache, not compute-bound.

---

## 2. Synthetic Acceptance (Bernoulli Model)

Draft token acceptance is simulated as independent Bernoulli(α) draws:

```cpp
std::bernoulli_distribution accept_dist(alpha);
for (int k = 0; k < K; ++k) {
    if (accept_dist(rng)) ++accepted;
    else break;
}
```

Real speculative decoding uses a **rejection-sampling** or
**modified-rejection-sampling** scheme (Leviathan et al., 2023) that
compares the draft model's probability distribution to the target model's
distribution token-by-token.  The acceptance rate depends on the
similarity of the two models' distributions, the input context, and the
temperature setting — none of which are modelled here.

**Implications:**
- The benchmark cannot show whether a given draft/target model pair
  achieves a particular α in practice.
- Results are parameterised by α as an external input; conclusions are
  conditional on that parameter.

---

## 3. Synthetic Cache Hits (Bernoulli phit)

Pre-speculation cache hits are modelled as `Bernoulli(phit)`:

```cpp
std::bernoulli_distribution hit_dist(phit);
if (!hit_dist(rng)) { /* cache miss: re-draft */ }
```

A real KV-cache hit system would track which previously-computed keys are
still valid after the target accepts or rejects draft tokens, using a
**tree or radix cache** structure.  The actual hit rate depends on
branching factor, access pattern, and eviction policy — all absent here.

---

## 4. No KV-Cache Growth With Sequence Length

In a real autoregressive decoder, each forward pass must load the full KV
cache for all previously generated tokens.  At long sequences, the KV-cache
load dominates latency and the model transitions from compute-bound to
memory-bandwidth-bound.

This benchmark fixes the GEMM sizes to the weight dimensions alone (M × d
input); there is no growing KV matrix.  Consequently:

- `target_step_ms` and `target_verify_ms` are constant across all sequence
  positions.
- Throughput does not degrade with context length.
- Prefill (processing the prompt) is not modelled.

---

## 5. Single-GPU Resource Contention (ssd-single)

In `ssd-single` mode, the draft worker and target worker both submit work
to the same physical GPU via two separate CUDA streams.  This means:

- Both streams compete for SMs, L2 cache, HBM bandwidth, and the cuBLAS
  cublasHandle kernel-dispatch pipeline.
- At LLM scale (e.g. 4096 × 14336 GEMM on an RTX 3050), each GEMM kernel
  typically saturates the GPU; the hardware serialises stream work rather
  than overlapping it.
- A production SSD deployment would run the draft model on a **dedicated
  GPU** (or a separate partition of a large multi-GPU node), avoiding
  this contention entirely.

See `docs/profiling.md` for measured stream concurrency analysis.

---

## 6. Multi-GPU Paths Not Validated

`src/bench_ssd.cu` (multi-GPU SSD with NCCL) and the tensor-parallel AR/SD
paths (`--tp N`) are **compiled conditionally** (`#ifdef USE_NCCL`) and
were **not validated** on the development machine (NVIDIA RTX 3050 Laptop,
single GPU).

These paths require:
- A multi-GPU system or NVLink topology.
- A valid NCCL installation.
- Tensor-parallelism degrees that evenly divide `q_dim`, `kv_dim`, and
  `d_ff`.

---

## 7. Throughput Is Relative, Not Absolute

The reported `throughput_tok_s` numbers reflect synthetic GEMM throughput
on the benchmark GPU under the described conditions.  They should be
interpreted as:

> "Mode A is X% faster than mode B under the same synthetic workload on
> the same hardware."

They **do not** predict real end-to-end LLM serving throughput, which
additionally depends on:
- KV-cache bandwidth
- Sampling overhead
- Request batching (the benchmark uses batch size 1)
- Network / communication overhead in multi-node settings
- Real model quality / acceptance rates

---

## Summary Table

| Limitation | Impact |
|---|---|
| No attention / KV cache | Latency does not grow with context length |
| Bernoulli acceptance | α is synthetic; real rates depend on model pair |
| Bernoulli cache hit | phit is synthetic; real hit rates depend on cache design |
| Single-GPU contention | Underestimates SSD benefit vs. multi-GPU deployment |
| No real weights | Cannot test model quality or real acceptance rates |
| Multi-GPU paths unvalidated | NCCL paths are compile-time optional, untested |

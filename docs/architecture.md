# Async-SSD Architecture

This document describes the three inference benchmark modes implemented in
this project, the data structures and queues that connect them, and what each
measurement means.

---

## 1. Execution Modes

### 1.1 AR – Autoregressive Decoding

AR is the baseline: the target model runs a single forward pass at batch
size M = 1 for each of the N tokens to generate.  There is no draft model,
no speculation, no parallelism.

```
for token in 1..N:
    target.forward(M=1)   # one cuBLAS GEMM sequence per layer
```

- **Latency per token**: one full target forward pass (`target_step_ms`).
- **Throughput**: N / (N × target_step_ms) = 1 / target_step_ms  tok/s.

---

### 1.2 SD – Speculative Decoding (Standard)

The draft model generates K draft tokens sequentially (K forward passes at
M = 1), then the target model verifies all K drafts in a single batched
forward pass at batch size M = K.

```
while tokens_generated < N:
    # Draft phase
    for k in 1..K:
        draft.forward(M=1)

    # Verify phase
    target.forward(M=K)          # one batched pass verifies K drafts

    # Acceptance
    accepted = 0
    for k in 1..K:
        if Bernoulli(alpha): accepted++
        else: break
    tokens_generated += accepted + 1   # +1 bonus token always accepted
```

**tokens_per_round** = `accepted + 1`.  Even if all K drafts are rejected
(accepted = 0), the target model's own predicted token at that position is
always kept — hence the guaranteed +1 bonus token.

---

### 1.3 SSD-single – Async Speculative Decoding (Single GPU)

SSD uses two host threads on one GPU to overlap draft work with target
verification.  There are two cuBLAS handles bound to two separate CUDA
streams on GPU 0.

#### Thread Roles

| Thread | Work |
|--------|------|
| `draft_worker` | K sequential forward passes (M=1) to produce draft tokens; fanout pre-speculation (F forward passes at M=F) while the target is verifying |
| `target_worker` | Blocks on queue, verifies K drafts (M=K), determines accepted count, signals draft |

#### Data-Flow Diagram

```
  draft_worker                            target_worker
  ────────────                            ─────────────
  run_draft_steps(K, M=1)
  → draft tokens ready
  push(SpecMessage{K tokens})
  ─────────────────────────────────────► pop() → SpecMessage

  ┌── loop while not done ──┐            forward(M=K)         # verify
  │  run_draft_steps(K,M=F) │            evaluate Bernoulli(alpha) × K
  │  [fanout pre-speculation]│            accepted = contiguous run of successes
  │                          │            tokens += accepted + 1
  │  pop() ◄────────────────┼──────────  push(OutcomeMessage{accepted, done})
  │  if cache miss (1-phit): │
  │    run_draft_steps(K,M=1)│ # re-draft
  │  push(SpecMessage)  ─────┼──────────►  ...
  └──────────────────────────┘
```

The two `ThreadSafeQueue` instances are blocking FIFO queues backed by
`std::mutex` + `std::condition_variable`:

- **`draft_to_target`**: carries `SpecMessage{std::vector<half>(K)}` —
  the K synthetic draft token embeddings.
- **`target_to_draft`**: carries `OutcomeMessage{accepted, bonus_id, done}` —
  how many tokens were accepted and whether generation is complete.

A `StartGate` barrier synchronises the two threads after warm-up so that
wall-clock timing begins simultaneously.

---

## 2. The "Round" Concept

One **round** is one complete verify cycle:

1. Draft produces K tokens.
2. Target verifies them (one forward pass at M = K).
3. The outcome (how many were accepted) is communicated back.

`avg_rounds` in the CSV is the average number of rounds needed to generate
N tokens across `bench_iters` repetitions.

`tokens_per_round = accepted + 1` because the target always contributes its
own predicted token even when it rejects the draft's position.

For a Bernoulli acceptance model with rate α and speculation length K:

```
E[tokens_per_round] = sum_{j=0}^{K} α^j * (1 - α^{K+1}) / (1 - α)   (approx)
                    ≈ (1 - α^{K+1}) / (1 - α)   for α < 1
```

---

## 3. Fanout Pre-Speculation (F)

After the draft worker sends its K drafts to the target, it immediately
starts **fanout pre-speculation**: running K additional forward passes at
batch size F instead of M = 1.

```
fanout_state = ModelState(draft_cfg, F)   # allocated once
run_draft_steps(fanout_state, K, M=F)
```

The intent is to speculatively pre-compute F candidate continuations in
parallel, mimicking a scenario where a KV-cache lookup hits on F branches.

- **phit**: probability that the fanout cache hit occurs.  Sampled as
  `Bernoulli(phit)` per round.
- **miss_rate**: fraction of rounds where `phit` draw failed — in those
  rounds the draft worker falls back to a fresh sequential K-step draft
  (`run_draft_steps(step_state, K, M=1)`).

`prespec_ms` is the wall time for one fanout pass (K forward passes at M=F).
When F = 1 it is identical to `draft_total_ms`.

---

## 4. Component Timing Definitions

| CSV column | What it measures |
|---|---|
| `target_step_ms` | One full target forward pass at M=1 (AR baseline) |
| `target_verify_ms` | One full target forward pass at M=K (verify batch) |
| `draft_step_ms` | One full draft forward pass at M=1 |
| `draft_total_ms` | K sequential draft forward passes at M=1 |
| `prespec_ms` | K draft forward passes at M=F (fanout batch) |
| `e2e_ms` | Wall time (steady_clock) for the target thread across bench_iters, divided by bench_iters |

All component timings use CUDA events (`cudaEventRecord` / `cudaEventElapsedTime`)
for GPU-side accuracy.  `e2e_ms` uses `std::chrono::steady_clock` on the
host to capture the true end-to-end latency including queue waits.

---

## 5. File Map

| File | Role |
|---|---|
| `src/bench_ssd_single.cu` | SSD-single implementation (threads, queues, timing) |
| `src/bench_sd.cu` | SD implementation (single-GPU and TP paths) |
| `src/bench_ar.cu` | AR implementation (single-GPU and TP paths) |
| `src/forward_single.cu` | `forward_layer` / `forward_model` (7 GEMMs per layer) |
| `src/config.h` | `ModelConfig` struct + model registry |
| `src/thread_safe_queue.h` | Blocking FIFO `ThreadSafeQueue<T>` |
| `src/bench_common.cu` | CSV output, printing helpers |
| `src/main.cu` | CLI argument parsing, top-level dispatch |

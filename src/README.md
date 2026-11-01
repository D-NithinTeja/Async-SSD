# src/ — Source Files

This directory contains all CUDA/C++ source files for the Async-SSD benchmark.

## Core files (single-GPU build — no NCCL required)

| File | Role |
|------|------|
| `main.cu` | CLI entry point — argument parsing, model lookup, dispatch to benchmark runners |
| `config.h` | `ModelConfig` struct and model presets (`tiny`, `llama-1b`, `llama-8b`, `llama-70b`) |
| `check.h` | `CHECK_CUDA` / `CHECK_CUBLAS` error-checking macros |
| `gemm.h` | Thin wrapper around cuBLAS GEMM for half-precision matrix multiply |
| `gpu_context.h` | `LayerWeights` and `ActivationBuffers` structs; GPU memory allocation helpers |
| `forward_single.cu` / `.h` | Single-GPU transformer layer forward pass (`forward_layer`, `forward_model`) |
| `bench_common.cu` / `.h` | Shared benchmark options (`BenchOpts`), CSV header/row printing, `lookup_model` |
| `bench_ar.cu` / `.h` | Autoregressive (AR) decode benchmark |
| `bench_sd.cu` / `.h` | Standard speculative decoding (SD) benchmark |
| `bench_ssd_single.cu` / `.h` | **Async SSD single-GPU** benchmark — two CPU threads + two CUDA streams to overlap draft pre-speculation with target verification |
| `thread_safe_queue.h` | Blocking FIFO (`ThreadSafeQueue<T>`) used to hand off work between the draft and target threads in SSD-single |

## Optional files (multi-GPU / NCCL build — `make USE_NCCL=1` only)

> [!NOTE]
> These files are compiled only when `USE_NCCL=1` is set. They require NCCL,
> `nvToolsExt`, and a multi-GPU machine (e.g. a PSC Bridges-2 V100 node).

| File | Role |
|------|------|
| `forward_tp.cu` / `.h` | Tensor-parallel transformer layer — splits weight matrices across GPUs and uses NCCL all-reduce for activation synchronization |
| `bench_ssd.cu` / `.h` | Multi-GPU async SSD benchmark — draft and target models run on separate GPUs |

All multi-GPU code paths are guarded with `#ifdef USE_NCCL` so the default
`make` produces a self-contained single-GPU binary with no NCCL dependency.

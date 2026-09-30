// bench_ssd_single.h
// SSD scheduling benchmark using two workers on one GPU.

#pragma once

#include "bench_common.h"
#include "config.h"

void run_ssd_single_benchmark(
  const ModelConfig& target_model,
  const ModelConfig& draft_model,
  int K,
  int F,
  int N,
  float alpha,
  float phit,
  int warmup,
  int iters,
  const BenchOpts& opts
);

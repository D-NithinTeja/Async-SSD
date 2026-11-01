#!/usr/bin/env bash
# scripts/run_single_gpu_sweep.sh
#
# Single-GPU experiment matrix: reproduces the PSC sbatch sweeps on one GPU
# using only CUDA + cuBLAS (no NCCL/multi-GPU required).
#
# Sweeps (default):
#   alpha : 0.3, 0.5, 0.7, 0.9
#   K     : 1, 2, 4, 8
#   F     : 1, 2, 4
#   phit  : 0.5, 0.7, 0.85, 0.95
#
# For every configuration all three modes are run: ar, sd, ssd-single.
# Results are collected into a timestamped CSV in results/.
#
# Usage:
#   bash scripts/run_single_gpu_sweep.sh [options]
#
# Options:
#   --dry-run      Print commands without executing them
#   --quick        Use tiny models and 1 iter for rapid testing
#   --output FILE  Override the output CSV path
#   --target-model NAME   (default: llama-8b)
#   --draft-model  NAME   (default: llama-1b)
#   --N N          Tokens per request        (default: 128)
#   --warmup W     Warmup iterations         (default: 5)
#   --iters I      Measurement iterations    (default: 10)
#   --help
#
# Exit code: 0 on success, non-zero on any failure.

set -euo pipefail

# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------
DRY_RUN=0
QUICK=0
OUTPUT_OVERRIDE=""

TARGET_MODEL="llama-2b"
DRAFT_MODEL="llama-100m"
N=128
WARMUP=2
ITERS=5

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dry-run)          DRY_RUN=1;                 shift ;;
    --quick)            QUICK=1;                   shift ;;
    --output)           OUTPUT_OVERRIDE="$2";      shift 2 ;;
    --target-model)     TARGET_MODEL="$2";         shift 2 ;;
    --draft-model)      DRAFT_MODEL="$2";          shift 2 ;;
    --N)                N="$2";                    shift 2 ;;
    --warmup)           WARMUP="$2";               shift 2 ;;
    --iters)            ITERS="$2";                shift 2 ;;
    --help|-h)
      sed -n '2,30p' "$0" | sed 's/^# \{0,1\}//'
      exit 0
      ;;
    *)
      echo "ERROR: unknown argument: $1" >&2
      echo "       Run with --help for usage." >&2
      exit 1
      ;;
  esac
done

# ---------------------------------------------------------------------------
# Apply --quick overrides
# ---------------------------------------------------------------------------
if [[ $QUICK -eq 1 ]]; then
  TARGET_MODEL="tiny"
  DRAFT_MODEL="tiny"
  N=8
  WARMUP=1
  ITERS=1
  echo "[quick] Using tiny models, N=8, iters=1"
fi

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

BINARY="${BINARY:-$PROJECT_ROOT/benchmark}"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
OUTPUT="${OUTPUT_OVERRIDE:-$PROJECT_ROOT/results/single_gpu_${TIMESTAMP}.csv}"
mkdir -p "$(dirname "$OUTPUT")"

# ---------------------------------------------------------------------------
# Sweep parameters
# ---------------------------------------------------------------------------
ALPHAS=(0.3 0.5 0.7 0.9)
KS=(1 2 4 8)
FS=(1 2 4)
PHITS=(0.5 0.7 0.85 0.95)

# Fixed values used when a dimension is not being swept
ALPHA_FIXED=0.9
K_FIXED=4
F_FIXED=1
PHIT_FIXED=0.85

# ---------------------------------------------------------------------------
# Dry-run / execute helpers
# ---------------------------------------------------------------------------
FAILURES=0
TOTAL_RUNS=0
COMPLETED_RUNS=0

maybe_run() {
  # Run "$@" or just print the command in dry-run mode.
  if [[ $DRY_RUN -eq 1 ]]; then
    echo "[dry-run] $*"
  else
    "$@"
  fi
}

# Run benchmark, append RESULT, rows to $OUTPUT.
# Usage: bench_run <progress-label> [benchmark args...]
bench_run() {
  local label="$1"; shift
  (( TOTAL_RUNS++ )) || true

  echo "[$(printf '%4d' "$TOTAL_RUNS")] $label"

  if [[ $DRY_RUN -eq 1 ]]; then
    echo "         cmd: $BINARY $* --csv"
    (( COMPLETED_RUNS++ )) || true
    return
  fi

  local tmp_out tmp_err
  tmp_out="$(mktemp)"
  tmp_err="$(mktemp)"
  local rc=0

  "$BINARY" "$@" --csv >"$tmp_out" 2>"$tmp_err" || rc=$?

  if [[ $rc -ne 0 ]]; then
    echo "         [FAIL] exit code $rc" >&2
    echo "         stderr: $(head -3 "$tmp_err")" >&2
    (( FAILURES++ )) || true
  else
    grep '^RESULT,' "$tmp_out" >> "$OUTPUT" || true
    (( COMPLETED_RUNS++ )) || true
  fi

  rm -f "$tmp_out" "$tmp_err"
}

# ---------------------------------------------------------------------------
# Preflight
# ---------------------------------------------------------------------------
echo "============================================================"
echo " Async-SSD single-GPU experiment sweep"
echo "============================================================"
echo "Binary     : $BINARY"
echo "Output CSV : $OUTPUT"
echo "Models     : target=$TARGET_MODEL  draft=$DRAFT_MODEL"
echo "Run params : N=$N  warmup=$WARMUP  iters=$ITERS"
echo "Dry-run    : $DRY_RUN"
echo "============================================================"
echo

if [[ $DRY_RUN -eq 0 && ! -x "$BINARY" ]]; then
  echo "ERROR: benchmark binary not found or not executable: $BINARY" >&2
  echo "       Run 'make' (or 'make USE_NCCL=0') first." >&2
  exit 1
fi

# GPU info
if [[ $DRY_RUN -eq 0 ]]; then
  echo "===== GPU info ====="
  if command -v nvidia-smi &>/dev/null; then
    nvidia-smi --query-gpu=name,memory.total,driver_version \
               --format=csv,noheader 2>/dev/null \
      | awk '{print "[gpu]", NR-1":", $0}' \
      || nvidia-smi -L 2>/dev/null || true
  else
    echo "[warn] nvidia-smi not found"
  fi
  echo "===================="
  echo
fi

# ---------------------------------------------------------------------------
# Write CSV header
# ---------------------------------------------------------------------------
if [[ $DRY_RUN -eq 0 ]]; then
  "$BINARY" --csv-header > "$OUTPUT"
  echo "[header] Written to $OUTPUT"
  echo
fi

# ---------------------------------------------------------------------------
# Sweep 1: alpha x K  (phit and F fixed)
# AR runs once (independent of alpha/K).
# ---------------------------------------------------------------------------
echo "------------------------------------------------------------"
echo " Sweep 1 of 4: alpha x K  (phit=${PHIT_FIXED}, F=${F_FIXED})"
echo "------------------------------------------------------------"

bench_run "AR baseline (alpha x K sweep)" \
  --mode ar \
  --target-model "$TARGET_MODEL" \
  --N "$N" --warmup "$WARMUP" --iters "$ITERS"

for A in "${ALPHAS[@]}"; do
  for K in "${KS[@]}"; do
    bench_run "SD  alpha=$A K=$K" \
      --mode sd \
      --target-model "$TARGET_MODEL" --draft-model "$DRAFT_MODEL" \
      --K "$K" --alpha "$A" \
      --N "$N" --warmup "$WARMUP" --iters "$ITERS"

    bench_run "SSD-SINGLE  alpha=$A K=$K F=${F_FIXED} phit=${PHIT_FIXED}" \
      --mode ssd-single \
      --target-model "$TARGET_MODEL" --draft-model "$DRAFT_MODEL" \
      --K "$K" --F "$F_FIXED" --alpha "$A" --phit "$PHIT_FIXED" \
      --N "$N" --warmup "$WARMUP" --iters "$ITERS"
  done
done

# ---------------------------------------------------------------------------
# Sweep 2: phit  (alpha, K, F fixed)
# ---------------------------------------------------------------------------
echo
echo "------------------------------------------------------------"
echo " Sweep 2 of 4: phit  (alpha=${ALPHA_FIXED}, K=${K_FIXED}, F=${F_FIXED})"
echo "------------------------------------------------------------"

bench_run "AR baseline (phit sweep)" \
  --mode ar \
  --target-model "$TARGET_MODEL" \
  --N "$N" --warmup "$WARMUP" --iters "$ITERS"

bench_run "SD  alpha=${ALPHA_FIXED} K=${K_FIXED} (phit sweep baseline)" \
  --mode sd \
  --target-model "$TARGET_MODEL" --draft-model "$DRAFT_MODEL" \
  --K "$K_FIXED" --alpha "$ALPHA_FIXED" \
  --N "$N" --warmup "$WARMUP" --iters "$ITERS"

for PHIT in "${PHITS[@]}"; do
  bench_run "SSD-SINGLE  phit=$PHIT alpha=${ALPHA_FIXED} K=${K_FIXED} F=${F_FIXED}" \
    --mode ssd-single \
    --target-model "$TARGET_MODEL" --draft-model "$DRAFT_MODEL" \
    --K "$K_FIXED" --F "$F_FIXED" --alpha "$ALPHA_FIXED" --phit "$PHIT" \
    --N "$N" --warmup "$WARMUP" --iters "$ITERS"
done

# ---------------------------------------------------------------------------
# Sweep 3: F (fan-out)  (alpha, K, phit fixed)
# ---------------------------------------------------------------------------
echo
echo "------------------------------------------------------------"
echo " Sweep 3 of 4: F (fan-out)  (alpha=${ALPHA_FIXED}, K=${K_FIXED})"
echo "------------------------------------------------------------"

bench_run "AR baseline (fanout sweep)" \
  --mode ar \
  --target-model "$TARGET_MODEL" \
  --N "$N" --warmup "$WARMUP" --iters "$ITERS"

for PHIT in "${PHITS[@]}"; do
  for F in "${FS[@]}"; do
    bench_run "SSD-SINGLE  F=$F phit=$PHIT alpha=${ALPHA_FIXED} K=${K_FIXED}" \
      --mode ssd-single \
      --target-model "$TARGET_MODEL" --draft-model "$DRAFT_MODEL" \
      --K "$K_FIXED" --F "$F" --alpha "$ALPHA_FIXED" --phit "$PHIT" \
      --N "$N" --warmup "$WARMUP" --iters "$ITERS"
  done
done

# ---------------------------------------------------------------------------
# Sweep 4: All three modes compared at representative operating points
# (matching the breakdown sbatch scripts for a single GPU)
# ---------------------------------------------------------------------------
echo
echo "------------------------------------------------------------"
echo " Sweep 4 of 4: AR vs SD vs SSD-SINGLE at representative points"
echo "------------------------------------------------------------"

bench_run "AR baseline (breakdown)" \
  --mode ar \
  --target-model "$TARGET_MODEL" \
  --N "$N" --warmup "$WARMUP" --iters "$ITERS"

for A in "${ALPHAS[@]}"; do
  bench_run "SD  alpha=$A K=${K_FIXED}" \
    --mode sd \
    --target-model "$TARGET_MODEL" --draft-model "$DRAFT_MODEL" \
    --K "$K_FIXED" --alpha "$A" \
    --N "$N" --warmup "$WARMUP" --iters "$ITERS"

  bench_run "SSD-SINGLE  alpha=$A K=${K_FIXED} F=${F_FIXED} phit=${PHIT_FIXED}" \
    --mode ssd-single \
    --target-model "$TARGET_MODEL" --draft-model "$DRAFT_MODEL" \
    --K "$K_FIXED" --F "$F_FIXED" --alpha "$A" --phit "$PHIT_FIXED" \
    --N "$N" --warmup "$WARMUP" --iters "$ITERS"
done

# Copy latest result
if [[ $DRY_RUN -eq 0 ]]; then
  cp "$OUTPUT" "$PROJECT_ROOT/results/single_gpu_latest.csv"
fi

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
echo
echo "============================================================"
echo " Sweep complete"
echo " Runs attempted : $TOTAL_RUNS"
echo " Runs succeeded : $COMPLETED_RUNS"
echo " Failures       : $FAILURES"
if [[ $DRY_RUN -eq 0 ]]; then
  RESULT_LINES=$(grep -c '^RESULT,' "$OUTPUT" 2>/dev/null || echo 0)
  echo " RESULT rows    : $RESULT_LINES"
  echo " Output file    : $OUTPUT"
fi
echo "============================================================"

if [[ $FAILURES -gt 0 ]]; then
  echo "OVERALL: FAIL ($FAILURES failures)" >&2
  exit 1
else
  echo "OVERALL: PASS"
  exit 0
fi

#!/usr/bin/env bash
# scripts/run_core.sh
#
# One-command core runner for the Async-SSD benchmark.
# Builds the binary (single-GPU, no NCCL) and runs all three core modes
# with a smoke-test configuration.
#
# Usage:
#   bash scripts/run_core.sh [--full] [--binary /path/to/benchmark]
#
# Flags:
#   --full     Run a heavier configuration (more iters, larger N)
#   --binary   Use a pre-built binary instead of rebuilding
#
# Requirements: nvcc, g++, cuBLAS (libcublas). No NCCL needed.
#
# Exit code: 0 if everything passes, non-zero on any build or run failure.

set -euo pipefail

# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------
FULL=0
BINARY_OVERRIDE=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --full)     FULL=1; shift ;;
    --binary)   BINARY_OVERRIDE="$2"; shift 2 ;;
    --help|-h)
      sed -n '2,20p' "$0" | sed 's/^# \{0,1\}//'
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
# Paths
# ---------------------------------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

BINARY="${BINARY_OVERRIDE:-$PROJECT_ROOT/benchmark}"

# ---------------------------------------------------------------------------
# Smoke-test vs full configuration
# ---------------------------------------------------------------------------
if [[ $FULL -eq 1 ]]; then
  TARGET_MODEL="${TARGET_MODEL:-llama-8b}"
  DRAFT_MODEL="${DRAFT_MODEL:-llama-1b}"
  N="${N:-128}"
  WARMUP="${WARMUP:-5}"
  ITERS="${ITERS:-10}"
  K="${K:-4}"
  F="${F:-4}"
  ALPHA="${ALPHA:-0.9}"
  PHIT="${PHIT:-0.85}"
  echo "[config] FULL mode (heavier run)"
else
  TARGET_MODEL="${TARGET_MODEL:-tiny}"
  DRAFT_MODEL="${DRAFT_MODEL:-tiny}"
  N="${N:-8}"
  WARMUP="${WARMUP:-1}"
  ITERS="${ITERS:-2}"
  K="${K:-2}"
  F="${F:-2}"
  ALPHA="${ALPHA:-0.9}"
  PHIT="${PHIT:-0.85}"
  echo "[config] SMOKE-TEST mode (use --full for heavier run)"
fi

# ---------------------------------------------------------------------------
# GPU info
# ---------------------------------------------------------------------------
echo
echo "===== GPU info ====="
if command -v nvidia-smi &>/dev/null; then
  nvidia-smi --query-gpu=name,memory.total,driver_version \
             --format=csv,noheader 2>/dev/null \
    | awk '{print "[gpu]", NR-1":", $0}' \
    || nvidia-smi -L 2>/dev/null || true
else
  echo "[warn] nvidia-smi not found — GPU info unavailable"
fi
echo "===================="
echo

# ---------------------------------------------------------------------------
# Build (single-GPU, no NCCL)
# ---------------------------------------------------------------------------
if [[ -n "$BINARY_OVERRIDE" ]]; then
  echo "[build] Using pre-built binary: $BINARY"
  if [[ ! -x "$BINARY" ]]; then
    echo "ERROR: binary not found or not executable: $BINARY" >&2
    exit 1
  fi
else
  echo "[build] Building benchmark (USE_NCCL=0) ..."
  make USE_NCCL=0
  echo "[build] Done."
fi
echo

# ---------------------------------------------------------------------------
# Helper: run one benchmark invocation
# ---------------------------------------------------------------------------
FAILURES=0

run_mode() {
  local label="$1"; shift
  echo "--------------------------------------------------------------"
  echo "[run] $label"
  echo "      cmd: $BINARY $*"
  echo "--------------------------------------------------------------"
  if "$BINARY" "$@"; then
    echo "[OK] $label completed successfully"
  else
    local rc=$?
    echo "[FAIL] $label exited with code $rc" >&2
    (( FAILURES++ )) || true
  fi
  echo
}

# ---------------------------------------------------------------------------
# Mode: ar (autoregressive decode baseline)
# ---------------------------------------------------------------------------
run_mode "AR (autoregressive decode)" \
  --mode ar \
  --target-model "$TARGET_MODEL" \
  --N "$N" --warmup "$WARMUP" --iters "$ITERS"

# ---------------------------------------------------------------------------
# Mode: sd (speculative decoding)
# ---------------------------------------------------------------------------
run_mode "SD (speculative decoding)" \
  --mode sd \
  --target-model "$TARGET_MODEL" --draft-model "$DRAFT_MODEL" \
  --K "$K" --alpha "$ALPHA" \
  --N "$N" --warmup "$WARMUP" --iters "$ITERS"

# ---------------------------------------------------------------------------
# Mode: ssd-single (async SSD on one GPU)
# ---------------------------------------------------------------------------
run_mode "SSD-SINGLE (async SSD, single GPU)" \
  --mode ssd-single \
  --target-model "$TARGET_MODEL" --draft-model "$DRAFT_MODEL" \
  --K "$K" --F "$F" --alpha "$ALPHA" --phit "$PHIT" \
  --N "$N" --warmup "$WARMUP" --iters "$ITERS"

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
echo "============================================================"
if [[ $FAILURES -eq 0 ]]; then
  echo " All modes completed successfully."
  exit 0
else
  echo " $FAILURES mode(s) FAILED — check output above." >&2
  exit 1
fi

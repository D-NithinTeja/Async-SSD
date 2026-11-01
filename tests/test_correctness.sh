#!/usr/bin/env bash
# tests/test_correctness.sh
#
# Deterministic correctness tests for the benchmark binary.
# Each test checks exit codes and expected output patterns.
#
# Usage:
#   bash tests/test_correctness.sh [--binary /path/to/benchmark]
#
# Requirements: the benchmark binary must already be built (run `make` first).
#
# Exit code: 0 if all tests pass, 1 if any test failed.

set -uo pipefail

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

BINARY="${BINARY:-$PROJECT_ROOT/benchmark}"

# Smoke-test parameters: small/fast so they work without real weights.
TARGET_MODEL="${TARGET_MODEL:-tiny}"
DRAFT_MODEL="${DRAFT_MODEL:-tiny}"
ITERS="${ITERS:-2}"
WARMUP="${WARMUP:-1}"
N="${N:-8}"
K="${K:-2}"
F="${F:-2}"

# ---------------------------------------------------------------------------
# Test harness
# ---------------------------------------------------------------------------
PASS=0
FAIL=0
TESTS_RUN=0

pass() { echo "  [PASS] $1"; (( PASS++ )) || true; }
fail() { echo "  [FAIL] $1"; (( FAIL++ )) || true; }

run_test() {
  local name="$1"; shift
  (( TESTS_RUN++ )) || true
  echo
  echo "TEST: $name"
  "$@"
}

# Runs benchmark and captures stdout+stderr into variables.
# Sets: BM_STDOUT, BM_STDERR, BM_EXIT
run_bm() {
  local tmp_out tmp_err
  tmp_out="$(mktemp)"
  tmp_err="$(mktemp)"
  BM_EXIT=0
  "$BINARY" "$@" >"$tmp_out" 2>"$tmp_err" || BM_EXIT=$?
  BM_STDOUT="$(cat "$tmp_out")"
  BM_STDERR="$(cat "$tmp_err")"
  rm -f "$tmp_out" "$tmp_err"
}

# ---------------------------------------------------------------------------
# Preflight: binary exists
# ---------------------------------------------------------------------------
echo "============================================================"
echo " Async-SSD benchmark correctness tests"
echo "============================================================"
echo "Binary : $BINARY"
echo "Config : target=$TARGET_MODEL draft=$DRAFT_MODEL N=$N iters=$ITERS warmup=$WARMUP K=$K F=$F"
echo

if [[ ! -x "$BINARY" ]]; then
  echo "ERROR: benchmark binary not found or not executable: $BINARY"
  echo "       Run 'make' in the project root first."
  exit 1
fi

# ---------------------------------------------------------------------------
# Test 1: --csv-header prints a header and exits 0
# ---------------------------------------------------------------------------
run_test "csv-header prints header and exits 0" bash -c '
  run_bm --csv-header
  if [[ $BM_EXIT -ne 0 ]]; then
    fail "exit code $BM_EXIT, expected 0"
    return
  fi
  if echo "$BM_STDOUT" | grep -q "^tag,"; then
    pass "header line present (starts with tag,)"
  elif echo "$BM_STDOUT" | grep -qi "mode\|throughput\|target"; then
    pass "header line present (contains expected column names)"
  else
    fail "expected CSV header not found in output"
    echo "    stdout: $(echo "$BM_STDOUT" | head -3)"
  fi
'

# ---------------------------------------------------------------------------
# Test 2: invalid --mode returns non-zero exit code
# ---------------------------------------------------------------------------
run_test "invalid --mode bogus returns non-zero" bash -c '
  run_bm --mode bogus --target-model '"$TARGET_MODEL"' --N '"$N"'
  if [[ $BM_EXIT -ne 0 ]]; then
    pass "exited with code $BM_EXIT (non-zero) as expected"
  else
    fail "expected non-zero exit for --mode bogus, got 0"
  fi
'

# ---------------------------------------------------------------------------
# Test 3: --mode ar exits 0 and produces throughput output
# ---------------------------------------------------------------------------
run_test "mode ar: exits 0 and reports throughput" bash -c '
  run_bm --mode ar \
    --target-model '"$TARGET_MODEL"' \
    --N '"$N"' --warmup '"$WARMUP"' --iters '"$ITERS"'
  if [[ $BM_EXIT -ne 0 ]]; then
    fail "exit code $BM_EXIT, expected 0"
    echo "    stderr: $(echo "$BM_STDERR" | head -5)"
    return
  fi
  combined="$BM_STDOUT"$'"'"'\n'"'"'"$BM_STDERR"
  if echo "$combined" | grep -qi "throughput\|tok/s\|ms\|tokens"; then
    pass "output contains timing/throughput information"
  else
    fail "no throughput/timing info found in output"
    echo "    stdout: $(echo "$BM_STDOUT" | head -5)"
  fi
'

# ---------------------------------------------------------------------------
# Test 4: --mode ar --csv emits a RESULT, row
# ---------------------------------------------------------------------------
run_test "mode ar: --csv emits RESULT, row" bash -c '
  run_bm --mode ar \
    --target-model '"$TARGET_MODEL"' \
    --N '"$N"' --warmup '"$WARMUP"' --iters '"$ITERS"' \
    --csv
  if [[ $BM_EXIT -ne 0 ]]; then
    fail "exit code $BM_EXIT, expected 0"
    echo "    stderr: $(echo "$BM_STDERR" | head -5)"
    return
  fi
  if echo "$BM_STDOUT" | grep -q "^RESULT,"; then
    pass "RESULT, prefix found in CSV output"
  else
    fail "no RESULT, row found in stdout"
    echo "    stdout: $(echo "$BM_STDOUT" | head -5)"
  fi
'

# ---------------------------------------------------------------------------
# Test 5: --mode sd exits 0 and produces throughput output
# ---------------------------------------------------------------------------
run_test "mode sd: exits 0 and reports throughput" bash -c '
  run_bm --mode sd \
    --target-model '"$TARGET_MODEL"' --draft-model '"$DRAFT_MODEL"' \
    --K '"$K"' --alpha 0.9 \
    --N '"$N"' --warmup '"$WARMUP"' --iters '"$ITERS"'
  if [[ $BM_EXIT -ne 0 ]]; then
    fail "exit code $BM_EXIT, expected 0"
    echo "    stderr: $(echo "$BM_STDERR" | head -5)"
    return
  fi
  combined="$BM_STDOUT"$'"'"'\n'"'"'"$BM_STDERR"
  if echo "$combined" | grep -qi "throughput\|tok/s\|ms\|tokens"; then
    pass "output contains timing/throughput information"
  else
    fail "no throughput/timing info found"
    echo "    stdout: $(echo "$BM_STDOUT" | head -5)"
  fi
'

# ---------------------------------------------------------------------------
# Test 6: --mode sd --csv emits a RESULT, row
# ---------------------------------------------------------------------------
run_test "mode sd: --csv emits RESULT, row" bash -c '
  run_bm --mode sd \
    --target-model '"$TARGET_MODEL"' --draft-model '"$DRAFT_MODEL"' \
    --K '"$K"' --alpha 0.9 \
    --N '"$N"' --warmup '"$WARMUP"' --iters '"$ITERS"' \
    --csv
  if [[ $BM_EXIT -ne 0 ]]; then
    fail "exit code $BM_EXIT, expected 0"
    echo "    stderr: $(echo "$BM_STDERR" | head -5)"
    return
  fi
  if echo "$BM_STDOUT" | grep -q "^RESULT,"; then
    pass "RESULT, prefix found in CSV output"
  else
    fail "no RESULT, row found in stdout"
    echo "    stdout: $(echo "$BM_STDOUT" | head -5)"
  fi
'

# ---------------------------------------------------------------------------
# Test 7: --mode ssd-single exits 0 and produces throughput output
# ---------------------------------------------------------------------------
run_test "mode ssd-single: exits 0 and reports throughput" bash -c '
  run_bm --mode ssd-single \
    --target-model '"$TARGET_MODEL"' --draft-model '"$DRAFT_MODEL"' \
    --K '"$K"' --F '"$F"' --alpha 0.9 --phit 0.85 \
    --N '"$N"' --warmup '"$WARMUP"' --iters '"$ITERS"'
  if [[ $BM_EXIT -ne 0 ]]; then
    fail "exit code $BM_EXIT, expected 0"
    echo "    stderr: $(echo "$BM_STDERR" | head -5)"
    return
  fi
  combined="$BM_STDOUT"$'"'"'\n'"'"'"$BM_STDERR"
  if echo "$combined" | grep -qi "throughput\|tok/s\|ms\|tokens"; then
    pass "output contains timing/throughput information"
  else
    fail "no throughput/timing info found"
    echo "    stdout: $(echo "$BM_STDOUT" | head -5)"
  fi
'

# ---------------------------------------------------------------------------
# Test 8: --mode ssd-single --csv emits a RESULT, row
# ---------------------------------------------------------------------------
run_test "mode ssd-single: --csv emits RESULT, row" bash -c '
  run_bm --mode ssd-single \
    --target-model '"$TARGET_MODEL"' --draft-model '"$DRAFT_MODEL"' \
    --K '"$K"' --F '"$F"' --alpha 0.9 --phit 0.85 \
    --N '"$N"' --warmup '"$WARMUP"' --iters '"$ITERS"' \
    --csv
  if [[ $BM_EXIT -ne 0 ]]; then
    fail "exit code $BM_EXIT, expected 0"
    echo "    stderr: $(echo "$BM_STDERR" | head -5)"
    return
  fi
  if echo "$BM_STDOUT" | grep -q "^RESULT,"; then
    pass "RESULT, prefix found in CSV output"
  else
    fail "no RESULT, row found in stdout"
    echo "    stdout: $(echo "$BM_STDOUT" | head -5)"
  fi
'

# ---------------------------------------------------------------------------
# Test 9: CSV header columns match RESULT row column count
# ---------------------------------------------------------------------------
run_test "csv-header and RESULT row have the same column count" bash -c '
  header_line="$('"$BINARY"' --csv-header 2>/dev/null | head -1)"
  result_line="$('"$BINARY"' --mode ar \
      --target-model '"$TARGET_MODEL"' \
      --N '"$N"' --warmup '"$WARMUP"' --iters '"$ITERS"' \
      --csv 2>/dev/null | grep "^RESULT," | head -1)"

  if [[ -z "$header_line" || -z "$result_line" ]]; then
    fail "could not get header or result line for comparison"
    return
  fi

  hcols=$(echo "$header_line" | awk -F, "{print NF}")
  rcols=$(echo "$result_line" | awk -F, "{print NF}")

  if [[ "$hcols" -eq "$rcols" ]]; then
    pass "header and RESULT row both have $hcols columns"
  else
    fail "column count mismatch: header=$hcols, RESULT row=$rcols"
    echo "    header : $header_line"
    echo "    result : $result_line"
  fi
'

# ---------------------------------------------------------------------------
# Test 10: --mode ssd-single --tp 2 returns non-zero (rejects invalid TP)
# ---------------------------------------------------------------------------
run_test "mode ssd-single rejects --tp 2 with non-zero exit" bash -c '
  run_bm --mode ssd-single --tp 2 \
    --target-model '"$TARGET_MODEL"' --draft-model '"$DRAFT_MODEL"' \
    --N '"$N"' --warmup '"$WARMUP"' --iters '"$ITERS"'
  if [[ $BM_EXIT -ne 0 ]]; then
    pass "exited with code $BM_EXIT (non-zero) as expected"
  else
    fail "expected non-zero exit for --tp 2 with ssd-single, got 0"
  fi
'

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
echo
echo "============================================================"
echo " Results: $PASS/$TESTS_RUN passed, $FAIL failed"
echo "============================================================"

if [[ $FAIL -gt 0 ]]; then
  echo "OVERALL: FAIL"
  exit 1
else
  echo "OVERALL: PASS"
  exit 0
fi


Build:

```bash
make clean
make
```

Check the new CLI mode is exposed:

```bash
./benchmark --help
```

You should see `ssd-single` listed under `--mode`.

Run a quick, low-cost smoke test:

```bash
./benchmark \
  --mode ssd-single \
  --target-model tiny \
  --draft-model tiny \
  --K 2 \
  --F 2 \
  --N 8 \
  --warmup 1 \
  --iters 1
```

Then run the requested configuration:

```bash
./benchmark \
  --mode ssd-single \
  --target-model llama-1b \
  --draft-model tiny
```

Check CSV output compatibility:

```bash
./benchmark \
  --mode ssd-single \
  --target-model tiny \
  --draft-model tiny \
  --K 2 --F 2 --N 8 --warmup 1 --iters 1 \
  --csv
```

Check queue/scheduling behavior through expected miss-rate extremes:

```bash
# Every pre-speculation is a cache hit: expected miss rate ~0.
./benchmark --mode ssd-single --target-model tiny --draft-model tiny \
  --K 2 --F 2 --N 32 --warmup 1 --iters 5 --phit 1.0

# Every pre-speculation misses: expected miss rate ~1.
./benchmark --mode ssd-single --target-model tiny --draft-model tiny \
  --K 2 --F 2 --N 32 --warmup 1 --iters 5 --phit 0.0
```

Check acceptance behavior:

```bash
# No speculative tokens accepted: roughly one token per round.
./benchmark --mode ssd-single --target-model tiny --draft-model tiny \
  --K 4 --F 2 --N 32 --warmup 1 --iters 5 --alpha 0.0

# All K speculative tokens accepted: roughly K+1 tokens per round.
./benchmark --mode ssd-single --target-model tiny --draft-model tiny \
  --K 4 --F 2 --N 32 --warmup 1 --iters 5 --alpha 1.0
```

Finally, confirm the original implementation is unaffected (requires its usual multi-GPU/NCCL build):

```bash
make clean
make USE_NCCL=1
./benchmark --mode ssd --target-model llama-1b --draft-model tiny --tp 1
```

The single-GPU mode intentionally rejects invalid TP settings:

```bash
./benchmark --mode ssd-single --tp 2
```

Expected: `Error: mode 'ssd-single' requires --tp 1`.
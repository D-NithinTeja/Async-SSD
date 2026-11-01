# analysis/

This directory contains tools for processing benchmark logs, generating plots from raw experimental data, and analyzing Nsight Systems profiling traces.

---

## Files

| File | Purpose |
|:---|:---|
| `plot_results.py` | Generates SVG/PNG benchmark plots and summary tables from `results/*.csv` |
| `parse_profile.py` | Inspects `ssd_single.sqlite` and computes stream concurrency & kernel overlap |
| `plots/` | Output directory containing generated vector plots (`.svg`) |

---

## 1. Generating Benchmark Plots

Run `plot_results.py` to parse all experimental sweeps under `results/`:

```bash
python3 analysis/plot_results.py
```

### Options
- `--results-dir <path>`: Directory containing CSV results (default: `results/`)
- `--output-dir <path>`: Directory to save generated plots (default: `analysis/plots/`)

### Generated Plots
1. **`01_throughput_vs_alpha.svg`**: Throughput (tokens/s) vs token acceptance rate ($\alpha \in [0.3, 1.0]$) for AR, SD, and Async-SSD.
2. **`02_throughput_vs_K.svg`**: Throughput scaling vs speculation length ($K \in [1, 8]$) comparing standard SD against Async-SSD.
3. **`03_throughput_vs_phit.svg`**: Impact of pre-speculation cache hit rate ($p_{\text{hit}} \in [0.3, 0.95]$) on Async-SSD throughput.
4. **`04_throughput_vs_fanout.svg`**: Effect of branch pre-speculation fan-out ($F \in [1, 4]$) across different cache hit regimes.
5. **`05_latency_breakdown.svg`**: Component latency breakdown comparing draft generation, pre-speculation fanout, and batched target verification.

---

## 2. Profiling & Stream Overlap Analysis

Run `parse_profile.py` on an Nsight Systems SQLite database to inspect execution timelines and kernel overlap:

```bash
python3 analysis/parse_profile.py --db ssd_single.sqlite
```

### What It Measures
- **Stream Execution Summary**: Kernel counts, overall timeline span, and mean kernel duration per stream.
- **Concurrent Overlap**: Number of overlapping kernel pairs and cumulative concurrent execution time between worker streams.
- **Synchronization Overhead**: Cumulative wait time across host threads spent in `cudaEventSynchronize` and `cudaStreamSynchronize`.

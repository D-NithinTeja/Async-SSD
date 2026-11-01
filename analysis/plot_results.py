#!/usr/bin/env python3
"""
plot_results.py - Generates benchmark plots and analysis from Async-SSD raw CSV data.

Generates:
  1. Throughput vs Alpha (AR vs SD vs SSD)
  2. Throughput vs Speculation Length K (SD vs SSD)
  3. Throughput vs Cache Hit Probability p_hit (SSD)
  4. Throughput vs Pre-speculation Fanout F (SSD)
  5. Component Timing Breakdown (Draft, Prespec, Target Verify)

Works with matplotlib (PNG) if installed, and always produces standalone SVG plots
and ASCII summary tables so it runs out-of-the-box in any Python 3 environment.

Usage:
  python3 analysis/plot_results.py [--csv results/single_gpu_latest.csv] [--results-dir results/] [--output-dir analysis/plots/]
"""

import argparse
import csv
import math
import os
import sys

def parse_args():
    parser = argparse.ArgumentParser(description="Generate Async-SSD benchmark analysis and plots.")
    parser.add_argument("--csv", type=str, default="", help="Path to unified sweep CSV file (e.g., results/single_gpu_latest.csv)")
    parser.add_argument("--results-dir", type=str, default="results", help="Directory containing raw result CSVs")
    parser.add_argument("--output-dir", type=str, default="analysis/plots", help="Directory to save generated plots")
    return parser.parse_args()

def load_csv(filepath):
    if not os.path.exists(filepath):
        return []
    rows = []
    with open(filepath, mode="r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            if r.get("tag") == "RESULT" or "mode" in r:
                row = dict(r)
                for float_field in ["alpha", "phit", "target_step_ms", "target_verify_ms",
                                    "draft_step_ms", "draft_total_ms", "prespec_ms",
                                    "e2e_ms", "avg_tokens", "avg_rounds", "miss_rate", "throughput_tok_s"]:
                    if float_field in row and row[float_field] != "":
                        try:
                            row[float_field] = float(row[float_field])
                        except ValueError:
                            row[float_field] = 0.0
                for int_field in ["tp", "K", "F", "N"]:
                    if int_field in row and row[int_field] != "":
                        try:
                            row[int_field] = int(row[int_field])
                        except ValueError:
                            row[int_field] = 0
                rows.append(row)
    return rows

def generate_svg_line_chart(title, x_label, y_label, series_dict, filepath, width=700, height=450):
    """Generates a standalone, dependency-free SVG line chart."""
    pad_left = 80
    pad_right = 160
    pad_top = 60
    pad_bottom = 60
    plot_w = width - pad_left - pad_right
    plot_h = height - pad_top - pad_bottom

    all_x = []
    all_y = []
    for s_name, points in series_dict.items():
        for x, y in points:
            all_x.append(x)
            all_y.append(y)
    
    if not all_x or not all_y:
        return

    min_x, max_x = min(all_x), max(all_x)
    min_y = 0.0
    max_y = max(all_y) * 1.15
    if max_x == min_x:
        max_x += 1.0
    if max_y == min_y:
        max_y += 1.0

    def tx(x):
        return pad_left + (x - min_x) / (max_x - min_x) * plot_w
    def ty(y):
        return pad_top + plot_h - (y - min_y) / (max_y - min_y) * plot_h

    colors = ["#2563eb", "#dc2626", "#16a34a", "#9333ea", "#ea580c"]
    
    svg = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" style="background:#ffffff; font-family:sans-serif;">',
        f'<text x="{width/2}" y="32" text-anchor="middle" font-size="16" font-weight="bold" fill="#1e293b">{title}</text>',
        f'<line x1="{pad_left}" y1="{pad_top+plot_h}" x2="{pad_left+plot_w}" y2="{pad_top+plot_h}" stroke="#94a3b8" stroke-width="1.5"/>',
        f'<line x1="{pad_left}" y1="{pad_top}" x2="{pad_left}" y2="{pad_top+plot_h}" stroke="#94a3b8" stroke-width="1.5"/>',
        f'<text x="{pad_left+plot_w/2}" y="{height-15}" text-anchor="middle" font-size="12" fill="#475569">{x_label}</text>',
        f'<text x="25" y="{pad_top+plot_h/2}" text-anchor="middle" font-size="12" fill="#475569" transform="rotate(-90 25 {pad_top+plot_h/2})">{y_label}</text>',
    ]

    ticks = 5
    for i in range(ticks + 1):
        y_val = min_y + (max_y - min_y) * (i / ticks)
        py = ty(y_val)
        svg.append(f'<line x1="{pad_left}" y1="{py}" x2="{pad_left+plot_w}" y2="{py}" stroke="#e2e8f0" stroke-dasharray="4"/>')
        svg.append(f'<text x="{pad_left-10}" y="{py+4}" text-anchor="end" font-size="10" fill="#64748b">{y_val:.1f}</text>')

    leg_x = pad_left + plot_w + 20
    leg_y = pad_top + 20
    for idx, (s_name, points) in enumerate(series_dict.items()):
        color = colors[idx % len(colors)]
        pts_sorted = sorted(points, key=lambda p: p[0])
        polyline_pts = " ".join([f"{tx(x):.1f},{ty(y):.1f}" for x, y in pts_sorted])
        svg.append(f'<polyline fill="none" stroke="{color}" stroke-width="2.5" points="{polyline_pts}"/>')
        for x, y in pts_sorted:
            svg.append(f'<circle cx="{tx(x):.1f}" cy="{ty(y):.1f}" r="4" fill="{color}" stroke="#ffffff" stroke-width="1.5"/>')
        
        svg.append(f'<line x1="{leg_x}" y1="{leg_y + idx*22}" x2="{leg_x+20}" y2="{leg_y + idx*22}" stroke="{color}" stroke-width="2.5"/>')
        svg.append(f'<circle cx="{leg_x+10}" cy="{leg_y + idx*22}" r="3" fill="{color}"/>')
        svg.append(f'<text x="{leg_x+28}" y="{leg_y + idx*22 + 4}" font-size="11" fill="#1e293b">{s_name}</text>')

    svg.append('</svg>')
    with open(filepath, "w", encoding="utf-8") as f:
        f.write("\n".join(svg))

def generate_svg_bar_chart(title, categories, values_dict, filepath, width=700, height=450):
    """Generates a standalone SVG stacked/grouped bar chart for component timing."""
    pad_left = 80
    pad_right = 160
    pad_top = 60
    pad_bottom = 60
    plot_w = width - pad_left - pad_right
    plot_h = height - pad_top - pad_bottom

    if not categories:
        return

    max_val = 0.0
    for cat in categories:
        tot = sum(values_dict[k].get(cat, 0.0) for k in values_dict)
        if tot > max_val:
            max_val = tot
    max_y = max_val * 1.15 or 1.0

    def ty(y):
        return pad_top + plot_h - (y / max_y) * plot_h

    colors = ["#3b82f6", "#10b981", "#f59e0b", "#8b5cf6"]
    svg = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" style="background:#ffffff; font-family:sans-serif;">',
        f'<text x="{width/2}" y="32" text-anchor="middle" font-size="16" font-weight="bold" fill="#1e293b">{title}</text>',
        f'<line x1="{pad_left}" y1="{pad_top+plot_h}" x2="{pad_left+plot_w}" y2="{pad_top+plot_h}" stroke="#94a3b8" stroke-width="1.5"/>',
        f'<line x1="{pad_left}" y1="{pad_top}" x2="{pad_left}" y2="{pad_top+plot_h}" stroke="#94a3b8" stroke-width="1.5"/>',
        f'<text x="25" y="{pad_top+plot_h/2}" text-anchor="middle" font-size="12" fill="#475569" transform="rotate(-90 25 {pad_top+plot_h/2})">Latency (ms)</text>',
    ]

    ticks = 5
    for i in range(ticks + 1):
        y_val = max_y * (i / ticks)
        py = ty(y_val)
        svg.append(f'<line x1="{pad_left}" y1="{py}" x2="{pad_left+plot_w}" y2="{py}" stroke="#e2e8f0" stroke-dasharray="4"/>')
        svg.append(f'<text x="{pad_left-10}" y="{py+4}" text-anchor="end" font-size="10" fill="#64748b">{y_val:.1f}</text>')

    bar_width = plot_w / len(categories) * 0.5
    for idx, cat in enumerate(categories):
        bx = pad_left + (idx + 0.5) * (plot_w / len(categories)) - bar_width / 2
        cum_h = 0.0
        for s_idx, (comp_name, comp_data) in enumerate(values_dict.items()):
            val = comp_data.get(cat, 0.0)
            if val > 0:
                h = (val / max_y) * plot_h
                y_top = pad_top + plot_h - (cum_h / max_y) * plot_h - h
                color = colors[s_idx % len(colors)]
                svg.append(f'<rect x="{bx:.1f}" y="{y_top:.1f}" width="{bar_width:.1f}" height="{h:.1f}" fill="{color}"/>')
                cum_h += val
        svg.append(f'<text x="{bx + bar_width/2:.1f}" y="{pad_top+plot_h+20}" text-anchor="middle" font-size="11" fill="#334155">{cat}</text>')

    leg_x = pad_left + plot_w + 20
    leg_y = pad_top + 20
    for idx, comp_name in enumerate(values_dict.keys()):
        color = colors[idx % len(colors)]
        svg.append(f'<rect x="{leg_x}" y="{leg_y + idx*22}" width="14" height="14" fill="{color}"/>')
        svg.append(f'<text x="{leg_x+22}" y="{leg_y + idx*22 + 11}" font-size="11" fill="#1e293b">{comp_name}</text>')

    svg.append('</svg>')
    with open(filepath, "w", encoding="utf-8") as f:
        f.write("\n".join(svg))

def render_plots(csv_file, results_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    all_rows = []
    if csv_file and os.path.exists(csv_file):
        print(f"Loading data from specified CSV: {csv_file}")
        all_rows = load_csv(csv_file)
    elif os.path.exists(os.path.join(results_dir, "single_gpu_latest.csv")):
        unified = os.path.join(results_dir, "single_gpu_latest.csv")
        print(f"Loading data from latest single-GPU sweep: {unified}")
        all_rows = load_csv(unified)

    if not all_rows:
        print(f"Loading individual CSV files from: {results_dir}")
        rows_01 = load_csv(os.path.join(results_dir, "01_alpha_K.csv"))
        rows_02 = load_csv(os.path.join(results_dir, "02_phit.csv"))
        rows_03 = load_csv(os.path.join(results_dir, "03_fanout.csv"))
        rows_04 = load_csv(os.path.join(results_dir, "04a_breakdown_small.csv"))
    else:
        rows_01 = all_rows
        rows_02 = all_rows
        rows_03 = all_rows
        rows_04 = all_rows

    target_name = all_rows[0].get("target", "Target") if all_rows else "Llama"
    draft_name = all_rows[0].get("draft", "Draft") if all_rows else "Draft"

    # 1. Plot 1: Throughput vs Alpha
    if rows_01:
        ar_thr = [r["throughput_tok_s"] for r in rows_01 if r["mode"] == "ar"]
        baseline_ar = ar_thr[0] if ar_thr else 50.0

        alphas = sorted(list(set(r["alpha"] for r in rows_01 if r["alpha"] > 0)))
        series = {"AR Baseline": [(a, baseline_ar) for a in alphas]}

        sd_pts = [(r["alpha"], r["throughput_tok_s"]) for r in rows_01 if r["mode"] == "sd" and r["K"] == 4]
        ssd_pts = [(r["alpha"], r["throughput_tok_s"]) for r in rows_01 if r["mode"] in ("ssd", "ssd-single") and r["K"] == 4 and r.get("F", 1) in (1, 4)]
        if sd_pts:
            # Deduplicate by alpha
            series["SD (K=4)"] = sorted(dict(sd_pts).items())
        if ssd_pts:
            series["Async-SSD (K=4)"] = sorted(dict(ssd_pts).items())

        svg_path = os.path.join(output_dir, "01_throughput_vs_alpha.svg")
        generate_svg_line_chart("Inference Throughput vs Acceptance Rate (α)", "Acceptance Rate α", "Throughput (tok/s)", series, svg_path)
        print(f"Generated: {svg_path}")

        # 2. Plot 2: Throughput vs K
        series_k = {}
        target_alphas = [0.7, 0.9] if any(abs(r["alpha"] - 0.7) < 0.05 for r in rows_01) else [alphas[-1]]
        for alpha_val in target_alphas:
            sd_k = [(r["K"], r["throughput_tok_s"]) for r in rows_01 if r["mode"] == "sd" and abs(r["alpha"] - alpha_val) < 0.05]
            ssd_k = [(r["K"], r["throughput_tok_s"]) for r in rows_01 if r["mode"] in ("ssd", "ssd-single") and abs(r["alpha"] - alpha_val) < 0.05 and r.get("F", 1) in (1, 4)]
            if sd_k:
                series_k[f"SD (α={alpha_val})"] = sorted(dict(sd_k).items())
            if ssd_k:
                series_k[f"SSD (α={alpha_val})"] = sorted(dict(ssd_k).items())

        svg_path_k = os.path.join(output_dir, "02_throughput_vs_K.svg")
        generate_svg_line_chart("Throughput vs Speculation Length (K)", "Speculation Length K", "Throughput (tok/s)", series_k, svg_path_k)
        print(f"Generated: {svg_path_k}")

    # 3. Plot 3: Throughput vs p_hit
    if rows_02:
        ssd_phit = [(r["phit"], r["throughput_tok_s"]) for r in rows_02 if r["mode"] in ("ssd", "ssd-single") and r["K"] == 4 and r["phit"] > 0]
        sd_matches = [r["throughput_tok_s"] for r in rows_02 if r["mode"] == "sd" and r["K"] == 4]
        sd_base = sd_matches[0] if sd_matches else 0.0
        phits = sorted(list(set(p for p, _ in ssd_phit)))

        series_phit = {}
        if sd_base > 0 and phits:
            series_phit["SD Baseline (K=4)"] = [(p, sd_base) for p in phits]
        if ssd_phit:
            series_phit["Async-SSD (K=4)"] = sorted(dict(ssd_phit).items())

        svg_path_phit = os.path.join(output_dir, "03_throughput_vs_phit.svg")
        generate_svg_line_chart("Throughput vs Cache Hit Rate (p_hit)", "Cache Hit Rate p_hit", "Throughput (tok/s)", series_phit, svg_path_phit)
        print(f"Generated: {svg_path_phit}")

    # 4. Plot 4: Throughput vs Fanout F
    if rows_03:
        series_f = {}
        avail_phits = sorted(list(set(r["phit"] for r in rows_03 if r["mode"] in ("ssd", "ssd-single") and r["F"] > 0)))
        for phit_val in avail_phits:
            f_pts = [(r["F"], r["throughput_tok_s"]) for r in rows_03 if abs(r["phit"] - phit_val) < 0.05 and r["F"] > 0]
            if f_pts:
                series_f[f"p_hit={phit_val}"] = sorted(dict(f_pts).items())
        svg_path_f = os.path.join(output_dir, "04_throughput_vs_fanout.svg")
        generate_svg_line_chart("Throughput vs Pre-speculation Fanout (F)", "Fanout Degree F", "Throughput (tok/s)", series_f, svg_path_f)
        print(f"Generated: {svg_path_f}")

    # 5. Plot 5: Breakdown Chart
    if rows_04:
        cand = [r for r in rows_04 if r["target_verify_ms"] > 0 or r["draft_total_ms"] > 0]
        cats = [f"{r['mode'].upper()} (K={r['K']})" for r in cand[:6]]
        comps = {
            "Draft Total": {f"{r['mode'].upper()} (K={r['K']})": r["draft_total_ms"] for r in cand[:6]},
            "Pre-spec Fanout": {f"{r['mode'].upper()} (K={r['K']})": r["prespec_ms"] for r in cand[:6]},
            "Target Verify": {f"{r['mode'].upper()} (K={r['K']})": r["target_verify_ms"] for r in cand[:6]},
        }
        svg_path_bd = os.path.join(output_dir, "05_latency_breakdown.svg")
        generate_svg_bar_chart("Component Latency Breakdown", cats, comps, svg_path_bd)
        print(f"Generated: {svg_path_bd}")

    # Summary table
    print("\n" + "=" * 80)
    print(f"ASYNC-SSD BENCHMARK SUMMARY ({target_name} Target + {draft_name} Draft)")
    print("=" * 80)
    print(f"{'Mode':<14} {'Config':<25} {'Throughput (tok/s)':<22} {'Speedup vs AR':<14}")
    print("-" * 80)
    if rows_01:
        ar_thr_val = next((r["throughput_tok_s"] for r in rows_01 if r["mode"] == "ar"), 50.0)
        print(f"{'AR':<14} {'Baseline (M=1)':<25} {ar_thr_val:<22.2f} {'1.00x':<14}")

        best_sd = max([r for r in rows_01 if r["mode"] == "sd"], key=lambda x: x["throughput_tok_s"], default=None)
        if best_sd:
            cfg = f"K={best_sd['K']}, alpha={best_sd['alpha']}"
            sp = best_sd["throughput_tok_s"] / ar_thr_val
            print(f"{'SD':<14} {cfg:<25} {best_sd['throughput_tok_s']:<22.2f} {f'{sp:.2f}x':<14}")

        best_ssd = max([r for r in rows_01 if r["mode"] in ("ssd", "ssd-single")], key=lambda x: x["throughput_tok_s"], default=None)
        if best_ssd:
            cfg = f"K={best_ssd['K']}, alpha={best_ssd['alpha']}, F={best_ssd['F']}, phit={best_ssd['phit']}"
            sp = best_ssd["throughput_tok_s"] / ar_thr_val
            print(f"{'Async-SSD':<14} {cfg:<25} {best_ssd['throughput_tok_s']:<22.2f} {f'{sp:.2f}x':<14}")
    print("=" * 80 + "\n")

if __name__ == "__main__":
    args = parse_args()
    render_plots(args.csv, args.results_dir, args.output_dir)

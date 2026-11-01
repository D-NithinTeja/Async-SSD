#!/usr/bin/env python3
"""
parse_profile.py - Analyzes Nsight Systems SQLite exports for Async-SSD.

Extracts CUDA kernel executions, stream concurrency, and kernel overlap metrics
from an Nsight Systems SQLite database (e.g., ssd_single.sqlite).

Usage:
    python3 analysis/parse_profile.py [--db PATH_TO_SQLITE]
"""

import argparse
import os
import sqlite3
import sys

def parse_args():
    parser = argparse.ArgumentParser(
        description="Analyze CUDA kernel overlap and stream concurrency from Nsight Systems SQLite."
    )
    parser.add_argument(
        "--db",
        type=str,
        default="ssd_single.sqlite",
        help="Path to Nsight Systems SQLite database (default: ssd_single.sqlite)",
    )
    return parser.parse_args()

def analyze_profile(db_path):
    if not os.path.exists(db_path):
        print(f"Error: Database file not found: {db_path}", file=sys.stderr)
        sys.exit(1)

    print(f"Opening Nsight Systems profile: {db_path}")
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()

    # Verify kernel activity table exists
    cur.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='CUPTI_ACTIVITY_KIND_KERNEL';")
    if not cur.fetchone():
        print("Error: Table 'CUPTI_ACTIVITY_KIND_KERNEL' not found in database.", file=sys.stderr)
        conn.close()
        sys.exit(1)

    # 1. Kernel statistics per stream
    query_streams = """
    SELECT 
        streamId, 
        COUNT(*) as kernel_count, 
        MIN(start) as min_start_ns, 
        MAX(end) as max_end_ns,
        SUM(end - start) as total_exec_ns
    FROM CUPTI_ACTIVITY_KIND_KERNEL
    GROUP BY streamId
    ORDER BY kernel_count DESC;
    """
    cur.execute(query_streams)
    stream_rows = cur.fetchall()

    print("\n" + "=" * 70)
    print("STREAM EXECUTION SUMMARY")
    print("=" * 70)
    print(f"{'Stream ID':<12} {'Kernels':<12} {'Span (ms)':<14} {'Total Work (ms)':<16} {'Avg Kernel (us)':<14}")
    print("-" * 70)

    active_streams = []
    for s_id, count, min_s, max_s, total_ns in stream_rows:
        span_ms = (max_s - min_s) / 1e6
        work_ms = total_ns / 1e6
        avg_us = (total_ns / count) / 1e3
        print(f"{s_id:<12} {count:<12} {span_ms:<14.2f} {work_ms:<16.2f} {avg_us:<14.2f}")
        if count > 1000:
            active_streams.append(s_id)

    print("=" * 70)

    # 2. Kernel overlap analysis between the primary worker streams
    if len(active_streams) >= 2:
        s1, s2 = active_streams[0], active_streams[1]
        print(f"\nAnalyzing kernel overlap between primary worker streams (Stream {s1} and Stream {s2})...")

        cur.execute("SELECT start, end FROM CUPTI_ACTIVITY_KIND_KERNEL WHERE streamId = ? ORDER BY start", (s1,))
        k1 = cur.fetchall()
        cur.execute("SELECT start, end FROM CUPTI_ACTIVITY_KIND_KERNEL WHERE streamId = ? ORDER BY start", (s2,))
        k2 = cur.fetchall()

        overlaps = 0
        total_overlap_ns = 0
        i = 0
        j = 0
        while i < len(k1) and j < len(k2):
            s_1, e_1 = k1[i]
            s_2, e_2 = k2[j]
            o_start = max(s_1, s_2)
            o_end = min(e_1, e_2)
            if o_start < o_end:
                overlaps += 1
                total_overlap_ns += (o_end - o_start)
            if e_1 < e_2:
                i += 1
            else:
                j += 1

        overall_start = min(k1[0][0], k2[0][0])
        overall_end = max(k1[-1][1], k2[-1][1])
        overall_span_ms = (overall_end - overall_start) / 1e6
        overlap_ms = total_overlap_ns / 1e6
        overlap_pct = (total_overlap_ns / (overall_end - overall_start)) * 100.0

        print("\n" + "=" * 70)
        print("CONCURRENT KERNEL OVERLAP ANALYSIS")
        print("=" * 70)
        print(f"Overall timeline duration:      {overall_span_ms:.2f} ms")
        print(f"Stream {s1} kernel count:         {len(k1)}")
        print(f"Stream {s2} kernel count:         {len(k2)}")
        print(f"Overlapping kernel pairs:       {overlaps}")
        print(f"Total concurrent execution time: {overlap_ms:.2f} ms")
        print(f"Timeline overlap duty cycle:    {overlap_pct:.2f}%")
        print("=" * 70)
    else:
        print("\nNote: Less than 2 active worker streams detected in this database.")

    # 3. Synchronization events summary
    cur.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='CUPTI_ACTIVITY_KIND_SYNCHRONIZATION';")
    if cur.fetchone():
        cur.execute("SELECT COUNT(*), SUM(end - start) FROM CUPTI_ACTIVITY_KIND_SYNCHRONIZATION;")
        sync_count, sync_time_ns = cur.fetchone()
        sync_time_ms = (sync_time_ns or 0) / 1e6
        print(f"\nCUDA Synchronization Events: {sync_count} calls totaling {sync_time_ms:.2f} ms")

    conn.close()

if __name__ == "__main__":
    args = parse_args()
    analyze_profile(args.db)

#!/usr/bin/env python3
"""Detailed Statistical Breakdown of the 360-Run Grand Mega-Benchmark (100 to 1000 Customers)."""

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
CSV_PATH = ROOT / "results" / "mega_benchmark" / "mega_benchmark_raw.csv"

df = pd.read_csv(CSV_PATH)

print("=" * 85)
print("GRAND MEGA-BENCHMARK: 6-SCALE COMPREHENSIVE PERFORMANCE (360 INDEPENDENT RUNS)")
print("=" * 85)

# Group by scale and solver
scales = ["Solomon-100", "Homberger-200", "Homberger-400", "Homberger-600", "Homberger-800", "Homberger-1000"]

for scale in scales:
    sub = df[df["scale"] == scale]
    alns = sub[sub["solver"] == "ALNS-Base"]
    hybrid = sub[sub["solver"] == "Hybrid-DDQN"]

    mean_nv_alns = alns["nv"].mean()
    mean_nv_hyb = hybrid["nv"].mean()
    mean_td_alns = alns["cost"].mean()
    mean_td_hyb = hybrid["cost"].mean()

    # Calculate TD savings across all instances
    td_saving = (mean_td_alns - mean_td_hyb) / mean_td_alns * 100.0

    print(f"\n--- SCALE: {scale:15s} ---")
    print(
        f"  ALNS-Base   : NV = {mean_nv_alns:6.2f} veh | TD = {mean_td_alns:10.2f} km | Time = {alns['time'].mean():5.1f}s"
    )
    print(
        f"  Hybrid-DDQN : NV = {mean_nv_hyb:6.2f} veh | TD = {mean_td_hyb:10.2f} km | Time = {hybrid['time'].mean():5.1f}s"
    )
    print(f"  TD Difference: {td_saving:+.2f}% (Hybrid-DDQN vs ALNS-Base)")

    # Per-instance breakdown
    insts = sub["instance"].unique()
    print(f"  {'Instance':8s} | {'ALNS NV':8s} | {'Hyb NV':8s} | {'ALNS TD':10s} | {'Hyb TD':10s} | {'Delta TD':10s}")
    print("  " + "-" * 65)
    for inst in insts:
        a_i = alns[alns["instance"] == inst]
        h_i = hybrid[hybrid["instance"] == inst]
        a_nv, h_nv = a_i["nv"].mean(), h_i["nv"].mean()
        a_td, h_td = a_i["cost"].mean(), h_i["cost"].mean()
        d_td = (h_td - a_td) / a_td * 100.0 if a_nv == h_nv else float("nan")
        d_str = f"{d_td:+.2f}%" if not np.isnan(d_td) else f"[NV: {h_nv - a_nv:+.1f}]"
        print(f"  {inst:8s} | {a_nv:8.2f} | {h_nv:8.2f} | {a_td:10.2f} | {h_td:10.2f} | {d_str:10s}")

print("\n" + "=" * 85)

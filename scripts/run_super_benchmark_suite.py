#!/usr/bin/env python3
"""Comprehensive Super-Benchmark Suite: Solomon-100, Homberger-200, and Homberger-400.

Evaluates ALNS-Base vs Tri-Level Hybrid DDQN-ALNS under strictly isolated independent cold-starts.
Instances: 18 representative topologies (6 classes x 3 scales) x 5 seeds x 2 solvers = 180 runs.
"""

from __future__ import annotations

import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import pandas as pd

from vrptw.config import BKS, Config
from vrptw.core import load_solomon_instance
from vrptw.solvers import ALNSSolver, HybridDDQNSolver

ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "results" / "super_benchmark_suite"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

CORE_BENCHMARK_MATRIX = [
    # Solomon 100
    ("Solomon-100", "C101", "data/Solomon/C101.txt"),
    ("Solomon-100", "C201", "data/Solomon/C201.txt"),
    ("Solomon-100", "R101", "data/Solomon/R101.txt"),
    ("Solomon-100", "R201", "data/Solomon/R201.txt"),
    ("Solomon-100", "RC101", "data/Solomon/RC101.txt"),
    ("Solomon-100", "RC201", "data/Solomon/RC201.txt"),
    # Homberger 200
    ("Homberger-200", "c1_2_1", "data/Gehring_Homberger/homberger_200_customer_instances/C1_2_1.TXT"),
    ("Homberger-200", "c2_2_1", "data/Gehring_Homberger/homberger_200_customer_instances/C2_2_1.TXT"),
    ("Homberger-200", "r1_2_1", "data/Gehring_Homberger/homberger_200_customer_instances/R1_2_1.TXT"),
    ("Homberger-200", "r2_2_1", "data/Gehring_Homberger/homberger_200_customer_instances/R2_2_1.TXT"),
    ("Homberger-200", "rc1_2_1", "data/Gehring_Homberger/homberger_200_customer_instances/RC1_2_1.TXT"),
    ("Homberger-200", "rc2_2_1", "data/Gehring_Homberger/homberger_200_customer_instances/RC2_2_1.TXT"),
    # Homberger 400
    ("Homberger-400", "c1_4_1", "data/Gehring_Homberger/homberger_400_customer_instances/C1_4_1.TXT"),
    ("Homberger-400", "c2_4_1", "data/Gehring_Homberger/homberger_400_customer_instances/C2_4_1.TXT"),
    ("Homberger-400", "r1_4_1", "data/Gehring_Homberger/homberger_400_customer_instances/R1_4_1.TXT"),
    ("Homberger-400", "r2_4_1", "data/Gehring_Homberger/homberger_400_customer_instances/R2_4_1.TXT"),
    ("Homberger-400", "rc1_4_1", "data/Gehring_Homberger/homberger_400_customer_instances/RC1_4_1.TXT"),
    ("Homberger-400", "rc2_4_1", "data/Gehring_Homberger/homberger_400_customer_instances/RC2_4_1.TXT"),
]

SEEDS = [42, 43, 44, 45, 46]


def run_single_eval(scale: str, inst_name: str, rel_path: str, seed: int, solver_type: str) -> dict:
    inst_path = str(ROOT / rel_path)
    inst = load_solomon_instance(inst_path)

    cfg = Config()
    cfg.seed = seed

    if solver_type == "ALNS-Base":
        solver = ALNSSolver(inst, cfg)
    else:
        solver = HybridDDQNSolver(inst, cfg, seed=seed)

    t0 = time.perf_counter()
    plan, _ = solver.solve(seed=seed)
    wall_time = time.perf_counter() - t0

    bks = BKS.get(inst_name, {})
    bks_nv = bks.get("nv", None)
    bks_td = bks.get("td", None)

    return {
        "scale": scale,
        "instance": inst_name,
        "seed": seed,
        "solver": solver_type,
        "nv": plan.nv,
        "cost": round(float(plan.cost), 2),
        "feasible": plan.feasible,
        "time_sec": round(wall_time, 2),
        "bks_nv": bks_nv,
        "bks_td": bks_td,
    }


def main():
    print("=" * 80)
    print("🔥 LAUNCHING SUPER-BENCHMARK SUITE: 18 TOPOLOGIES x 5 SEEDS x 2 SOLVERS (180 RUNS) 🔥")
    print("=" * 80, flush=True)

    tasks = []
    for scale, inst_name, rel_path in CORE_BENCHMARK_MATRIX:
        for seed in SEEDS:
            tasks.append((scale, inst_name, rel_path, seed, "ALNS-Base"))
            tasks.append((scale, inst_name, rel_path, seed, "Tri-Level Hybrid"))

    num_workers = min(6, max(1, (os.cpu_count() or 4) - 1))
    print(f"Spawning {num_workers} parallel workers on {len(tasks)} runs...", flush=True)

    results = []
    completed = 0
    total = len(tasks)

    t_start_all = time.time()
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        future_to_task = {executor.submit(run_single_eval, *t): t for t in tasks}
        for fut in as_completed(future_to_task):
            res = fut.result()
            results.append(res)
            completed += 1
            if completed % 6 == 0 or completed == total:
                elapsed = time.time() - t_start_all
                print(
                    f"[{completed:03d}/{total:03d}] {res['scale']:14s} | {res['instance']:8s} | {res['solver']:16s} | s={res['seed']} -> NV={res['nv']:2d}, TD={res['cost']:8.2f} ({res['time_sec']:4.1f}s) [Elapsed: {elapsed:.0f}s]",
                    flush=True,
                )

    df = pd.DataFrame(results)
    raw_path = RESULTS_DIR / "super_benchmark_raw.csv"
    df.to_csv(raw_path, index=False)
    print(f"\n✅ Raw benchmark results saved to {raw_path}", flush=True)

    # ── AGGREGATE PER INSTANCE COMPARISON ──
    print("\n" + "=" * 90)
    print("📊 SUPER-BENCHMARK AGGREGATE SUMMARY PER INSTANCE (5 SEEDS AVERAGE)")
    print("=" * 90)

    summary_rows = []
    for scale, inst_name, _ in CORE_BENCHMARK_MATRIX:
        sub = df[df["instance"] == inst_name]
        alns_sub = sub[sub["solver"] == "ALNS-Base"]
        hyb_sub = sub[sub["solver"] == "Tri-Level Hybrid"]

        alns_nv = alns_sub["nv"].mean()
        alns_td = alns_sub["cost"].mean()
        alns_t = alns_sub["time_sec"].mean()

        hyb_nv = hyb_sub["nv"].mean()
        hyb_td = hyb_sub["cost"].mean()
        hyb_t = hyb_sub["time_sec"].mean()

        bks_nv = sub["bks_nv"].iloc[0]
        bks_td = sub["bks_td"].iloc[0]

        delta_nv = hyb_nv - alns_nv
        delta_td_pct = ((hyb_td - alns_td) / alns_td * 100) if alns_nv == hyb_nv else None
        gap_bks_td = ((hyb_td - bks_td) / bks_td * 100) if bks_td else None

        if hyb_nv < alns_nv:
            winner = "Ours (NV)"
        elif hyb_nv > alns_nv:
            winner = "ALNS (NV)"
        else:
            winner = "Ours (TD)" if hyb_td < alns_td else ("ALNS (TD)" if hyb_td > alns_td else "TIE")

        summary_rows.append(
            {
                "Scale": scale,
                "Instance": inst_name,
                "BKS_NV": bks_nv,
                "BKS_TD": bks_td,
                "ALNS_NV": alns_nv,
                "ALNS_TD": alns_td,
                "Ours_NV": hyb_nv,
                "Ours_TD": hyb_td,
                "Delta_NV": delta_nv,
                "Delta_TD%": f"{delta_td_pct:+.2f}%" if delta_td_pct is not None else "---",
                "Gap_BKS%": f"{gap_bks_td:+.2f}%" if gap_bks_td is not None else "---",
                "Winner": winner,
                "ALNS_Time": alns_t,
                "Ours_Time": hyb_t,
            }
        )

    summary_df = pd.DataFrame(summary_rows)
    sum_path = RESULTS_DIR / "super_benchmark_summary.csv"
    summary_df.to_csv(sum_path, index=False)

    print(summary_df.to_string(index=False))

    # Overall win record
    nv_wins = sum(1 for r in summary_rows if r["Ours_NV"] < r["ALNS_NV"])
    nv_ties = sum(1 for r in summary_rows if r["Ours_NV"] == r["ALNS_NV"])
    nv_losses = sum(1 for r in summary_rows if r["Ours_NV"] > r["ALNS_NV"])

    td_wins = sum(1 for r in summary_rows if r["Ours_NV"] == r["ALNS_NV"] and r["Ours_TD"] < r["ALNS_TD"])
    td_ties = sum(1 for r in summary_rows if r["Ours_NV"] == r["ALNS_NV"] and r["Ours_TD"] == r["ALNS_TD"])
    td_losses = sum(1 for r in summary_rows if r["Ours_NV"] == r["ALNS_NV"] and r["Ours_TD"] > r["ALNS_TD"])

    print("\n" + "=" * 50)
    print(f"🏆 OVERALL SCORECARD ACROSS {len(CORE_BENCHMARK_MATRIX)} INSTANCES:")
    print(f"  Fleet Size (NV): {nv_wins} WINS / {nv_ties} TIES / {nv_losses} LOSSES")
    print(f"  Matched TD:     {td_wins} WINS / {td_ties} TIES / {td_losses} LOSSES")
    print("=" * 50)


if __name__ == "__main__":
    main()

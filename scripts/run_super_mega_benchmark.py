#!/usr/bin/env python3
"""Grand Comprehensive Mega-Benchmark: Solomon-100 to Gehring-Homberger 1000 (1k) Customers.

Evaluates ALNS-Base vs Hybrid-DDQN v2.0 Ultra across:
- Solomon 100 (6 instances)
- Homberger 200 (6 instances)
- Homberger 400 (6 instances)
- Homberger 600 (6 instances)
- Homberger 800 (6 instances)
- Homberger 1000 (6 instances)
Total: 36 instance topologies x 5 seeds x 2 solvers = 360 runs.
"""

from __future__ import annotations

import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import pandas as pd

from vrptw.config import Config
from vrptw.core import load_solomon_instance
from vrptw.solvers import ALNSSolver, HybridDDQNSolver

ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "results" / "mega_benchmark"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# 36 benchmark instances spanning all 6 problem scales
BENCHMARK_MATRIX = [
    # Solomon 100
    ("Solomon-100", "C101", "data/Solomon/c101.txt", 15.0),
    ("Solomon-100", "R101", "data/Solomon/r101.txt", 15.0),
    ("Solomon-100", "RC101", "data/Solomon/rc101.txt", 15.0),
    ("Solomon-100", "C201", "data/Solomon/c201.txt", 15.0),
    ("Solomon-100", "R201", "data/Solomon/r201.txt", 15.0),
    ("Solomon-100", "RC201", "data/Solomon/rc201.txt", 15.0),
    # Homberger 200
    ("Homberger-200", "c1_2_1", "data/Gehring_Homberger/homberger_200_customer_instances/C1_2_1.TXT", 20.0),
    ("Homberger-200", "r1_2_1", "data/Gehring_Homberger/homberger_200_customer_instances/R1_2_1.TXT", 20.0),
    ("Homberger-200", "rc1_2_1", "data/Gehring_Homberger/homberger_200_customer_instances/RC1_2_1.TXT", 20.0),
    ("Homberger-200", "c2_2_1", "data/Gehring_Homberger/homberger_200_customer_instances/C2_2_1.TXT", 20.0),
    ("Homberger-200", "r2_2_1", "data/Gehring_Homberger/homberger_200_customer_instances/R2_2_1.TXT", 20.0),
    ("Homberger-200", "rc2_2_1", "data/Gehring_Homberger/homberger_200_customer_instances/RC2_2_1.TXT", 20.0),
    # Homberger 400
    ("Homberger-400", "c1_4_1", "data/Gehring_Homberger/homberger_400_customer_instances/C1_4_1.TXT", 30.0),
    ("Homberger-400", "r1_4_1", "data/Gehring_Homberger/homberger_400_customer_instances/R1_4_1.TXT", 30.0),
    ("Homberger-400", "rc1_4_1", "data/Gehring_Homberger/homberger_400_customer_instances/RC1_4_1.TXT", 30.0),
    ("Homberger-400", "c2_4_1", "data/Gehring_Homberger/homberger_400_customer_instances/C2_4_1.TXT", 30.0),
    ("Homberger-400", "r2_4_1", "data/Gehring_Homberger/homberger_400_customer_instances/R2_4_1.TXT", 30.0),
    ("Homberger-400", "rc2_4_1", "data/Gehring_Homberger/homberger_400_customer_instances/RC2_4_1.TXT", 30.0),
    # Homberger 600
    ("Homberger-600", "c1_6_1", "data/Gehring_Homberger/homberger_600_customer_instances/C1_6_1.TXT", 40.0),
    ("Homberger-600", "r1_6_1", "data/Gehring_Homberger/homberger_600_customer_instances/R1_6_1.TXT", 40.0),
    ("Homberger-600", "rc1_6_1", "data/Gehring_Homberger/homberger_600_customer_instances/RC1_6_1.TXT", 40.0),
    ("Homberger-600", "c2_6_1", "data/Gehring_Homberger/homberger_600_customer_instances/C2_6_1.TXT", 40.0),
    ("Homberger-600", "r2_6_1", "data/Gehring_Homberger/homberger_600_customer_instances/R2_6_1.TXT", 40.0),
    ("Homberger-600", "rc2_6_1", "data/Gehring_Homberger/homberger_600_customer_instances/RC2_6_1.TXT", 40.0),
    # Homberger 800
    ("Homberger-800", "c1_8_1", "data/Gehring_Homberger/homberger_800_customer_instances/C1_8_1.TXT", 50.0),
    ("Homberger-800", "r1_8_1", "data/Gehring_Homberger/homberger_800_customer_instances/R1_8_1.TXT", 50.0),
    ("Homberger-800", "rc1_8_1", "data/Gehring_Homberger/homberger_800_customer_instances/RC1_8_1.TXT", 50.0),
    ("Homberger-800", "c2_8_1", "data/Gehring_Homberger/homberger_800_customer_instances/C2_8_1.TXT", 50.0),
    ("Homberger-800", "r2_8_1", "data/Gehring_Homberger/homberger_800_customer_instances/R2_8_1.TXT", 50.0),
    ("Homberger-800", "rc2_8_1", "data/Gehring_Homberger/homberger_800_customer_instances/RC2_8_1.TXT", 50.0),
    # Homberger 1000 (1k)
    ("Homberger-1000", "c1_10_1", "data/Gehring_Homberger/homberger_1000_customer_instances/C1_10_1.TXT", 60.0),
    ("Homberger-1000", "r1_10_1", "data/Gehring_Homberger/homberger_1000_customer_instances/R1_10_1.TXT", 60.0),
    ("Homberger-1000", "rc1_10_1", "data/Gehring_Homberger/homberger_1000_customer_instances/RC1_10_1.TXT", 60.0),
    ("Homberger-1000", "c2_10_1", "data/Gehring_Homberger/homberger_1000_customer_instances/C2_10_1.TXT", 60.0),
    ("Homberger-1000", "r2_10_1", "data/Gehring_Homberger/homberger_1000_customer_instances/R2_10_1.TXT", 60.0),
    ("Homberger-1000", "rc2_10_1", "data/Gehring_Homberger/homberger_1000_customer_instances/RC2_10_1.TXT", 60.0),
]

SEEDS = [42, 43, 44, 45, 46]


def run_eval(scale: str, inst_name: str, rel_path: str, t_lim: float, seed: int, solver_type: str) -> dict:
    inst_path = str(ROOT / rel_path)
    inst = load_solomon_instance(inst_path)

    cfg = Config(
        time_limit=t_lim,
        gec_max_depth=3 if solver_type == "Hybrid-DDQN" else 0,
        lac_tau_min=0.40 if solver_type == "Hybrid-DDQN" else 0.50,
        lac_tau_max=0.75 if solver_type == "Hybrid-DDQN" else 0.50,
        dynamic_pool_factor=4.0 if solver_type == "Hybrid-DDQN" else 0.0,
    )

    if solver_type == "ALNS-Base":
        solver = ALNSSolver(inst, cfg)
    else:
        solver = HybridDDQNSolver(inst, cfg)

    try:
        t0 = time.perf_counter()
        plan, _ = solver.solve(seed=seed)
        wall_time = time.perf_counter() - t0
        return {
            "scale": scale,
            "instance": inst_name,
            "seed": seed,
            "solver": solver_type,
            "nv": plan.nv,
            "cost": plan.cost,
            "feasible": plan.feasible,
            "time": wall_time,
            "status": "OK",
        }
    except Exception as e:
        print(f"ERROR on {scale} {inst_name} {solver_type} seed={seed}: {e}", file=sys.stderr, flush=True)
        return {
            "scale": scale,
            "instance": inst_name,
            "seed": seed,
            "solver": solver_type,
            "nv": -1,
            "cost": -1.0,
            "feasible": False,
            "time": 0.0,
            "status": f"ERROR: {e}",
        }


def main():
    print("=== GRAND COMPREHENSIVE MEGA-BENCHMARK: 100 TO 1000 (1K) CUSTOMERS ===", flush=True)
    tasks = []
    for scale, inst_name, rel_path, t_lim in BENCHMARK_MATRIX:
        for seed in SEEDS:
            tasks.append((scale, inst_name, rel_path, t_lim, seed, "ALNS-Base"))
            tasks.append((scale, inst_name, rel_path, t_lim, seed, "Hybrid-DDQN"))

    results = []
    completed = 0
    total = len(tasks)
    num_workers = min(7, max(1, os.cpu_count() or 4))
    print(f"Spawning {num_workers} parallel worker processes for {total} total runs...", flush=True)

    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = {executor.submit(run_eval, *t): t for t in tasks}
        for fut in as_completed(futures):
            res = fut.result()
            results.append(res)
            completed += 1
            if completed % 5 == 0 or completed == total:
                print(
                    f"[{completed:03d}/{total:03d}] {res['scale']:14s} | {res['instance']:8s} | {res['solver']:11s} | s={res['seed']:2d} -> NV={res['nv']:2d}, TD={res['cost']:8.2f} ({res['time']:4.1f}s)",
                    flush=True,
                )

    df = pd.DataFrame(results)
    raw_csv = RESULTS_DIR / "mega_benchmark_raw.csv"
    df.to_csv(raw_csv, index=False)
    print(f"\nWrote raw results to {raw_csv}", flush=True)

    print("\n=== SCALE-BY-SCALE AGGREGATE SUMMARY TABLE ===", flush=True)
    summary = (
        df.groupby(["scale", "solver"])
        .agg(
            {
                "nv": "mean",
                "cost": "mean",
                "time": "mean",
            }
        )
        .round(2)
    )
    print(summary, flush=True)
    summary.to_csv(RESULTS_DIR / "mega_benchmark_summary.csv")


if __name__ == "__main__":
    main()

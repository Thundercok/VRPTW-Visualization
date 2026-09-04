#!/usr/bin/env python3
"""Parallel A/B Benchmark Comparison: Baseline vs v2.0 Ultra (with GEC & Adaptive LAC)."""

from __future__ import annotations

import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import pandas as pd

from vrptw.config import Config
from vrptw.core import load_solomon_instance
from vrptw.solvers import HybridDDQNSolver

ROOT = Path(__file__).resolve().parents[1]

INSTANCES = [
    ("RC101", "data/Solomon/rc101.txt", 15.0),
    ("R101", "data/Solomon/r101.txt", 15.0),
    ("C101", "data/Solomon/c101.txt", 15.0),
    ("r1_2_1", "data/Gehring_Homberger/homberger_200_customer_instances/R1_2_1.TXT", 20.0),
]

SEEDS = [42, 43, 44, 45, 46]


def run_single_experiment(inst_name: str, rel_path: str, t_lim: float, seed: int, mode: str) -> dict:
    inst_path = str(ROOT / rel_path)
    inst = load_solomon_instance(inst_path)

    if mode == "Baseline":
        cfg = Config(
            time_limit=t_lim,
            gec_max_depth=0,
            lac_tau_min=0.50,
            lac_tau_max=0.50,
            dynamic_pool_factor=0.0,
        )
    else:  # v2.0 Ultra
        cfg = Config(
            time_limit=t_lim,
            gec_max_depth=3,
            lac_tau_min=0.40,
            lac_tau_max=0.75,
            dynamic_pool_factor=4.0,
        )

    solver = HybridDDQNSolver(inst, cfg)
    t0 = time.perf_counter()
    plan, _ = solver.solve(seed=seed)
    wall_time = time.perf_counter() - t0

    return {
        "instance": inst_name,
        "seed": seed,
        "mode": mode,
        "nv": plan.nv,
        "cost": plan.cost,
        "feasible": plan.feasible,
        "time": wall_time,
    }


def main():
    print("=== RUNNING PARALLEL A/B BENCHMARK (6 WORKERS) ===", flush=True)
    tasks = []
    for inst_name, rel_path, t_lim in INSTANCES:
        for seed in SEEDS:
            tasks.append((inst_name, rel_path, t_lim, seed, "Baseline"))
            tasks.append((inst_name, rel_path, t_lim, seed, "v2.0 Ultra"))

    results = []
    completed = 0
    total = len(tasks)

    with ProcessPoolExecutor(max_workers=6) as executor:
        futures = {executor.submit(run_single_experiment, *t): t for t in tasks}
        for fut in as_completed(futures):
            res = fut.result()
            results.append(res)
            completed += 1
            print(
                f"[{completed:02d}/{total:02d}] {res['instance']:8s} | {res['mode']:10s} | seed={res['seed']:2d} -> NV={res['nv']:2d}, TD={res['cost']:8.2f} ({res['time']:4.1f}s)",
                flush=True,
            )

    df = pd.DataFrame(results)
    df.to_csv(ROOT / "results/ab_test_v2_ultra.csv", index=False)

    print("\n=== FINAL EMPIRICAL COMPARISON TABLE ===", flush=True)
    summary = (
        df.groupby(["instance", "mode"])
        .agg(
            {
                "nv": ["mean", "min", "max"],
                "cost": ["mean", "min"],
                "time": "mean",
            }
        )
        .round(2)
    )
    print(summary, flush=True)


if __name__ == "__main__":
    main()

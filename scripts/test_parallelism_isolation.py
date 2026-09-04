#!/usr/bin/env python3
"""Isolates the parallelism variable for ALNS-Base across workers=1 vs workers=7."""

import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from vrptw.benchmark_suite import BenchmarkTask, execute_benchmark_task
from vrptw.config import Config
from vrptw.solvers import ALNSSolver


def main():
    solomon_dir = ROOT / "data" / "Solomon"
    instances = [
        ("C101", str(solomon_dir / "c101.txt")),
        ("R101", str(solomon_dir / "r101.txt")),
        ("RC101", str(solomon_dir / "rc101.txt")),
    ]
    seeds = [42, 43]
    t_max = 2000

    cfg = Config(
        alns_iterations=t_max,
        hybrid_iterations=t_max,
        early_stop_patience=max(100, t_max // 4),
        polish_iterations=50,
        polish_patience=25,
    )

    tasks_w1 = []
    tasks_w7 = []
    for name, path in instances:
        for seed in seeds:
            tasks_w1.append(BenchmarkTask(name, path, "ALNS-Base", seed, cfg, ALNSSolver, tag="w1"))
            tasks_w7.append(BenchmarkTask(name, path, "ALNS-Base", seed, cfg, ALNSSolver, tag="w7"))

    print("=" * 80)
    print("RUNNING WORKERS=1 (Sequential)")
    print("=" * 80)
    res_w1 = {}
    for t in tasks_w1:
        res = execute_benchmark_task(t)
        res_w1[(res.instance, res.seed)] = res
        print(f"  [W1] {res.instance:<6} Seed {res.seed} -> NV: {res.nv:2d}, TD: {res.td:8.2f} ({res.time_sec:4.1f}s)")

    print("\n" + "=" * 80)
    print("RUNNING WORKERS=7 (Multiprocessing ProcessPoolExecutor)")
    print("=" * 80)
    res_w7 = {}
    with ProcessPoolExecutor(max_workers=7) as executor:
        future_to_task = {executor.submit(execute_benchmark_task, t): t for t in tasks_w7}
        for future in as_completed(future_to_task):
            res = future.result()
            res_w7[(res.instance, res.seed)] = res
            print(
                f"  [W7] {res.instance:<6} Seed {res.seed} -> NV: {res.nv:2d}, TD: {res.td:8.2f} ({res.time_sec:4.1f}s)"
            )

    print("\n" + "=" * 80)
    print("SIDE-BY-SIDE COMPARISON (Workers=1 vs Workers=7)")
    print("=" * 80)
    print(f"{'Instance':<8} {'Seed':<5} | {'W1 NV':<6} {'W1 TD':<10} | {'W7 NV':<6} {'W7 TD':<10} | {'Match?':<8}")
    print("-" * 65)

    all_matched = True
    for name, _ in instances:
        for seed in seeds:
            r1 = res_w1[(name, seed)]
            r7 = res_w7[(name, seed)]
            match = (r1.nv == r7.nv) and (abs(r1.td - r7.td) < 1e-4)
            if not match:
                all_matched = False
            status = "EXACT MATCH" if match else "MISMATCH"
            print(f"{name:<8} {seed:<5} | {r1.nv:<6} {r1.td:<10.2f} | {r7.nv:<6} {r7.td:<10.2f} | {status}")

    print("=" * 80)
    if all_matched:
        print("ALL 6 RUNS MATCH EXACTLY BIT-FOR-BIT BETWEEN WORKERS=1 AND WORKERS=7.")
    else:
        print("MISMATCH DETECTED BETWEEN WORKERS=1 AND WORKERS=7!")
    print("=" * 80)


if __name__ == "__main__":
    main()

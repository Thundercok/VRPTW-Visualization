#!/usr/bin/env python3
"""Runner for Config (6) Rule-Macro and Config (7) Rule-Micro Ablations.

Executes 6 representative instances across 5 seeds under strict cold-starts.
"""

from __future__ import annotations

import copy
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from vrptw.benchmark_suite import (
    ABLATION_INSTANCES,
    BenchmarkResult,
    BenchmarkTask,
    execute_benchmark_task,
)
from vrptw.config import Config
from vrptw.solvers import RuleMacroHybridSolver, RuleMicroHybridSolver


def main():
    solomon_dir = ROOT / "data" / "Solomon"
    gh200_dir = ROOT / "data" / "Gehring_Homberger" / "homberger_200_customer_instances"
    gh400_dir = ROOT / "data" / "Gehring_Homberger" / "homberger_400_customer_instances"

    inst_map = {
        "C101": str(solomon_dir / "c101.txt"),
        "R101": str(solomon_dir / "r101.txt"),
        "RC101": str(solomon_dir / "rc101.txt"),
        "c2_2_1": str(gh200_dir / "C2_2_1.TXT"),
        "r1_2_1": str(gh200_dir / "R1_2_1.TXT"),
        "rc2_4_1": str(gh400_dir / "RC2_4_1.TXT"),
    }

    seeds = [42, 43, 44, 45, 46]
    n_iters = 2000

    base_cfg = Config(
        alns_iterations=n_iters,
        hybrid_iterations=n_iters,
        early_stop_patience=max(100, n_iters // 4),
        polish_iterations=50,
        polish_patience=25,
        op_softmax_tau=1.0,
        lac_enabled=True,
        recombine_after_main_search=True,
        plateau_start=72,
    )

    configs_to_run = {
        "Rule-Macro": {
            "solver_cls": RuleMacroHybridSolver,
            "cfg": copy.deepcopy(base_cfg),
        },
        "Rule-Micro": {
            "solver_cls": RuleMicroHybridSolver,
            "cfg": copy.deepcopy(base_cfg),
        },
    }

    tasks: list[BenchmarkTask] = []
    for inst_name in ABLATION_INSTANCES:
        path = inst_map[inst_name]
        for cfg_name, cfg_info in configs_to_run.items():
            for seed in seeds:
                tasks.append(
                    BenchmarkTask(
                        instance_name=inst_name,
                        instance_path=path,
                        solver_name=cfg_name,
                        seed=seed,
                        cfg=copy.deepcopy(cfg_info["cfg"]),
                        solver_cls=cfg_info["solver_cls"],
                        tag="ablation_rule",
                    )
                )

    print("=" * 80)
    print(f"  RUNNING RULE ABLATIONS: {len(tasks)} tasks (2 configs x 6 instances x 5 seeds)")
    print("=" * 80)

    t0 = time.time()
    results: list[BenchmarkResult] = []
    workers = min(8, mp.cpu_count()) if hasattr(os, "cpu_count") else 4

    with ProcessPoolExecutor(max_workers=workers) as executor:
        future_to_task = {executor.submit(execute_benchmark_task, t): t for t in tasks}
        completed = 0
        for future in as_completed(future_to_task):
            res = future.result()
            results.append(res)
            completed += 1
            print(
                f"  [{completed:2d}/{len(tasks):2d}] {res.instance:<10} {res.solver:<15} "
                f"Seed {res.seed:2d} -> NV: {res.nv:2d}, TD: {res.td:7.2f} ({res.time_sec:5.1f}s)"
            )

    elapsed = time.time() - t0
    print(f"\nAll {len(tasks)} tasks finished in {elapsed:.1f}s ({elapsed/60:.2f} mins).")

    out_dir = ROOT / "results" / "paper_benchmark_suite"
    df_rules = pd.DataFrame([r.to_dict() for r in results])
    df_rules.to_csv(out_dir / "benchmark_ablation_rules_raw.csv", index=False)
    print(f"Rules raw CSV saved -> {out_dir / 'benchmark_ablation_rules_raw.csv'}")

    # Merge with original 5-config ablation CSV if present
    orig_csv = out_dir / "benchmark_ablation_raw.csv"
    if orig_csv.exists():
        df_orig = pd.read_csv(orig_csv)
        df_merged = pd.concat([df_orig, df_rules], ignore_index=True)
        merged_csv = out_dir / "benchmark_ablation_7configs_raw.csv"
        df_merged.to_csv(merged_csv, index=False)
        print(f"Merged 7-config raw CSV saved -> {merged_csv}")


if __name__ == "__main__":
    import multiprocessing as mp
    main()

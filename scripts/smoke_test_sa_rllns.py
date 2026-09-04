#!/usr/bin/env python3
"""Smoke test for Single-Agent RL-LNS baseline on 3 instances x 2 seeds (6 runs)."""

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from vrptw.benchmark_suite import (
    BenchmarkTask,
    execute_benchmark_task,
)
from vrptw.config import Config
from vrptw.solvers import HybridDDQNSolver


def main():
    solomon_dir = ROOT / "data" / "Solomon"
    test_instances = [
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
        op_softmax_tau=1.0,
        macro_enabled=False,
        lac_enabled=False,
        pool_recombine_enabled=False,
        recombine_after_main_search=False,
        gnn_model_path=None,
    )

    print("=" * 80)
    print(f"  SMOKE TEST: Single-Agent RL-LNS (3 instances x 2 seeds = 6 runs, T_max={t_max})")
    print("  Macro Controller: OFF | LAC: OFF | RoutePool Recombine: OFF | GNN: OFF")
    print("  Micro DDQN Operator Selection: ON (Dueling DDQN + Entropy Gate)")
    print("=" * 80)

    for inst_name, inst_path in test_instances:
        for seed in seeds:
            task = BenchmarkTask(
                instance_name=inst_name,
                instance_path=inst_path,
                solver_name="Single-Agent RL-LNS",
                seed=seed,
                cfg=cfg,
                solver_cls=HybridDDQNSolver,
                tag="smoke_sa_rllns",
            )
            t0 = time.time()
            res = execute_benchmark_task(task)
            elapsed = time.time() - t0
            print(
                f"RESULT | Instance: {res.instance:<6} | Seed: {res.seed} | "
                f"NV: {res.nv:2d} | TD: {res.td:9.2f} | "
                f"Gap_TD%: {res.gap_td_pct:+6.2f}% | Feasible: {res.feasible!s:<5} | "
                f"Time: {res.time_sec:5.1f}s | Wall: {elapsed:5.1f}s"
            )


if __name__ == "__main__":
    main()

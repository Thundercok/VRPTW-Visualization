#!/usr/bin/env python3
"""Unified Paper Benchmark Runner for IEEE Access VRPTW Publication.

Features:
- Tri-Paradigm Execution: Best Heuristics (ALNS-Base, OR-Tools) vs Hybrid-DDQN.
- Full 5-Configuration Ablation Matrix on 6 representative instances.
- Strict Independent Cold-Starts: Isolated temporary scratch spaces for every single run.
- Multiprocessing Worker Pool with budget consistency (matching iterations).
- Automated Statistical Analysis (Wilcoxon signed-rank + Binomial Sign test + Bonferroni).
- Automatic LaTeX Table Generation (Table III, IV, V) for docs/sections/05_experiments.tex.
"""

from __future__ import annotations

import argparse
import multiprocessing as mp
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from vrptw.benchmark_suite import (
    ABLATION_CONFIGS,
    ABLATION_INSTANCES,
    BenchmarkResult,
    BenchmarkTask,
    execute_benchmark_task,
)
from vrptw.config import Config
from vrptw.solvers import ALNSSolver, HybridDDQNSolver


def collect_solomon_instances(data_dir: Path) -> list[tuple[str, str]]:
    """Discovers all 56 Solomon benchmark instances."""
    instances = []
    for f in sorted(data_dir.glob("*.txt")):
        name = f.stem.upper()
        if any(name.startswith(p) for p in ["C1", "C2", "R1", "R2", "RC1", "RC2"]):
            instances.append((name, str(f)))
    return instances


def collect_homberger_instances(
    data_dir: Path, n_customers: int = 200, limit: int | None = None
) -> list[tuple[str, str]]:
    """Discovers Homberger benchmark instances."""
    instances = []
    for f in sorted(data_dir.glob("*.TXT")) + sorted(data_dir.glob("*.txt")):
        name = f.stem.lower()
        if f"_{n_customers // 100}_" in name or f"_{n_customers}_" in name:
            instances.append((name, str(f)))
    if limit and len(instances) > limit:
        # Select evenly spaced representative instances
        step = max(1, len(instances) // limit)
        instances = instances[::step][:limit]
    return instances


def resolve_solver_task(solver: str, base_cfg: Config) -> tuple[Any, Config]:
    if solver == "ALNS-Base":
        return ALNSSolver, base_cfg
    elif solver in ("Single-Agent RL-LNS", "Single-Agent-RL-LNS", "SA-RL-LNS"):
        sa_cfg = copy_cfg_with_overrides(
            base_cfg,
            {
                "macro_enabled": False,
                "lac_enabled": False,
                "pool_recombine_enabled": False,
                "recombine_after_main_search": False,
                "op_softmax_tau": 1.0,
                "gnn_model_path": None,
            },
        )
        return HybridDDQNSolver, sa_cfg
    elif solver in ABLATION_CONFIGS:
        conf = ABLATION_CONFIGS[solver]
        return conf["solver_cls"], copy_cfg_with_overrides(base_cfg, conf["config_overrides"])
    else:
        return HybridDDQNSolver, base_cfg


def build_tasks_for_mode(
    mode: str,
    solvers: list[str],
    seeds: list[int],
    cfg: Config,
    solomon_dir: Path,
    gh200_dir: Path,
    gh400_dir: Path,
) -> list[BenchmarkTask]:
    """Constructs the list of tasks for the selected benchmark mode."""
    tasks: list[BenchmarkTask] = []

    # ── 1. Quick Mode (5 Representative Instances, 3 Seeds) ────────────────
    if mode == "quick":
        quick_insts = [
            ("C101", str(solomon_dir / "c101.txt")),
            ("R101", str(solomon_dir / "r101.txt")),
            ("RC101", str(solomon_dir / "rc101.txt")),
            ("R201", str(solomon_dir / "r201.txt")),
            ("r1_2_1", str(gh200_dir / "R1_2_1.TXT")),
        ]
        for name, path in quick_insts:
            if not os.path.exists(path):
                continue
            for solver in solvers:
                s_cls, s_cfg = resolve_solver_task(solver, cfg)
                for seed in seeds[:3]:
                    tasks.append(BenchmarkTask(name, path, solver, seed, s_cfg, s_cls, tag="quick"))

    # ── 2. Ablation Mode (5 Configurations on 6 Instances) ────────────────
    if mode in ("ablation", "all"):
        inst_map = {
            "C101": str(solomon_dir / "c101.txt"),
            "R101": str(solomon_dir / "r101.txt"),
            "RC101": str(solomon_dir / "rc101.txt"),
            "c2_2_1": str(gh200_dir / "C2_2_1.TXT"),
            "r1_2_1": str(gh200_dir / "R1_2_1.TXT"),
            "rc2_4_1": str(gh400_dir / "RC2_4_1.TXT"),
        }
        for name in ABLATION_INSTANCES:
            path = inst_map.get(name)
            if not path or not os.path.exists(path):
                continue
            for config_name, conf_data in ABLATION_CONFIGS.items():
                ablation_cfg = copy_cfg_with_overrides(cfg, conf_data["config_overrides"])
                for seed in seeds:
                    tasks.append(
                        BenchmarkTask(
                            name,
                            path,
                            config_name,
                            seed,
                            ablation_cfg,
                            conf_data["solver_cls"],
                            tag="ablation",
                        )
                    )

    # ── 3. Solomon Mode (56 Instances) ───────────────────────────────────
    if mode in ("solomon", "all"):
        solomon_insts = collect_solomon_instances(solomon_dir)
        for name, path in solomon_insts:
            for solver in solvers:
                s_cls, s_cfg = resolve_solver_task(solver, cfg)
                for seed in seeds:
                    tasks.append(BenchmarkTask(name, path, solver, seed, s_cfg, s_cls, tag="solomon"))

    # ── 4. Homberger-200 Mode ─────────────────────────────────────────────
    if mode in ("homberger200", "all"):
        gh200_insts = collect_homberger_instances(gh200_dir, 200, limit=12 if mode == "all" else None)
        for name, path in gh200_insts:
            for solver in solvers:
                s_cls, s_cfg = resolve_solver_task(solver, cfg)
                for seed in seeds:
                    tasks.append(BenchmarkTask(name, path, solver, seed, s_cfg, s_cls, tag="gh200"))

    # ── 5. Homberger-400 Mode ─────────────────────────────────────────────
    if mode in ("homberger400", "all"):
        rep_400 = ["c1_4_1", "c2_4_1", "r1_4_1", "r2_4_1", "rc1_4_1", "rc2_4_1"]
        for r_name in rep_400:
            target_path = gh400_dir / f"{r_name.upper()}.TXT"
            if not target_path.exists():
                target_path = gh400_dir / f"{r_name.lower()}.txt"
            if target_path.exists():
                for solver in solvers:
                    s_cls, s_cfg = resolve_solver_task(solver, cfg)
                    for seed in seeds:
                        tasks.append(BenchmarkTask(r_name, str(target_path), solver, seed, s_cfg, s_cls, tag="gh400"))

    return tasks


def copy_cfg_with_overrides(base_cfg: Config, overrides: dict[str, Any]) -> Config:
    import copy

    new_cfg = copy.deepcopy(base_cfg)
    for k, v in overrides.items():
        setattr(new_cfg, k, v)
    return new_cfg


def main() -> int:
    parser = argparse.ArgumentParser(description="Tri-Paradigm Paper Benchmark Runner for IEEE Access.")
    parser.add_argument(
        "--mode",
        type=str,
        choices=["quick", "ablation", "solomon", "homberger200", "homberger400", "all"],
        default="quick",
        help="Benchmark execution mode.",
    )
    parser.add_argument(
        "--solvers",
        nargs="+",
        default=["ALNS-Base", "Hybrid-DDQN"],
        help="Solvers to evaluate.",
    )
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=[42, 43, 44, 45, 46],
        help="Random seeds.",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=600,
        help="Max ALNS / Hybrid iterations per instance.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=max(1, mp.cpu_count() // 2),
        help="Multiprocessing worker pool size.",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default=str(ROOT / "results" / "paper_benchmark_suite"),
        help="Directory to store benchmark CSVs and LaTeX outputs.",
    )
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    solomon_dir = ROOT / "data" / "Solomon"
    gh200_dir = ROOT / "data" / "Gehring_Homberger" / "homberger_200_customer_instances"
    gh400_dir = ROOT / "data" / "Gehring_Homberger" / "homberger_400_customer_instances"

    base_cfg = Config(
        alns_iterations=args.iterations,
        hybrid_iterations=args.iterations,
        early_stop_patience=max(100, args.iterations // 4),
        polish_iterations=50,
        polish_patience=25,
        op_softmax_tau=1.0,
    )

    tasks = build_tasks_for_mode(
        args.mode,
        args.solvers,
        args.seeds,
        base_cfg,
        solomon_dir,
        gh200_dir,
        gh400_dir,
    )

    print("=" * 80)
    print(f"  IEEE ACCESS VRPTW BENCHMARK RUNNER [Mode: {args.mode.upper()}]")
    print(f"  Total Tasks: {len(tasks)} | Workers: {args.workers} | Iterations: {args.iterations}")
    print(f"  Output Directory: {out_dir}")
    print("  Independent Cold-Starts: ENFORCED (Isolated Scopes)")
    print("=" * 80)

    if not tasks:
        print("No tasks generated. Please verify data directory paths.")
        return 1

    t0 = time.time()
    results: list[BenchmarkResult] = []
    completed_count = 0

    csv_path = out_dir / f"benchmark_{args.mode}_raw.csv"
    if args.workers == 1:
        for t in tasks:
            res = execute_benchmark_task(t)
            results.append(res)
            completed_count += 1
            print(
                f"  [{completed_count:3d}/{len(tasks):3d}] {res.instance:<10} {res.solver:<15} Seed {res.seed:2d} -> NV: {res.nv:2d}, TD: {res.td:7.2f} ({res.time_sec:5.1f}s)"
            )
            if completed_count % 10 == 0:
                pd.DataFrame([r.to_dict() for r in results]).to_csv(csv_path, index=False)
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            future_to_task = {executor.submit(execute_benchmark_task, t): t for t in tasks}
            for future in as_completed(future_to_task):
                res = future.result()
                results.append(res)
                completed_count += 1
                print(
                    f"  [{completed_count:3d}/{len(tasks):3d}] {res.instance:<10} {res.solver:<15} Seed {res.seed:2d} -> NV: {res.nv:2d}, TD: {res.td:7.2f} ({res.time_sec:5.1f}s)"
                )
                if completed_count % 10 == 0 or completed_count == len(tasks):
                    pd.DataFrame([r.to_dict() for r in results]).to_csv(csv_path, index=False)

    total_time = time.time() - t0
    print(f"\nAll {len(tasks)} benchmark tasks finished in {total_time:.1f}s.")

    # ── Save Results ──────────────────────────────────────────────────────────
    df_raw = pd.DataFrame([r.to_dict() for r in results])
    csv_path = out_dir / f"benchmark_{args.mode}_raw.csv"
    df_raw.to_csv(csv_path, index=False)
    print(f"Raw results saved -> {csv_path}")

    # ── Run Statistical Tests ────────────────────────────────────────────────
    stat_script = ROOT / "scripts" / "compute_paper_statistics.py"
    stat_md = out_dir / f"statistical_report_{args.mode}.md"
    os.system(f'"{sys.executable}" "{stat_script}" --csv "{csv_path}" --out-md "{stat_md}"')

    # ── Update LaTeX Tables ──────────────────────────────────────────────────
    latex_script = ROOT / "scripts" / "generate_latex_tables.py"
    os.system(f'"{sys.executable}" "{latex_script}" --csv "{csv_path}"')

    return 0


if __name__ == "__main__":
    sys.exit(main())

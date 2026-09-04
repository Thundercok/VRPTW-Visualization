#!/usr/bin/env python3
"""Diagnose exact mode distribution and QNet head for DDQN vs Rule-Macro."""

from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from vrptw.config import MODES, Config
from vrptw.core import load_solomon_instance
from vrptw.solvers import HybridDDQNSolver, RuleMacroHybridSolver

ROOT = Path(__file__).resolve().parents[1]
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


def run_one(task):
    name, path, solver_type = task
    inst = load_solomon_instance(path)
    cfg = Config(
        hybrid_iterations=2000,
        alns_iterations=2000,
        polish_iterations=50,
        polish_patience=25,
        plateau_start=72,
        lac_enabled=True,
        recombine_after_main_search=True,
    )
    if solver_type == "ddqn":
        s = HybridDDQNSolver(inst, cfg, seed=42)
    else:
        s = RuleMacroHybridSolver(inst, cfg, seed=42)
    p, _ = s.solve(seed=42)
    mode_names = {idx: m.name for idx, m in enumerate(MODES)}
    trace_with_names = {f"{k}_{mode_names.get(k, k)}": v for k, v in dict(s.mode_trace).items()}
    return name, solver_type, p.nv, p.cost, trace_with_names, len(s.modes)


def main():
    tasks = []
    for name, path in inst_map.items():
        tasks.append((name, path, "ddqn"))
        tasks.append((name, path, "rule"))

    with ProcessPoolExecutor(max_workers=8) as executor:
        results = list(executor.map(run_one, tasks))

    print("=== MODE TRACE RESULTS (Seed 42) ===")
    for name in inst_map:
        sub = [r for r in results if r[0] == name]
        print(f"\n--- Instance: {name} ---")
        for _, stype, nv, td, trace, n_modes in sub:
            print(f"  [{stype.upper():<4}] (n_modes={n_modes}) NV={nv:2d}, TD={td:7.2f} | Trace: {trace}")


if __name__ == "__main__":
    main()

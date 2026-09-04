#!/usr/bin/env python3
"""
LOCO Ablation on Homberger-200 (c1_2_1, r1_2_1, rc1_2_1).
7 Configurations x 3 Instances x 5 Seeds = 105 runs.
"""

from __future__ import annotations

import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import pandas as pd

from vrptw.config import Config
from vrptw.core import load_solomon_instance
from vrptw.solvers import HybridDDQNSolver

CONFIGS = {
    "Full": {},
    "Full_no_Macro": {"macro_mode": "none"},
    "Full_no_Micro": {"micro_mode": "bandit"},
    "Full_no_LAC": {"use_lac": False},
    "Full_no_Pool": {"pool_recombine": False},
    "Full_no_GNN": {"use_gnn": False},
    "Full_no_Gate": {"entropy_gating": False},
}

INSTANCES = {
    "c1_2_1": "data/Gehring_Homberger/homberger_200_customer_instances/C1_2_1.TXT",
    "r1_2_1": "data/Gehring_Homberger/homberger_200_customer_instances/R1_2_1.TXT",
    "rc1_2_1": "data/Gehring_Homberger/homberger_200_customer_instances/RC1_2_1.TXT",
}

SEEDS = [42, 43, 44, 45, 46]


def run_single(config_name: str, inst_name: str, path: str, seed: int) -> dict:
    inst = load_solomon_instance(path)
    cfg = Config()
    cfg.seed = seed

    flags = CONFIGS[config_name]
    if flags.get("macro_mode") == "none":
        cfg.use_rl = False
        cfg.use_macro_rl = False
    if flags.get("micro_mode") == "bandit":
        cfg.use_op_rl = False
        cfg.use_op_bandit = True
    if flags.get("use_lac") is False:
        cfg.use_lac = False
    if flags.get("pool_recombine") is False:
        cfg.rl_recombine_enabled = False
        cfg.recombine_with_pool = False
    if flags.get("use_gnn") is False:
        cfg.gnn_guidance = False
    if flags.get("entropy_gating") is False:
        cfg.entropy_gating = False

    t0 = time.time()
    solver = HybridDDQNSolver(inst, cfg, seed=seed)

    if flags.get("macro_mode") == "none":
        solver.use_rl = False
    if flags.get("micro_mode") == "bandit":
        solver.use_op_rl = False
    if flags.get("use_lac") is False:
        solver.use_lac = False
    if flags.get("use_gnn") is False:
        solver.gamma = 0.0
        solver.heatmap = None

    plan, _ = solver.solve(seed=seed)
    dur = time.time() - t0

    return {
        "config": config_name,
        "instance": inst_name,
        "seed": seed,
        "nv": int(plan.nv),
        "cost": float(plan.cost),
        "feasible": bool(plan.feasible),
        "time": dur,
    }


def main():
    jobs = []
    for cfg_name in CONFIGS:
        for inst_name, p in INSTANCES.items():
            for s in SEEDS:
                jobs.append((cfg_name, inst_name, p, s))

    out_dir = Path("results/loco_200")
    out_dir.mkdir(parents=True, exist_ok=True)
    raw_path = out_dir / "loco_200_raw.csv"

    print("Launching 105 LOCO runs on Homberger-200 across 6 workers...")
    results = []
    with ProcessPoolExecutor(max_workers=6) as executor:
        futures = {executor.submit(run_single, c, i, p, s): (c, i, s) for c, i, p, s in jobs}
        for fut in as_completed(futures):
            res = fut.result()
            results.append(res)
            print(
                f"[{len(results)}/105] {res['config']:15s} {res['instance']:8s} seed={res['seed']}: NV={res['nv']} TD={res['cost']:.2f}"
            )

    df = pd.DataFrame(results)
    df.to_csv(raw_path, index=False)

    summary = df.groupby(["config", "instance"])[["nv", "cost", "time"]].mean().reset_index()
    summary.to_csv(out_dir / "loco_200_summary.csv", index=False)
    print("\n=== SUMMARY ===")
    print(summary.to_string())


if __name__ == "__main__":
    main()

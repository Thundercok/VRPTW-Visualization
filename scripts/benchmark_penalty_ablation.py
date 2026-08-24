"""
Targeted 3-Way Isolated Benchmark for Penalty Mechanism Ablation.
Compares:
  - Arm 2: Adaptive Feasibility (AdaptiveFeasibilityManager)
  - Arm 3a: Lagrangian-Only (LagrangianPenaltyController without HiGHS plateau synthesis)
  - Arm 3b: Lagrangian + HiGHS (LagrangianPenaltyController WITH HiGHS plateau synthesis)
on representative benchmark instances under strict independent cold-starts.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from vrptw.config import BKS, Config
from vrptw.core import Inst
from vrptw.solvers import HybridDDQNSolver


def load_instance(path: Path) -> Inst:
    with open(path, encoding="utf-8") as f:
        lines = f.readlines()
    name = lines[0].strip()
    capacity = float(lines[4].strip().split()[1])
    data = []
    for line in lines[9:]:
        if line.strip():
            data.append(list(map(float, line.split())))
    return Inst({
        "name": name,
        "capacity": capacity,
        "data": np.array(data, dtype=np.float64)
    })


def run_ablation(
    instances: list[str],
    seeds: list[int],
    iters: int = 1000,
    early_stop: int = 250
):
    print("=" * 125)
    print("🚀 RUNNING 3-WAY ISOLATED ABLATION BENCHMARK (Strict Independent Cold-Starts)")
    print(f"   Instances : {', '.join(instances)}")
    print(f"   Seeds     : {seeds}")
    print(f"   Iters     : {iters} (Early stop: {early_stop})")
    print("=" * 125)

    results = {
        "Adaptive": {},
        "Lagrangian-Only": {},
        "Lagrangian+HiGHS": {}
    }

    for inst_name in instances:
        file_path = ROOT / "data" / "Solomon" / f"{inst_name.lower()}.txt"
        if not file_path.exists():
            print(f"File not found: {file_path}")
            continue

        inst = load_instance(file_path)
        bks_ref = BKS.get(inst_name, {"nv": 0, "td": 0.0})
        bks_nv = bks_ref["nv"]
        bks_td = bks_ref["td"]

        results["Adaptive"][inst_name] = []
        results["Lagrangian-Only"][inst_name] = []
        results["Lagrangian+HiGHS"][inst_name] = []

        print(f"\n▶ Instance: {inst_name} (SINTEF BKS: NV={bks_nv}, TD={bks_td:.2f})")
        print(f"  {'Seed':<6} | {'Mode':<18} | {'NV':<6} | {'TD':<10} | {'NV Gap':<8} | {'TD Gap %':<10} | {'Time (s)':<8}")
        print("  " + "-" * 85)

        for seed in seeds:
            # 1. Arm 2: Adaptive Feasibility
            cfg_adapt = Config(
                hybrid_iterations=iters,
                alns_iterations=iters,
                early_stop_patience=early_stop,
                adaptive_feasibility=True,
                lagrangian_penalty=False,
                penalty_search_enabled=False,
                highs_plateau_synthesis=False,
            )
            t0 = time.time()
            solver_adapt = HybridDDQNSolver(inst, cfg_adapt)
            plan_adapt, _ = solver_adapt.solve(seed=seed)
            t_adapt = time.time() - t0
            nv_gap_adapt = plan_adapt.nv - bks_nv
            td_gap_adapt = (plan_adapt.cost - bks_td) / bks_td * 100.0 if bks_td > 0 else 0.0

            results["Adaptive"][inst_name].append({
                "seed": seed, "nv": plan_adapt.nv, "td": plan_adapt.cost, "time": t_adapt
            })
            print(f"  {seed:<6} | {'Adaptive':<18} | {plan_adapt.nv:<6} | {plan_adapt.cost:<10.2f} | {nv_gap_adapt:<+8} | {td_gap_adapt:<+9.2f}% | {t_adapt:<7.1f}s")

            # 2. Arm 3a: Lagrangian-Only (No HiGHS plateau synthesis)
            cfg_lag_only = Config(
                hybrid_iterations=iters,
                alns_iterations=iters,
                early_stop_patience=early_stop,
                adaptive_feasibility=False,
                lagrangian_penalty=True,
                lagrangian_theta=2.0,
                lagrangian_stall_limit=20,
                penalty_search_enabled=False,
                highs_plateau_synthesis=False,
            )
            t0 = time.time()
            solver_lag_only = HybridDDQNSolver(inst, cfg_lag_only)
            plan_lag_only, _ = solver_lag_only.solve(seed=seed)
            t_lag_only = time.time() - t0
            nv_gap_lag_only = plan_lag_only.nv - bks_nv
            td_gap_lag_only = (plan_lag_only.cost - bks_td) / bks_td * 100.0 if bks_td > 0 else 0.0

            results["Lagrangian-Only"][inst_name].append({
                "seed": seed, "nv": plan_lag_only.nv, "td": plan_lag_only.cost, "time": t_lag_only
            })
            print(f"  {seed:<6} | {'Lagrangian-Only':<18} | {plan_lag_only.nv:<6} | {plan_lag_only.cost:<10.2f} | {nv_gap_lag_only:<+8} | {td_gap_lag_only:<+9.2f}% | {t_lag_only:<7.1f}s")

            # 3. Arm 3b: Lagrangian + HiGHS Supercharger
            cfg_lag_highs = Config(
                hybrid_iterations=iters,
                alns_iterations=iters,
                early_stop_patience=early_stop,
                adaptive_feasibility=False,
                lagrangian_penalty=True,
                lagrangian_theta=2.0,
                lagrangian_stall_limit=20,
                penalty_search_enabled=False,
                highs_plateau_synthesis=True,
            )
            t0 = time.time()
            solver_lag_highs = HybridDDQNSolver(inst, cfg_lag_highs)
            plan_lag_highs, _ = solver_lag_highs.solve(seed=seed)
            t_lag_highs = time.time() - t0
            nv_gap_lag_highs = plan_lag_highs.nv - bks_nv
            td_gap_lag_highs = (plan_lag_highs.cost - bks_td) / bks_td * 100.0 if bks_td > 0 else 0.0

            results["Lagrangian+HiGHS"][inst_name].append({
                "seed": seed, "nv": plan_lag_highs.nv, "td": plan_lag_highs.cost, "time": t_lag_highs
            })
            print(f"  {seed:<6} | {'Lagrangian+HiGHS':<18} | {plan_lag_highs.nv:<6} | {plan_lag_highs.cost:<10.2f} | {nv_gap_lag_highs:<+8} | {td_gap_lag_highs:<+9.2f}% | {t_lag_highs:<7.1f}s")

    # Summary Table
    print("\n" + "=" * 135)
    print("📊 3-WAY ISOLATED ABLATION SUMMARY (Mean over seeds)")
    print("=" * 135)
    print(f"{'Instance':<10} | {'BKS (NV/TD)':<16} | {'Adaptive (NV/TD)':<22} | {'Lagrangian-Only (NV/TD)':<26} | {'Lagrangian+HiGHS (NV/TD)':<26} | {'Net ΔNV':<8}")
    print("-" * 135)

    for inst_name in instances:
        bks_ref = BKS.get(inst_name, {"nv": 0, "td": 0.0})
        bks_str = f"{bks_ref['nv']} / {bks_ref['td']:.1f}"

        ad_nvs = [r["nv"] for r in results["Adaptive"][inst_name]]
        ad_tds = [r["td"] for r in results["Adaptive"][inst_name]]
        lo_nvs = [r["nv"] for r in results["Lagrangian-Only"][inst_name]]
        lo_tds = [r["td"] for r in results["Lagrangian-Only"][inst_name]]
        lh_nvs = [r["nv"] for r in results["Lagrangian+HiGHS"][inst_name]]
        lh_tds = [r["td"] for r in results["Lagrangian+HiGHS"][inst_name]]

        str_ad = f"{np.mean(ad_nvs):.2f} / {np.mean(ad_tds):.2f}"
        str_lo = f"{np.mean(lo_nvs):.2f} / {np.mean(lo_tds):.2f}"
        str_lh = f"{np.mean(lh_nvs):.2f} / {np.mean(lh_tds):.2f}"
        net_dnv = np.mean(lh_nvs) - np.mean(ad_nvs)

        print(f"{inst_name:<10} | {bks_str:<16} | {str_ad:<22} | {str_lo:<26} | {str_lh:<26} | {net_dnv:<+8.2f}")

    print("=" * 135)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--instances", nargs="+", default=["RC101", "RC104", "RC202"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 123, 7])
    parser.add_argument("--iters", type=int, default=1000)
    parser.add_argument("--early_stop", type=int, default=250)
    args = parser.parse_args()

    run_ablation(args.instances, args.seeds, iters=args.iters, early_stop=args.early_stop)

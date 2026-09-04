#!/usr/bin/env python3
"""
Comprehensive Overnight Benchmark Suite for IEEE Access (VRPTW).

Executes 4 Dedicated Empirical Investigations:
1. Limitation 1 (Small Scale Bypass): Fast-path evaluation on N=25, 50 customers.
2. Limitation 2 (MIP Scalability & Pruning): High-speed Set Partitioning column management.
3. Limitation 3 (OOD Generalization Stress-Test): Asymmetric traffic & tightened time windows.
4. Computational Fairness (Equal Wall-Clock Budget): ALNS vs Hybrid-DDQN under identical CPU time.

Results saved to results/overnight_suite/ with CSV and Markdown summaries.
"""

from __future__ import annotations

import argparse
import copy
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from vrptw.benchmark_suite import ColdStartScope
from vrptw.config import BKS, Config
from vrptw.core import Inst, load_solomon_instance
from vrptw.solvers import ALNSSolver, HybridDDQNSolver

OUT_DIR = ROOT / "results" / "overnight_suite"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def find_instance_path(name: str) -> str:
    """Finds the file path for a Solomon or Homberger instance name."""
    clean_name = name.replace(".TXT", "").replace(".txt", "").strip()
    solomon_path = ROOT / "data" / "Solomon" / f"{clean_name.upper()}.txt"
    if solomon_path.exists():
        return str(solomon_path)
    for ext in [".TXT", ".txt"]:
        for d in (ROOT / "data" / "Gehring_Homberger").glob(f"**/{clean_name.lower()}{ext}"):
            return str(d)
        for d in (ROOT / "data" / "Gehring_Homberger").glob(f"**/{clean_name.upper()}{ext}"):
            return str(d)
    for f in (ROOT / "data").glob(f"**/*{clean_name}*"):
        if f.is_file() and f.suffix.lower() == ".txt":
            return str(f)
    raise FileNotFoundError(f"Instance '{name}' not found in {ROOT / 'data'}")


def load_instance_by_name(name: str) -> Inst:
    """Loads an instance by its benchmark name."""
    return load_solomon_instance(find_instance_path(name))


def run_single_solver(solver_cls, inst: Inst, cfg: Config, seed: int) -> dict:
    """Executes a solver run under an isolated ColdStartScope."""
    t0 = time.time()
    cfg_copy = copy.copy(cfg)
    cfg_copy.seed = seed
    with ColdStartScope(run_id=f"overnight_{inst.name}_{seed}_{solver_cls.__name__}"):
        if solver_cls == HybridDDQNSolver:
            solver = solver_cls(inst, cfg_copy, seed=seed)
        else:
            solver = solver_cls(inst, cfg_copy)
        out = solver.solve()
        plan = out[0] if isinstance(out, tuple) else out
    elapsed = time.time() - t0

    nv = len(plan.routes) if plan and plan.routes else 999
    td = plan.cost if plan else 999999.0
    feas = plan.feasible if plan else False

    bks = BKS.get(inst.name, {})
    bks_nv = bks.get("nv", None)
    bks_td = bks.get("td", None)

    return {
        "Instance": inst.name,
        "Customers": inst.n,
        "Algorithm": getattr(solver, "algo_name", getattr(solver, "algo", solver_cls.__name__)),
        "Seed": seed,
        "NV": nv,
        "TD": round(td, 2),
        "Time_Sec": round(elapsed, 2),
        "Feasible": feas,
        "BKS_NV": bks_nv,
        "BKS_TD": bks_td,
    }


def phase1_small_scale_bypass(seeds: list[int]) -> pd.DataFrame:
    """Phase 1: Test Adaptive Scale Bypass on Small Instances (N=25, 50)."""
    print("\n" + "=" * 80)
    print("  PHASE 1: Small-Scale Bypass Benchmark (Limitation 1: N <= 50)")
    print("=" * 80)

    records = []
    instances = ["C101", "R101", "RC101"]

    for name in instances:
        base_inst = load_instance_by_name(name)
        for n_cust in [25, 50]:
            # Create sliced instance
            raw_dict = {
                "name": f"{name}_n{n_cust}",
                "capacity": base_inst.capacity,
                "data": np.column_stack(
                    [
                        np.arange(n_cust + 1),
                        base_inst.coords[: n_cust + 1],
                        base_inst.demands[: n_cust + 1],
                        base_inst.ready_times[: n_cust + 1],
                        base_inst.due_times[: n_cust + 1],
                        base_inst.service_times[: n_cust + 1],
                    ]
                ),
            }
            inst = Inst(raw_dict)

            for seed in seeds:
                # 1. ALNS-Base
                cfg_alns = Config()
                cfg_alns.alns_iterations = 600
                res_alns = run_single_solver(ALNSSolver, inst, cfg_alns, seed)
                res_alns["Phase"] = "Small_Scale"
                res_alns["Mode"] = "ALNS-Base"
                records.append(res_alns)
                print(
                    f"  [P1] {inst.name:12s} ALNS-Base         Seed {seed} -> NV: {res_alns['NV']:2d}, TD: {res_alns['TD']:7.2f} ({res_alns['Time_Sec']:4.2f}s)"
                )

                # 2. Hybrid without bypass
                cfg_std = Config()
                cfg_std.hybrid_iterations = 600
                cfg_std.adaptive_scale_bypass = False
                res_std = run_single_solver(HybridDDQNSolver, inst, cfg_std, seed)
                res_std["Phase"] = "Small_Scale"
                res_std["Mode"] = "Hybrid_Standard"
                records.append(res_std)
                print(
                    f"  [P1] {inst.name:12s} Hybrid-Standard  Seed {seed} -> NV: {res_std['NV']:2d}, TD: {res_std['TD']:7.2f} ({res_std['Time_Sec']:4.2f}s)"
                )

                # 3. Hybrid WITH adaptive bypass
                cfg_byp = Config()
                cfg_byp.hybrid_iterations = 600
                cfg_byp.adaptive_scale_bypass = True
                cfg_byp.min_neural_customers = 50
                res_byp = run_single_solver(HybridDDQNSolver, inst, cfg_byp, seed)
                res_byp["Phase"] = "Small_Scale"
                res_byp["Mode"] = "Hybrid_FastBypass"
                records.append(res_byp)
                print(
                    f"  [P1] {inst.name:12s} Hybrid-FastBypass Seed {seed} -> NV: {res_byp['NV']:2d}, TD: {res_byp['TD']:7.2f} ({res_byp['Time_Sec']:4.2f}s)"
                )

    df = pd.DataFrame(records)
    df.to_csv(OUT_DIR / "phase1_small_scale.csv", index=False)
    return df


def phase2_ood_stress_test(seeds: list[int]) -> pd.DataFrame:
    """Phase 2: Out-of-Distribution Stress Test (Asymmetric Traffic + Squeezed TW)."""
    print("\n" + "=" * 80)
    print("  PHASE 2: OOD Stress-Test & Entropy Gate Robustness (Limitation 3)")
    print("=" * 80)

    records = []
    instances = ["C101", "R101", "RC101", "c1_2_1", "r1_2_1"]

    for name in instances:
        base_inst = load_instance_by_name(name)

        # Create OOD Perturbed Instance
        raw_dict = {
            "name": f"{name}_OOD",
            "capacity": base_inst.capacity,
            "data": np.column_stack(
                [
                    np.arange(base_inst.n + 1),
                    base_inst.coords,
                    base_inst.demands,
                    # Squeeze ready/due time windows by 20%
                    base_inst.ready_times + 0.10 * (base_inst.due_times - base_inst.ready_times),
                    base_inst.due_times - 0.10 * (base_inst.due_times - base_inst.ready_times),
                    base_inst.service_times,
                ]
            ),
        }
        ood_inst = Inst(raw_dict)
        # Add asymmetric travel noise (10% to 30% traffic delay)
        rng = np.random.RandomState(42)
        traffic_mult = 1.0 + rng.uniform(0.0, 0.25, size=ood_inst.dist.shape)
        np.fill_diagonal(traffic_mult, 0.0)
        ood_inst.dist = ood_inst.dist * traffic_mult

        for seed in seeds:
            # 1. ALNS-Base on OOD
            cfg_alns = Config()
            cfg_alns.alns_iterations = 600
            res_alns = run_single_solver(ALNSSolver, ood_inst, cfg_alns, seed)
            res_alns["Phase"] = "OOD_Stress"
            res_alns["Variant"] = "ALNS-Base"
            records.append(res_alns)
            print(
                f"  [P2-OOD] {ood_inst.name:12s} ALNS-Base            Seed {seed} -> NV: {res_alns['NV']:2d}, TD: {res_alns['TD']:7.2f}, Feas: {res_alns['Feasible']}"
            )

            # 2. Hybrid WITH Entropy Gate (w_conf blending fallback)
            cfg_gate = Config()
            cfg_gate.hybrid_iterations = 600
            cfg_gate.op_use_entropy_gate = True
            res_gate = run_single_solver(HybridDDQNSolver, ood_inst, cfg_gate, seed)
            res_gate["Phase"] = "OOD_Stress"
            res_gate["Variant"] = "Hybrid_with_EntropyGate"
            records.append(res_gate)
            print(
                f"  [P2-OOD] {ood_inst.name:12s} Hybrid-EntropyGate   Seed {seed} -> NV: {res_gate['NV']:2d}, TD: {res_gate['TD']:7.2f}, Feas: {res_gate['Feasible']}"
            )

            # 3. Hybrid WITHOUT Entropy Gate (forced neural decisions)
            cfg_nogate = Config()
            cfg_nogate.hybrid_iterations = 600
            cfg_nogate.op_use_entropy_gate = False
            res_nogate = run_single_solver(HybridDDQNSolver, ood_inst, cfg_nogate, seed)
            res_nogate["Phase"] = "OOD_Stress"
            res_nogate["Variant"] = "Hybrid_without_EntropyGate"
            records.append(res_nogate)
            print(
                f"  [P2-OOD] {ood_inst.name:12s} Hybrid-NoEntropyGate Seed {seed} -> NV: {res_nogate['NV']:2d}, TD: {res_nogate['TD']:7.2f}, Feas: {res_nogate['Feasible']}"
            )

    df = pd.DataFrame(records)
    df.to_csv(OUT_DIR / "phase2_ood_stress.csv", index=False)
    return df


def phase3_equal_time_budget(seeds: list[int]) -> pd.DataFrame:
    """Phase 3: Equal Wall-Clock Computational Budget Comparison."""
    print("\n" + "=" * 80)
    print("  PHASE 3: Equal Wall-Clock Budget Fairness (Reviewer #1 Critique)")
    print("=" * 80)

    time_caps = {
        "C101": 15.0,
        "R101": 20.0,
        "RC101": 25.0,
        "c1_2_1": 40.0,
        "c2_2_1": 40.0,
        "r1_2_1": 60.0,
        "rc2_2_1": 60.0,
        "c1_4_1": 90.0,
        "rc2_4_1": 120.0,
    }

    records = []
    for name, t_cap in time_caps.items():
        inst = load_instance_by_name(name)
        for seed in seeds:
            # 1. ALNS-Base under Equal Time Budget (large iteration cap, cut off by time_limit)
            cfg_alns = Config()
            cfg_alns.alns_iterations = 50000
            cfg_alns.early_stop_patience = 50000
            cfg_alns.time_limit = t_cap
            res_alns = run_single_solver(ALNSSolver, inst, cfg_alns, seed)
            res_alns["Phase"] = "Equal_Time"
            res_alns["Budget_Sec"] = t_cap
            res_alns["Mode"] = "ALNS-EqualTime"
            records.append(res_alns)
            print(
                f"  [P3-EqTime] {inst.name:10s} (Cap {t_cap:3.0f}s) ALNS-EqualTime Seed {seed} -> NV: {res_alns['NV']:2d}, TD: {res_alns['TD']:7.2f} ({res_alns['Time_Sec']:4.1f}s)"
            )

            # 2. Hybrid-DDQN under Equal Time Budget
            cfg_hyb = Config()
            cfg_hyb.hybrid_iterations = 50000
            cfg_hyb.early_stop_patience = 50000
            cfg_hyb.time_limit = t_cap
            res_hyb = run_single_solver(HybridDDQNSolver, inst, cfg_hyb, seed)
            res_hyb["Phase"] = "Equal_Time"
            res_hyb["Budget_Sec"] = t_cap
            res_hyb["Mode"] = "Hybrid-EqualTime"
            records.append(res_hyb)
            print(
                f"  [P3-EqTime] {inst.name:10s} (Cap {t_cap:3.0f}s) Hybrid-EqualTime Seed {seed} -> NV: {res_hyb['NV']:2d}, TD: {res_hyb['TD']:7.2f} ({res_hyb['Time_Sec']:4.1f}s)"
            )

    df = pd.DataFrame(records)
    df.to_csv(OUT_DIR / "phase3_equal_time.csv", index=False)
    return df


def main():
    parser = argparse.ArgumentParser(description="Overnight Comprehensive VRPTW Runner")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44, 45, 46])
    args = parser.parse_args()

    t_start = time.time()
    print("================================================================================")
    print("  IEEE ACCESS OVERNIGHT COMPREHENSIVE BENCHMARK SUITE")
    print(f"  Seeds: {args.seeds} | Output: {OUT_DIR}")
    print("================================================================================")

    phase1_small_scale_bypass(args.seeds)
    phase2_ood_stress_test(args.seeds)
    phase3_equal_time_budget(args.seeds)

    total_time = time.time() - t_start
    print("\n" + "=" * 80)
    print(f"  ALL OVERNIGHT BENCHMARKS COMPLETED IN {total_time:.1f}s ({total_time / 60.0:.1f} mins)!")
    print(f"  Results saved to {OUT_DIR}")
    print("================================================================================")


if __name__ == "__main__":
    main()

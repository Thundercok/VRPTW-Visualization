"""
Step 1+2+3: MILP Column Cap Ablation on Homberger-200 and Homberger-400.

Measures:
1. Is the column cap (800) actually binding (i.e. pool_size > 800 during SP recombination)?
2. HiGHS solve time (avg ms, max ms, total s) as cap increases from 800 -> 1500 -> 2000.
3. Solution quality (NV, TD) across multiple seeds under strict cold-starts.
"""

from __future__ import annotations

import os
import sys
import time

import numpy as np

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_REPO, "src"))

import vrptw.pool as pool_module
from vrptw.config import Config
from vrptw.core import load_solomon_instance
from vrptw.solvers import HybridDDQNSolver

# Instrument _milp_recombine to track solve time and column statistics
_original_milp_recombine = pool_module._milp_recombine


class MILPInstrumenter:
    def __init__(self):
        self.calls: list[dict] = []

    def clear(self):
        self.calls.clear()

    def wrap_milp_recombine(self, *args, **kwargs):
        route_records = args[0] if args else kwargs.get("route_records")
        cfg = args[2] if len(args) > 2 else kwargs.get("cfg")
        max_cols = getattr(cfg, "milp_max_cols", 800)

        pool_size_before = len(route_records) if route_records else 0
        t0 = time.perf_counter()
        result = _original_milp_recombine(*args, **kwargs)
        solve_time_ms = (time.perf_counter() - t0) * 1000.0

        is_capped = pool_size_before > max_cols
        self.calls.append({
            "pool_size": pool_size_before,
            "max_cols": max_cols,
            "capped": is_capped,
            "solve_time_ms": solve_time_ms,
            "success": result is not None,
        })
        return result


instrumenter = MILPInstrumenter()
pool_module._milp_recombine = instrumenter.wrap_milp_recombine


BENCHMARK_INSTANCES = [
    # 200 customers
    ("R1_2_1 (200c)", "data/Gehring_Homberger/homberger_200_customer_instances/R1_2_1.TXT"),
    ("RC1_2_1 (200c)", "data/Gehring_Homberger/homberger_200_customer_instances/RC1_2_1.TXT"),
    ("C1_2_1 (200c)", "data/Gehring_Homberger/homberger_200_customer_instances/C1_2_1.TXT"),
    # 400 customers
    ("R1_4_1 (400c)", "data/Gehring_Homberger/homberger_400_customer_instances/R1_4_1.TXT"),
    ("RC1_4_1 (400c)", "data/Gehring_Homberger/homberger_400_customer_instances/RC1_4_1.TXT"),
    ("C1_4_1 (400c)", "data/Gehring_Homberger/homberger_400_customer_instances/C1_4_1.TXT"),
]

CAP_VARIANTS = [
    ("Cap=800 (Baseline)", 800),
    ("Cap=1500", 1500),
    ("Cap=2000", 2000),
]
SEEDS = [1, 2]
ITERS = 400


def run_experiment():
    print("=" * 115, flush=True)
    print("EXPERIMENT: MILP Column Cap Ablation on Homberger-200 & Homberger-400 Instances", flush=True)
    print("=" * 115, flush=True)

    summary_rows = []

    for inst_label, rel_path in BENCHMARK_INSTANCES:
        path = os.path.join(_REPO, rel_path)
        if not os.path.exists(path):
            print(f"Skipping {inst_label}: file not found ({path})", flush=True)
            continue
        inst = load_solomon_instance(path)

        print(f"\n---> INSTANCE: {inst_label} (N={inst.n}, Capacity={inst.capacity})", flush=True)
        print(f"{'Cap Variant':<22} | {'Seed':<5} | {'NV':<4} {'TD':<10} | {'SP Calls':<9} {'Capped Calls':<13} {'Max Pool':<9} | {'Avg MILP (ms)':<14} {'Total MILP (s)':<15}", flush=True)
        print("-" * 115, flush=True)

        for cap_label, max_cols in CAP_VARIANTS:
            for seed in SEEDS:
                cfg = Config(
                    alns_iterations=ITERS,
                    hybrid_iterations=ITERS,
                    early_stop_patience=10**9,
                    split_enabled=False,
                    time_limit=None,
                    time_limit_per_customer=0.0,
                    milp_max_cols=max_cols,
                )

                instrumenter.clear()
                solver = HybridDDQNSolver(inst, cfg, seed=seed)
                best_plan, _ = solver.solve(seed=seed)

                calls = instrumenter.calls
                n_calls = len(calls)
                capped_calls = sum(1 for c in calls if c["capped"])
                max_pool = max((c["pool_size"] for c in calls), default=0)
                avg_ms = float(np.mean([c["solve_time_ms"] for c in calls])) if calls else 0.0
                total_s = float(np.sum([c["solve_time_ms"] for c in calls])) / 1000.0 if calls else 0.0

                summary_rows.append({
                    "inst": inst_label,
                    "cap": cap_label,
                    "max_cols": max_cols,
                    "seed": seed,
                    "nv": best_plan.nv,
                    "cost": best_plan.cost,
                    "n_calls": n_calls,
                    "capped_calls": capped_calls,
                    "max_pool": max_pool,
                    "avg_ms": avg_ms,
                    "total_s": total_s,
                })

                print(
                    f"{cap_label:<22} | {seed:<5} | {best_plan.nv:<4} {best_plan.cost:<10.2f} | "
                    f"{n_calls:<9} {capped_calls:<13} {max_pool:<9} | {avg_ms:<14.2f} {total_s:<15.3f}",
                    flush=True,
                )

    print("\n" + "=" * 115, flush=True)
    print("AGGREGATED SUMMARY TABLE", flush=True)
    print("=" * 115, flush=True)
    for cap_label, _ in CAP_VARIANTS:
        subset = [r for r in summary_rows if r["cap"] == cap_label]
        if subset:
            mean_nv = np.mean([r["nv"] for r in subset])
            mean_td = np.mean([r["cost"] for r in subset])
            mean_capped_frac = np.mean([r["capped_calls"] / max(r["n_calls"], 1) for r in subset]) * 100.0
            avg_solve_ms = np.mean([r["avg_ms"] for r in subset])
            total_solve_s = np.sum([r["total_s"] for r in subset])
            print(
                f"{cap_label:<22} | Mean NV={mean_nv:.2f} | Mean TD={mean_td:.2f} | "
                f"Capped Calls={mean_capped_frac:.1f}% | Avg HiGHS Time={avg_solve_ms:.1f}ms | Total HiGHS Time={total_solve_s:.2f}s",
                flush=True,
            )
    print("=" * 115, flush=True)


if __name__ == "__main__":
    run_experiment()

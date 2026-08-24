"""
Diagnostic and Stress-Test A/B Script for RoutePool Slot A Eviction.

1. Test with default pool limit (n=100 -> limit=1000, threshold=1100) and track:
   - final_pool_size
   - trim_call_count (did _trim() ever execute?)

2. Test with constrained pool limit (e.g. limit=150, threshold=150) and track:
   - final_pool_size
   - trim_call_count
   - Variant A (cost / len) vs Variant B ((cost / len, -len))
"""

from __future__ import annotations

import os
import sys

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_REPO, "src"))

from vrptw.config import Config
from vrptw.core import load_solomon_instance
from vrptw.pool import RoutePool, RouteRecord, _cover_key
from vrptw.solvers import HybridDDQNSolver


class InstrumentableRoutePool(RoutePool):
    def __init__(self, inst, cfg, slot_a_mode="cost_per_stop", custom_limit=None):
        super().__init__(inst, cfg)
        self.slot_a_mode = slot_a_mode
        self.trim_count = 0
        if custom_limit is not None:
            self.cfg.route_pool_limit = custom_limit
            self.cfg.route_pool_max_per_customer = max(8, custom_limit // 15)

    def _trim(self) -> None:
        limit = self.cfg.route_pool_limit
        # If custom limit is set, trim whenever length exceeds limit; else default limit+100
        threshold = limit if self.cfg.route_pool_limit < 500 else limit + 100
        if len(self._routes) <= threshold:
            return

        self.trim_count += 1
        slot_b = max(limit // 4, 8)
        usage: dict[int, int] = {}
        kept: dict[tuple[int, ...], RouteRecord] = {}

        len_ranked = sorted(self._routes.values(), key=lambda r: -len(r.nodes))

        if self.slot_a_mode == "cost_per_stop":
            # Variant A: cost / len
            eff_ranked = sorted(
                self._routes.values(),
                key=lambda r: r.cost / max(len(r.nodes), 1),
            )
        elif self.slot_a_mode == "cost_per_stop_and_len":
            # Variant B: (cost / len, -len)
            eff_ranked = sorted(
                self._routes.values(),
                key=lambda r: (r.cost / max(len(r.nodes), 1), -len(r.nodes)),
            )
        else:
            raise ValueError(f"Unknown slot_a_mode {self.slot_a_mode}")

        max_per = self.cfg.route_pool_max_per_customer

        def _admit(rec: RouteRecord) -> bool:
            if rec.nodes in kept:
                return False
            under = all(usage.get(n, 0) < max_per for n in rec.nodes)
            if not under and len(kept) >= limit // 3:
                return False
            kept[rec.nodes] = rec
            for n in rec.nodes:
                usage[n] = usage.get(n, 0) + 1
            return True

        for rec in len_ranked:
            if len(kept) >= slot_b:
                break
            _admit(rec)

        for rec in eff_ranked:
            if len(kept) >= limit:
                break
            _admit(rec)

        if len(kept) < limit:
            for rec in len_ranked:
                if len(kept) >= limit:
                    break
                _admit(rec)

        self._routes = kept
        self._cover_to_key = {_cover_key(k): k for k in kept}


INSTANCES = [
    ("RC101", "data/Solomon/rc101.txt"),
    ("RC105", "data/Solomon/rc105.txt"),
    ("R101", "data/Solomon/r101.txt"),
    ("C101", "data/Solomon/c101.txt"),
]


def run_diagnostics():
    print("=" * 80, flush=True)
    print("DIAGNOSTIC 1: Did _trim() execute under default route_pool_limit (1000)?", flush=True)
    print("=" * 80, flush=True)
    print(f"{'Inst':<8} {'Seed':<5} | {'Final Pool Size':<17} {'Limit (Threshold)':<20} {'Trim Count':<12} {'Did Trim?':<10}", flush=True)
    print("-" * 80, flush=True)

    for name, rel_path in INSTANCES:
        path = os.path.join(_REPO, rel_path)
        if not os.path.exists(path):
            continue
        inst = load_solomon_instance(path)

        for seed in [1, 2]:
            cfg = Config(
                alns_iterations=400,
                hybrid_iterations=400,
                early_stop_patience=10**9,
                split_enabled=False,
                time_limit=None,
                time_limit_per_customer=0.0,
            )
            solver = HybridDDQNSolver(inst, cfg, seed=seed)
            pool = InstrumentableRoutePool(inst, cfg, slot_a_mode="cost_per_stop")
            solver.pool = pool
            solver.solve(seed=seed)

            limit_thresh = f"{pool.cfg.route_pool_limit} ({pool.cfg.route_pool_limit + 100})"
            did_trim = "YES" if pool.trim_count > 0 else "NO (NEVER)"
            print(f"{name:<8} {seed:<5} | {len(pool._routes):<17} {limit_thresh:<20} {pool.trim_count:<12} {did_trim:<10}", flush=True)

    print("\n" + "=" * 80, flush=True)
    print("DIAGNOSTIC 2: Stress-Test A/B with Constrained Pool Limit (Limit=120) to FORCE Trimming", flush=True)
    print("=" * 80, flush=True)
    print(f"{'Inst':<8} {'Seed':<5} | {'Var A NV':<9} {'Var A TD':<10} {'Trim A':<7} | {'Var B NV':<9} {'Var B TD':<10} {'Trim B':<7} | {'Winner':<8}", flush=True)
    print("-" * 80, flush=True)

    a_wins = 0
    b_wins = 0
    ties = 0

    for name, rel_path in INSTANCES:
        path = os.path.join(_REPO, rel_path)
        if not os.path.exists(path):
            continue
        inst = load_solomon_instance(path)

        for seed in [1, 2, 3]:
            cfg = Config(
                alns_iterations=400,
                hybrid_iterations=400,
                early_stop_patience=10**9,
                split_enabled=False,
                time_limit=None,
                time_limit_per_customer=0.0,
            )

            # Variant A (forced trim at limit=120)
            solver_a = HybridDDQNSolver(inst, cfg, seed=seed)
            pool_a = InstrumentableRoutePool(inst, cfg, slot_a_mode="cost_per_stop", custom_limit=120)
            solver_a.pool = pool_a
            plan_a, _ = solver_a.solve(seed=seed)

            # Variant B (forced trim at limit=120)
            solver_b = HybridDDQNSolver(inst, cfg, seed=seed)
            pool_b = InstrumentableRoutePool(inst, cfg, slot_a_mode="cost_per_stop_and_len", custom_limit=120)
            solver_b.pool = pool_b
            plan_b, _ = solver_b.solve(seed=seed)

            if plan_b.dominates(plan_a):
                winner = "B (Cand)"
                b_wins += 1
            elif plan_a.dominates(plan_b):
                winner = "A (Base)"
                a_wins += 1
            elif abs(plan_a.cost - plan_b.cost) < 1e-4:
                winner = "Tie"
                ties += 1
            elif plan_b.cost < plan_a.cost:
                winner = "B (TD)"
                b_wins += 1
            else:
                winner = "A (TD)"
                a_wins += 1

            print(
                f"{name:<8} {seed:<5} | {plan_a.nv:<9} {plan_a.cost:<10.2f} {pool_a.trim_count:<7} | "
                f"{plan_b.nv:<9} {plan_b.cost:<10.2f} {pool_b.trim_count:<7} | {winner:<8}",
                flush=True,
            )

    print("=" * 80, flush=True)
    print(f"Stress-Test Summary: Variant A wins = {a_wins}, Variant B wins = {b_wins}, Ties = {ties}", flush=True)


if __name__ == "__main__":
    run_diagnostics()

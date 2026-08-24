"""
A/B Test Slot A Sorting in RoutePool:
Variant A (Baseline): key = cost / len
Variant B (Candidate): key = (cost / len, -len) [prioritise longer routes on cost/len tie]
"""

import os
import sys

import numpy as np

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_REPO, "src"))

from vrptw.config import Config
from vrptw.core import load_solomon_instance
from vrptw.pool import RoutePool, RouteRecord, _cover_key
from vrptw.solvers import HybridDDQNSolver


class RoutePoolSlotABaseline(RoutePool):
    """Slot A sorts strictly by cost / len."""
    def _trim(self) -> None:
        limit = self.cfg.route_pool_limit
        if len(self._routes) <= limit + 100:
            return

        slot_b = max(limit // 4, 8)
        usage: dict[int, int] = {}
        kept: dict[tuple[int, ...], RouteRecord] = {}

        len_ranked = sorted(self._routes.values(), key=lambda r: -len(r.nodes))
        eff_ranked = sorted(
            self._routes.values(),
            key=lambda r: r.cost / max(len(r.nodes), 1),
        )

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


class RoutePoolSlotACandidate(RoutePool):
    """Slot A sorts by (cost / len, -len) — breaking cost/len ties in favor of longer routes."""
    def _trim(self) -> None:
        limit = self.cfg.route_pool_limit
        if len(self._routes) <= limit + 100:
            return

        slot_b = max(limit // 4, 8)
        usage: dict[int, int] = {}
        kept: dict[tuple[int, ...], RouteRecord] = {}

        len_ranked = sorted(self._routes.values(), key=lambda r: -len(r.nodes))
        eff_ranked = sorted(
            self._routes.values(),
            key=lambda r: (r.cost / max(len(r.nodes), 1), -len(r.nodes)),
        )

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
    ("RC108", "data/Solomon/rc108.txt"),
    ("R101", "data/Solomon/r101.txt"),
    ("R105", "data/Solomon/r105.txt"),
    ("C101", "data/Solomon/c101.txt"),
    ("C201", "data/Solomon/c201.txt"),
]
SEEDS = [1, 2, 3]
ITERS = 400


def run_ab():
    print("=== A/B TEST: Slot A Sorting in RoutePool ===", flush=True)
    print("Variant A: cost / len", flush=True)
    print("Variant B: (cost / len, -len)", flush=True)
    print(f"{'Inst':<8} {'Seed':<5} | {'Var A NV':<9} {'Var A TD':<10} | {'Var B NV':<9} {'Var B TD':<10} | {'Winner':<8}", flush=True)
    print("-" * 75, flush=True)

    a_wins = 0
    b_wins = 0
    ties = 0

    a_nvs = []
    b_nvs = []
    a_tds = []
    b_tds = []

    for name, rel_path in INSTANCES:
        path = os.path.join(_REPO, rel_path)
        if not os.path.exists(path):
            continue
        inst = load_solomon_instance(path)

        for seed in SEEDS:
            cfg = Config(
                alns_iterations=ITERS,
                hybrid_iterations=ITERS,
                early_stop_patience=10**9,
                split_enabled=False,
                time_limit=None,
                time_limit_per_customer=0.0,
            )

            # Run Variant A
            solver_a = HybridDDQNSolver(inst, cfg, seed=seed)
            solver_a.pool = RoutePoolSlotABaseline(inst, cfg)
            plan_a, _ = solver_a.solve(seed=seed)

            # Run Variant B
            solver_b = HybridDDQNSolver(inst, cfg, seed=seed)
            solver_b.pool = RoutePoolSlotACandidate(inst, cfg)
            plan_b, _ = solver_b.solve(seed=seed)

            a_nvs.append(plan_a.nv)
            b_nvs.append(plan_b.nv)
            a_tds.append(plan_a.cost)
            b_tds.append(plan_b.cost)

            # Dominance check
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

            print(f"{name:<8} {seed:<5} | {plan_a.nv:<9} {plan_a.cost:<10.2f} | {plan_b.nv:<9} {plan_b.cost:<10.2f} | {winner:<8}", flush=True)

    print("=" * 75, flush=True)
    print(f"Summary: Variant A wins = {a_wins}, Variant B wins = {b_wins}, Ties = {ties}", flush=True)
    print(f"Mean NV: Variant A = {np.mean(a_nvs):.2f}, Variant B = {np.mean(b_nvs):.2f}", flush=True)
    print(f"Mean TD: Variant A = {np.mean(a_tds):.2f}, Variant B = {np.mean(b_tds):.2f}", flush=True)


if __name__ == "__main__":
    run_ab()

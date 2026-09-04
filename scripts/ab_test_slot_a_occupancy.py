"""
Ablation Stress-Test (Limit=120):
Compare Slot A Sorting Formulations:
1. Baseline: cost / len
2. Occupancy Boosted: cost / (len ** 1.3)
3. Load Occupancy: cost / max(load / capacity, 0.05)
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


class CustomSlotAPool(RoutePool):
    def __init__(self, inst, cfg, mode="baseline", limit=120):
        super().__init__(inst, cfg)
        self.mode = mode
        self.cfg.route_pool_limit = limit
        self.cfg.route_pool_max_per_customer = max(8, limit // 15)
        self.trim_count = 0

    def _trim(self) -> None:
        limit = self.cfg.route_pool_limit
        if len(self._routes) <= limit:
            return

        self.trim_count += 1
        slot_b = max(limit // 4, 8)
        usage: dict[int, int] = {}
        kept: dict[tuple[int, ...], RouteRecord] = {}

        len_ranked = sorted(self._routes.values(), key=lambda r: -len(r.nodes))

        if self.mode == "baseline":
            # 1. Baseline: cost / len
            eff_ranked = sorted(
                self._routes.values(),
                key=lambda r: r.cost / max(len(r.nodes), 1),
            )
        elif self.mode == "occupancy_boosted":
            # 2. Superlinear length scaling: cost / (len ^ 1.3)
            eff_ranked = sorted(
                self._routes.values(),
                key=lambda r: r.cost / max(len(r.nodes) ** 1.3, 1.0),
            )
        elif self.mode == "load_occupancy":
            # 3. Load capacity utilization: cost / (load / capacity)
            cap = max(self.inst.capacity, 1.0)
            eff_ranked = sorted(
                self._routes.values(),
                key=lambda r: r.cost / max(r.load / cap, 0.05),
            )
        else:
            raise ValueError(f"Unknown mode: {self.mode}")

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
SEEDS = [1, 2, 3]


def run():
    print("=" * 105, flush=True)
    print("SLOT A ABLATION STRESS-TEST (Limit=120, Forced Trimming > 1000 times/run)", flush=True)
    print("=" * 105, flush=True)
    print(
        f"{'Inst':<8} {'Seed':<5} | "
        f"{'Base (cost/len)':<18} | "
        f"{'OccBoost (len^1.3)':<18} | "
        f"{'LoadOcc (cost/load)':<18} | "
        f"{'Comparison':<15}",
        flush=True,
    )
    print("-" * 105, flush=True)

    results = []

    for name, rel_path in INSTANCES:
        path = os.path.join(_REPO, rel_path)
        if not os.path.exists(path):
            continue
        inst = load_solomon_instance(path)

        for seed in SEEDS:
            cfg = Config(
                alns_iterations=400,
                hybrid_iterations=400,
                early_stop_patience=10**9,
                split_enabled=False,
                time_limit=None,
                time_limit_per_customer=0.0,
            )

            # 1. Baseline
            s1 = HybridDDQNSolver(inst, cfg, seed=seed)
            p1 = CustomSlotAPool(inst, cfg, mode="baseline", limit=120)
            s1.pool = p1
            res1, _ = s1.solve(seed=seed)

            # 2. Occupancy Boosted
            s2 = HybridDDQNSolver(inst, cfg, seed=seed)
            p2 = CustomSlotAPool(inst, cfg, mode="occupancy_boosted", limit=120)
            s2.pool = p2
            res2, _ = s2.solve(seed=seed)

            # 3. Load Occupancy
            s3 = HybridDDQNSolver(inst, cfg, seed=seed)
            p3 = CustomSlotAPool(inst, cfg, mode="load_occupancy", limit=120)
            s3.pool = p3
            res3, _ = s3.solve(seed=seed)

            b_str = f"nv={res1.nv} td={res1.cost:.2f}"
            ob_str = f"nv={res2.nv} td={res2.cost:.2f}"
            lo_str = f"nv={res3.nv} td={res3.cost:.2f}"

            all_equal = (res1.nv == res2.nv == res3.nv) and (
                abs(res1.cost - res2.cost) < 1e-4 and abs(res1.cost - res3.cost) < 1e-4
            )
            comp = "ALL TIE" if all_equal else "DIVERGENCE"
            results.append((name, seed, res1, res2, res3, all_equal))

            print(
                f"{name:<8} {seed:<5} | {b_str:<18} | {ob_str:<18} | {lo_str:<18} | {comp:<15}",
                flush=True,
            )

    print("=" * 105, flush=True)
    all_ties = sum(1 for r in results if r[5])
    print(f"Summary: {all_ties}/{len(results)} exact ties across all 3 variants under forced eviction.", flush=True)


if __name__ == "__main__":
    run()

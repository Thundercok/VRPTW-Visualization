"""
Does the trained GNN heatmap actually help the solver?

Training loss going down only shows the network fits elite plans; it says nothing
about whether the resulting heatmap guides the search to better solutions. This
runs the same instances and seeds with and without the heatmap and reports the
paired difference in vehicle count and distance.

    python scripts/validate_gnn.py [--iters 400] [--seeds 3]
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_REPO, "src"))

from vrptw.config import BKS, Config  # noqa: E402
from vrptw.core import load_solomon_instance  # noqa: E402
from vrptw.solvers import HybridDDQNSolver  # noqa: E402

INSTANCES = [
    ("R101", "data/Solomon/r101.txt"),
    ("C203", "data/Solomon/c203.txt"),
    ("RC105", "data/Solomon/rc105.txt"),
    ("RC207", "data/Solomon/rc207.txt"),
    ("r1_2_1", "data/Gehring_Homberger/homberger_200_customer_instances/R1_2_1.TXT"),
]

MODEL_PATH = os.path.join(_REPO, "docs", "model", "gnn_edge_predictor.pt")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=400)
    ap.add_argument("--seeds", type=int, default=3)
    args = ap.parse_args()

    if not os.path.exists(MODEL_PATH):
        print(f"No checkpoint at {MODEL_PATH}; train it first.")
        return

    def cfg() -> Config:
        return Config(
            hybrid_iterations=args.iters, alns_iterations=args.iters,
            early_stop_patience=10**9, split_enabled=False,
            time_limit=None, time_limit_per_customer=0.0,
        )

    rows = []
    for label, rel in INSTANCES:
        path = os.path.join(_REPO, rel)
        if not os.path.exists(path):
            print(f"  SKIP {label}")
            continue
        inst = load_solomon_instance(path)
        for seed in range(1, args.seeds + 1):
            rec = {"instance": label, "seed": seed}
            for tag, use_gnn in (("off", False), ("on", True)):
                solver = HybridDDQNSolver(inst, cfg())
                if use_gnn:
                    solver.load_gnn_model(MODEL_PATH)
                    if solver.gnn_model is None:
                        raise RuntimeError("checkpoint failed to load")
                t0 = time.time()
                best, _ = solver.solve(seed=seed)
                rec[tag] = (best.nv, best.cost, time.time() - t0, best.feasible)
            rows.append(rec)
            (nv0, td0, _t0, _f0), (nv1, td1, _t1, _f1) = rec["off"], rec["on"]
            print(f"  {label:8s} s{seed}  GNN off: nv={nv0:3d} td={td0:9.2f}  |  "
                  f"on: nv={nv1:3d} td={td1:9.2f}  | dNV={nv1-nv0:+d} dTD={td1-td0:+8.2f}",
                  flush=True)

    if not rows:
        return
    nv0 = np.array([r["off"][0] for r in rows], float)
    nv1 = np.array([r["on"][0] for r in rows], float)
    td0 = np.array([r["off"][1] for r in rows], float)
    td1 = np.array([r["on"][1] for r in rows], float)
    assert all(r["off"][3] and r["on"][3] for r in rows), "infeasible result produced"

    def gap(td):
        return np.mean([(t - BKS[r["instance"]]["td"]) / BKS[r["instance"]]["td"] * 100
                        for t, r in zip(td, rows) if r["instance"] in BKS])

    print(f"\n{len(rows)} paired runs")
    print(f"  mean NV   {nv0.mean():8.3f} -> {nv1.mean():8.3f}  ({nv1.mean()-nv0.mean():+.3f})")
    print(f"  mean TD   {td0.mean():8.2f} -> {td1.mean():8.2f}  ({td1.mean()-td0.mean():+.2f})")
    print(f"  gap-BKS   {gap(td0):7.2f}% -> {gap(td1):7.2f}%  ({gap(td1)-gap(td0):+.2f} pp)")
    print(f"  NV better {int((nv1<nv0).sum())}, worse {int((nv1>nv0).sum())}, tie {int((nv1==nv0).sum())}")
    print(f"  TD better {int((td1<td0-1e-6).sum())}, worse {int((td1>td0+1e-6).sum())}")
    try:
        from scipy.stats import wilcoxon
        for name, a, b in (("NV", nv0, nv1), ("TD", td0, td1)):
            if not np.allclose(a, b):
                print(f"  Wilcoxon {name}: p={wilcoxon(a, b)[1]:.4f}")
    except ImportError:
        pass


if __name__ == "__main__":
    main()

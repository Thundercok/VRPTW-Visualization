"""
A/B harness for solver changes that intentionally alter the search trajectory.

Deliberately version-agnostic — it parses instances itself and only passes Config
fields that exist in the checked-out code — so the same script can be run against
an older revision (e.g. via ``git stash``) and the current one.

    python scripts/ab_compare.py run --out before.json
    ... change code ...
    python scripts/ab_compare.py run --out after.json
    python scripts/ab_compare.py compare before.json after.json
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
import sys
import time

import numpy as np

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_REPO, "src"))

from vrptw.config import BKS, Config  # noqa: E402
from vrptw.core import Inst  # noqa: E402
from vrptw.solvers import ALNSSolver, HybridDDQNSolver  # noqa: E402

INSTANCES = [
    ("R101", "data/Solomon/r101.txt"),
    ("R110", "data/Solomon/r110.txt"),
    ("C101", "data/Solomon/c101.txt"),
    ("C203", "data/Solomon/c203.txt"),
    ("RC105", "data/Solomon/rc105.txt"),
    ("RC207", "data/Solomon/rc207.txt"),
    ("r1_2_1", "data/Gehring_Homberger/homberger_200_customer_instances/R1_2_1.TXT"),
    ("c1_2_1", "data/Gehring_Homberger/homberger_200_customer_instances/C1_2_1.TXT"),
    ("rc1_2_1", "data/Gehring_Homberger/homberger_200_customer_instances/RC1_2_1.TXT"),
]

SEEDS = [1, 2, 3, 4, 5]
ITERS = 400


def load_instance(path: str) -> Inst:
    with open(path, encoding="utf-8") as fh:
        lines = fh.readlines()
    return Inst(
        {
            "name": lines[0].strip(),
            "capacity": float(lines[4].strip().split()[1]),
            "data": np.array([list(map(float, ln.split())) for ln in lines[9:] if ln.strip()]),
        }
    )


def make_cfg(iters: int) -> Config:
    """Build a Config using only fields the checked-out revision defines, so this
    script runs unchanged against both sides of the comparison."""
    known = {f.name for f in dataclasses.fields(Config)}
    kwargs = {
        "alns_iterations": iters,
        "hybrid_iterations": iters,
        "early_stop_patience": 10**9,
        "split_enabled": False,
    }
    # Compare at equal iteration counts, not equal wall time: the point is to
    # isolate the effect of the search changes from the speedups.
    if "time_limit_per_customer" in known:
        kwargs["time_limit_per_customer"] = 0.0
    if "time_limit" in known:
        kwargs["time_limit"] = None
    return Config(**{k: v for k, v in kwargs.items() if k in known})


def run(iters: int) -> dict:
    records = []
    for label, rel in INSTANCES:
        path = os.path.join(_REPO, rel)
        if not os.path.exists(path):
            print(f"  SKIP {label}")
            continue
        inst = load_instance(path)
        for solver_name, cls in (("Hybrid-DDQN", HybridDDQNSolver), ("ALNS-Base", ALNSSolver)):
            for seed in SEEDS:
                solver = cls(inst, make_cfg(iters))
                t0 = time.time()
                best, _ = solver.solve(seed=seed)
                elapsed = time.time() - t0
                records.append(
                    {
                        "instance": label,
                        "solver": solver_name,
                        "seed": seed,
                        "nv": int(best.nv),
                        "cost": float(best.cost),
                        "feasible": bool(best.feasible),
                        "wall_time": round(elapsed, 3),
                    }
                )
                print(
                    f"  {label:9s} {solver_name:12s} s{seed} nv={best.nv:3d} td={best.cost:10.2f} {elapsed:7.2f}s",
                    flush=True,
                )
    return {"iters": iters, "seeds": SEEDS, "records": records}


def _key(r: dict) -> tuple:
    return (r["instance"], r["solver"], r["seed"])


def compare(before_path: str, after_path: str) -> None:
    with open(before_path, encoding="utf-8") as fh:
        before = {_key(r): r for r in json.load(fh)["records"]}
    with open(after_path, encoding="utf-8") as fh:
        after = {_key(r): r for r in json.load(fh)["records"]}

    shared = sorted(before.keys() & after.keys())
    if not shared:
        print("No overlapping records.")
        return

    nv_b = np.array([before[k]["nv"] for k in shared], dtype=float)
    nv_a = np.array([after[k]["nv"] for k in shared], dtype=float)
    td_b = np.array([before[k]["cost"] for k in shared], dtype=float)
    td_a = np.array([after[k]["cost"] for k in shared], dtype=float)
    wt_b = np.array([before[k]["wall_time"] for k in shared], dtype=float)
    wt_a = np.array([after[k]["wall_time"] for k in shared], dtype=float)

    infeasible = [k for k in shared if not after[k]["feasible"]]

    print(f"\n{len(shared)} paired runs\n")
    print(f"{'metric':16s} {'before':>12s} {'after':>12s} {'delta':>12s}")
    print("-" * 56)
    print(f"{'mean NV':16s} {nv_b.mean():12.3f} {nv_a.mean():12.3f} {nv_a.mean() - nv_b.mean():+12.3f}")
    print(f"{'mean TD':16s} {td_b.mean():12.2f} {td_a.mean():12.2f} {td_a.mean() - td_b.mean():+12.2f}")
    print(f"{'total wall(s)':16s} {wt_b.sum():12.1f} {wt_a.sum():12.1f} {(wt_b.sum() / max(wt_a.sum(), 1e-9)):11.2f}x")
    print(f"\nNV: better {int((nv_a < nv_b).sum())}, worse {int((nv_a > nv_b).sum())}, tie {int((nv_a == nv_b).sum())}")
    print(
        f"TD: better {int((td_a < td_b - 1e-6).sum())}, worse {int((td_a > td_b + 1e-6).sum())}, "
        f"tie {int((np.abs(td_a - td_b) <= 1e-6).sum())}"
    )
    if infeasible:
        print(f"\n!! {len(infeasible)} INFEASIBLE results after: {infeasible[:5]}")

    try:
        from scipy.stats import wilcoxon

        for name, b, a in (("NV", nv_b, nv_a), ("TD", td_b, td_a)):
            if np.allclose(b, a):
                print(f"\nWilcoxon {name}: identical, no test")
                continue
            stat, p = wilcoxon(b, a)
            direction = "improved" if a.mean() < b.mean() else "worsened"
            print(f"Wilcoxon {name}: p={p:.5f} ({direction})")
    except ImportError:
        print("\nscipy unavailable; skipping Wilcoxon")

    # Gap to BKS, the metric the paper reports.
    gaps_b, gaps_a = [], []
    for k in shared:
        bks = BKS.get(before[k]["instance"]) or BKS.get(before[k]["instance"].upper())
        if not bks:
            continue
        gaps_b.append((before[k]["cost"] - bks["td"]) / bks["td"] * 100)
        gaps_a.append((after[k]["cost"] - bks["td"]) / bks["td"] * 100)
    if gaps_b:
        print(
            f"\nmean gap-to-BKS TD: {np.mean(gaps_b):.2f}% -> {np.mean(gaps_a):.2f}% "
            f"({np.mean(gaps_a) - np.mean(gaps_b):+.2f} pp)"
        )


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--out", required=True)
    r.add_argument(
        "--iters", type=int, default=ITERS, help="iteration budget; raise it to compare a slower variant iso-time"
    )
    c = sub.add_parser("compare")
    c.add_argument("before")
    c.add_argument("after")
    args = ap.parse_args()

    if args.cmd == "run":
        payload = run(args.iters)
        with open(args.out, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2)
        print(f"\nWrote {len(payload['records'])} records to {args.out}")
    else:
        compare(args.before, args.after)


if __name__ == "__main__":
    main()

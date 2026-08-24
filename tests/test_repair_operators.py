"""
Every repair operator must return a feasible plan from a feasible partial plan.

``op_fts_greedy`` used to fail this on 100% of trials: its insertion test read the
*arrival* time at the predecessor without adding that node's service time, and
only checked the immediate successor's due date instead of propagating the delay
down the route. One of the five repair operators was therefore producing nothing
but rejected candidates, silently burning ~20% of the ALNS iteration budget and
13 of the 65 operator-controller actions.
"""

from __future__ import annotations

import os
import random

import numpy as np
import pytest

from vrptw.core import Inst
from vrptw.heuristics import build_greedy
from vrptw.operators import REPAIR, op_random

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

INSTANCES = [
    ("R101", "data/Solomon/r101.txt"),
    ("RC207", "data/Solomon/rc207.txt"),
    ("r1_2_1", "data/Gehring_Homberger/homberger_200_customer_instances/R1_2_1.TXT"),
]

N_TRIALS = 20


def _load(path: str) -> Inst:
    with open(path, encoding="utf-8") as fh:
        lines = fh.readlines()
    return Inst(
        {
            "name": lines[0].strip(),
            "capacity": float(lines[4].strip().split()[1]),
            "data": np.array([list(map(float, ln.split())) for ln in lines[9:] if ln.strip()]),
        }
    )


@pytest.mark.parametrize("repair_fn", REPAIR, ids=[f.__name__ for f in REPAIR])
@pytest.mark.parametrize("label,rel_path", INSTANCES, ids=[i[0] for i in INSTANCES])
def test_repair_returns_feasible_plan(repair_fn, label: str, rel_path: str) -> None:
    path = os.path.join(_REPO, rel_path)
    if not os.path.exists(path):
        pytest.skip(f"instance file missing: {path}")

    inst = _load(path)
    base = build_greedy(inst, "test")
    assert base.feasible, f"{label}: greedy construction is infeasible, test premise broken"

    random.seed(0)
    np.random.seed(0)
    infeasible = 0
    for _ in range(N_TRIALS):
        destroyed, removed = op_random(base.copy(), max(5, inst.n // 10))
        cand = repair_fn(destroyed.copy(), list(removed))
        if not cand.feasible:
            infeasible += 1

    assert infeasible == 0, (
        f"{label}/{repair_fn.__name__}: {infeasible}/{N_TRIALS} repairs produced an "
        "infeasible plan; those iterations can never be accepted."
    )


@pytest.mark.parametrize("label,rel_path", INSTANCES, ids=[i[0] for i in INSTANCES])
def test_repair_covers_all_customers(label: str, rel_path: str) -> None:
    """Repair must reinsert every removed customer exactly once."""
    path = os.path.join(_REPO, rel_path)
    if not os.path.exists(path):
        pytest.skip(f"instance file missing: {path}")

    inst = _load(path)
    base = build_greedy(inst, "test")
    random.seed(1)
    np.random.seed(1)

    for repair_fn in REPAIR:
        destroyed, removed = op_random(base.copy(), max(5, inst.n // 10))
        cand = repair_fn(destroyed.copy(), list(removed))
        visited = [n for r in cand.routes for n in r]
        assert sorted(visited) == list(range(1, inst.n + 1)), (
            f"{label}/{repair_fn.__name__}: customer set corrupted "
            f"({len(visited)} visits for {inst.n} customers)"
        )

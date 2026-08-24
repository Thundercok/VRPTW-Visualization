"""Guided Ejection Search invariants: whatever it returns must be a feasible
exact cover with exactly one route fewer — and perturbation must never break
feasibility of the partial solution it reshapes."""

import os
import random

import numpy as np
import pytest

from vrptw.core import Inst, _check_route
from vrptw.heuristics import build_greedy
from vrptw.local_search import _ges_perturb, _guided_ejection_search

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _load(name: str) -> Inst:
    path = os.path.join(_REPO, "data", "Solomon", name)
    if not os.path.exists(path):
        pytest.skip(f"instance file missing: {path}")
    with open(path, encoding="utf-8") as fh:
        lines = fh.readlines()
    return Inst(
        {
            "name": lines[0].strip(),
            "capacity": float(lines[4].strip().split()[1]),
            "data": np.array([list(map(float, ln.split())) for ln in lines[9:] if ln.strip()]),
        }
    )


@pytest.mark.parametrize("fname", ["r101.txt", "rc105.txt", "c101.txt"])
def test_ges_result_is_feasible_one_route_fewer(fname):
    inst = _load(fname)
    plan = build_greedy(inst)
    assert plan.feasible

    random.seed(3)
    target_idx = min(range(len(plan.routes)), key=lambda i: len(plan.routes[i]))
    result = _guided_ejection_search(plan, target_idx, p_counters={}, max_steps=400)

    if result is None:
        return  # elimination may genuinely be impossible from this state
    assert result.feasible
    assert result.nv == plan.nv - 1
    served = [c for r in result.routes for c in r]
    assert len(served) == inst.n and len(set(served)) == inst.n


def test_ges_perturb_preserves_feasibility():
    inst = _load("rc207.txt")
    plan = build_greedy(inst)
    assert plan.feasible
    routes = [r[:] for r in plan.routes]
    served_before = sorted(c for r in routes for c in r)

    random.seed(11)
    _ges_perturb(routes, inst, n_moves=60)

    assert sorted(c for r in routes for c in r) == served_before
    for r in routes:
        if r:
            assert _check_route(r, inst)


def test_ges_penalty_counters_accumulate():
    inst = _load("rc105.txt")
    plan = build_greedy(inst)
    random.seed(5)
    counters: dict[int, int] = {}
    target_idx = min(range(len(plan.routes)), key=lambda i: len(plan.routes[i]))
    _guided_ejection_search(plan, target_idx, p_counters=counters, max_steps=200)
    # Counters only ever grow, and only through the +1 path.
    assert all(v >= 2 for v in counters.values())

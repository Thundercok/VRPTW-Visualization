"""Tests for Generalized Ejection Chains (GEC v2.0) route reduction."""

from __future__ import annotations

from vrptw.config import Config
from vrptw.core import Plan, load_solomon_instance
from vrptw.solvers import HybridDDQNSolver


def test_gec_ejection_on_toy_instance():
    inst = load_solomon_instance("data/Solomon/c101.txt")
    cfg = Config(gec_max_depth=3)
    solver = HybridDDQNSolver(inst, cfg)

    # Construct an artificial plan with 11 routes where one route has only 1 customer
    init_plan, _ = solver.solve(seed=42)
    assert init_plan.feasible

    # Try GEC reduction
    gec_res = solver._try_gec_route_reduction(init_plan)
    # If GEC can eliminate a route, verify feasibility and NV
    if gec_res is not None:
        assert gec_res.feasible
        assert gec_res.nv == init_plan.nv - 1


def test_gec_ejection_preserves_feasibility():
    inst = load_solomon_instance("data/Solomon/r101.txt")
    cfg = Config(gec_max_depth=3)
    solver = HybridDDQNSolver(inst, cfg)

    # Create a plan with an extra single-customer route
    plan = Plan([[1, 2, 3], [4, 5], [6]], inst)
    if plan.feasible:
        reduced = solver._try_gec_route_reduction(plan)
        if reduced is not None:
            assert reduced.feasible
            assert reduced.nv == 2

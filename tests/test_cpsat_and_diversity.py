#!/usr/bin/env python3
"""
Unit and regression tests for:
1. Post-Hoc CP-SAT Exact TSP-TW Refinement.
2. EliteArchive Edge-Jaccard Diversity Enforcement.
"""
import numpy as np

from vrptw.config import Config
from vrptw.core import Inst, Plan, load_solomon_instance
from vrptw.rl import EliteArchive
from vrptw.solvers import ALNSSolver


def test_cpsat_exact_refinement_rc101():
    from vrptw.local_search import refine_plan_cpsat
    inst = load_solomon_instance("data/Solomon/RC101.txt")
    cfg = Config(alns_iterations=200, hybrid_iterations=200)
    solver = ALNSSolver(inst, cfg)
    plan, _ = solver.solve(seed=42)

    assert plan.feasible
    orig_cost = plan.cost
    orig_nv = plan.nv

    refined = refine_plan_cpsat(plan, time_limit_per_route=0.5)
    assert refined.feasible
    assert refined.nv == orig_nv
    assert refined.cost <= orig_cost + 1e-6


def test_elite_archive_diversity_enforcement():
    raw_data = np.zeros((6, 7), dtype=np.float64)
    raw_data[:, 0] = np.arange(6)
    raw_data[0, 1:3] = [50, 50]
    raw_data[1:, 1:3] = [[10, 10], [10, 20], [80, 80], [80, 90], [50, 60]]
    raw_data[1:, 3] = 10.0
    raw_data[:, 4] = 0.0
    raw_data[:, 5] = 1000.0
    raw_data[:, 6] = 5.0
    inst = Inst({"name": "mock_div", "capacity": 50.0, "data": raw_data})

    archive = EliteArchive(k=3)

    # Solution 1
    p1 = Plan([[1, 2], [3, 4, 5]], inst, "P1")
    archive.update(p1)
    assert len(archive._plans["mock_div"]) == 1

    # Solution 1 clone with tiny cost noise
    p1_clone = Plan([[1, 2], [3, 4, 5]], inst, "P1_CLONE")
    archive.update(p1_clone)
    # Should NOT add identical route structure
    assert len(archive._plans["mock_div"]) == 1

    # Solution 2 (structurally distinct)
    p2 = Plan([[1, 5], [2, 3, 4]], inst, "P2")
    archive.update(p2)
    assert len(archive._plans["mock_div"]) == 2

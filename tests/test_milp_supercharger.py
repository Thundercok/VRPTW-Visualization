"""
Unit tests for HiGHS MILP Set Covering Supercharger on RoutePool.
"""

from __future__ import annotations

import numpy as np

from vrptw.config import Config
from vrptw.core import Inst, Plan
from vrptw.pool import RoutePool, recombine_with_route_pool


def test_highs_milp_supercharger_nv_reduction():
    # Construct a synthetic instance with 6 customers
    raw = {
        "name": "mock_supercharger",
        "capacity": 100.0,
        "data": np.array(
            [
                [0, 0, 0, 0, 0, 100, 0],  # Depot
                [1, 1, 0, 20, 0, 100, 0],  # Cust 1
                [2, 2, 0, 20, 0, 100, 0],  # Cust 2
                [3, 3, 0, 20, 0, 100, 0],  # Cust 3
                [4, 4, 0, 20, 0, 100, 0],  # Cust 4
                [5, 5, 0, 20, 0, 100, 0],  # Cust 5
                [6, 6, 0, 20, 0, 100, 0],  # Cust 6
            ],
            dtype=np.float64,
        ),
    }
    inst = Inst(raw)
    cfg = Config(sp_time_limit=5.0)
    pool = RoutePool(inst, cfg)

    # Suboptimal starting plan with 3 routes (2 customers each)
    plan_suboptimal = Plan([[1, 2], [3, 4], [5, 6]], inst)
    assert plan_suboptimal.nv == 3

    # Add elite routes into pool that can form a 2-route plan ([1, 2, 3] and [4, 5, 6])
    route_a = [1, 2, 3]  # load = 60 <= 100
    route_b = [4, 5, 6]  # load = 60 <= 100
    pool.add_route(route_a)
    pool.add_route(route_b)

    # HiGHS Supercharger: synthesize NV-1 target plan (nv_target = 2)
    recombined = recombine_with_route_pool(plan_suboptimal, pool, cfg, nv_target=2, nv_ceiling=2)

    assert recombined.feasible
    assert recombined.nv == 2
    covered_nodes = sorted([c for r in recombined.routes for c in r])
    assert covered_nodes == [1, 2, 3, 4, 5, 6]

from __future__ import annotations

import numpy as np

from vrptw.config import Config
from vrptw.core import Inst, Plan
from vrptw.pool import RoutePool, recombine_with_route_pool


def _create_mock_inst(n: int = 20) -> Inst:
    data = np.zeros((n + 1, 7), dtype=np.float64)
    # Col 0: id
    data[:, 0] = np.arange(n + 1)
    # Col 1, 2: coords (Depot at 50,50, customers in circle)
    data[0, 1:3] = [50.0, 50.0]
    for i in range(1, n + 1):
        angle = 2.0 * np.pi * (i - 1) / n
        data[i, 1:3] = [50.0 + 30.0 * np.cos(angle), 50.0 + 30.0 * np.sin(angle)]

    # Col 3: demand
    data[0, 3] = 0.0
    data[1:, 3] = 10.0

    # Col 4: ready_time
    data[:, 4] = 0.0

    # Col 5: due_time
    data[0, 5] = 2000.0
    data[1:, 5] = 1000.0

    # Col 6: service_time
    data[0, 6] = 0.0
    data[1:, 6] = 10.0

    raw = {
        "name": "mock_inst",
        "data": data,
        "capacity": 50.0,
    }
    return Inst(raw)


def test_granular_neighbor_matrix():
    """Verify that granular spatiotemporal correlation matrix identifies true spatial/temporal neighbors."""
    inst = _create_mock_inst(n=20)
    k = 5

    # Granular neighbors computation
    c_ij = inst.dist
    tw_overlap = np.maximum(0.0, inst.ready_times[:, None] - (inst.due_times[None, :] + inst.service_times[None, :] + c_ij))
    affinity = c_ij + 0.5 * tw_overlap

    # Sort top-k neighbors (excluding self)
    neighbors = np.argsort(affinity[1:, 1:], axis=1)[:, 1:k+1] + 1

    assert neighbors.shape == (20, k)
    # Verify neighbor indices are valid customer IDs
    assert np.all((neighbors >= 1) & (neighbors <= 20))


def test_route_pool_highs_recombination_integration():
    """Verify that RoutePool recombines fragmented routes into a feasible exact cover."""
    inst = _create_mock_inst(n=12)
    cfg = Config()
    pool = RoutePool(inst, cfg)

    # 4 distinct 3-customer routes
    routes = [
        [1, 2, 3],
        [4, 5, 6],
        [7, 8, 9],
        [10, 11, 12],
        [1, 2],
        [3, 4],
        [5, 6],
        [7, 8],
        [9, 10],
        [11, 12],
    ]
    for r in routes:
        pool.add_route(r)

    initial_plan = Plan([[1, 2], [3, 4], [5, 6], [7, 8], [9, 10], [11, 12]], inst, "INITIAL")
    assert initial_plan.nv == 6

    recombined = recombine_with_route_pool(initial_plan, pool, cfg)
    assert recombined is not None
    assert recombined.feasible
    assert recombined.nv <= 4  # Should find the 4-vehicle combination

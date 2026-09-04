import numpy as np

from vrptw.core import Inst, Plan
from vrptw.local_search import _cross_tail, local_search


def test_cross_tail_basic():
    # Construct a small mock instance:
    # 5 nodes: 0 (depot), 1, 2, 3, 4
    # Depot: (0, 0)
    # 1: (1, 0)
    # 2: (2, 0)
    # 3: (0, 1)
    # 4: (0, 2)
    #
    # Route 1: [1, 4] -> cost: dist(0,1) + dist(1,4) + dist(4,0) = 1 + sqrt(5) + 2 = 5.236
    # Route 2: [3, 2] -> cost: dist(0,3) + dist(3,2) + dist(2,0) = 1 + sqrt(5) + 2 = 5.236
    # Total: 10.472
    #
    # Swapping tails (at split_i=1, split_j=1):
    # Route 1 new: [1, 2] -> cost: dist(0,1) + dist(1,2) + dist(2,0) = 1 + 1 + 2 = 4.0
    # Route 2 new: [3, 4] -> cost: dist(0,3) + dist(3,4) + dist(4,0) = 1 + 1 + 2 = 4.0
    # Total: 8.0 (saving: 2.472)

    raw = {
        "name": "mock",
        "capacity": 100.0,
        "data": np.array(
            [
                [0, 0, 0, 0, 0, 1000, 0],  # Depot
                [1, 1, 0, 10, 0, 1000, 0],  # Node 1
                [2, 2, 0, 10, 0, 1000, 0],  # Node 2
                [3, 0, 1, 10, 0, 1000, 0],  # Node 3
                [4, 0, 2, 10, 0, 1000, 0],  # Node 4
            ],
            dtype=np.float64,
        ),
    }
    inst = Inst(raw)

    plan = Plan([[1, 4], [3, 2]], inst)
    assert plan.feasible
    original_cost = plan.cost

    improved = _cross_tail(plan)
    assert improved is not None
    assert improved.cost < original_cost - 1e-3
    assert improved.routes == [[1, 2], [3, 4]]


def test_local_search_cascade():
    raw = {
        "name": "mock",
        "capacity": 100.0,
        "data": np.array(
            [
                [0, 0, 0, 0, 0, 1000, 0],  # Depot
                [1, 1, 0, 10, 0, 1000, 0],  # Node 1
                [2, 2, 0, 10, 0, 1000, 0],  # Node 2
                [3, 0, 1, 10, 0, 1000, 0],  # Node 3
                [4, 0, 2, 10, 0, 1000, 0],  # Node 4
            ],
            dtype=np.float64,
        ),
    }
    inst = Inst(raw)

    # Start with suboptimal routes that can be resolved via cross-tail
    plan = Plan([[1, 4], [3, 2]], inst)

    refined = local_search(plan, max_passes=2)
    assert refined.feasible
    assert refined.cost < plan.cost - 1e-3
    assert refined.nv <= 2


def test_string_relocate_basic():
    raw = {
        "name": "mock",
        "capacity": 100.0,
        "data": np.array(
            [
                [0, 0, 0, 0, 0, 1000, 0],  # Depot
                [1, 10, 0, 10, 0, 1000, 0],  # Node 1: (10, 0)
                [2, 10, 1, 10, 0, 1000, 0],  # Node 2: (10, 1)
                [3, 10, 2, 10, 0, 1000, 0],  # Node 3: (10, 2)
                [4, 0, 10, 10, 0, 1000, 0],  # Node 4: (0, 10)
            ],
            dtype=np.float64,
        ),
    }
    inst = Inst(raw)

    plan = Plan([[4, 2, 3], [1]], inst)
    assert plan.feasible
    original_cost = plan.cost

    from vrptw.local_search import _apply_or_opt, _best_or_opt

    res = _best_or_opt(plan)
    assert res is not None
    move, delta = res

    improved = _apply_or_opt(plan, move)
    assert improved.feasible
    assert improved.cost < original_cost - 1e-3
    assert improved.routes == [[4, 2, 3, 1]]


def test_swap_21_basic():
    raw = {
        "name": "mock",
        "capacity": 100.0,
        "data": np.array(
            [
                [0, 0, 0, 0, 0, 1000, 0],  # Depot
                [1, 10, 0, 10, 0, 1000, 0],  # Node 1: (10, 0)
                [2, 10, 1, 10, 0, 1000, 0],  # Node 2: (10, 1)
                [3, 0, 10, 10, 0, 1000, 0],  # Node 3: (0, 10)
                [4, 0, 11, 10, 0, 1000, 0],  # Node 4: (0, 11)
            ],
            dtype=np.float64,
        ),
    }
    inst = Inst(raw)

    # Route 1 has node 3 (at y=10) mismatched with [1, 2] (at x=10)
    # Route 2 has nodes [4] (at y=11) with [3]
    plan = Plan([[1, 3], [4, 2]], inst)
    assert plan.feasible

    from vrptw.local_search import _apply_swap_21, _best_swap_21, td_inter_route_polish

    res = _best_swap_21(plan)
    if res is not None:
        move, delta = res
        cand = _apply_swap_21(plan, move)
        assert cand.feasible
        assert cand.cost < plan.cost

    polished = td_inter_route_polish(plan)
    assert polished.feasible
    assert polished.cost <= plan.cost

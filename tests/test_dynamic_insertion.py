"""
Unit tests and throughput benchmarks for Pillar 3: Online Dynamic Customer Insertion.
"""

from __future__ import annotations

import os

import pytest

from vrptw.core import Inst, Plan, load_solomon_instance
from vrptw.dynamic_insertion import (
    DynamicCustomerInserter,
    insert_dynamic_batch,
)
from vrptw.heuristics import build_greedy


@pytest.fixture
def r101_inst() -> Inst:
    repo_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    path = os.path.join(repo_dir, "data", "Solomon", "r101.txt")
    if not os.path.exists(path):
        pytest.skip(f"R101 instance not found at {path}")
    return load_solomon_instance(path)


@pytest.fixture
def c101_inst() -> Inst:
    repo_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    path = os.path.join(repo_dir, "data", "Solomon", "c101.txt")
    if not os.path.exists(path):
        pytest.skip(f"C101 instance not found at {path}")
    return load_solomon_instance(path)


def test_dynamic_insertion_single_customer_feasible(r101_inst: Inst) -> None:
    # Build a partial initial plan with subset of customers (e.g. 1..50)
    initial_plan = build_greedy(r101_inst, "ALNS-Base")
    assert initial_plan.feasible

    # Pick an unvisited customer or remove one from plan to re-insert
    removed_node = initial_plan.routes[0].pop(1)
    initial_plan.invalidate()
    assert initial_plan.feasible

    inserter = DynamicCustomerInserter(r101_inst)
    result = inserter.insert(initial_plan, removed_node)

    assert result.success is True
    assert result.updated_plan is not None
    assert result.updated_plan.feasible is True
    assert removed_node in result.updated_plan.routes[result.route_idx]
    assert result.latency_ms > 0.0
    assert result.latency_ms < 5.0  # Well under 5ms, typically < 0.5ms


def test_dynamic_insertion_cost_delta_matches_brute_force(c101_inst: Inst) -> None:
    """Validate cost delta against independent brute-force route & plan evaluation."""
    initial_plan = build_greedy(c101_inst, "ALNS-Base")
    inserter = DynamicCustomerInserter(c101_inst)

    for r_idx in range(min(3, len(initial_plan.routes))):
        if len(initial_plan.routes[r_idx]) < 2:
            continue
        test_plan = initial_plan.copy()
        removed_node = test_plan.routes[r_idx].pop(1)
        test_plan.invalidate()

        old_route = list(test_plan.routes[r_idx])
        old_plan_cost = test_plan.cost

        result = inserter.insert(test_plan, removed_node)
        assert result.success is True
        assert result.updated_plan.feasible is True

        # Ground truth 1: Full plan cost difference
        new_plan_cost = result.updated_plan.cost
        assert abs((new_plan_cost - old_plan_cost) - result.cost_delta) < 1e-4

        # Ground truth 2: Inserted route cost difference calculated purely from raw distance matrix
        new_route = result.updated_plan.routes[result.route_idx]

        def _raw_route_dist(r: list[int]) -> float:
            if not r:
                return 0.0
            chain = [0] + r + [0]
            return sum(c101_inst.dist[chain[i], chain[i + 1]] for i in range(len(chain) - 1))

        expected_route_delta = _raw_route_dist(new_route) - _raw_route_dist(old_route)
        if result.route_idx == r_idx:
            assert abs(expected_route_delta - result.cost_delta) < 1e-4


def test_dynamic_insertion_already_served_customer(r101_inst: Inst) -> None:
    plan = build_greedy(r101_inst, "ALNS-Base")
    served_node = plan.routes[0][0]

    inserter = DynamicCustomerInserter(r101_inst)
    result = inserter.insert(plan, served_node)

    assert result.success is False
    assert "already served" in result.message.lower()


def test_dynamic_insertion_invalid_customer_id(r101_inst: Inst) -> None:
    plan = build_greedy(r101_inst, "ALNS-Base")
    inserter = DynamicCustomerInserter(r101_inst)

    res_neg = inserter.insert(plan, -1)
    assert res_neg.success is False
    assert "invalid" in res_neg.message.lower()

    res_overflow = inserter.insert(plan, r101_inst.n + 99)
    assert res_overflow.success is False
    assert "invalid" in res_overflow.message.lower()


def test_dynamic_insertion_vehicle_limit_enforcement(r101_inst: Inst) -> None:
    # Empty plan with max_vehicles = 1
    empty_plan = Plan([], r101_inst)
    inserter = DynamicCustomerInserter(r101_inst, max_vehicles=1)

    # Insert 1st customer -> creates 1st route
    res1 = inserter.insert(empty_plan, 1)
    assert res1.success is True
    assert res1.is_new_route is True

    # Attempt to insert customer with allow_new_route=False when it cannot fit
    # or when vehicle limit is saturated
    saturated_plan = Plan([[1]], r101_inst)
    res_disallowed = inserter.insert(saturated_plan, 2, allow_new_route=False)
    # If customer 2 cannot fit into route [1], it must fail because new route is disabled
    if not res_disallowed.success:
        assert "cannot be feasibly inserted" in res_disallowed.message.lower()


def test_evaluate_candidates_ranking(r101_inst: Inst) -> None:
    plan = build_greedy(r101_inst, "ALNS-Base")
    node = plan.routes[0].pop(0)
    plan.invalidate()

    inserter = DynamicCustomerInserter(r101_inst)
    candidates = inserter.evaluate_candidates(plan, node)

    assert len(candidates) > 0
    # Verify sorted in ascending order of cost_delta
    deltas = [c.cost_delta for c in candidates]
    assert deltas == sorted(deltas)


def test_insert_batch_dynamic(c101_inst: Inst) -> None:
    plan = Plan([], c101_inst)
    batch_nodes = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

    updated_plan, results = insert_dynamic_batch(plan, batch_nodes, c101_inst)

    assert len(results) == len(batch_nodes)
    assert all(r.success for r in results)
    assert updated_plan.feasible is True

    # Verify all batch nodes are in the final plan
    all_served = {node for r in updated_plan.routes for node in r}
    assert set(batch_nodes).issubset(all_served)


def test_dynamic_insertion_submillisecond_benchmark(r101_inst: Inst) -> None:
    plan = build_greedy(r101_inst, "ALNS-Base")
    node = plan.routes[0].pop(0)
    plan.invalidate()

    inserter = DynamicCustomerInserter(r101_inst)
    bench = inserter.benchmark_throughput(plan, customer_pool=[node], n_trials=500)

    # Sub-millisecond performance assertions
    assert bench["n_trials"] == 500
    assert bench["mean_ms"] < 1.0, f"Mean latency {bench['mean_ms']:.4f}ms exceeds 1.0ms SLA target"
    assert bench["p95_ms"] < 2.0, f"P95 latency {bench['p95_ms']:.4f}ms exceeds 2.0ms target"
    assert bench["ops_per_sec"] > 1000.0, f"Throughput {bench['ops_per_sec']:.1f} ops/s below 1000 target"

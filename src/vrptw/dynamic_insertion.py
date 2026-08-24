"""
Pillar 3: Online Dynamic Customer Insertion Engine.

High-throughput, sub-millisecond dynamic insertion of incoming customers
into an active dispatch Plan using Savelsbergh O(1) Forward Time Slack and
push-forward timing profiles.
"""

from __future__ import annotations

import time
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from .core import Inst, Plan, _check_route
from .heuristics import (
    _NO_HEATMAP,
    _best_insert_over_routes_numba,
    _insert_feasible_numba,
    _route_cost_list,
    _route_timing_numba,
    pack_routes,
)


@dataclass(frozen=True)
class CandidateInsertion:
    """Evaluated insertion position in an active route."""
    route_idx: int
    insert_pos: int
    cost_delta: float
    wait_delta: float = 0.0
    composite_cost: float = 0.0


@dataclass(frozen=True)
class DynamicInsertionResult:
    """Result of an online dynamic customer insertion request."""
    customer_id: int
    success: bool
    route_idx: int | None
    insert_pos: int | None
    cost_delta: float
    is_new_route: bool
    latency_ms: float
    updated_plan: Plan | None
    message: str = ""


class DynamicCustomerInserter:
    """
    Real-time online customer insertion engine.

    Targets < 1ms latency per single insertion by leveraging precomputed
    distance structures and compiled Numba kernels.
    """

    def __init__(
        self,
        inst: Inst,
        max_vehicles: int | None = None,
        tw_wait_weight: float = 0.20,
    ) -> None:
        self.inst = inst
        self.max_vehicles = max_vehicles
        self.tw_wait_weight = tw_wait_weight

    def evaluate_candidates(
        self,
        plan: Plan,
        customer_id: int,
    ) -> list[CandidateInsertion]:
        """
        Evaluate all feasible insertion points for customer_id across active routes.
        Returns candidates ranked from cheapest to most expensive cost delta.
        """
        inst = self.inst
        node = int(customer_id)
        if node < 1 or node > inst.n:
            return []

        candidates: list[CandidateInsertion] = []
        node_demand = inst.demands[node]

        for r_idx, route in enumerate(plan.routes):
            route_load = sum(inst.demands[c] for c in route)
            if route_load + node_demand > inst.capacity:
                continue

            route_arr = np.array(route, dtype=np.int64)
            n_nodes = len(route_arr)
            arrivals, latest, first_violation = _route_timing_numba(
                route_arr, inst.dist, inst.ready_times, inst.due_times, inst.service_times
            )

            for pos in range(n_nodes + 1):
                if pos > first_violation:
                    break

                if _insert_feasible_numba(
                    node, pos, route_arr, arrivals, latest,
                    inst.dist, inst.ready_times, inst.due_times, inst.service_times
                ):
                    prev = route_arr[pos - 1] if pos > 0 else 0
                    nxt = route_arr[pos] if pos < n_nodes else 0
                    delta_dist = (
                        inst.dist[prev, node]
                        + inst.dist[node, nxt]
                        - inst.dist[prev, nxt]
                    )

                    t_prev_depart = (arrivals[pos - 1] + inst.service_times[prev]) if pos > 0 else 0.0
                    t_arrive_node = t_prev_depart + inst.dist[prev, node]
                    wait_added = max(0.0, inst.ready_times[node] - t_arrive_node)
                    composite = delta_dist + self.tw_wait_weight * wait_added

                    candidates.append(
                        CandidateInsertion(
                            route_idx=r_idx,
                            insert_pos=pos,
                            cost_delta=float(delta_dist),
                            wait_delta=float(wait_added),
                            composite_cost=float(composite),
                        )
                    )

        candidates.sort(key=lambda c: (c.cost_delta, c.wait_delta))
        return candidates

    def insert(
        self,
        plan: Plan,
        customer_id: int,
        allow_new_route: bool = True,
    ) -> DynamicInsertionResult:
        """
        Fast dynamic insertion of a customer into an active plan with latency measurement.
        """
        t0 = time.perf_counter()
        node = int(customer_id)
        inst = self.inst

        if node < 1 or node > inst.n:
            t1 = time.perf_counter()
            return DynamicInsertionResult(
                customer_id=node,
                success=False,
                route_idx=None,
                insert_pos=None,
                cost_delta=float("inf"),
                is_new_route=False,
                latency_ms=(t1 - t0) * 1000.0,
                updated_plan=None,
                message=f"Invalid customer ID {node}. Must be in [1, {inst.n}].",
            )

        # Check if customer is already present in any route
        for r_idx, r in enumerate(plan.routes):
            if node in r:
                t1 = time.perf_counter()
                return DynamicInsertionResult(
                    customer_id=node,
                    success=False,
                    route_idx=r_idx,
                    insert_pos=r.index(node),
                    cost_delta=0.0,
                    is_new_route=False,
                    latency_ms=(t1 - t0) * 1000.0,
                    updated_plan=plan,
                    message=f"Customer {node} is already served in route {r_idx}.",
                )

        # Single-call vectorized Numba scan over all routes
        if plan.routes:
            routes_flat, route_lens, route_loads = pack_routes(plan.routes, inst)
            delta, best_route, best_pos = _best_insert_over_routes_numba(
                node,
                routes_flat,
                route_lens,
                route_loads,
                inst.dist,
                inst.demands,
                inst.capacity,
                inst.ready_times,
                inst.due_times,
                inst.service_times,
                _NO_HEATMAP,
                0.0,
                False,
            )
        else:
            delta = float("inf")
            best_route = -1
            best_pos = -1

        new_plan = plan.copy()

        if best_route >= 0:
            new_plan.routes[best_route].insert(best_pos, node)
            new_plan.invalidate()
            t1 = time.perf_counter()
            return DynamicInsertionResult(
                customer_id=node,
                success=True,
                route_idx=int(best_route),
                insert_pos=int(best_pos),
                cost_delta=float(delta),
                is_new_route=False,
                latency_ms=(t1 - t0) * 1000.0,
                updated_plan=new_plan,
                message=f"Customer {node} inserted into route {best_route} at position {best_pos}.",
            )

        # Customer does not fit into any existing route -> evaluate opening a new route
        if allow_new_route:
            if self.max_vehicles is not None and len(plan.routes) >= self.max_vehicles:
                t1 = time.perf_counter()
                return DynamicInsertionResult(
                    customer_id=node,
                    success=False,
                    route_idx=None,
                    insert_pos=None,
                    cost_delta=float("inf"),
                    is_new_route=False,
                    latency_ms=(t1 - t0) * 1000.0,
                    updated_plan=None,
                    message=f"Vehicle limit ({self.max_vehicles}) reached. Cannot dispatch new vehicle.",
                )

            # Check if singleton route [node] is feasible
            if _check_route([node], inst):
                singleton_cost = _route_cost_list([node], inst)
                new_plan.routes.append([node])
                new_plan.invalidate()
                t1 = time.perf_counter()
                return DynamicInsertionResult(
                    customer_id=node,
                    success=True,
                    route_idx=len(new_plan.routes) - 1,
                    insert_pos=0,
                    cost_delta=float(singleton_cost),
                    is_new_route=True,
                    latency_ms=(t1 - t0) * 1000.0,
                    updated_plan=new_plan,
                    message=f"Customer {node} dispatched on new vehicle route.",
                )

        t1 = time.perf_counter()
        return DynamicInsertionResult(
            customer_id=node,
            success=False,
            route_idx=None,
            insert_pos=None,
            cost_delta=float("inf"),
            is_new_route=False,
            latency_ms=(t1 - t0) * 1000.0,
            updated_plan=None,
            message=f"Customer {node} cannot be feasibly inserted into any existing route or singleton.",
        )

    def insert_batch(
        self,
        plan: Plan,
        customer_ids: Sequence[int],
        allow_new_route: bool = True,
    ) -> tuple[Plan, list[DynamicInsertionResult]]:
        """
        Dynamically insert a sequence of customers one by one.
        """
        current_plan = plan.copy()
        results: list[DynamicInsertionResult] = []

        for cid in customer_ids:
            res = self.insert(current_plan, cid, allow_new_route=allow_new_route)
            results.append(res)
            if res.success and res.updated_plan is not None:
                current_plan = res.updated_plan

        return current_plan, results

    def benchmark_throughput(
        self,
        plan: Plan,
        customer_pool: Sequence[int] | None = None,
        n_trials: int = 1000,
    ) -> dict[str, float]:
        """
        Benchmark throughput and latency distribution (mean, p50, p95, p99) over n_trials.
        """
        if customer_pool is None:
            # Pick valid customers from instance
            customer_pool = list(range(1, min(self.inst.n + 1, 20)))

        latencies_ms: list[float] = []

        # Warm-up JIT compilation
        _ = self.insert(plan, customer_pool[0])

        for i in range(n_trials):
            node = customer_pool[i % len(customer_pool)]
            res = self.insert(plan, node)
            latencies_ms.append(res.latency_ms)

        arr = np.array(latencies_ms)
        total_time_s = float(np.sum(arr)) / 1000.0
        ops_per_sec = n_trials / max(total_time_s, 1e-9)

        return {
            "n_trials": float(n_trials),
            "mean_ms": float(np.mean(arr)),
            "p50_ms": float(np.percentile(arr, 50)),
            "p95_ms": float(np.percentile(arr, 95)),
            "p99_ms": float(np.percentile(arr, 99)),
            "min_ms": float(np.min(arr)),
            "max_ms": float(np.max(arr)),
            "ops_per_sec": float(ops_per_sec),
        }


def insert_dynamic_customer(
    plan: Plan,
    customer_id: int,
    inst: Inst,
    max_vehicles: int | None = None,
    allow_new_route: bool = True,
) -> DynamicInsertionResult:
    """Convenience function for dynamic customer insertion."""
    inserter = DynamicCustomerInserter(inst, max_vehicles=max_vehicles)
    return inserter.insert(plan, customer_id, allow_new_route=allow_new_route)


def insert_dynamic_batch(
    plan: Plan,
    customer_ids: Sequence[int],
    inst: Inst,
    max_vehicles: int | None = None,
    allow_new_route: bool = True,
) -> tuple[Plan, list[DynamicInsertionResult]]:
    """Convenience function for dynamic batch customer insertion."""
    inserter = DynamicCustomerInserter(inst, max_vehicles=max_vehicles)
    return inserter.insert_batch(plan, customer_ids, allow_new_route=allow_new_route)

from __future__ import annotations

import numpy as np
from numba import njit

from .core import Inst, Plan, _route_cost


@njit(cache=True)
def _route_timing_numba(
    route: np.ndarray,
    dist: np.ndarray,
    ready: np.ndarray,
    due: np.ndarray,
    service: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Precompute the timing profile of an existing route.

    Returns ``(arrivals, latest, first_violation)`` where

    * ``arrivals[i]`` is the service start time at ``route[i]`` (after any wait),
    * ``latest[i]`` is the *latest* service start at ``route[i]`` for which the
      remainder of the route — including the return leg to the depot — stays
      feasible,
    * ``first_violation`` is the index of the first node whose service start
      already exceeds its due date (``len(route)`` when the route is feasible).

    ``latest`` is the exact feasibility threshold, not a conservative bound:
    since ``ready[i+1] <= latest[i+1]`` whenever the suffix is feasible at all,
    ``max(arrival, ready[i+1]) <= latest[i+1]`` reduces to ``arrival <=
    latest[i+1]``, so waiting is handled correctly by the recursion. This is what
    lets the insertion feasibility test below run in O(1) per position rather
    than re-simulating the whole route.
    """
    m = len(route)
    arrivals = np.empty(m, dtype=np.float64)
    latest = np.empty(m, dtype=np.float64)
    first_violation = m

    t = 0.0
    prev = 0
    for i in range(m):
        node = route[i]
        t += dist[prev, node]
        if t < ready[node]:
            t = ready[node]
        arrivals[i] = t
        if t > due[node] and first_violation == m:
            first_violation = i
        t += service[node]
        prev = node

    if m > 0:
        last = route[m - 1]
        latest[m - 1] = min(due[last], due[0] - service[last] - dist[last, 0])
        for i in range(m - 2, -1, -1):
            node = route[i]
            nxt = route[i + 1]
            cand = latest[i + 1] - service[node] - dist[node, nxt]
            latest[i] = min(due[node], cand)

    return arrivals, latest, first_violation


@njit(cache=True, fastmath=True)
def compute_forward_slack(
    tw_early: np.ndarray,
    tw_late: np.ndarray,
    travel: np.ndarray,
    service: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Savelsbergh (1992) Forward Time Slack calculation.

    F_i is the maximum delay that can be absorbed at node i without violating any
    downstream time window in the route suffix.
    """
    n = tw_early.shape[0]
    b = np.empty(n, dtype=np.float64)
    wait = np.zeros(n, dtype=np.float64)
    b[0] = tw_early[0]
    for i in range(1, n):
        raw = b[i - 1] + service[i - 1] + travel[i - 1]
        b[i] = max(raw, tw_early[i])
        wait[i] = b[i] - raw
    F = np.empty(n, dtype=np.float64)
    F[n - 1] = tw_late[n - 1] - b[n - 1]
    for i in range(n - 2, -1, -1):
        F[i] = min(wait[i + 1] + F[i + 1], tw_late[i] - b[i])
    return b, F


@njit(cache=True)
def move_feasible(F: np.ndarray, insert_pos: int, delta_time: float) -> bool:
    """O(1) feasibility check via Forward Time Slack."""
    return delta_time <= F[insert_pos]


@njit(cache=True)
def _insert_feasible_numba(
    node: int,
    pos: int,
    route: np.ndarray,
    arrivals: np.ndarray,
    latest: np.ndarray,
    dist: np.ndarray,
    ready: np.ndarray,
    due: np.ndarray,
    service: np.ndarray,
) -> bool:
    """O(1) push-forward feasibility test for inserting ``node`` at ``pos``.

    The caller is responsible for the capacity check and for ensuring the prefix
    ``route[:pos]`` is itself feasible (``pos <= first_violation``).
    """
    n_nodes = len(route)
    prev = route[pos - 1] if pos > 0 else 0
    t_prev_depart = (arrivals[pos - 1] + service[prev]) if pos > 0 else 0.0

    t_node = t_prev_depart + dist[prev, node]
    if t_node < ready[node]:
        t_node = ready[node]
    if t_node > due[node]:
        return False
    t_depart = t_node + service[node]

    if pos == n_nodes:
        return t_depart + dist[node, 0] <= due[0]

    nxt = route[pos]
    t_nxt = t_depart + dist[node, nxt]
    if t_nxt < ready[nxt]:
        t_nxt = ready[nxt]
    return t_nxt <= latest[pos]


@njit(cache=True)
def _best_insert_position_numba(
    node: int,
    route: np.ndarray,
    dist: np.ndarray,
    demands: np.ndarray,
    capacity: float,
    ready: np.ndarray,
    due: np.ndarray,
    service: np.ndarray,
) -> tuple[float, int]:
    best_cost = 1e18
    best_pos = -1

    n_nodes = len(route)
    current_load = 0.0
    for idx in range(n_nodes):
        current_load += demands[route[idx]]
    if current_load + demands[node] > capacity:
        return 1e18, -1

    arrivals, latest, first_violation = _route_timing_numba(route, dist, ready, due, service)

    for pos in range(n_nodes + 1):
        if pos > first_violation:
            break
        prev = route[pos - 1] if pos > 0 else 0
        nxt = route[pos] if pos < n_nodes else 0
        delta = dist[prev, node] + dist[node, nxt] - dist[prev, nxt]
        if delta >= best_cost:
            continue

        if _insert_feasible_numba(node, pos, route, arrivals, latest, dist, ready, due, service):
            best_cost = delta
            best_pos = pos

    return best_cost, best_pos


@njit(cache=True)
def _best_insert_in_route_numba(
    node: int,
    route: np.ndarray,
    route_load: float,
    arrivals: np.ndarray,
    latest: np.ndarray,
    first_violation: int,
    dist: np.ndarray,
    demands: np.ndarray,
    capacity: float,
    ready: np.ndarray,
    due: np.ndarray,
    service: np.ndarray,
    heatmap: np.ndarray,
    gamma: float,
    use_bias: bool,
) -> tuple[float, float, int]:
    """Best insertion of ``node`` into one route whose timing profile is already known.

    Separating the timing precomputation from the position scan is what makes the
    batched kernels below cheap: the O(m) profile is built once per route and
    reused across every node being inserted, instead of once per (node, route).
    """
    if route_load + demands[node] > capacity:
        return 1e18, 1e18, -1

    n_nodes = len(route)
    best_key = 1e18
    best_delta = 1e18
    best_pos = -1

    for pos in range(n_nodes + 1):
        if pos > first_violation:
            break
        prev = route[pos - 1] if pos > 0 else 0
        nxt = route[pos] if pos < n_nodes else 0
        delta = dist[prev, node] + dist[node, nxt] - dist[prev, nxt]

        if use_bias:
            key = delta * (1.0 - gamma * heatmap[prev, node]) * (1.0 - gamma * heatmap[node, nxt])
        else:
            key = delta

        if key >= best_key:
            continue

        if _insert_feasible_numba(node, pos, route, arrivals, latest, dist, ready, due, service):
            best_key = key
            best_delta = delta
            best_pos = pos

    return best_key, best_delta, best_pos


@njit(cache=True)
def _best_insert_in_route_pruned_numba(
    node: int,
    route: np.ndarray,
    route_load: float,
    arrivals: np.ndarray,
    latest: np.ndarray,
    first_violation: int,
    dist: np.ndarray,
    demands: np.ndarray,
    capacity: float,
    ready: np.ndarray,
    due: np.ndarray,
    service: np.ndarray,
    heatmap: np.ndarray,
    pruning_threshold: float,
) -> tuple[float, int]:
    """Heatmap-pruned counterpart of :func:`_best_insert_in_route_numba`."""
    if route_load + demands[node] > capacity:
        return 1e18, -1

    n_nodes = len(route)
    best_cost = 1e18
    best_pos = -1

    for pos in range(n_nodes + 1):
        if pos > first_violation:
            break
        prev = route[pos - 1] if pos > 0 else 0
        nxt = route[pos] if pos < n_nodes else 0

        if heatmap[prev, node] < pruning_threshold or heatmap[node, nxt] < pruning_threshold:
            continue

        delta = dist[prev, node] + dist[node, nxt] - dist[prev, nxt]
        if delta >= best_cost:
            continue

        if _insert_feasible_numba(node, pos, route, arrivals, latest, dist, ready, due, service):
            best_cost = delta
            best_pos = pos

    return best_cost, best_pos


@njit(cache=True)
def _insert_costs_matrix_numba(
    nodes: np.ndarray,
    routes_flat: np.ndarray,
    route_lens: np.ndarray,
    route_loads: np.ndarray,
    dist: np.ndarray,
    demands: np.ndarray,
    capacity: float,
    ready: np.ndarray,
    due: np.ndarray,
    service: np.ndarray,
    heatmap: np.ndarray,
    gamma: float,
    use_bias: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Best-insertion cost of every ``node`` into every route, in one call.

    Returns ``(keys, deltas, positions)`` each shaped ``(len(nodes), n_routes)``;
    ``positions[i, r] == -1`` marks an infeasible pair. ``keys`` is the (possibly
    heatmap-biased) selection criterion, ``deltas`` the true distance increase.

    Every route is evaluated. A kNN candidate-route filter was tried here and
    removed: it cost 1.15pp of gap-to-BKS for no speedup, because after the
    batching above this scan is no longer the bottleneck (9% of runtime at n=400).
    """
    n = len(nodes)
    r_count = len(route_lens)
    keys = np.full((n, r_count), 1e18, dtype=np.float64)
    deltas = np.full((n, r_count), 1e18, dtype=np.float64)
    positions = np.full((n, r_count), -1, dtype=np.int64)

    for r in range(r_count):
        route = routes_flat[r, : route_lens[r]]
        arrivals, latest, first_violation = _route_timing_numba(route, dist, ready, due, service)
        for i in range(n):
            k, d, p = _best_insert_in_route_numba(
                nodes[i],
                route,
                route_loads[r],
                arrivals,
                latest,
                first_violation,
                dist,
                demands,
                capacity,
                ready,
                due,
                service,
                heatmap,
                gamma,
                use_bias,
            )
            keys[i, r] = k
            deltas[i, r] = d
            positions[i, r] = p

    return keys, deltas, positions


@njit(cache=True)
def _insert_costs_column_numba(
    nodes: np.ndarray,
    route: np.ndarray,
    route_load: float,
    dist: np.ndarray,
    demands: np.ndarray,
    capacity: float,
    ready: np.ndarray,
    due: np.ndarray,
    service: np.ndarray,
    heatmap: np.ndarray,
    gamma: float,
    use_bias: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Best-insertion cost of every ``node`` into a single route.

    Used to refresh only the column invalidated by an insertion, instead of
    rebuilding the whole matrix each round.
    """
    n = len(nodes)
    keys = np.full(n, 1e18, dtype=np.float64)
    deltas = np.full(n, 1e18, dtype=np.float64)
    positions = np.full(n, -1, dtype=np.int64)

    arrivals, latest, first_violation = _route_timing_numba(route, dist, ready, due, service)
    for i in range(n):
        k, d, p = _best_insert_in_route_numba(
            nodes[i],
            route,
            route_load,
            arrivals,
            latest,
            first_violation,
            dist,
            demands,
            capacity,
            ready,
            due,
            service,
            heatmap,
            gamma,
            use_bias,
        )
        keys[i] = k
        deltas[i] = d
        positions[i] = p

    return keys, deltas, positions


@njit(cache=True)
def _best_insert_over_routes_numba(
    node: int,
    routes_flat: np.ndarray,
    route_lens: np.ndarray,
    route_loads: np.ndarray,
    dist: np.ndarray,
    demands: np.ndarray,
    capacity: float,
    ready: np.ndarray,
    due: np.ndarray,
    service: np.ndarray,
    heatmap: np.ndarray,
    gamma: float,
    use_bias: bool,
) -> tuple[float, int, int]:
    """Cheapest insertion of a single node across every route, in one call.

    Returns ``(delta, route_index, pos)`` with ``route_index == -1`` when the node
    fits nowhere. Ties resolve to the lowest route index, matching the strict
    ``<`` comparison of the Python loop this replaces.
    """
    best_key = 1e18
    best_delta = 1e18
    best_ri = -1
    best_pos = -1

    for r in range(len(route_lens)):
        route = routes_flat[r, : route_lens[r]]
        arrivals, latest, first_violation = _route_timing_numba(route, dist, ready, due, service)
        key, delta, pos = _best_insert_in_route_numba(
            node,
            route,
            route_loads[r],
            arrivals,
            latest,
            first_violation,
            dist,
            demands,
            capacity,
            ready,
            due,
            service,
            heatmap,
            gamma,
            use_bias,
        )
        if pos >= 0 and key < best_key:
            best_key = key
            best_delta = delta
            best_ri = r
            best_pos = pos

    return best_delta, best_ri, best_pos


_NO_HEATMAP = np.zeros((1, 1), dtype=np.float32)


def pack_routes(routes: list[list[int]], inst: Inst) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Pack routes into the padded (routes_flat, route_lens, route_loads) form the
    batched insertion kernels expect."""
    r_count = len(routes)
    max_len = max((len(r) for r in routes), default=0)
    routes_flat = np.zeros((r_count, max(max_len, 1)), dtype=np.int64)
    route_lens = np.empty(r_count, dtype=np.int64)
    route_loads = np.empty(r_count, dtype=np.float64)
    for i, route in enumerate(routes):
        route_lens[i] = len(route)
        if route:
            routes_flat[i, : len(route)] = route
            route_loads[i] = float(inst.demands[routes_flat[i, : len(route)]].sum())
        else:
            route_loads[i] = 0.0
    return routes_flat, route_lens, route_loads


def _best_insert_position(node: int, route: list[int], inst: Inst) -> tuple[float, int | None]:
    # No Python-side capacity pre-check: the kernel recomputes the load itself and
    # returns (1e18, -1) in exactly that case. The pre-check cost 19% of total
    # runtime at n=400 purely to duplicate work the kernel already does.
    route_arr = np.array(route, dtype=np.int64)
    best_cost, best_pos = _best_insert_position_numba(
        node,
        route_arr,
        inst.dist,
        inst.demands,
        inst.capacity,
        inst.ready_times,
        inst.due_times,
        inst.service_times,
    )
    if best_pos == -1:
        return float("inf"), None
    return float(best_cost), int(best_pos)


def _insert_into_cheapest_route(
    plan: Plan,
    node: int,
    inst: Inst,
    heatmap: np.ndarray | None = None,
    gamma: float = 0.0,
) -> None:
    """Insert ``node`` at its cheapest feasible position, opening a new route if it
    fits nowhere. One kernel call covers every route."""
    use_bias = heatmap is not None and gamma > 0.0
    if plan.routes:
        routes_flat, route_lens, route_loads = pack_routes(plan.routes, inst)
        _delta, best_route, best_pos = _best_insert_over_routes_numba(
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
            heatmap if use_bias else _NO_HEATMAP,
            gamma,
            use_bias,
        )
    else:
        best_route = -1
    if best_route >= 0:
        plan.routes[best_route].insert(best_pos, node)
    else:
        plan.routes.append([node])
    plan.invalidate()


def _insert_customer(plan: Plan, node: int, inst: Inst) -> None:
    _insert_into_cheapest_route(plan, node, inst)


def _route_cost_list(route: list[int], inst: Inst) -> float:
    if not route:
        return 0.0
    return float(_route_cost(np.array(route, np.int64), inst.dist))


def _route_load(route: list[int], inst: Inst) -> float:
    return float(sum(inst.demands[n] for n in route))


def _route_avg_slack(route: list[int], inst: Inst) -> float:
    if not route:
        return 0.0
    slack, t, prev = 0.0, 0.0, 0
    for node in route:
        t += inst.dist[prev, node]
        t = max(t, inst.ready_times[node])
        slack += inst.due_times[node] - t
        t += inst.service_times[node]
        prev = node
    return slack / len(route)


def build_greedy(inst: Inst, algo: str = "", heatmap: np.ndarray | None = None, gnn_strength: float = 0.0) -> Plan:
    has_gnn = heatmap is not None and gnn_strength > 0.0

    def get_dist(i, j):
        if has_gnn:
            return inst.dist[i, j] * (1.0 - gnn_strength * heatmap[i, j])
        return inst.dist[i, j]

    def arrival(route, pos, node, arrivals):
        prev = route[pos - 1] if pos > 0 else 0
        t = arrivals[pos - 1] if pos > 0 else 0.0
        return max(t + inst.dist[prev, node], inst.ready_times[node])

    def feasible_insert(route, pos, node, arrivals, load):
        if load + inst.demands[node] > inst.capacity:
            return False, None
        t = arrival(route, pos, node, arrivals)
        if t > inst.due_times[node]:
            return False, None
        ft, prev = t + inst.service_times[node], node
        for idx in range(pos, len(route)):
            nxt = route[idx]
            ft += inst.dist[prev, nxt]
            ft = max(ft, inst.ready_times[nxt])
            if ft > inst.due_times[nxt]:
                return False, None
            ft += inst.service_times[nxt]
            prev = nxt
        return True, t

    def compute_arrivals(route):
        arrivals, t, prev = [], 0.0, 0
        for node in route:
            t += inst.dist[prev, node]
            t = max(t, inst.ready_times[node])
            arrivals.append(t)
            t += inst.service_times[node]
            prev = node
        return arrivals

    def best_insert_cost(route, node, arrivals, load):
        best_cost, best_pos = float("inf"), None
        for pos in range(len(route) + 1):
            ok, _ = feasible_insert(route, pos, node, arrivals, load)
            if not ok:
                continue
            prev = route[pos - 1] if pos > 0 else 0
            nxt = route[pos] if pos < len(route) else 0
            delta = get_dist(prev, node) + get_dist(node, nxt) - get_dist(prev, nxt)
            if delta < best_cost:
                best_cost, best_pos = delta, pos
        return best_cost, best_pos

    unrouted = list(range(1, inst.n + 1))
    routes: list[list[int]] = []
    while unrouted:
        seed = max(unrouted, key=lambda n: get_dist(0, n))
        if max(inst.dist[0, seed], inst.ready_times[seed]) > inst.due_times[seed]:
            seed = min(unrouted, key=lambda n: inst.due_times[n])
        route = [seed]
        load = inst.demands[seed]
        arrivals = [max(inst.dist[0, seed], inst.ready_times[seed])]
        unrouted.remove(seed)
        improved = True
        while improved and unrouted:
            improved = False
            best_regret, best_node, best_pos = -float("inf"), None, None
            for node in unrouted:
                c1, pos = best_insert_cost(route, node, arrivals, load)
                if pos is None:
                    continue
                c2 = get_dist(0, node) + get_dist(node, 0) - c1
                if c2 > best_regret:
                    best_regret, best_node, best_pos = c2, node, pos
            if best_node is not None:
                route.insert(best_pos, best_node)
                load += inst.demands[best_node]
                arrivals = compute_arrivals(route)
                unrouted.remove(best_node)
                improved = True
        routes.append(route)

    plan = Plan(routes, inst, algo)
    if plan.feasible:
        return plan

    customers = sorted(range(1, inst.n + 1), key=lambda n: (inst.due_times[n], inst.ready_times[n]))
    unrouted_set = set(customers)
    fallback: list[list[int]] = []
    while unrouted_set:
        route_fb: list[int] = []
        node, load, t = 0, 0.0, 0.0
        while unrouted_set:
            feasible = [
                c
                for c in unrouted_set
                if load + inst.demands[c] <= inst.capacity and t + inst.dist[node, c] <= inst.due_times[c]
            ]
            if not feasible:
                break
            nxt = min(feasible, key=lambda c: get_dist(node, c))
            route_fb.append(nxt)
            unrouted_set.remove(nxt)
            load += inst.demands[nxt]
            t = max(t + inst.dist[node, nxt], inst.ready_times[nxt]) + inst.service_times[nxt]
            node = nxt
        if route_fb:
            fallback.append(route_fb)
        elif unrouted_set:
            nxt = next(iter(unrouted_set))
            fallback.append([nxt])
            unrouted_set.remove(nxt)
    return Plan(fallback, inst, algo)

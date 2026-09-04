from __future__ import annotations

import math
import random

import numpy as np
from numba import njit

from .config import MODES, Config
from .core import Inst, Plan, _invalidate, _route_duration_no_return
from .heuristics import (
    _NO_HEATMAP,
    _insert_costs_column_numba,
    _insert_costs_matrix_numba,
    _insert_feasible_numba,
    _route_timing_numba,
    pack_routes,
)
from .penalty import PenaltyManager


def _remove_nodes(plan: Plan, rs: set[int]) -> Plan:
    routes = [[n for n in r if n not in rs] for r in plan.routes]
    plan.routes = [r for r in routes if r]
    return _invalidate(plan)


def op_random(plan: Plan, size: int) -> tuple[Plan, list[int]]:
    nodes = [n for r in plan.routes for n in r]
    removed = random.sample(nodes, min(size, len(nodes)))
    rs = set(removed)
    return _remove_nodes(plan, rs), removed


def op_worst(plan: Plan, size: int) -> tuple[Plan, list[int]]:
    inst = plan.inst
    gains: list[tuple[float, int]] = []
    for route in plan.routes:
        for idx, node in enumerate(route):
            prev = route[idx - 1] if idx > 0 else 0
            nxt = route[idx + 1] if idx < len(route) - 1 else 0
            gains.append((inst.dist[prev, node] + inst.dist[node, nxt] - inst.dist[prev, nxt], node))
    gains.sort(reverse=True)
    p = 3.0
    removed: list[int] = []
    rs = set()
    while len(removed) < size and gains:
        idx = int((random.random() ** p) * len(gains))
        _, node = gains.pop(idx)
        removed.append(node)
        rs.add(node)
    return _remove_nodes(plan, rs), removed


def op_shaw(plan: Plan, size: int) -> tuple[Plan, list[int]]:
    inst = plan.inst
    nodes = [n for r in plan.routes for n in r]
    if not nodes:
        return plan, []
    seed_node = random.choice(nodes)
    removed = [seed_node]
    rs = {seed_node}
    relatedness = inst.relatedness
    while len(removed) < size:
        ref_node = random.choice(removed)
        neighbors = inst.neighbors_k[ref_node]
        candidates = [(n, relatedness[ref_node, n]) for n in neighbors if n not in rs]
        if not candidates:
            candidates = [(n, relatedness[ref_node, n]) for n in nodes if n not in rs]
        if not candidates:
            break
        candidates.sort(key=lambda x: x[1])
        p = 3.0
        idx = int((random.random() ** p) * len(candidates))
        chosen = candidates[idx][0]
        removed.append(chosen)
        rs.add(chosen)
    return _remove_nodes(plan, rs), removed


def op_route_portion_removal(plan: Plan, size: int) -> tuple[Plan, list[int]]:
    if len(plan.routes) <= 1:
        return op_shaw(plan, size)
    inst = plan.inst
    target = min(max(3, size), sum(len(r) for r in plan.routes))
    removed: list[int] = []
    routes = [r[:] for r in plan.routes]
    while len(removed) < target:
        nonempty = [r for r in routes if r]
        if not nonempty:
            break
        durations = [_route_duration_no_return(r, inst) for r in nonempty]
        avg_dur = max(float(np.mean(durations)), 1e-9)
        max_len = max(len(r) for r in nonempty)
        max_dist = max(inst.max_dist, 1.0)
        scored: list[tuple[float, int]] = []
        for ridx, route in enumerate(routes):
            if len(route) < 2:
                continue
            coords = inst.coords[np.array(route, dtype=np.int64)]
            centroid = coords.mean(axis=0)
            spatial = float(np.sqrt(((coords - centroid) ** 2).sum(axis=1)).mean()) / max_dist
            duration = _route_duration_no_return(route, inst) / avg_dur
            length_pressure = len(route) / max(max_len, 1)
            score = 0.45 * length_pressure + 0.35 * duration + 0.20 * spatial
            score += random.random() * 0.03
            scored.append((score, ridx))
        if not scored:
            break
        _, ridx = max(scored, key=lambda x: x[0])
        route = routes[ridx]
        remaining = target - len(removed)
        lower = max(1, int(math.floor(0.05 * len(route))))
        upper = min(max(lower, int(math.ceil(0.30 * len(route)))), len(route) - 1, remaining)
        if upper < 1:
            break
        seg_len = random.randint(min(lower, upper), upper)
        strain: list[tuple[float, int]] = []
        for pos, node in enumerate(route):
            prev = route[pos - 1] if pos > 0 else 0
            nxt = route[pos + 1] if pos < len(route) - 1 else 0
            arc = inst.dist[prev, node] + inst.dist[node, nxt] - inst.dist[prev, nxt]
            tw_width = inst.due_times[node] - inst.ready_times[node]
            urgency = 1.0 - min(tw_width / max(inst.max_tw_width, 1.0), 1.0)
            strain.append((arc / max_dist + 0.35 * urgency, pos))
        strain.sort(reverse=True, key=lambda x: x[0])
        p = 2.0
        idx = int((random.random() ** p) * len(strain))
        pivot = strain[idx][1]
        start = int(np.clip(pivot - seg_len // 2, 0, len(route) - seg_len))
        segment = route[start : start + seg_len]
        removed.extend(segment)
        routes[ridx] = route[:start] + route[start + seg_len :]
    if not removed:
        return op_shaw(plan, size)
    plan.routes = [r for r in routes if r]
    return _invalidate(plan), removed


def op_tw_urgent(plan: Plan, size: int) -> tuple[Plan, list[int]]:
    inst = plan.inst
    nodes = [n for r in plan.routes for n in r]
    if not nodes:
        return plan, []
    candidates = sorted(nodes, key=lambda n: (inst.due_times[n] - inst.ready_times[n]) * (1.0 + random.random() * 0.3))
    removed = candidates[:size]
    rs = set(removed)
    return _remove_nodes(plan, rs), removed


def op_route_eliminate(plan: Plan, size: int) -> tuple[Plan, list[int]]:
    if len(plan.routes) <= 1:
        return op_random(plan, size)
    inst = plan.inst
    ranked = sorted(
        enumerate(plan.routes),
        key=lambda x: (
            len(x[1]) + random.random() * 0.8,
            sum(inst.demands[n] for n in x[1]) / max(inst.capacity, 1) + random.random() * 0.1,
        ),
    )
    removed: list[int] = []
    drop_ids: set = set()
    for idx, route in ranked:
        if len(removed) >= size:
            break
        removed.extend(route)
        drop_ids.add(idx)
    plan.routes = [r for i, r in enumerate(plan.routes) if i not in drop_ids]
    return _invalidate(plan), removed


def op_route_dispersion_eliminate(plan: Plan, size: int) -> tuple[Plan, list[int]]:
    if len(plan.routes) <= 1:
        return op_random(plan, size)
    inst = plan.inst
    durations = [_route_duration_no_return(r, inst) for r in plan.routes if r]
    avg_dur = max(float(np.mean(durations)), 1e-9)
    max_dist = max(inst.max_dist, 1.0)
    scored: list[tuple[float, int]] = []
    for idx, route in enumerate(plan.routes):
        if not route:
            continue
        coords = inst.coords[np.array(route, dtype=np.int64)]
        centroid = coords.mean(axis=0)
        spatial = float(np.sqrt(((coords - centroid) ** 2).sum(axis=1)).mean()) / max_dist
        temporal = _route_duration_no_return(route, inst) / avg_dur
        noise = random.random() * 0.25
        scored.append((1.5 * spatial + 0.5 * temporal + noise, idx))
    removed: list[int] = []
    drop_ids: set = set()
    for _, idx in sorted(scored, reverse=True):
        if len(removed) >= size:
            break
        removed.extend(plan.routes[idx])
        drop_ids.add(idx)
    plan.routes = [r for i, r in enumerate(plan.routes) if i not in drop_ids]
    return _invalidate(plan), removed


def op_cross_route_shaw(plan: Plan, size: int) -> tuple[Plan, list[int]]:
    inst = plan.inst
    nodes = [n for r in plan.routes for n in r]
    if not nodes:
        return plan, []
    node_to_route = {n: ri for ri, route in enumerate(plan.routes) for n in route}
    seed_node = random.choice(nodes)
    removed = [seed_node]
    rs = {seed_node}
    relatedness = inst.relatedness
    while len(removed) < size:
        ref_node = random.choice(removed)
        ref_route = node_to_route.get(ref_node, -1)
        candidates = []
        for n in nodes:
            if n in rs:
                continue
            cross_route_bonus = -0.2 if node_to_route.get(n, -2) != ref_route else 0.2
            rel = relatedness[ref_node, n] + cross_route_bonus
            candidates.append((n, rel))
        if not candidates:
            break
        candidates.sort(key=lambda x: x[1])
        p = 3.0
        idx = int((random.random() ** p) * len(candidates))
        nxt = candidates[idx][0]
        removed.append(nxt)
        rs.add(nxt)
    return _remove_nodes(plan, rs), removed


def _remove_neighborhood_additional(plan: Plan, removed: list[int], size: int) -> list[int]:
    inst = plan.inst
    nodes = [n for r in plan.routes for n in r]
    rs = set(removed)
    candidates = [n for n in nodes if n not in rs]
    if not candidates or len(removed) >= size:
        return removed

    max_dist = inst.max_dist + 1e-9
    max_tw = max(inst.due_times - inst.ready_times) + 1e-9

    scores = []
    for n in candidates:
        min_score = float("inf")
        for ref in removed:
            score = (
                0.5 * inst.dist[ref, n] / max_dist
                + 0.4 * abs(inst.ready_times[ref] - inst.ready_times[n]) / max_tw
                + 0.1 * abs(inst.demands[ref] - inst.demands[n]) / inst.capacity
            )
            if score < min_score:
                min_score = score
        scores.append((n, min_score))

    scores.sort(key=lambda x: x[1])

    p = 3.0
    while len(removed) < size and scores:
        idx = int((random.random() ** p) * len(scores))
        chosen, _ = scores.pop(idx)
        removed.append(chosen)
        rs.add(chosen)

    return removed


def op_route_costly_eliminate(plan: Plan, size: int) -> tuple[Plan, list[int]]:
    if len(plan.routes) <= 1:
        return op_random(plan, size)
    inst = plan.inst
    scored = []
    from .heuristics import _route_cost_list

    for idx, route in enumerate(plan.routes):
        if not route:
            continue
        cost = _route_cost_list(route, inst)
        noise = random.random() * 0.1 * cost
        scored.append((cost + noise, idx))

    best_idx = max(scored, key=lambda x: x[0])[1]
    removed = list(plan.routes[best_idx])
    drop_ids = {best_idx}

    plan.routes = [r for i, r in enumerate(plan.routes) if i not in drop_ids]

    if len(removed) < size:
        removed = _remove_neighborhood_additional(plan, removed, size)
        rs = set(removed)
        plan.routes = [[n for n in r if n not in rs] for r in plan.routes]
        plan.routes = [r for r in plan.routes if r]

    return _invalidate(plan), removed


def op_route_merge_sample(plan: Plan, size: int) -> tuple[Plan, list[int]]:
    if len(plan.routes) <= 1:
        return op_random(plan, size)
    inst = plan.inst

    candidates = []
    for i, r1 in enumerate(plan.routes):
        load1 = sum(inst.demands[n] for n in r1)
        for j, r2 in enumerate(plan.routes):
            if j <= i:
                continue
            load2 = sum(inst.demands[n] for n in r2)
            if load1 + load2 > inst.capacity:
                continue
            combined_len = len(r1) + len(r2)
            load_fill = (load1 + load2) / max(inst.capacity, 1)
            score = combined_len - load_fill * 4.0 + random.random() * 1.5
            candidates.append((score, i, j))

    if not candidates:
        return op_route_eliminate(plan, size)

    candidates.sort()
    p = 3.0
    idx = int((random.random() ** p) * len(candidates))
    _, i, j = candidates[idx]

    small_idx = i if len(plan.routes[i]) <= len(plan.routes[j]) else j
    removed = list(plan.routes[small_idx])
    plan.routes = [r for k, r in enumerate(plan.routes) if k != small_idx]
    return _invalidate(plan), removed


def op_route_absorb_disrupt(plan: Plan, size: int) -> tuple[Plan, list[int]]:
    """Absorb the smallest route and disrupt border zones of neighboring routes.

    1. Select the route with fewest customers.
    2. Remove all customers from that route.
    3. For each absorbed customer, identify the nearest customers still in
       other routes and remove those 'border zone' neighbors.
    4. Cap total removals at ``size``.
    """
    if len(plan.routes) <= 1:
        return op_random(plan, size)
    inst = plan.inst
    # Find smallest route
    ranked = sorted(
        enumerate(plan.routes),
        key=lambda x: (len(x[1]) + random.random() * 0.5,),
    )
    target_idx, target_route = ranked[0]
    removed: list[int] = list(target_route)

    # Build remaining node list for border-zone disruption
    remaining_nodes = [n for i, r in enumerate(plan.routes) if i != target_idx for n in r]

    # For each absorbed customer, find nearest border-zone neighbors
    if remaining_nodes and len(removed) < size:
        absorbed_set = set(removed)
        border_candidates: list[tuple[float, int]] = []
        for node in removed:
            for rn in remaining_nodes:
                if rn not in absorbed_set:
                    dist_score = float(inst.dist[node, rn]) + random.random() * 0.1
                    border_candidates.append((dist_score, rn))

        # Sort by proximity and select unique border nodes
        border_candidates.sort(key=lambda x: x[0])
        seen = set(removed)
        for _, rn in border_candidates:
            if len(removed) >= size:
                break
            if rn not in seen:
                removed.append(rn)
                seen.add(rn)

    rs = set(removed)
    plan.routes = [[n for n in r if n not in rs] for r in plan.routes]
    plan.routes = [r for r in plan.routes if r]
    return _invalidate(plan), removed


def op_neural_worst(plan: Plan, size: int, heatmap: np.ndarray | None = None) -> tuple[Plan, list[int]]:
    """Remove customers whose current edges have low GNN-predicted probability.

    For each customer *node* currently placed between *prev* and *nxt* in its
    route, compute a removal score:
        score = (1 - heatmap[prev, node]) + (1 - heatmap[node, nxt])
    High score ⇒ the GNN thinks these edges are unlikely in a good solution
    ⇒ the customer is misplaced ⇒ prioritise it for removal.

    Falls back to ``op_worst`` when no heatmap is available.
    """
    if heatmap is None:
        return op_worst(plan, size)

    inst = plan.inst
    scores: list[tuple[float, int]] = []
    for route in plan.routes:
        for idx, node in enumerate(route):
            prev = route[idx - 1] if idx > 0 else 0
            nxt = route[idx + 1] if idx < len(route) - 1 else 0
            # Low heatmap probability → high removal score
            neural_score = (1.0 - heatmap[prev, node]) + (1.0 - heatmap[node, nxt])
            # Blend with detour cost for stability (30% distance, 70% neural)
            detour = inst.dist[prev, node] + inst.dist[node, nxt] - inst.dist[prev, nxt]
            detour_norm = detour / max(inst.max_dist, 1e-9)
            score = 0.7 * neural_score + 0.3 * detour_norm
            scores.append((score, node))
    scores.sort(reverse=True)

    p = 3.0
    removed: list[int] = []
    rs: set = set()
    while len(removed) < size and scores:
        idx = int((random.random() ** p) * len(scores))
        _, node = scores.pop(idx)
        if node not in rs:
            removed.append(node)
            rs.add(node)
    return _remove_nodes(plan, rs), removed


def op_neural_shaw(plan: Plan, size: int, heatmap: np.ndarray | None = None) -> tuple[Plan, list[int]]:
    """Remove a coherent cluster of poorly-connected customers using Shaw relatedness and GNN edge probability.

    Falls back to ``op_shaw`` when no heatmap is available.
    """
    if heatmap is None:
        return op_shaw(plan, size)

    inst = plan.inst
    nodes = [n for r in plan.routes for n in r]
    if not nodes:
        return plan, []

    # 1. Seed: pick customer with lowest avg GNN edge probability to neighbors in current route
    seed_node = None
    min_avg_prob = 2.0
    for route in plan.routes:
        for idx, node in enumerate(route):
            prev = route[idx - 1] if idx > 0 else 0
            nxt = route[idx + 1] if idx < len(route) - 1 else 0
            avg_prob = 0.5 * (heatmap[prev, node] + heatmap[node, nxt])
            if avg_prob < min_avg_prob:
                min_avg_prob = avg_prob
                seed_node = node

    if seed_node is None:
        seed_node = random.choice(nodes)

    removed = [seed_node]
    rs = {seed_node}

    while len(removed) < size:
        ref_node = random.choice(removed)
        neighbors = inst.neighbors_k[ref_node]
        candidates = []
        for n in neighbors:
            if n not in rs:
                avg_heatmap_prob = sum(0.5 * (heatmap[r, n] + heatmap[n, r]) for r in removed) / len(removed)
                score = (
                    0.35 * inst.spatial_rel[ref_node, n]
                    + 0.25 * inst.temporal_rel[ref_node, n]
                    + 0.30 * (1.0 - avg_heatmap_prob)
                    + 0.10 * inst.demand_rel[ref_node, n]
                )
                candidates.append((n, score))

        if not candidates:
            for n in nodes:
                if n not in rs:
                    avg_heatmap_prob = sum(0.5 * (heatmap[r, n] + heatmap[n, r]) for r in removed) / len(removed)
                    score = (
                        0.35 * inst.spatial_rel[ref_node, n]
                        + 0.25 * inst.temporal_rel[ref_node, n]
                        + 0.30 * (1.0 - avg_heatmap_prob)
                        + 0.10 * inst.demand_rel[ref_node, n]
                    )
                    candidates.append((n, score))

        if not candidates:
            break

        candidates.sort(key=lambda x: x[1])
        p = 3.0
        idx = int((random.random() ** p) * len(candidates))
        chosen = candidates[idx][0]
        removed.append(chosen)
        rs.add(chosen)

    return _remove_nodes(plan, rs), removed


DESTROY = [
    op_random,
    op_worst,
    op_shaw,
    op_route_portion_removal,
    op_tw_urgent,
    op_route_eliminate,
    op_route_dispersion_eliminate,
    op_route_costly_eliminate,
    op_cross_route_shaw,
    op_route_merge_sample,
    op_route_absorb_disrupt,
    op_neural_worst,
    op_neural_shaw,
]


def _sequential_cheapest_insert(plan: Plan, order: list[int], heatmap: np.ndarray | None, gamma: float) -> Plan:
    """Insert ``order`` one node at a time, each at its cheapest feasible position.

    Behaviour-identical to calling ``_insert_into_cheapest_route`` per node, but
    the (node x route) insertion-cost matrix is built once and refreshed one
    column at a time — the same pattern ``_regret`` uses. The per-node path
    re-derived every route's timing profile for every insertion, O(|order| x R x m)
    instead of O(R x m + |order| x m).

    A node that fits nowhere opens a new route, which becomes a new column so
    later nodes can insert into it, exactly as before.
    """
    inst = plan.inst
    if not order:
        return Plan(plan.routes, inst, plan.algo)
    use_bias = heatmap is not None and gamma > 0.0
    hm = heatmap if use_bias else _NO_HEATMAP

    nodes = np.asarray(order, dtype=np.int64)
    n = len(nodes)

    n_routes = len(plan.routes)
    width = n_routes + n  # worst case: every node opens a new route
    keys = np.full((n, width), 1e18, dtype=np.float64)
    positions = np.full((n, width), -1, dtype=np.int64)
    loads = np.zeros(width, dtype=np.float64)

    if n_routes > 0:
        routes_flat, route_lens, route_loads = pack_routes(plan.routes, inst)
        k0, _d0, p0 = _insert_costs_matrix_numba(
            nodes,
            routes_flat,
            route_lens,
            route_loads,
            inst.dist,
            inst.demands,
            inst.capacity,
            inst.ready_times,
            inst.due_times,
            inst.service_times,
            hm,
            gamma,
            use_bias,
        )
        keys[:, :n_routes] = k0
        positions[:, :n_routes] = p0
        loads[:n_routes] = route_loads

    for i in range(n):
        node = int(nodes[i])
        ri = -1
        if n_routes > 0:
            cand_ri = int(np.argmin(keys[i, :n_routes]))
            # key < 1e18 iff the kernel found a feasible position; argmin picks
            # the first (lowest-index) minimum, matching the strict `<` scan of
            # _best_insert_over_routes_numba.
            if positions[i, cand_ri] >= 0:
                ri = cand_ri
        if ri >= 0:
            plan.routes[ri].insert(int(positions[i, ri]), node)
            loads[ri] += float(inst.demands[node])
        else:
            plan.routes.append([node])
            ri = n_routes
            n_routes += 1
            loads[ri] = float(inst.demands[node])
        if i + 1 < n:
            # Only the column of the modified (or newly opened) route is stale.
            new_route = np.array(plan.routes[ri], dtype=np.int64)
            col_k, _cd, col_p = _insert_costs_column_numba(
                nodes,
                new_route,
                loads[ri],
                inst.dist,
                inst.demands,
                inst.capacity,
                inst.ready_times,
                inst.due_times,
                inst.service_times,
                hm,
                gamma,
                use_bias,
            )
            keys[:, ri] = col_k
            positions[:, ri] = col_p

    plan.invalidate()
    return Plan(plan.routes, inst, plan.algo)


def op_greedy(plan: Plan, removed: list[int], heatmap: np.ndarray | None = None, gamma: float = 0.0) -> Plan:
    inst = plan.inst
    return _sequential_cheapest_insert(plan, sorted(removed, key=lambda n: inst.due_times[n]), heatmap, gamma)


def _regret(plan: Plan, removed: list[int], k: int, heatmap: np.ndarray | None = None, gamma: float = 0.0) -> Plan:
    """Regret-k insertion.

    The (node x route) insertion-cost matrix is built in a single kernel call and
    then refreshed one column at a time: inserting a node only invalidates the
    route it landed in, so recomputing the whole matrix each round — as this used
    to — repeated O(|removed|) times the work actually needed.

    The Python-level selection loop is kept deliberately: it iterates ``remaining``
    in set order, and reproducing that order exactly is what keeps this change
    behaviour-preserving.
    """
    inst = plan.inst
    remaining: set = set(removed)
    if not remaining or not plan.routes:
        if remaining:
            for node in remaining:
                plan.routes.append([node])
        return Plan(plan.routes, inst, plan.algo)

    use_bias = heatmap is not None and gamma > 0.0
    hm = heatmap if use_bias else _NO_HEATMAP

    nodes = np.fromiter(sorted(remaining), dtype=np.int64, count=len(remaining))
    row_of = {int(nd): i for i, nd in enumerate(nodes)}

    routes_flat, route_lens, route_loads = pack_routes(plan.routes, inst)
    keys, _deltas, positions = _insert_costs_matrix_numba(
        nodes,
        routes_flat,
        route_lens,
        route_loads,
        inst.dist,
        inst.demands,
        inst.capacity,
        inst.ready_times,
        inst.due_times,
        inst.service_times,
        hm,
        gamma,
        use_bias,
    )

    while remaining:
        best_regret, chosen, choice = -float("inf"), None, None
        for node in remaining:
            i = row_of[node]
            row = keys[i]
            n_opt = int((positions[i] >= 0).sum())
            if n_opt == 0:
                continue

            if n_opt >= k:
                # Infeasible pairs carry a 1e18 key, so they sort last and never
                # enter the k smallest while a feasible option remains.
                best_k = np.sort(np.partition(row, k - 1)[:k])
                regret = 0.0
                for j in range(1, k):
                    regret += best_k[j] - best_k[0]
            elif n_opt >= 2:
                best_2 = np.sort(np.partition(row, 1)[:2])
                regret = best_2[1] - best_2[0]
            else:
                regret = float("inf")

            if regret > best_regret:
                best_ri = int(np.argmin(row))
                best_regret, chosen, choice = regret, node, (best_ri, int(positions[i, best_ri]))

        if chosen is not None and choice is not None:
            ri, pos = choice
            plan.routes[ri].insert(pos, chosen)
            plan.invalidate()
            remaining.discard(chosen)

            # Only the modified route's column is stale.
            route_loads[ri] += float(inst.demands[chosen])
            new_route = np.array(plan.routes[ri], dtype=np.int64)
            col_keys, col_deltas, col_pos = _insert_costs_column_numba(
                nodes,
                new_route,
                route_loads[ri],
                inst.dist,
                inst.demands,
                inst.capacity,
                inst.ready_times,
                inst.due_times,
                inst.service_times,
                hm,
                gamma,
                use_bias,
            )
            keys[:, ri] = col_keys
            _deltas[:, ri] = col_deltas
            positions[:, ri] = col_pos
        else:
            for node in remaining:
                plan.routes.append([node])
            break
    return Plan(plan.routes, inst, plan.algo)


def op_regret_2(plan: Plan, removed: list[int], heatmap: np.ndarray | None = None, gamma: float = 0.0) -> Plan:
    return _regret(plan, removed, 2, heatmap, gamma)


def op_regret_3(plan: Plan, removed: list[int], heatmap: np.ndarray | None = None, gamma: float = 0.0) -> Plan:
    return _regret(plan, removed, 3, heatmap, gamma)


def op_tw_greedy(plan: Plan, removed: list[int], heatmap: np.ndarray | None = None, gamma: float = 0.0) -> Plan:
    inst = plan.inst
    return _sequential_cheapest_insert(
        plan,
        sorted(removed, key=lambda n: inst.due_times[n] - inst.ready_times[n]),
        heatmap,
        gamma,
    )


@njit(cache=True)
def _fts_best_insert_position_numba(
    node: int,
    route: np.ndarray,
    dist: np.ndarray,
    demands: np.ndarray,
    capacity: float,
    ready: np.ndarray,
    due: np.ndarray,
    service: np.ndarray,
    tw_tight_frac: float,
) -> tuple[float, int]:
    """Time-window-aware cheapest insertion: distance plus a penalty for waiting.

    This operator previously added a bonus for preserving downstream slack,
    weighted up to 0.85 * max_dist. Since max_dist dwarfs a typical insertion
    delta, that term drove position choice almost entirely by slack preservation,
    spreading customers over extra vehicles: measured against baseline it pushed
    RC207 from 3 to 4 vehicles on 3 of 5 seeds and rc1_2_1 from 19.2 to 19.8,
    trading vehicles for distance in a problem whose objective is lexicographic in
    vehicles first. The weights had never been validated because the operator
    returned an infeasible plan on every trial until the push-forward fix below,
    so nothing ever exercised them. Dropping the term restores the vehicle counts
    and still improves gap-to-BKS over baseline.
    """
    best_cost = 1e18
    best_pos = -1

    n_nodes = len(route)
    current_load = 0.0
    for idx in range(n_nodes):
        current_load += demands[route[idx]]
    if current_load + demands[node] > capacity:
        return 1e18, -1

    wait_weight = 0.10 + 0.35 * tw_tight_frac

    arrivals, latest, first_violation = _route_timing_numba(route, dist, ready, due, service)

    for pos in range(n_nodes + 1):
        if pos > first_violation:
            break
        prev = route[pos - 1] if pos > 0 else 0
        nxt = route[pos] if pos < n_nodes else 0

        # Full push-forward feasibility. The previous test used the *arrival* time
        # at ``prev`` without adding its service time, and only validated the
        # immediate successor's due date rather than propagating the delay along
        # the rest of the route — so this operator returned an infeasible plan on
        # 100% of trials (measured 60/60 on both R101 and r1_2_1).
        if not _insert_feasible_numba(node, pos, route, arrivals, latest, dist, ready, due, service):
            continue

        t_prev_depart = (arrivals[pos - 1] + service[prev]) if pos > 0 else 0.0
        dist_added = dist[prev, node] + dist[node, nxt] - dist[prev, nxt]
        wait_added = max(0.0, ready[node] - (t_prev_depart + dist[prev, node]))

        composite = dist_added + wait_weight * wait_added
        if composite < best_cost:
            best_cost = composite
            best_pos = pos

    return best_cost, best_pos


def _fts_best_insert_position(node: int, route: list[int], inst: Inst) -> tuple[float, int | None]:
    route_arr = np.array(route, dtype=np.int64)
    best_cost, best_pos = _fts_best_insert_position_numba(
        node,
        route_arr,
        inst.dist,
        inst.demands,
        inst.capacity,
        inst.ready_times,
        inst.due_times,
        inst.service_times,
        inst.tw_tight_frac,
    )
    if best_pos == -1:
        return float("inf"), None
    return float(best_cost), int(best_pos)


def op_fts_greedy(plan: Plan, removed: list[int], heatmap: np.ndarray | None = None, gamma: float = 0.0) -> Plan:
    inst = plan.inst
    urgent_first = sorted(
        removed,
        key=lambda n: (inst.due_times[n] - inst.ready_times[n], inst.due_times[n], -inst.demands[n]),
    )
    for node in urgent_first:
        best_cost, best_route, best_pos = float("inf"), None, None
        for ri, route in enumerate(plan.routes):
            cost, pos = _fts_best_insert_position(node, route, inst)
            if pos is not None and cost < best_cost:
                best_cost, best_route, best_pos = cost, ri, pos
        if best_route is not None and best_pos is not None:
            plan.routes[best_route].insert(best_pos, node)
        else:
            plan.routes.append([node])
        plan.invalidate()
    return Plan(plan.routes, inst, plan.algo)


REPAIR = [op_greedy, op_regret_2, op_regret_3, op_tw_greedy, op_fts_greedy]
N_D, N_R = len(DESTROY), len(REPAIR)
N_ACTIONS = N_D * N_R

assert N_D == 13
assert N_R == 5
for _m in MODES:
    assert len(_m.destroy_bias) == N_D
    assert len(_m.repair_bias) == N_R


def accept(cur: Plan, cand: Plan, temp: float, allow_nv_increase: bool = False) -> bool:
    if not cand.feasible:
        return False
    if cand.nv < cur.nv:
        return True
    if cand.nv == cur.nv:
        if cand.cost < cur.cost:
            return True
        return random.random() < math.exp(-(cand.cost - cur.cost) / max(temp, 1e-6))
    if allow_nv_increase and cand.nv == cur.nv + 1:
        return random.random() < math.exp(-2.0)
    return False


def accept_with_nv_ceiling(
    cur: Plan, cand: Plan, temp: float, nv_ceiling: int, allow_nv_increase: bool = False
) -> bool:
    allowed_nv = max(nv_ceiling, cur.nv + 1) if allow_nv_increase else nv_ceiling
    if not cand.feasible or cand.nv > allowed_nv:
        return False
    return accept(cur, cand, temp, allow_nv_increase=allow_nv_increase)


def accept_penalized(cur: Plan, cand: Plan, temp: float, penalty_manager: PenaltyManager) -> bool:
    cur_p = penalty_manager.penalized_cost(cur)
    cand_p = penalty_manager.penalized_cost(cand)
    if cand_p < cur_p:
        return True
    return random.random() < math.exp(-(cand_p - cur_p) / max(temp, 1e-6))


def destroy_size(it: int, n_iters: int, cfg: Config, n_customers: int, scale: float = 1.0) -> int:
    """Monotone ramp from ratio_max down to ratio_min across the run.

    A sawtooth (cyclic) variant was tried here, motivated by the measurement that
    the incumbent's excursion above the best shrinks from 3.44% to 0.93% over a
    3000-iteration run while population restarts reheat the temperature but not
    the perturbation size. At equal *iterations* it looked like a win (net -5
    vehicles, distance unchanged at matched vehicle count). At equal *wall time*
    it lost on both metrics: the cyclic schedule costs ~2x per iteration, and
    simply running this monotone schedule for twice as many iterations beat it
    (13.289 vs 13.344 vehicles, 3.85% vs 4.46% gap-to-BKS). Do not re-add it
    without an iso-time comparison.
    """
    ratio = cfg.destroy_ratio_max - ((cfg.destroy_ratio_max - cfg.destroy_ratio_min) * (it / max(n_iters, 1)))
    ratio = min(cfg.destroy_ratio_max, max(cfg.destroy_ratio_min, ratio * scale))
    return max(3, int(ratio * n_customers))

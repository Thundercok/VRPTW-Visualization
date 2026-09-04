from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np

from .core import Inst, Plan, _check_route, _route_cost
from .heuristics import (
    _NO_HEATMAP,
    _best_insert_in_route_numba,
    _best_insert_in_route_pruned_numba,
    _best_insert_position,
    _route_cost_list,
    _route_load,
    _route_timing_numba,
)
from .numba_kernels import (
    _cross_exchange_pair_numba,
    _cross_exchange_pair_pruned_numba,
    _cross_tail_pair_numba,
    _cross_tail_pair_pruned_numba,
    _intra_route_optimize_numba,
    _or_opt_intra_numba,
    _string_relocate_pair_numba,
    _string_relocate_pair_pruned_numba,
    _swap_21_evaluate_numba,
    _swap_evaluate_numba,
    _swap_evaluate_pruned_numba,
    _two_opt_best_numba,
)


@dataclass
class _PlanCache:
    inst: Inst
    route_arrays: list[np.ndarray]
    centroids: list[np.ndarray]
    route_sets: list[set[int]]
    route_neighbors: list[set[int]]
    route_loads: list[float]
    node_to_route: np.ndarray
    route_readys: list[float]
    route_dues: list[float]
    centroid_arr: np.ndarray
    centroid_sqdist: np.ndarray
    route_timings: list[tuple[np.ndarray, np.ndarray, int]]
    content_map: dict[bytes, int]
    route_keys: list[bytes]
    # Memo of pure kernel results keyed by route-content bytes:
    #   (kernel_tag, key_i, key_j)  for pairwise scans,
    #   (kernel_tag, node, key_j)   for per-(node, dest-route) insert scans.
    # Kernel outputs are pure functions of route contents (+inst), so entries
    # stay valid across moves and are carried over via ``prev``; after a move
    # only pairs touching a changed route miss. Only the heatmap-free paths are
    # memoised — heatmap variants depend on external state.
    scan_memo: dict

    @classmethod
    def from_plan(cls, plan: Plan, prev: _PlanCache | None = None) -> _PlanCache:
        """Build the per-plan move-scan cache.

        When ``prev`` is the cache of the plan this one was derived from (the
        local-search loop passes the previous iteration's cache), per-route data
        is reused for every route whose node sequence is unchanged — a single
        move touches at most two routes, so rebuilding all of it per accepted
        move repeated O(R) work for O(1) change. Routes are keyed by the raw
        bytes of their node array; all reused entries are pure functions of
        (route content, inst) and are never mutated by the scans, so sharing
        them across caches is safe.
        """
        inst = plan.inst
        route_arrays = plan.route_arrays
        n_r = len(route_arrays)
        prev_map = prev.content_map if prev is not None and prev.inst is inst else None

        route_timings: list = [None] * n_r
        centroids: list = [None] * n_r
        route_sets: list = [None] * n_r
        route_neighbors: list = [None] * n_r
        route_loads: list = [0.0] * n_r
        route_readys: list = [0.0] * n_r
        route_dues: list = [0.0] * n_r
        content_map: dict[bytes, int] = {}
        route_keys: list = [b""] * n_r

        for i, arr in enumerate(route_arrays):
            key = arr.tobytes()
            content_map[key] = i
            route_keys[i] = key
            j = prev_map.get(key, -1) if prev_map is not None else -1
            if j >= 0:
                route_timings[i] = prev.route_timings[j]
                centroids[i] = prev.centroids[j]
                route_sets[i] = prev.route_sets[j]
                route_neighbors[i] = prev.route_neighbors[j]
                route_loads[i] = prev.route_loads[j]
                route_readys[i] = prev.route_readys[j]
                route_dues[i] = prev.route_dues[j]
                continue
            # Arrival/latest profiles are a property of the route, not of the
            # node being inserted: built once per route, reused across the whole
            # move scan instead of per (node, route) pair.
            route_timings[i] = _route_timing_numba(arr, inst.dist, inst.ready_times, inst.due_times, inst.service_times)
            centroids[i] = inst.coords[arr].mean(axis=0) if len(arr) > 0 else inst.coords[0]
            r = plan.routes[i]
            route_sets[i] = set(r)
            nbrs: set[int] = set()
            for u in r:
                if u < len(inst.neighbors_k):
                    nbrs.update(inst.neighbors_k[u])
            route_neighbors[i] = nbrs
            route_loads[i] = sum(inst.demands[c] for c in r)
            route_readys[i] = float(np.min(inst.ready_times[arr])) if len(arr) > 0 else 0.0
            route_dues[i] = float(np.max(inst.due_times[arr])) if len(arr) > 0 else 0.0

        # Pairwise centroid distances, precomputed once and compared squared. The
        # per-pair np.linalg.norm this replaces accounted for 3.8M calls / 22s at
        # n=400.
        centroid_arr = np.asarray(centroids, dtype=np.float64) if centroids else np.zeros((0, 2), dtype=np.float64)
        cdiff = centroid_arr[:, None, :] - centroid_arr[None, :, :]
        centroid_sqdist = (cdiff * cdiff).sum(axis=2)

        # Vectorised per-route writes replace the O(n) Python dict rebuild.
        node_to_route = np.full(inst.n + 1, -1, dtype=np.int64)
        for ri, arr in enumerate(route_arrays):
            node_to_route[arr] = ri

        return cls(
            inst=inst,
            route_arrays=route_arrays,
            centroids=centroids,
            route_sets=route_sets,
            route_neighbors=route_neighbors,
            route_loads=route_loads,
            node_to_route=node_to_route,
            route_readys=route_readys,
            route_dues=route_dues,
            centroid_arr=centroid_arr,
            centroid_sqdist=centroid_sqdist,
            route_timings=route_timings,
            content_map=content_map,
            route_keys=route_keys,
            # Cache, not state: dropping it is always safe. The size cap only
            # matters if a caller chains caches across an unusually long run.
            scan_memo=(
                prev.scan_memo if prev is not None and prev.inst is inst and len(prev.scan_memo) < 100_000 else {}
            ),
        )


def _two_opt_best(route: list[int], inst: Inst) -> list[int]:
    if len(route) < 4:
        return route[:]
    route_arr = np.array(route, dtype=np.int64)
    result, _ = _two_opt_best_numba(
        route_arr,
        inst.dist,
        inst.demands,
        inst.capacity,
        inst.ready_times,
        inst.due_times,
        inst.service_times,
    )
    return list(result)


def _capacity_ok(route: list[int] | np.ndarray, inst: Inst) -> bool:
    return sum(inst.demands[c] for c in route) <= inst.capacity


def _best_relocate(
    plan: Plan,
    nv_ceiling: int | None = None,
    heatmap: np.ndarray | None = None,
    pruning_threshold: float = 0.0,
    cache: _PlanCache | None = None,
):
    inst = plan.inst
    best_delta, best_move = -1e-9, None
    max_dist = max(inst.max_dist, 1.0)

    if cache is None:
        cache = _PlanCache.from_plan(plan)

    route_arrays = cache.route_arrays
    route_loads = cache.route_loads
    node_to_route = cache.node_to_route
    centroid_arr = cache.centroid_arr
    route_timings = cache.route_timings
    route_keys = cache.route_keys
    scan_memo = cache.scan_memo
    relocate_thresh_sq = (0.55 * max_dist) ** 2

    for si in range(len(plan.routes)):
        source_route = plan.routes[si]
        for sp in range(len(source_route)):
            node = source_route[sp]
            sn_empty = len(source_route) == 1
            # Compute removal delta in O(1) — removing a node from a feasible
            # route always yields a feasible route (earlier arrivals, lower load)
            prev_s = source_route[sp - 1] if sp > 0 else 0
            next_s = source_route[sp + 1] if sp < len(source_route) - 1 else 0
            remove_delta = inst.dist[prev_s, next_s] - inst.dist[prev_s, node] - inst.dist[node, next_s]
            node_coord = inst.coords[node]

            # kNN-filtered destination routes. Same insertion order as the dict
            # this replaces (-1 marks unrouted, mirroring the old `in` check),
            # so equal-delta tie-breaking is unchanged.
            candidate_routes = set()
            if node < len(inst.neighbors_k):
                for neighbor in inst.neighbors_k[node]:
                    ri_n = node_to_route[neighbor]
                    if ri_n >= 0:
                        candidate_routes.add(int(ri_n))
            candidate_routes.discard(si)
            # Fallback: if kNN yields too few candidates, use centroid filter
            if len(candidate_routes) < 3:
                cdiff = centroid_arr - node_coord
                near = np.flatnonzero((cdiff * cdiff).sum(axis=1) <= relocate_thresh_sq)
                for di in near:
                    di = int(di)
                    if di != si:
                        candidate_routes.add(di)

            for di in candidate_routes:
                # Fast capacity pre-check
                if route_loads[di] + inst.demands[node] > inst.capacity:
                    continue

                # Check if pruning is enabled and heatmap is provided
                if heatmap is not None and pruning_threshold > 0.0:
                    arrivals, latest, first_violation = route_timings[di]
                    insert_delta, best_pos = _best_insert_in_route_pruned_numba(
                        node,
                        route_arrays[di],
                        route_loads[di],
                        arrivals,
                        latest,
                        first_violation,
                        inst.dist,
                        inst.demands,
                        inst.capacity,
                        inst.ready_times,
                        inst.due_times,
                        inst.service_times,
                        heatmap,
                        pruning_threshold,
                    )
                else:
                    # Pure function of (node, destination route content):
                    # memoised across moves that leave the destination unchanged.
                    mkey = (1, node, route_keys[di])
                    hit = scan_memo.get(mkey)
                    if hit is None:
                        arrivals, latest, first_violation = route_timings[di]
                        hit = _best_insert_in_route_numba(
                            node,
                            route_arrays[di],
                            route_loads[di],
                            arrivals,
                            latest,
                            first_violation,
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
                        scan_memo[mkey] = hit
                    _key, insert_delta, best_pos = hit
                if best_pos == -1:
                    continue
                new_nv = plan.nv - (1 if sn_empty else 0)
                if nv_ceiling is not None and new_nv > nv_ceiling:
                    continue
                delta = remove_delta + insert_delta
                if new_nv < plan.nv:
                    delta -= 1000.0
                if delta < best_delta:
                    best_delta = delta
                    best_move = (si, sp, di, int(best_pos))
                    best_cost_delta = remove_delta + insert_delta
    if best_move is None:
        return None
    return best_move, best_cost_delta


def _apply_relocate(plan: Plan, move: tuple[int, int, int, int]) -> Plan:
    si, sp, di, ip = move
    routes = [r[:] for r in plan.routes]
    node = routes[si].pop(sp)
    if si < di and len(routes[si]) == 0:
        di -= 1
    routes = [r for r in routes if r]
    routes[di].insert(ip, node)
    return Plan(routes, plan.inst, plan.algo)


def _best_swap(
    plan: Plan,
    heatmap: np.ndarray | None = None,
    pruning_threshold: float = 0.0,
    cache: _PlanCache | None = None,
):
    inst = plan.inst
    best_delta, best_move = -1e-9, None
    max_dist = max(inst.max_dist, 1.0)

    if cache is None:
        cache = _PlanCache.from_plan(plan)

    route_arrays = cache.route_arrays
    centroid_sqdist = cache.centroid_sqdist
    route_sets = cache.route_sets
    route_neighbors = cache.route_neighbors
    route_keys = cache.route_keys
    scan_memo = cache.scan_memo
    pair_thresh_sq = (0.65 * max_dist) ** 2

    for si in range(len(plan.routes)):
        for di in range(si + 1, len(plan.routes)):
            # Neighbor list pruning: skip route pairs with no neighboring customers
            if route_sets[si].isdisjoint(route_neighbors[di]) and route_sets[di].isdisjoint(route_neighbors[si]):
                continue
            # Spatial filter: skip route pairs whose centroids are too far apart
            if centroid_sqdist[si, di] > pair_thresh_sq:
                continue
            if heatmap is not None and pruning_threshold > 0.0:
                delta, sp, dp = _swap_evaluate_pruned_numba(
                    route_arrays[si],
                    route_arrays[di],
                    inst.dist,
                    inst.demands,
                    inst.capacity,
                    inst.ready_times,
                    inst.due_times,
                    inst.service_times,
                    heatmap,
                    pruning_threshold,
                )
            else:
                mkey = (2, route_keys[si], route_keys[di])
                hit = scan_memo.get(mkey)
                if hit is None:
                    hit = _swap_evaluate_numba(
                        route_arrays[si],
                        route_arrays[di],
                        inst.dist,
                        inst.demands,
                        inst.capacity,
                        inst.ready_times,
                        inst.due_times,
                        inst.service_times,
                    )
                    scan_memo[mkey] = hit
                delta, sp, dp = hit
            if sp >= 0 and delta < best_delta:
                best_delta = delta
                best_move = (si, int(sp), di, int(dp))
    if best_move is None:
        return None
    return best_move, best_delta


def _apply_swap(plan: Plan, move: tuple[int, int, int, int]) -> Plan:
    si, sp, di, dp = move
    routes = [r[:] for r in plan.routes]
    routes[si][sp], routes[di][dp] = routes[di][dp], routes[si][sp]
    return Plan(routes, plan.inst, plan.algo)


def _best_swap_21(
    plan: Plan,
    heatmap: np.ndarray | None = None,
    pruning_threshold: float = 0.0,
    cache: _PlanCache | None = None,
):
    inst = plan.inst
    best_delta, best_move = -1e-9, None
    max_dist = max(inst.max_dist, 1.0)

    if cache is None:
        cache = _PlanCache.from_plan(plan)

    route_arrays = cache.route_arrays
    centroid_sqdist = cache.centroid_sqdist
    route_sets = cache.route_sets
    route_neighbors = cache.route_neighbors
    route_keys = cache.route_keys
    scan_memo = cache.scan_memo
    pair_thresh_sq = (0.65 * max_dist) ** 2

    for si in range(len(plan.routes)):
        for di in range(si + 1, len(plan.routes)):
            if route_sets[si].isdisjoint(route_neighbors[di]) and route_sets[di].isdisjoint(route_neighbors[si]):
                continue
            if centroid_sqdist[si, di] > pair_thresh_sq:
                continue
            mkey = (21, route_keys[si], route_keys[di])
            hit = scan_memo.get(mkey)
            if hit is None:
                hit = _swap_21_evaluate_numba(
                    route_arrays[si],
                    route_arrays[di],
                    inst.dist,
                    inst.demands,
                    inst.capacity,
                    inst.ready_times,
                    inst.due_times,
                    inst.service_times,
                )
                scan_memo[mkey] = hit
            delta, mtype, sp, dp = hit
            if mtype > 0 and delta < best_delta:
                best_delta = delta
                best_move = (si, di, int(mtype), int(sp), int(dp))
    if best_move is None:
        return None
    return best_move, best_delta


def _apply_swap_21(plan: Plan, move: tuple[int, int, int, int, int]) -> Plan:
    si, di, mtype, sp, dp = move
    routes = [r[:] for r in plan.routes]
    r1, r2 = routes[si], routes[di]
    if mtype == 21:
        n_a, n_b = r1[sp], r1[sp + 1]
        n_c = r2[dp]
        nr1 = r1[:sp] + [n_c] + r1[sp + 2 :]
        nr2 = r2[:dp] + [n_a, n_b] + r2[dp + 1 :]
        routes[si], routes[di] = nr1, nr2
    elif mtype == 12:
        n_c = r1[sp]
        n_a, n_b = r2[dp], r2[dp + 1]
        nr1 = r1[:sp] + [n_a, n_b] + r1[sp + 1 :]
        nr2 = r2[:dp] + [n_c] + r2[dp + 2 :]
        routes[si], routes[di] = nr1, nr2
    return Plan(routes, plan.inst, plan.algo)


def _cross_exchange(
    plan: Plan,
    nv_ceiling: int | None = None,
    heatmap: np.ndarray | None = None,
    pruning_threshold: float = 0.0,
    cache: _PlanCache | None = None,
) -> Plan | None:
    inst = plan.inst
    if nv_ceiling is not None and plan.nv > nv_ceiling:
        return None
    max_dist = max(inst.max_dist, 1.0)
    best_delta = -1e-9
    best_routes: list[list[int]] | None = None

    if cache is None:
        cache = _PlanCache.from_plan(plan)

    route_arrays = cache.route_arrays
    centroid_sqdist = cache.centroid_sqdist
    route_readys = cache.route_readys
    route_dues = cache.route_dues
    route_sets = cache.route_sets
    route_neighbors = cache.route_neighbors
    route_keys = cache.route_keys
    scan_memo = cache.scan_memo
    pair_thresh_sq = (0.55 * max_dist) ** 2

    for i in range(len(plan.routes)):
        for j in range(i + 1, len(plan.routes)):
            r1, r2 = plan.routes[i], plan.routes[j]
            if not r1 or not r2:
                continue

            # Neighbor list pruning: skip route pairs with no neighboring customers
            if route_sets[i].isdisjoint(route_neighbors[j]) and route_sets[j].isdisjoint(route_neighbors[i]):
                continue

            r1_arr, r2_arr = route_arrays[i], route_arrays[j]
            # Route-level spatial/temporal filter
            r1_ready, r1_due = route_readys[i], route_dues[i]
            r2_ready, r2_due = route_readys[j], route_dues[j]
            if centroid_sqdist[i, j] > pair_thresh_sq and not (min(r1_due, r2_due) >= max(r1_ready, r2_ready)):
                continue
            # Delegate entire (p1, p2, len1, len2) search to JIT
            if heatmap is not None and pruning_threshold > 0.0:
                old_pair = float(_route_cost(r1_arr, inst.dist)) + float(_route_cost(r2_arr, inst.dist))
                max_len1 = 4 if len(r1) >= 20 else (3 if len(r1) >= 12 else 2)
                max_len2 = 4 if len(r2) >= 20 else (3 if len(r2) >= 12 else 2)
                delta, p1, len1, p2, len2 = _cross_exchange_pair_pruned_numba(
                    r1_arr,
                    r2_arr,
                    max_len1,
                    max_len2,
                    inst.dist,
                    inst.demands,
                    inst.capacity,
                    inst.ready_times,
                    inst.due_times,
                    inst.service_times,
                    old_pair,
                    heatmap,
                    pruning_threshold,
                )
            else:
                mkey = (3, route_keys[i], route_keys[j])
                hit = scan_memo.get(mkey)
                if hit is None:
                    old_pair = float(_route_cost(r1_arr, inst.dist)) + float(_route_cost(r2_arr, inst.dist))
                    max_len1 = 4 if len(r1) >= 20 else (3 if len(r1) >= 12 else 2)
                    max_len2 = 4 if len(r2) >= 20 else (3 if len(r2) >= 12 else 2)
                    hit = _cross_exchange_pair_numba(
                        r1_arr,
                        r2_arr,
                        max_len1,
                        max_len2,
                        inst.dist,
                        inst.demands,
                        inst.capacity,
                        inst.ready_times,
                        inst.due_times,
                        inst.service_times,
                        old_pair,
                    )
                    scan_memo[mkey] = hit
                delta, p1, len1, p2, len2 = hit
            if p1 >= 0 and delta < best_delta:
                seg2 = list(r2[p2 : p2 + len2])
                seg1 = list(r1[p1 : p1 + len1])
                nr1 = r1[:p1] + seg2 + r1[p1 + len1 :]
                nr2 = r2[:p2] + seg1 + r2[p2 + len2 :]
                routes = [r[:] for r in plan.routes]
                routes[i], routes[j] = nr1, nr2
                best_routes = routes
                best_delta = delta
    if best_routes is not None:
        cand = Plan(best_routes, inst, plan.algo)
        cand._cost = plan.cost + best_delta
        cand._ok = True
        return cand
    return None


def _cross_tail(
    plan: Plan,
    nv_ceiling: int | None = None,
    heatmap: np.ndarray | None = None,
    pruning_threshold: float = 0.0,
    cache: _PlanCache | None = None,
) -> Plan | None:
    inst = plan.inst
    if nv_ceiling is not None and plan.nv > nv_ceiling:
        return None
    max_dist = max(inst.max_dist, 1.0)
    best_delta = -1e-9
    best_routes: list[list[int]] | None = None
    best_cost_delta = 0.0

    if cache is None:
        cache = _PlanCache.from_plan(plan)

    route_arrays = cache.route_arrays
    centroid_sqdist = cache.centroid_sqdist
    route_sets = cache.route_sets
    route_neighbors = cache.route_neighbors
    route_keys = cache.route_keys
    scan_memo = cache.scan_memo
    pair_thresh_sq = (0.65 * max_dist) ** 2

    for i in range(len(plan.routes)):
        for j in range(i + 1, len(plan.routes)):
            r1, r2 = plan.routes[i], plan.routes[j]
            if not r1 or not r2:
                continue

            # Neighbor list pruning: skip route pairs with no neighboring customers
            if route_sets[i].isdisjoint(route_neighbors[j]) and route_sets[j].isdisjoint(route_neighbors[i]):
                continue

            # Spatial filter: skip route pairs whose centroids are too far apart
            if centroid_sqdist[i, j] > pair_thresh_sq:
                continue

            r1_arr, r2_arr = route_arrays[i], route_arrays[j]

            if heatmap is not None and pruning_threshold > 0.0:
                old_pair = float(_route_cost(r1_arr, inst.dist)) + float(_route_cost(r2_arr, inst.dist))
                delta, split_i, split_j = _cross_tail_pair_pruned_numba(
                    r1_arr,
                    r2_arr,
                    inst.dist,
                    inst.demands,
                    inst.capacity,
                    inst.ready_times,
                    inst.due_times,
                    inst.service_times,
                    old_pair,
                    heatmap,
                    pruning_threshold,
                )
            else:
                mkey = (4, route_keys[i], route_keys[j])
                hit = scan_memo.get(mkey)
                if hit is None:
                    old_pair = float(_route_cost(r1_arr, inst.dist)) + float(_route_cost(r2_arr, inst.dist))
                    hit = _cross_tail_pair_numba(
                        r1_arr,
                        r2_arr,
                        inst.dist,
                        inst.demands,
                        inst.capacity,
                        inst.ready_times,
                        inst.due_times,
                        inst.service_times,
                        old_pair,
                    )
                    scan_memo[mkey] = hit
                delta, split_i, split_j = hit

            if split_i >= 0:
                # A vehicle is eliminated if either tail swap makes a route empty.
                len1_new = split_i + (len(r2) - split_j)
                len2_new = split_j + (len(r1) - split_i)
                reduced = (len1_new == 0) or (len2_new == 0)

                new_nv = plan.nv - (1 if reduced else 0)
                if nv_ceiling is not None and new_nv > nv_ceiling:
                    continue

                actual_delta = delta
                if new_nv < plan.nv:
                    actual_delta -= 1000.0

                if actual_delta < best_delta:
                    best_delta = actual_delta
                    nr1 = list(r1[:split_i]) + list(r2[split_j:])
                    nr2 = list(r2[:split_j]) + list(r1[split_i:])
                    routes = [r[:] for r in plan.routes]
                    routes[i], routes[j] = nr1, nr2
                    routes = [r for r in routes if r]
                    best_routes = routes
                    best_cost_delta = delta

    if best_routes is not None:
        cand = Plan(best_routes, inst, plan.algo)
        cand._cost = plan.cost + best_cost_delta
        cand._ok = True
        return cand
    return None


def _try_route_compact(plan: Plan, nv_ceiling: int | None = None) -> Plan | None:
    if len(plan.routes) <= 1:
        return None
    inst = plan.inst
    ranked = sorted(
        range(len(plan.routes)),
        key=lambda i: (len(plan.routes[i]), _route_load(plan.routes[i], inst), _route_cost_list(plan.routes[i], inst)),
    )
    for ridx in ranked:
        source = plan.routes[ridx]
        others = [r[:] for i, r in enumerate(plan.routes) if i != ridx]
        ok = True
        for node in sorted(source, key=lambda n: (inst.due_times[n] - inst.ready_times[n], -inst.demands[n])):
            best_c, best_r, best_p = float("inf"), None, None
            for oi, route in enumerate(others):
                delta, pos = _best_insert_position(node, route, inst)
                if pos is not None and delta < best_c:
                    best_c, best_r, best_p = delta, oi, pos
            if best_r is None:
                ok = False
                break
            others[best_r].insert(best_p, node)
        if not ok:
            continue
        cand = Plan([r for r in others if r], inst, plan.algo)
        if not cand.feasible:
            continue
        if nv_ceiling is not None and cand.nv > nv_ceiling:
            continue
        if cand.dominates(plan) or (cand.nv == plan.nv and cand.cost + 1e-9 < plan.cost):
            return cand
    return None


def _best_or_opt(
    plan: Plan,
    nv_ceiling: int | None = None,
    heatmap: np.ndarray | None = None,
    pruning_threshold: float = 0.0,
    cache: _PlanCache | None = None,
):
    inst = plan.inst
    best_delta, best_move = -1e-9, None
    best_cost_delta = 0.0
    max_dist = max(inst.max_dist, 1.0)

    if cache is None:
        cache = _PlanCache.from_plan(plan)

    route_arrays = cache.route_arrays
    centroid_sqdist = cache.centroid_sqdist
    route_sets = cache.route_sets
    route_neighbors = cache.route_neighbors
    route_keys = cache.route_keys
    scan_memo = cache.scan_memo
    pair_thresh_sq = (0.65 * max_dist) ** 2

    for si in range(len(plan.routes)):
        for di in range(si + 1, len(plan.routes)):
            # Neighbor list pruning: skip route pairs with no neighboring customers
            if route_sets[si].isdisjoint(route_neighbors[di]) and route_sets[di].isdisjoint(route_neighbors[si]):
                continue
            # Spatial filter: skip route pairs whose centroids are too far apart
            if centroid_sqdist[si, di] > pair_thresh_sq:
                continue

            r1_arr, r2_arr = route_arrays[si], route_arrays[di]

            if heatmap is not None and pruning_threshold > 0.0:
                old_pair = float(_route_cost(r1_arr, inst.dist)) + float(_route_cost(r2_arr, inst.dist))
                delta, direction, p1, L, p2, rev = _string_relocate_pair_pruned_numba(
                    r1_arr,
                    r2_arr,
                    inst.dist,
                    inst.demands,
                    inst.capacity,
                    inst.ready_times,
                    inst.due_times,
                    inst.service_times,
                    old_pair,
                    heatmap,
                    pruning_threshold,
                )
            else:
                mkey = (5, route_keys[si], route_keys[di])
                hit = scan_memo.get(mkey)
                if hit is None:
                    old_pair = float(_route_cost(r1_arr, inst.dist)) + float(_route_cost(r2_arr, inst.dist))
                    hit = _string_relocate_pair_numba(
                        r1_arr,
                        r2_arr,
                        inst.dist,
                        inst.demands,
                        inst.capacity,
                        inst.ready_times,
                        inst.due_times,
                        inst.service_times,
                        old_pair,
                    )
                    scan_memo[mkey] = hit
                delta, direction, p1, L, p2, rev = hit

            if direction != -1:
                if direction == 1:
                    # Relocate from si to di
                    source_idx, dest_idx = si, di
                else:
                    # Relocate from di to si
                    source_idx, dest_idx = di, si

                sn_empty = len(plan.routes[source_idx]) == L
                new_nv = plan.nv - (1 if sn_empty else 0)
                if nv_ceiling is not None and new_nv > nv_ceiling:
                    continue

                actual_delta = delta
                if new_nv < plan.nv:
                    actual_delta -= 1000.0

                if actual_delta < best_delta:
                    best_delta = actual_delta
                    best_move = (source_idx, int(p1), int(L), dest_idx, int(p2), bool(rev))
                    best_cost_delta = delta

    if best_move is None:
        return None
    return best_move, best_cost_delta


def _apply_or_opt(plan: Plan, move: tuple[int, int, int, int, int, bool]) -> Plan:
    si, sp, L, di, ip, is_reversed = move
    routes = [r[:] for r in plan.routes]
    seg = routes[si][sp : sp + L]
    del routes[si][sp : sp + L]
    if si < di and len(routes[si]) == 0:
        di -= 1
    routes = [r for r in routes if r]
    if is_reversed:
        seg = list(reversed(seg))
    routes[di] = routes[di][:ip] + seg + routes[di][ip:]
    return Plan(routes, plan.inst, plan.algo)


def _merge_orderings(r1: list[int], r2: list[int], inst: Inst) -> list[list[int]]:
    combined = list(r1) + list(r2)
    depot = 0
    center = inst.coords[np.array(combined, dtype=np.int64)].mean(axis=0)
    polar = sorted(
        combined,
        key=lambda n: np.arctan2(inst.coords[n][1] - center[1], inst.coords[n][0] - center[0]),
    )
    nearest_seed = min(combined, key=lambda n: inst.dist[depot, n])
    farthest_seed = max(combined, key=lambda n: inst.dist[depot, n])

    orderings = [
        sorted(combined, key=lambda n: (inst.due_times[n], inst.ready_times[n], inst.dist[depot, n])),
        sorted(combined, key=lambda n: (inst.ready_times[n], inst.due_times[n], inst.dist[depot, n])),
        sorted(combined, key=lambda n: inst.dist[depot, n]),
        sorted(combined, key=lambda n: inst.dist[depot, n], reverse=True),
        polar,
        list(reversed(polar)),
        sorted(combined, key=lambda n: (inst.dist[nearest_seed, n], inst.due_times[n])),
        sorted(combined, key=lambda n: (inst.dist[farthest_seed, n], inst.due_times[n])),
    ]

    deduped: list[list[int]] = []
    seen: set[tuple[int, ...]] = set()
    for ordering in orderings:
        key = tuple(ordering)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(ordering)
    return deduped


def _build_merged_route(ordering: list[int], inst: Inst) -> list[int] | None:
    route: list[int] = []
    for node in ordering:
        _, pos = _best_insert_position(node, route, inst)
        if pos is None:
            return None
        route.insert(pos, node)
    return route if _check_route(route, inst) else None


def _ranked_merge_pairs(plan: Plan, max_pairs: int) -> list[tuple[int, int]]:
    inst = plan.inst
    route_stats = []
    for i, route in enumerate(plan.routes):
        route_stats.append((i, _route_load(route, inst), _route_cost_list(route, inst), len(route)))

    candidates: list[tuple[float, int, int]] = []
    for pos_i, (i, load_i, cost_i, len_i) in enumerate(route_stats):
        for j, load_j, cost_j, len_j in route_stats[pos_i + 1 :]:
            if load_i + load_j > inst.capacity:
                continue
            depot_saving = inst.dist[0, plan.routes[i][0]] + inst.dist[plan.routes[i][-1], 0]
            depot_saving += inst.dist[0, plan.routes[j][0]] + inst.dist[plan.routes[j][-1], 0]
            size_bias = len_i + len_j
            pair_score = cost_i + cost_j - 0.15 * depot_saving - 2.0 * size_bias
            candidates.append((float(pair_score), i, j))
    candidates.sort(key=lambda x: x[0])
    return [(i, j) for _, i, j in candidates[:max_pairs]]


def merged_route_candidates(plan: Plan, max_pairs: int = 24, max_routes: int = 32) -> list[list[int]]:
    if len(plan.routes) <= 1:
        return []

    inst = plan.inst
    candidates: list[list[int]] = []
    seen: set[tuple[int, ...]] = set()
    for i, j in _ranked_merge_pairs(plan, max_pairs=max_pairs):
        for ordering in _merge_orderings(plan.routes[i], plan.routes[j], inst):
            merged_route = _build_merged_route(ordering, inst)
            if merged_route is None:
                continue
            key = tuple(merged_route)
            if key in seen:
                continue
            seen.add(key)
            candidates.append(merged_route)
            if len(candidates) >= max_routes:
                return candidates
    return candidates


def _try_route_merge(plan: Plan, max_pairs: int = 24) -> Plan | None:
    inst = plan.inst
    if len(plan.routes) <= 1:
        return None

    best_merged: Plan | None = None

    for i, j in _ranked_merge_pairs(plan, max_pairs=max_pairs):
        r1 = plan.routes[i]
        r2 = plan.routes[j]
        for ordering in _merge_orderings(r1, r2, inst):
            merged_route = _build_merged_route(ordering, inst)
            if merged_route is None:
                continue

            surviving = [r[:] for k, r in enumerate(plan.routes) if k != i and k != j]
            surviving.append(merged_route)
            cand = Plan(surviving, inst, plan.algo)
            if not cand.feasible:
                continue
            if best_merged is None or cand.dominates(best_merged) or cand.nv < best_merged.nv:
                best_merged = cand

    return best_merged


def _covers_all_customers(routes: list[list[int]], inst: Inst) -> bool:
    nodes = [n for route in routes for n in route]
    return len(nodes) == inst.n and len(set(nodes)) == inst.n and all(1 <= n <= inst.n for n in nodes)


def _insertion_options(
    node: int,
    routes: list[list[int]],
    inst: Inst,
    max_options: int = 3,
) -> list[tuple[float, int, int]]:
    options: list[tuple[float, int, int]] = []
    for ri, route in enumerate(routes):
        delta, pos = _best_insert_position(node, route, inst)
        if pos is not None:
            options.append((float(delta), ri, pos))
    options.sort(key=lambda x: x[0])
    return options[:max_options]


def _select_buffered_pending_node(pending: list[int], routes: list[list[int]], inst: Inst) -> int:
    best_idx = 0
    best_key: tuple[float, ...] | None = None
    for idx, node in enumerate(pending):
        options = 0
        best_delta = float("inf")
        for route in routes:
            delta, pos = _best_insert_position(node, route, inst)
            if pos is not None:
                options += 1
                best_delta = min(best_delta, float(delta))
        width = float(inst.due_times[node] - inst.ready_times[node])
        key = (float(options), width, float(inst.due_times[node]), -float(inst.demands[node]), best_delta)
        if best_key is None or key < best_key:
            best_key = key
            best_idx = idx
    return best_idx


def _buffered_ejection_options(
    node: int,
    routes: list[list[int]],
    inst: Inst,
    max_options: int = 3,
) -> list[tuple[float, int, list[int], int]]:
    options: list[tuple[float, int, list[int], int]] = []
    max_dist = max(inst.max_dist, 1.0)
    horizon = max(inst.horizon, 1.0)

    for ri, route in enumerate(routes):
        old_cost = _route_cost_list(route, inst)
        for eject_pos, ejected in enumerate(route):
            stripped = route[:eject_pos] + route[eject_pos + 1 :]
            _, pos_node = _best_insert_position(node, stripped, inst)
            if pos_node is None:
                continue

            new_route = stripped[:pos_node] + [node] + stripped[pos_node:]
            if not _check_route(new_route, inst):
                continue

            trial_routes = [r[:] for r in routes]
            trial_routes[ri] = new_route
            landing = _insertion_options(ejected, trial_routes, inst, max_options=1)
            if landing:
                landing_score = landing[0][0]
                orphan_penalty = 0.0
            else:
                landing_score = max_dist
                orphan_penalty = max_dist

            width_penalty = (inst.due_times[ejected] - inst.ready_times[ejected]) / horizon
            route_delta = _route_cost_list(new_route, inst) - old_cost
            score = float(route_delta + 0.35 * landing_score + orphan_penalty + 0.05 * width_penalty * max_dist)
            options.append((score, ri, new_route, ejected))

    options.sort(key=lambda x: x[0])
    return options[:max_options]


def _try_buffered_route_elimination(
    plan: Plan,
    target_idx: int,
    max_ejections: int = 4,
    beam_width: int = 8,
) -> Plan | None:
    if len(plan.routes) <= 1:
        return None

    inst = plan.inst
    target = sorted(
        plan.routes[target_idx],
        key=lambda n: (inst.due_times[n] - inst.ready_times[n], inst.due_times[n], -inst.demands[n]),
    )
    routes = [r[:] for i, r in enumerate(plan.routes) if i != target_idx]
    states: list[tuple[list[list[int]], list[int], int, float]] = [(routes, target, 0, 0.0)]
    max_steps = len(target) + max_ejections

    def finish_if_complete(routes_: list[list[int]], pending_: list[int]) -> Plan | None:
        if pending_:
            return None
        cand = Plan([r for r in routes_ if r], inst, plan.algo)
        if cand.nv == plan.nv - 1 and cand.feasible and _covers_all_customers(cand.routes, inst):
            return cand
        return None

    for _ in range(max_steps):
        next_states: list[tuple[list[list[int]], list[int], int, float]] = []
        for routes_cur, pending, ejections, score in states:
            completed = finish_if_complete(routes_cur, pending)
            if completed is not None:
                return completed
            if not pending:
                continue

            node_idx = _select_buffered_pending_node(pending, routes_cur, inst)
            node = pending[node_idx]
            rest = pending[:node_idx] + pending[node_idx + 1 :]

            for delta, ri, pos in _insertion_options(node, routes_cur, inst):
                new_routes = [r[:] for r in routes_cur]
                new_routes[ri] = new_routes[ri][:pos] + [node] + new_routes[ri][pos:]
                next_states.append((new_routes, rest[:], ejections, score + delta))

            if ejections >= max_ejections:
                continue

            for eject_score, ri, new_route, ejected in _buffered_ejection_options(node, routes_cur, inst):
                new_routes = [r[:] for r in routes_cur]
                new_routes[ri] = new_route
                next_states.append((new_routes, rest + [ejected], ejections + 1, score + eject_score))

        if not next_states:
            break

        deduped: list[tuple[list[list[int]], list[int], int, float]] = []
        seen: set[tuple[tuple[tuple[int, ...], ...], tuple[int, ...]]] = set()
        for item in sorted(next_states, key=lambda s: (len(s[1]), s[2], s[3])):
            routes_cur, pending, _, _ = item
            signature = (tuple(tuple(route) for route in routes_cur), tuple(sorted(pending)))
            if signature in seen:
                continue
            seen.add(signature)
            deduped.append(item)
            if len(deduped) >= beam_width:
                break
        states = deduped

    for routes_cur, pending, _, _ in states:
        completed = finish_if_complete(routes_cur, pending)
        if completed is not None:
            return completed
    return None


def _ges_perturb(routes: list[list[int]], inst: Inst, n_moves: int) -> None:
    """Random feasible relocations that reshape the partial solution in place.

    This is the escape hatch GES needs when no single ejection admits the
    pending customer: the beam search this complements simply gave up here.
    Moves are feasibility-checked and applied regardless of cost — the goal is
    reachability, not improvement.
    """
    import random as _random

    for _ in range(n_moves):
        non_empty = [i for i, r in enumerate(routes) if r]
        if len(non_empty) < 2:
            return
        si = _random.choice(non_empty)
        sp = _random.randrange(len(routes[si]))
        node = routes[si][sp]
        di = _random.choice([i for i in non_empty if i != si])
        trial = routes[si][:sp] + routes[si][sp + 1 :]
        cost, pos = _best_insert_position(node, routes[di], inst)
        if pos is None:
            continue
        routes[si] = trial
        routes[di] = routes[di][:pos] + [node] + routes[di][pos:]


def _guided_ejection_search(
    plan: Plan,
    target_idx: int,
    p_counters: dict[int, int],
    max_steps: int | None = None,
    k_max: int = 2,
    n_perturb: int = 20,
    deadline: float | None = None,
) -> Plan | None:
    """Guided Ejection Search route elimination (Nagata & Bräysy 2009, adapted).

    Four ingredients the beam search in :func:`_try_buffered_route_elimination`
    lacks, which is why this runs where the beam has already failed:

    * a LIFO **ejection pool** of unplaced customers,
    * per-customer **penalty counters** ``p_counters`` that persist across
      targets — customers that keep getting ejected become expensive to eject,
    * ejections chosen to **minimise the summed penalty** of the ejected set,
    * a **perturbation phase** when no ejection is admissible, instead of
      giving up.

    Returns a feasible plan with one route fewer, or None.
    """
    inst = plan.inst
    if len(plan.routes) <= 1:
        return None
    routes = [r[:] for i, r in enumerate(plan.routes) if i != target_idx]
    pool_stack = list(plan.routes[target_idx])
    if max_steps is None:
        max_steps = min(1500, 60 * max(len(pool_stack), 1))

    nbrs_of = inst.neighbors_k
    steps = 0
    while pool_stack:
        if steps >= max_steps:
            return None
        if deadline is not None and time.time() >= deadline:
            return None
        steps += 1
        v = pool_stack.pop()

        # 1) Plain feasible insertion, cheapest across all routes.
        best_delta, best_ri, best_pos = float("inf"), -1, -1
        for ri, r in enumerate(routes):
            if not r:
                continue
            cost, pos = _best_insert_position(v, r, inst)
            if pos is not None and cost < best_delta:
                best_delta, best_ri, best_pos = cost, ri, pos
        if best_ri >= 0:
            routes[best_ri] = routes[best_ri][:best_pos] + [v] + routes[best_ri][best_pos:]
            continue

        # 2) Ejection: replace 1..k_max consecutive customers by v, choosing the
        #    set with minimal summed penalty (ties: fewer ejected customers).
        v_nbrs = set(nbrs_of[v]) if v < len(nbrs_of) else set()
        best_score: tuple[int, int] | None = None
        best_apply = None
        for ri, r in enumerate(routes):
            if not r or (v_nbrs and v_nbrs.isdisjoint(r)):
                continue
            for start in range(len(r)):
                for k in range(1, min(k_max, len(r) - start) + 1):
                    cand = r[:start] + [v] + r[start + k :]
                    if not _check_route(cand, inst):
                        continue
                    ejected = r[start : start + k]
                    score = (sum(p_counters.get(w, 1) for w in ejected), k)
                    if best_score is None or score < best_score:
                        best_score = score
                        best_apply = (ri, cand, ejected)
        if best_apply is not None:
            ri, cand, ejected = best_apply
            routes[ri] = cand
            pool_stack.extend(ejected)  # LIFO: most recently ejected pops first
            p_counters[v] = p_counters.get(v, 1) + 1
            continue

        # 3) No admissible ejection: perturb and retry this customer.
        _ges_perturb(routes, inst, n_perturb)
        pool_stack.append(v)

    cand_plan = Plan([r for r in routes if r], inst, plan.algo)
    if cand_plan.nv == plan.nv - 1 and cand_plan.feasible and _covers_all_customers(cand_plan.routes, inst):
        return cand_plan
    return None


def _buffered_route_elimination(
    plan: Plan,
    max_rounds: int = 2,
    max_ejections: int = 6,
    beam_width: int = 16,
    pool=None,
    hard_mode: bool = False,
    deadline: float | None = None,
) -> Plan:
    if hard_mode:
        beam_width = max(beam_width * 2, 32)
        max_ejections = max(max_ejections + 4, 10)
    best = plan.copy()
    ges_penalties: dict[int, int] = {}  # persists across targets and rounds
    for _ in range(max_rounds):
        if len(best.routes) <= 1:
            break
        ranked = sorted(
            range(len(best.routes)),
            key=lambda i: (
                len(best.routes[i]),
                _route_load(best.routes[i], best.inst),
                _route_cost_list(best.routes[i], best.inst),
            ),
        )
        # Try all routes as targets when the smallest route has ≤ 4 customers
        # (signals the instance is close to an NV drop, e.g. RC101's [61,81,90])
        smallest_len = len(best.routes[ranked[0]]) if ranked else 0
        n_targets = len(ranked) if smallest_len <= 4 else min(6, len(ranked))
        improved = False
        # GES runs on every target the beam gives up on, and the beam failing is
        # the common case: a measured run saw 18/18 (n=200) and 35/35 (n=400)
        # beam failures, at ~130-190ms per GES call. When the smallest route is
        # tiny, `n_targets` above becomes one-per-route, so an unbudgeted round
        # costs O(R) GES calls — and with `deadline is None` (the default, and
        # what --no-time-limit and scripts/capture_golden.py produce) nothing
        # else bounds it. Measured maxima were 6 (n=200) and 23 (n=400) GES
        # calls per round, so this ceiling does not bind at those scales; it
        # caps the O(R) growth past them.
        ges_budget = max(6, min(24, len(best.routes)))
        ges_used = 0
        for target_idx in ranked[:n_targets]:
            if deadline is not None and time.time() >= deadline:
                break
            local_ejections = max(2, min(max_ejections, len(best.routes[target_idx]) // 2 + 1))
            cand = _try_buffered_route_elimination(
                best,
                target_idx,
                max_ejections=local_ejections,
                beam_width=beam_width,
            )
            if cand is None and ges_used < ges_budget:
                # Guided Ejection Search picks up exactly where the beam gave
                # up: LIFO pool + penalty-guided ejections + perturbation.
                ges_used += 1
                cand = _guided_ejection_search(best, target_idx, ges_penalties, deadline=deadline)
            if cand is None:
                continue
            cand = local_search(cand, max_passes=1, nv_ceiling=cand.nv, max_ls_moves=10, pool=pool)
            if cand.feasible and cand.nv < best.nv and _covers_all_customers(cand.routes, best.inst):
                best = cand
                if pool is not None:
                    pool.add_plan(best)
                improved = True
                break
        if not improved:
            break
    return best


def _intra_route_or_opt(plan: Plan, nv_ceiling: int | None = None) -> Plan | None:
    """
    Intra-Route Or-Opt — JIT-accelerated.
    Moves customer segments (length 1-3) to better positions within
    the SAME route to minimise total distance (TD).
    """
    inst = plan.inst
    improved = False
    routes: list[list[int]] = []
    for route in plan.routes:
        if len(route) < 4:
            routes.append(route[:])
            continue
        route_arr = np.array(route, dtype=np.int64)
        result, imp = _or_opt_intra_numba(
            route_arr,
            inst.dist,
            inst.demands,
            inst.capacity,
            inst.ready_times,
            inst.due_times,
            inst.service_times,
        )
        if imp:
            improved = True
        routes.append(list(result))
    if not improved:
        return None
    cand = Plan(routes, inst, plan.algo)
    cand._ok = True
    if nv_ceiling is not None and cand.nv > nv_ceiling:
        return None
    return cand if cand.dominates(plan) else None


def _intra_route_optimize(route: list[int], inst: Inst, max_passes: int = 25) -> list[int]:
    """
    Run 2-opt and or-opt(1,2,3) on a single route until convergence.
    Fully JIT-compiled — designed for wide-TW instances where routes
    carry 25+ customers.
    """
    if len(route) < 4:
        return route[:]
    route_arr = np.array(route, dtype=np.int64)
    result = _intra_route_optimize_numba(
        route_arr,
        inst.dist,
        inst.demands,
        inst.capacity,
        inst.ready_times,
        inst.due_times,
        inst.service_times,
        max_passes,
    )
    return list(result)


def td_converge_polish(plan: Plan, max_passes: int = 25, deadline: float | None = None) -> Plan:
    """
    Apply _intra_route_optimize to every route independently.
    Called during the BKS-NV TD polish phase. Unlike local_search(),
    this never modifies route assignments — only improves sequence quality.

    ``deadline`` (absolute ``time.time()`` value) makes this anytime-safe:
    routes not yet optimised when the deadline passes are kept as-is. Without
    it this phase was a measured contributor to the 8% budget overrun at n=1000.
    """
    routes = []
    for r in plan.routes:
        if deadline is not None and time.time() >= deadline:
            routes.append(r[:])
        else:
            routes.append(_intra_route_optimize(r, plan.inst, max_passes))
    cand = Plan(routes, plan.inst, plan.algo)
    return cand if cand.feasible and cand.cost + 1e-9 < plan.cost else plan


def td_inter_route_polish(
    plan: Plan,
    max_passes: int = 8,
    deadline: float | None = None,
    heatmap: np.ndarray | None = None,
) -> Plan:
    """
    Convergent inter-route travel distance optimization.
    Unlike local_search(), this has NO move cap per pass.
    Iteratively executes inter-route local search operators (Relocate, Or-Opt,
    Swap 1-1, Swap 2-1, Cross-Tail, Cross-Exchange) until local minimum.
    """
    if not plan.feasible:
        return plan
    best = plan.copy()
    cache = None

    for _ in range(max_passes):
        if deadline is not None and time.time() >= deadline:
            break
        improved = False

        routes = []
        for route in best.routes:
            nr = _two_opt_best(route, best.inst)
            routes.append(nr)
            if nr != route:
                improved = True
        if improved:
            best = Plan(routes, best.inst, best.algo)
            best._ok = True

        while True:
            if deadline is not None and time.time() >= deadline:
                break
            cache = _PlanCache.from_plan(best, prev=cache)
            step_improved = False

            res = _best_relocate(best, nv_ceiling=best.nv, heatmap=heatmap, cache=cache)
            if res is not None:
                move, cost_delta = res
                cand = _apply_relocate(best, move)
                cand._cost = best.cost + cost_delta
                cand._ok = True
                if cand.nv <= best.nv and cand.cost + 1e-9 < best.cost:
                    best, improved, step_improved = cand, True, True
                    continue

            res = _best_or_opt(best, nv_ceiling=best.nv, heatmap=heatmap, cache=cache)
            if res is not None:
                move, cost_delta = res
                cand = _apply_or_opt(best, move)
                cand._cost = best.cost + cost_delta
                cand._ok = True
                if cand.nv <= best.nv and cand.cost + 1e-9 < best.cost:
                    best, improved, step_improved = cand, True, True
                    continue

            intra = _intra_route_or_opt(best, nv_ceiling=best.nv)
            if intra is not None and intra.cost + 1e-9 < best.cost:
                best, improved, step_improved = intra, True, True
                continue

            res = _best_swap(best, heatmap=heatmap, cache=cache)
            if res is not None:
                move, cost_delta = res
                cand = _apply_swap(best, move)
                cand._cost = best.cost + cost_delta
                cand._ok = True
                if cand.cost + 1e-9 < best.cost:
                    best, improved, step_improved = cand, True, True
                    continue

            res21 = _best_swap_21(best, heatmap=heatmap, cache=cache)
            if res21 is not None:
                move, cost_delta = res21
                cand = _apply_swap_21(best, move)
                cand._cost = best.cost + cost_delta
                cand._ok = True
                if cand.cost + 1e-9 < best.cost:
                    best, improved, step_improved = cand, True, True
                    continue

            cross_tail = _cross_tail(best, nv_ceiling=best.nv, heatmap=heatmap, cache=cache)
            if cross_tail is not None and cross_tail.cost + 1e-9 < best.cost:
                best, improved, step_improved = cross_tail, True, True
                continue

            cross = _cross_exchange(best, nv_ceiling=best.nv, heatmap=heatmap, cache=cache)
            if cross is not None and cross.cost + 1e-9 < best.cost:
                best, improved, step_improved = cross, True, True
                continue

            if not step_improved:
                break

        if not improved:
            break

    return best


def local_search(
    plan: Plan,
    max_passes: int = 1,
    nv_ceiling: int | None = None,
    max_ls_moves: int = 5,
    pool=None,
    heatmap: np.ndarray | None = None,
    pruning_threshold: float = 0.0,
) -> Plan:
    if not plan.feasible:
        return plan
    best = plan.copy()

    cache = None
    for _ in range(max_passes):
        improved = False
        routes = []
        for route in best.routes:
            nr = _two_opt_best(route, best.inst)
            routes.append(nr)
            if nr != route:
                improved = True
        if improved:
            best = Plan(routes, best.inst, best.algo)
            best._ok = True

        moves = 0
        while moves < max_ls_moves:
            # Passing the previous iteration's cache reuses per-route data for
            # every route the last move didn't touch.
            cache = _PlanCache.from_plan(best, prev=cache)
            # 1. Relocate Move
            res = _best_relocate(
                best, nv_ceiling=nv_ceiling, heatmap=heatmap, pruning_threshold=pruning_threshold, cache=cache
            )
            if res is not None:
                move, cost_delta = res
                cand = _apply_relocate(best, move)
                cand._cost = best.cost + cost_delta
                cand._ok = True
                if cand.dominates(best) or (cand.nv == best.nv and cand.cost + 1e-9 < best.cost):
                    best, improved = cand, True
                    moves += 1
                    if pool is not None:
                        pool.add_plan(best)  # Immediate hot-commit to active pool
                    continue

            # 1.5. String Relocate (Or-Opt Move)
            res = _best_or_opt(
                best, nv_ceiling=nv_ceiling, heatmap=heatmap, pruning_threshold=pruning_threshold, cache=cache
            )
            if res is not None:
                move, cost_delta = res
                cand = _apply_or_opt(best, move)
                cand._cost = best.cost + cost_delta
                cand._ok = True
                if cand.dominates(best) or (cand.nv == best.nv and cand.cost + 1e-9 < best.cost):
                    best, improved = cand, True
                    moves += 1
                    if pool is not None:
                        pool.add_plan(best)
                    continue

            # 2. Intra-Route Or-Opt
            intra = _intra_route_or_opt(best, nv_ceiling=nv_ceiling)
            if intra is not None:
                best, improved = intra, True
                moves += 1
                if pool is not None:
                    pool.add_plan(best)
                continue

            # 3. Swap Move
            res = _best_swap(best, heatmap=heatmap, pruning_threshold=pruning_threshold, cache=cache)
            if res is not None:
                move, cost_delta = res
                cand = _apply_swap(best, move)
                cand._cost = best.cost + cost_delta
                cand._ok = True
                if cand.cost + 1e-9 < best.cost:
                    best, improved = cand, True
                    moves += 1
                    if pool is not None:
                        pool.add_plan(best)
                    continue

            # 3.25. Swap(2,1) Move
            res21 = _best_swap_21(best, heatmap=heatmap, pruning_threshold=pruning_threshold, cache=cache)
            if res21 is not None:
                move, cost_delta = res21
                cand = _apply_swap_21(best, move)
                cand._cost = best.cost + cost_delta
                cand._ok = True
                if cand.cost + 1e-9 < best.cost:
                    best, improved = cand, True
                    moves += 1
                    if pool is not None:
                        pool.add_plan(best)
                    continue

            # 3.5. CROSS Tail-Swap
            cross_tail = _cross_tail(
                best, nv_ceiling=nv_ceiling, heatmap=heatmap, pruning_threshold=pruning_threshold, cache=cache
            )
            if cross_tail is not None:
                best, improved = cross_tail, True
                moves += 1
                if pool is not None:
                    pool.add_plan(best)
                continue

            # 4. Cross Exchange
            cross = _cross_exchange(
                best, nv_ceiling=nv_ceiling, heatmap=heatmap, pruning_threshold=pruning_threshold, cache=cache
            )
            if cross is not None:
                best, improved = cross, True
                moves += 1
                if pool is not None:
                    pool.add_plan(best)
                continue

            # 5. Route Compaction
            compact = _try_route_compact(best, nv_ceiling=nv_ceiling)
            if compact is not None:
                best, improved = compact, True
                moves += 1
                if pool is not None:
                    pool.add_plan(best)
                continue
            break

        if not improved:
            break

    # ── Post-Search Diversification Seeding (Executed exactly once at exit) ──
    if pool is not None:
        for route in merged_route_candidates(best):
            pool.add_route(route)
        merged = _try_route_merge(best)
        if merged is not None:
            pool.add_plan(merged)

    return best


def _try_chain_elimination(plan: Plan, target_idx: int, beam_width: int = 3, max_depth: int = 3) -> Plan | None:
    """
    Generalized ejection chain search: c -> Ri (displacing d) -> d -> Rj ... -> Rk.
    Falls back to direct insertion (depth 1), then depth 2, then depth 3, etc. up to max_depth.
    """
    inst = plan.inst
    target = plan.routes[target_idx]
    routes = [r[:] for i, r in enumerate(plan.routes) if i != target_idx]

    for c in sorted(target, key=lambda n: inst.due_times[n] - inst.ready_times[n]):
        # We search for a chain to insert customer `c`
        # active_states is a list of tuples: (current_routes, node, visited_route_indices, cost)
        active_states = [(routes, c, set(), 0.0)]
        placed_routes = None

        # Try increasing depth d from 1 to max_depth
        for d in range(1, max_depth + 1):
            completed_states_at_this_depth = []
            next_active_states = []

            for curr_routes, node, visited, cost in active_states:
                for rj, route in enumerate(curr_routes):
                    if rj in visited:
                        continue

                    # 1. Try direct insertion of node into curr_routes[rj]
                    if _capacity_ok(route + [node], inst):
                        delta, pos = _best_insert_position(node, route, inst)
                        if pos is not None:
                            new_routes = curr_routes[:]
                            new_routes[rj] = route[:pos] + [node] + route[pos:]
                            completed_states_at_this_depth.append((new_routes, cost + delta))

                    # 2. Try ejections from curr_routes[rj] if we can go deeper
                    if d < max_depth:
                        for eject_pos, eject_val in enumerate(route):
                            stripped = route[:eject_pos] + route[eject_pos + 1 :]
                            if not _capacity_ok(stripped + [node], inst):
                                continue
                            delta, pos = _best_insert_position(node, stripped, inst)
                            if pos is not None:
                                new_routes = curr_routes[:]
                                new_routes[rj] = stripped[:pos] + [node] + stripped[pos:]
                                next_active_states.append((new_routes, eject_val, visited | {rj}, cost + delta))

            # If we found any complete insertions at this depth, choose the best one
            if completed_states_at_this_depth:
                completed_states_at_this_depth.sort(key=lambda x: x[1])
                placed_routes, _ = completed_states_at_this_depth[0]
                break

            # Otherwise, sort and prune the next active states to keep only the top beam_width candidates
            if not next_active_states:
                break
            # Sort by cumulative delta cost
            next_active_states.sort(key=lambda x: x[3])
            active_states = next_active_states[:beam_width]

        if placed_routes is not None:
            # Successfully placed customer c! Update routes for the next customer in target
            routes = placed_routes
        else:
            # c could not be placed at any depth; target route cannot be eliminated
            return None

    cand = Plan([r for r in routes if r], inst, plan.algo)
    return cand if cand.feasible else None


def _ejection_chain_eliminate(plan: Plan, beam_width: int | None = None, max_depth: int | None = None) -> Plan | None:
    """
    Tries to eliminate up to the 4 smallest routes via depth-2/depth-3 ejection chains.
    Complements _iterative_route_elimination: handles cases where greedy
    insertion fails due to TW blocking but a 2-step chain resolves the conflict.
    """
    if len(plan.routes) <= 1:
        return None

    n = plan.inst.n
    if beam_width is None:
        beam_width = min(32, max(16, n // 6))
    if max_depth is None:
        max_depth = min(10, max(6, n // 15))

    ranked = sorted(
        range(len(plan.routes)),
        key=lambda i: (len(plan.routes[i]), sum(plan.inst.demands[n] for n in plan.routes[i])),
    )
    for target_idx in ranked[:4]:
        result = _try_chain_elimination(plan, target_idx, beam_width=beam_width, max_depth=max_depth)
        if result is not None:
            return result
    return None


def _iterative_route_elimination(
    plan: Plan,
    inst: Inst,
    max_rounds: int = 6,
    pool=None,  # RoutePool | None — seeds pool after each success
) -> Plan:
    best = plan.copy()
    for _ in range(max_rounds):
        if len(best.routes) <= 1:
            break
        sorted_idxs = sorted(
            range(len(best.routes)),
            key=lambda i: (
                len(best.routes[i]),
                _route_load(best.routes[i], inst),
                _route_cost_list(best.routes[i], inst),
            ),
        )
        eliminated = False
        for target_idx in sorted_idxs[:5]:
            target = best.routes[target_idx]
            others = [r[:] for i, r in enumerate(best.routes) if i != target_idx]
            ok = True
            for node in sorted(target, key=lambda n: inst.due_times[n] - inst.ready_times[n]):
                best_d, best_r, best_p = float("inf"), None, None
                for oi, route in enumerate(others):
                    delta, pos = _best_insert_position(node, route, inst)
                    if pos is not None and delta < best_d:
                        best_d, best_r, best_p = delta, oi, pos
                if best_r is None:
                    ok = False
                    break
                others[best_r].insert(best_p, node)
            if not ok:
                continue
            cand = Plan([r for r in others if r], inst, best.algo)
            if not cand.feasible:
                continue
            cand = local_search(cand, max_passes=2, nv_ceiling=cand.nv, pool=pool)
            if cand.feasible:
                if pool is not None:
                    pool.add_plan(cand)
                best = cand
                eliminated = True
                break
        if not eliminated:
            break
    return best


def exact_tsptw_cpsat(
    route: list[int] | np.ndarray,
    inst: Inst,
    time_limit_sec: float = 0.5,
    scale: int = 1000,
) -> tuple[list[int], float]:
    """Exact single-vehicle TSP-TW sequencing using OR-Tools CP-SAT.

    Guarantees mathematically optimal ordering for the given subset of customers.
    """
    route_list = [int(x) for x in route]
    m = len(route_list)
    arr_orig = np.asarray(route_list, dtype=np.int64)
    orig_cost = _route_cost(arr_orig, inst.dist)
    if m <= 2:
        if m == 2:
            rev = [route_list[1], route_list[0]]
            arr_rev = np.asarray(rev, dtype=np.int64)
            cost_rev = _route_cost(arr_rev, inst.dist)
            if cost_rev < orig_cost and _check_route(rev, inst):
                return rev, cost_rev
        return route_list, orig_cost

    try:
        from ortools.sat.python import cp_model
    except ImportError:
        return route_list, orig_cost

    node_to_orig = {0: 0, m + 1: 0}
    for idx, c in enumerate(route_list, start=1):
        node_to_orig[idx] = c

    model = cp_model.CpModel()
    time_vars = {}
    for i in range(m + 2):
        orig_id = node_to_orig[i]
        r_t = int(round(float(inst.ready_times[orig_id]) * scale))
        d_t = int(round(float(inst.due_times[orig_id]) * scale))
        time_vars[i] = model.NewIntVar(r_t, d_t, f"time_{i}")

    arcs = []
    arc_literals = {}

    for j in range(1, m + 1):
        c_j = node_to_orig[j]
        d = int(round(float(inst.dist[0, c_j]) * scale))
        lit = model.NewBoolVar(f"arc_0_{j}")
        arcs.append((0, j, lit))
        arc_literals[(0, j)] = (lit, d)

    for i in range(1, m + 1):
        c_i = node_to_orig[i]
        for j in range(1, m + 1):
            if i == j:
                continue
            c_j = node_to_orig[j]
            d = int(round(float(inst.dist[c_i, c_j]) * scale))
            if inst.ready_times[c_i] + inst.service_times[c_i] + inst.dist[c_i, c_j] <= inst.due_times[c_j]:
                lit = model.NewBoolVar(f"arc_{i}_{j}")
                arcs.append((i, j, lit))
                arc_literals[(i, j)] = (lit, d)

    for i in range(1, m + 1):
        c_i = node_to_orig[i]
        d = int(round(float(inst.dist[c_i, 0]) * scale))
        lit = model.NewBoolVar(f"arc_{i}_{m + 1}")
        arcs.append((i, m + 1, lit))
        arc_literals[(i, m + 1)] = (lit, d)

    dummy_lit = model.NewBoolVar("arc_end_start")
    arcs.append((m + 1, 0, dummy_lit))
    arc_literals[(m + 1, 0)] = (dummy_lit, 0)

    model.AddCircuit(arcs)

    for (u, v), (lit, dist_val) in arc_literals.items():
        if (u, v) == (m + 1, 0):
            continue
        orig_u = node_to_orig[u]
        serv_u = int(round(float(inst.service_times[orig_u]) * scale))
        model.Add(time_vars[v] >= time_vars[u] + serv_u + dist_val).OnlyEnforceIf(lit)

    model.Minimize(sum(dist_val * lit for (lit, dist_val) in arc_literals.values()))

    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = time_limit_sec
    solver.parameters.num_search_workers = 1
    status = solver.Solve(model)

    if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        next_node = {}
        for (u, v), (lit, _) in arc_literals.items():
            if solver.Value(lit) == 1:
                next_node[u] = v

        new_route = []
        curr = next_node.get(0)
        while curr is not None and curr != (m + 1):
            new_route.append(node_to_orig[curr])
            curr = next_node.get(curr)

        if len(new_route) == m and _check_route(new_route, inst):
            arr_new = np.asarray(new_route, dtype=np.int64)
            new_cost = _route_cost(arr_new, inst.dist)
            if new_cost < orig_cost - 1e-6:
                return new_route, new_cost

    return route_list, orig_cost


def refine_plan_cpsat(plan: Plan, time_limit_per_route: float = 0.5) -> Plan:
    """Applies exact TSP-TW sequencing across all routes in a Plan."""
    if not plan.feasible or not plan.routes:
        return plan

    inst = plan.inst
    improved_routes = []

    for route in plan.routes:
        new_route, new_c = exact_tsptw_cpsat(route, inst, time_limit_sec=time_limit_per_route)
        arr_orig = np.asarray(route, dtype=np.int64)
        orig_c = _route_cost(arr_orig, inst.dist)
        if new_c < orig_c - 1e-6:
            improved_routes.append(new_route)
        else:
            improved_routes.append(route)

    return Plan(improved_routes, inst, plan.algo)

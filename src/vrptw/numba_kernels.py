"""
JIT-compiled Numba kernels for VRPTW local-search hot paths.

Every function is decorated with ``@njit(cache=True)`` and operates
exclusively on NumPy arrays and scalars (no Python objects).
"""

from __future__ import annotations

import numpy as np
from numba import njit

from .core import _route_cost, _route_ok

# ──────────────────────────────────────────────────────────────────────
# 1.  2-opt (intra-route)
# ──────────────────────────────────────────────────────────────────────


@njit(cache=True, fastmath=True)
def _compute_route_forward_slack_numba(
    route_arr: np.ndarray,
    dist: np.ndarray,
    ready: np.ndarray,
    due: np.ndarray,
    service: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Compute (arrival, departure, wait, F_slack) for route_arr."""
    n = len(route_arr)
    arrival = np.empty(n, dtype=np.float64)
    departure = np.empty(n, dtype=np.float64)
    wait = np.empty(n, dtype=np.float64)

    t = 0.0
    prev = 0
    for k in range(n):
        node = route_arr[k]
        arr = t + dist[prev, node]
        arrival[k] = arr
        start = max(arr, ready[node])
        wait[k] = start - arr
        dep = start + service[node]
        departure[k] = dep
        t = dep
        prev = node

    F = np.empty(n, dtype=np.float64)
    if n > 0:
        last = route_arr[n - 1]
        latest_start = min(due[last], due[0] - dist[last, 0] - service[last])
        start_last = max(arrival[n - 1], ready[last])
        F[n - 1] = latest_start - start_last

        for i in range(n - 2, -1, -1):
            node = route_arr[i]
            start_i = max(arrival[i], ready[node])
            slack_self = due[node] - start_i
            F[i] = min(slack_self, wait[i + 1] + F[i + 1])

    return arrival, departure, wait, F


@njit(cache=True)
def _two_opt_best_numba(
    route_arr: np.ndarray,
    dist: np.ndarray,
    demands: np.ndarray,
    capacity: float,
    ready: np.ndarray,
    due: np.ndarray,
    service: np.ndarray,
):
    """Return (best_route, improved) after trying every 2-opt reversal with O(1) Savelsbergh Forward Slack."""
    n = len(route_arr)
    if n < 4:
        return route_arr.copy(), False

    # Precompute arrival, departure, wait, and Savelsbergh forward slack F
    arrival, departure, wait, F = _compute_route_forward_slack_numba(route_arr, dist, ready, due, service)

    best_delta = -1e-9
    best_i = -1
    best_j = -1

    for i in range(n - 2):
        for j in range(i + 2, n):
            u = route_arr[i - 1] if i > 0 else 0
            v = route_arr[i]
            w = route_arr[j]
            x = route_arr[j + 1] if j < n - 1 else 0

            # O(1) cost delta check
            delta = dist[u, w] + dist[v, x] - dist[u, v] - dist[w, x]
            if delta >= best_delta:
                continue

            # Feasibility check: only check if delta is better than best_delta
            dep_prev = departure[i - 1] if i > 0 else 0.0
            prev_node = u
            feasible = True

            # Traverse reversed segment: route_arr[j] down to route_arr[i]
            for k in range(j, i - 1, -1):
                node = route_arr[k]
                arr_time = dep_prev + dist[prev_node, node]
                if arr_time > due[node]:
                    feasible = False
                    break
                dep_prev = max(arr_time, ready[node]) + service[node]
                prev_node = node

            if not feasible:
                continue

            # O(1) Suffix feasibility check via Savelsbergh Forward Slack F
            if j < n - 1:
                nxt = route_arr[j + 1]
                new_arr = dep_prev + dist[prev_node, nxt]
                delta_t = new_arr - arrival[j + 1]
                if new_arr > due[nxt] or delta_t > F[j + 1]:
                    continue
            else:
                # Return to depot
                if dep_prev + dist[prev_node, 0] > due[0]:
                    continue

            # If we reach here, it's feasible and improves best_delta
            best_delta = delta
            best_i = i
            best_j = j

    if best_i != -1:
        # Reconstruct best route
        best_arr = route_arr.copy()
        idx = best_i
        for k in range(best_j, best_i - 1, -1):
            best_arr[idx] = route_arr[k]
            idx += 1
        return best_arr, True

    return route_arr.copy(), False


# ──────────────────────────────────────────────────────────────────────
# 2.  Intra-route Or-Opt  (segment lengths 1, 2, 3)
# ──────────────────────────────────────────────────────────────────────


@njit(cache=True)
def _or_opt_intra_numba(
    route_arr: np.ndarray,
    dist: np.ndarray,
    demands: np.ndarray,
    capacity: float,
    ready: np.ndarray,
    due: np.ndarray,
    service: np.ndarray,
):
    """Return (best_route, improved) after trying all intra-route or-opt moves."""
    n = len(route_arr)
    if n < 4:
        return route_arr.copy(), False

    best_cost = _route_cost(route_arr, dist)
    original_cost = best_cost
    best_arr = route_arr.copy()
    improved = False

    remainder = np.empty(n, dtype=np.int64)
    cand = np.empty(n, dtype=np.int64)
    seg = np.empty(3, dtype=np.int64)

    for L in range(1, 4):  # segment lengths 1, 2, 3
        if n < L + 2:
            continue
        for sp in range(n - L + 1):
            # Extract segment
            for k in range(L):
                seg[k] = route_arr[sp + k]

            # Build remainder (route without segment)
            rn = 0
            for k in range(n):
                if k < sp or k >= sp + L:
                    remainder[rn] = route_arr[k]
                    rn += 1
            # rn == n - L

            # Precalculate segment boundaries in original route
            prev_seg = route_arr[sp - 1] if sp > 0 else 0
            next_seg = route_arr[sp + L] if sp + L < n else 0
            orig_edges_cost = dist[prev_seg, route_arr[sp]] + dist[route_arr[sp + L - 1], next_seg]

            for ip in range(rn + 1):
                u_rem = remainder[ip - 1] if ip > 0 else 0
                v_rem = remainder[ip] if ip < rn else 0

                for rev in range(2):  # 0 = forward, 1 = reversed
                    first_seg_node = seg[L - 1] if rev == 1 else seg[0]
                    last_seg_node = seg[0] if rev == 1 else seg[L - 1]

                    # O(1) cost delta calculation
                    delta = (dist[prev_seg, next_seg] + dist[u_rem, first_seg_node] + dist[last_seg_node, v_rem]) - (
                        orig_edges_cost + dist[u_rem, v_rem]
                    )

                    if original_cost + delta >= best_cost - 1e-9:
                        continue

                    # Build candidate only if cost is promising
                    ci = 0
                    for k in range(ip):
                        cand[ci] = remainder[k]
                        ci += 1
                    if rev == 0:
                        for k in range(L):
                            cand[ci] = seg[k]
                            ci += 1
                    else:
                        for k in range(L - 1, -1, -1):
                            cand[ci] = seg[k]
                            ci += 1
                    for k in range(ip, rn):
                        cand[ci] = remainder[k]
                        ci += 1

                    # Skip identity move
                    is_identity = True
                    for k in range(n):
                        if cand[k] != route_arr[k]:
                            is_identity = False
                            break
                    if is_identity:
                        continue

                    if not _route_ok(cand, demands, capacity, ready, due, service, dist):
                        continue

                    # Feasible and improves best_cost
                    best_cost = original_cost + delta
                    for k in range(n):
                        best_arr[k] = cand[k]
                    improved = True

    return best_arr, improved


# ──────────────────────────────────────────────────────────────────────
# 3.  Combined intra-route optimise  (alternates 2-opt ↔ or-opt)
# ──────────────────────────────────────────────────────────────────────


@njit(cache=True)
def _intra_route_optimize_numba(
    route_arr: np.ndarray,
    dist: np.ndarray,
    demands: np.ndarray,
    capacity: float,
    ready: np.ndarray,
    due: np.ndarray,
    service: np.ndarray,
    max_passes: int,
):
    """Alternate 2-opt and or-opt passes until convergence or *max_passes*."""
    best = route_arr.copy()
    for _pass in range(max_passes):
        changed = False

        arr2, imp2 = _two_opt_best_numba(best, dist, demands, capacity, ready, due, service)
        if imp2:
            best = arr2
            changed = True

        arr3, imp3 = _or_opt_intra_numba(best, dist, demands, capacity, ready, due, service)
        if imp3:
            best = arr3
            changed = True

        if not changed:
            break

    return best


# ──────────────────────────────────────────────────────────────────────
# 4.  Inter-route single-node swap evaluation
# ──────────────────────────────────────────────────────────────────────


@njit(cache=True)
def _swap_evaluate_numba(
    r1: np.ndarray,
    r2: np.ndarray,
    dist: np.ndarray,
    demands: np.ndarray,
    capacity: float,
    ready: np.ndarray,
    due: np.ndarray,
    service: np.ndarray,
):
    """
    Evaluate every single-node swap between *r1* and *r2*.

    Returns ``(best_delta, sp, dp)`` where *sp*/*dp* are positions in
    *r1*/*r2* respectively.  If no improving swap exists the sentinel
    ``(-1e-9, -1, -1)`` is returned.
    """
    n1 = len(r1)
    n2 = len(r2)
    c1 = _route_cost(r1, dist)
    c2 = _route_cost(r2, dist)
    old_cost = c1 + c2

    best_delta = -1e-9
    best_sp = -1
    best_dp = -1

    # Working copies
    t1 = r1.copy()
    t2 = r2.copy()

    for sp in range(n1):
        for dp in range(n2):
            if r1[sp] == r2[dp]:
                continue

            # Swap in-place
            t1[sp] = r2[dp]
            t2[dp] = r1[sp]

            if _route_ok(t1, demands, capacity, ready, due, service, dist) and _route_ok(
                t2, demands, capacity, ready, due, service, dist
            ):
                new_cost = _route_cost(t1, dist) + _route_cost(t2, dist)
                delta = new_cost - old_cost
                if delta < best_delta:
                    best_delta = delta
                    best_sp = sp
                    best_dp = dp

            # Undo swap
            t1[sp] = r1[sp]
            t2[dp] = r2[dp]

    return best_delta, best_sp, best_dp


@njit(cache=True)
def _swap_21_evaluate_numba(
    r1: np.ndarray,
    r2: np.ndarray,
    dist: np.ndarray,
    demands: np.ndarray,
    capacity: float,
    ready: np.ndarray,
    due: np.ndarray,
    service: np.ndarray,
):
    """
    Evaluate Swap(2,1) and Swap(1,2) between r1 and r2.
    Returns (best_delta, move_type, sp, dp)
    move_type = 21 for 2-from-r1 / 1-from-r2
    move_type = 12 for 1-from-r1 / 2-from-r2
    """
    n1 = len(r1)
    n2 = len(r2)
    c1 = _route_cost(r1, dist)
    c2 = _route_cost(r2, dist)
    old_cost = c1 + c2

    best_delta = -1e-9
    best_type = 0
    best_sp = -1
    best_dp = -1

    # Part 1: Swap 2 from r1 (sp, sp+1) with 1 from r2 (dp)
    if n1 >= 2 and n2 >= 1:
        for sp in range(n1 - 1):
            n_a = r1[sp]
            n_b = r1[sp + 1]
            for dp in range(n2):
                n_c = r2[dp]
                t1 = np.empty(n1 - 1, dtype=np.int64)
                t1[:sp] = r1[:sp]
                t1[sp] = n_c
                t1[sp + 1 :] = r1[sp + 2 :]

                t2 = np.empty(n2 + 1, dtype=np.int64)
                t2[:dp] = r2[:dp]
                t2[dp] = n_a
                t2[dp + 1] = n_b
                t2[dp + 2 :] = r2[dp + 1 :]

                if _route_ok(t1, demands, capacity, ready, due, service, dist) and _route_ok(
                    t2, demands, capacity, ready, due, service, dist
                ):
                    new_cost = _route_cost(t1, dist) + _route_cost(t2, dist)
                    delta = new_cost - old_cost
                    if delta < best_delta:
                        best_delta = delta
                        best_type = 21
                        best_sp = sp
                        best_dp = dp

    # Part 2: Swap 1 from r1 (sp) with 2 from r2 (dp, dp+1)
    if n1 >= 1 and n2 >= 2:
        for sp in range(n1):
            n_c = r1[sp]
            for dp in range(n2 - 1):
                n_a = r2[dp]
                n_b = r2[dp + 1]
                t1 = np.empty(n1 + 1, dtype=np.int64)
                t1[:sp] = r1[:sp]
                t1[sp] = n_a
                t1[sp + 1] = n_b
                t1[sp + 2 :] = r1[sp + 1 :]

                t2 = np.empty(n2 - 1, dtype=np.int64)
                t2[:dp] = r2[:dp]
                t2[dp] = n_c
                t2[dp + 1 :] = r2[dp + 2 :]

                if _route_ok(t1, demands, capacity, ready, due, service, dist) and _route_ok(
                    t2, demands, capacity, ready, due, service, dist
                ):
                    new_cost = _route_cost(t1, dist) + _route_cost(t2, dist)
                    delta = new_cost - old_cost
                    if delta < best_delta:
                        best_delta = delta
                        best_type = 12
                        best_sp = sp
                        best_dp = dp

    return best_delta, best_type, best_sp, best_dp


# ──────────────────────────────────────────────────────────────────────
# 5.  Best segment insertion into a route
# ──────────────────────────────────────────────────────────────────────


@njit(cache=True)
def _best_segment_insert_numba(
    seg: np.ndarray,
    route: np.ndarray,
    dist: np.ndarray,
    demands: np.ndarray,
    capacity: float,
    ready: np.ndarray,
    due: np.ndarray,
    service: np.ndarray,
):
    """
    Find the best position to insert *seg* into *route*.

    Both forward and reversed orientations are tried.  Capacity is
    checked first for an early exit.

    Returns ``(best_delta, best_pos, rev_int)`` or ``(1e18, -1, 0)``
    if no feasible insertion exists.
    """
    seg_len = len(seg)
    rn = len(route)
    new_len = rn + seg_len

    # ── Capacity early-exit ──
    seg_load = 0.0
    for k in range(seg_len):
        seg_load += demands[seg[k]]
    route_load = 0.0
    for k in range(rn):
        route_load += demands[route[k]]
    if seg_load + route_load > capacity:
        return 1e18, -1, 0

    old_cost = _route_cost(route, dist)

    best_delta = 1e18
    best_pos = -1
    best_rev = 0

    cand = np.empty(new_len, dtype=np.int64)
    seg_rev = np.empty(seg_len, dtype=np.int64)
    for k in range(seg_len):
        seg_rev[k] = seg[seg_len - 1 - k]

    for rev in range(2):  # 0 = forward, 1 = reversed
        s = seg if rev == 0 else seg_rev
        for pos in range(rn + 1):
            ci = 0
            for k in range(pos):
                cand[ci] = route[k]
                ci += 1
            for k in range(seg_len):
                cand[ci] = s[k]
                ci += 1
            for k in range(pos, rn):
                cand[ci] = route[k]
                ci += 1

            if not _route_ok(cand, demands, capacity, ready, due, service, dist):
                continue
            new_cost = _route_cost(cand, dist)
            delta = new_cost - old_cost
            if delta < best_delta:
                best_delta = delta
                best_pos = pos
                best_rev = rev

    if best_pos == -1:
        return 1e18, -1, 0
    return best_delta, best_pos, best_rev


# ──────────────────────────────────────────────────────────────────────
# 6.  Cross-exchange segment swap between two routes
# ──────────────────────────────────────────────────────────────────────


@njit(cache=True)
def _cross_exchange_pair_numba(
    r1: np.ndarray,
    r2: np.ndarray,
    max_len1: int,
    max_len2: int,
    dist: np.ndarray,
    demands: np.ndarray,
    capacity: float,
    ready: np.ndarray,
    due: np.ndarray,
    service: np.ndarray,
    old_pair_cost: float,
):
    """
    Evaluate all segment-swap moves between *r1* and *r2*.

    For each combination of segment lengths and starting positions the
    two candidate routes are built, feasibility-checked, and the cost
    delta is computed.

    Returns ``(delta, p1, len1, p2, len2)`` for the best improving move,
    or ``(-1e-9, -1, -1, -1, -1)`` if none exists.
    """
    n1 = len(r1)
    n2 = len(r2)
    max_cand = n1 + n2  # upper bound on candidate length

    cand1 = np.empty(max_cand, dtype=np.int64)
    cand2 = np.empty(max_cand, dtype=np.int64)

    best_delta = -1e-9
    best_p1 = -1
    best_l1 = -1
    best_p2 = -1
    best_l2 = -1

    # Precompute route loads for capacity early-exit
    load1 = 0.0
    for k in range(n1):
        load1 += demands[r1[k]]
    load2 = 0.0
    for k in range(n2):
        load2 += demands[r2[k]]

    for len1 in range(1, max_len1 + 1):
        if n1 < len1:
            continue
        for len2 in range(1, max_len2 + 1):
            if n2 < len2:
                continue
            for p1 in range(n1 - len1 + 1):
                # Precompute segment 1 load
                seg1_load = 0.0
                for k in range(p1, p1 + len1):
                    seg1_load += demands[r1[k]]
                for p2 in range(n2 - len2 + 1):
                    # Precompute segment 2 load
                    seg2_load = 0.0
                    for k in range(p2, p2 + len2):
                        seg2_load += demands[r2[k]]
                    # Capacity early-exit: skip if either swapped route exceeds capacity
                    new_load1 = load1 - seg1_load + seg2_load
                    new_load2 = load2 - seg2_load + seg1_load
                    if new_load1 > capacity or new_load2 > capacity:
                        continue

                    # Build nr1 = r1[:p1] + r2[p2:p2+len2] + r1[p1+len1:]
                    cn1 = 0
                    for k in range(p1):
                        cand1[cn1] = r1[k]
                        cn1 += 1
                    for k in range(p2, p2 + len2):
                        cand1[cn1] = r2[k]
                        cn1 += 1
                    for k in range(p1 + len1, n1):
                        cand1[cn1] = r1[k]
                        cn1 += 1

                    # Build nr2 = r2[:p2] + r1[p1:p1+len1] + r2[p2+len2:]
                    cn2 = 0
                    for k in range(p2):
                        cand2[cn2] = r2[k]
                        cn2 += 1
                    for k in range(p1, p1 + len1):
                        cand2[cn2] = r1[k]
                        cn2 += 1
                    for k in range(p2 + len2, n2):
                        cand2[cn2] = r2[k]
                        cn2 += 1

                    # Skip empty routes
                    if cn1 == 0 or cn2 == 0:
                        continue

                    # Feasibility checks on the variable-length slices
                    c1_slice = cand1[:cn1]
                    c2_slice = cand2[:cn2]

                    if not _route_ok(c1_slice, demands, capacity, ready, due, service, dist):
                        continue
                    if not _route_ok(c2_slice, demands, capacity, ready, due, service, dist):
                        continue

                    new_cost = _route_cost(c1_slice, dist) + _route_cost(c2_slice, dist)
                    delta = new_cost - old_pair_cost
                    if delta < best_delta:
                        best_delta = delta
                        best_p1 = p1
                        best_l1 = len1
                        best_p2 = p2
                        best_l2 = len2

    return best_delta, best_p1, best_l1, best_p2, best_l2


# ──────────────────────────────────────────────────────────────────────
# 7.  Pruned kernels using GNN heatmaps
# ──────────────────────────────────────────────────────────────────────


@njit(cache=True)
def _swap_evaluate_pruned_numba(
    r1: np.ndarray,
    r2: np.ndarray,
    dist: np.ndarray,
    demands: np.ndarray,
    capacity: float,
    ready: np.ndarray,
    due: np.ndarray,
    service: np.ndarray,
    heatmap: np.ndarray,
    pruning_threshold: float,
):
    n1 = len(r1)
    n2 = len(r2)
    c1 = _route_cost(r1, dist)
    c2 = _route_cost(r2, dist)
    old_cost = c1 + c2

    best_delta = -1e-9
    best_sp = -1
    best_dp = -1

    t1 = r1.copy()
    t2 = r2.copy()

    for sp in range(n1):
        prev1 = r1[sp - 1] if sp > 0 else 0
        nxt1 = r1[sp + 1] if sp < n1 - 1 else 0
        u = r1[sp]

        for dp in range(n2):
            v = r2[dp]
            if u == v:
                continue

            prev2 = r2[dp - 1] if dp > 0 else 0
            nxt2 = r2[dp + 1] if dp < n2 - 1 else 0

            # GNN edge probability check
            if (
                heatmap[prev1, v] < pruning_threshold
                or heatmap[v, nxt1] < pruning_threshold
                or heatmap[prev2, u] < pruning_threshold
                or heatmap[u, nxt2] < pruning_threshold
            ):
                continue

            # Swap in-place
            t1[sp] = v
            t2[dp] = u

            if _route_ok(t1, demands, capacity, ready, due, service, dist) and _route_ok(
                t2, demands, capacity, ready, due, service, dist
            ):
                new_cost = _route_cost(t1, dist) + _route_cost(t2, dist)
                delta = new_cost - old_cost
                if delta < best_delta:
                    best_delta = delta
                    best_sp = sp
                    best_dp = dp

            # Undo swap
            t1[sp] = u
            t2[dp] = v

    return best_delta, best_sp, best_dp


@njit(cache=True)
def _best_segment_insert_pruned_numba(
    seg: np.ndarray,
    route: np.ndarray,
    dist: np.ndarray,
    demands: np.ndarray,
    capacity: float,
    ready: np.ndarray,
    due: np.ndarray,
    service: np.ndarray,
    heatmap: np.ndarray,
    pruning_threshold: float,
):
    seg_len = len(seg)
    rn = len(route)
    new_len = rn + seg_len

    # Capacity early-exit
    seg_load = 0.0
    for k in range(seg_len):
        seg_load += demands[seg[k]]
    route_load = 0.0
    for k in range(rn):
        route_load += demands[route[k]]
    if seg_load + route_load > capacity:
        return 1e18, -1, 0

    old_cost = _route_cost(route, dist)

    best_delta = 1e18
    best_pos = -1
    best_rev = 0

    cand = np.empty(new_len, dtype=np.int64)
    seg_rev = np.empty(seg_len, dtype=np.int64)
    for k in range(seg_len):
        seg_rev[k] = seg[seg_len - 1 - k]

    for rev in range(2):  # 0 = forward, 1 = reversed
        s = seg if rev == 0 else seg_rev
        for pos in range(rn + 1):
            prev = route[pos - 1] if pos > 0 else 0
            nxt = route[pos] if pos < rn else 0

            # GNN edge probability check
            if heatmap[prev, s[0]] < pruning_threshold or heatmap[s[-1], nxt] < pruning_threshold:
                continue

            ci = 0
            for k in range(pos):
                cand[ci] = route[k]
                ci += 1
            for k in range(seg_len):
                cand[ci] = s[k]
                ci += 1
            for k in range(pos, rn):
                cand[ci] = route[k]
                ci += 1

            if not _route_ok(cand, demands, capacity, ready, due, service, dist):
                continue
            new_cost = _route_cost(cand, dist)
            delta = new_cost - old_cost
            if delta < best_delta:
                best_delta = delta
                best_pos = pos
                best_rev = rev

    if best_pos == -1:
        return 1e18, -1, 0
    return best_delta, best_pos, best_rev


@njit(cache=True)
def _cross_exchange_pair_pruned_numba(
    r1: np.ndarray,
    r2: np.ndarray,
    max_len1: int,
    max_len2: int,
    dist: np.ndarray,
    demands: np.ndarray,
    capacity: float,
    ready: np.ndarray,
    due: np.ndarray,
    service: np.ndarray,
    old_pair_cost: float,
    heatmap: np.ndarray,
    pruning_threshold: float,
):
    n1 = len(r1)
    n2 = len(r2)
    max_cand = n1 + n2

    cand1 = np.empty(max_cand, dtype=np.int64)
    cand2 = np.empty(max_cand, dtype=np.int64)

    best_delta = -1e-9
    best_p1 = -1
    best_l1 = -1
    best_p2 = -1
    best_l2 = -1

    # Precompute route loads for capacity early-exit
    load1 = 0.0
    for k in range(n1):
        load1 += demands[r1[k]]
    load2 = 0.0
    for k in range(n2):
        load2 += demands[r2[k]]

    for len1 in range(1, max_len1 + 1):
        if n1 < len1:
            continue
        for len2 in range(1, max_len2 + 1):
            if n2 < len2:
                continue
            for p1 in range(n1 - len1 + 1):
                prev1 = r1[p1 - 1] if p1 > 0 else 0
                nxt1 = r1[p1 + len1] if p1 + len1 < n1 else 0

                # Precompute segment 1 load
                seg1_load = 0.0
                for k in range(p1, p1 + len1):
                    seg1_load += demands[r1[k]]

                for p2 in range(n2 - len2 + 1):
                    prev2 = r2[p2 - 1] if p2 > 0 else 0
                    nxt2 = r2[p2 + len2] if p2 + len2 < n2 else 0

                    # Precompute segment 2 load
                    seg2_load = 0.0
                    for k in range(p2, p2 + len2):
                        seg2_load += demands[r2[k]]
                    # Capacity early-exit
                    new_load1 = load1 - seg1_load + seg2_load
                    new_load2 = load2 - seg2_load + seg1_load
                    if new_load1 > capacity or new_load2 > capacity:
                        continue

                    # GNN boundary edge checks
                    if (
                        heatmap[prev1, r2[p2]] < pruning_threshold
                        or heatmap[r2[p2 + len2 - 1], nxt1] < pruning_threshold
                        or heatmap[prev2, r1[p1]] < pruning_threshold
                        or heatmap[r1[p1 + len1 - 1], nxt2] < pruning_threshold
                    ):
                        continue

                    # Build nr1 = r1[:p1] + r2[p2:p2+len2] + r1[p1+len1:]
                    cn1 = 0
                    for k in range(p1):
                        cand1[cn1] = r1[k]
                        cn1 += 1
                    for k in range(p2, p2 + len2):
                        cand1[cn1] = r2[k]
                        cn1 += 1
                    for k in range(p1 + len1, n1):
                        cand1[cn1] = r1[k]
                        cn1 += 1

                    # Build nr2 = r2[:p2] + r1[p1:p1+len1] + r2[p2+len2:]
                    cn2 = 0
                    for k in range(p2):
                        cand2[cn2] = r2[k]
                        cn2 += 1
                    for k in range(p1, p1 + len1):
                        cand2[cn2] = r1[k]
                        cn2 += 1
                    for k in range(p2 + len2, n2):
                        cand2[cn2] = r2[k]
                        cn2 += 1

                    if cn1 == 0 or cn2 == 0:
                        continue

                    c1_slice = cand1[:cn1]
                    c2_slice = cand2[:cn2]

                    if not _route_ok(c1_slice, demands, capacity, ready, due, service, dist):
                        continue
                    if not _route_ok(c2_slice, demands, capacity, ready, due, service, dist):
                        continue

                    new_cost = _route_cost(c1_slice, dist) + _route_cost(c2_slice, dist)
                    delta = new_cost - old_pair_cost
                    if delta < best_delta:
                        best_delta = delta
                        best_p1 = p1
                        best_l1 = len1
                        best_p2 = p2
                        best_l2 = len2

    return best_delta, best_p1, best_l1, best_p2, best_l2


# ──────────────────────────────────────────────────────────────────────
# 8.  CROSS tail-swap inter-route evaluation
# ──────────────────────────────────────────────────────────────────────


@njit(cache=True)
def _cross_tail_pair_numba(
    r1: np.ndarray,
    r2: np.ndarray,
    dist: np.ndarray,
    demands: np.ndarray,
    capacity: float,
    ready: np.ndarray,
    due: np.ndarray,
    service: np.ndarray,
    old_pair_cost: float,
):
    """
    Evaluate all inter-route tail swaps (exchanging suffixes r1[i:] and r2[j:])
    without reversing the suffixes.

    Returns (best_delta, best_i, best_j) or (-1e-9, -1, -1) if no improving swap exists.
    """
    n1 = len(r1)
    n2 = len(r2)

    best_delta = -1e-9
    best_i = -1
    best_j = -1

    load1 = 0.0
    for k in range(n1):
        load1 += demands[r1[k]]
    load2 = 0.0
    for k in range(n2):
        load2 += demands[r2[k]]

    pref_load1 = np.empty(n1 + 1)
    pref_load1[0] = 0.0
    for k in range(n1):
        pref_load1[k + 1] = pref_load1[k] + demands[r1[k]]

    pref_load2 = np.empty(n2 + 1)
    pref_load2[0] = 0.0
    for k in range(n2):
        pref_load2[k + 1] = pref_load2[k] + demands[r2[k]]

    max_cand = n1 + n2 + 2
    cand1 = np.empty(max_cand, dtype=np.int64)
    cand2 = np.empty(max_cand, dtype=np.int64)

    for i in range(n1 + 1):
        t1_load_part = pref_load1[i]
        t2_load_part = load1 - t1_load_part
        for j in range(n2 + 1):
            if (i == 0 and j == 0) or (i == n1 and j == n2):
                continue

            new_load1 = t1_load_part + (load2 - pref_load2[j])
            new_load2 = pref_load2[j] + t2_load_part

            if new_load1 > capacity or new_load2 > capacity:
                continue

            u = r1[i - 1] if i > 0 else 0
            u_prime = r1[i] if i < n1 else 0
            v = r2[j - 1] if j > 0 else 0
            v_prime = r2[j] if j < n2 else 0

            delta = dist[u, v_prime] + dist[v, u_prime] - dist[u, u_prime] - dist[v, v_prime]

            if delta >= best_delta:
                continue

            cn1 = 0
            for k in range(i):
                cand1[cn1] = r1[k]
                cn1 += 1
            for k in range(j, n2):
                cand1[cn1] = r2[k]
                cn1 += 1

            cn2 = 0
            for k in range(j):
                cand2[cn2] = r2[k]
                cn2 += 1
            for k in range(i, n1):
                cand2[cn2] = r1[k]
                cn2 += 1

            c1_slice = cand1[:cn1]
            c2_slice = cand2[:cn2]

            feasible = True
            if cn1 > 0:
                if not _route_ok(c1_slice, demands, capacity, ready, due, service, dist):
                    feasible = False
            if feasible and cn2 > 0:
                if not _route_ok(c2_slice, demands, capacity, ready, due, service, dist):
                    feasible = False

            if not feasible:
                continue

            best_delta = delta
            best_i = i
            best_j = j

    return best_delta, best_i, best_j


@njit(cache=True)
def _cross_tail_pair_pruned_numba(
    r1: np.ndarray,
    r2: np.ndarray,
    dist: np.ndarray,
    demands: np.ndarray,
    capacity: float,
    ready: np.ndarray,
    due: np.ndarray,
    service: np.ndarray,
    old_pair_cost: float,
    heatmap: np.ndarray,
    pruning_threshold: float,
):
    """
    GNN heatmap pruned version of _cross_tail_pair_numba.
    """
    n1 = len(r1)
    n2 = len(r2)

    best_delta = -1e-9
    best_i = -1
    best_j = -1

    load1 = 0.0
    for k in range(n1):
        load1 += demands[r1[k]]
    load2 = 0.0
    for k in range(n2):
        load2 += demands[r2[k]]

    pref_load1 = np.empty(n1 + 1)
    pref_load1[0] = 0.0
    for k in range(n1):
        pref_load1[k + 1] = pref_load1[k] + demands[r1[k]]

    pref_load2 = np.empty(n2 + 1)
    pref_load2[0] = 0.0
    for k in range(n2):
        pref_load2[k + 1] = pref_load2[k] + demands[r2[k]]

    max_cand = n1 + n2 + 2
    cand1 = np.empty(max_cand, dtype=np.int64)
    cand2 = np.empty(max_cand, dtype=np.int64)

    for i in range(n1 + 1):
        t1_load_part = pref_load1[i]
        t2_load_part = load1 - t1_load_part
        for j in range(n2 + 1):
            if (i == 0 and j == 0) or (i == n1 and j == n2):
                continue

            new_load1 = t1_load_part + (load2 - pref_load2[j])
            new_load2 = pref_load2[j] + t2_load_part

            if new_load1 > capacity or new_load2 > capacity:
                continue

            u = r1[i - 1] if i > 0 else 0
            u_prime = r1[i] if i < n1 else 0
            v = r2[j - 1] if j > 0 else 0
            v_prime = r2[j] if j < n2 else 0

            # GNN boundary check: check probability of the two new edges
            if heatmap[u, v_prime] < pruning_threshold or heatmap[v, u_prime] < pruning_threshold:
                continue

            delta = dist[u, v_prime] + dist[v, u_prime] - dist[u, u_prime] - dist[v, v_prime]

            if delta >= best_delta:
                continue

            cn1 = 0
            for k in range(i):
                cand1[cn1] = r1[k]
                cn1 += 1
            for k in range(j, n2):
                cand1[cn1] = r2[k]
                cn1 += 1

            cn2 = 0
            for k in range(j):
                cand2[cn2] = r2[k]
                cn2 += 1
            for k in range(i, n1):
                cand2[cn2] = r1[k]
                cn2 += 1

            c1_slice = cand1[:cn1]
            c2_slice = cand2[:cn2]

            feasible = True
            if cn1 > 0:
                if not _route_ok(c1_slice, demands, capacity, ready, due, service, dist):
                    feasible = False
            if feasible and cn2 > 0:
                if not _route_ok(c2_slice, demands, capacity, ready, due, service, dist):
                    feasible = False

            if not feasible:
                continue

            best_delta = delta
            best_i = i
            best_j = j

    return best_delta, best_i, best_j


@njit(cache=True)
def _string_relocate_pair_numba(
    r1: np.ndarray,
    r2: np.ndarray,
    dist: np.ndarray,
    demands: np.ndarray,
    capacity: float,
    ready: np.ndarray,
    due: np.ndarray,
    service: np.ndarray,
    old_pair_cost: float,
):
    """
    Evaluate all inter-route segment relocations of length 2 or 3 from r1 to r2,
    and from r2 to r1 (both orientations: forward and reversed).

    Returns (best_delta, direction, p1, length, p2, rev)
    where:
      direction: 1 if r1 -> r2, 2 if r2 -> r1, -1 if no move
      p1: starting position of segment in source route
      length: length of the segment (2 or 3)
      p2: insertion position in destination route
      rev: 1 if segment is reversed, 0 if forward
    """
    n1 = len(r1)
    n2 = len(r2)

    best_delta = -1e-9
    best_direction = -1
    best_p1 = -1
    best_length = -1
    best_p2 = -1
    best_rev = 0

    load1 = 0.0
    for k in range(n1):
        load1 += demands[r1[k]]
    load2 = 0.0
    for k in range(n2):
        load2 += demands[r2[k]]

    max_cand = n1 + n2 + 3
    cand1 = np.empty(max_cand, dtype=np.int64)
    cand2 = np.empty(max_cand, dtype=np.int64)

    # 1. Direction: r1 -> r2
    for L in (2, 3):
        if n1 < L:
            continue
        for p1 in range(n1 - L + 1):
            seg_load = 0.0
            for k in range(p1, p1 + L):
                seg_load += demands[r1[k]]

            if load2 + seg_load > capacity:
                continue

            # Build cand1 = r1 without r1[p1:p1+L]
            cn1 = 0
            for k in range(p1):
                cand1[cn1] = r1[k]
                cn1 += 1
            for k in range(p1 + L, n1):
                cand1[cn1] = r1[k]
                cn1 += 1

            # Loop over insertion positions in r2
            for p2 in range(n2 + 1):
                for rev in (0, 1):
                    # Build cand2 = r2 with r1[p1:p1+L] inserted at p2
                    cn2 = 0
                    for k in range(p2):
                        cand2[cn2] = r2[k]
                        cn2 += 1

                    if rev == 0:
                        for k in range(L):
                            cand2[cn2] = r1[p1 + k]
                            cn2 += 1
                    else:
                        for k in range(L - 1, -1, -1):
                            cand2[cn2] = r1[p1 + k]
                            cn2 += 1

                    for k in range(p2, n2):
                        cand2[cn2] = r2[k]
                        cn2 += 1

                    c1_slice = cand1[:cn1]
                    c2_slice = cand2[:cn2]

                    feasible1 = True
                    if cn1 > 0:
                        feasible1 = _route_ok(c1_slice, demands, capacity, ready, due, service, dist)

                    feasible2 = True
                    if cn2 > 0:
                        feasible2 = _route_ok(c2_slice, demands, capacity, ready, due, service, dist)

                    if not (feasible1 and feasible2):
                        continue

                    cost1 = _route_cost(c1_slice, dist) if cn1 > 0 else 0.0
                    cost2 = _route_cost(c2_slice, dist) if cn2 > 0 else 0.0
                    new_cost = cost1 + cost2
                    delta = new_cost - old_pair_cost

                    if delta < best_delta:
                        best_delta = delta
                        best_direction = 1
                        best_p1 = p1
                        best_length = L
                        best_p2 = p2
                        best_rev = rev

    # 2. Direction: r2 -> r1
    for L in (2, 3):
        if n2 < L:
            continue
        for p1 in range(n2 - L + 1):
            seg_load = 0.0
            for k in range(p1, p1 + L):
                seg_load += demands[r2[k]]

            if load1 + seg_load > capacity:
                continue

            # Build cand2 = r2 without r2[p1:p1+L]
            cn2 = 0
            for k in range(p1):
                cand2[cn2] = r2[k]
                cn2 += 1
            for k in range(p1 + L, n2):
                cand2[cn2] = r2[k]
                cn2 += 1

            # Loop over insertion positions in r1
            for p2 in range(n1 + 1):
                for rev in (0, 1):
                    # Build cand1 = r1 with r2[p1:p1+L] inserted at p2
                    cn1 = 0
                    for k in range(p2):
                        cand1[cn1] = r1[k]
                        cn1 += 1

                    if rev == 0:
                        for k in range(L):
                            cand1[cn1] = r2[p1 + k]
                            cn1 += 1
                    else:
                        for k in range(L - 1, -1, -1):
                            cand1[cn1] = r2[p1 + k]
                            cn1 += 1

                    for k in range(p2, n1):
                        cand1[cn1] = r1[k]
                        cn1 += 1

                    c1_slice = cand1[:cn1]
                    c2_slice = cand2[:cn2]

                    feasible1 = True
                    if cn1 > 0:
                        feasible1 = _route_ok(c1_slice, demands, capacity, ready, due, service, dist)

                    feasible2 = True
                    if cn2 > 0:
                        feasible2 = _route_ok(c2_slice, demands, capacity, ready, due, service, dist)

                    if not (feasible1 and feasible2):
                        continue

                    cost1 = _route_cost(c1_slice, dist) if cn1 > 0 else 0.0
                    cost2 = _route_cost(c2_slice, dist) if cn2 > 0 else 0.0
                    new_cost = cost1 + cost2
                    delta = new_cost - old_pair_cost

                    if delta < best_delta:
                        best_delta = delta
                        best_direction = 2
                        best_p1 = p1
                        best_length = L
                        best_p2 = p2
                        best_rev = rev

    return best_delta, best_direction, best_p1, best_length, best_p2, best_rev


@njit(cache=True)
def _string_relocate_pair_pruned_numba(
    r1: np.ndarray,
    r2: np.ndarray,
    dist: np.ndarray,
    demands: np.ndarray,
    capacity: float,
    ready: np.ndarray,
    due: np.ndarray,
    service: np.ndarray,
    old_pair_cost: float,
    heatmap: np.ndarray,
    pruning_threshold: float,
):
    """
    GNN heatmap pruned version of _string_relocate_pair_numba.
    """
    n1 = len(r1)
    n2 = len(r2)

    best_delta = -1e-9
    best_direction = -1
    best_p1 = -1
    best_length = -1
    best_p2 = -1
    best_rev = 0

    load1 = 0.0
    for k in range(n1):
        load1 += demands[r1[k]]
    load2 = 0.0
    for k in range(n2):
        load2 += demands[r2[k]]

    max_cand = n1 + n2 + 3
    cand1 = np.empty(max_cand, dtype=np.int64)
    cand2 = np.empty(max_cand, dtype=np.int64)

    # 1. Direction: r1 -> r2
    for L in (2, 3):
        if n1 < L:
            continue
        for p1 in range(n1 - L + 1):
            prev1 = r1[p1 - 1] if p1 > 0 else 0
            nxt1 = r1[p1 + L] if p1 + L < n1 else 0

            # GNN boundary check: check edge created when segment is removed from r1
            if heatmap[prev1, nxt1] < pruning_threshold:
                continue

            seg_load = 0.0
            for k in range(p1, p1 + L):
                seg_load += demands[r1[k]]

            if load2 + seg_load > capacity:
                continue

            # Build cand1 = r1 without r1[p1:p1+L]
            cn1 = 0
            for k in range(p1):
                cand1[cn1] = r1[k]
                cn1 += 1
            for k in range(p1 + L, n1):
                cand1[cn1] = r1[k]
                cn1 += 1

            # Loop over insertion positions in r2
            for p2 in range(n2 + 1):
                prev2 = r2[p2 - 1] if p2 > 0 else 0
                nxt2 = r2[p2] if p2 < n2 else 0

                for rev in (0, 1):
                    # GNN boundary check: check edges created by segment insertion
                    if rev == 0:
                        if (
                            heatmap[prev2, r1[p1]] < pruning_threshold
                            or heatmap[r1[p1 + L - 1], nxt2] < pruning_threshold
                        ):
                            continue
                    else:
                        if (
                            heatmap[prev2, r1[p1 + L - 1]] < pruning_threshold
                            or heatmap[r1[p1], nxt2] < pruning_threshold
                        ):
                            continue

                    # Build cand2 = r2 with r1[p1:p1+L] inserted at p2
                    cn2 = 0
                    for k in range(p2):
                        cand2[cn2] = r2[k]
                        cn2 += 1

                    if rev == 0:
                        for k in range(L):
                            cand2[cn2] = r1[p1 + k]
                            cn2 += 1
                    else:
                        for k in range(L - 1, -1, -1):
                            cand2[cn2] = r1[p1 + k]
                            cn2 += 1

                    for k in range(p2, n2):
                        cand2[cn2] = r2[k]
                        cn2 += 1

                    c1_slice = cand1[:cn1]
                    c2_slice = cand2[:cn2]

                    feasible1 = True
                    if cn1 > 0:
                        feasible1 = _route_ok(c1_slice, demands, capacity, ready, due, service, dist)

                    feasible2 = True
                    if cn2 > 0:
                        feasible2 = _route_ok(c2_slice, demands, capacity, ready, due, service, dist)

                    if not (feasible1 and feasible2):
                        continue

                    cost1 = _route_cost(c1_slice, dist) if cn1 > 0 else 0.0
                    cost2 = _route_cost(c2_slice, dist) if cn2 > 0 else 0.0
                    new_cost = cost1 + cost2
                    delta = new_cost - old_pair_cost

                    if delta < best_delta:
                        best_delta = delta
                        best_direction = 1
                        best_p1 = p1
                        best_length = L
                        best_p2 = p2
                        best_rev = rev

    # 2. Direction: r2 -> r1
    for L in (2, 3):
        if n2 < L:
            continue
        for p1 in range(n2 - L + 1):
            prev2 = r2[p1 - 1] if p1 > 0 else 0
            nxt2 = r2[p1 + L] if p1 + L < n2 else 0

            # GNN boundary check: check edge created when segment is removed from r2
            if heatmap[prev2, nxt2] < pruning_threshold:
                continue

            seg_load = 0.0
            for k in range(p1, p1 + L):
                seg_load += demands[r2[k]]

            if load1 + seg_load > capacity:
                continue

            # Build cand2 = r2 without r2[p1:p1+L]
            cn2 = 0
            for k in range(p1):
                cand2[cn2] = r2[k]
                cn2 += 1
            for k in range(p1 + L, n2):
                cand2[cn2] = r2[k]
                cn2 += 1

            # Loop over insertion positions in r1
            for p2 in range(n1 + 1):
                prev1 = r1[p2 - 1] if p2 > 0 else 0
                nxt1 = r1[p2] if p2 < n1 else 0

                for rev in (0, 1):
                    # GNN boundary check: check edges created by segment insertion
                    if rev == 0:
                        if (
                            heatmap[prev1, r2[p1]] < pruning_threshold
                            or heatmap[r2[p1 + L - 1], nxt1] < pruning_threshold
                        ):
                            continue
                    else:
                        if (
                            heatmap[prev1, r2[p1 + L - 1]] < pruning_threshold
                            or heatmap[r2[p1], nxt1] < pruning_threshold
                        ):
                            continue

                    # Build cand1 = r1 with r2[p1:p1+L] inserted at p2
                    cn1 = 0
                    for k in range(p2):
                        cand1[cn1] = r1[k]
                        cn1 += 1

                    if rev == 0:
                        for k in range(L):
                            cand1[cn1] = r2[p1 + k]
                            cn1 += 1
                    else:
                        for k in range(L - 1, -1, -1):
                            cand1[cn1] = r2[p1 + k]
                            cn1 += 1

                    for k in range(p2, n1):
                        cand1[cn1] = r1[k]
                        cn1 += 1

                    c1_slice = cand1[:cn1]
                    c2_slice = cand2[:cn2]

                    feasible1 = True
                    if cn1 > 0:
                        feasible1 = _route_ok(c1_slice, demands, capacity, ready, due, service, dist)

                    feasible2 = True
                    if cn2 > 0:
                        feasible2 = _route_ok(c2_slice, demands, capacity, ready, due, service, dist)

                    if not (feasible1 and feasible2):
                        continue

                    cost1 = _route_cost(c1_slice, dist) if cn1 > 0 else 0.0
                    cost2 = _route_cost(c2_slice, dist) if cn2 > 0 else 0.0
                    new_cost = cost1 + cost2
                    delta = new_cost - old_pair_cost

                    if delta < best_delta:
                        best_delta = delta
                        best_direction = 2
                        best_p1 = p1
                        best_length = L
                        best_p2 = p2
                        best_rev = rev

    return best_delta, best_direction, best_p1, best_length, best_p2, best_rev

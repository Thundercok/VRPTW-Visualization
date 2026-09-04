from __future__ import annotations

import copy
import hashlib
from dataclasses import dataclass

import numpy as np

from .config import Config
from .core import Inst, Plan, _check_route
from .heuristics import _insert_customer, _route_avg_slack, _route_cost_list, _route_load

try:
    from scipy.optimize import Bounds, LinearConstraint
    from scipy.optimize import milp as _scipy_milp

    milp = _scipy_milp
    MILP_OK = True
except Exception:
    Bounds = LinearConstraint = milp = None
    MILP_OK = False


@dataclass(frozen=True)
class RouteRecord:
    nodes: tuple[int, ...]
    cost: float
    load: float
    slack: float
    protected: bool = False


def _cover_key(nodes: tuple[int, ...] | list[int]) -> tuple[int, ...]:
    return tuple(sorted(nodes))


def _same_cover_priority(rec: RouteRecord) -> tuple[float, float]:
    return (rec.cost, -rec.slack)


def _is_exact_cover(plan: Plan) -> bool:
    nodes = [n for route in plan.routes for n in route]
    return (
        len(nodes) == plan.inst.n and len(set(nodes)) == plan.inst.n and all(1 <= node <= plan.inst.n for node in nodes)
    )


class RoutePool:
    def __init__(self, inst: Inst, cfg: Config):
        self.inst = inst
        # Scale pool limits with instance size
        n = inst.n
        self.cfg = copy.copy(cfg)
        self.cfg.route_pool_limit = min(2000, 600 + 4 * n)
        self.cfg.route_pool_max_per_customer = min(60, 28 + n // 10)
        self.cfg.sp_time_limit = min(30.0, 4.0 + 0.05 * n)
        self._routes: dict[tuple[int, ...], RouteRecord] = {}
        self._cover_to_key: dict[tuple[int, ...], tuple[int, ...]] = {}
        # Memoised _milp_recombine solves, keyed by (column set, parameters).
        self._milp_cache: dict = {}

    def _priority(self, rec: RouteRecord) -> tuple[float, ...]:
        lr = rec.load / max(self.inst.capacity, 1)
        cps = rec.cost / max(len(rec.nodes), 1)
        return (-len(rec.nodes), cps, -lr, -rec.slack)

    def _trim(self) -> None:
        limit = self.cfg.route_pool_limit
        if len(self._routes) <= limit + 100:
            return

        slot_b = max(limit // 4, 8)  # 25% → longest routes (NV-1 MILP)
        limit - slot_b  # 75% → cheapest-per-stop routes

        usage: dict[int, int] = {}
        kept: dict[tuple[int, ...], RouteRecord] = {}

        # Slot B: sorted by route LENGTH only (decoupled from cost)
        len_ranked = sorted(self._routes.values(), key=lambda r: -len(r.nodes))

        # Slot A: sorted by cost PER CUSTOMER only (short efficient routes survive here)
        # Previously _priority used (-len, cps, ...) making both slots sort by length first.
        # Now Slot A explicitly ignores length so long routes don't crowd out efficient ones.
        eff_ranked = sorted(
            self._routes.values(),
            key=lambda r: r.cost / max(len(r.nodes), 1),
        )

        max_per = self.cfg.route_pool_max_per_customer

        def _admit(rec: RouteRecord) -> bool:
            if rec.nodes in kept:
                return False
            under = all(usage.get(n, 0) < max_per for n in rec.nodes)
            if not under and len(kept) >= limit // 3:
                return False
            kept[rec.nodes] = rec
            for n in rec.nodes:
                usage[n] = usage.get(n, 0) + 1
            return True

        for rec in len_ranked:  # Fill Slot B with longest routes
            if len(kept) >= slot_b:
                break
            _admit(rec)

        for rec in eff_ranked:  # Fill Slot A with cheapest-per-stop routes
            if len(kept) >= limit:
                break
            _admit(rec)

        if len(kept) < limit:  # Backfill if either slot undersaturated
            for rec in len_ranked:
                if len(kept) >= limit:
                    break
                _admit(rec)

        self._routes = kept
        self._cover_to_key = {_cover_key(k): k for k in kept}

    def add_route(self, route: list[int], protected: bool = False) -> None:
        if not route or not _check_route(route, self.inst):
            return
        key = tuple(route)
        if key in self._routes:
            if protected and not self._routes[key].protected:
                self._routes[key] = RouteRecord(
                    nodes=key,
                    cost=self._routes[key].cost,
                    load=self._routes[key].load,
                    slack=self._routes[key].slack,
                    protected=True,
                )
            return
        rec = RouteRecord(
            nodes=key,
            cost=_route_cost_list(route, self.inst),
            load=_route_load(route, self.inst),
            slack=_route_avg_slack(route, self.inst),
            protected=protected,
        )
        cover = _cover_key(key)
        old_key = self._cover_to_key.get(cover)
        if old_key is not None:
            old_rec = self._routes[old_key]
            if _same_cover_priority(old_rec) <= _same_cover_priority(rec):
                return
            del self._routes[old_key]
            del self._cover_to_key[cover]
        self._routes[key] = rec
        self._cover_to_key[cover] = key
        self._trim()

    def add_plan(self, plan: Plan) -> None:
        for r in plan.routes:
            self.add_route(r)

    def records(self, incumbent: Plan | None = None) -> list[RouteRecord]:
        recs = dict(self._routes)
        if incumbent is not None:
            for r in incumbent.routes:
                key = tuple(r)
                recs[key] = RouteRecord(
                    nodes=key,
                    cost=_route_cost_list(r, incumbent.inst),
                    load=_route_load(r, incumbent.inst),
                    slack=_route_avg_slack(r, incumbent.inst),
                )
        best_by_cover: dict[tuple[int, ...], RouteRecord] = {}
        for rec in recs.values():
            cover = _cover_key(rec.nodes)
            incumbent_rec = best_by_cover.get(cover)
            if incumbent_rec is None or _same_cover_priority(rec) < _same_cover_priority(incumbent_rec):
                best_by_cover[cover] = rec
        return sorted(best_by_cover.values(), key=self._priority)


def _sp_vehicle_penalty(inst: Inst, cfg: Config) -> float:
    return cfg.sp_vehicle_penalty_scale * max(inst.max_dist, 1.0) * max(inst.n, 1)


_MILP_CACHE_LIMIT = 256

# id(array) -> (array, digest). Holding the array itself is the point: `id()` is
# unique only among *live* objects, so a bare id key lets a freed heatmap's
# address be reused by a different array, hitting the entry cached for the old
# one and returning a recombination optimised against a different objective.
# Keeping a strong reference makes the id stable for as long as the entry lives,
# which is what makes the identity check below sound.
_HEATMAP_DIGESTS: dict[int, tuple[np.ndarray, bytes]] = {}
_HEATMAP_DIGEST_LIMIT = 4


def _heatmap_key(heatmap: np.ndarray | None) -> bytes | None:
    """Stable key identifying ``heatmap`` for the MILP memo.

    Digests the buffer, but memoises per array object: the digest is ~14 ms on
    the 8 MB heatmap at n=1000, and this runs before the memo lookup, so hashing
    on every call would blunt the very cache it keys. Heatmaps are assigned
    wholesale and only ever read, never mutated in place, so a cached digest
    cannot go stale.
    """
    if heatmap is None:
        return None
    cached = _HEATMAP_DIGESTS.get(id(heatmap))
    if cached is not None and cached[0] is heatmap:
        return cached[1]
    arr = np.ascontiguousarray(heatmap)
    digest = hashlib.blake2b(memoryview(arr).cast("B"), digest_size=16).digest()
    if len(_HEATMAP_DIGESTS) >= _HEATMAP_DIGEST_LIMIT:
        _HEATMAP_DIGESTS.clear()
    _HEATMAP_DIGESTS[id(heatmap)] = (heatmap, digest)
    return digest


def _milp_cache_store(cache: dict | None, key, value) -> None:
    """Write a memo entry under a bounded size guard.

    Applies to the "no solution" path too: a run whose MILP consistently fails
    would otherwise grow the dict without bound.
    """
    if cache is None or key is None:
        return
    if len(cache) > _MILP_CACHE_LIMIT:
        cache.clear()
    cache[key] = value


def _select_milp_columns(route_records: list[RouteRecord], inst: Inst, max_cols: int) -> list[RouteRecord]:
    """Truncate the SP column set, guaranteeing every customer keeps a column.

    Returns exactly ``route_records[:max_cols]`` — same set, same order —
    whenever that slice already covers every customer, so the search trajectory
    is untouched in the normal case. Only when some customer would lose every
    column (the case where ``_milp_recombine`` used to die on
    ``row_sums == 0``) are rescue columns swapped in, displacing the
    lowest-priority slice columns that are not themselves sole coverage.

    A broader reshuffle (guaranteeing >=3 columns per customer) was tried here
    and reverted: it perturbed every recombination and cost a vehicle on
    rc1_2_1 in the paired A/B. Keep this surgical.
    """
    if len(route_records) <= max_cols:
        return route_records
    head = list(route_records[:max_cols])

    counts = np.zeros(inst.n + 1, dtype=np.int64)
    for rec in head:
        counts[np.asarray(rec.nodes, dtype=np.int64)] += 1
    missing = set(int(c) for c in np.flatnonzero(counts[1:] == 0) + 1)
    if not missing:
        return head

    rescues: list[RouteRecord] = []
    for rec in route_records[max_cols:]:
        if not missing:
            break
        nodes = set(rec.nodes)
        if nodes & missing:
            rescues.append(rec)
            missing -= nodes

    # Displace lowest-priority head columns whose customers all keep >=1 other
    # column; if too few are droppable, overflow max_cols by the difference —
    # preserving coverage wins over the cap here, because a stranded customer
    # kills the solve outright. `_milp_recombine` bounds the overflow by
    # skipping the MILP altogether past a hard ceiling.
    droppable = 0
    for idx in range(len(head) - 1, -1, -1):
        if droppable >= len(rescues):
            break
        rec = head[idx]
        nodes = np.asarray(rec.nodes, dtype=np.int64)
        if (counts[nodes] >= 2).all():
            counts[nodes] -= 1
            head.pop(idx)
            droppable += 1
    return head + rescues


def _milp_recombine(
    route_records: list[RouteRecord],
    inst: Inst,
    cfg: Config,
    nv_ceiling: int | None = None,
    vehicle_penalty: float | None = None,
    heatmap: np.ndarray | None = None,
    alpha: float = 0.15,
    _stats: dict | None = None,
    _cache: dict | None = None,
) -> Plan | None:
    if not MILP_OK or not route_records:
        return None
    _MILP_MAX_COLS = getattr(cfg, "milp_max_cols", 800)
    pool_size_before = len(route_records)
    route_records = _select_milp_columns(route_records, inst, _MILP_MAX_COLS)
    if getattr(cfg, "log_milp_cols", False):
        print(
            f"[MILP_COLS] pool_size={pool_size_before}, selected={len(route_records)}, capped={pool_size_before > _MILP_MAX_COLS}",
            flush=True,
        )
    # _select_milp_columns may overflow the cap: it must never strand a customer,
    # so it stops displacing head columns once none are droppable, and the excess
    # is bounded only by the number of uncovered customers. Measured at 0
    # overflows across 56 truncating calls (n<=200), but since the cap exists
    # solely to keep SciPy's O(N^2) extraction from hanging, enforce it here
    # rather than trusting that. Recombination is an optional improvement step
    # and every caller already handles None, so skipping is the safe response.
    if len(route_records) > 2 * _MILP_MAX_COLS:
        return None
    n_routes = len(route_records)

    # The same column set is often re-solved with identical parameters within a
    # run (recombination fires on a fixed cadence while the pool is stable), so
    # completed solves are memoised on the pool.
    cache_key = None
    if _cache is not None:
        cache_key = (
            tuple(rec.nodes for rec in route_records),
            nv_ceiling,
            round(vehicle_penalty, 6) if vehicle_penalty is not None else None,
            _heatmap_key(heatmap),
            round(alpha, 6),
        )
        hit = _cache.get(cache_key)
        if hit is not None:
            if hit == ():  # cached "no solution"
                return None
            plan = Plan([list(nodes) for nodes in hit], inst, "SP-RECOMBINE")
            return plan if plan.feasible and _is_exact_cover(plan) else None
    from scipy.sparse import csc_matrix

    rows = []
    cols = []
    data = []
    for ridx, rec in enumerate(route_records):
        for node in rec.nodes:
            rows.append(node - 1)
            cols.append(ridx)
            data.append(1.0)
    cover = csc_matrix((data, (rows, cols)), shape=(inst.n, n_routes), dtype=float)
    row_sums = np.asarray(cover.sum(axis=1)).flatten()
    if np.any(row_sums == 0):
        return None
    # Try 1: Exact Set Partitioning (ub = 1)
    constraints_sp = [LinearConstraint(cover, lb=np.ones(inst.n), ub=np.ones(inst.n))]
    if nv_ceiling is not None:
        cover_nv = csc_matrix(np.ones((1, n_routes), dtype=float))
        constraints_sp.append(LinearConstraint(cover_nv, lb=np.array([0.0]), ub=np.array([float(nv_ceiling)])))

    penalty = vehicle_penalty if vehicle_penalty is not None else _sp_vehicle_penalty(inst, cfg)
    costs = []
    for rec in route_records:
        r_cost = rec.cost
        if heatmap is not None and alpha > 0.0:
            nodes = rec.nodes
            if nodes:
                edges = [(0, nodes[0])]
                for i in range(len(nodes) - 1):
                    edges.append((nodes[i], nodes[i + 1]))
                edges.append((nodes[-1], 0))
                gnn_score = float(np.mean([heatmap[u, v] for u, v in edges]))
                r_cost = rec.cost * (1.0 - alpha * gnn_score)
        costs.append(penalty + r_cost)
    costs = np.array(costs)

    result = milp(
        c=costs,
        constraints=constraints_sp,
        integrality=np.ones(n_routes, dtype=int),
        bounds=Bounds(np.zeros(n_routes), np.ones(n_routes)),
        options={"time_limit": float(cfg.sp_time_limit), "disp": False},
    )

    # Try 2: Set Covering Relaxation (ub = inf) + Duplicate Cleanup if exact partitioning failed
    if result is None or result.x is None or getattr(result, "status", 1) != 0:
        constraints_sc = [LinearConstraint(cover, lb=np.ones(inst.n), ub=np.full(inst.n, np.inf))]
        if nv_ceiling is not None:
            constraints_sc.append(LinearConstraint(cover_nv, lb=np.array([0.0]), ub=np.array([float(nv_ceiling)])))
        result = milp(
            c=costs,
            constraints=constraints_sc,
            integrality=np.ones(n_routes, dtype=int),
            bounds=Bounds(np.zeros(n_routes), np.ones(n_routes)),
            options={"time_limit": float(cfg.sp_time_limit), "disp": False},
        )

    if _stats is not None:
        _stats["calls"] = _stats.get("calls", 0) + 1
        if result is not None and getattr(result, "status", None) == 1:
            _stats["timeouts"] = _stats.get("timeouts", 0) + 1
        _stats["milp_fired"] = True
        _stats["milp_cadence_skip"] = 0

    if result is None or result.x is None:
        _milp_cache_store(_cache, cache_key, ())
        return None

    chosen = [list(route_records[i].nodes) for i, v in enumerate(result.x) if v >= 0.5]
    if not _is_exact_cover(Plan(chosen, inst, "SP-RECOMBINE")):
        chosen = _cleanup_duplicate_nodes(chosen, inst)

    plan = Plan(chosen, inst, "SP-RECOMBINE")
    _milp_cache_store(_cache, cache_key, tuple(tuple(r) for r in chosen))
    return plan if plan.feasible and _is_exact_cover(plan) else None


def _cleanup_duplicate_nodes(routes: list[list[int]], inst: Inst) -> list[list[int]]:
    """Remove duplicate customer visits from Set Covering routes while maintaining feasibility."""
    cleaned = [r[:] for r in routes if r]
    node_counts: dict[int, int] = {}
    for r in cleaned:
        for node in r:
            node_counts[node] = node_counts.get(node, 0) + 1

    duplicates = [node for node, count in node_counts.items() if count > 1]
    if not duplicates:
        return cleaned

    for dup in duplicates:
        occurrences = []
        for r_idx, r in enumerate(cleaned):
            if dup in r:
                pos = r.index(dup)
                r_without = r[:pos] + r[pos + 1 :]
                old_cost = _route_cost_list(r, inst)
                new_cost = _route_cost_list(r_without, inst) if r_without else 0.0
                saved_cost = old_cost - new_cost
                occurrences.append((saved_cost, r_idx, pos))

        occurrences.sort(key=lambda x: x[0], reverse=True)
        to_remove = occurrences[:-1]
        for _, r_idx, _pos in to_remove:
            if dup in cleaned[r_idx]:
                cleaned[r_idx].remove(dup)

    return [r for r in cleaned if r]


def _greedy_recombine(route_records: list[RouteRecord], incumbent: Plan, nv_ceiling: int | None = None) -> Plan:
    uncovered = set(range(1, incumbent.inst.n + 1))
    selected: list[list[int]] = []
    used: set[tuple[int, ...]] = set()
    while uncovered:
        best_rec, best_score = None, -float("inf")
        for rec in route_records:
            if rec.nodes in used:
                continue
            rec_nodes = set(rec.nodes)
            if not rec_nodes or not rec_nodes.issubset(uncovered):
                continue
            gain = len(rec_nodes)
            score = gain * 10.0 + len(rec.nodes) - rec.cost / max(len(rec.nodes), 1)
            if score > best_score:
                best_score, best_rec = score, rec
        if best_rec is None:
            break
        selected.append(list(best_rec.nodes))
        used.add(best_rec.nodes)
        uncovered.difference_update(best_rec.nodes)
        if nv_ceiling is not None and len(selected) > nv_ceiling:
            return incumbent.copy()
    plan = Plan(selected, incumbent.inst, "SP-GREEDY")
    for node in sorted(uncovered):
        _insert_customer(plan, node, incumbent.inst)
    if nv_ceiling is not None and plan.nv > nv_ceiling:
        return incumbent.copy()
    return plan if plan.feasible and _is_exact_cover(plan) else incumbent.copy()


def recombine_with_route_pool(
    incumbent: Plan,
    pool: RoutePool,
    cfg: Config,
    nv_ceiling: int | None = None,
    nv_target: int | None = None,
    td_only: bool = False,
    heatmap: np.ndarray | None = None,
    alpha: float = 0.15,
    _stats: dict | None = None,
) -> Plan:
    pool.add_plan(incumbent)
    recs = pool.records(incumbent)
    if not recs:
        return incumbent.copy()

    # ── TD-only fast path ────────────────────────────────────────────────────
    if td_only:
        effective_ceiling = nv_ceiling if nv_ceiling is not None else incumbent.nv
        candidate = _milp_recombine(
            recs,
            incumbent.inst,
            cfg,
            nv_ceiling=effective_ceiling,
            vehicle_penalty=0.0,
            _stats=_stats,
            _cache=pool._milp_cache,
        )
        if candidate is None:
            candidate = _greedy_recombine(recs, incumbent, nv_ceiling=effective_ceiling)
        if (
            candidate.feasible
            and _is_exact_cover(candidate)
            and candidate.nv <= effective_ceiling
            and candidate.cost + 1e-6 < incumbent.cost
        ):
            return candidate
        return incumbent.copy()

    mean_cost = float(np.mean([r.cost for r in recs])) if recs else 100.0
    use_penalty = (nv_ceiling is not None) or (nv_target is not None)
    effective_ceiling = nv_target if nv_target is not None else nv_ceiling

    if not use_penalty:
        # Standard recombination: no NV pressure
        candidate = _milp_recombine(
            recs,
            incumbent.inst,
            cfg,
            nv_ceiling=effective_ceiling,
            vehicle_penalty=0.0,
            heatmap=heatmap,
            alpha=alpha,
            _stats=_stats,
            _cache=pool._milp_cache,
        )
        if candidate is None:
            candidate = _greedy_recombine(recs, incumbent, nv_ceiling=effective_ceiling)
        return candidate if candidate.dominates(incumbent) else incumbent.copy()

    # NV-targeted: try multiple penalty scales so one scale finds the partition
    # if it exists in the pool, even when mean_cost * 2.0 is insufficient.
    # Scales 30.0/50.0 are extreme: they force the MILP to prioritize NV
    # reduction over TD quality, which is correct when BKS-guided routes
    # are in the pool and we know the NV-target partition exists.
    penalty_scales = (2.0, 5.0, 12.0, 30.0, 50.0)
    per_query_limit = max(1.0, cfg.sp_time_limit / max(len(penalty_scales) - 2, 1))

    # Temporarily override time limit per query
    class _TmpCfg:
        def __getattr__(self, name):
            if name == "sp_time_limit":
                return per_query_limit
            return getattr(cfg, name)

    tmp_cfg = _TmpCfg()

    k_max = int(effective_ceiling if effective_ceiling is not None else incumbent.nv)
    c_max_route = max((r.cost for r in recs), default=1.0)
    eps_margin = getattr(cfg, "sp_cert_margin", 0.20)
    lambda_cert = (1.0 + eps_margin) * k_max * c_max_route
    enforce_cert = getattr(cfg, "sp_enforce_certified_penalty", True)

    for scale in penalty_scales:
        grid_value = max(cfg.sp_vehicle_penalty_scale, mean_cost * scale)
        if enforce_cert and scale >= 50.0:
            penalty = max(grid_value, lambda_cert)
        else:
            penalty = grid_value
        candidate = _milp_recombine(
            recs,
            incumbent.inst,
            tmp_cfg,
            nv_ceiling=effective_ceiling,
            vehicle_penalty=penalty,
            heatmap=heatmap,
            alpha=alpha,
            _stats=_stats,
            _cache=pool._milp_cache,
        )
        if candidate is not None and (effective_ceiling is None or candidate.nv <= effective_ceiling):
            # Run LS at the new NV to recover TD
            from .local_search import local_search

            candidate = local_search(
                candidate,
                max_passes=1,
                nv_ceiling=candidate.nv,
                max_ls_moves=10,
            )
            if (
                candidate.feasible
                and _is_exact_cover(candidate)
                and (effective_ceiling is None or candidate.nv <= effective_ceiling)
            ):
                return candidate

    # All scales failed: greedy fallback
    candidate = _greedy_recombine(recs, incumbent, nv_ceiling=effective_ceiling)
    if effective_ceiling is not None and candidate.nv > effective_ceiling:
        return incumbent.copy()
    return (
        candidate
        if candidate.feasible and _is_exact_cover(candidate) and candidate.dominates(incumbent)
        else incumbent.copy()
    )

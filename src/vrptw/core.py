from __future__ import annotations

import os

_N_PARALLEL = min(3, max(1, (os.cpu_count() or 1) // 2))
_NUMBA_THREADS = max(1, (os.cpu_count() or 1) // _N_PARALLEL)
os.environ.setdefault("NUMBA_NUM_THREADS", str(_NUMBA_THREADS))
os.environ.setdefault("OMP_NUM_THREADS", str(_NUMBA_THREADS))
os.environ.setdefault("MKL_NUM_THREADS", str(_NUMBA_THREADS))


import numpy as np
from numba import njit

from .config import BKS


class Inst:
    def __init__(self, raw: dict):
        self.name = raw["name"]
        data = raw["data"]
        self.capacity = raw["capacity"]
        self.coords = data[:, 1:3]
        self.demands = data[:, 3]
        self.ready_times = data[:, 4]
        self.due_times = data[:, 5]
        self.service_times = data[:, 6]
        self.horizon = self.due_times[0]
        if self.horizon <= 0:
            raise ValueError(
                f"Instance '{self.name}': depot due_time={self.horizon} is invalid. "
                f"Expected a positive planning horizon (due_times[0] > 0)."
            )
        self.n = len(data) - 1
        diff = self.coords[:, None, :] - self.coords[None, :, :]
        self.dist = np.sqrt((diff**2).sum(axis=2))
        self.max_dist = float(self.dist.max())
        self.tw_width = self.due_times - self.ready_times
        self.max_tw_width = float(self.tw_width[1:].max() + 1e-9)
        avg_cross_time = self.max_dist
        self.tw_tight_frac = sum(1 for i in range(1, self.n + 1) if self.tw_width[i] < 0.5 * avg_cross_time) / max(
            self.n, 1
        )

        # Precompute k-nearest neighbors for each customer (excluding depot 0, excluding self)
        k_neighbors = min(15, self.n - 1) if self.n > 1 else 0
        self.neighbors_k = [[]]  # depot 0 has no neighbors in this list
        for i in range(1, self.n + 1):
            dists = self.dist[i].copy()
            dists[0] = float("inf")  # exclude depot
            dists[i] = float("inf")  # exclude self
            # NB: argpartition would be asymptotically cheaper here, but it breaks
            # ties among equidistant customers differently from argsort, which
            # perturbs every downstream kNN filter and changes search results.
            # The saving (~0.16s at n=1000, once per instance) is not worth it.
            nearest = list(np.argsort(dists)[:k_neighbors])
            self.neighbors_k.append(nearest)

        # Precompute Shaw relatedness components and matrix
        max_d_val = self.max_dist + 1e-9
        max_tw_val = max(self.due_times - self.ready_times) + 1e-9
        self.spatial_rel = (self.dist / max_d_val).astype(np.float32)
        ready_diff = np.abs(self.ready_times[:, None] - self.ready_times[None, :])
        self.temporal_rel = (ready_diff / max_tw_val).astype(np.float32)
        demand_diff = np.abs(self.demands[:, None] - self.demands[None, :])
        self.demand_rel = (demand_diff / self.capacity).astype(np.float32)
        self.relatedness = 0.5 * self.spatial_rel + 0.4 * self.temporal_rel + 0.1 * self.demand_rel
        # Optional roadmap extensions: Multi-depot coordinates & Heterogeneous vehicle capacities
        self.multi_depots = raw.get("multi_depots", None)
        self.vehicle_capacities = raw.get("vehicle_capacities", None)


def load_solomon_instance(path: str) -> Inst:
    """Parse a Solomon / Gehring-Homberger instance file into an :class:`Inst`.

    Note that :class:`Inst` takes an already-parsed dict, not a path — passing a
    path silently produced a TypeError at several call sites.
    """
    with open(path, encoding="utf-8") as fh:
        lines = fh.readlines()
    name = lines[0].strip()
    capacity = float(lines[4].strip().split()[1])
    rows = [list(map(float, ln.split())) for ln in lines[9:] if ln.strip()]
    return Inst({"name": name, "capacity": capacity, "data": np.array(rows)})


@njit(cache=True)
def _route_cost(route: np.ndarray, dist: np.ndarray) -> float:
    cost = dist[0, route[0]]
    for idx in range(len(route) - 1):
        cost += dist[route[idx], route[idx + 1]]
    return cost + dist[route[-1], 0]


@njit(cache=True)
def _route_violations(route, demands, capacity, ready, due, service, dist):
    load = 0.0
    tw_violation = 0.0
    t, prev = 0.0, 0
    for node in route:
        load += demands[node]
        t += dist[prev, node]
        if t > due[node]:
            tw_violation += t - due[node]
        t = max(t, ready[node]) + service[node]
        prev = node

    # Return to depot
    t += dist[prev, 0]
    if t > due[0]:
        tw_violation += t - due[0]

    cap_violation = max(0.0, load - capacity)
    return cap_violation, tw_violation


@njit(cache=True)
def _route_ok(route, demands, capacity, ready, due, service, dist) -> bool:
    load = 0.0
    for node in route:
        load += demands[node]
    if load > capacity:
        return False

    t, prev = 0.0, 0
    for node in route:
        t += dist[prev, node]
        if t < ready[node]:
            t = ready[node]
        if t > due[node]:
            return False
        t += service[node]
        prev = node
    return t + dist[prev, 0] <= due[0]


class Plan:
    __slots__ = (
        "routes",
        "inst",
        "_cost",
        "_ok",
        "algo",
        "_route_arrays",
        "_violation_capacity",
        "_violation_tw",
    )

    def __init__(self, routes: list[list[int]], inst: Inst, algo: str = ""):
        self.routes = [r for r in routes if r]
        self.inst = inst
        self._cost: float | None = None
        self._ok: bool | None = None
        self.algo = algo
        self._route_arrays: list[np.ndarray] | None = None
        self._violation_capacity: float | None = None
        self._violation_tw: float | None = None

    @property
    def route_arrays(self) -> list[np.ndarray]:
        if self._route_arrays is None:
            self._route_arrays = [np.array(r, np.int64) for r in self.routes]
        return self._route_arrays

    @property
    def cost(self) -> float:
        if self._cost is None:
            self._cost = sum(_route_cost(arr, self.inst.dist) for arr in self.route_arrays)
        return self._cost

    @property
    def feasible(self) -> bool:
        if self._ok is None:
            self._ok = all(
                _route_ok(
                    arr,
                    self.inst.demands,
                    self.inst.capacity,
                    self.inst.ready_times,
                    self.inst.due_times,
                    self.inst.service_times,
                    self.inst.dist,
                )
                for arr in self.route_arrays
            )
        return self._ok

    @property
    def violation_capacity(self) -> float:
        if self._violation_capacity is None:
            self._compute_violations()
        return self._violation_capacity

    @property
    def violation_tw(self) -> float:
        if self._violation_tw is None:
            self._compute_violations()
        return self._violation_tw

    def _compute_violations(self) -> None:
        cap_viol = 0.0
        tw_viol = 0.0
        for arr in self.route_arrays:
            c_v, t_v = _route_violations(
                arr,
                self.inst.demands,
                self.inst.capacity,
                self.inst.ready_times,
                self.inst.due_times,
                self.inst.service_times,
                self.inst.dist,
            )
            cap_viol += c_v
            tw_viol += t_v
        self._violation_capacity = cap_viol
        self._violation_tw = tw_viol

    @property
    def nv(self) -> int:
        return len(self.routes)

    @property
    def on_time_rate(self) -> float:
        on_time = total = 0
        for route in self.routes:
            t, prev = 0.0, 0
            for node in route:
                t += self.inst.dist[prev, node]
                t = max(t, self.inst.ready_times[node])
                total += 1
                if t <= self.inst.due_times[node]:
                    on_time += 1
                t += self.inst.service_times[node]
                prev = node
        return on_time / max(total, 1)

    def gap(self) -> tuple[float | None, int | None]:
        bks = BKS.get(self.inst.name)
        if not bks:
            return None, None
        return (self.cost - bks["td"]) / bks["td"] * 100, int(self.nv) - int(bks["nv"])

    def dominates(self, other: Plan) -> bool:
        return self.nv < other.nv or (self.nv == other.nv and self.cost < other.cost - 1e-6)

    def copy(self) -> Plan:
        p = Plan([r[:] for r in self.routes], self.inst, self.algo)
        p._cost = self._cost
        p._ok = self._ok
        p._violation_capacity = self._violation_capacity
        p._violation_tw = self._violation_tw
        if self._route_arrays is not None:
            p._route_arrays = [arr.copy() for arr in self._route_arrays]
        return p

    def invalidate(self) -> None:
        self._cost = None
        self._ok = None
        self._route_arrays = None
        self._violation_capacity = None
        self._violation_tw = None

    def calculate_workload_balance(self) -> float:
        """
        Calculates driver workload balance as the variance of route durations.
        Lower variance indicates more balanced work distribution among drivers.
        """
        if not self.routes:
            return 0.0
        durations = [_route_duration_no_return(r, self.inst) for r in self.routes]
        return float(np.var(durations))

    def calculate_pareto_metrics(self) -> dict[str, float]:
        """
        Generates multi-objective Pareto evaluation metrics for enterprise dispatchers:
        - nv: Fleet size
        - td: Total travel distance
        - workload_variance: Driver workload balance variance
        - delay_risk: Time window tightness / risk score
        """
        return {
            "nv": float(self.nv),
            "td": float(self.cost),
            "workload_variance": self.calculate_workload_balance(),
            "delay_risk": float(1.0 - self.on_time_rate),
        }


def _invalidate(plan: Plan) -> Plan:
    plan.invalidate()
    return plan


def _route_duration_no_return(route: list[int], inst: Inst) -> float:
    """Calculate duration of a route without returning to depot."""
    if not route:
        return 0.0
    t, prev = 0.0, 0
    for node in route:
        t += inst.dist[prev, node]
        t = max(t, float(inst.ready_times[node])) + float(inst.service_times[node])
        prev = node
    return float(t)


def _check_route(route: list[int] | np.ndarray, inst: Inst) -> bool:
    """Check capacity + time-window feasibility (delegates to numba ``_route_ok``)."""
    arr = route if isinstance(route, np.ndarray) else np.array(route, np.int64)
    return bool(
        _route_ok(
            arr,
            inst.demands,
            inst.capacity,
            inst.ready_times,
            inst.due_times,
            inst.service_times,
            inst.dist,
        )
    )


def _avg_slack(plan: Plan) -> float:
    inst = plan.inst
    slack_sum = count = 0
    for route in plan.routes:
        t, prev = 0.0, 0
        for node in route:
            t += inst.dist[prev, node]
            t = max(t, inst.ready_times[node])
            slack_sum += max(0.0, inst.due_times[node] - t)
            t += inst.service_times[node]
            prev = node
            count += 1
    return (slack_sum / count) / max(inst.horizon, 1) if count else 0.0


def _plan_spread(plan: Plan, inst: Inst) -> tuple[float, float]:
    lengths = [len(r) for r in plan.routes] or [0]
    loads = [sum(inst.demands[n] for n in r) for r in plan.routes] or [0]
    rb = min(float(np.std(lengths)) / max(float(np.mean(lengths)), 1.0), 1.0) if len(lengths) > 1 else 0.0
    lb = min(float(np.std(loads)) / max(float(inst.capacity), 1.0), 1.0)
    return rb, lb


def _fleet_fill(plan: Plan) -> float:
    if not plan.routes:
        return 0.0
    capacity = max(plan.inst.capacity, 1)
    fills = [plan.inst.demands[arr].sum() / capacity for arr in plan.route_arrays]
    return float(np.mean(fills))

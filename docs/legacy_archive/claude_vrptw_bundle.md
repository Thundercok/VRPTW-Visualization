# NAMI VRPTW Solver Models & Core Algorithms Bundle

This document bundles the complete, up-to-date Python implementation of the **NAMI** (AI-driven VRPTW research and dispatch platform) solver suite. It includes the config settings, low-level feasibility checking, heuristic construction methods, destroy & repair operators, local search routines, route-recombination pool, and the reinforcement learning (DDQN) plateau/operator controllers.

Use this code block as reference context to help design optimizations to beat Google **OR-Tools** (CP-SAT Guided Local Search).

---

## Codebase Map
1. **`config.py`**: Configuration parameters, best-known-solutions (BKS) reference, Mode specifications.
2. **`core.py`**: Core classes (`Inst` for problem definition, `Plan` for solution routes) with fast Numba-JIT constraints check.
3. **`heuristics.py`**: Greedy initialization procedures.
4. **`operators.py`**: The 10 Destroy and 5 Repair operators for the ALNS search.
5. **`local_search.py`**: Post-destruction neighborhood local search moves, route compaction, ejection-chain elimination, and buffered route elimination.
6. **`pool.py`**: Set-partitioning recombination using MILP/greedy route pools.
7. **`rl.py`**: Prioritized Experience Replay buffer, Dueling DDQN Q-networks, Thompson Bandit decay, and Learned Acceptance Criterion (LAC).
8. **`solvers.py`**: The top-level solver classes (`ALNSSolver`, `HybridDDQNSolver`, `HybridFixedSolver`, `HybridRuleSolver`).

---

## File: `src/vrptw/config.py`

```python
from __future__ import annotations

import os
from dataclasses import dataclass, field

import pandas as pd

# ---------------------------------------------------------------------------
# BKS table
# ---------------------------------------------------------------------------
BKS: dict[str, dict[str, float]] = {
    "C101": {"nv": 10, "td": 828.94},
    "C102": {"nv": 10, "td": 828.94},
    "C103": {"nv": 10, "td": 828.06},
    "C104": {"nv": 10, "td": 824.78},
    "C105": {"nv": 10, "td": 828.94},
    "C106": {"nv": 10, "td": 828.94},
    "C107": {"nv": 10, "td": 828.94},
    "C108": {"nv": 10, "td": 828.94},
    "C109": {"nv": 10, "td": 828.94},
    # C2
    "C201": {"nv": 3, "td": 591.56},
    "C202": {"nv": 3, "td": 591.56},
    "C203": {"nv": 3, "td": 591.17},
    "C204": {"nv": 3, "td": 590.60},
    "C205": {"nv": 3, "td": 588.88},
    "C206": {"nv": 3, "td": 588.49},
    "C207": {"nv": 3, "td": 588.29},
    "C208": {"nv": 3, "td": 588.32},
    # R1
    "R101": {"nv": 19, "td": 1650.80},
    "R102": {"nv": 17, "td": 1486.12},
    "R103": {"nv": 13, "td": 1292.68},
    "R104": {"nv": 9, "td": 1007.31},
    "R105": {"nv": 14, "td": 1377.11},
    "R106": {"nv": 12, "td": 1252.03},
    "R107": {"nv": 10, "td": 1104.66},
    "R108": {"nv": 9, "td": 960.88},
    "R109": {"nv": 11, "td": 1194.73},
    "R110": {"nv": 10, "td": 1118.84},
    "R111": {"nv": 10, "td": 1096.72},
    "R112": {"nv": 9, "td": 982.14},
    "R201": {"nv": 4, "td": 1252.37},
    "R202": {"nv": 3, "td": 1191.70},
    "R203": {"nv": 3, "td": 939.50},
    "R204": {"nv": 2, "td": 825.52},
    "R205": {"nv": 3, "td": 994.43},
    "R206": {"nv": 3, "td": 906.14},
    "R207": {"nv": 2, "td": 890.61},
    "R208": {"nv": 2, "td": 726.82},
    "R209": {"nv": 3, "td": 909.16},
    "R210": {"nv": 3, "td": 939.37},
    "R211": {"nv": 2, "td": 885.71},
    "RC101": {"nv": 14, "td": 1696.94},
    "RC102": {"nv": 12, "td": 1554.75},
    "RC103": {"nv": 11, "td": 1261.67},
    "RC104": {"nv": 10, "td": 1135.48},
    "RC105": {"nv": 13, "td": 1629.44},
    "RC106": {"nv": 11, "td": 1424.73},
    "RC107": {"nv": 11, "td": 1230.48},
    "RC108": {"nv": 10, "td": 1139.82},
    "RC201": {"nv": 4, "td": 1406.94},
    "RC202": {"nv": 3, "td": 1365.65},
    "RC203": {"nv": 3, "td": 1049.62},
    "RC204": {"nv": 3, "td": 798.46},
    "RC205": {"nv": 4, "td": 1297.65},
    "RC206": {"nv": 3, "td": 1146.32},
    "RC207": {"nv": 3, "td": 1061.14},
    "RC208": {"nv": 3, "td": 828.14},
}

ALGO_ORTOOLS = "OR-Tools"
ALGO_ALNS_BASE = "ALNS-Base"
ALGO_HYBRID_FIXED = "Hybrid-Fixed"
ALGO_HYBRID_RULE = "Hybrid-Rule"
ALGO_HYBRID_DDQN = "Hybrid-DDQN"
ALGO_HYBRID_DDQN_TRANSFER = "Hybrid-DDQN-Transfer"
ALGO_HYBRID_DDQN_TRANSFER_RC2 = "Hybrid-DDQN-Transfer-RC2"
ALGO_HYBRID_DDQN_TRANSFER_DR = "Hybrid-DDQN-Transfer-DR"

ALGO_ORDER = [
    ALGO_ORTOOLS,
    ALGO_ALNS_BASE,
    ALGO_HYBRID_FIXED,
    ALGO_HYBRID_RULE,
    ALGO_HYBRID_DDQN,
    ALGO_HYBRID_DDQN_TRANSFER,
    ALGO_HYBRID_DDQN_TRANSFER_RC2,
    ALGO_HYBRID_DDQN_TRANSFER_DR,
]

LEGACY_ALGO_LABELS = {
    "ALNS": ALGO_ALNS_BASE,
    "ALNS-Base": ALGO_ALNS_BASE,
    "ALNS+": ALGO_HYBRID_FIXED,
    "ALNS-FAIR": ALGO_HYBRID_FIXED,
    "Hybrid-Fixed": ALGO_HYBRID_FIXED,
    "ALNS++": ALGO_HYBRID_RULE,
    "SCHED-ALNS": ALGO_HYBRID_RULE,
    "Hybrid-Rule": ALGO_HYBRID_RULE,
    "DDQN-ALNS": ALGO_HYBRID_DDQN,
    "PLATEAU-HYBRID": ALGO_HYBRID_DDQN,
    "Hybrid-DDQN": ALGO_HYBRID_DDQN,
    "DDQN-ALNS*": ALGO_HYBRID_DDQN_TRANSFER,
    "DDQN-ALNSâ˜…": ALGO_HYBRID_DDQN_TRANSFER,
    "Hybrid-DDQN-Transfer": ALGO_HYBRID_DDQN_TRANSFER,
    "Hybrid-DDQN-Transfer-RC2": ALGO_HYBRID_DDQN_TRANSFER_RC2,
    "Hybrid-DDQN-Transfer-DR": ALGO_HYBRID_DDQN_TRANSFER_DR,
}


def canonical_algo_label(label: str) -> str:
    if label in LEGACY_ALGO_LABELS:
        return LEGACY_ALGO_LABELS[label]
    normalized = label.lower().replace("_", "-")
    for k, v in LEGACY_ALGO_LABELS.items():
        if k.lower().replace("_", "-") == normalized:
            return v
    for val in ALGO_ORDER:
        if val.lower().replace("_", "-") == normalized:
            return val
    return label


def normalize_algorithm_frame(df: pd.DataFrame) -> pd.DataFrame:
    if "Algorithm" not in df.columns:
        return df
    out = df.copy()
    out["Algorithm"] = out["Algorithm"].map(canonical_algo_label)
    extra = [a for a in out["Algorithm"].dropna().unique() if a not in ALGO_ORDER]
    out["Algorithm"] = pd.Categorical(out["Algorithm"], categories=ALGO_ORDER + extra, ordered=True)
    sort_cols = [c for c in ("Dataset", "Instance", "Algorithm") if c in out.columns]
    if sort_cols:
        out = out.sort_values(sort_cols).reset_index(drop=True)
    return out


def default_data_path() -> str:
    candidates = [
        "./data/Solomon",
        os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "data", "Solomon"),
        os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "docs", "data", "Solomon"
        ),
        "/workspace/data/Solomon",
        "/root/data/Solomon",
        "/kaggle/input/vrptw-benchmark-datasets/data/Solomon",
        "/kaggle/input/datasets/senju14/vrptw-benchmark-datasets/data/Solomon",
        "/content/vrptw-benchmark/data/Solomon",
    ]
    for path in candidates:
        if os.path.isdir(path):
            return path
    return candidates[0]


def default_output_dir() -> str:
    for d in ("/workspace", "/root", "/kaggle/working", "/content"):
        if os.path.exists(d):
            return d
    return os.getcwd()


# ---------------------------------------------------------------------------
# Config â€” tuned for i7-14700KF (28 cores), n_runs=3
# ---------------------------------------------------------------------------
@dataclass
class Config:
    data_path: str = field(default_factory=default_data_path)
    output_dir: str = field(default_factory=default_output_dir)

    # â”€â”€ iterations (reduced from 3500; early-stop exits stagnation faster) â”€
    alns_iterations: int = 1200
    hybrid_iterations: int = 1200
    early_stop_patience: int = 250
    polish_iterations: int = 80
    polish_patience: int = 40

    destroy_ratio_min: float = 0.10
    destroy_ratio_max: float = 0.40
    temp_control: float = 0.05
    temp_decay: float = 0.99975
    sigma1: int = 33
    sigma2: int = 9
    sigma3: int = 3
    weight_decay: float = 0.10
    segment_size: int = 100
    max_wall_hours: float = 9.5
    n_runs: int = 5
    seed: int = 42

    # â”€â”€ plateau controller â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    ctrl_state_dim: int = 12
    ctrl_hidden: int = 128
    ctrl_lr: float = 3e-4
    ctrl_gamma: float = 0.95
    ctrl_buffer: int = 20_000
    ctrl_batch: int = 64
    ctrl_target_freq: int = 100
    ctrl_eps_start: float = 0.40
    ctrl_eps_end: float = 0.02
    ctrl_eps_decay: float = 0.9997
    ctrl_start: int = 24
    plateau_start: int = 72
    ctrl_start_floor: int = 10  # minimum non-improvement threshold floor to trigger plateau controller
    nv_increase_penalty: float = 15.0
    rl_recombine_min_routes: int = 24
    ctrl_tau: float = 0.005  # soft target update rate for PlateauController
    per_beta_steps: int = 50_000  # steps over which beta anneals 0.4 â†’ 1.0
    # â”€â”€ operator controller â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    op_state_dim: int = 15
    op_hidden: int = 128
    op_lr: float = 3e-4
    op_gamma: float = 0.97
    op_buffer: int = 30_000
    op_batch: int = 64
    op_target_freq: int = 120
    op_eps_start: float = 0.35
    op_eps_end: float = 0.02
    op_eps_decay: float = 0.9996
    op_warmup: int = 256
    op_prior_strength: float = 0.55
    op_bandit_strength: float = 0.20
    op_tau: float = 0.005  # soft target update rate for OperatorController

    bandit_decay: float = 0.95
    bandit_prior_strength: float = 0.18
    potential_nv_scale: float = 15.0
    potential_cost_scale: float = 0.18
    segment_reward_scale: float = 0.30
    iteration_reward_scale: float = 0.45

    # â”€â”€ route pool / set-partitioning â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    route_pool_limit: int = 480
    route_pool_max_per_customer: int = 18
    sp_time_limit: float = 4.0
    sp_vehicle_penalty_scale: float = 200.0

    # â”€â”€ polish â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    polish_ls_passes: int = 2
    max_ls_moves: int = 15
    recombine_after_main_search: bool = True
    recombine_after_polish: bool = True

    # â”€â”€ transfer â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    transfer_epochs: int = 1
    transfer_shuffle: bool = True
    rc2_transfer_split: int = 4

    # â”€â”€ elite archive â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    elite_archive_k: int = 5

    # â”€â”€ OR-Tools â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    ortools_time_limit: float = 60.0

    # â”€â”€ Learned Acceptance Criterion â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    lac_enabled: bool = True
    lac_state_dim: int = 9
    lac_hidden: int = 48
    lac_lr: float = 1e-3
    lac_warmup: int = 300
    lac_horizon: int = 80
    lac_train_freq: int = 20
    lac_buf_size: int = 5000
    lac_batch: int = 64  # batch size for training the learned acceptance criterion
    # â”€â”€ domain randomization â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    domain_randomization_epochs: int = 20
    domain_randomization_batch: int = 15

    def validate(self) -> None:
        """Validate configuration settings to prevent runtime failures during long-running tasks."""
        if self.alns_iterations < 0:
            raise ValueError(f"alns_iterations must be >= 0, got {self.alns_iterations}")
        if self.hybrid_iterations < 0:
            raise ValueError(f"hybrid_iterations must be >= 0, got {self.hybrid_iterations}")
        if self.early_stop_patience <= 0:
            raise ValueError(f"early_stop_patience must be > 0, got {self.early_stop_patience}")
        if self.max_ls_moves <= 0:
            raise ValueError(f"max_ls_moves must be > 0, got {self.max_ls_moves}")
        if not (0.0 <= self.destroy_ratio_min < self.destroy_ratio_max <= 1.0):
            raise ValueError(
                f"Invalid destroy ratios: min={self.destroy_ratio_min}, max={self.destroy_ratio_max}. "
                "Must satisfy 0.0 <= min < max <= 1.0"
            )
        if self.max_wall_hours <= 0.0:
            raise ValueError(f"max_wall_hours must be > 0, got {self.max_wall_hours}")
        if self.n_runs < 1:
            raise ValueError(f"n_runs must be >= 1, got {self.n_runs}")
        if self.ctrl_tau <= 0.0 or self.ctrl_tau >= 1.0:
            raise ValueError(f"ctrl_tau must be > 0 and < 1, got {self.ctrl_tau}")
        if self.op_tau <= 0.0 or self.op_tau >= 1.0:
            raise ValueError(f"op_tau must be > 0 and < 1, got {self.op_tau}")
        if self.per_beta_steps <= 0:
            raise ValueError(f"per_beta_steps must be > 0, got {self.per_beta_steps}")
        if self.lac_batch < 16:
            raise ValueError(f"lac_batch must be >= 16, got {self.lac_batch}")
        if self.ctrl_start_floor < 1:
            raise ValueError(f"ctrl_start_floor must be >= 1, got {self.ctrl_start_floor}")


# ---------------------------------------------------------------------------
# Mode specifications
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class ModeSpec:
    name: str
    destroy_scale: float
    temp_boost: float
    temp_decay_scale: float
    destroy_bias: tuple[float, ...]  # length == N_D = 8
    repair_bias: tuple[float, ...]  # length == N_R = 5
    ls_passes: int
    use_recombine: bool


MODES: tuple[ModeSpec, ...] = (
    ModeSpec(
        "default",
        1.00,
        1.00,
        1.000,
        (1.0, 1.0, 1.0, 1.0, 1.0, 0.8, 0.8, 0.8, 1.0, 0.8),
        (1.0, 1.0, 1.0, 1.0, 1.1),
        0,
        False,
    ),
    ModeSpec(
        "intensify",
        0.70,
        0.98,
        0.995,
        (0.5, 1.3, 1.2, 0.5, 1.0, 0.7, 0.8, 0.8, 0.9, 0.6),
        (1.3, 1.2, 0.8, 1.0, 1.3),
        1,
        False,
    ),
    ModeSpec(
        "diversify",
        1.35,
        1.08,
        1.002,
        (1.5, 0.9, 1.3, 1.4, 1.0, 0.7, 1.4, 1.4, 1.6, 1.2),
        (0.9, 1.0, 1.3, 1.0, 0.9),
        0,
        False,
    ),
    ModeSpec(
        "tw_rescue",
        1.10,
        1.05,
        1.000,
        (0.6, 0.9, 1.1, 0.8, 1.8, 0.4, 0.8, 0.8, 1.0, 0.4),
        (0.8, 1.0, 1.2, 1.8, 2.2),
        1,
        False,
    ),
    ModeSpec(
        "pool_recombine",
        0.90,
        1.01,
        0.997,
        (0.7, 1.2, 0.9, 1.1, 0.8, 1.8, 1.6, 1.6, 1.1, 1.8),
        (0.7, 1.1, 1.5, 0.9, 1.1),
        1,
        True,
    ),
    ModeSpec(
        "route_reduce",
        0.95,
        1.02,
        0.998,
        (0.6, 1.0, 0.9, 1.7, 0.6, 2.2, 2.4, 2.4, 1.2, 2.6),
        (0.8, 1.2, 1.5, 1.0, 1.4),
        1,
        True,
    ),
)

MODE_DEFAULT, MODE_INTENSIFY, MODE_DIVERSIFY = 0, 1, 2
MODE_TW_RESCUE, MODE_POOL_RECOMBINE, MODE_ROUTE_REDUCE = 3, 4, 5
```

## File: `src/vrptw/core.py`

```python
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
            nearest = list(np.argsort(dists)[:k_neighbors])
            self.neighbors_k.append(nearest)


@njit(cache=True)
def _route_cost(route: np.ndarray, dist: np.ndarray) -> float:
    cost = dist[0, route[0]]
    for idx in range(len(route) - 1):
        cost += dist[route[idx], route[idx + 1]]
    return cost + dist[route[-1], 0]


@njit(cache=True)
def _route_ok(route, demands, capacity, ready, due, service, dist) -> bool:
    load = 0.0
    t, prev = 0.0, 0
    for node in route:
        load += demands[node]
        t += dist[prev, node]
        if t < ready[node]:
            t = ready[node]
        if t > due[node]:
            return False
        t += service[node]
        prev = node
    if load > capacity:
        return False
    return t + dist[prev, 0] <= due[0]


class Plan:
    __slots__ = ("routes", "inst", "_cost", "_ok", "algo")

    def __init__(self, routes: list[list[int]], inst: Inst, algo: str = ""):
        self.routes = [r for r in routes if r]
        self.inst = inst
        self._cost: float | None = None
        self._ok: bool | None = None
        self.algo = algo

    @property
    def cost(self) -> float:
        if self._cost is None:
            self._cost = sum(_route_cost(np.array(r, np.int64), self.inst.dist) for r in self.routes)
        return self._cost

    @property
    def feasible(self) -> bool:
        if self._ok is None:
            self._ok = all(
                _route_ok(
                    np.array(r, np.int64),
                    self.inst.demands,
                    self.inst.capacity,
                    self.inst.ready_times,
                    self.inst.due_times,
                    self.inst.service_times,
                    self.inst.dist,
                )
                for r in self.routes
            )
        return self._ok

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
        return Plan([r[:] for r in self.routes], self.inst, self.algo)

    def invalidate(self) -> None:
        self._cost = None
        self._ok = None


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


def _check_route(route: list[int], inst: Inst) -> bool:
    """Check capacity + time-window feasibility (delegates to numba ``_route_ok``)."""
    return bool(
        _route_ok(
            np.array(route, np.int64),
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
    fills = [plan.inst.demands[np.array(r, np.int64)].sum() / capacity for r in plan.routes]
    return float(np.mean(fills))
```

## File: `src/vrptw/heuristics.py`

```python
from __future__ import annotations

import numpy as np

from .core import Inst, Plan, _check_route, _route_cost


def _best_insert_position(node: int, route: list[int], inst: Inst) -> tuple[float, int | None]:
    best_cost, best_pos = float("inf"), None
    for pos in range(len(route) + 1):
        prev = route[pos - 1] if pos > 0 else 0
        nxt = route[pos] if pos < len(route) else 0
        delta = inst.dist[prev, node] + inst.dist[node, nxt] - inst.dist[prev, nxt]
        if delta < best_cost and _check_route(route[:pos] + [node] + route[pos:], inst):
            best_cost, best_pos = delta, pos
    return best_cost, best_pos


def _insert_customer(plan: Plan, node: int, inst: Inst) -> None:
    best_cost, best_route, best_pos = float("inf"), None, None
    for ri, route in enumerate(plan.routes):
        delta, pos = _best_insert_position(node, route, inst)
        if pos is not None and delta < best_cost:
            best_cost, best_route, best_pos = delta, ri, pos
    if best_route is not None:
        plan.routes[best_route].insert(best_pos, node)
    else:
        plan.routes.append([node])
    plan.invalidate()


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


def build_greedy(inst: Inst, algo: str = "") -> Plan:
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
            delta = inst.dist[prev, node] + inst.dist[node, nxt] - inst.dist[prev, nxt]
            if delta < best_cost:
                best_cost, best_pos = delta, pos
        return best_cost, best_pos

    unrouted = list(range(1, inst.n + 1))
    routes: list[list[int]] = []
    while unrouted:
        seed = max(unrouted, key=lambda n: inst.dist[0, n])
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
                c2 = inst.dist[0, node] + inst.dist[node, 0] - c1
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
            nxt = min(feasible, key=lambda c: inst.dist[node, c])
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
```

## File: `src/vrptw/operators.py`

```python
from __future__ import annotations

import math
import random

import numpy as np

from .config import MODES, Config
from .core import Inst, Plan, _invalidate, _route_duration_no_return
from .heuristics import _best_insert_position, _insert_customer


def op_random(plan: Plan, size: int) -> tuple[Plan, list[int]]:
    nodes = [n for r in plan.routes for n in r]
    removed = random.sample(nodes, min(size, len(nodes)))
    rs = set(removed)
    plan.routes = [[n for n in r if n not in rs] for r in plan.routes]
    plan.routes = [r for r in plan.routes if r]
    return _invalidate(plan), removed


def op_worst(plan: Plan, size: int) -> tuple[Plan, list[int]]:
    inst = plan.inst
    gains: list[tuple[float, int]] = []
    for route in plan.routes:
        for idx, node in enumerate(route):
            prev = route[idx - 1] if idx > 0 else 0
            nxt = route[idx + 1] if idx < len(route) - 1 else 0
            gains.append((inst.dist[prev, node] + inst.dist[node, nxt] - inst.dist[prev, nxt], node))
    gains.sort(reverse=True)
    # ALNS power-law randomized selection to introduce search diversity
    p = 3.0
    removed: list[int] = []
    rs = set()
    while len(removed) < size and gains:
        idx = int((random.random() ** p) * len(gains))
        _, node = gains.pop(idx)
        removed.append(node)
        rs.add(node)
    plan.routes = [[n for n in r if n not in rs] for r in plan.routes]
    plan.routes = [r for r in plan.routes if r]
    return _invalidate(plan), removed


def op_shaw(plan: Plan, size: int) -> tuple[Plan, list[int]]:
    inst = plan.inst
    nodes = [n for r in plan.routes for n in r]
    if not nodes:
        return plan, []
    seed_node = random.choice(nodes)
    removed = [seed_node]
    rs = {seed_node}
    max_dist = inst.max_dist + 1e-9
    max_tw = max(inst.due_times - inst.ready_times) + 1e-9
    while len(removed) < size:
        # Dynamically select a reference node from the already removed set
        ref_node = random.choice(removed)
        neighbors = inst.neighbors_k[ref_node]
        candidates = [
            (
                n,
                0.5 * inst.dist[ref_node, n] / max_dist
                + 0.4 * abs(inst.ready_times[ref_node] - inst.ready_times[n]) / max_tw
                + 0.1 * abs(inst.demands[ref_node] - inst.demands[n]) / inst.capacity,
            )
            for n in neighbors
            if n not in rs
        ]
        if not candidates:
            # Fallback to all remaining nodes if neighbors are exhausted
            candidates = [
                (
                    n,
                    0.5 * inst.dist[ref_node, n] / max_dist
                    + 0.4 * abs(inst.ready_times[ref_node] - inst.ready_times[n]) / max_tw
                    + 0.1 * abs(inst.demands[ref_node] - inst.demands[n]) / inst.capacity,
                )
                for n in nodes
                if n not in rs
            ]
        if not candidates:
            break
        candidates.sort(key=lambda x: x[1])
        # Use power law selection (p = 3.0) to select similar nodes with high probability
        p = 3.0
        idx = int((random.random() ** p) * len(candidates))
        chosen = candidates[idx][0]
        removed.append(chosen)
        rs.add(chosen)
    plan.routes = [[n for n in r if n not in rs] for r in plan.routes]
    plan.routes = [r for r in plan.routes if r]
    return _invalidate(plan), removed


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
        # Use randomized selection for the pivot to prevent determinism
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
    # Add random noise to the time window width to vary the selection of tight windows
    candidates = sorted(nodes, key=lambda n: (inst.due_times[n] - inst.ready_times[n]) * (1.0 + random.random() * 0.3))
    removed = candidates[:size]
    rs = set(removed)
    plan.routes = [[n for n in r if n not in rs] for r in plan.routes]
    plan.routes = [r for r in plan.routes if r]
    return _invalidate(plan), removed


def op_route_eliminate(plan: Plan, size: int) -> tuple[Plan, list[int]]:
    if len(plan.routes) <= 1:
        return op_random(plan, size)
    inst = plan.inst
    # Add small random noise to route length and load fraction to vary route choices
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
        # Add random noise to increase variety
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
    max_dist = inst.max_dist + 1e-9
    max_tw = max(inst.due_times - inst.ready_times) + 1e-9
    while len(removed) < size:
        ref_node = random.choice(removed)  # â† once per outer iteration
        ref_route = node_to_route.get(ref_node, -1)
        candidates = []
        for n in nodes:
            if n in rs:
                continue
            cross_route_bonus = -0.2 if node_to_route.get(n, -2) != ref_route else 0.2
            rel = (
                0.5 * inst.dist[ref_node, n] / max_dist
                + 0.4 * abs(inst.ready_times[ref_node] - inst.ready_times[n]) / max_tw
                + 0.1 * abs(inst.demands[ref_node] - inst.demands[n]) / inst.capacity
                + cross_route_bonus
            )
            candidates.append((n, rel))
        if not candidates:
            break
        candidates.sort(key=lambda x: x[1])
        p = 3.0
        idx = int((random.random() ** p) * len(candidates))
        nxt = candidates[idx][0]
        removed.append(nxt)
        rs.add(nxt)
    plan.routes = [[n for n in r if n not in rs] for r in plan.routes]
    plan.routes = [r for r in plan.routes if r]
    return _invalidate(plan), removed


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
    """
    Route Merge Sampling (10th destroy operator).

    Selects the capacity-compatible route pair with the smallest combined customer
    count and removes the smaller route's customers. The repair operator must then
    insert them back â€” often into the surviving route, producing a longer merged
    route that the pool otherwise under-represents.

    Key difference from op_route_eliminate: this operator explicitly verifies
    combined capacity compatibility before selection, maximising the probability
    that repair produces a single consolidated route rather than creating a new one.
    """
    if len(plan.routes) <= 1:
        return op_random(plan, size)
    inst = plan.inst

    # Find capacity-compatible pairs sorted by combined customer count
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
            # Prefer denser pairs (high fill = fewer wasted seats after merge)
            score = combined_len - load_fill * 4.0 + random.random() * 1.5
            candidates.append((score, i, j))

    if not candidates:
        return op_route_eliminate(plan, size)

    # Power-law selection: bias toward lower-score (denser / smaller) pairs
    candidates.sort()
    p = 3.0
    idx = int((random.random() ** p) * len(candidates))
    _, i, j = candidates[idx]

    small_idx = i if len(plan.routes[i]) <= len(plan.routes[j]) else j
    removed = list(plan.routes[small_idx])
    plan.routes = [r for k, r in enumerate(plan.routes) if k != small_idx]
    return _invalidate(plan), removed


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
    op_route_merge_sample,  # idx 9 â€” new
]


def op_greedy(plan: Plan, removed: list[int]) -> Plan:
    inst = plan.inst
    for node in sorted(removed, key=lambda n: inst.due_times[n]):
        _insert_customer(plan, node, inst)
    return Plan(plan.routes, inst, plan.algo)


def _regret(plan: Plan, removed: list[int], k: int) -> Plan:
    inst = plan.inst
    remaining: set = set(removed)
    while remaining:
        best_regret, chosen, choice = -float("inf"), None, None
        for node in remaining:
            options = sorted(
                (delta, ri, pos)
                for ri, route in enumerate(plan.routes)
                for delta, pos in [_best_insert_position(node, route, inst)]
                if pos is not None
            )
            if not options:
                continue
            # sum-of-gaps variant of regret-k: penalises nodes with multiple
            # poor alternatives more aggressively than classical Î”_{k-1} - Î”_0
            regret = (
                sum(options[i][0] - options[0][0] for i in range(1, k))
                if len(options) >= k
                else (options[1][0] - options[0][0] if len(options) >= 2 else float("inf"))
            )
            if regret > best_regret:
                best_regret, chosen, choice = regret, node, options[0]
        if chosen is not None and choice is not None:
            _, ri, pos = choice
            if pos is not None:
                plan.routes[ri].insert(pos, chosen)
            plan.invalidate()
            remaining.discard(chosen)  # â† O(1)
        else:
            for node in remaining:
                plan.routes.append([node])
            break
    return Plan(plan.routes, inst, plan.algo)


def op_regret_2(plan: Plan, removed: list[int]) -> Plan:
    return _regret(plan, removed, 2)


def op_regret_3(plan: Plan, removed: list[int]) -> Plan:
    return _regret(plan, removed, 3)


def op_tw_greedy(plan: Plan, removed: list[int]) -> Plan:
    inst = plan.inst
    for node in sorted(removed, key=lambda n: inst.due_times[n] - inst.ready_times[n]):
        _insert_customer(plan, node, inst)
    return Plan(plan.routes, inst, plan.algo)


def _route_arrivals_wait(route: list[int], inst: Inst) -> tuple[list[float], float]:
    arrivals: list[float] = []
    total_wait = 0.0
    t, prev = 0.0, 0
    for node in route:
        raw = t + inst.dist[prev, node]
        wait = max(0.0, inst.ready_times[node] - raw)
        t = raw + wait
        arrivals.append(float(t))
        total_wait += wait
        t += inst.service_times[node]
        prev = node
    return arrivals, float(total_wait)


def _route_forward_time_slacks(route: list[int], inst: Inst) -> list[float]:
    if not route:
        return []
    arrivals, _ = _route_arrivals_wait(route, inst)
    latest = [0.0] * len(route)
    latest[-1] = float(inst.due_times[route[-1]])
    for idx in range(len(route) - 2, -1, -1):
        node = route[idx]
        nxt = route[idx + 1]
        latest[idx] = min(
            float(inst.due_times[node]),
            latest[idx + 1] - float(inst.service_times[node]) - float(inst.dist[node, nxt]),
        )
    return [max(0.0, latest[idx] - arrivals[idx]) for idx in range(len(route))]


def _fts_best_insert_position(node: int, route: list[int], inst: Inst) -> tuple[float, int | None]:
    best_cost, best_pos = float("inf"), None
    horizon = max(inst.horizon, 1.0)
    max_dist = max(inst.max_dist, 1.0)
    wait_weight = 0.10 + 0.35 * inst.tw_tight_frac
    long_route_pressure = min((len(route) + 1) / 30.0, 1.0)
    fts_weight = 0.15 + 0.45 * inst.tw_tight_frac + 0.25 * long_route_pressure

    # precompute once â€” O(route_len)
    base_arrivals, base_wait = _route_arrivals_wait(route, inst)

    for pos in range(len(route) + 1):
        prev = route[pos - 1] if pos > 0 else 0
        nxt = route[pos] if pos < len(route) else 0

        # incremental feasibility: check node insertion only
        t_prev = base_arrivals[pos - 1] if pos > 0 else 0.0
        t_arrive = t_prev + inst.dist[prev, node]
        if t_arrive > inst.due_times[node]:
            continue
        t_depart = max(t_arrive, inst.ready_times[node]) + inst.service_times[node]
        # check downstream propagation: time push at nxt
        if nxt != 0:
            t_nxt_new = t_depart + inst.dist[node, nxt]
            t_nxt_old = base_arrivals[pos] if pos < len(base_arrivals) else None
            if t_nxt_old is not None and t_nxt_new > inst.due_times[nxt]:
                continue

        dist_added = inst.dist[prev, node] + inst.dist[node, nxt] - inst.dist[prev, nxt]

        # approximate wait_added from time push at insertion point
        wait_node = max(0.0, inst.ready_times[node] - (t_prev + inst.dist[prev, node]))
        wait_added = wait_node  # dominant contribution; downstream wait unchanged to first order

        # downstream FTS: min slack from pos onward in base route
        if pos < len(base_arrivals):
            downstream_fts = min(max(0.0, inst.due_times[route[i]] - base_arrivals[i]) for i in range(pos, len(route)))
        else:
            downstream_fts = horizon
        fts_norm = min(downstream_fts / horizon, 1.0)

        composite = float(dist_added + wait_weight * wait_added - fts_weight * fts_norm * max_dist)
        if composite < best_cost:
            best_cost, best_pos = composite, pos

    return best_cost, best_pos


def op_fts_greedy(plan: Plan, removed: list[int]) -> Plan:
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

assert N_D == 10
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


def destroy_size(it: int, n_iters: int, cfg: Config, n_customers: int, scale: float = 1.0) -> int:
    ratio = cfg.destroy_ratio_max - ((cfg.destroy_ratio_max - cfg.destroy_ratio_min) * (it / max(n_iters, 1)))
    ratio = min(cfg.destroy_ratio_max, max(cfg.destroy_ratio_min, ratio * scale))
    return max(3, int(ratio * n_customers))
```

## File: `src/vrptw/local_search.py`

```python
from __future__ import annotations

import numpy as np

from .core import Inst, Plan, _check_route
from .heuristics import _best_insert_position, _route_cost_list, _route_load


def _two_opt_best(route: list[int], inst: Inst) -> list[int]:
    if len(route) < 4:
        return route[:]
    best, best_cost = route[:], _route_cost_list(route, inst)
    for i in range(len(route) - 2):
        for j in range(i + 2, len(route)):
            cand = route[:i] + list(reversed(route[i : j + 1])) + route[j + 1 :]
            if not _check_route(cand, inst):
                continue
            cc = _route_cost_list(cand, inst)
            if cc + 1e-9 < best_cost:
                best, best_cost = cand, cc
    return best


def _best_relocate(plan: Plan, nv_ceiling: int | None = None):
    inst = plan.inst
    best_delta, best_move = -1e-9, None
    for si, source_route in enumerate(plan.routes):
        sc = _route_cost_list(source_route, inst)
        for sp, node in enumerate(source_route):
            sn = source_route[:sp] + source_route[sp + 1 :]
            if sn and not _check_route(sn, inst):
                continue
            snc = _route_cost_list(sn, inst)
            for di, dest_route in enumerate(plan.routes):
                if di == si:
                    continue
                dc = _route_cost_list(dest_route, inst)
                for ip in range(len(dest_route) + 1):
                    dn = dest_route[:ip] + [node] + dest_route[ip:]
                    if not _check_route(dn, inst):
                        continue
                    new_nv = plan.nv - (1 if not sn else 0)
                    if nv_ceiling is not None and new_nv > nv_ceiling:
                        continue
                    delta = snc + _route_cost_list(dn, inst) - sc - dc
                    if new_nv < plan.nv:
                        delta -= 1000.0
                    if delta < best_delta:
                        best_delta, best_move = delta, (si, sp, di, ip)
    return best_move


def _apply_relocate(plan: Plan, move: tuple[int, int, int, int]) -> Plan:
    si, sp, di, ip = move
    routes = [r[:] for r in plan.routes]
    node = routes[si].pop(sp)
    if si < di and len(routes[si]) == 0:
        di -= 1
    routes = [r for r in routes if r]
    routes[di].insert(ip, node)
    return Plan(routes, plan.inst, plan.algo)


def _best_swap(plan: Plan):
    inst = plan.inst
    best_delta, best_move = -1e-9, None
    for si, sr in enumerate(plan.routes):
        sc = _route_cost_list(sr, inst)
        for di in range(si + 1, len(plan.routes)):
            dr = plan.routes[di]
            dc = _route_cost_list(dr, inst)
            for sp, sn in enumerate(sr):
                for dp, dn in enumerate(dr):
                    if sn == dn:
                        continue
                    srn, drn = sr[:], dr[:]
                    srn[sp], drn[dp] = dn, sn
                    if not _check_route(srn, inst) or not _check_route(drn, inst):
                        continue
                    delta = _route_cost_list(srn, inst) + _route_cost_list(drn, inst) - sc - dc
                    if delta < best_delta:
                        best_delta, best_move = delta, (si, sp, di, dp)
    return best_move


def _apply_swap(plan: Plan, move: tuple[int, int, int, int]) -> Plan:
    si, sp, di, dp = move
    routes = [r[:] for r in plan.routes]
    routes[si][sp], routes[di][dp] = routes[di][dp], routes[si][sp]
    return Plan(routes, plan.inst, plan.algo)


def _cross_exchange(plan: Plan, nv_ceiling: int | None = None) -> Plan | None:
    inst = plan.inst
    if nv_ceiling is not None and plan.nv > nv_ceiling:
        return None
    max_dist = max(inst.max_dist, 1.0)
    granular_radius = max(10.0, 0.18 * max_dist)
    best_delta = -1e-9
    best_routes: list[list[int]] | None = None

    def interval_overlap(a0, a1, b0, b1):
        return min(a1, b1) >= max(a0, b0)

    def route_meta(route):
        coords = inst.coords[np.array(route, dtype=np.int64)]
        return (coords.mean(axis=0), float(np.min(inst.ready_times[route])), float(np.max(inst.due_times[route])))

    def seg_meta(seg):
        coords = inst.coords[np.array(seg, dtype=np.int64)]
        return (coords.mean(axis=0), float(np.min(inst.ready_times[seg])), float(np.max(inst.due_times[seg])))

    route_info = [route_meta(r) if r else None for r in plan.routes]

    for i in range(len(plan.routes)):
        for j in range(i + 1, len(plan.routes)):
            r1, r2 = plan.routes[i], plan.routes[j]
            if not r1 or not r2:
                continue
            info1, info2 = route_info[i], route_info[j]
            if info1 is None or info2 is None:
                continue
            c1, r1_ready, r1_due = info1
            c2, r2_ready, r2_due = info2
            if float(np.linalg.norm(c1 - c2)) > 0.55 * max_dist and not interval_overlap(
                r1_ready, r1_due, r2_ready, r2_due
            ):
                continue
            old_pair = _route_cost_list(r1, inst) + _route_cost_list(r2, inst)
            lc1 = (1, 2, 3) if len(r1) >= 12 else (1, 2)
            lc2 = (1, 2, 3) if len(r2) >= 12 else (1, 2)
            for len1 in lc1:
                if len(r1) < len1:
                    continue
                for len2 in lc2:
                    if len(r2) < len2:
                        continue
                    for p1 in range(len(r1) - len1 + 1):
                        seg1 = r1[p1 : p1 + len1]
                        s1c, s1r, s1d = seg_meta(seg1)
                        for p2 in range(len(r2) - len2 + 1):
                            seg2 = r2[p2 : p2 + len2]
                            s2c, s2r, s2d = seg_meta(seg2)
                            if float(np.linalg.norm(s1c - s2c)) > granular_radius and not interval_overlap(
                                s1r, s1d, s2r, s2d
                            ):
                                continue
                            nr1 = r1[:p1] + seg2 + r1[p1 + len1 :]
                            nr2 = r2[:p2] + seg1 + r2[p2 + len2 :]
                            if not _check_route(nr1, inst) or not _check_route(nr2, inst):
                                continue
                            delta = _route_cost_list(nr1, inst) + _route_cost_list(nr2, inst) - old_pair
                            if delta < best_delta:
                                routes = [r[:] for r in plan.routes]
                                routes[i], routes[j] = nr1, nr2
                                best_routes = routes
                                best_delta = delta
    return Plan(best_routes, inst, plan.algo) if best_routes is not None else None


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


def _best_or_opt(plan: Plan, nv_ceiling: int | None = None):
    inst = plan.inst
    best_delta, best_move = -1e-9, None
    for si, source_route in enumerate(plan.routes):
        sc = _route_cost_list(source_route, inst)
        for L in (2, 3):
            for sp in range(len(source_route) - L + 1):
                seg = source_route[sp : sp + L]
                sn = source_route[:sp] + source_route[sp + L :]
                if sn and not _check_route(sn, inst):
                    continue
                snc = _route_cost_list(sn, inst) if sn else 0.0
                for di, dest_route in enumerate(plan.routes):
                    if di == si:
                        continue
                    dc = _route_cost_list(dest_route, inst)
                    for ip in range(len(dest_route) + 1):
                        # Try original segment order
                        dn = dest_route[:ip] + seg + dest_route[ip:]
                        if _check_route(dn, inst):
                            new_nv = plan.nv - (1 if not sn else 0)
                            if nv_ceiling is None or new_nv <= nv_ceiling:
                                delta = snc + _route_cost_list(dn, inst) - sc - dc
                                if new_nv < plan.nv:
                                    delta -= 1000.0
                                if delta < best_delta:
                                    best_delta, best_move = delta, (si, sp, L, di, ip, False)

                        # Try reversed segment order
                        dn_rev = dest_route[:ip] + list(reversed(seg)) + dest_route[ip:]
                        if _check_route(dn_rev, inst):
                            new_nv = plan.nv - (1 if not sn else 0)
                            if nv_ceiling is None or new_nv <= nv_ceiling:
                                delta = snc + _route_cost_list(dn_rev, inst) - sc - dc
                                if new_nv < plan.nv:
                                    delta -= 1000.0
                                if delta < best_delta:
                                    best_delta, best_move = delta, (si, sp, L, di, ip, True)
    return best_move


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


def _try_route_merge(plan: Plan) -> Plan | None:
    """
    Route Consolidation Move.

    Attempts to merge any two capacity-compatible routes into a single feasible
    route via EDF-ordered best-insertion. Unlike _try_route_compact (which
    scatters customers across many routes), this produces ONE longer route â€”
    exactly the route type MILP needs to assemble an NV-1 partition.

    Checks at most the 4 shortest routes for O(k * n^2) tractability.
    Returns the best merged plan (nv-1), or None if no merge is feasible.
    """
    inst = plan.inst
    if len(plan.routes) <= 1:
        return None

    ranked = sorted(range(len(plan.routes)), key=lambda i: len(plan.routes[i]))
    check_idxs = ranked[: min(4, len(ranked))]
    best_merged: Plan | None = None

    for pos_i, i in enumerate(check_idxs):
        r1 = plan.routes[i]
        load1 = sum(inst.demands[n] for n in r1)
        for j in check_idxs[pos_i + 1 :]:
            r2 = plan.routes[j]
            if load1 + sum(inst.demands[n] for n in r2) > inst.capacity:
                continue  # capacity gate

            # EDF ordering minimises downstream TW violations
            combined = sorted(list(r1) + list(r2), key=lambda n: inst.due_times[n])
            merged_route: list[int] = []
            ok = True
            for node in combined:
                _, pos = _best_insert_position(node, merged_route, inst)
                if pos is None:
                    ok = False
                    break
                merged_route.insert(pos, node)
            if not ok or not _check_route(merged_route, inst):
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
    """
    Eliminate one route with a bounded ejection buffer.

    Direct route elimination fails on random/mixed instances when the target
    customers need a few recipient-route customers to move first. This beam
    search keeps those displaced customers in a pending buffer and reinserts
    them without ever creating a new route.
    """
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


def _buffered_route_elimination(
    plan: Plan,
    max_rounds: int = 2,
    max_ejections: int = 4,
    beam_width: int = 8,
    pool=None,
) -> Plan:
    best = plan.copy()
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
        improved = False
        for target_idx in ranked[:6]:
            local_ejections = max(2, min(max_ejections, len(best.routes[target_idx]) // 2 + 1))
            cand = _try_buffered_route_elimination(
                best,
                target_idx,
                max_ejections=local_ejections,
                beam_width=beam_width,
            )
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


def local_search(
    plan: Plan,
    max_passes: int = 1,
    nv_ceiling: int | None = None,
    max_ls_moves: int = 5,
    pool=None,  # RoutePool | None â€” seeds pool with every improvement
) -> Plan:
    """
    pool: when provided, each improvement along the LS trajectory is added to
          the route pool, generating diverse intermediate routes for MILP.
          Also triggers _try_route_merge at the end to seed merged-route types.
    """
    best = plan.copy()
    for _ in range(max_passes):
        improved = False
        routes = []
        for route in best.routes:
            nr = _two_opt_best(route, best.inst)
            routes.append(nr)
            if nr != route:
                improved = True
        best = Plan(routes, best.inst, best.algo)

        moves = 0
        while moves < max_ls_moves:
            move = _best_relocate(best, nv_ceiling=nv_ceiling)
            if move is not None:
                cand = _apply_relocate(best, move)
                if cand.feasible and (cand.dominates(best) or (cand.nv == best.nv and cand.cost + 1e-9 < best.cost)):
                    best, improved = cand, True
                    if pool is not None:
                        pool.add_plan(best)
                    moves += 1
                    continue
            move = _best_swap(best)
            if move is not None:
                cand = _apply_swap(best, move)
                if cand.feasible and cand.cost + 1e-9 < best.cost:
                    best, improved = cand, True
                    if pool is not None:
                        pool.add_plan(best)
                    moves += 1
                    continue
            move = _best_or_opt(best, nv_ceiling=nv_ceiling)
            if move is not None:
                cand = _apply_or_opt(best, move)
                if cand.feasible and (cand.dominates(best) or (cand.nv == best.nv and cand.cost + 1e-9 < best.cost)):
                    best, improved = cand, True
                    if pool is not None:
                        pool.add_plan(best)
                    moves += 1
                    continue
            cross = _cross_exchange(best, nv_ceiling=nv_ceiling)
            if cross is not None:
                best, improved = cross, True
                if pool is not None:
                    pool.add_plan(best)
                moves += 1
                continue
            compact = _try_route_compact(best, nv_ceiling=nv_ceiling)
            if compact is not None:
                best, improved = compact, True
                if pool is not None:
                    pool.add_plan(best)
                moves += 1
                continue
            break

        if not improved:
            break

    # Pool-seeding bonus: generate merged-route type not produced by normal LS
    if pool is not None:
        merged = _try_route_merge(best)
        if merged is not None:
            pool.add_plan(merged)

    return best


def _try_chain_elimination(plan: Plan, target_idx: int) -> Plan | None:
    """
    Single-target ejection chain.  Processes each customer in the target route;
    for customers that fail direct insertion, attempts a depth-2 chain:
      c â†’ Ri  (displacing d from Ri)  â†’  d â†’ Rj
    Returns a feasible Plan with nv-1, or None if any customer is unplaceable.
    """
    inst = plan.inst
    target = plan.routes[target_idx]
    routes = [r[:] for i, r in enumerate(plan.routes) if i != target_idx]

    # Tightest TW first â€” hardest-to-place customers block everything downstream
    for c in sorted(target, key=lambda n: inst.due_times[n] - inst.ready_times[n]):
        # --- Level 1: direct best-insertion ---
        best_delta, best_ri, best_pos = float("inf"), None, None
        for ri, route in enumerate(routes):
            delta, pos = _best_insert_position(c, route, inst)
            if pos is not None and delta < best_delta:
                best_delta, best_ri, best_pos = delta, ri, pos

        if best_ri is not None:
            routes[best_ri].insert(best_pos, c)
            continue

        # --- Level 2: ejection chain  (c ejects d from Ri, d moves to Rj) ---
        best_chain_cost = float("inf")
        best_chain: tuple | None = None  # (ri, eject_pos, d, rj)

        for ri, route in enumerate(routes):
            for eject_pos, d in enumerate(route):
                # Temporary route with d removed
                stripped = route[:eject_pos] + route[eject_pos + 1 :]
                delta_c, pos_c = _best_insert_position(c, stripped, inst)
                if pos_c is None:
                    continue
                # Check if d can land anywhere else
                for rj, other in enumerate(routes):
                    if rj == ri:
                        continue
                    delta_d, pos_d = _best_insert_position(d, other, inst)
                    if pos_d is not None:
                        total = delta_c + delta_d
                        if total < best_chain_cost:
                            best_chain_cost = total
                            best_chain = (ri, eject_pos, d, rj)

        if best_chain is None:
            return None  # c unplaceable even with chain

        ri, eject_pos, d, rj = best_chain
        # Apply: remove d, insert c into Ri, insert d into Rj
        stripped_ri = routes[ri][:eject_pos] + routes[ri][eject_pos + 1 :]
        _, pos_c = _best_insert_position(c, stripped_ri, inst)
        if pos_c is None:
            return None
        routes[ri] = stripped_ri[:pos_c] + [c] + stripped_ri[pos_c:]

        _, pos_d = _best_insert_position(d, routes[rj], inst)
        if pos_d is None:
            return None
        routes[rj].insert(pos_d, d)

    cand = Plan([r for r in routes if r], inst, plan.algo)
    return cand if cand.feasible else None


def _ejection_chain_eliminate(plan: Plan) -> Plan | None:
    """
    Tries to eliminate up to the 4 smallest routes via depth-2 ejection chains.
    Complements _iterative_route_elimination: handles cases where greedy
    insertion fails due to TW blocking but a 2-step chain resolves the conflict.

    Complexity: O(k Ã— n_target Ã— n_routes Ã— route_len Ã— n_routes) per call â€”
    fast enough for post-search use on 100-customer instances.
    Returns first feasible NV-1 plan found, or None.
    """
    if len(plan.routes) <= 1:
        return None

    ranked = sorted(
        range(len(plan.routes)),
        key=lambda i: (len(plan.routes[i]), sum(plan.inst.demands[n] for n in plan.routes[i])),
    )
    for target_idx in ranked[:4]:
        result = _try_chain_elimination(plan, target_idx)
        if result is not None:
            return result
    return None


def _iterative_route_elimination(
    plan: Plan,
    inst: Inst,
    max_rounds: int = 6,
    pool=None,  # RoutePool | None â€” seeds pool after each success
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
```

## File: `src/vrptw/pool.py`

```python
from __future__ import annotations

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


class RoutePool:
    def __init__(self, inst: Inst, cfg: Config):
        self.inst = inst
        self.cfg = cfg
        self._routes: dict[tuple[int, ...], RouteRecord] = {}

    def _priority(self, rec: RouteRecord) -> tuple[float, ...]:
        lr = rec.load / max(self.inst.capacity, 1)
        cps = rec.cost / max(len(rec.nodes), 1)
        return (-len(rec.nodes), cps, -lr, -rec.slack)

    def _trim(self) -> None:
        limit = self.cfg.route_pool_limit
        if len(self._routes) <= limit:
            return
        usage: dict[int, int] = {}
        kept: dict[tuple[int, ...], RouteRecord] = {}
        ranked = sorted(self._routes.values(), key=self._priority)
        for rec in ranked:
            if len(kept) >= limit:
                break
            under = all(usage.get(n, 0) < self.cfg.route_pool_max_per_customer for n in rec.nodes)
            if under or len(kept) < limit // 3:
                kept[rec.nodes] = rec
                for n in rec.nodes:
                    usage[n] = usage.get(n, 0) + 1
        if len(kept) < limit:
            for rec in ranked:
                if rec.nodes not in kept:
                    kept[rec.nodes] = rec
                if len(kept) >= limit:
                    break
        self._routes = kept

    def add_route(self, route: list[int]) -> None:
        if not route or not _check_route(route, self.inst):
            return
        key = tuple(route)
        if key in self._routes:
            return
        self._routes[key] = RouteRecord(
            nodes=key,
            cost=_route_cost_list(route, self.inst),
            load=_route_load(route, self.inst),
            slack=_route_avg_slack(route, self.inst),
        )
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
        return sorted(recs.values(), key=self._priority)


def _sp_vehicle_penalty(inst: Inst, cfg: Config) -> float:
    return cfg.sp_vehicle_penalty_scale * max(inst.max_dist, 1.0) * max(inst.n, 1)


def _milp_recombine(
    route_records: list[RouteRecord],
    inst: Inst,
    cfg: Config,
    nv_ceiling: int | None = None,
    vehicle_penalty: float | None = None,
) -> Plan | None:
    if not MILP_OK or not route_records:
        return None
    n_routes = len(route_records)
    cover = np.zeros((inst.n, n_routes), dtype=float)
    for ridx, rec in enumerate(route_records):
        for node in rec.nodes:
            cover[node - 1, ridx] = 1.0
    if np.any(cover.sum(axis=1) == 0):
        return None
    constraints = [LinearConstraint(cover, lb=np.ones(inst.n), ub=np.ones(inst.n))]
    if nv_ceiling is not None:
        constraints.append(
            LinearConstraint(np.ones((1, n_routes)), lb=np.array([0.0]), ub=np.array([float(nv_ceiling)]))
        )
    penalty = vehicle_penalty if vehicle_penalty is not None else _sp_vehicle_penalty(inst, cfg)
    costs = np.array([penalty + rec.cost for rec in route_records])
    result = milp(
        c=costs,
        constraints=constraints,
        integrality=np.ones(n_routes, dtype=int),
        bounds=Bounds(np.zeros(n_routes), np.ones(n_routes)),
        options={"time_limit": float(cfg.sp_time_limit), "disp": False},
    )
    if result is None or not getattr(result, "success", False) or result.x is None:
        return None
    chosen = [list(route_records[i].nodes) for i, v in enumerate(result.x) if v >= 0.5]
    plan = Plan(chosen, inst, "SP-RECOMBINE")
    return plan if plan.feasible else None


def _greedy_recombine(route_records: list[RouteRecord], incumbent: Plan, nv_ceiling: int | None = None) -> Plan:
    uncovered = set(range(1, incumbent.inst.n + 1))
    selected: list[list[int]] = []
    used: set = set()
    while uncovered:
        best_rec, best_score = None, -float("inf")
        for rec in route_records:
            if rec.nodes in used:
                continue
            gain = len(set(rec.nodes) & uncovered)
            if gain == 0:
                continue
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
    return plan if plan.feasible else incumbent.copy()


def recombine_with_route_pool(
    incumbent: Plan,
    pool: RoutePool,
    cfg: Config,
    nv_ceiling: int | None = None,
    nv_target: int | None = None,
) -> Plan:
    pool.add_plan(incumbent)
    recs = pool.records(incumbent)
    if not recs:
        return incumbent.copy()

    mean_cost = float(np.mean([r.cost for r in recs])) if recs else 100.0
    use_penalty = (nv_ceiling is not None) or (nv_target is not None)
    effective_ceiling = nv_target if nv_target is not None else nv_ceiling

    if not use_penalty:
        # Standard recombination: no NV pressure
        candidate = _milp_recombine(recs, incumbent.inst, cfg, nv_ceiling=effective_ceiling, vehicle_penalty=0.0)
        if candidate is None:
            candidate = _greedy_recombine(recs, incumbent, nv_ceiling=effective_ceiling)
        return candidate if candidate.dominates(incumbent) else incumbent.copy()

    # NV-targeted: try multiple penalty scales so one scale finds the partition
    # if it exists in the pool, even when mean_cost * 2.0 is insufficient.
    # Time per query: sp_time_limit / n_scales; total budget = sp_time_limit.
    penalty_scales = (2.0, 5.0, 12.0)
    per_query_limit = max(1.0, cfg.sp_time_limit / len(penalty_scales))

    # Temporarily override time limit per query
    class _TmpCfg:
        def __getattr__(self, name):
            if name == "sp_time_limit":
                return per_query_limit
            return getattr(cfg, name)

    tmp_cfg = _TmpCfg()

    for scale in penalty_scales:
        penalty = max(cfg.sp_vehicle_penalty_scale, mean_cost * scale)
        candidate = _milp_recombine(
            recs,
            incumbent.inst,
            tmp_cfg,
            nv_ceiling=effective_ceiling,
            vehicle_penalty=penalty,
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
            if candidate.feasible and (effective_ceiling is None or candidate.nv <= effective_ceiling):
                return candidate

    # All scales failed: greedy fallback
    candidate = _greedy_recombine(recs, incumbent, nv_ceiling=effective_ceiling)
    if effective_ceiling is not None and candidate.nv > effective_ceiling:
        return incumbent.copy()
    return candidate if candidate.dominates(incumbent) else incumbent.copy()
```

## File: `src/vrptw/rl.py`

```python
from __future__ import annotations

import math
import os
import random
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from .config import MODES, Config
from .core import Inst, Plan
from .operators import N_ACTIONS, N_R

DEVICE = torch.device("cpu")
torch.set_num_threads(max(1, int(os.environ.get("NUMBA_NUM_THREADS", "1")) // 2))


class PrioritizedReplayBuffer:
    """
    Proportional PER (Schaul et al., 2016).
    alpha=0.6: prioritization strength.
    beta anneals 0.4â†’1.0 over training to correct IS bias.
    """

    def __init__(
        self,
        capacity: int,
        alpha: float = 0.6,
        beta_start: float = 0.4,
        beta_end: float = 1.0,
        expected_steps: int = 50_000,
    ):
        self.capacity: int = capacity
        self.alpha: float = float(alpha)
        self.beta: float = beta_start
        self.beta_end: float = beta_end
        self.beta_inc: float = (beta_end - beta_start) / max(expected_steps, 1)
        self.buf: list = []
        self.pos: int = 0
        self.priorities: np.ndarray = np.zeros(capacity, dtype=np.float32)
        self.max_pri = 1.0

    def push(self, *transition) -> None:
        if len(self.buf) < self.capacity:
            self.buf.append(transition)
        else:
            self.buf[self.pos] = transition
        self.priorities[self.pos] = self.max_pri
        self.pos = (self.pos + 1) % self.capacity

    def sample(self, batch_size: int):
        n = len(self.buf)
        probs = self.priorities[:n] ** self.alpha
        probs /= probs.sum()
        idxs = np.random.choice(n, batch_size, p=probs, replace=True)
        ws = (n * probs[idxs]) ** -self.beta
        ws /= ws.max()
        self.beta = float(min(self.beta_end, self.beta + self.beta_inc))
        s, a, r, ns, d = zip(*[self.buf[i] for i in idxs])
        return (
            (
                np.array(s, np.float32),
                np.array(a, np.int64),
                np.array(r, np.float32),
                np.array(ns, np.float32),
                np.array(d, np.float32),
            ),
            idxs,
            torch.tensor(ws, dtype=torch.float32).to(DEVICE),
        )

    def update_priorities(self, idxs, td_errors: np.ndarray) -> None:
        for i, err in zip(idxs, td_errors):
            p = float(abs(err)) + 1e-6
            self.priorities[i] = p
            self.max_pri = max(self.max_pri, p)

    def __len__(self) -> int:
        return len(self.buf)


# ---------------------------------------------------------------------------
# QNet
# ---------------------------------------------------------------------------
class QNet(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int):
        super().__init__()
        hid2 = max(hidden_dim // 2, 32)
        self.trunk = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
        )
        self.value_head = nn.Sequential(nn.Linear(hidden_dim, hid2), nn.ReLU(), nn.Linear(hid2, 1))
        self.adv_head = nn.Sequential(nn.Linear(hidden_dim, hid2), nn.ReLU(), nn.Linear(hid2, action_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.trunk(x)
        v = self.value_head(h)
        a = self.adv_head(h)
        return v + a - a.mean(dim=1, keepdim=True)


# ---------------------------------------------------------------------------
# Thompson bandit
# ---------------------------------------------------------------------------
class ThompsonBandit:
    def __init__(self, n_d: int, n_r: int):
        self.alpha = np.ones((n_d, n_r), dtype=np.float64)
        self.beta = np.ones((n_d, n_r), dtype=np.float64)

    def mean(self) -> np.ndarray:
        return self.alpha / (self.alpha + self.beta)

    def select(
        self,
        prior: np.ndarray | None = None,  # â† indented back in
        prior_strength: float = 0.0,
    ) -> tuple[int, int]:
        if prior is not None and prior_strength > 0.0:
            p = np.asarray(prior, dtype=np.float64)
            p /= max(p.sum(), 1e-9)
            alpha = self.alpha + prior_strength * p * self.alpha.sum()
            beta = self.beta + prior_strength * p * self.beta.sum()
            samples = np.random.beta(np.maximum(alpha, 1e-9), np.maximum(beta, 1e-9))
        else:
            samples = np.random.beta(self.alpha, self.beta)
        idx = np.unravel_index(int(samples.argmax()), samples.shape)
        return int(idx[0]), int(idx[1])

    def update(self, di: int, ri: int, score: float, sigma1: int) -> None:
        success = float(np.clip(score / max(sigma1, 1), 0.0, 1.0))
        self.alpha[di, ri] += success
        self.beta[di, ri] += 1.0 - success

    def decay(self, rate: float = 0.95) -> None:
        np.multiply(self.alpha - 1.0, rate, out=self.alpha)
        np.add(self.alpha, 1.0, out=self.alpha)
        np.multiply(self.beta - 1.0, rate, out=self.beta)
        np.add(self.beta, 1.0, out=self.beta)

    def clone(self) -> ThompsonBandit:
        b = ThompsonBandit(self.alpha.shape[0], self.alpha.shape[1])
        b.alpha = self.alpha.copy()
        b.beta = self.beta.copy()
        return b


# ---------------------------------------------------------------------------
# Elite archive
# ---------------------------------------------------------------------------
class EliteArchive:
    def __init__(self, k: int = 5):
        self.k = k
        self._plans: dict[str, list[Plan]] = {}

    def update(self, plan: Plan) -> None:
        if not plan.feasible:
            return
        key = plan.inst.name
        bucket = self._plans.setdefault(key, [])
        bucket.append(plan.copy())
        bucket.sort(key=lambda p: (p.nv, p.cost))
        self._plans[key] = bucket[: self.k]

    def load_plans(self, folder: str, insts_dict: dict[str, Inst]) -> None:
        if not os.path.exists(folder):
            return
        import json

        for fname in os.listdir(folder):
            if not fname.endswith(".json"):
                continue
            path = os.path.join(folder, fname)
            try:
                with open(path) as f:
                    data = json.load(f)
                inst_name = data["instance"]
                if inst_name in insts_dict:
                    plan = Plan(data["routes"], insts_dict[inst_name], data.get("algo", ""))
                    bucket = self._plans.setdefault(inst_name, [])
                    bucket.append(plan)
                    bucket.sort(key=lambda p: (p.nv, p.cost))
                    self._plans[inst_name] = bucket[: self.k]
            except Exception:
                pass

    def update_and_save(self, plan: Plan, folder: str) -> None:
        if not plan.feasible:
            return
        key = plan.inst.name
        bucket = self._plans.setdefault(key, [])
        old_best = bucket[0] if bucket else None
        bucket.append(plan.copy())
        bucket.sort(key=lambda p: (p.nv, p.cost))
        self._plans[key] = bucket[: self.k]

        new_best = self._plans[key][0]
        is_improved = (
            old_best is None
            or new_best.nv < old_best.nv
            or (new_best.nv == old_best.nv and new_best.cost < old_best.cost - 1e-6)
        )
        if is_improved:
            os.makedirs(folder, exist_ok=True)
            path = os.path.join(folder, f"{key}.json")
            import json

            try:
                with open(path, "w") as f:
                    json.dump(
                        {
                            "instance": key,
                            "cost": new_best.cost,
                            "nv": new_best.nv,
                            "routes": new_best.routes,
                            "algo": new_best.algo,
                        },
                        f,
                    )
            except Exception:
                pass

    def best(self, inst_name: str) -> Plan | None:
        bucket = self._plans.get(inst_name, [])
        return bucket[0].copy() if bucket else None

    def summary(self) -> str:
        lines = []
        for name, bucket in sorted(self._plans.items()):
            p = bucket[0]
            td_gap, _ = p.gap()
            gap_str = f"{td_gap:+.2f}%" if td_gap is not None else "--"
            lines.append(f"  {name}: nv={p.nv} cost={p.cost:.1f} gap={gap_str}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# LS Budget, UCB, & Welford Reward Normalizer
# ---------------------------------------------------------------------------
class WelfordRewardNormalizer:
    def __init__(self, clip_sigma: float = 8.0, warmup: int = 128, eps: float = 1e-8):
        self.clip = clip_sigma
        self.warmup = warmup
        self.eps = eps
        self._n = 0
        self._mean = 0.0
        self._M2 = 0.0

    def observe(self, r: float) -> None:
        self._n += 1
        delta = r - self._mean
        self._mean += delta / self._n
        self._M2 += delta * (r - self._mean)

    @property
    def std(self) -> float:
        if self._n < 2:
            return 1.0
        return math.sqrt(max(self._M2 / (self._n - 1), self.eps**2))

    def normalize(self, r: float) -> float | None:
        self.observe(r)
        if self._n < self.warmup:
            return None  # caller must check before pushing to buffer
        z = (r - self._mean) / (self.std + self.eps)
        return float(np.clip(z, -self.clip, self.clip))

    def state_dict(self) -> dict:
        return {"n": self._n, "mean": self._mean, "M2": self._M2}

    def load_state_dict(self, d: dict) -> None:
        self._n = int(d["n"])
        self._mean = float(d["mean"])
        self._M2 = float(d["M2"])


class LSBudgetController:
    def __init__(self, ls_time_frac: float = 0.30, ema_alpha: float = 0.15):
        self.ls_time_frac = ls_time_frac
        self.ema_alpha = ema_alpha
        self._budget_total = float("inf")
        self._budget_used = 0.0
        self._yield_ema = 0.05
        self._n_calls = 0

    def initialize(self, cfg: Config) -> None:
        est_total_s = cfg.hybrid_iterations * 0.025
        self._budget_total = est_total_s * self.ls_time_frac
        self._budget_used = 0.0
        self._yield_ema = 0.05

    def should_trigger(self, action: int, accepted: bool, is_new_best: bool, modes) -> bool:
        if self._budget_used >= self._budget_total:
            return False
        if is_new_best:
            return True
        if modes[action].ls_passes == 0:
            return False
        if not accepted:
            return False
        if self._yield_ema > 0.0:
            return True
        return random.random() < 0.10

    def record(self, time_s: float, cost_before: float, cost_after: float) -> None:
        improvement_pct = max(0.0, (cost_before - cost_after) / max(cost_before, 1.0) * 100.0)
        time_cost = time_s / max(self._budget_total * 0.02, 1e-9)
        self._yield_ema = self.ema_alpha * (improvement_pct - time_cost) + (1.0 - self.ema_alpha) * self._yield_ema
        self._budget_used += time_s
        self._n_calls += 1


class UCBActionAugmenter:
    def __init__(self, n_actions: int = 40, c_ucb: float = 1.0, gamma: float = 0.993, alpha_blend: float = 0.35):
        self.n = n_actions
        self.c = c_ucb
        self.gamma = gamma
        self.alpha = alpha_blend
        self._cnt = np.ones(n_actions, dtype=np.float64) * 0.5
        self._mu = np.zeros(n_actions, dtype=np.float64)
        self._m2 = np.ones(n_actions, dtype=np.float64) * 0.5
        self._N = float(n_actions) * 0.5

    def reset(self) -> None:
        self._cnt[:] = 0.5
        self._mu[:] = 0.0
        self._m2[:] = 0.5
        self._N = float(self.n) * 0.5

    def update(self, action: int, reward: float) -> None:
        # exponential decay on counts only â€” keeps forgetting semantics clean
        self._cnt *= self.gamma
        self._N = self._cnt.sum()
        # update selected arm with decayed Welford
        self._cnt[action] += 1.0
        delta = reward - self._mu[action]
        self._mu[action] += delta / self._cnt[action]
        delta2 = reward - self._mu[action]
        self._m2[action] += delta * delta2
        # decay non-selected arms toward global mean (not toward 0)
        global_mean = float(self._mu[self._cnt > 0.6].mean()) if (self._cnt > 0.6).any() else 0.0
        mask = np.ones(self.n, dtype=bool)
        mask[action] = False
        self._mu[mask] = self._mu[mask] * self.gamma + global_mean * (1.0 - self.gamma)

    def augment_qvalues(self, q: np.ndarray) -> np.ndarray:
        variance = self._m2 / np.maximum(self._cnt - 1.0, 1.0)
        std = np.sqrt(np.maximum(variance, 0.0))
        log_n = math.log(max(self._N, math.e))
        conf = self.c * std * np.sqrt(log_n / np.maximum(self._cnt, 1e-9))
        scores = self._mu + conf
        centered = scores - scores.mean()
        scale = max(scores.std(), 1e-6)
        return q + self.alpha * (centered / scale)


# ---------------------------------------------------------------------------
# DDQN controllers  (now using PrioritizedReplayBuffer)
# ---------------------------------------------------------------------------
class PlateauController:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.q = QNet(cfg.ctrl_state_dim, len(MODES), cfg.ctrl_hidden).to(DEVICE)
        self.q_t = QNet(cfg.ctrl_state_dim, len(MODES), cfg.ctrl_hidden).to(DEVICE)
        self.q_t.load_state_dict(self.q.state_dict())
        self.opt = optim.Adam(self.q.parameters(), lr=cfg.ctrl_lr)
        self.buf = PrioritizedReplayBuffer(cfg.ctrl_buffer, expected_steps=cfg.per_beta_steps)
        self.eps = cfg.ctrl_eps_start
        self.step = 0

    def reset(self) -> None:
        self.eps = self.cfg.ctrl_eps_start

    def act(self, state: np.ndarray) -> int:
        if random.random() < self.eps:
            return random.randrange(len(MODES))
        with torch.no_grad():
            return int(self.q(torch.tensor(state).unsqueeze(0).to(DEVICE))[0].argmax().item())

    def observe(self, s, a, r, ns, done=0.0):
        self.buf.push(s, a, r, ns, done)

    def train_step(self) -> None:
        self.step += 1
        if len(self.buf) < self.cfg.ctrl_batch:
            return
        (s, a, r, ns, d), idxs, is_w = self.buf.sample(self.cfg.ctrl_batch)
        s = torch.tensor(s).to(DEVICE)
        a = torch.tensor(a, dtype=torch.long).to(DEVICE)
        r = torch.tensor(r).to(DEVICE)
        ns = torch.tensor(ns).to(DEVICE)
        d = torch.tensor(d).to(DEVICE)
        qp = self.q(s).gather(1, a.unsqueeze(1)).squeeze(1)
        with torch.no_grad():
            best_a = self.q(ns).argmax(1).unsqueeze(1)
            qn = self.q_t(ns).gather(1, best_a).squeeze(1)
            target = r + self.cfg.ctrl_gamma * qn * (1 - d)
        td_errors = (qp - target).detach().cpu().numpy()
        self.buf.update_priorities(idxs, td_errors)
        loss = (is_w * F.smooth_l1_loss(qp, target, reduction="none")).mean()
        self.opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.q.parameters(), 1.0)
        self.opt.step()
        tau = self.cfg.op_tau
        for target_param, local_param in zip(self.q_t.parameters(), self.q.parameters()):
            target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)
        self.eps = max(self.cfg.ctrl_eps_end, self.eps * self.cfg.ctrl_eps_decay)


class OperatorController:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.q = QNet(cfg.op_state_dim, N_ACTIONS, cfg.op_hidden).to(DEVICE)
        self.q_t = QNet(cfg.op_state_dim, N_ACTIONS, cfg.op_hidden).to(DEVICE)
        self.q_t.load_state_dict(self.q.state_dict())
        self.opt = optim.Adam(self.q.parameters(), lr=cfg.op_lr)
        self.buf = PrioritizedReplayBuffer(cfg.op_buffer, expected_steps=cfg.per_beta_steps)
        self.eps = cfg.op_eps_start
        self.step = 0

    def reset(self) -> None:
        self.eps = self.cfg.op_eps_start

    def _prior(self, dw: np.ndarray, rw: np.ndarray) -> np.ndarray:
        dw = np.asarray(dw, np.float32)
        dw /= max(dw.sum(), 1e-9)
        rw = np.asarray(rw, np.float32)
        rw /= max(rw.sum(), 1e-9)
        return np.outer(dw, rw)

    def _sample_prior(self, prior: np.ndarray, bandit: ThompsonBandit) -> int:
        probs = prior.reshape(-1) * bandit.mean().reshape(-1)
        probs /= max(probs.sum(), 1e-9)
        return int(np.random.choice(N_ACTIONS, p=probs))

    def act(self, state, dw, rw, bandit, frozen=False, ucb_aug=None) -> tuple[int, int, int]:
        prior = self._prior(dw, rw)
        if not frozen and len(self.buf) < self.cfg.op_warmup:
            di, ri = bandit.select(prior=prior, prior_strength=self.cfg.bandit_prior_strength)
            action = di * N_R + ri
        elif not frozen and random.random() < self.eps:
            action = self._sample_prior(prior, bandit)
            di, ri = divmod(action, N_R)
        else:
            with torch.no_grad():
                q = self.q(torch.tensor(state).unsqueeze(0).to(DEVICE))[0].cpu().numpy()
            q = (
                q
                + self.cfg.op_prior_strength * np.log(prior.reshape(-1) + 1e-8)
                + self.cfg.op_bandit_strength * bandit.mean().reshape(-1)
            )
            if ucb_aug is not None:
                q = ucb_aug.augment_qvalues(q)
            action = int(q.argmax())
            di, ri = divmod(action, N_R)
        return int(action), int(di), int(ri)

    def observe(self, s, a, r, ns, done=0.0):
        self.buf.push(s, a, r, ns, done)

    def train_step(self) -> None:
        self.step += 1
        if len(self.buf) < self.cfg.op_batch:
            return
        (s, a, r, ns, d), idxs, is_w = self.buf.sample(self.cfg.op_batch)
        s = torch.tensor(s).to(DEVICE)
        a = torch.tensor(a, dtype=torch.long).to(DEVICE)
        r = torch.tensor(r).to(DEVICE)
        ns = torch.tensor(ns).to(DEVICE)
        d = torch.tensor(d).to(DEVICE)
        qp = self.q(s).gather(1, a.unsqueeze(1)).squeeze(1)
        with torch.no_grad():
            best_a = self.q(ns).argmax(1).unsqueeze(1)
            qn = self.q_t(ns).gather(1, best_a).squeeze(1)
            target = r + self.cfg.op_gamma * qn * (1 - d)
        td_errors = (qp - target).detach().cpu().numpy()
        self.buf.update_priorities(idxs, td_errors)
        loss = (is_w * F.smooth_l1_loss(qp, target, reduction="none")).mean()
        self.opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.q.parameters(), 1.0)
        self.opt.step()
        tau = self.cfg.ctrl_tau
        for target_param, local_param in zip(self.q_t.parameters(), self.q.parameters()):
            target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)
        self.eps = max(self.cfg.op_eps_end, self.eps * self.cfg.op_eps_decay)


# ---------------------------------------------------------------------------
# Learned Acceptance Criterion
# ---------------------------------------------------------------------------
class LearnedAcceptanceCriterion:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.net = nn.Sequential(
            nn.Linear(cfg.lac_state_dim, cfg.lac_hidden),
            nn.ReLU(),
            nn.Linear(cfg.lac_hidden, cfg.lac_hidden // 2),
            nn.ReLU(),
            nn.Linear(cfg.lac_hidden // 2, 1),
            nn.Sigmoid(),
        ).to(DEVICE)
        self.opt = optim.Adam(self.net.parameters(), lr=cfg.lac_lr)
        self.step = 0
        self._pending: deque = deque()
        self._train_buf: deque = deque(maxlen=cfg.lac_buf_size)

    def features(
        self,
        cost_delta,
        cur_cost,
        temp,
        temp_init,
        no_imp,
        patience,
        nv_diff,
        progress,
        tw_tight_frac,
        fleet_fill,
        avg_slack_val,
    ) -> np.ndarray:
        metro = math.exp(-max(cost_delta, 0.0) / max(temp, 1e-6))
        return np.array(
            [
                cost_delta / max(abs(cur_cost), 1.0),
                temp / max(temp_init, 1e-6),
                no_imp / max(patience, 1),
                float(np.clip(nv_diff, -2, 2)),
                progress,
                tw_tight_frac,
                fleet_fill,
                avg_slack_val,
                metro,
            ],
            dtype=np.float32,
        )

    def decide(self, feats: np.ndarray, cur_best_cost: float) -> tuple[bool, float]:
        self.step += 1
        self._pending.append((feats.copy(), cur_best_cost, self.step))
        metro_p = float(feats[-1])
        if self.step < self.cfg.lac_warmup:
            return random.random() < metro_p, metro_p
        with torch.no_grad():
            p = float(self.net(torch.tensor(feats).unsqueeze(0).to(DEVICE))[0, 0])
        return random.random() < p, p

    def observe(self, current_best_cost: float) -> None:
        cutoff = self.step - self.cfg.lac_horizon
        while self._pending and self._pending[0][2] <= cutoff:
            feats, best_at_t, _ = self._pending.popleft()
            label = 1.0 if current_best_cost < best_at_t - 1e-6 else 0.0
            self._train_buf.append((feats, label))
        if self.step % self.cfg.lac_train_freq == 0 and len(self._train_buf) >= self.cfg.lac_batch:
            self._train()

    def _train(self) -> None:
        batch = random.sample(self._train_buf, min(self.cfg.lac_batch, len(self._train_buf)))
        feats, labels = zip(*batch)
        x = torch.tensor(np.array(feats), dtype=torch.float32).to(DEVICE)
        y = torch.tensor(labels, dtype=torch.float32).to(DEVICE)
        n_neg = max((y == 0).sum().item(), 1)
        n_pos = max((y == 1).sum().item(), 1)
        sample_weights = torch.where(
            y == 1,
            torch.full_like(y, n_neg / n_pos),
            torch.ones_like(y),
        )
        pred = self.net(x).squeeze(1)
        loss = F.binary_cross_entropy(pred, y, weight=sample_weights)
        self.opt.zero_grad()
        loss.backward()
        self.opt.step()

    def state_dict(self) -> dict:
        return {f"lac.{k}": v.clone().cpu() for k, v in self.net.state_dict().items()}

    def load_state_dict(self, weights: dict) -> None:
        sd = self.net.state_dict()
        updates = {}
        for k, v in weights.items():
            bare = k[4:] if k.startswith("lac.") else k
            if bare in sd and tuple(v.shape) == tuple(sd[bare].shape):
                updates[bare] = v.to(DEVICE)
        sd.update(updates)
        self.net.load_state_dict(sd)
```

## File: `src/vrptw/solvers.py`

```python
from __future__ import annotations

import math
import random
import time
from collections import deque

import numpy as np
import torch

from .config import (
    ALGO_ALNS_BASE,
    ALGO_HYBRID_DDQN,
    ALGO_HYBRID_FIXED,
    ALGO_HYBRID_RULE,
    ALGO_ORTOOLS,
    MODE_DEFAULT,
    MODE_DIVERSIFY,
    MODE_INTENSIFY,
    MODE_POOL_RECOMBINE,
    MODE_ROUTE_REDUCE,
    MODE_TW_RESCUE,
    MODES,
    Config,
)
from .core import Inst, Plan, _avg_slack, _fleet_fill, _plan_spread
from .heuristics import build_greedy
from .local_search import (
    _buffered_route_elimination,
    _ejection_chain_eliminate,
    _iterative_route_elimination,
    _try_route_merge,
    local_search,
)
from .operators import DESTROY, N_D, N_R, REPAIR, accept, accept_with_nv_ceiling, destroy_size
from .pool import RoutePool, recombine_with_route_pool
from .rl import (
    DEVICE,
    EliteArchive,
    LearnedAcceptanceCriterion,
    LSBudgetController,
    OperatorController,
    PlateauController,
    ThompsonBandit,
    UCBActionAugmenter,
    WelfordRewardNormalizer,
)

try:
    import ortools  # noqa: F401

    ORTOOLS_OK = True
except ImportError:
    ORTOOLS_OK = False


class ALNSSolver:
    def __init__(self, inst: Inst, cfg: Config):
        self.inst = inst
        self.cfg = cfg
        self.bandit = ThompsonBandit(N_D, N_R)

    def solve(self, seed: int | None = None, init: Plan | None = None) -> tuple[Plan, list[float]]:
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
        cfg = self.cfg
        self.bandit = ThompsonBandit(N_D, N_R)
        cur = init.copy() if init is not None else build_greedy(self.inst, ALGO_ALNS_BASE)
        best = cur.copy()
        temp = cfg.temp_control * cur.cost / math.log(2)
        history = [best.cost]
        no_imp = 0
        self.q_scale = 1.0
        for it in range(cfg.alns_iterations):
            di, ri = self.bandit.select()
            size = destroy_size(it, cfg.alns_iterations, cfg, self.inst.n, scale=self.q_scale)
            dest, removed = DESTROY[di](cur.copy(), size)
            cand = REPAIR[ri](dest, removed)
            score = 0
            if accept(cur, cand, temp):
                if cand.dominates(best):
                    best, score, no_imp = cand.copy(), cfg.sigma1, 0
                elif cand.dominates(cur):
                    score, no_imp = cfg.sigma2, 0
                else:
                    score, no_imp = cfg.sigma3, no_imp + 1
                cur = cand
            else:
                no_imp += 1

            # Adapt q_scale based on whether search is improving or stuck
            if no_imp == 0:
                self.q_scale = max(0.6, self.q_scale * 0.98)
            else:
                self.q_scale = min(1.6, self.q_scale * 1.005)

            self.bandit.update(di, ri, score, cfg.sigma1)
            if (it + 1) % cfg.segment_size == 0:
                self.bandit.decay(cfg.bandit_decay)
            temp *= cfg.temp_decay
            history.append(best.cost)
            if no_imp >= cfg.early_stop_patience:
                break
        best.algo = ALGO_ALNS_BASE
        return best, history


class HybridDDQNSolver:
    algo_name = ALGO_HYBRID_DDQN
    use_op_rl = True

    def __init__(self, inst: Inst, cfg: Config):
        self.inst = inst
        self.cfg = cfg
        self.ctrl = PlateauController(cfg)
        self.op_ctrl = OperatorController(cfg)
        self.lac = LearnedAcceptanceCriterion(cfg)
        self.ls_budget = LSBudgetController(ls_time_frac=0.30)
        self.ucb_aug = UCBActionAugmenter(n_actions=N_D * N_R)
        self.reward_norm = WelfordRewardNormalizer(clip_sigma=8.0, warmup=128)
        self.mode_bandits: list[ThompsonBandit] = [ThompsonBandit(N_D, N_R) for _ in MODES]
        self._segment_recombine_used = False
        self._pool_seeding_done: bool = False  # guard: fires once per solve()
        self._init_nv = 1
        self.archive = EliteArchive(k=cfg.elite_archive_k)

    def clone_weights(self) -> dict:
        weights: dict[str, torch.Tensor] = {}
        for prefix, sd in (("plateau", self.ctrl.q.state_dict()), ("operator", self.op_ctrl.q.state_dict())):
            for k, v in sd.items():
                weights[f"{prefix}.{k}"] = v.clone().cpu()
        weights.update(self.lac.state_dict())
        weights["ucb.mu"] = torch.tensor(self.ucb_aug._mu, dtype=torch.float32)
        weights["ucb.cnt"] = torch.tensor(self.ucb_aug._cnt, dtype=torch.float32)
        weights["ucb.m2"] = torch.tensor(self.ucb_aug._m2, dtype=torch.float32)
        for k, v in self.reward_norm.state_dict().items():
            weights[f"reward_norm.{k}"] = torch.tensor(float(v))
        return weights

    def load_weights(self, weights: dict) -> None:
        plateau_sd = self.ctrl.q.state_dict()
        operator_sd = self.op_ctrl.q.state_dict()
        p_up: dict[str, torch.Tensor] = {}
        o_up: dict[str, torch.Tensor] = {}
        legacy = not any(k.startswith(("plateau.", "operator.")) for k in weights)
        if legacy:
            import warnings

            warnings.warn(
                "load_weights: legacy unprefixed weight format detected. "
                "Re-save weights using clone_weights() to suppress this warning.",
                DeprecationWarning,
                stacklevel=2,
            )
            for k, v in weights.items():
                if k in plateau_sd and tuple(v.shape) == tuple(plateau_sd[k].shape):
                    p_up[k] = v.to(DEVICE)
        else:
            for k, v in weights.items():
                if k.startswith("plateau."):
                    bare = k.split(".", 1)[1]
                    if bare in plateau_sd and tuple(v.shape) == tuple(plateau_sd[bare].shape):
                        p_up[bare] = v.to(DEVICE)
                elif k.startswith("operator."):
                    bare = k.split(".", 1)[1]
                    if bare in operator_sd:
                        if tuple(v.shape) != tuple(operator_sd[bare].shape):
                            # Pad shape from (40, hid2) to (45, hid2) to support the 9th destroy operator
                            if "adv_head.2.weight" in bare:
                                padded = operator_sd[bare].clone()
                                loaded_n = v.shape[0]
                                padded[:loaded_n] = v
                                # Copy weights of actions 25..29 (route_eliminate) to actions 35..39 (costly_route_eliminate)
                                for new_act in range(loaded_n, padded.shape[0]):
                                    rep_idx = new_act % 5
                                    src_act = 5 * 5 + rep_idx
                                    padded[new_act] = v[src_act]
                                o_up[bare] = padded.to(DEVICE)
                            elif "adv_head.2.bias" in bare:
                                padded = operator_sd[bare].clone()
                                loaded_n = v.shape[0]
                                padded[:loaded_n] = v
                                for new_act in range(loaded_n, padded.shape[0]):
                                    rep_idx = new_act % 5
                                    src_act = 5 * 5 + rep_idx
                                    padded[new_act] = v[src_act]
                                o_up[bare] = padded.to(DEVICE)
                        else:
                            o_up[bare] = v.to(DEVICE)
        plateau_sd.update(p_up)
        operator_sd.update(o_up)
        self.ctrl.q.load_state_dict(plateau_sd)
        self.ctrl.q_t.load_state_dict(plateau_sd)
        self.op_ctrl.q.load_state_dict(operator_sd)
        self.op_ctrl.q_t.load_state_dict(operator_sd)
        lac_weights = {k: v for k, v in weights.items() if k.startswith("lac.")}
        if lac_weights:
            self.lac.load_state_dict(lac_weights)

        if "ucb.mu" in weights:
            loaded_mu = weights["ucb.mu"].numpy().astype(np.float64)
            loaded_cnt = weights["ucb.cnt"].numpy().astype(np.float64)
            loaded_m2 = weights["ucb.m2"].numpy().astype(np.float64)
            if len(loaded_mu) < self.ucb_aug.n:
                padded_mu = np.zeros(self.ucb_aug.n, dtype=np.float64)
                padded_cnt = np.ones(self.ucb_aug.n, dtype=np.float64) * 0.5
                padded_m2 = np.ones(self.ucb_aug.n, dtype=np.float64) * 0.5

                padded_mu[: len(loaded_mu)] = loaded_mu
                padded_cnt[: len(loaded_cnt)] = loaded_cnt
                padded_m2[: len(loaded_m2)] = loaded_m2

                for new_act in range(len(loaded_mu), self.ucb_aug.n):
                    rep_idx = new_act % 5
                    src_act = 5 * 5 + rep_idx
                    padded_mu[new_act] = loaded_mu[src_act]
                    padded_cnt[new_act] = loaded_cnt[src_act]
                    padded_m2[new_act] = loaded_m2[src_act]

                self.ucb_aug._mu = padded_mu
                self.ucb_aug._cnt = padded_cnt
                self.ucb_aug._m2 = padded_m2
            else:
                self.ucb_aug._mu = loaded_mu
                self.ucb_aug._cnt = loaded_cnt
                self.ucb_aug._m2 = loaded_m2
            self.ucb_aug._N = float(self.ucb_aug._cnt.sum())

        norm_d = {k.split(".", 1)[1]: float(weights[k]) for k in weights if k.startswith("reward_norm.")}
        if norm_d:
            self.reward_norm.load_state_dict(norm_d)

    def _fleet_pressure(self, plan: Plan, best_nv: float) -> float:
        nv_excess = (plan.nv - best_nv) / max(self._init_nv, 1.0)
        return float(1.0 / (1.0 + math.exp(-8.0 * nv_excess)))

    def _adaptive_potential(self, plan: Plan, best_nv: float, best_td: float) -> float:
        lam = self._fleet_pressure(plan, best_nv)
        nv_penalty_norm = max(plan.nv - best_nv, 0.0) / max(self._init_nv, 1.0)
        td_gap = float(np.clip((plan.cost - best_td) / max(best_td, 1.0) * 100.0, -25.0, 25.0))
        return float(
            -lam * self.cfg.potential_nv_scale * nv_penalty_norm - (1 - lam) * self.cfg.potential_cost_scale * td_gap
        )

    def _state(self, cur, best, no_imp, temp, imp_rate, progress, pool) -> np.ndarray:
        rb, lb = _plan_spread(cur, self.inst)
        t0 = self.cfg.temp_control * max(best.cost, 1.0) / math.log(2)
        pool_fill = min(len(pool._routes) / max(self.cfg.route_pool_limit, 1), 1.0)
        return np.array(
            [
                min(no_imp / max(self.cfg.early_stop_patience, 1), 1.0),
                min((cur.cost - best.cost) / max(best.cost, 1), 1.0),
                min(temp / max(t0, 1e-6), 1.5),
                imp_rate,
                min(cur.nv / max(self._init_nv, 1), 2.0),
                rb,
                lb,
                self.inst.tw_tight_frac,
                _avg_slack(cur),
                _fleet_fill(cur),
                pool_fill,
                progress,
            ],
            dtype=np.float32,
        )

    def _op_state(self, cur, best, mode_idx, it, temp, no_imp, pool, recent_imp) -> np.ndarray:
        rb, lb = _plan_spread(cur, self.inst)
        t0 = self.cfg.temp_control * max(best.cost, 1.0) / math.log(2)
        pool_fill = min(len(pool._routes) / max(self.cfg.route_pool_limit, 1), 1.0)
        return np.array(
            [
                min((cur.cost - best.cost) / max(best.cost, 1), 1.0),
                min(cur.nv / max(self._init_nv, 1), 2.0),
                it / max(self.cfg.hybrid_iterations, 1),
                (it % self.cfg.segment_size) / max(self.cfg.segment_size, 1),
                min(temp / max(t0, 1e-6), 1.5),
                min(no_imp / max(self.cfg.early_stop_patience, 1), 1.0),
                rb,
                lb,
                self.inst.tw_tight_frac,
                _avg_slack(cur),
                _fleet_fill(cur),
                pool_fill,
                mode_idx / max(len(MODES) - 1, 1),
                float(cur.nv - best.nv) / max(self._init_nv, 1),
                recent_imp,
            ],
            dtype=np.float32,
        )

    def _segment_reward(self, best_before, best_after, cur_before, cur_after, accepted_moves, action) -> float:
        lam = self._fleet_pressure(cur_after, best_before.nv)
        base = -0.20 - 0.04 * MODES[action].ls_passes
        if MODES[action].use_recombine:
            base -= 0.06
        denom = max(self._init_nv, 1.0)
        best_nv_gain = (best_before.nv - best_after.nv) / denom
        cur_nv_gain = (cur_before.nv - cur_after.nv) / denom
        best_cost_gain = max((best_before.cost - best_after.cost) / max(best_before.cost, 1) * 100, 0.0)
        cur_cost_gain = max((cur_before.cost - cur_after.cost) / max(cur_before.cost, 1) * 100, 0.0)
        nv_component = lam * (
            8.0 * best_nv_gain * denom + 1.2 * best_cost_gain + 5.0 * cur_nv_gain * denom + 0.6 * cur_cost_gain
        )
        td_component = (1.0 - lam) * (
            3.0 * best_nv_gain * denom + 3.5 * best_cost_gain + 2.0 * cur_nv_gain * denom + 1.8 * cur_cost_gain
        )
        if best_after.nv < best_before.nv:
            base += 15.0 * (best_before.nv - best_after.nv)
        if cur_after.nv < cur_before.nv:
            base += 5.0 * (cur_before.nv - cur_after.nv)
        base += nv_component + td_component
        if accepted_moves <= max(1, self.cfg.segment_size // 10):
            base -= 0.15
        shaped = self.cfg.ctrl_gamma * self._adaptive_potential(
            cur_after, best_before.nv, best_before.cost
        ) - self._adaptive_potential(cur_before, best_before.nv, best_before.cost)
        return float(self.cfg.segment_reward_scale * base + shaped)

    def _iteration_reward(self, cur_before, best_before, cur_after, best_after, accepted) -> float:
        lam = self._fleet_pressure(cur_after, best_before.nv)
        if not accepted:
            base = -0.08
        else:
            base = 0.05
            denom = max(self._init_nv, 1.0)
            best_nv_gain = (best_before.nv - best_after.nv) / denom
            cur_nv_gain = (cur_before.nv - cur_after.nv) / denom
            best_cost_gain = max((best_before.cost - best_after.cost) / max(best_before.cost, 1) * 100, 0.0)
            cur_cost_gain = max((cur_before.cost - cur_after.cost) / max(cur_before.cost, 1) * 100, 0.0)
            nv_component = lam * (
                3.0 * best_nv_gain * denom + 0.40 * best_cost_gain + 2.0 * cur_nv_gain * denom + 0.20 * cur_cost_gain
            )
            td_component = (1.0 - lam) * (
                0.50 * best_nv_gain * denom + 1.80 * best_cost_gain + 0.30 * cur_nv_gain * denom + 0.90 * cur_cost_gain
            )
            if best_after.nv < best_before.nv:
                base += 15.0 * (best_before.nv - best_after.nv)
            if cur_after.nv < cur_before.nv:
                base += 5.0 * (cur_before.nv - cur_after.nv)
            base += nv_component + td_component
            if cur_after.nv > cur_before.nv:
                base -= 0.5 * ((cur_after.nv - cur_before.nv) / denom) * denom
        shaped = self.cfg.op_gamma * self._adaptive_potential(
            cur_after, best_before.nv, best_before.cost
        ) - self._adaptive_potential(cur_before, best_before.nv, best_before.cost)
        return float(self.cfg.iteration_reward_scale * base + shaped)

    def _route_reduce_trigger(self, cur: Plan, no_imp: int) -> bool:
        return no_imp >= self.cfg.plateau_start and _fleet_fill(cur) < max(0.52, 0.80 - 0.25 * self.inst.tw_tight_frac)

    def _select_action(self, state_before, cur, best, no_imp, progress, pool, frozen) -> tuple[int, bool]:
        if no_imp >= max(self.cfg.ctrl_start_floor, self.cfg.ctrl_start // 2):
            return self.ctrl.act(state_before), (not frozen)
        return MODE_DEFAULT, False

    def _refine_candidate(self, cand, action, pool, cur, best, no_imp, iter_idx) -> Plan:
        del cur, iter_idx
        mode = MODES[action]
        refined = cand
        if (
            mode.use_recombine
            and not self._segment_recombine_used
            and no_imp >= max(self.cfg.ctrl_start, self.cfg.plateau_start // 2)
            and len(pool._routes) >= self.cfg.rl_recombine_min_routes
        ):
            self._segment_recombine_used = True
            nv_cap = min(best.nv, refined.nv)
            recombined = recombine_with_route_pool(refined, pool, self.cfg, nv_ceiling=nv_cap)
            if recombined.dominates(refined):
                refined = local_search(
                    recombined, max_passes=1, nv_ceiling=recombined.nv, max_ls_moves=self.cfg.max_ls_moves
                )
        return refined

    def _fixed_nv_polish(self, start: Plan, pool: RoutePool, inherited_bandit: ThompsonBandit | None = None) -> Plan:
        cfg = self.cfg
        target_nv = start.nv
        # Inherit operator statistics from main search instead of cold-starting
        polish_bandit = inherited_bandit.clone() if inherited_bandit is not None else ThompsonBandit(N_D, N_R)
        cur = local_search(start, max_passes=cfg.polish_ls_passes, nv_ceiling=target_nv, max_ls_moves=cfg.max_ls_moves)
        best = cur.copy()
        pool.add_plan(best)
        temp = cfg.temp_control * best.cost / math.log(2)
        no_imp = 0
        for it in range(cfg.polish_iterations):
            di, ri = polish_bandit.select()
            size = destroy_size(it, cfg.polish_iterations, cfg, self.inst.n, scale=0.70)
            dest, removed = DESTROY[di](cur.copy(), size)
            cand = REPAIR[ri](dest, removed)
            cand = local_search(cand, max_passes=1, nv_ceiling=target_nv, max_ls_moves=cfg.max_ls_moves)
            pool.add_plan(cand)
            score, cur_before = 0, cur
            if accept_with_nv_ceiling(cur, cand, temp, target_nv):
                cur = cand
                if cand.nv < target_nv:
                    target_nv = cand.nv
                if cand.dominates(best):
                    best, score, no_imp = cand.copy(), cfg.sigma1, 0
                elif cand.nv == cur_before.nv and cand.cost + 1e-9 < cur_before.cost:
                    score, no_imp = cfg.sigma2, 0
                else:
                    score, no_imp = cfg.sigma3, no_imp + 1
            else:
                no_imp += 1
            polish_bandit.update(di, ri, score, cfg.sigma1)
            if (it + 1) % cfg.segment_size == 0:
                polish_bandit.decay(cfg.bandit_decay)
            temp *= cfg.temp_decay * 0.997
            if no_imp >= cfg.polish_patience:
                break
        best = local_search(best, max_passes=cfg.polish_ls_passes, nv_ceiling=best.nv, max_ls_moves=cfg.max_ls_moves)
        pool.add_plan(best)
        return best

    def _seed_pool_large_destroy(self, best: Plan, pool: RoutePool, n_seeds: int = 30) -> None:
        """
        Dedicated pool-diversity seeding pass.

        Runs n_seeds destroy-repair cycles with large destroy ratios (40-65%) to
        generate consolidated routes covering more customers per vehicle.

        Rationale: normal ALNS (10-40% destroy) produces routes of 5-7 customers.
        RC101 BKS NV=14 requires routes averaging 7.1 customers per route. Large
        destroys force regret-3 repair to build fewer, longer routes â€” exactly
        the missing pool diversity preventing MILP from finding an NV-1 partition.

        NEVER updates best. Only seeds pool with individual routes and full plans.
        """
        inst = self.inst

        for it in range(n_seeds):
            # Stagger ratios: 60% (large) and 42% (medium-large)
            ratio = 0.60 if it % 3 != 1 else 0.42
            size = max(5, int(ratio * inst.n))

            # Alternate shaw (spatial clustering) and route_eliminate (full route removal)
            di = 2 if it % 2 == 0 else 5  # DESTROY[2]=op_shaw, DESTROY[5]=op_route_eliminate
            destroyed, removed = DESTROY[di](best.copy(), size)

            # regret-3 repair: tends to build fewer, longer routes than greedy
            candidate = REPAIR[2](destroyed, removed)  # REPAIR[2] = op_regret_3

            # Add individual routes unconditionally (each route is self-contained feasible)
            for route in candidate.routes:
                pool.add_route(route)

            if candidate.feasible:
                pool.add_plan(candidate)

            # Every 5th seed: attempt explicit route merges on best for direct NV-1 seeds
            if it % 5 == 0:
                merged = _try_route_merge(best)
                if merged is not None:
                    pool.add_plan(merged)
                    for route in merged.routes:
                        pool.add_route(route)

    def _committed_nv_search(
        self,
        start: Plan,
        pool: RoutePool,
        target_nv: int,
        n_iters: int = 300,
    ) -> Plan | None:
        """
        Focused ALNS with hard NV ceiling = target_nv.

        Mechanism:
        - 4Ã— initial temperature: accepts up to ~20% TD regression for NV gain
        - Accepts NV=target_nv+1 candidates and immediately applies LS with
          nv_ceiling=target_nv, catching solutions 'one step away'
        - Always seeds pool with individual routes regardless of acceptance
        - Never modifies self.best or persistent solver state

        Returns best feasible plan at target_nv found, or None.
        """
        cfg = self.cfg
        inst = self.inst

        if start.nv <= target_nv:
            return start.copy()

        cur = start.copy()
        best_found: Plan | None = None
        # High temperature: prioritise NV over TD for this search phase
        temp = cfg.temp_control * cur.cost / math.log(2) * 4.0
        bandit = ThompsonBandit(N_D, N_R)

        for it in range(n_iters):
            di, ri = bandit.select()
            size = destroy_size(it, n_iters, cfg, inst.n, scale=1.0)
            dest, removed = DESTROY[di](cur.copy(), size)
            cand = REPAIR[ri](dest, removed)

            # Unconditional pool seeding â€” every route encountered is valuable
            for route in cand.routes:
                pool.add_route(route)

            score = 0
            if cand.feasible and cand.nv <= target_nv:
                pool.add_plan(cand)
                # SA acceptance within the hard NV ceiling
                if (
                    cand.nv < cur.nv
                    or (cand.nv == cur.nv and cand.cost <= cur.cost)
                    or random.random() < math.exp(-(cand.cost - cur.cost) / max(temp, 1e-6))
                ):
                    cur = cand
                if best_found is None or cand.dominates(best_found) or cand.nv < best_found.nv:
                    best_found = cand.copy()
                    score = cfg.sigma1

            elif cand.feasible and cand.nv == target_nv + 1:
                # One vehicle over target â€” cheap LS pass may close the gap
                cand_ls = local_search(
                    cand,
                    max_passes=1,
                    nv_ceiling=target_nv,
                    max_ls_moves=cfg.max_ls_moves,
                    pool=pool,
                )
                if cand_ls.feasible and cand_ls.nv <= target_nv:
                    pool.add_plan(cand_ls)
                    cur = cand_ls
                    if best_found is None or cand_ls.dominates(best_found) or cand_ls.nv < best_found.nv:
                        best_found = cand_ls.copy()
                        score = cfg.sigma1

            bandit.update(di, ri, score, cfg.sigma1)
            temp *= cfg.temp_decay

        return best_found

    def solve(
        self,
        seed: int | None = None,
        frozen: bool = False,
        init: Plan | None = None,
        shared_norm: WelfordRewardNormalizer | None = None,
    ) -> tuple[Plan, list[float]]:
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
        cfg = self.cfg
        self.ctrl.reset()
        self.op_ctrl.reset()
        self.ls_budget.initialize(cfg)
        self.ucb_aug.reset()
        norm = shared_norm if shared_norm is not None else self.reward_norm
        if shared_norm is None:
            self.reward_norm = WelfordRewardNormalizer(clip_sigma=8.0, warmup=128)
            norm = self.reward_norm
        if getattr(self, "use_op_rl", True):
            self.lac = LearnedAcceptanceCriterion(cfg)
        self.mode_bandits = [ThompsonBandit(N_D, N_R) for _ in MODES]
        pool = RoutePool(self.inst, cfg)
        cur = init.copy() if init is not None else build_greedy(self.inst, self.algo_name)
        best = cur.copy()
        pool.add_plan(cur)
        self._init_nv = cur.nv
        temp = cfg.temp_control * cur.cost / math.log(2)
        all_dw = np.ones((len(MODES), N_D), dtype=np.float32)
        all_rw = np.ones((len(MODES), N_R), dtype=np.float32)
        history: list[float] = [best.cost]
        recent_improvements: deque[int] = deque(maxlen=cfg.segment_size)
        no_imp = 0
        self.q_scale = 1.0
        n_segments = math.ceil(cfg.hybrid_iterations / cfg.segment_size)

        for seg_idx in range(n_segments):
            progress = seg_idx / max(n_segments, 1)
            imp_rate = sum(recent_improvements) / max(len(recent_improvements), 1)
            self._segment_recombine_used = False
            # Plateau-triggered seeding: fires once when plateau first reached
            if (
                not self._pool_seeding_done
                and no_imp >= cfg.plateau_start
                and len(pool._routes) >= cfg.rl_recombine_min_routes
            ):
                self._pool_seeding_done = True
                self._seed_pool_large_destroy(best, pool, n_seeds=25)

            state_before = self._state(cur, best, no_imp, temp, imp_rate, progress, pool)
            action, ctrl_active = self._select_action(state_before, cur, best, no_imp, progress, pool, frozen)
            mode = MODES[action]
            dw = all_dw[action].copy()
            rw = all_rw[action].copy()
            biased_dw = np.maximum(dw * np.array(mode.destroy_bias, np.float32), 0.1)
            biased_rw = np.maximum(rw * np.array(mode.repair_bias, np.float32), 0.1)
            mode_bandit = self.mode_bandits[action]
            temp *= mode.temp_boost
            seg_scores = np.zeros((N_D, N_R))
            seg_counts = np.zeros((N_D, N_R))
            seg_best_pre = best.copy()
            seg_cur_pre = cur.copy()
            accepted_moves = 0

            for offset in range(cfg.segment_size):
                it = seg_idx * cfg.segment_size + offset
                if it >= cfg.hybrid_iterations:
                    break
                op_state = self._op_state(cur, best, action, it, temp, no_imp, pool, imp_rate)
                if getattr(self, "use_op_rl", True):
                    op_action, di, ri = self.op_ctrl.act(
                        op_state,
                        biased_dw,
                        biased_rw,
                        mode_bandit,
                        frozen=frozen,
                        ucb_aug=self.ucb_aug if not frozen else None,
                    )
                else:
                    di, ri = mode_bandit.select(
                        prior=self.op_ctrl._prior(biased_dw, biased_rw),
                        prior_strength=self.cfg.bandit_prior_strength,
                    )
                    op_action = di * N_R + ri
                size = destroy_size(
                    it, cfg.hybrid_iterations, cfg, self.inst.n, scale=mode.destroy_scale * self.q_scale
                )
                cur_before = cur.copy()
                best_before = best.copy()
                dest, removed = DESTROY[di](cur.copy(), size)
                cand = REPAIR[ri](dest, removed)
                cand = self._refine_candidate(cand, action, pool, cur, best, no_imp, it)

                lac_decided = False
                allow_nv_increase = action == MODE_DIVERSIFY
                if not cand.feasible:
                    accepted = False
                elif cand.nv > cur.nv and not (allow_nv_increase and cand.nv == cur.nv + 1):
                    accepted = False
                elif cand.nv < cur.nv or (cand.nv == cur.nv and cand.cost <= cur.cost):
                    accepted = True
                elif cand.nv == cur.nv + 1:
                    accepted = accept(cur, cand, temp, allow_nv_increase=True)
                else:
                    if cfg.lac_enabled and getattr(self, "use_op_rl", True) and not frozen:
                        t0_init = cfg.temp_control * max(best.cost, 1.0) / math.log(2)
                        lac_feats = self.lac.features(
                            cost_delta=cand.cost - cur.cost,
                            cur_cost=cur.cost,
                            temp=temp,
                            temp_init=t0_init,
                            no_imp=no_imp,
                            patience=cfg.early_stop_patience,
                            nv_diff=cand.nv - cur.nv,
                            progress=it / max(cfg.hybrid_iterations, 1),
                            tw_tight_frac=self.inst.tw_tight_frac,
                            fleet_fill=_fleet_fill(cur),
                            avg_slack_val=_avg_slack(cur),
                        )
                        accepted, _ = self.lac.decide(lac_feats, best.cost)
                        lac_decided = True
                    else:
                        accepted = random.random() < math.exp(-(cand.cost - cur.cost) / max(temp, 1e-6))

                if lac_decided:
                    self.lac.observe(best.cost)

                score = 0
                improved = False
                if accepted:
                    accepted_moves += 1
                    is_new_best = cand.dominates(best)
                    if not frozen and self.ls_budget.should_trigger(action, True, is_new_best, MODES):
                        t_ls = time.time()
                        cost_pre = cand.cost
                        nv_cap = (
                            best.nv
                            if action in (MODE_INTENSIFY, MODE_TW_RESCUE, MODE_POOL_RECOMBINE, MODE_ROUTE_REDUCE)
                            else None
                        )
                        cand = local_search(
                            cand, max_passes=MODES[action].ls_passes, nv_ceiling=nv_cap, max_ls_moves=cfg.max_ls_moves
                        )
                        self.ls_budget.record(time.time() - t_ls, cost_pre, cand.cost)
                    improved = cand.dominates(cur)
                    pool.add_plan(cand)
                    if cand.nv <= best.nv and cand.dominates(best):
                        best, score, no_imp = cand.copy(), cfg.sigma1, 0
                        pool.add_plan(best)
                    elif improved:
                        score, no_imp = cfg.sigma2, 0
                    else:
                        score, no_imp = cfg.sigma3, no_imp + 1
                    cur = cand
                else:
                    no_imp += 1

                recent_improvements.append(1 if improved else 0)
                seg_scores[di, ri] += score
                seg_counts[di, ri] += 1
                mode_bandit.update(di, ri, score, cfg.sigma1)
                cur_after = cur.copy()
                best_after = best.copy()
                next_imp = sum(recent_improvements) / max(len(recent_improvements), 1)
                next_state = self._op_state(cur_after, best_after, action, it + 1, temp, no_imp, pool, next_imp)
                done = 1.0 if no_imp >= cfg.early_stop_patience else 0.0
                if not frozen and getattr(self, "use_op_rl", True):
                    iter_rew_raw = self._iteration_reward(cur_before, best_before, cur_after, best_after, accepted)
                    iter_rew_norm = norm.normalize(iter_rew_raw)
                    self.ucb_aug.update(op_action, iter_rew_raw)
                    if iter_rew_norm is not None:
                        self.op_ctrl.observe(
                            op_state,
                            op_action,
                            iter_rew_norm,
                            next_state,
                            done,
                        )
                    if (it + 1) % 4 == 0:
                        self.op_ctrl.train_step()

                # Adapt q_scale based on whether search is improving or stuck
                if no_imp == 0:
                    self.q_scale = max(0.6, self.q_scale * 0.98)
                else:
                    self.q_scale = min(1.6, self.q_scale * 1.005)

                temp *= cfg.temp_decay * mode.temp_decay_scale
                history.append(best.cost)
                if no_imp >= cfg.early_stop_patience:
                    break

            for mb in self.mode_bandits:
                mb.decay(cfg.bandit_decay)
            for d in range(N_D):
                for r in range(N_R):
                    if seg_counts[d, r] > 0:
                        avg = seg_scores[d, r] / seg_counts[d, r]
                        dw[d] = dw[d] * (1 - cfg.weight_decay) + avg * cfg.weight_decay
                        rw[r] = rw[r] * (1 - cfg.weight_decay) + avg * cfg.weight_decay
            all_dw[action] = np.maximum(dw, 0.1)
            all_rw[action] = np.maximum(rw, 0.1)

            state_after = self._state(
                cur,
                best,
                no_imp,
                temp,
                sum(recent_improvements) / max(len(recent_improvements), 1),
                min((seg_idx + 1) / max(n_segments, 1), 1.0),
                pool,
            )
            if ctrl_active:
                self.ctrl.observe(
                    state_before,
                    action,
                    self._segment_reward(seg_best_pre, best, seg_cur_pre, cur, accepted_moves, action),
                    state_after,
                    0.0,
                )
                self.ctrl.train_step()
            if no_imp >= cfg.early_stop_patience:
                break

        if cfg.recombine_after_main_search:
            recombined = recombine_with_route_pool(best, pool, cfg, nv_ceiling=best.nv)
            if recombined.dominates(best):
                best = recombined
                pool.add_plan(best)
                history.append(best.cost)

        best = self._fixed_nv_polish(best, pool, inherited_bandit=self.mode_bandits[MODE_INTENSIFY])
        history.append(best.cost)

        if cfg.recombine_after_polish:
            recombined = recombine_with_route_pool(best, pool, cfg, nv_ceiling=best.nv)
            if recombined.dominates(best):
                best = local_search(
                    recombined, max_passes=cfg.polish_ls_passes, nv_ceiling=recombined.nv, max_ls_moves=cfg.max_ls_moves
                )
                history.append(best.cost)

        # Enrich pool before MILP queries: large-destroy seeds cover the 7-9 customer
        # route types that normal search under-generates.
        self._seed_pool_large_destroy(best, pool, n_seeds=20)

        # Explicit NV-reduction loop: try MILP with nv_target = best.nv-1, best.nv-2
        # Accept any feasible result with fewer vehicles (BKS has *worse* TD for fewer NV).
        for _target_nv in range(best.nv - 1, max(best.nv - 3, 0), -1):
            _rec = recombine_with_route_pool(best, pool, cfg, nv_target=_target_nv)
            if not _rec.feasible or _rec.nv > _target_nv:
                break  # pool lacks routes to cover at this NV; lower targets will also fail
            # Deep LS to improve TD at the new NV (2x default moves)
            _rec = local_search(
                _rec,
                max_passes=cfg.polish_ls_passes + 1,
                nv_ceiling=_rec.nv,
                max_ls_moves=cfg.max_ls_moves * 2,
                pool=pool,  # seed intermediates into pool
            )
            if _rec.feasible and _rec.nv <= _target_nv:
                best = _rec
                pool.add_plan(best)
                history.append(best.cost)
                # Don't break â€” try to go one lower

        # Pass 1: depth-2 ejection chain (handles TW-blocked direct insertions)
        chain_result = _ejection_chain_eliminate(best)
        if (
            chain_result is not None
            and chain_result.feasible
            and (chain_result.nv < best.nv or chain_result.dominates(best))
        ):
            best = chain_result
            pool.add_plan(best)
            history.append(best.cost)

        # Pass 2: buffered elimination (bounded multi-route rebalancing)
        buffered = _buffered_route_elimination(best, pool=pool)
        if buffered.feasible and (buffered.nv < best.nv or buffered.dominates(best)):
            best = buffered
            pool.add_plan(best)
            history.append(best.cost)

        # Pass 3: iterative greedy elimination (catches easier direct insertions)
        eliminated = _iterative_route_elimination(best, self.inst, pool=pool)
        if eliminated.feasible and (eliminated.nv < best.nv or eliminated.dominates(best)):
            best = eliminated
            pool.add_plan(best)
            history.append(best.cost)

        # Pass 4: committed NV search -- only if still above target
        from .config import BKS as _BKS

        _bks = _BKS.get(self.inst.name)
        _bks_nv = int(_bks["nv"]) if _bks else max(1, best.nv - 1)
        if best.nv > _bks_nv:
            committed = self._committed_nv_search(best, pool, target_nv=best.nv - 1)
            if committed is not None and committed.feasible and (committed.nv < best.nv or committed.dominates(best)):
                best = committed
                pool.add_plan(best)
                history.append(best.cost)
                # One more ejection attempt at the new, lower NV
                chain2 = _ejection_chain_eliminate(best)
                if chain2 is not None and chain2.feasible and (chain2.nv < best.nv or chain2.dominates(best)):
                    best = chain2
                    pool.add_plan(best)
                    history.append(best.cost)

        best.algo = self.algo_name
        self.archive.update(best)
        return best, history


# ---------------------------------------------------------------------------
# Hybrid-Fixed
# ---------------------------------------------------------------------------
class HybridFixedSolver(HybridDDQNSolver):
    algo_name = ALGO_HYBRID_FIXED
    use_op_rl = False

    def _select_action(self, state_before, cur, best, no_imp, progress, pool, frozen):
        del state_before, best, progress, pool, frozen
        if self._route_reduce_trigger(cur, no_imp):
            return MODE_ROUTE_REDUCE, False
        return MODE_DEFAULT, False

    def solve(self, seed=None, frozen=True, init=None):
        plan, history = super().solve(seed=seed, frozen=True, init=init)
        plan.algo = self.algo_name
        return plan, history


# ---------------------------------------------------------------------------
# Hybrid-Rule
# ---------------------------------------------------------------------------
class HybridRuleSolver(HybridDDQNSolver):
    algo_name = ALGO_HYBRID_RULE
    use_op_rl = False

    def _select_action(self, state_before, cur, best, no_imp, progress, pool, frozen) -> tuple[int, bool]:
        del state_before, best, frozen
        if self._route_reduce_trigger(cur, no_imp):
            return MODE_ROUTE_REDUCE, False
        pool_ready = len(pool._routes) >= max(self.cfg.rl_recombine_min_routes, max(12, cur.nv * 2))
        fleet_fill = _fleet_fill(cur)
        slack = _avg_slack(cur)
        if pool_ready and no_imp >= max(10, self.cfg.ctrl_start // 2) and fleet_fill >= 0.66 and progress < 0.92:
            return MODE_POOL_RECOMBINE, False
        if self.inst.tw_tight_frac >= 0.18 and slack < 0.16 and no_imp >= max(8, self.cfg.ctrl_start // 2):
            return MODE_TW_RESCUE, False
        if no_imp >= max(self.cfg.ctrl_start_floor, self.cfg.ctrl_start // 2):
            return (MODE_DIVERSIFY if progress < 0.45 else MODE_INTENSIFY), False
        return MODE_DEFAULT, False

    def solve(self, seed=None, frozen=True, init=None):
        plan, history = super().solve(seed=seed, frozen=True, init=init)
        plan.algo = self.algo_name
        return plan, history


PlateauHybridSolver = HybridDDQNSolver
ScheduledHybridSolver = HybridRuleSolver
RLALNSSolver = HybridDDQNSolver


def run_ortools(inst: Inst, cfg: Config) -> tuple[Plan | None, float]:
    if not ORTOOLS_OK:
        print("  [OR-Tools] not installed â€” skipping")
        return None, 0.0
    from ortools.constraint_solver import pywrapcp, routing_enums_pb2

    scale = 100
    n_nodes = inst.n + 1
    n_vehicles = inst.n
    manager = pywrapcp.RoutingIndexManager(n_nodes, n_vehicles, 0)
    routing = pywrapcp.RoutingModel(manager)
    dist_mat = (inst.dist * scale).astype(int)
    serv_int = (inst.service_times * scale).astype(int)

    def transit_cb(fi, ti):
        fn, tn = manager.IndexToNode(fi), manager.IndexToNode(ti)
        return int(dist_mat[fn, tn]) + int(serv_int[fn])

    transit_idx = routing.RegisterTransitCallback(transit_cb)
    routing.SetArcCostEvaluatorOfAllVehicles(transit_idx)
    demands_int = inst.demands.astype(int)

    def demand_cb(fi):
        return int(demands_int[manager.IndexToNode(fi)])

    demand_idx = routing.RegisterUnaryTransitCallback(demand_cb)
    routing.AddDimensionWithVehicleCapacity(demand_idx, 0, [int(inst.capacity)] * n_vehicles, True, "Capacity")
    routing.AddDimension(transit_idx, int(inst.horizon * scale), int(inst.horizon * scale), False, "Time")
    time_dim = routing.GetDimensionOrDie("Time")
    for node in range(1, inst.n + 1):
        idx = manager.NodeToIndex(node)
        time_dim.CumulVar(idx).SetRange(int(inst.ready_times[node] * scale), int(inst.due_times[node] * scale))
    for v in range(n_vehicles):
        routing.AddVariableMinimizedByFinalizer(time_dim.CumulVar(routing.Start(v)))
        routing.AddVariableMinimizedByFinalizer(time_dim.CumulVar(routing.End(v)))
    params = pywrapcp.DefaultRoutingSearchParameters()
    params.first_solution_strategy = routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
    params.local_search_metaheuristic = routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
    params.time_limit.seconds = int(cfg.ortools_time_limit)
    params.log_search = False
    t0 = time.time()
    solution = routing.SolveWithParameters(params)
    elapsed = time.time() - t0
    if not solution:
        print(f"  [OR-Tools] no solution ({elapsed:.1f}s)")
        return None, elapsed
    routes: list[list[int]] = []
    for v in range(n_vehicles):
        route: list[int] = []
        idx = routing.Start(v)
        while not routing.IsEnd(idx):
            node = manager.IndexToNode(idx)
            if node != 0:
                route.append(node)
            idx = solution.Value(routing.NextVar(idx))
        if route:
            routes.append(route)
    plan = Plan(routes, inst, ALGO_ORTOOLS)
    if not plan.feasible:
        print(f"  [OR-Tools] infeasible ({elapsed:.1f}s)")
        return None, elapsed
    return plan, elapsed
```

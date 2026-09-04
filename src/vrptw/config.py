from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any

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
    "C201": {"nv": 3, "td": 591.56},
    "C202": {"nv": 3, "td": 591.56},
    "C203": {"nv": 3, "td": 591.17},
    "C204": {"nv": 3, "td": 590.60},
    "C205": {"nv": 3, "td": 588.88},
    "C206": {"nv": 3, "td": 588.49},
    "C207": {"nv": 3, "td": 588.29},
    "C208": {"nv": 3, "td": 588.32},
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
    "R111": {"nv": 10, "td": 1096.73},
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
    "RC101": {"nv": 14, "td": 1696.95},
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
    "c1_2_1": {"nv": 20, "td": 2704.57},
    "c1_2_2": {"nv": 18, "td": 2917.89},
    "c1_2_3": {"nv": 18, "td": 2707.35},
    "c1_2_4": {"nv": 18, "td": 2643.31},
    "c1_2_5": {"nv": 20, "td": 2702.05},
    "c1_2_6": {"nv": 20, "td": 2701.04},
    "c1_2_7": {"nv": 20, "td": 2701.04},
    "c1_2_8": {"nv": 19, "td": 2775.48},
    "c1_2_9": {"nv": 18, "td": 2687.83},
    "c1_2_10": {"nv": 18, "td": 2643.51},
    "c2_2_1": {"nv": 6, "td": 1931.44},
    "c2_2_2": {"nv": 6, "td": 1863.16},
    "c2_2_3": {"nv": 6, "td": 1775.08},
    "c2_2_4": {"nv": 6, "td": 1703.43},
    "c2_2_5": {"nv": 6, "td": 1878.85},
    "c2_2_6": {"nv": 6, "td": 1857.35},
    "c2_2_7": {"nv": 6, "td": 1849.46},
    "c2_2_8": {"nv": 6, "td": 1820.53},
    "c2_2_9": {"nv": 6, "td": 1830.05},
    "c2_2_10": {"nv": 6, "td": 1806.58},
    "r1_2_1": {"nv": 20, "td": 4784.11},
    "r1_2_2": {"nv": 18, "td": 4039.86},
    "r1_2_3": {"nv": 18, "td": 3381.96},
    "r1_2_4": {"nv": 18, "td": 3057.81},
    "r1_2_5": {"nv": 18, "td": 4107.86},
    "r1_2_6": {"nv": 18, "td": 3583.14},
    "r1_2_7": {"nv": 18, "td": 3150.11},
    "r1_2_8": {"nv": 18, "td": 2951.99},
    "r1_2_9": {"nv": 18, "td": 3760.58},
    "r1_2_10": {"nv": 18, "td": 3301.18},
    "r2_2_1": {"nv": 4, "td": 4483.16},
    "r2_2_2": {"nv": 4, "td": 3621.20},
    "r2_2_3": {"nv": 4, "td": 2880.62},
    "r2_2_4": {"nv": 4, "td": 1981.29},
    "r2_2_5": {"nv": 4, "td": 3366.79},
    "r2_2_6": {"nv": 4, "td": 2913.03},
    "r2_2_7": {"nv": 4, "td": 2451.14},
    "r2_2_8": {"nv": 4, "td": 1849.87},
    "r2_2_9": {"nv": 4, "td": 3092.04},
    "r2_2_10": {"nv": 4, "td": 2654.97},
    "rc1_2_1": {"nv": 18, "td": 3602.80},
    "rc1_2_2": {"nv": 18, "td": 3249.05},
    "rc1_2_3": {"nv": 18, "td": 3008.33},
    "rc1_2_4": {"nv": 18, "td": 2851.68},
    "rc1_2_5": {"nv": 18, "td": 3371.00},
    "rc1_2_6": {"nv": 18, "td": 3324.80},
    "rc1_2_7": {"nv": 18, "td": 3189.32},
    "rc1_2_8": {"nv": 18, "td": 3083.93},
    "rc1_2_9": {"nv": 18, "td": 3081.13},
    "rc1_2_10": {"nv": 18, "td": 3000.30},
    "rc2_2_1": {"nv": 6, "td": 3099.53},
    "rc2_2_2": {"nv": 5, "td": 2825.24},
    "rc2_2_3": {"nv": 4, "td": 2601.87},
    "rc2_2_4": {"nv": 4, "td": 2038.56},
    "rc2_2_5": {"nv": 4, "td": 2911.46},
    "rc2_2_6": {"nv": 4, "td": 2873.12},
    "rc2_2_7": {"nv": 4, "td": 2525.83},
    "rc2_2_8": {"nv": 4, "td": 2292.53},
    "rc2_2_9": {"nv": 4, "td": 2175.04},
    "rc2_2_10": {"nv": 4, "td": 2015.60},
    "c1_4_1": {"nv": 40, "td": 7152.02},
    "c1_4_2": {"nv": 36, "td": 7686.38},
    "c1_4_3": {"nv": 36, "td": 7060.67},
    "c1_4_4": {"nv": 36, "td": 6803.24},
    "c1_4_5": {"nv": 40, "td": 7152.02},
    "c1_4_6": {"nv": 40, "td": 7153.41},
    "c1_4_7": {"nv": 39, "td": 7417.92},
    "c1_4_8": {"nv": 37, "td": 7347.23},
    "c1_4_9": {"nv": 36, "td": 7042.53},
    "c1_4_10": {"nv": 36, "td": 6860.63},
    "c2_4_1": {"nv": 12, "td": 4116.05},
    "c2_4_2": {"nv": 12, "td": 3929.89},
    "c2_4_3": {"nv": 11, "td": 4018.02},
    "c2_4_4": {"nv": 11, "td": 3702.49},
    "c2_4_5": {"nv": 12, "td": 3938.69},
    "c2_4_6": {"nv": 12, "td": 3875.94},
    "c2_4_7": {"nv": 12, "td": 3894.13},
    "c2_4_8": {"nv": 11, "td": 4233.20},
    "c2_4_9": {"nv": 12, "td": 3864.68},
    "c2_4_10": {"nv": 11, "td": 3825.67},
    "r1_4_1": {"nv": 40, "td": 10372.31},
    "r1_4_2": {"nv": 36, "td": 8898.15},
    "r1_4_3": {"nv": 36, "td": 7808.34},
    "r1_4_4": {"nv": 36, "td": 7282.78},
    "r1_4_5": {"nv": 36, "td": 9222.96},
    "r1_4_6": {"nv": 36, "td": 8358.69},
    "r1_4_7": {"nv": 36, "td": 7616.15},
    "r1_4_8": {"nv": 36, "td": 7257.28},
    "r1_4_9": {"nv": 36, "td": 8694.78},
    "r1_4_10": {"nv": 36, "td": 8094.10},
    "r2_4_1": {"nv": 8, "td": 9210.15},
    "r2_4_2": {"nv": 8, "td": 7606.75},
    "r2_4_3": {"nv": 8, "td": 5911.07},
    "r2_4_4": {"nv": 8, "td": 4241.47},
    "r2_4_5": {"nv": 8, "td": 7127.83},
    "r2_4_6": {"nv": 8, "td": 6122.60},
    "r2_4_7": {"nv": 8, "td": 5018.53},
    "r2_4_8": {"nv": 8, "td": 4015.60},
    "r2_4_9": {"nv": 8, "td": 6400.10},
    "r2_4_10": {"nv": 8, "td": 5773.17},
    "rc1_4_1": {"nv": 36, "td": 8571.32},
    "rc1_4_2": {"nv": 36, "td": 7892.52},
    "rc1_4_3": {"nv": 36, "td": 7533.05},
    "rc1_4_4": {"nv": 36, "td": 7308.55},
    "rc1_4_5": {"nv": 36, "td": 8172.64},
    "rc1_4_6": {"nv": 36, "td": 8164.98},
    "rc1_4_7": {"nv": 36, "td": 7948.51},
    "rc1_4_8": {"nv": 36, "td": 7772.57},
    "rc1_4_9": {"nv": 36, "td": 7733.35},
    "rc1_4_10": {"nv": 36, "td": 7596.04},
    "rc2_4_1": {"nv": 11, "td": 6682.37},
    "rc2_4_2": {"nv": 9, "td": 6180.62},
    "rc2_4_3": {"nv": 8, "td": 4930.84},
    "rc2_4_4": {"nv": 8, "td": 3631.01},
    "rc2_4_5": {"nv": 8, "td": 6706.44},
    "rc2_4_6": {"nv": 8, "td": 5766.61},
    "rc2_4_7": {"nv": 8, "td": 5334.72},
    "rc2_4_8": {"nv": 8, "td": 4790.14},
    "rc2_4_9": {"nv": 8, "td": 4551.11},
    "rc2_4_10": {"nv": 8, "td": 4278.61},
}

ALGO_ORTOOLS = "OR-Tools"
ALGO_ALNS_BASE = "ALNS-Base"
ALGO_ALNS_BASE_PLUS = "ALNS-Base+"
ALGO_HYBRID_FIXED = "Hybrid-Fixed"
ALGO_HYBRID_RULE = "Hybrid-Rule"
ALGO_FLAT_DDQN = "Flat-DDQN"
ALGO_HYBRID_DDQN = "Hybrid-DDQN"
ALGO_HYBRID_DDQN_TRANSFER = "Hybrid-DDQN-Transfer"
ALGO_HYBRID_DDQN_TRANSFER_RC2 = "Hybrid-DDQN-Transfer-RC2"
ALGO_HYBRID_DDQN_TRANSFER_DR = "Hybrid-DDQN-Transfer-DR"
ALGO_DQN = "DQN"

ALGO_ORDER = [
    ALGO_ORTOOLS,
    ALGO_ALNS_BASE,
    ALGO_ALNS_BASE_PLUS,
    ALGO_HYBRID_FIXED,
    ALGO_HYBRID_RULE,
    ALGO_FLAT_DDQN,
    ALGO_HYBRID_DDQN,
    ALGO_HYBRID_DDQN_TRANSFER,
    ALGO_HYBRID_DDQN_TRANSFER_RC2,
    ALGO_HYBRID_DDQN_TRANSFER_DR,
    ALGO_DQN,
]

LEGACY_ALGO_LABELS = {
    "ALNS": ALGO_ALNS_BASE,
    "ALNS-Base": ALGO_ALNS_BASE,
    "ALNS-Base+": ALGO_ALNS_BASE_PLUS,
    "ALNS+": ALGO_HYBRID_FIXED,
    "ALNS-FAIR": ALGO_HYBRID_FIXED,
    "Hybrid-Fixed": ALGO_HYBRID_FIXED,
    "ALNS++": ALGO_HYBRID_RULE,
    "SCHED-ALNS": ALGO_HYBRID_RULE,
    "Hybrid-Rule": ALGO_HYBRID_RULE,
    "Flat-DDQN": ALGO_FLAT_DDQN,
    "Flat-DDQN-ALNS": ALGO_FLAT_DDQN,
    "DDQN-ALNS": ALGO_HYBRID_DDQN,
    "PLATEAU-HYBRID": ALGO_HYBRID_DDQN,
    "Hybrid-DDQN": ALGO_HYBRID_DDQN,
    "DDQN-ALNS*": ALGO_HYBRID_DDQN_TRANSFER,
    "DDQN-ALNS★": ALGO_HYBRID_DDQN_TRANSFER,
    "Hybrid-DDQN-Transfer": ALGO_HYBRID_DDQN_TRANSFER,
    "Hybrid-DDQN-Transfer-RC2": ALGO_HYBRID_DDQN_TRANSFER_RC2,
    "Hybrid-DDQN-Transfer-DR": ALGO_HYBRID_DDQN_TRANSFER_DR,
    "dqn": ALGO_DQN,
    "DQN": ALGO_DQN,
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
# Config — tuned for i7-14700KF (28 cores), n_runs=3
# ---------------------------------------------------------------------------
@dataclass
class Config:
    data_path: str = field(default_factory=default_data_path)
    output_dir: str = field(default_factory=default_output_dir)

    # ── iterations (reduced from 3500; early-stop exits stagnation faster) ─
    alns_iterations: int = 5000
    hybrid_iterations: int = 5000
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

    # ── plateau controller ─────────────────────────────────────────────────
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
    macro_enabled: bool = True  # If False, disables Macro Plateau Controller completely (pins mode=DEFAULT)
    pool_recombine_enabled: bool = True  # If False, disables periodic and plateau Set Partitioning recombination
    nv_increase_penalty: float = 15.0
    rl_recombine_min_routes: int = 24
    ctrl_tau: float = 0.005  # soft target update rate for PlateauController
    per_beta_auto_scale: bool = True
    per_beta_steps: int = 50_000  # steps over which beta anneals 0.4 → 1.0
    # ── operator controller ────────────────────────────────────────────────
    op_state_dim: int = 20
    op_hidden: int = 128
    op_lr: float = 3e-4
    op_gamma: float = 0.97
    distilled_weights_path: str | None = None
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
    op_softmax_tau: float = 1.0  # Boltzmann temperature for entropy confidence gate blending
    op_use_entropy_gate: bool = True  # If False, disables entropy confidence weighting (w_conf = 1.0)

    bandit_decay: float = 0.95
    bandit_prior_strength: float = 0.18
    potential_nv_scale: float = 15.0
    potential_cost_scale: float = 0.18
    segment_reward_scale: float = 0.30
    iteration_reward_scale: float = 0.45

    # ── route pool / set-partitioning ─────────────────────────────────────
    route_pool_limit: int = 600
    dynamic_pool_factor: float = 4.0  # Dynamic pool capacity scaling: max(limit, dynamic_pool_factor * N)
    route_pool_max_per_customer: int = 28
    sp_time_limit: float = 4.0
    sp_vehicle_penalty_scale: float = 200.0
    sp_cert_margin: float = 0.20  # Analytic clearance margin over worst-case route distance disparity
    sp_enforce_certified_penalty: bool = True  # Enforce lambda_pen^cert = max(grid, (1+eps)*K_max*c_max^route)
    recombine_interval: int = 100
    milp_max_cols: int = 800

    # ── generalized ejection chains (GEC v2.0) ───────────────────────────
    gec_max_depth: int = 3  # Depth of recursive 3-level customer displacement tree in MODE_ROUTE_REDUCE
    gec_candidate_k: int = 15  # Candidate neighbor branching factor in GEC

    # ── dynamic spatiotemporal GNN attention ──────────────────────────────
    dynamic_gnn_tw_factor: float = 0.5  # Modulates GNN edge heatmap with dynamic TW slack

    # ── adaptive scale bypass (Limitation 1) ──────────────────────────────
    adaptive_scale_bypass: bool = False
    min_neural_customers: int = 50

    pool_cls: Any = None

    # ── polish ────────────────────────────────────────────────────────────
    polish_ls_passes: int = 2
    max_ls_moves: int = 15
    recombine_after_main_search: bool = True
    recombine_after_polish: bool = True

    # ── transfer ──────────────────────────────────────────────────────────
    transfer_epochs: int = 1
    transfer_shuffle: bool = True
    rc2_transfer_split: int = 4

    # ── elite archive ─────────────────────────────────────────────────────
    elite_archive_k: int = 5

    # ── constructive DQN (ablation only) ──────────────────────────────────
    dqn_state_dim: int = 13
    dqn_hidden: int = 128
    dqn_lr: float = 1e-3
    dqn_gamma: float = 0.99
    dqn_buffer: int = 8192
    dqn_batch: int = 64
    dqn_eps_start: float = 1.0
    dqn_eps_end: float = 0.05
    dqn_eps_decay: float = 0.995
    dqn_target_freq: int = 20
    dqn_train_freq: int = 5
    dqn_vehicle_penalty: float = 5.0

    # ── RL-guided split controller ────────────────────────────────────────
    split_state_dim: int = 18
    split_hidden: int = 64
    split_lr: float = 3e-4
    split_gamma: float = 0.95
    split_buffer: int = 10_000
    split_batch: int = 32
    split_eps_start: float = 0.30
    split_eps_end: float = 0.02
    split_tau: float = 0.005  # soft target update rate for SplitController
    split_trigger_interval: int = 50  # try split every N iterations
    split_trigger_after: int = 200  # don't try before iteration 200
    split_nv_penalty: float = 5.0  # episode reward scale for NV reduction
    split_infeasible_penalty: float = 10.0

    # ── OR-Tools ──────────────────────────────────────────────────────────
    ortools_time_limit: float = 15.0

    # ── Learned Acceptance Criterion ──────────────────────────────────────
    lac_enabled: bool = True
    lac_state_dim: int = 9
    lac_hidden: int = 48
    lac_lr: float = 1e-3
    lac_warmup: int = 300
    lac_horizon: int = 80
    lac_train_freq: int = 20
    lac_buf_size: int = 5000
    lac_batch: int = 64  # batch size for training the learned acceptance criterion
    lac_tau_min: float = 0.40  # Initial exploratory acceptance probability threshold
    lac_tau_max: float = 0.75  # Final intensive exploitation acceptance threshold
    # ── domain randomization ──────────────────────────────────────────────
    domain_randomization_epochs: int = 20
    domain_randomization_batch: int = 15
    # ── SOTA GNN guidance ────────────────────────────────────────────────
    gnn_guidance_strength: float = 0.45
    gnn_pruning_threshold: float = 0.01
    gnn_pruning_threshold_start: float = 0.05
    gnn_pruning_threshold_end: float = 0.003
    gnn_model_path: str | None = None

    # ── Penalty-Based Infeasible Search ───────────────────────────────────
    penalty_search_enabled: bool = False

    # ── Adaptive Feasibility Management (NAMI) ────────────────────────────
    adaptive_feasibility: bool = True
    target_feasible_ratio: float = 0.5
    phase1_ratio: float = 0.60

    # ── Lagrangian Dual Penalty Search (ALPS) ─────────────────────────────
    lagrangian_penalty: bool = False
    lagrangian_theta: float = 2.0
    lagrangian_stall_limit: int = 20
    highs_plateau_synthesis: bool = True

    # ── Split Controller ──────────────────────────────────────────────────
    split_enabled: bool = False

    # ── Anytime wall-clock budget ─────────────────────────────────────────
    # The search is otherwise bounded only by iteration count, which makes run
    # times unpredictable (measured ~5.6h/run at n=400) and makes the OR-Tools
    # comparison non-iso-time. When ``time_limit`` is None it is derived per
    # instance as ``time_limit_per_customer * inst.n``; set it explicitly (or set
    # ``time_limit_per_customer`` to 0.0) to opt out.
    time_limit: float | None = None
    time_limit_per_customer: float = 0.6
    # Fraction of the budget the main destroy/repair loop may consume. The
    # remainder is reserved for the tail phases (route elimination, TD polish,
    # recombination), which contribute most of the final TD quality.
    time_limit_main_loop_frac: float = 0.80

    def __post_init__(self) -> None:
        if self.per_beta_auto_scale:
            self.per_beta_steps = self.hybrid_iterations * 10
        self.validate()

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
        if self.split_tau <= 0.0 or self.split_tau >= 1.0:
            raise ValueError(f"split_tau must be > 0 and < 1, got {self.split_tau}")
        if self.split_trigger_interval <= 0:
            raise ValueError(f"split_trigger_interval must be > 0, got {self.split_trigger_interval}")
        if self.per_beta_steps <= 0:
            raise ValueError(f"per_beta_steps must be > 0, got {self.per_beta_steps}")
        if self.lac_batch < 16:
            raise ValueError(f"lac_batch must be >= 16, got {self.lac_batch}")
        if self.ctrl_start_floor < 1:
            raise ValueError(f"ctrl_start_floor must be >= 1, got {self.ctrl_start_floor}")
        if self.time_limit is not None and self.time_limit <= 0.0:
            raise ValueError(f"time_limit must be > 0 when set, got {self.time_limit}")
        if self.time_limit_per_customer < 0.0:
            raise ValueError(f"time_limit_per_customer must be >= 0, got {self.time_limit_per_customer}")
        if not (0.0 < self.time_limit_main_loop_frac <= 1.0):
            raise ValueError(f"time_limit_main_loop_frac must be in (0.0, 1.0], got {self.time_limit_main_loop_frac}")


# ---------------------------------------------------------------------------
# Mode specifications
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class ModeSpec:
    name: str
    destroy_scale: float
    temp_boost: float
    temp_decay_scale: float
    destroy_bias: tuple[float, ...]  # length == N_D = 13
    repair_bias: tuple[float, ...]  # length == N_R = 5
    ls_passes: int
    use_recombine: bool


MODES: tuple[ModeSpec, ...] = (
    ModeSpec(
        "default",
        1.00,
        1.00,
        1.000,
        (1.0, 1.0, 1.0, 1.0, 1.0, 0.8, 0.8, 0.8, 1.0, 0.8, 0.5, 1.0, 1.0),
        (1.0, 1.0, 1.0, 1.0, 1.1),
        0,
        False,
    ),
    ModeSpec(
        "intensify",
        0.70,
        0.98,
        0.995,
        (0.5, 1.3, 1.2, 0.5, 1.0, 0.7, 0.8, 0.8, 0.9, 0.6, 0.3, 1.5, 1.4),
        (1.3, 1.2, 0.8, 1.0, 1.3),
        1,
        False,
    ),
    ModeSpec(
        "diversify",
        1.35,
        1.08,
        1.002,
        (1.5, 0.9, 1.3, 1.4, 1.0, 0.7, 1.4, 1.4, 1.6, 1.2, 1.5, 0.6, 1.2),
        (0.9, 1.0, 1.3, 1.0, 0.9),
        0,
        False,
    ),
    ModeSpec(
        "tw_rescue",
        1.10,
        1.05,
        1.000,
        (0.6, 0.9, 1.1, 0.8, 1.8, 0.4, 0.8, 0.8, 1.0, 0.4, 0.5, 0.8, 0.9),
        (0.8, 1.0, 1.2, 1.8, 2.2),
        1,
        False,
    ),
    ModeSpec(
        "pool_recombine",
        0.90,
        1.01,
        0.997,
        (0.7, 1.2, 0.9, 1.1, 0.8, 1.8, 1.6, 1.6, 1.1, 1.8, 1.2, 1.0, 1.1),
        (0.7, 1.1, 1.5, 0.9, 1.1),
        1,
        True,
    ),
    ModeSpec(
        "route_reduce",
        0.95,
        1.02,
        0.998,
        (0.6, 1.0, 0.9, 1.7, 0.6, 2.2, 2.4, 2.4, 1.2, 2.6, 2.2, 0.7, 0.8),
        (0.8, 1.2, 1.5, 1.0, 1.4),
        1,
        True,
    ),
    ModeSpec(
        "infeasible_descent",
        1.10,
        1.05,
        0.998,
        (1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0),
        (1.0, 1.0, 1.0, 1.0, 1.0),
        1,
        False,
    ),
)

MODE_DEFAULT, MODE_INTENSIFY, MODE_DIVERSIFY = 0, 1, 2
MODE_TW_RESCUE, MODE_POOL_RECOMBINE, MODE_ROUTE_REDUCE = 3, 4, 5
MODE_INFEASIBLE_DESCENT = 6

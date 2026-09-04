<div align="center">

# 🚛 vrptw-neural-hybrid-optimizer

**Neural Hybrid DDQN-ALNS Metaheuristic with GNN Edge-Heatmap Guidance for the Vehicle Routing Problem with Time Windows**

A research platform benchmarking **ALNS**, **Hybrid-Fixed**, **Hybrid-Rule**, **DDQN-ALNS**, and **GNN-Hybrid-DDQN** solvers  
on the Solomon & Gehring–Homberger VRPTW benchmarks — with a web-based dispatch portal for live demos.

[![Python](https://img.shields.io/badge/Python-3.12-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.4+-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.116+-009688?style=for-the-badge&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![NumPy](https://img.shields.io/badge/NumPy-013243?style=for-the-badge&logo=numpy&logoColor=white)](https://numpy.org)
[![Numba](https://img.shields.io/badge/Numba_JIT-00A3E0?style=for-the-badge&logo=numba&logoColor=white)](https://numba.pydata.org)
[![Vite](https://img.shields.io/badge/Vite-646CFF?style=for-the-badge&logo=vite&logoColor=white)](https://vitejs.dev)
[![Firebase](https://img.shields.io/badge/Firebase-FFCA28?style=for-the-badge&logo=firebase&logoColor=black)](https://firebase.google.com)
[![License: MIT](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)](LICENSE)

</div>

---

## Table of Contents

1.  [Highlights](#1-highlights)
2.  [Architecture](#2-architecture)
3.  [Project Structure](#3-project-structure)
4.  [Installation](#4-installation)
5.  [Quick Smoke Test](#5-quick-smoke-test)
6.  [Running the Full Benchmark](#6-running-the-full-benchmark)
7.  [Benchmark CLI](#7-benchmark-cli)
8.  [Transfer Learning & Domain Randomization](#8-transfer-learning--domain-randomization)
9.  [Web App (Dispatch Portal)](#9-web-app-dispatch-portal)
10. [Configuration Reference](#10-configuration-reference)
11. [Algorithm Overview](#11-algorithm-overview)
12. [Outputs & Artifacts](#12-outputs--artifacts)
13. [GPU Acceleration](#13-gpu-acceleration)
14. [Testing](#14-testing)
15. [Research Paper](#15-research-paper)
16. [Best-Known Solutions (BKS)](#16-best-known-solutions-bks)
17. [Contributing](#17-contributing)
18. [License](#18-license)

---

## 1. Highlights

- **12 solver variants** — from pure ALNS to GNN-guided neural hybrid solvers (`GNN-Hybrid-DDQN`) with multi-objective Pareto optimization, dynamic insertion, and C++ FFI bindings.
- **62 benchmark instances** — all 56 Solomon (C, R, RC × 100-customer) + 6 Gehring & Homberger 200-customer instances.
- **Statistically validated** — Wilcoxon signed-rank tests ($p < 0.05$) confirm neural hybrid solver superiority on head-to-head comparisons.
- **Numba JIT-compiled** cost and feasibility checks for maximum single-thread throughput.
- **Unified benchmark CLI** (`scripts/benchmark.py`) — one command to prepare, run, monitor, analyze, and clean benchmark sweeps.
- **Production web app** — FastAPI + Vite + Firebase Auth/Firestore dispatch portal with interactive route visualization.
- **Automatic checkpointing** — crash-safe; just re-run to resume.
- **Thread-contention mitigation** — all math/DL libraries pinned to 1 thread per worker for optimal parallel scaling.

---

## 2. Architecture

| Algorithm | Description |
|-----------|-------------|
| `ALNS-Base` | Pure ALNS with Thompson-bandit operator selection |
| `Hybrid-Fixed` | ALNS + rule-triggered route-reduction mode |
| `Hybrid-Rule` | ALNS + full heuristic mode-switching policy (6 modes) |
| `Hybrid-DDQN` | ALNS + online-trained DDQN plateau & operator controllers + Learned Acceptance Criterion |
| `Hybrid-DDQN-Transfer` | Hybrid-DDQN with weights pre-trained on RC1, tested on RC2 |
| `Hybrid-DDQN-Transfer-RC2` | Within-RC2 transfer (train on first 4, test on last 4) |
| `Hybrid-DDQN-Transfer-DR` | Domain-randomization pre-training (3-phase curriculum), then frozen inference |
| `GNN-Hybrid-DDQN` | **State-of-the-Art:** Hybrid-DDQN with Graph Attention Network (GAT) spatial-temporal embeddings, dynamic GNN edge-probability heatmap guidance, workload balance, and C++ LS hooks |
| `OR-Tools` | Google OR-Tools CP-SAT baseline |

### Key Components

| Component | Description |
|-----------|-------------|
| **GAT Policy Encoder** | Graph Attention Network utilizing multi-head self-attention to generate 64-dimensional spatial-temporal node embeddings |
| **GNN Edge Predictor** | Generates static/dynamic edge connectivity probabilities to guide construction, local search pruning, and route pool recombination |
| **Plateau Controller** | DDQN that selects search modes (`default`, `intensify`, `diversify`, `tw_rescue`, `pool_recombine`, `route_reduce`) when the search stagnates |
| **Operator Controller** | DDQN that selects destroy/repair operator pairs with prior-augmented exploration |
| **Learned Acceptance Criterion (LAC)** | Neural network replacing simulated-annealing acceptance for adaptive solution acceptance |
| **Prioritized Experience Replay (PER)** | TD-error prioritized sampling with β-annealing for stable off-policy learning |
| **Welford Reward Normalizer** | Online mean/variance normalization for stable RL training across diverse instance scales |
| **Route Pool + Set Partitioning** | Collect high-quality routes during search; recombine via MILP/greedy set-partitioning with GNN-discounted costs |
| **Elite Archive** | Top-k solution archive for warm-starting and diversification |
| **Thompson Bandit** | Bayesian bandit for operator selection in non-RL solvers |

---

## 3. Project Structure

```
vrptw-neural-hybrid-optimizer/
├── src/
│   ├── vrptw/                        # Research solver package
│   │   ├── __init__.py               # Public API (all exports)
│   │   ├── __main__.py               # Entry point: python3 -m vrptw
│   │   ├── config.py                 # Config dataclass, BKS table, algo labels
│   │   ├── core.py                   # Inst, Plan, Numba-JIT cost/feasibility
│   │   ├── generators.py             # SyntheticVRPTWGenerator, load_datasets
│   │   ├── heuristics.py             # Greedy construction, insertion utilities
│   │   ├── operators.py              # 8 destroy + 5 repair operators
│   │   ├── local_search.py           # 2-opt, relocate, swap, cross-exchange, route-compact
│   │   ├── pool.py                   # RoutePool, MILP/greedy set-partitioning
│   │   ├── rl.py                     # QNet, DDQN controllers, PER, LAC, EliteArchive
│   │   ├── solvers.py                # ALNSSolver → HybridDDQNSolver hierarchy
│   │   └── benchmark.py              # run_instance, run_benchmark, transfer training
│   ├── backend/                      # FastAPI application
│   │   ├── main.py                   # App factory, CORS, routes
│   │   ├── api/                      # REST endpoints (solve, solomon, health, config)
│   │   ├── core/                     # Auth, security, middleware
│   │   ├── models/                   # Pydantic schemas
│   │   ├── services/                 # Solver orchestration, Firebase integration
│   │   └── database/                 # Firestore persistence layer
│   └── frontend/                     # Browser UI
│       ├── index.html                # Single-page app
│       ├── css/                      # Stylesheets
│       └── js/                       # Client-side JS modules
├── scripts/
│   └── benchmark.py                  # Unified benchmark CLI (prepare/run/monitor/analyze/clean)
├── data/
│   ├── Solomon/                      # 56 Solomon .txt files (committed)
│   └── Gehring_Homberger/            # 200-customer instances
├── docs/
│   ├── paper.tex                     # IEEE-format research paper
│   ├── thesis.tex                    # Vietnamese thesis document
│   ├── fig3.tex                      # Standalone TikZ architecture diagram
│   ├── data/Solomon/                 # Backup Solomon data
│   ├── logs/                         # Benchmark CSVs, run logs
│   ├── model/                        # Saved safetensors weights
│   └── scripts/                      # GPU install, utility scripts
├── results/                          # Benchmark output directories
├── tests/
│   └── e2e/                          # Playwright E2E tests
├── main.py                           # Web app entry point
├── Makefile                          # Dev commands (dev, test, dist, emulators, test-e2e)
├── Dockerfile                        # Container build
├── pyproject.toml                    # Project metadata & dependencies
├── requirements.txt                  # Pip-compatible dependency list
├── firebase.json                     # Firebase hosting & emulator config
└── vite.config.js                    # Vite build config
```

---

## 4. Installation

### Prerequisites

- **Python ≥ 3.11, < 3.13** (3.12 recommended; 3.13+ not yet supported by Numba/PyTorch)
- **Node.js ≥ 18** (for the web app frontend)
- Recommended: [`uv`](https://docs.astral.sh/uv/) for fast Python installs

### Option A — uv (recommended, ~30 s)

```bash
git clone https://github.com/Thundercok/vrptw-neural-hybrid-optimizer.git
cd vrptw-neural-hybrid-optimizer

uv venv .venv --python 3.12
source .venv/bin/activate          # macOS/Linux
# .venv\Scripts\Activate.ps1       # Windows PowerShell

uv pip install -r requirements.txt
npm install                        # frontend dependencies
```

### Option B — plain pip

```bash
git clone https://github.com/Thundercok/vrptw-neural-hybrid-optimizer.git
cd vrptw-neural-hybrid-optimizer

python3.12 -m venv .venv
source .venv/bin/activate

pip install -r requirements.txt
npm install
```

> **Note:** The Solomon data files (`C101.txt` … `RC208.txt`) are **already committed** under `data/Solomon/` — no download needed.

---

## 5. Quick Smoke Test

Runs all 4 main solvers on a small synthetic 25-customer instance. Completes in **< 10 seconds**.

```bash
PYTHONPATH=src python3 -m vrptw
```

Expected output:

```
Launching VRPTW Application...
ALNS-Base                nv=  4 cost=   580.0 BKS TD N/A NV N/A (0.4s)
Hybrid-Fixed             nv=  3 cost=   648.0 BKS TD N/A NV N/A (1.5s)
Hybrid-Rule              nv=  3 cost=   648.0 BKS TD N/A NV N/A (0.7s)
Hybrid-DDQN              nv=  3 cost=   641.5 BKS TD N/A NV N/A (1.3s)
```

`BKS TD N/A` is expected — synthetic instances have no Best-Known Solution entry.

---

## 6. Running the Full Benchmark

The full benchmark runs all algorithms across **56 Solomon + 6 Homberger-200** instances (62 total) with multiple seeds. On a modern desktop it takes **~4–8 hours** for a 5-run pass of all algorithms.

### Scripted benchmark

Create a file, e.g. `run_benchmark.py`:

```python
import sys, os

sys.path.insert(0, os.path.abspath("src"))

from vrptw import (
    Config,
    load_datasets,
    run_benchmark,
    print_summary_table,
    ALGO_ALNS_BASE,
    ALGO_HYBRID_FIXED,
    ALGO_HYBRID_RULE,
    ALGO_HYBRID_DDQN,
    ALGO_ORTOOLS,
)

cfg = Config(
    data_path="./data/Solomon",
    output_dir="./results/my_run",
    n_runs=5,
    alns_iterations=5000,
    hybrid_iterations=5000,
    early_stop_patience=250,
    max_wall_hours=9.5,
)

datasets = load_datasets(cfg.data_path)
all_insts = datasets["c1"] + datasets["c2"] + datasets["r1"] + datasets["r2"] + datasets["rc1"] + datasets["rc2"]
algorithms = [ALGO_ALNS_BASE, ALGO_HYBRID_FIXED, ALGO_HYBRID_RULE, ALGO_HYBRID_DDQN, ALGO_ORTOOLS]

df = run_benchmark(
    instances=all_insts,
    algorithms=algorithms,
    cfg=cfg,
    result_path="./results/my_run/benchmark_clean.csv",
    checkpoint_path="./results/my_run/benchmark_checkpoint.csv",
)
print_summary_table(df)
```

### Checkpointing & resuming

The benchmark saves a checkpoint CSV every 4 instances. If it crashes or you stop it, **just re-run the same script** — completed `(instance, algorithm)` pairs are skipped automatically.

### Running a single instance

```python
from vrptw import Config, load_datasets, run_instance, ALGO_HYBRID_DDQN

cfg = Config(data_path="./data/Solomon")
datasets = load_datasets(cfg.data_path)
inst = datasets["rc1"][0]  # RC101

result, plan = run_instance(inst, ALGO_HYBRID_DDQN, cfg, seed=42)
print(f"NV={result['nv']}  cost={result['cost']:.1f}  gap={result['td_gap']:+.2f}%")
```

---

## 7. Benchmark CLI

The unified CLI at `scripts/benchmark.py` replaces all separate shell scripts with a single entry point:

| Command | Description |
|---------|-------------|
| `python3 scripts/benchmark.py prepare` | Aggregate Solomon + Homberger datasets into `data/combined_sweep` |
| `python3 scripts/benchmark.py run` | Run the full benchmark (all 4 shards) |
| `python3 scripts/benchmark.py run --shard 2` | Run a specific shard only |
| `python3 scripts/benchmark.py run --bg` | Run detached in background (macOS `caffeinate` auto-enabled) |
| `python3 scripts/benchmark.py run --runs 3` | Override the number of seeds per combo |
| `python3 scripts/benchmark.py run --no-checkpoint` | Start fresh, ignore existing checkpoints |
| `python3 scripts/benchmark.py monitor` | Live console dashboard with progress bars |
| `python3 scripts/benchmark.py status` | Print completion summary from checkpoints |
| `python3 scripts/benchmark.py analyze` | Aggregate results, compute NV/TD averages, run Wilcoxon tests |
| `python3 scripts/benchmark.py clean` | Delete checkpoints (prompt-guarded) |

### Shards

| Shard | Name | Instances |
|-------|------|-----------|
| 1 | Clustered (C1/C2) | 17 Solomon instances |
| 2 | Short-Horizon (R1/RC1) | 20 Solomon instances |
| 3 | Wide-Horizon (R2/RC2) | 19 Solomon instances |
| 4 | Homberger-200 | 6 × 200-customer instances |

---

## 8. Transfer Learning & Domain Randomization

Pre-train a DDQN policy, then apply it **frozen** to unseen instances.

### Train on RC1, test on RC2 (Cross-Distribution Transfer)

```python
from vrptw import (
    Config,
    load_datasets,
    train_transfer_model,
    load_transfer_model,
    run_benchmark,
    ALGO_HYBRID_DDQN_TRANSFER,
)

cfg = Config(data_path="./data/Solomon", output_dir="./results/transfer", transfer_epochs=1)
datasets = load_datasets(cfg.data_path)

# Train on RC1 → saves rl_alns_transfer_rc1_v15.safetensors
weights = train_transfer_model(datasets["rc1"], cfg, seed=42, label="RC1")

# Benchmark on RC2 with frozen weights
df = run_benchmark(
    instances=datasets["rc2"],
    algorithms=[ALGO_HYBRID_DDQN_TRANSFER],
    cfg=cfg,
    transfer_weights=weights,
)
```

### Within-RC2 Transfer

```python
from vrptw import train_transfer_model_within_rc2, ALGO_HYBRID_DDQN_TRANSFER_RC2

weights = train_transfer_model_within_rc2(datasets["rc2"], cfg, seed=42)
df = run_benchmark(
    instances=datasets["rc2"][cfg.rc2_transfer_split :],
    algorithms=[ALGO_HYBRID_DDQN_TRANSFER_RC2],
    cfg=cfg,
    transfer_weights=weights,
)
```

### Domain Randomization Pre-Training

```python
from vrptw import train_domain_randomization, ALGO_HYBRID_DDQN_TRANSFER_DR

# 3-phase curriculum on synthetic instances
weights = train_domain_randomization(cfg, seed=42)

# Test frozen on all Solomon instances
df = run_benchmark(
    instances=datasets["rc1"] + datasets["rc2"],
    algorithms=[ALGO_HYBRID_DDQN_TRANSFER_DR],
    cfg=cfg,
    transfer_weights=weights,
)
```

### Load Previously Saved Weights

```python
from vrptw import load_transfer_model

weights = load_transfer_model(cfg, label="rc1")
```

---

## 9. Web App (Dispatch Portal)

The web app provides an interactive UI for loading Solomon instances or custom CSV data, running solvers, and viewing routes on an interactive map.

### Start the server

```bash
# From the repo root
cp .env.example .env       # first time only
make dev
# or: python main.py
```

Then open **http://127.0.0.1:8000** in your browser.

### Development with hot-reload

```bash
make dev-all   # starts backend + Vite dev server with HMR
```

### Demo mode (no auth required)

By default `DEMO_AUTH_BYPASS=true` in `.env` — you can use the app immediately without Firebase credentials.

### API Endpoints

| Endpoint | Description |
|----------|-------------|
| `GET /api/health` | Firebase status, Torch device, model loaded state |
| `GET /api/config` | Public config the SPA reads on boot |
| `GET /api/solomon?name=rc101` | Load a Solomon instance by name |
| `POST /api/solve` | Submit a solve job (JSON body) |

### Enable Firebase Auth + Firestore

1. Create a Firebase project and download a service-account JSON.
2. Place it at `firebase-adminsdk.json` (already gitignored).
3. Set `DEMO_AUTH_BYPASS=false` and `FIREBASE_SERVICE_ACCOUNT_PATH=firebase-adminsdk.json` in `.env`.
4. Restart: `make dev`.

---

## 10. Configuration Reference

All options are fields of the `Config` dataclass in `src/vrptw/config.py`:

```python
from vrptw import Config

cfg = Config(
    # ── Data paths ────────────────────────────────────────────────────────
    data_path="./data/Solomon",
    output_dir="./results/my_run",
    # ── Search budget ────────────────────────────────────────────────────
    alns_iterations=5000,  # main ALNS iterations
    hybrid_iterations=5000,  # Hybrid solver iterations
    early_stop_patience=250,  # stop if no improvement for N iterations
    polish_iterations=80,  # post-search polish phase
    polish_patience=40,  # polish early-stop
    n_runs=5,  # runs per (instance, algo) combination
    max_wall_hours=9.5,  # hard wall-clock limit
    # ── Simulated annealing ─────────────────────────────────────────────
    temp_control=0.05,
    temp_decay=0.99975,
    # ── DDQN plateau controller ─────────────────────────────────────────
    ctrl_lr=3e-4,
    ctrl_tau=0.005,  # soft target update rate
    per_beta_steps=50_000,  # PER β annealing steps
    # ── DDQN operator controller ────────────────────────────────────────
    op_lr=3e-4,
    op_tau=0.005,
    # ── Learned Acceptance Criterion ────────────────────────────────────
    lac_enabled=True,
    # ── Transfer learning ───────────────────────────────────────────────
    transfer_epochs=1,
    rc2_transfer_split=4,
    # ── OR-Tools ────────────────────────────────────────────────────────
    ortools_time_limit=15.0,  # seconds
    # ── Route pool / set-partitioning ───────────────────────────────────
    route_pool_limit=600,
    sp_time_limit=4.0,
)
```

---

## 11. Algorithm Overview

All solvers share the same `solve(seed, init)` interface and return `(Plan, history)`.

### Solver Hierarchy

```
HybridDDQNSolver            (DDQN plateau + operator + LAC controllers)
    ├── HybridRuleSolver     (heuristic mode-switching, no RL)
    ├── HybridFixedSolver    (fixed mode-switching rules, no RL)
    └── ScheduledHybridSolver
ALNSSolver                   (pure ALNS, no hybrid modes)
```

### Operators

| Type | Operators |
|------|-----------|
| **Destroy (8)** | Random, Worst, Shaw, Route-Segment, TW-Urgent, Route-Eliminate, Proximity-Eliminate, Cross-Route-Shaw |
| **Repair (5)** | Greedy, Regret-2, Regret-3, TW-Greedy, FTS-Greedy |
| **Local Search** | 2-opt, Relocate, Swap, Cross-Exchange (granular), Route-Compact |

### Key Modules

| Module | Responsibility |
|--------|----------------|
| `core.py` | `Inst` (problem data), `Plan` (solution), Numba-JIT cost + feasibility |
| `operators.py` | 8 destroy + 5 repair operators with bias-weighted selection |
| `local_search.py` | 5 local search moves, iterative route elimination |
| `pool.py` | Route pool with MILP (scipy) or greedy set-partitioning recombination |
| `rl.py` | DDQN with PER, Thompson bandit, LAC, Welford normalizer, Elite Archive, UCB augmenter |
| `solvers.py` | Solver class hierarchy with 6-mode switching |
| `benchmark.py` | Parallel runner (`ProcessPoolExecutor` + spawn), checkpointing, transfer training |
| `generators.py` | Synthetic instance generator, Solomon/Homberger dataset loader |
| `config.py` | `Config` dataclass, BKS table (62 instances), algorithm labels |

---

## 12. Outputs & Artifacts

### Benchmark CSV Columns

| Column | Description |
|--------|-------------|
| `Dataset` | C1, C2, R1, R2, RC1, RC2, or GH200 |
| `Instance` | e.g. RC101, r1_2_1 |
| `Algorithm` | Canonical algorithm label |
| `NV_mean` | Mean number of vehicles across runs |
| `NV_std` | Std dev of NV |
| `NV_diff` | Mean NV minus BKS NV (negative = fewer vehicles) |
| `TD_mean` | Mean total distance |
| `TD_std` | Std dev of total distance |
| `Gap%` | `(TD_mean - BKS_TD) / BKS_TD × 100` |
| `OnTime` | On-time delivery rate (%) |
| `Time_s` | Mean wall time per run (seconds) |
| `NV_inflated` | Flag: NV > BKS_NV, making Gap% comparison misleading |

### Print the summary table from a saved CSV

```python
import pandas as pd
from vrptw import print_summary_table

df = pd.read_csv("./results/ultimate-publication-suite/benchmark_clean.csv")
print_summary_table(df)
```

---

## 13. GPU Acceleration

The DDQN policy uses PyTorch. The default `requirements.txt` installs CPU-only PyTorch to keep the install small. To use a GPU:

```bash
# Auto-detect CUDA version via nvidia-smi
python docs/scripts/install_torch_gpu.py

# Or force a specific CUDA version (cu118 / cu121 / cu124 / cu126)
python docs/scripts/install_torch_gpu.py --cuda 124
```

Restart afterwards. The startup log will confirm: `Torch device: GPU (NVIDIA ..., CUDA 12.x)`.

> **Note:** The DDQN Q-network is small — GPU speedup for the solver is modest. The bigger win is if you run many parallel benchmark workers.

---

## 14. Testing

### Unit tests

```bash
make test
# or: PYTHONPATH=src uv run pytest tests/ -v
```

### E2E tests (Playwright)

Requires Firebase Emulators:

```bash
make test-e2e
```

This will:
1. Build the frontend (`dist/`)
2. Start Firebase Emulators (Auth + Firestore + Hosting)
3. Seed a test user (`test@vrptw.local` / `testpass123`)
4. Start the backend
5. Run Playwright tests
6. Clean up all processes

---

## 15. Research Paper

The LaTeX source for the accompanying IEEE-format research paper is at `docs/paper.tex`.

### Key results (62 instances, 5 seeds, 310 combos)

- **Hybrid-DDQN** achieves the **lowest vehicle inflation** above BKS (+0.139) vs. ALNS-Base (+0.161) and OR-Tools (+1.911) on all 56 Solomon instances.
- On fair NV-filtered comparisons (N=47), Hybrid-DDQN achieves the **lowest distance gap** (+0.204%) vs. ALNS-Base (+0.220%) and Hybrid-Rule (+0.215%).
- On the 17 Clustered instances where OR-Tools matched BKS fleet size, Hybrid-DDQN's distance reduction (**−1.43%**) is **statistically significant** at α = 0.05 (Wilcoxon _p_ = 0.043).

### Build the paper

```bash
cd docs
pdflatex paper.tex && pdflatex paper.tex   # two passes for references
```

### Architecture diagram

The unified TikZ block diagram (Figure 3) shows the complete Hybrid DDQN-ALNS architecture:
- **Left column**: Training & Memory Loop (PER → DDQN update → Controllers)
- **Right column**: System Inference Pipeline (Mode Select → Operators → ALNS Core → Route Pool → Evaluation)
- Physical arrows show all data flows (minibatch B, TD error δ, valid routes, SP columns Ω, rewards)

---

## 16. Best-Known Solutions (BKS)

BKS values from the [SINTEF TOP benchmark](https://www.sintef.no/projectweb/top/vrptw/):

<details>
<summary><strong>Solomon 100-customer instances (56)</strong></summary>

| Instance | BKS NV | BKS TD | | Instance | BKS NV | BKS TD |
|----------|--------|--------|---|----------|--------|--------|
| C101 | 10 | 828.94 | | R101 | 19 | 1650.80 |
| C102 | 10 | 828.94 | | R102 | 17 | 1486.12 |
| C103 | 10 | 828.06 | | R103 | 13 | 1292.68 |
| C104 | 10 | 824.78 | | R104 |  9 | 1007.31 |
| C105 | 10 | 828.94 | | R105 | 14 | 1377.11 |
| C106 | 10 | 828.94 | | R106 | 12 | 1252.03 |
| C107 | 10 | 828.94 | | R107 | 10 | 1104.66 |
| C108 | 10 | 828.94 | | R108 |  9 |  960.88 |
| C109 | 10 | 828.94 | | R109 | 11 | 1194.73 |
| C201 |  3 | 591.56 | | R110 | 10 | 1118.84 |
| C202 |  3 | 591.56 | | R111 | 10 | 1096.72 |
| C203 |  3 | 591.17 | | R112 |  9 |  982.14 |
| C204 |  3 | 590.60 | | R201 |  4 | 1252.37 |
| C205 |  3 | 588.88 | | R202 |  3 | 1191.70 |
| C206 |  3 | 588.49 | | R203 |  3 |  939.50 |
| C207 |  3 | 588.29 | | R204 |  2 |  825.52 |
| C208 |  3 | 588.32 | | R205 |  3 |  994.43 |
| RC101 | 14 | 1696.94 | | R206 |  3 |  906.14 |
| RC102 | 12 | 1554.75 | | R207 |  2 |  890.61 |
| RC103 | 11 | 1261.67 | | R208 |  2 |  726.82 |
| RC104 | 10 | 1135.48 | | R209 |  3 |  909.16 |
| RC105 | 13 | 1629.44 | | R210 |  3 |  939.37 |
| RC106 | 11 | 1424.73 | | R211 |  2 |  885.71 |
| RC107 | 11 | 1230.48 | | RC201 |  4 | 1406.94 |
| RC108 | 10 | 1139.82 | | RC202 |  3 | 1365.65 |
| | | | | RC203 |  3 | 1049.62 |
| | | | | RC204 |  3 |  798.46 |
| | | | | RC205 |  4 | 1297.65 |
| | | | | RC206 |  3 | 1146.32 |
| | | | | RC207 |  3 | 1061.14 |
| | | | | RC208 |  3 |  828.14 |

</details>

<details>
<summary><strong>Gehring & Homberger 200-customer instances (6)</strong></summary>

| Instance | BKS NV | BKS TD |
|----------|--------|--------|
| c1_2_1 | 20 | 2704.57 |
| c2_2_1 |  6 | 1931.44 |
| r1_2_1 | 20 | 4784.11 |
| r2_2_1 |  4 | 4483.16 |
| rc1_2_1 | 18 | 3602.80 |
| rc2_2_1 |  6 | 3099.53 |

</details>

---

## 17. Contributing

1. Fork the repository.
2. Create a feature branch (`git checkout -b feat/my-feature`).
3. Add tests or a smoke script for your change.
4. Run the linter: `ruff check src/`
5. Open a Pull Request with a clear summary and test steps.

---

## 18. License

This project is licensed under the [MIT License](LICENSE).

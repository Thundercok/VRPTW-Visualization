# VRPTW Research Optimization Benchmark Suite

A self-contained, high-performance Python package for solving the **Vehicle Routing Problem with Time Windows (VRPTW)**. It integrates classical heuristics with Deep Reinforcement Learning (DRL) and multi-agent solvers.

---

## 🚀 Getting Started

### 📋 Prerequisites
Make sure dependencies are installed:
```bash
pip install -r docs/requirements.txt
```

### 🏃 Running Benchmarks
The primary benchmark runner is `run_benchmark.py`. It has a built-in CLI to customize runs without editing any code:

```bash
# Run a fast smoke test against one instance and one run
python3 docs/run_benchmark.py --instances RC101 --runs 1 --alns-iters 300 --hybrid-iters 300

# Run a full benchmark suite across all RC1 & RC2 instances
python3 docs/run_benchmark.py --runs 3 --alns-iters 1200 --hybrid-iters 1200
```

---

## 🛠️ Command-Line Interface (CLI)

Customize your benchmark runs using these options:

| Flag | Type | Default | Description |
| :--- | :---: | :---: | :--- |
| `--data-path` | `str` | `docs/data/Solomon` | Path to Solomon datasets folder |
| `--output-dir` | `str` | `docs/logs` | Directory to save logs/results |
| `--runs` | `int` | `3` | Number of independent runs per combination |
| `--alns-iters` | `int` | `1200` | ALNS base iteration limit |
| `--hybrid-iters` | `int` | `1200` | Hybrid ALNS/DDQN iteration limit |
| `--early-stop` | `int` | `250` | Early stop patience (iterations) |
| `--polish-iters` | `int` | `80` | Local search route-polishing iterations |
| `--max-hours` | `float` | `9.5` | Max wall-clock execution time (hours) before graceful shutdown |
| `--algorithms` | `list` | *All Four* | Choices: `ALNS-Base`, `Hybrid-Fixed`, `Hybrid-Rule`, `Hybrid-DDQN` |
| `--instances` | `list` | *All* | Run specific instances (e.g. `RC101 RC102 C101`) |

---

## 📊 Output & Checkpointing

- **`benchmark_clean.csv`**: Contains finalized results (mean + standard deviations) for all completed instances and algorithms.
- **`benchmark_checkpoint.csv`**: Implements a robust checkpointing mechanism. If a run is interrupted or reaches the `--max-hours` limit, re-running the command will **automatically resume** right where it left off.
- **Averages & Warnings**: The summary table automatically displays your gap against **Best Known Solutions (BKS)**. If your solution uses more vehicles (`NV`) than BKS, a warning is printed since minimizing vehicles is the primary objective in standard benchmarks.

---

## 📂 Project Architecture

```
docs/
├── run_benchmark.py          # CLI runner & coordinator
├── requirements.txt          # Package dependencies
├── data/
│   └── Solomon/              # 56 classic benchmark files (C, R, RC)
└── vrptw/                    # Modularized optimization source
    ├── __init__.py           # Package exports
    ├── __main__.py           # Quick CLI testing entrypoint
    ├── config.py             # Config schema, BKS values, defaults
    ├── core.py               # Instance, Plan, and low-level Numba checks
    ├── generators.py         # Synthetic instance generator & dataset loaders
    ├── heuristics.py         # ALNS operators & greedy builders
    ├── local_search.py       # 2-opt, relocate, and exchange routes
    ├── operators.py          # Shaw, random, regret, and noise operators
    ├── pool.py               # Route pools & exact MILP set partitioning solver
    ├── rl.py                 # DDQN agent, networks, and Safetensors IO
    └── solvers.py            # Unified solvers (ALNS, Hybrid-Fixed/Rule/DDQN)
```

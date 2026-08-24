"""
run_benchmark.py — Full Solomon RC1+RC2 benchmark runner.

Usage:
    cd docs
    python3 run_benchmark.py

Resumes automatically from benchmark_checkpoint.csv if interrupted.
Edit the Config block below to change algorithms, iteration counts, etc.
"""
import sys
import os


def _requested_sequential(argv: list[str]) -> bool:
    for i, a in enumerate(argv):
        if a == "--max-workers" and i + 1 < len(argv):
            return argv[i + 1] == "1"
        if a.startswith("--max-workers="):
            return a.split("=", 1)[1] == "1"
    return False


# Pin BLAS/Numba/torch threads to 1 per process for parallel runs. The defaults
# in vrptw.core assume 3 workers (NUMBA/OMP/MKL=4 on 12 cores), but this runner
# defaults to cpu_count-1 workers, which oversubscribed ~4x whenever torch or
# BLAS ran and added noise to every Time_s measurement. Must happen before
# `import vrptw` — core.py only uses setdefault, so an explicit value here wins,
# and spawned workers inherit it via os.environ.
if not _requested_sequential(sys.argv):
    for _v in ("NUMBA_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[_v] = "1"

# Ensure the vrptw package is importable when run from docs/
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

from vrptw import (
    Config,
    load_datasets,
    run_benchmark,
    print_summary_table,
    ALGO_ALNS_BASE,
    ALGO_ALNS_BASE_PLUS,
    ALGO_HYBRID_FIXED,
    ALGO_HYBRID_RULE,
    ALGO_HYBRID_DDQN,
    ALGO_ORTOOLS,
    ALGO_DQN,
)

# ── REQUIRED: all execution must be inside this guard so that spawn workers
# ── that re-import this script do NOT re-execute the benchmark call.
if __name__ == "__main__":
    import argparse

    base_dir = os.path.dirname(os.path.abspath(__file__))
    default_data = os.path.join(base_dir, "data", "Solomon")
    default_logs = os.path.join(base_dir, "logs")

    parser = argparse.ArgumentParser(
        description="Run VRPTW Solomon benchmark suite.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--data-path", type=str, default=default_data, help="Path to Solomon datasets")
    parser.add_argument("--output-dir", type=str, default=default_logs, help="Directory to save logs/results")
    parser.add_argument("--runs", type=int, default=3, help="Number of runs per algorithm/instance combo")
    parser.add_argument("--alns-iters", type=int, default=5000, help="ALNS iteration limit")
    parser.add_argument("--hybrid-iters", type=int, default=5000, help="Hybrid ALNS/DDQN iteration limit")
    parser.add_argument("--early-stop", type=int, default=250, help="Early stop patience")
    parser.add_argument("--polish-iters", type=int, default=80, help="Polish iterations")
    parser.add_argument("--max-hours", type=float, default=9.5, help="Max wall-clock execution time limit in hours")
    parser.add_argument("--gnn-path", type=str, default=None, help="Path to pre-trained GNN model weights")
    parser.add_argument(
        "--algorithms",
        nargs="+",
        choices=[
            ALGO_ALNS_BASE, ALGO_ALNS_BASE_PLUS, ALGO_HYBRID_FIXED, ALGO_HYBRID_RULE, ALGO_HYBRID_DDQN, ALGO_ORTOOLS, ALGO_DQN,
            "GNN-ALNS-Base", "GNN-Hybrid-Fixed", "GNN-Hybrid-Rule", "GNN-Hybrid-DDQN"
        ],
        default=[ALGO_ALNS_BASE, ALGO_HYBRID_FIXED, ALGO_HYBRID_RULE, ALGO_HYBRID_DDQN],
        help="Algorithms to include in benchmark"
    )
    parser.add_argument(
        "--instances",
        nargs="+",
        default=[],
        help="List of specific instance names to run (e.g. RC101 RC201). If empty, runs all available."
    )

    parser.add_argument(
        "--no-checkpoint",
        action="store_true",
        help="Ignore existing checkpoints and start a fresh run."
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=None,
        help="Maximum number of parallel workers (set to 1 for sequential)"
    )
    parser.add_argument(
        "--ortools-time-limit",
        type=float,
        default=15.0,
        help="Time limit for OR-Tools solver in seconds"
    )
    parser.add_argument(
        "--penalty-search",
        action="store_true",
        help="Enable penalty-based infeasible search"
    )
    budget = parser.add_mutually_exclusive_group()
    budget.add_argument(
        "--time-limit",
        type=float,
        default=None,
        help="Absolute anytime wall-clock budget per solve, in seconds (sets cfg.time_limit)"
    )
    budget.add_argument(
        "--no-time-limit",
        action="store_true",
        help="Disable the anytime budget entirely: pure iteration-bounded runs "
             "(sets cfg.time_limit_per_customer=0)"
    )

    args = parser.parse_args()

    gnn_path = args.gnn_path
    if gnn_path is None and any(a.startswith("GNN-") for a in args.algorithms):
        default_gnn = "docs/model/gnn_edge_predictor.pt"
        if os.path.exists(default_gnn):
            gnn_path = default_gnn
            print(f"Auto-configured GNN model path to: {default_gnn}")

    budget_kwargs = {}
    if args.no_time_limit:
        budget_kwargs["time_limit"] = None
        budget_kwargs["time_limit_per_customer"] = 0.0
    elif args.time_limit is not None:
        budget_kwargs["time_limit"] = args.time_limit

    cfg = Config(
        data_path=args.data_path,
        output_dir=args.output_dir,
        n_runs=args.runs,
        alns_iterations=args.alns_iters,
        hybrid_iterations=args.hybrid_iters,
        early_stop_patience=args.early_stop,
        polish_iterations=args.polish_iters,
        max_wall_hours=args.max_hours,
        gnn_model_path=gnn_path,
        ortools_time_limit=args.ortools_time_limit,
        penalty_search_enabled=args.penalty_search,
        **budget_kwargs,
    )

    # ── Load Solomon instances ─────────────────────────────────────────────
    print(f"Loading datasets from: {cfg.data_path}")
    datasets = load_datasets(cfg.data_path)
    all_insts = []
    counts_str = []
    for g, insts in datasets.items():
        all_insts.extend(insts)
        counts_str.append(f"{g.upper()}: {len(insts)}")
    
    # Filter by user-requested instances if provided
    if args.instances:
        req_lower = {inst.lower() for inst in args.instances}
        all_insts = [inst for inst in all_insts if inst.name.lower() in req_lower]
        print(f"Filtered to {len(all_insts)} requested instances: {[i.name for i in all_insts]}")
    else:
        print("  " + "  |  ".join(counts_str))

    if not all_insts:
        print("ERROR: No matching instances found. Check data-path and instance filters.")
        sys.exit(1)

    os.makedirs(cfg.output_dir, exist_ok=True)

    # ── Run ───────────────────────────────────────────────────────────────
    df = run_benchmark(
        instances=all_insts,
        algorithms=args.algorithms,
        cfg=cfg,
        result_path=os.path.join(cfg.output_dir, "benchmark_clean.csv"),
        checkpoint_path=os.path.join(cfg.output_dir, "benchmark_checkpoint.csv"),
        no_checkpoint=args.no_checkpoint,
        max_workers=args.max_workers,
    )

    print("\n\n═══ BENCHMARK SUMMARY ═══")
    print_summary_table(df)
    print(f"\nFull results saved to: {os.path.join(cfg.output_dir, 'benchmark_clean.csv')}")

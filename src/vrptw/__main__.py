from __future__ import annotations

import argparse
import os
import sys

from .benchmark import smoke_test
from .config import Config
from .generators import SyntheticVRPTWGenerator
from .solvers import ALNSSolver, HybridDDQNSolver, HybridFixedSolver, HybridRuleSolver


def cmd_smoke_test(args):
    print(f"Running synthetic smoke test (nodes={args.nodes}, distribution={args.dist})...")
    inst = SyntheticVRPTWGenerator(n_nodes=args.nodes, distribution=args.dist, seed=args.seed).generate()
    smoke_test(inst, seed=args.seed)


def cmd_solve(args):
    if not os.path.exists(args.file):
        print(f"Error: file not found: {args.file}")
        sys.exit(1)

    print(f"Loading instance from {args.file}...")
    from .core import load_solomon_instance

    inst = load_solomon_instance(args.file)

    print(
        f"Solving instance {inst.name} (capacity={inst.capacity}, customers={len(inst.demands) - 1}) with {args.algo}..."
    )

    use_gnn = args.algo.startswith("GNN-")
    target_algo = args.algo[4:] if use_gnn else args.algo

    cfg = Config(
        alns_iterations=args.iters,
        hybrid_iterations=args.iters,
        early_stop_patience=args.early_stop,
        polish_iterations=args.polish,
        penalty_search_enabled=args.penalty_search,
        adaptive_feasibility=args.adaptive_feasibility,
    )

    if use_gnn:
        gnn_path = args.gnn_path
        if gnn_path is None:
            default_gnn = os.path.join(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "docs", "model", "gnn_edge_predictor.pt"
            )
            if os.path.exists(default_gnn):
                gnn_path = default_gnn
        cfg.gnn_model_path = gnn_path

    if target_algo == "ALNS-Base":
        solver = ALNSSolver(inst, cfg)
    elif target_algo == "Hybrid-Fixed":
        solver = HybridFixedSolver(inst, cfg)
    elif target_algo == "Hybrid-Rule":
        solver = HybridRuleSolver(inst, cfg)
    elif target_algo == "Hybrid-DDQN":
        solver = HybridDDQNSolver(inst, cfg)
    else:
        print(f"Error: Unknown algorithm: {args.algo}")
        sys.exit(1)

    if use_gnn and hasattr(solver, "load_gnn_model") and cfg.gnn_model_path:
        print(f"Loading GNN model from {cfg.gnn_model_path}...")
        solver.load_gnn_model(cfg.gnn_model_path)

    import time

    t0 = time.time()
    plan, history = solver.solve(seed=args.seed)
    dur = time.time() - t0

    print("\n═══ SOLUTION FOUND ═══")
    print(f"Status:   {'Feasible' if plan.feasible else 'INFEASIBLE'}")
    print(f"Vehicles: {len(plan.routes)}")
    print(f"Distance: {plan.cost:.2f} km")
    print(f"Runtime:  {dur:.2f} seconds")
    print("\nRoutes:")
    for i, route in enumerate(plan.routes, 1):
        print(f"  Route #{i}: Depot -> {' -> '.join(map(str, route))} -> Depot")


def cmd_benchmark(args):
    from .benchmark import print_summary_table, run_benchmark
    from .config import Config, canonical_algo_label, default_data_path, default_output_dir
    from .generators import load_datasets

    cfg = Config(
        data_path=args.data_path or default_data_path(),
        output_dir=args.output_dir or default_output_dir(),
        n_runs=args.runs,
        alns_iterations=args.iters,
        hybrid_iterations=args.iters,
        early_stop_patience=args.early_stop,
        polish_iterations=args.polish,
        seed=args.seed,
        penalty_search_enabled=args.penalty_search,
        adaptive_feasibility=args.adaptive_feasibility,
    )

    gnn_path = args.gnn_path
    if gnn_path is None:
        default_gnn = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "docs", "model", "gnn_edge_predictor.pt"
        )
        if os.path.exists(default_gnn):
            gnn_path = default_gnn
    cfg.gnn_model_path = gnn_path

    print(f"Loading datasets from: {cfg.data_path}")
    datasets = load_datasets(cfg.data_path)

    # Combine all instances
    all_insts = []
    for group in ("c1", "c2", "r1", "r2", "rc1", "rc2"):
        if group in datasets:
            all_insts.extend(datasets[group])

    # Filter by requested instances if provided
    if args.instances:
        req_lower = {inst.lower() for inst in args.instances}
        all_insts = [inst for inst in all_insts if inst.name.lower() in req_lower]
        print(f"Filtered to {len(all_insts)} requested instances: {[i.name for i in all_insts]}")
    else:
        print(f"Loaded {len(all_insts)} instances total.")

    if not all_insts:
        print("Error: No matching instances found. Check data-path and instance filters.")
        sys.exit(1)

    # Handle algorithms
    algorithms = [canonical_algo_label(a) for a in args.algo]

    os.makedirs(cfg.output_dir, exist_ok=True)

    print(f"Running benchmarks for algorithms: {algorithms}")
    df = run_benchmark(
        instances=all_insts,
        algorithms=algorithms,
        cfg=cfg,
        result_path=os.path.join(cfg.output_dir, "benchmark_clean.csv"),
        checkpoint_path=os.path.join(cfg.output_dir, "benchmark_checkpoint.csv"),
        max_workers=getattr(args, "max_workers", None),
    )

    print("\n\n═══ BENCHMARK SUMMARY ═══")
    print_summary_table(df)
    print(f"\nFull results saved to: {os.path.join(cfg.output_dir, 'benchmark_clean.csv')}")


def main():
    parser = argparse.ArgumentParser(description="VRPTW Optimization Research Suite CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Smoke-test command
    p_smoke = subparsers.add_parser("smoke-test", help="Run a quick smoke test with synthetic data")
    p_smoke.add_argument("--nodes", type=int, default=25, help="Number of customer nodes")
    p_smoke.add_argument("--dist", type=str, default="RC", choices=["C", "R", "RC"], help="Geographical distribution")
    p_smoke.add_argument("--seed", type=int, default=42, help="Random seed")
    p_smoke.set_defaults(func=cmd_smoke_test)

    # Solve command
    p_solve = subparsers.add_parser("solve", help="Solve a specific Solomon dataset file")
    p_solve.add_argument("file", type=str, help="Path to Solomon .txt file")
    p_solve.add_argument(
        "--algo",
        type=str,
        default="Hybrid-DDQN",
        choices=[
            "ALNS-Base",
            "Hybrid-Fixed",
            "Hybrid-Rule",
            "Hybrid-DDQN",
            "GNN-ALNS-Base",
            "GNN-Hybrid-Fixed",
            "GNN-Hybrid-Rule",
            "GNN-Hybrid-DDQN",
        ],
        help="Optimization solver to run",
    )
    p_solve.add_argument("--iters", type=int, default=1200, help="Solver iteration limit")
    p_solve.add_argument("--early-stop", type=int, default=250, help="Early stop patience")
    p_solve.add_argument("--polish", type=int, default=80, help="Local search polishing iterations")
    p_solve.add_argument("--seed", type=int, default=42, help="Random seed")
    p_solve.add_argument("--gnn-path", type=str, default=None, help="Path to pre-trained GNN model weights")
    p_solve.add_argument("--penalty-search", action="store_true", help="Enable penalty-based infeasible search")
    p_solve.add_argument(
        "--no-adaptive-feasibility",
        action="store_false",
        dest="adaptive_feasibility",
        help="Disable adaptive feasibility management",
    )
    p_solve.set_defaults(func=cmd_solve, adaptive_feasibility=True)

    # Benchmark command
    p_bench = subparsers.add_parser("benchmark", help="Run benchmark suite on Solomon instances")
    p_bench.add_argument("--instances", nargs="+", default=[], help="List of specific instance names to run")
    p_bench.add_argument(
        "--algo",
        "--algorithms",
        nargs="+",
        dest="algo",
        default=["ALNS-Base", "Hybrid-Fixed", "Hybrid-Rule", "Hybrid-DDQN", "GNN-Hybrid-DDQN"],
        help="Algorithms to include in benchmark",
    )
    p_bench.add_argument("--runs", type=int, default=3, help="Number of runs per algorithm/instance combo")
    p_bench.add_argument("--iters", type=int, default=1200, help="Solver iteration limit")
    p_bench.add_argument("--early-stop", type=int, default=250, help="Early stop patience")
    p_bench.add_argument("--polish", type=int, default=80, help="Polish iterations")
    p_bench.add_argument("--seed", type=int, default=42, help="Random seed")
    p_bench.add_argument("--gnn-path", type=str, default=None, help="Path to pre-trained GNN model weights")
    p_bench.add_argument("--data-path", type=str, default=None, help="Path to Solomon datasets")
    p_bench.add_argument("--output-dir", type=str, default=None, help="Directory to save logs/results")
    p_bench.add_argument("--max-workers", type=int, default=None, help="Maximum number of parallel workers")
    p_bench.add_argument("--penalty-search", action="store_true", help="Enable penalty-based infeasible search")
    p_bench.add_argument(
        "--no-adaptive-feasibility",
        action="store_false",
        dest="adaptive_feasibility",
        help="Disable adaptive feasibility management",
    )
    p_bench.set_defaults(func=cmd_benchmark, adaptive_feasibility=True)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Standard Solver CLI Wrapper for Phase 2 Benchmarking.

Provides a unified contract for:
  --instance PATH
  --seed INT
  --time-limit FLOAT
  --ablation [alns|gns|gns_pool|gns_pool_lac|micro|full|full_no_macro|full_no_micro|full_no_lac|full_no_pool|full_no_gnn|full_no_gate]
  --output-json PATH
"""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from vrptw.config import Config
from vrptw.core import load_solomon_instance
from vrptw.solvers import ALNSSolver, HybridDDQNSolver, RuleMicroHybridSolver


def get_env_info() -> dict[str, Any]:
    import numpy as np
    import torch

    git_hash = None
    try:
        git_hash = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, text=True).strip()
    except Exception:
        pass

    return {
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "processor": platform.processor(),
        "torch_version": torch.__version__,
        "numpy_version": np.__version__,
        "cuda_available": torch.cuda.is_available(),
        "git_commit": git_hash,
    }


def get_solver_and_config(ablation: str, time_limit: float) -> tuple[type, Config]:
    cfg = Config(
        alns_iterations=10000,
        hybrid_iterations=10000,
        early_stop_patience=2500,
        polish_iterations=500,
        time_limit=float(time_limit),
    )

    # 1. Cumulative Ladder
    if ablation == "alns":
        cfg.lac_enabled = False
        cfg.route_pool_limit = 0
        cfg.recombine_after_main_search = False
        cfg.recombine_after_polish = False
        return ALNSSolver, cfg
    elif ablation == "gns":
        cfg.lac_enabled = False
        cfg.route_pool_limit = 0
        cfg.recombine_after_main_search = False
        cfg.recombine_after_polish = False
        return ALNSSolver, cfg
    elif ablation == "gns_pool":
        cfg.lac_enabled = False
        cfg.route_pool_limit = 480
        cfg.recombine_after_main_search = True
        cfg.recombine_after_polish = True
        return ALNSSolver, cfg
    elif ablation == "gns_pool_lac":
        cfg.lac_enabled = True
        cfg.route_pool_limit = 480
        cfg.recombine_after_main_search = True
        cfg.recombine_after_polish = True
        return ALNSSolver, cfg
    elif ablation == "micro":
        cfg.lac_enabled = True
        cfg.route_pool_limit = 480
        cfg.recombine_after_main_search = True
        cfg.recombine_after_polish = True
        cfg.macro_enabled = False
        return HybridDDQNSolver, cfg
    elif ablation == "full":
        cfg.lac_enabled = True
        cfg.route_pool_limit = 480
        cfg.recombine_after_main_search = True
        cfg.recombine_after_polish = True
        cfg.macro_enabled = True
        return HybridDDQNSolver, cfg

    # 2. Leave-One-Component-Out (LOCO) Ablations
    elif ablation == "full_no_macro":
        cfg.lac_enabled = True
        cfg.route_pool_limit = 480
        cfg.recombine_after_main_search = True
        cfg.recombine_after_polish = True
        cfg.macro_enabled = False
        return HybridDDQNSolver, cfg
    elif ablation == "full_no_micro":
        cfg.lac_enabled = True
        cfg.route_pool_limit = 480
        cfg.recombine_after_main_search = True
        cfg.recombine_after_polish = True
        cfg.macro_enabled = True
        cfg.use_op_rl = False
        return RuleMicroHybridSolver, cfg
    elif ablation == "full_no_lac":
        cfg.lac_enabled = False
        cfg.route_pool_limit = 480
        cfg.recombine_after_main_search = True
        cfg.recombine_after_polish = True
        cfg.macro_enabled = True
        return HybridDDQNSolver, cfg
    elif ablation == "full_no_pool":
        cfg.lac_enabled = True
        cfg.route_pool_limit = 0
        cfg.recombine_after_main_search = False
        cfg.recombine_after_polish = False
        cfg.macro_enabled = True
        return HybridDDQNSolver, cfg
    elif ablation in ("full_no_gns", "full_no_gnn"):
        cfg.lac_enabled = True
        cfg.route_pool_limit = 480
        cfg.recombine_after_main_search = True
        cfg.recombine_after_polish = True
        cfg.macro_enabled = True
        cfg.gnn_model_path = ""
        return HybridDDQNSolver, cfg
    elif ablation == "full_no_gate":
        cfg.lac_enabled = True
        cfg.route_pool_limit = 480
        cfg.recombine_after_main_search = True
        cfg.recombine_after_polish = True
        cfg.macro_enabled = True
        cfg.op_use_entropy_gate = False
        return HybridDDQNSolver, cfg
    elif ablation in ("full_no_gec", "no_gec"):
        cfg.lac_enabled = True
        cfg.route_pool_limit = 480
        cfg.recombine_after_main_search = True
        cfg.recombine_after_polish = True
        cfg.macro_enabled = True
        cfg.gec_max_depth = 1  # Single-level direct insertion fallback (disables recursive ejection chains)
        return HybridDDQNSolver, cfg
    else:
        raise ValueError(f"Unknown ablation configuration: {ablation}")


def extract_instrumentation(solver: Any, cfg: Config, ablation: str) -> dict[str, Any]:
    highs_calls = int(getattr(solver, "sp_stats", {}).get("calls", 0))
    highs_timeouts = int(getattr(solver, "sp_stats", {}).get("timeouts", 0))
    macro_decisions = sum(getattr(solver, "mode_trace", {}).values()) if hasattr(solver, "mode_trace") else 0
    micro_decisions = len(getattr(solver, "solver_history", [])) if hasattr(solver, "solver_history") else 0
    lac_queries = int(getattr(getattr(solver, "lac", None), "step", 0))
    lac_acceptances = int(getattr(getattr(solver, "lac", None), "accept_count", 0))

    resolved_components = {
        "macro_ddqn": bool(getattr(cfg, "macro_enabled", False)),
        "micro_ddqn": bool(getattr(cfg, "use_op_rl", True) and not getattr(solver, "use_op_rule", False)),
        "policy_gate": bool(getattr(cfg, "op_use_entropy_gate", True)),
        "lac": bool(getattr(cfg, "lac_enabled", False)),
        "gnn_edge_guidance": bool(getattr(cfg, "gnn_model_path", "") != ""),
        "gns": bool(getattr(cfg, "k_nearest", 25) > 0),
        "route_pool": bool(
            getattr(cfg, "route_pool_limit", 0) > 0 and getattr(cfg, "recombine_after_main_search", False)
        ),
        "gec": bool(getattr(cfg, "gec_max_depth", 3) >= 2),
    }

    return {
        "ablation": ablation,
        "resolved_components": resolved_components,
        "highs_calls": highs_calls,
        "highs_timeouts": highs_timeouts,
        "macro_decisions": macro_decisions,
        "micro_decisions": micro_decisions,
        "lac_queries": lac_queries,
        "lac_acceptances": lac_acceptances,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--instance", required=True, help="Path to instance file")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--time-limit", type=float, default=60.0)
    parser.add_argument(
        "--ablation",
        default="full",
        choices=[
            "alns",
            "gns",
            "gns_pool",
            "gns_pool_lac",
            "micro",
            "full",
            "full_no_macro",
            "full_no_micro",
            "full_no_lac",
            "full_no_pool",
            "full_no_gns",
            "full_no_gnn",
            "full_no_gate",
            "full_no_gec",
            "no_gec",
        ],
    )
    parser.add_argument("--output-json", required=True, help="Path to output JSON")
    args = parser.parse_args()

    inst_path = Path(args.instance)
    if not inst_path.exists():
        print(f"Error: instance {inst_path} does not exist", file=sys.stderr)
        sys.exit(1)

    inst = load_solomon_instance(str(inst_path))
    solver_cls, cfg = get_solver_and_config(args.ablation, args.time_limit)

    solver = solver_cls(inst, cfg)
    start_time = time.perf_counter()
    try:
        plan, history = solver.solve(seed=args.seed)
        elapsed = time.perf_counter() - start_time
        instrumentation = extract_instrumentation(solver, cfg, args.ablation)
        trajectory = getattr(solver, "trajectory_log", [])

        result = {
            "requested_ablation": args.ablation,
            "resolved_components": instrumentation.get("resolved_components", {}),
            "feasible": bool(plan.feasible),
            "nv": int(plan.nv),
            "td": float(plan.cost),
            "runtime_sec": float(elapsed),
            "iterations": len(history) if isinstance(history, list) else int(getattr(history, "iterations", 0)),
            "instrumentation": instrumentation,
            "trajectory": trajectory,
            "environment": get_env_info(),
            "metadata": {
                "instance": inst.name,
                "ablation": args.ablation,
                "seed": args.seed,
                "time_limit": args.time_limit,
            },
        }
    except Exception as e:
        elapsed = time.perf_counter() - start_time
        result = {
            "feasible": False,
            "nv": None,
            "td": None,
            "runtime_sec": float(elapsed),
            "error": str(e),
            "environment": get_env_info(),
            "metadata": {
                "instance": inst.name,
                "ablation": args.ablation,
                "seed": args.seed,
                "time_limit": args.time_limit,
            },
        }

    out_path = Path(args.output_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"[{args.ablation}] {inst.name} -> NV={result['nv']}, TD={result['td']} in {result['runtime_sec']:.2f}s")


if __name__ == "__main__":
    raise SystemExit(main())

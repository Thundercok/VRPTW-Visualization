#!/usr/bin/env python3
"""
Multi-Scale Ablation & Benchmark Suite (Option A)
Compares:
  - ALNS-Base (Unchanged Baseline)
  - Hybrid-Fixed (Fixed Local Search Schedule)
  - Hybrid-Rule (Handcrafted Rule Transitions)
  - Hybrid-DDQN (Proposed: Hierarchical MDP + Spatiotemporal GNS + HiGHS RoutePool Recombination)

Instances:
  - Solomon 100: C101, C201, R101, R201, RC101, RC201
  - Homberger 200: C1_2_1, C2_2_1, R1_2_1, R2_2_1, RC1_2_1, RC2_2_1
  - Homberger 400: C1_4_1, C2_4_1, R1_4_1, R2_4_1, RC1_4_1, RC2_4_1
"""

import os
import sys
import time

import pandas as pd

# Pin threading to avoid CPU oversubscription
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["NUMBA_NUM_THREADS"] = "1"
os.environ["TORCH_NUM_THREADS"] = "1"

# Add src to sys.path
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

from vrptw.benchmark import print_summary_table, run_benchmark
from vrptw.config import Config
from vrptw.generators import load_datasets

OUTPUT_DIR = os.path.join(ROOT, "results", "option_a_ablation")
os.makedirs(OUTPUT_DIR, exist_ok=True)

ALGORITHMS = ["ALNS-Base", "Hybrid-Fixed", "Hybrid-Rule", "Hybrid-DDQN"]
RUNS = 3

BENCHMARK_STAGES = [
    {
        "name": "Solomon-100",
        "data_path": os.path.join(ROOT, "data", "Solomon"),
        "instances": ["C101", "C201", "R101", "R201", "RC101", "RC201"],
        "iters": 1000,
        "early_stop": 500,
        "polish": 200,
    },
    {
        "name": "Homberger-200",
        "data_path": os.path.join(ROOT, "data", "Gehring_Homberger", "homberger_200_customer_instances"),
        "instances": ["C1_2_1", "C2_2_1", "R1_2_1", "R2_2_1", "RC1_2_1", "RC2_2_1"],
        "iters": 600,
        "early_stop": 250,
        "polish": 100,
    },
    {
        "name": "Homberger-400",
        "data_path": os.path.join(ROOT, "data", "Gehring_Homberger", "homberger_400_customer_instances"),
        "instances": ["C1_4_1", "C2_4_1", "R1_4_1", "R2_4_1", "RC1_4_1", "RC2_4_1"],
        "iters": 600,
        "early_stop": 250,
        "polish": 100,
    },
]


def main():
    print("==========================================================================")
    print(" STARTING MULTI-SCALE SIDE-BY-SIDE ABLATION (OPTION A)")
    print(f" Algorithms: {ALGORITHMS}")
    print(f" Runs per instance: {RUNS}")
    print(f" Output directory: {OUTPUT_DIR}")
    print("==========================================================================\n")

    all_stage_dfs = []
    t_start_total = time.time()

    for stage in BENCHMARK_STAGES:
        stage_name = stage["name"]
        print(f"\n>>>> [STAGE] Running {stage_name} ...")

        datasets = load_datasets(stage["data_path"])
        all_insts = []
        for group in ("c1", "c2", "r1", "r2", "rc1", "rc2"):
            if group in datasets:
                all_insts.extend(datasets[group])

        req_lower = {name.lower() for name in stage["instances"]}
        stage_insts = [inst for inst in all_insts if inst.name.lower() in req_lower]

        print(f"  Instances found ({len(stage_insts)}): {[i.name for i in stage_insts]}")
        if not stage_insts:
            print(f"  [ERROR] No instances found for stage {stage_name} at {stage['data_path']}")
            continue

        stage_out = os.path.join(OUTPUT_DIR, stage_name.lower().replace("-", "_"))
        os.makedirs(stage_out, exist_ok=True)

        cfg = Config(
            data_path=stage["data_path"],
            output_dir=stage_out,
            n_runs=RUNS,
            alns_iterations=stage["iters"],
            hybrid_iterations=stage["iters"],
            early_stop_patience=stage["early_stop"],
            polish_iterations=stage["polish"],
            time_limit_per_customer=0.0,
            seed=42,
        )

        stage_df = run_benchmark(
            instances=stage_insts,
            algorithms=ALGORITHMS,
            cfg=cfg,
            result_path=os.path.join(stage_out, "benchmark_clean.csv"),
            checkpoint_path=os.path.join(stage_out, "benchmark_checkpoint.csv"),
            max_workers=min(len(stage_insts), os.cpu_count() or 4),
        )

        all_stage_dfs.append(stage_df)
        print(f"\n--- Stage {stage_name} Summary ---")
        print_summary_table(stage_df)

    # Combine all results into one master clean file
    if all_stage_dfs:
        master_df = pd.concat(all_stage_dfs, ignore_index=True)
        master_path = os.path.join(OUTPUT_DIR, "master_ablation_clean.csv")
        master_df.to_csv(master_path, index=False)

        print("\n\n==========================================================================")
        print(" ALL STAGES COMPLETED SUCCESSFULLY!")
        print(f" Total Elapsed Time: {(time.time() - t_start_total) / 60.0:.2f} minutes")
        print(f" Master Results saved to: {master_path}")
        print("==========================================================================")
        print_summary_table(master_df)


if __name__ == "__main__":
    main()

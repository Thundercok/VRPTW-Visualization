#!/usr/bin/env python3
import os
import sys

# Limit thread counts for math/deep learning libraries to prevent CPU oversubscription/contention
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["NUMBA_NUM_THREADS"] = "1"

import argparse
import re
import shutil
import subprocess
import time

# Path Configuration
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_BASE = os.path.join(ROOT, "results", "ultimate-publication-suite")
LOG_DIR = os.path.join(OUTPUT_BASE, "runtime_logs")
LIVE_LOG = os.path.join(ROOT, "results", "live_terminal_stream.log")

# Generate all 60 instances per Homberger group
HOMBERGER_200_ALL = [
    f"{prefix}_2_{i}" for prefix in ["C1", "C2", "R1", "R2", "RC1", "RC2"] for i in range(1, 11)
]
HOMBERGER_400_ALL = [
    f"{prefix}_4_{i}" for prefix in ["C1", "C2", "R1", "R2", "RC1", "RC2"] for i in range(1, 11)
]

# Benchmark Configurations
ALGS = ["ALNS-Base", "Hybrid-Fixed", "Hybrid-Rule", "Hybrid-DDQN", "GNN-Hybrid-DDQN", "OR-Tools"]
SHARDS = {
    1: {
        "name": "Clustered (C1/C2)",
        "data_path": "data/Solomon",
        "output_dir": "solomon_clustered",
        "alns_iters": 5000,
        "hybrid_iters": 5000,
        "early_stop": 1000,
        "polish_iters": 300,
        "instances": ["C101", "C102", "C103", "C104", "C105", "C106", "C107", "C108", "C109",
                      "C201", "C202", "C203", "C204", "C205", "C206", "C207", "C208"]
    },
    2: {
        "name": "Short-Horizon (R1/RC1)",
        "data_path": "data/Solomon",
        "output_dir": "solomon_short_horizon",
        "alns_iters": 5000,
        "hybrid_iters": 5000,
        "early_stop": 1000,
        "polish_iters": 300,
        "instances": ["R101", "R102", "R103", "R104", "R105", "R106", "R107", "R108", "R109", "R110", "R111", "R112",
                      "RC101", "RC102", "RC103", "RC104", "RC105", "RC106", "RC107", "RC108"]
    },
    3: {
        "name": "Wide-Horizon (R2/RC2)",
        "data_path": "data/Solomon",
        "output_dir": "solomon_wide_horizon",
        "alns_iters": 5000,
        "hybrid_iters": 5000,
        "early_stop": 1000,
        "polish_iters": 300,
        "instances": ["R201", "R202", "R203", "R204", "R205", "R206", "R207", "R208", "R209", "R210", "R211",
                      "RC201", "RC202", "RC203", "RC204", "RC205", "RC206", "RC207", "RC208"]
    },
    4: {
        "name": "Homberger-200",
        "data_path": "data/Gehring_Homberger/homberger_200_customer_instances",
        "output_dir": "gehring_homberger_200",
        "alns_iters": 600,
        "hybrid_iters": 600,
        "early_stop": 150,
        "polish_iters": 50,
        "instances": HOMBERGER_200_ALL,
    },
    5: {
        "name": "Homberger-400",
        "data_path": "data/Gehring_Homberger/homberger_400_customer_instances",
        "output_dir": "gehring_homberger_400",
        "alns_iters": 600,
        "hybrid_iters": 600,
        "early_stop": 150,
        "polish_iters": 50,
        "instances": HOMBERGER_400_ALL,
    },
}
TOTAL_INSTANCES = sum(len(s["instances"]) for s in SHARDS.values())

# Colors for TUI
C_GREEN = "\033[32m"
C_RED = "\033[31m"
C_YELLOW = "\033[33m"
C_CYAN = "\033[36m"
C_RESET = "\033[0m"
C_BOLD = "\033[1m"

def print_header(title):
    print(f"\n{C_BOLD}{C_CYAN}=========================================================================={C_RESET}")
    print(f"  {C_BOLD}{title.upper()}{C_RESET}")
    print(f"{C_BOLD}{C_CYAN}=========================================================================={C_RESET}")

def run_prepare():
    """Aggregates Solomon and Homberger dataset files into combined_sweep."""
    print_header("Dataset Preparation")
    combined_dir = os.path.join(ROOT, "data", "combined_sweep")

    # Recreate target dir
    if os.path.exists(combined_dir):
        shutil.rmtree(combined_dir)
    os.makedirs(combined_dir)

    # 1. Copy Solomon
    solomon_dir = os.path.join(ROOT, "data", "Solomon")
    copied_solomon = 0
    if os.path.exists(solomon_dir):
        import glob
        for f in glob.glob(os.path.join(solomon_dir, "*.TXT")) + glob.glob(os.path.join(solomon_dir, "*.txt")):
            shutil.copy(f, combined_dir)
            copied_solomon += 1

    # 2. Copy Gehring & Homberger 200
    hg_dir = os.path.join(ROOT, "data", "Gehring_Homberger", "homberger_200_customer_instances")
    hg_instances = ["C1_2_1.TXT", "C2_2_1.TXT", "R1_2_1.TXT", "R2_2_1.TXT", "RC1_2_1.TXT", "RC2_2_1.TXT"]
    copied_hg = 0
    if os.path.exists(hg_dir):
        for name in hg_instances:
            # Handle case insensitivity
            for ext in (".TXT", ".txt"):
                src = os.path.join(hg_dir, name[:-4] + ext)
                if os.path.exists(src):
                    shutil.copy(src, combined_dir)
                    copied_hg += 1
                    break

    print(f"{C_GREEN}✓ Successfully aggregated sweep datasets:{C_RESET}")
    print(f"  - {copied_solomon} Solomon instances")
    print(f"  - {copied_hg} Gehring & Homberger 200-customer instances")
    print(f"  - Saved to: {combined_dir}")

def run_clean():
    """Cleans up checkpoint outputs."""
    print_header("Clean Output Directory")
    print(f"This will delete all checkpoint files under: {OUTPUT_BASE}")
    confirm = input("Are you absolutely sure you want to clean? (y/N): ").strip().lower()
    if confirm == "y":
        if os.path.exists(OUTPUT_BASE):
            shutil.rmtree(OUTPUT_BASE)
            print(f"{C_GREEN}✓ Cleaned outputs and checkpoints successfully.{C_RESET}")
        else:
            print("No output directory found. Nothing to clean.")
    else:
        print("Clean cancelled.")

class Tee:
    def __init__(self, original_stdout, file_path):
        self.stdout = original_stdout
        self.file = open(file_path, "a")

    def write(self, data):
        self.stdout.write(data)
        self.file.write(data)
        self.file.flush()

    def flush(self):
        self.stdout.flush()
        self.file.flush()

    def close(self):
        self.file.close()

def run_benchmark_shards(
    selected_shard=None,
    runs=5,
    no_checkpoint=False,
    instances=None,
    alns_iters=None,
    hybrid_iters=None,
    algorithms=None,
):
    """Executes the Python benchmark for specific shards."""
    os.makedirs(LOG_DIR, exist_ok=True)
    shards_to_run = [selected_shard] if selected_shard else sorted(SHARDS.keys())

    is_bg = os.environ.get("BENCHMARK_BG") == "1"
    tee = None
    if not is_bg:
        os.makedirs(os.path.dirname(LIVE_LOG), exist_ok=True)
        with open(LIVE_LOG, "w") as f:
            f.write("")
        tee = Tee(sys.stdout, LIVE_LOG)
        sys.stdout = tee

    # Start caffeinate to prevent sleep on macOS
    caff_proc = None
    if shutil.which("caffeinate"):
        try:
            caff_proc = subprocess.Popen(["caffeinate", "-dism"])
            print(f"{C_GREEN}✓ caffeinate started (system sleep disabled){C_RESET}")
        except Exception:
            pass

    try:
        for sid in shards_to_run:
            shard = SHARDS[sid]
            inst_list = instances if instances else shard["instances"]
            alg_list = algorithms if algorithms else ALGS
            a_iters = alns_iters if alns_iters is not None else shard["alns_iters"]
            h_iters = hybrid_iters if hybrid_iters is not None else shard["hybrid_iters"]

            print(f"\n--> {C_BOLD}Running Shard: {shard['name']}{C_RESET} ({len(inst_list)} instances in parallel)")

            shard_log_name = shard['name'].replace(" ", "_").replace("/", "_").replace("(", "_").replace(")", "_")
            log_file_path = os.path.join(LOG_DIR, f"shard_{shard_log_name}.log")

            cmd = [
                sys.executable, "-u", os.path.join(ROOT, "docs", "run_benchmark.py"),
                "--data-path", shard["data_path"],
                "--output-dir", os.path.join(OUTPUT_BASE, shard["output_dir"]),
                "--runs", str(runs),
                "--alns-iters", str(a_iters),
                "--hybrid-iters", str(h_iters),
                "--early-stop", str(shard["early_stop"]),
                "--polish-iters", str(shard["polish_iters"]),
            ]
            if no_checkpoint:
                cmd.append("--no-checkpoint")
            cmd += ["--algorithms"] + alg_list + ["--instances"] + inst_list

            # Run the command and capture logs
            env = os.environ.copy()
            env["PYTHONPATH"] = os.path.join(ROOT, "src")

            with open(log_file_path, "w") as log_file:
                # We pipe stdin from devnull, stdout/stderr to log file and console
                process = subprocess.Popen(
                    cmd,
                    env=env,
                    stdin=subprocess.DEVNULL,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True
                )

                # Stream outputs live to console (which goes to live_terminal_stream.log if redirected) and to shard log
                for line in process.stdout:
                    sys.stdout.write(line)
                    sys.stdout.flush()
                    log_file.write(line)

                process.wait()
                if process.returncode == 0:
                    print(f"--> {C_GREEN}[SUCCESS] Shard {shard['name']} completed.{C_RESET}")
                else:
                    print(f"--> {C_RED}[!!! CRITICAL ERROR !!!] Shard {shard['name']} failed (exit code {process.returncode}).{C_RESET}", file=sys.stderr)

    finally:
        if caff_proc:
            caff_proc.terminate()
            caff_proc.wait()
            print(f"{C_GREEN}✓ caffeinate stopped (system sleep re-enabled){C_RESET}")
        if tee:
            sys.stdout = tee.stdout
            tee.close()

def run_bg_benchmark(selected_shard=None, runs=5, no_checkpoint=False):
    """Launches the benchmark suite in the background."""
    print_header("Background Benchmark Execution")
    os.makedirs(os.path.dirname(LIVE_LOG), exist_ok=True)

    # Formulate cmd
    script_path = os.path.abspath(__file__)
    cmd = [sys.executable, script_path, "run"]
    if selected_shard:
        cmd += ["--shard", str(selected_shard)]
    cmd += ["--runs", str(runs)]
    if no_checkpoint:
        cmd += ["--no-checkpoint"]

    # Spawn background process detached
    log_fh = open(LIVE_LOG, "w")
    env = os.environ.copy()
    env["BENCHMARK_BG"] = "1"
    process = subprocess.Popen(
        cmd,
        env=env,
        stdin=subprocess.DEVNULL,
        stdout=log_fh,
        stderr=subprocess.STDOUT,
        close_fds=True,
        start_new_session=True # Detach cleanly
    )

    print(f"{C_GREEN}✓ Benchmark successfully detached and running in the background.{C_RESET}")
    print(f"  - PID: {process.pid}")
    print(f"  - Logs redirected to: {LIVE_LOG}")
    print("\nTo monitor live results, run:")
    print(f"  {C_BOLD}python3 scripts/benchmark.py monitor{C_RESET}")

def run_monitor():
    """Monitors the active run in a clean, redrawing console UI."""
    if not os.path.exists(LIVE_LOG):
        print(f"Waiting for log file '{LIVE_LOG}' to be created...")
        while not os.path.exists(LIVE_LOG):
            time.sleep(1)

    print("\033[2J\033[H", end="") # Clear terminal and home cursor

    last_modified = 0
    while True:
        try:
            mtime = os.path.getmtime(LIVE_LOG)
            if mtime == last_modified:
                time.sleep(2)
                continue
            last_modified = mtime

            with open(LIVE_LOG) as f:
                content = f.read()

            # Parse instances
            processing = re.findall(r"\[PROCESSING\]\s+(\w+)...", content)
            successes = re.findall(r"\[SUCCESS\]\s+(\w+)\s+finished in\s+(\d+)s", content)
            failures = re.findall(r"\[!!! CRITICAL ERROR !!!\]\s+(\w+)\s+failed", content)

            succ_set = {s[0] for s in successes}
            fail_set = set(failures)

            # Find currently processing (in processing but not in success or failure)
            active = [p for p in processing if p not in succ_set and p not in fail_set]

            completed = len(succ_set) + len(fail_set)
            pct = (completed / TOTAL_INSTANCES) * 100

            # Render clean UI
            print("\033[2J\033[H", end="") # Clear terminal and home cursor
            print("==========================================================================")
            print(f"                  {C_BOLD}VRPTW SOLVER BENCHMARK MONITOR{C_RESET}                          ")
            print("==========================================================================")
            print(f"Progress: [{completed}/{TOTAL_INSTANCES}]  {pct:.1f}%")

            # Draw progress bar
            bar_len = 40
            filled = int(bar_len * completed / TOTAL_INSTANCES)
            bar = "█" * filled + "-" * (bar_len - filled)
            print(f"[{bar}]")
            print("--------------------------------------------------------------------------")
            print(f"Successes: {len(succ_set)} | Failures: {len(fail_set)}")
            if active:
                print(f"Currently Running: {C_YELLOW}{', '.join(active)}{C_RESET}")
            else:
                print("Currently Running: None (Waiting or Finished)")
            print("--------------------------------------------------------------------------")

            # Print last 5 status changes
            print("Recent activity:")
            lines = content.splitlines()
            activity = []
            for line in reversed(lines):
                if "[PROCESSING]" in line or "[SUCCESS]" in line or "[!!! CRITICAL ERROR !!!]" in line:
                    activity.append(line.strip())
                    if len(activity) == 6:
                        break

            # Pad to 6 lines to maintain clean layout
            while len(activity) < 6:
                activity.append("")

            for act in reversed(activity):
                if not act:
                    print("")
                elif "SUCCESS" in act:
                    print(f"  {C_GREEN}{act}{C_RESET}")
                elif "ERROR" in act:
                    print(f"  {C_RED}{act}{C_RESET}")
                else:
                    print(f"  {act}")
            print("==========================================================================")
            print("Press Ctrl+C to stop monitoring. The benchmark will keep running in bg.   ")

        except KeyboardInterrupt:
            print("\nMonitoring stopped.")
            break
        except Exception:
            time.sleep(2)

def run_status():
    """Prints summary of completed instances in the output directory."""
    print_header("Benchmark Progress Status")

    completed_combos = 0
    total_expected = TOTAL_INSTANCES * len(ALGS)
    shards_details = []

    # Keep track of fully completed instances (all 5 algorithms completed)
    completed_instances = set()

    for _sid, shard in SHARDS.items():
        ckpt_path = os.path.join(OUTPUT_BASE, shard["output_dir"], "benchmark_checkpoint.csv")
        s_completed = 0
        s_total = len(shard["instances"]) * len(ALGS)

        if os.path.exists(ckpt_path):
            try:
                import pandas as pd
                df = pd.read_csv(ckpt_path)
                s_completed = len(df)

                # Check which instances have all 5 algorithms done
                counts = df.groupby("Instance")["Algorithm"].nunique()
                for inst, count in counts.items():
                    if count == len(ALGS):
                        completed_instances.add(inst)
            except Exception:
                pass
        completed_combos += s_completed
        shards_details.append((shard["name"], s_completed, s_total))

    print(f"Overall Progress: {C_BOLD}{completed_combos}/{total_expected}{C_RESET} combos complete ({completed_combos / max(1, total_expected) * 100:.1f}%)")

    # Progress Bar
    bar_len = 50
    filled = int(bar_len * completed_combos / max(1, total_expected))
    bar = "█" * filled + "-" * (bar_len - filled)
    print(f"[{C_GREEN}{bar}{C_RESET}]")
    print("\nShard Breakdown:")
    for name, comp, tot in shards_details:
        color = C_GREEN if comp == tot else (C_YELLOW if comp > 0 else C_RESET)
        print(f"  - {name:<30}: {color}{comp}/{tot}{C_RESET} ({comp/max(1, tot)*100:.1f}%)")

    # Check failures, filtering out resolved ones
    failed_file = os.path.join(OUTPUT_BASE, "failed_instances.txt")
    if os.path.exists(failed_file):
        active_failures = []
        completed_instances_lower = {ci.lower() for ci in completed_instances}
        with open(failed_file) as f:
            for line in f:
                parts = line.strip().split(",")
                if len(parts) >= 2:
                    inst = parts[1].strip()
                    if inst.lower() not in completed_instances_lower:
                        active_failures.append(line.strip())

        if active_failures:
            print(f"\n{C_RED}⚠️ Failures Recorded (Not yet resolved):{C_RESET}")
            for failure in active_failures:
                print(f"  {failure}")
        else:
            print(f"\n{C_GREEN}✓ No unresolved failures.{C_RESET}")
    else:
        print(f"\n{C_GREEN}✓ No failures recorded.{C_RESET}")

def run_analyze():
    """Combines clean outputs and runs Wilcoxon signed-rank tests."""
    print_header("Benchmark Results Analysis")

    import numpy as np
    import pandas as pd
    from scipy.stats import wilcoxon

    # Find clean outputs
    all_dfs = []
    for _sid, shard in SHARDS.items():
        clean_path = os.path.join(OUTPUT_BASE, shard["output_dir"], "benchmark_clean.csv")
        if not os.path.exists(clean_path):
            # Try checkpoint fallback
            clean_path = os.path.join(OUTPUT_BASE, shard["output_dir"], "benchmark_checkpoint.csv")

        if os.path.exists(clean_path):
            try:
                all_dfs.append(pd.read_csv(clean_path))
            except Exception as e:
                print(f"Warning: Failed to load results from {clean_path} ({e})")

    if not all_dfs:
        print(f"{C_RED}Error: No result files found under {OUTPUT_BASE}. Please run benchmarks first.{C_RESET}")
        sys.exit(1)

    df = pd.concat(all_dfs, ignore_index=True)
    df["Algorithm"] = df["Algorithm"].str.strip()

    # Save combined_clean.csv
    combined_path = os.path.join(OUTPUT_BASE, "combined_clean.csv")
    df.to_csv(combined_path, index=False)
    print(f"Combined clean dataset saved to {combined_path} ({len(df)} rows across {df['Instance'].nunique()} instances).")

    nv_df = df.pivot(index="Instance", columns="Algorithm", values="NV_mean")
    td_df = df.pivot(index="Instance", columns="Algorithm", values="TD_mean")
    dataset_map = df.drop_duplicates("Instance").set_index("Instance")["Dataset"]

    # 1. Summary Stats Table
    print(f"\n{C_BOLD}=== VEHICLE COUNT (NV) / DISTANCE (TD) AVERAGES ==={C_RESET}")
    groups = df["Dataset"].unique()

    print(f"{'DS':<25}{'Algorithm':<20}{'NV Mean':>10}{'TD Mean':>12}{'Time (s)':>10}")
    print("-" * 80)
    for g in sorted(groups):
        insts_g = dataset_map[dataset_map == g].index
        for algo in ["OR-Tools", "ALNS-Base", "Hybrid-Fixed", "Hybrid-Rule", "Hybrid-DDQN", "GNN-Hybrid-DDQN"]:
            if algo in nv_df.columns and algo in td_df.columns:
                valid_g = nv_df.loc[insts_g, algo].dropna().index
                if len(valid_g) > 0:
                    avg_nv = nv_df.loc[valid_g, algo].mean()
                    avg_td = td_df.loc[valid_g, algo].mean()
                    avg_time = df[(df["Instance"].isin(valid_g)) & (df["Algorithm"] == algo)]["Time_s"].mean()
                    print(f"{g:<25}{algo:<20}{avg_nv:>10.2f}{avg_td:>12.2f}{avg_time:>10.1f}s")
        print("-" * 80)

    # 2. Wilcoxon signed-rank tests
    print(f"\n{C_BOLD}=== PAIRWISE WILCOXON SIGNED-RANK TESTS ==={C_RESET}")

    def run_test(x_col, y_col, name_x, name_y, metric, sub_df):
        try:
            pair = sub_df[[name_x, name_y]].dropna()
            if len(pair) < 5:
                return
            x = pair[name_x]
            y = pair[name_y]
            diff = x - y
            if np.all(diff == 0):
                print(f"  {metric:<3} (N={len(pair):>3}) {name_x:<15} vs {name_y:<15}: Identical solutions (p-value N/A)")
                return
            stat, p = wilcoxon(x, y)
            sig_color = C_GREEN if p < 0.05 else C_RESET
            print(f"  {metric:<3} (N={len(pair):>3}) {name_x:<15} vs {name_y:<15}: {sig_color}p-value = {p:.4e}{C_RESET} (stat={stat:.1f})")
        except Exception as exc:
            print(f"  {metric:<3} {name_x} vs {name_y}: Error ({exc})")

    if "Hybrid-DDQN" in nv_df.columns:
        for comparison in ["ALNS-Base", "Hybrid-Fixed", "Hybrid-Rule"]:
            if comparison in nv_df.columns:
                run_test(nv_df, nv_df, "Hybrid-DDQN", comparison, "NV", nv_df)
                run_test(td_df, td_df, "Hybrid-DDQN", comparison, "TD", td_df)
                print()

def main():
    parser = argparse.ArgumentParser(
        description="Unified VRPTW Solver Benchmark Manager CLI",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    subparsers = parser.add_subparsers(dest="command", help="Sub-commands")

    # 1. prepare
    subparsers.add_parser("prepare", help="Prepare & aggregate datasets into data/combined_sweep")

    # 2. run
    run_parser = subparsers.add_parser("run", help="Execute benchmark sweeps")
    run_parser.add_argument("--shard", type=int, choices=[1, 2, 3, 4, 5], default=None, help="Specific Shard ID to run (default: run all)")
    run_parser.add_argument("--runs", type=int, default=5, help="Number of runs per algo/instance combination")
    run_parser.add_argument("--bg", action="store_true", help="Launch execution in the background detached")
    run_parser.add_argument("--no-checkpoint", action="store_true", help="Ignore checkpoints and start fresh")
    run_parser.add_argument("--instances", nargs="+", default=None, help="Optional specific instance names to override shard default")
    run_parser.add_argument("--algorithms", nargs="+", default=None, help="Optional algorithms to include")
    run_parser.add_argument("--alns-iters", type=int, default=None, help="Override ALNS iterations")
    run_parser.add_argument("--hybrid-iters", type=int, default=None, help="Override Hybrid iterations")

    # 3. monitor
    subparsers.add_parser("monitor", help="Monitor real-time progress of background benchmark runs")

    # 4. status
    subparsers.add_parser("status", help="Print progress status breakdown")

    # 5. analyze
    subparsers.add_parser("analyze", help="Combine outputs, show averages, and run Wilcoxon signed-rank tests")

    # 6. clean
    subparsers.add_parser("clean", help="Clean up output/checkpoint folders")

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(0)

    if args.command == "prepare":
        run_prepare()
    elif args.command == "clean":
        run_clean()
    elif args.command == "run":
        if args.bg:
            run_bg_benchmark(selected_shard=args.shard, runs=args.runs, no_checkpoint=args.no_checkpoint)
        else:
            run_benchmark_shards(
                selected_shard=args.shard,
                runs=args.runs,
                no_checkpoint=args.no_checkpoint,
                instances=args.instances,
                alns_iters=args.alns_iters,
                hybrid_iters=args.hybrid_iters,
                algorithms=args.algorithms,
            )
    elif args.command == "monitor":
        run_monitor()
    elif args.command == "status":
        run_status()
    elif args.command == "analyze":
        run_analyze()

if __name__ == "__main__":
    main()

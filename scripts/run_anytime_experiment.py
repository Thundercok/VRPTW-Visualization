#!/usr/bin/env python3
"""Anytime Wall-Clock Benchmark Runner (Continuous 300s Trajectory Sampling).

Executes a single continuous 300s run per (instance, solver, seed) combination,
records full search trajectories, samples checkpoints at:
    t in {1s, 5s, 10s, 30s, 60s, 120s, 300s},
and outputs:
    1. results/phase2_anytime_300s/anytime_raw.csv
    2. results/phase2_anytime_300s/anytime_summary.csv
    3. docs/figures/anytime_convergence.png
"""

from __future__ import annotations

import argparse
import csv
import json
import shlex
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]

INSTANCES = [
    ("RC101", "data/Solomon/rc101.txt"),
    ("R101", "data/Solomon/r101.txt"),
    ("C101", "data/Solomon/c101.txt"),
]

SOLVERS = [
    ("ALNS-Base", "--ablation alns"),
    ("Hybrid-DDQN", "--ablation full"),
]

CHECKPOINTS = [1.0, 5.0, 10.0, 30.0, 60.0, 120.0, 300.0]


def run_single_job(job: tuple[str, str, str, str, int, float, Path]) -> dict[str, Any]:
    inst_name, inst_path, solver_name, solver_args, seed, time_limit, out_dir = job
    token = f"{solver_name}__{inst_name}__seed{seed}"
    json_path = out_dir / f"{token}.json"
    stdout_path = out_dir / f"{token}.stdout"
    stderr_path = out_dir / f"{token}.stderr"

    cmd = [
        "uv",
        "run",
        "python",
        str(ROOT / "scripts" / "benchmark.py"),
        "--instance",
        str(ROOT / inst_path),
        "--seed",
        str(seed),
        "--time-limit",
        str(time_limit),
        "--output-json",
        str(json_path),
    ] + shlex.split(solver_args)

    t0 = time.perf_counter()
    proc = subprocess.run(
        cmd,
        stdout=stdout_path.open("w", encoding="utf-8"),
        stderr=stderr_path.open("w", encoding="utf-8"),
        timeout=time_limit + 60.0,
        check=False,
    )
    wall_elapsed = time.perf_counter() - t0

    payload = {}
    if json_path.exists():
        try:
            payload = json.loads(json_path.read_text(encoding="utf-8"))
        except Exception as e:
            payload = {"error": str(e)}

    print(
        f"  [{solver_name:12s}] {inst_name} (seed {seed}) -> NV={payload.get('nv')}, TD={payload.get('td')} ({wall_elapsed:.1f}s, code {proc.returncode})"
    )
    return {
        "inst_name": inst_name,
        "solver_name": solver_name,
        "seed": seed,
        "payload": payload,
        "json_path": json_path,
        "wall_elapsed": wall_elapsed,
    }


def sample_trajectory_at_checkpoints(
    trajectory: list[dict[str, Any]],
    checkpoints: list[float],
) -> dict[float, tuple[int, float]]:
    """Determines best-so-far (NV, TD) at or before each cutoff time t."""
    if not trajectory:
        return {t: (999, 99999.0) for t in checkpoints}

    # Sort trajectory by t_sec
    sorted_traj = sorted(trajectory, key=lambda x: float(x.get("t_sec", 0.0)))

    results = {}
    for cp in checkpoints:
        best_nv = 999
        best_td = float("inf")

        for entry in sorted_traj:
            t = float(entry.get("t_sec", 0.0))
            if t <= cp:
                nv = int(entry.get("nv", 999))
                td = float(entry.get("td", float("inf")))
                if (nv < best_nv) or (nv == best_nv and td < best_td):
                    best_nv = nv
                    best_td = td
            else:
                break

        results[cp] = (best_nv, best_td)
    return results


def plot_anytime_convergence(df_samples: list[dict[str, Any]], out_png: Path):
    try:
        import matplotlib.pyplot as plt
        import pandas as pd

        df = pd.DataFrame(df_samples)
        fig, axes = plt.subplots(1, 3, figsize=(16, 5), dpi=300)

        inst_order = ["C101", "R101", "RC101"]
        colors = {"ALNS-Base": "#1f77b4", "Hybrid-DDQN": "#d62728"}
        markers = {"ALNS-Base": "o", "Hybrid-DDQN": "s"}

        for idx, inst in enumerate(inst_order):
            ax = axes[idx]
            sub = df[df["instance"] == inst]

            for solver in ["ALNS-Base", "Hybrid-DDQN"]:
                solver_sub = sub[sub["solver"] == solver]
                grp = solver_sub.groupby("cutoff_sec")[["td", "nv"]].mean().reset_index()

                ax.plot(
                    grp["cutoff_sec"],
                    grp["td"],
                    label=solver,
                    color=colors[solver],
                    marker=markers[solver],
                    linewidth=2.2,
                    markersize=6,
                )

            ax.set_xscale("log")
            ax.set_title(f"Instance {inst}", fontsize=13, fontweight="bold")
            ax.set_xlabel("Wall-Clock Budget (s, log-scale)", fontsize=11)
            ax.set_ylabel("Mean Travel Distance (km)", fontsize=11)
            ax.grid(True, linestyle="--", alpha=0.6)
            ax.legend(frameon=True, fontsize=10)

        plt.tight_layout()
        out_png.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(out_png, dpi=300)
        plt.close()
        print(f"✓ Saved anytime convergence figure to {out_png}")
    except Exception as e:
        print(f"Warning: Could not plot anytime convergence: {e}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44, 45, 46])
    parser.add_argument("--time-limit", type=float, default=300.0)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--output", default="results/phase2_anytime_300s")
    args = parser.parse_args()

    out_dir = ROOT / args.output
    out_dir.mkdir(parents=True, exist_ok=True)

    jobs = []
    for inst_name, inst_path in INSTANCES:
        for solver_name, solver_args in SOLVERS:
            for seed in args.seeds:
                jobs.append((inst_name, inst_path, solver_name, solver_args, seed, args.time_limit, out_dir))

    print(f"Launching {len(jobs)} continuous 300s runs across {args.workers} workers...")
    completed_runs = []

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = [executor.submit(run_single_job, job) for job in jobs]
        for fut in as_completed(futures):
            res = fut.result()
            completed_runs.append(res)

    print(f"\nAll {len(completed_runs)} runs finished. Sampling checkpoints at {CHECKPOINTS}s...")

    sample_rows = []
    for run in completed_runs:
        inst = run["inst_name"]
        solver = run["solver_name"]
        seed = run["seed"]
        payload = run["payload"]
        trajectory = payload.get("trajectory", [])

        cp_samples = sample_trajectory_at_checkpoints(trajectory, CHECKPOINTS)
        for cp, (nv, td) in cp_samples.items():
            sample_rows.append(
                {
                    "instance": inst,
                    "solver": solver,
                    "seed": seed,
                    "cutoff_sec": cp,
                    "nv": nv,
                    "td": td,
                }
            )

    raw_csv = out_dir / "anytime_raw.csv"
    with raw_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["instance", "solver", "seed", "cutoff_sec", "nv", "td"])
        writer.writeheader()
        writer.writerows(sample_rows)
    print(f"✓ Saved raw sampled checkpoints to {raw_csv}")

    # Generate summary
    import pandas as pd

    df = pd.DataFrame(sample_rows)
    piv = df.pivot_table(index=["instance", "cutoff_sec"], columns="solver", values=["nv", "td"], aggfunc="mean")
    print("\n=== ANYTIME BENCHMARK AGGREGATE SUMMARY (5 SEEDS) ===")
    print(piv.round(2))

    summary_csv = out_dir / "anytime_summary.csv"
    piv.to_csv(summary_csv)
    print(f"✓ Saved summary table to {summary_csv}")

    # Plot convergence
    fig_path = ROOT / "docs" / "figures" / "anytime_convergence.png"
    plot_anytime_convergence(sample_rows, fig_path)


if __name__ == "__main__":
    main()

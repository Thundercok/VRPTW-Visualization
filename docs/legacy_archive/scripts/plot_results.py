#!/usr/bin/env python3
"""Plot benchmark results for travel distance, vehicle count, and runtime.

Outputs highly polished, modern research charts comparing performance.
"""

from __future__ import annotations
import os
import sys
import argparse

def main():
    parser = argparse.ArgumentParser(description="Plot VRPTW benchmark results.")
    parser.add_argument(
        "--results-file",
        type=str,
        default="docs/logs/benchmark_clean.csv",
        help="Path to benchmark_clean.csv results file"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="docs/logs",
        help="Directory to save generated charts"
    )
    args = parser.parse_args()

    if not os.path.exists(args.results_file):
        print(f"Error: Results file not found: {args.results_file}")
        print("Please run a benchmark first using run_benchmark.py to generate results.")
        return 1

    try:
        import pandas as pd
    except ImportError:
        print("Error: pandas is required to plot results. Install it using: pip install pandas")
        return 1

    try:
        import matplotlib.pyplot as plt
        import matplotlib.style as style
    except ImportError:
        print("\n[INFO] Matplotlib is required to produce visualization plots.")
        print("Please install it using: pip install matplotlib\n")
        return 1

    print(f"Reading results from: {args.results_file}")
    df = pd.read_csv(args.results_file)

    if df.empty:
        print("Error: Results file is empty.")
        return 1

    # Rename columns to standard names if they exist in the CSV
    rename_map = {
        "NV_mean": "NV",
        "TD_mean": "TD",
        "Time_s": "Time"
    }
    df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})

    # Ensure required columns are present
    required = {"Algorithm", "Instance", "NV", "TD", "Time"}
    missing = required - set(df.columns)
    if missing:
        print(f"Error: Missing columns in CSV: {missing}")
        return 1

    os.makedirs(args.output_dir, exist_ok=True)

    # ── Modern Styling ────────────────────────────────────────────────────────
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = ["Inter", "DejaVu Sans", "Arial"]
    plt.rcParams["axes.edgecolor"] = "#E0E0E0"
    plt.rcParams["axes.linewidth"] = 0.8
    plt.rcParams["grid.color"] = "#F0F0F0"
    plt.rcParams["grid.linestyle"] = "--"
    plt.rcParams["grid.linewidth"] = 0.5

    # Sophisticated color palette
    colors = {
        "ALNS-Base": "#FF6B6B",        # Sleek Coral
        "Hybrid-Fixed": "#4D96FF",     # Royal Blue
        "Hybrid-Rule": "#6BCB77",      # Emerald Green
        "Hybrid-DDQN": "#9B5DE5",      # Elegant Purple
        "OR-Tools": "#F77F00",         # Warm Amber
    }

    algos = [a for a in colors.keys() if a in df["Algorithm"].unique()]
    if not algos:
        # Fallback to whatever unique algorithms are present
        algos = list(df["Algorithm"].unique())
        import matplotlib.cm as cm
        palette = cm.get_cmap("tab10", len(algos))
        colors = {algo: palette(i) for i, algo in enumerate(algos)}

    instances = df["Instance"].unique()
    print(f"Generating charts for algorithms: {algos} across {len(instances)} instances...")

    # Plot 1: Total Distance Comparison (TD)
    fig, ax = plt.subplots(figsize=(10, 6), dpi=150)
    df_pivot_td = df.pivot(index="Instance", columns="Algorithm", values="TD")[algos]
    df_pivot_td.plot(kind="bar", ax=ax, color=[colors[a] for a in algos], width=0.8)
    ax.set_title("Total Travel Distance Comparison (Lower is Better)", fontsize=14, fontweight="bold", pad=15)
    ax.set_ylabel("Travel Distance (km)", fontsize=12)
    ax.set_xlabel("Solomon Instance", fontsize=12)
    ax.grid(True, axis="y")
    ax.legend(title="Algorithm", framealpha=0.9, edgecolor="#E0E0E0")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plot_td_path = os.path.join(args.output_dir, "benchmark_distance.png")
    plt.savefig(plot_td_path)
    plt.close()

    # Plot 2: Vehicle Count Comparison (NV)
    fig, ax = plt.subplots(figsize=(10, 6), dpi=150)
    df_pivot_nv = df.pivot(index="Instance", columns="Algorithm", values="NV")[algos]
    df_pivot_nv.plot(kind="bar", ax=ax, color=[colors[a] for a in algos], width=0.8)
    ax.set_title("Vehicle Count Comparison (Lower is Better)", fontsize=14, fontweight="bold", pad=15)
    ax.set_ylabel("Number of Vehicles", fontsize=12)
    ax.set_xlabel("Solomon Instance", fontsize=12)
    ax.grid(True, axis="y")
    ax.legend(title="Algorithm", framealpha=0.9, edgecolor="#E0E0E0")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plot_nv_path = os.path.join(args.output_dir, "benchmark_vehicles.png")
    plt.savefig(plot_nv_path)
    plt.close()

    # Plot 3: Solve Time Comparison
    fig, ax = plt.subplots(figsize=(10, 6), dpi=150)
    df_pivot_time = df.pivot(index="Instance", columns="Algorithm", values="Time")[algos]
    df_pivot_time.plot(kind="bar", ax=ax, color=[colors[a] for a in algos], width=0.8)
    ax.set_title("Computational Solve Time Comparison (Lower is Better)", fontsize=14, fontweight="bold", pad=15)
    ax.set_ylabel("Solve Time (seconds)", fontsize=12)
    ax.set_xlabel("Solomon Instance", fontsize=12)
    ax.grid(True, axis="y")
    ax.legend(title="Algorithm", framealpha=0.9, edgecolor="#E0E0E0")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plot_time_path = os.path.join(args.output_dir, "benchmark_runtime.png")
    plt.savefig(plot_time_path)
    plt.close()

    print("\n🎉 Success! Beautiful comparative charts generated:")
    print(f"  • Distance: {plot_td_path}")
    print(f"  • Vehicles: {plot_nv_path}")
    print(f"  • Runtime:  {plot_time_path}")
    return 0

if __name__ == "__main__":
    sys.exit(main())

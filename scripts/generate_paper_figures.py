"""
Generate publication-quality figures for Paper 1 using pure matplotlib:
1. NV Distribution Boxplots (Solomon-100, GH-200, GH-400, and Overall N=176)
2. Convergence Curves across Search Iterations for representative instances
"""

from __future__ import annotations

import os
import sys

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Setup paths
_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_REPO, "src"))

from vrptw.config import Config
from vrptw.core import load_solomon_instance
from vrptw.solvers import ALNSSolver, HybridDDQNSolver, HybridFixedSolver, HybridRuleSolver

FIG_DIR = os.path.join(_REPO, "docs", "figures")
os.makedirs(FIG_DIR, exist_ok=True)
CSV_PATH = os.path.join(_REPO, "results", "ultimate-publication-suite", "combined_clean.csv")

# Set publication style
plt.rcParams.update({
    "font.size": 10,
    "axes.labelsize": 11,
    "axes.titlesize": 11.5,
    "xtick.labelsize": 9.5,
    "ytick.labelsize": 9.5,
    "legend.fontsize": 9.5,
    "figure.titlesize": 13,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.35,
    "grid.linestyle": "--",
})

ALGO_ORDER = ["ALNS-Base", "Hybrid-Fixed", "Hybrid-Rule", "Hybrid-DDQN"]
ALGO_PALETTE = {
    "ALNS-Base": "#64748b",     # Slate
    "Hybrid-Fixed": "#f59e0b",   # Amber
    "Hybrid-Rule": "#3b82f6",    # Blue
    "Hybrid-DDQN": "#10b981",    # Emerald
}

def generate_nv_boxplots():
    print(f"[1/2] Generating NV distribution boxplots from {CSV_PATH}...")
    df = pd.read_csv(CSV_PATH)
    df = df[df["Algorithm"].isin(ALGO_ORDER)].copy()

    # Categorize suites
    def assign_suite(row):
        d = row["Dataset"]
        if d in ["C1", "C2", "R1", "R2", "RC1", "RC2"]:
            return "Solomon-100 (N=56)"
        elif d in ["c1_2", "c2_2", "r1_2", "r2_2", "rc1_2", "rc2_2"]:
            return "Homberger-200 (N=60)"
        elif d in ["c1_4", "c2_4", "r1_4", "r2_4", "rc1_4", "rc2_4"]:
            return "Homberger-400 (N=60)"
        return "Other"

    df["Suite"] = df.apply(assign_suite, axis=1)

    suite_order = [
        "Solomon-100 (N=56)",
        "Homberger-200 (N=60)",
        "Homberger-400 (N=60)",
        "Overall Suite (N=176)"
    ]

    fig, axes = plt.subplots(1, 4, figsize=(14, 4.0), sharey=True)

    for idx, (ax, suite_name) in enumerate(zip(axes, suite_order)):
        if suite_name == "Overall Suite (N=176)":
            sub_df = df
        else:
            sub_df = df[df["Suite"] == suite_name]

        data_to_plot = []
        for algo in ALGO_ORDER:
            algo_vals = sub_df[sub_df["Algorithm"] == algo]["NV_diff"].dropna().values
            data_to_plot.append(algo_vals)

        bp = ax.boxplot(
            data_to_plot,
            positions=np.arange(1, len(ALGO_ORDER) + 1),
            widths=0.55,
            patch_artist=True,
            showmeans=True,
            meanprops={
                "marker": "D",
                "markerfacecolor": "white",
                "markeredgecolor": "black",
                "markersize": 5
            },
            medianprops={"color": "black", "linewidth": 1.5},
            flierprops={"marker": "o", "markersize": 3.5, "alpha": 0.5}
        )

        for patch, algo in zip(bp["boxes"], ALGO_ORDER):
            c = ALGO_PALETTE[algo]
            patch.set_facecolor(c)
            patch.set_alpha(0.65)
            patch.set_edgecolor(c)
            patch.set_linewidth(1.3)

        # Add slight jittered scatter points for raw observations
        np.random.seed(42)
        for i, vals in enumerate(data_to_plot, start=1):
            jitter = np.random.normal(0, 0.06, size=len(vals))
            ax.scatter(
                np.full_like(vals, i) + jitter,
                vals,
                alpha=0.35,
                color="black",
                s=12,
                zorder=3
            )

        ax.set_title(suite_name, fontweight="bold", pad=8)
        if idx == 0:
            ax.set_ylabel(r"Fleet Size Gap to BKS ($NV - NV_{\mathrm{BKS}}$)", fontweight="bold")

        ax.set_xticks(np.arange(1, len(ALGO_ORDER) + 1))
        ax.set_xticklabels(["ALNS\nBase", "Hybrid\nFixed", "Hybrid\nRule", "Hybrid\nDDQN"], fontsize=9)
        ax.axhline(0, color="#dc2626", linestyle=":", linewidth=1.2, alpha=0.7)
        ax.set_xlim(0.4, len(ALGO_ORDER) + 0.6)

    plt.tight_layout()
    out_pdf = os.path.join(FIG_DIR, "nv_distribution_boxplots.pdf")
    out_png = os.path.join(FIG_DIR, "nv_distribution_boxplots.png")
    fig.savefig(out_pdf, dpi=300, bbox_inches="tight")
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_pdf} and {out_png}")


def generate_convergence_curves():
    print("[2/2] Generating convergence search trajectories on landmark instances...")

    instances = [
        ("Solomon RC101", "data/Solomon/rc101.txt", 800),
        ("Solomon R101", "data/Solomon/r101.txt", 800),
        ("Solomon C101", "data/Solomon/c101.txt", 800),
        ("GH r1_2_1 (200c)", "data/Gehring_Homberger/homberger_200_customer_instances/R1_2_1.TXT", 500),
    ]

    solver_map = {
        "ALNS-Base": ALNSSolver,
        "Hybrid-Fixed": HybridFixedSolver,
        "Hybrid-Rule": HybridRuleSolver,
        "Hybrid-DDQN": HybridDDQNSolver,
    }

    fig, axes = plt.subplots(1, 4, figsize=(14.5, 3.8))

    for ax, (inst_label, inst_relpath, iters) in zip(axes, instances):
        inst_path = os.path.join(_REPO, inst_relpath)
        if not os.path.exists(inst_path):
            print(f"Skipping {inst_label}, file not found: {inst_path}")
            continue

        inst = load_solomon_instance(inst_path)

        cfg = Config(
            alns_iterations=iters,
            hybrid_iterations=iters,
            early_stop_patience=iters,
            time_limit=None,
            time_limit_per_customer=0.0
        )

        print(f"  Simulating search trajectories for {inst_label} ({iters} iters)...")

        for algo_name, solver_cls in solver_map.items():
            histories = []
            for seed in [1, 7]:
                solver = solver_cls(inst, cfg)
                _, hist = solver.solve(seed=seed)
                histories.append(hist)

            min_len = min(len(h) for h in histories)
            arr = np.array([h[:min_len] for h in histories])
            mean_hist = np.mean(arr, axis=0)
            std_hist = np.std(arr, axis=0)

            x = np.arange(len(mean_hist))
            color = ALGO_PALETTE[algo_name]
            ax.plot(x, mean_hist, label=algo_name, color=color, linewidth=1.6)
            ax.fill_between(x, mean_hist - std_hist, mean_hist + std_hist, color=color, alpha=0.12)

        ax.set_title(inst_label, fontweight="bold")
        ax.set_xlabel("Search Iteration", fontweight="bold")
        if ax == axes[0]:
            ax.set_ylabel("Best Incumbent Cost", fontweight="bold")
        ax.grid(True, alpha=0.35)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 1.06), ncol=4, frameon=True)

    plt.tight_layout()
    out_pdf = os.path.join(FIG_DIR, "convergence_curves.pdf")
    out_png = os.path.join(FIG_DIR, "convergence_curves.png")
    fig.savefig(out_pdf, dpi=300, bbox_inches="tight")
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_pdf} and {out_png}")


if __name__ == "__main__":
    generate_nv_boxplots()
    generate_convergence_curves()
    print("All figures successfully generated!")

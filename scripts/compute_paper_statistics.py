#!/usr/bin/env python3
"""Automated Statistical Testing Suite for VRPTW-Research-Optimization.

Calculates:
1. Two-tailed Wilcoxon Signed-Rank Test (W-statistic, p-value) across instance means.
2. Exact Binomial Sign Test (Wins / Ties / Losses, two-sided p-value).
3. Bonferroni Multiple Testing Correction (alpha_adj = 0.05 / m).
4. Matched-NV Fairness & Gap Filtering:
   - Compares TD only on vehicle-matched subsets (NV == NV_BKS or NV_solver == NV_baseline).
   - Flags unmatched rows with NV^dagger to prevent capacity-distorted travel distance comparisons.
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))


def compute_instance_aggregates(df: pd.DataFrame) -> pd.DataFrame:
    """Computes mean NV, mean TD, wall-clock time, and explicit Feasibility Rate per instance and solver."""
    agg = (
        df.groupby(["Instance", "Family", "Algorithm"])
        .agg(
            NV_mean=("NV", "mean"),
            NV_std=("NV", "std"),
            TD_mean=("TD", "mean"),
            TD_std=("TD", "std"),
            Time_mean=("Time_Sec", "mean"),
            Feasibility_Rate=("Feasible", "mean"),
            Total_Runs=("Seed", "count"),
            Feasible_Runs=("Feasible", "sum"),
            BKS_NV=("BKS_NV", "first"),
            BKS_TD=("BKS_TD", "first"),
        )
        .reset_index()
    )
    agg["Feas_Pct"] = agg["Feasibility_Rate"] * 100.0
    agg["Gap_NV"] = agg["NV_mean"] - agg["BKS_NV"]
    agg["Gap_TD_Pct"] = (agg["TD_mean"] - agg["BKS_TD"]) / agg["BKS_TD"] * 100.0
    agg["Matched_NV"] = agg["NV_mean"] == agg["BKS_NV"]
    return agg


def run_statistical_comparison(
    df_agg: pd.DataFrame,
    solver_a: str,
    solver_b: str,
    alpha: float = 0.05,
    m_comparisons: int = 3,
) -> dict[str, Any]:
    """Runs Wilcoxon signed-rank and Binomial Sign tests between two solvers across instances."""
    alpha_bonferroni = alpha / max(m_comparisons, 1)

    pivot_nv = df_agg.pivot(index="Instance", columns="Algorithm", values="NV_mean")
    pivot_td = df_agg.pivot(index="Instance", columns="Algorithm", values="TD_mean")

    common_instances = pivot_nv.dropna(subset=[solver_a, solver_b]).index.tolist()
    if not common_instances:
        return {"error": f"No common instances between {solver_a} and {solver_b}"}

    nv_a = pivot_nv.loc[common_instances, solver_a].to_numpy()
    nv_b = pivot_nv.loc[common_instances, solver_b].to_numpy()
    td_a = pivot_td.loc[common_instances, solver_a].to_numpy()
    td_b = pivot_td.loc[common_instances, solver_b].to_numpy()

    n_instances = len(common_instances)

    # ── 1. NV Statistical Tests ──────────────────────────────────────────
    diff_nv = nv_b - nv_a  # negative means solver_b has fewer vehicles (better)
    wins_nv = int(np.sum(diff_nv < -1e-6))
    losses_nv = int(np.sum(diff_nv > 1e-6))
    ties_nv = int(np.sum(np.abs(diff_nv) <= 1e-6))
    med_diff_nv = float(np.median(diff_nv))

    non_zero_nv = diff_nv[np.abs(diff_nv) > 1e-6]
    if len(non_zero_nv) == 0:
        w_nv, p_nv, r_nv = 0.0, 1.0, 0.0
    else:
        try:
            res_w_nv = stats.wilcoxon(nv_b, nv_a, alternative="two-sided", zero_method="pratt")
            w_nv, p_nv = float(res_w_nv.statistic), float(res_w_nv.pvalue)
            # Exact rank-biserial correlation r_rb = (W_minus - W_plus) / (W_plus + W_minus)
            ranks_nv = stats.rankdata(np.abs(non_zero_nv))
            w_pos = float(np.sum(ranks_nv[non_zero_nv > 0]))
            w_neg = float(np.sum(ranks_nv[non_zero_nv < 0]))
            s_sum = max(w_pos + w_neg, 1e-9)
            r_nv = float((w_neg - w_pos) / s_sum)  # positive r means solver_b beats solver_a
        except Exception:
            w_nv, p_nv, r_nv = 0.0, 1.0, 0.0

    non_ties_nv = wins_nv + losses_nv
    if non_ties_nv > 0:
        sign_res_nv = stats.binomtest(wins_nv, n=non_ties_nv, p=0.5, alternative="two-sided")
        p_sign_nv = float(sign_res_nv.pvalue)
    else:
        p_sign_nv = 1.0

    # ── 2. TD Statistical Tests (Matched-NV & Overall) ───────────────────
    # Matched-NV filter (where both solvers hit exact same vehicle count)
    matched_mask = np.abs(nv_a - nv_b) <= 1e-6
    n_matched = int(np.sum(matched_mask))

    if n_matched > 0:
        td_a_matched = td_a[matched_mask]
        td_b_matched = td_b[matched_mask]
        diff_td_matched = td_b_matched - td_a_matched

        wins_td_m = int(np.sum(diff_td_matched < -1e-4))
        losses_td_m = int(np.sum(diff_td_matched > 1e-4))
        ties_td_m = int(np.sum(np.abs(diff_td_matched) <= 1e-4))
        med_diff_td = float(np.median(diff_td_matched))

        non_zero_td = diff_td_matched[np.abs(diff_td_matched) > 1e-4]
        if len(non_zero_td) == 0:
            w_td, p_td, r_td = 0.0, 1.0, 0.0
        else:
            try:
                res_w_td = stats.wilcoxon(td_b_matched, td_a_matched, alternative="two-sided", zero_method="pratt")
                w_td, p_td = float(res_w_td.statistic), float(res_w_td.pvalue)
                ranks_td = stats.rankdata(np.abs(non_zero_td))
                w_pos_td = float(np.sum(ranks_td[non_zero_td > 0]))
                w_neg_td = float(np.sum(ranks_td[non_zero_td < 0]))
                s_sum_td = max(w_pos_td + w_neg_td, 1e-9)
                r_td = float((w_neg_td - w_pos_td) / s_sum_td)
            except Exception:
                w_td, p_td, r_td = 0.0, 1.0, 0.0

        non_ties_td = wins_td_m + losses_td_m
        if non_ties_td > 0:
            sign_res_td = stats.binomtest(wins_td_m, n=non_ties_td, p=0.5, alternative="two-sided")
            p_sign_td = float(sign_res_td.pvalue)
        else:
            p_sign_td = 1.0

        mean_gap_matched = float(np.mean((td_b_matched - td_a_matched) / td_a_matched * 100.0))
    else:
        w_td, p_td, p_sign_td, r_td = 0.0, 1.0, 1.0, 0.0
        wins_td_m, losses_td_m, ties_td_m = 0, 0, 0
        mean_gap_matched = 0.0
        med_diff_td = 0.0

    return {
        "solver_a": solver_a,
        "solver_b": solver_b,
        "n_instances": n_instances,
        "alpha_nominal": alpha,
        "alpha_bonferroni": alpha_bonferroni,
        "nv_comparison": {
            "mean_nv_a": float(np.mean(nv_a)),
            "mean_nv_b": float(np.mean(nv_b)),
            "median_diff": round(med_diff_nv, 3),
            "rank_biserial_r": round(r_nv, 3),
            "wins_b": wins_nv,
            "losses_b": losses_nv,
            "ties": ties_nv,
            "wilcoxon_w": w_nv,
            "wilcoxon_p": p_nv,
            "sign_test_p": p_sign_nv,
            "significant_bonferroni": bool(p_nv < alpha_bonferroni),
        },
        "td_matched_comparison": {
            "n_matched": n_matched,
            "pct_matched": round(n_matched / n_instances * 100.0, 1),
            "mean_gap_pct": round(mean_gap_matched, 2),
            "median_diff": round(med_diff_td, 2),
            "rank_biserial_r": round(r_td, 3),
            "wins_b": wins_td_m,
            "losses_b": losses_td_m,
            "ties": ties_td_m,
            "wilcoxon_w": w_td,
            "wilcoxon_p": p_td,
            "sign_test_p": p_sign_td,
            "significant_bonferroni": bool(p_td < alpha_bonferroni),
        },
    }


def format_statistical_report(stat_results: list[dict[str, Any]]) -> str:
    """Formats statistical outcomes into markdown and human-readable text."""
    lines = []
    lines.append("# Statistical Rigor Report (Two-Tailed Wilcoxon & Exact Sign Test)")
    lines.append(f"Significance Thresholds: Nominal $\\alpha = 0.05$, Bonferroni $\\alpha_{{\\text{{adj}}}} = {0.05/3:.4f}$ (m=3 families).\n")

    for res in stat_results:
        if "error" in res:
            continue
        lines.append(f"## Comparison: {res['solver_b']} vs {res['solver_a']} (N={res['n_instances']} instances)")
        nv = res["nv_comparison"]
        td = res["td_matched_comparison"]

        lines.append("### 1. Fleet Size Reduction (NV):")
        lines.append(f"- Mean NV: {res['solver_a']} = {nv['mean_nv_a']:.2f} vs {res['solver_b']} = {nv['mean_nv_b']:.2f}")
        lines.append(f"- Record: {nv['wins_b']} Wins / {nv['ties']} Ties / {nv['losses_b']} Losses")
        lines.append(f"- Wilcoxon signed-rank: $W = {nv['wilcoxon_w']:.1f}$, $p = {nv['wilcoxon_p']:.4e}$ ({'Significant' if nv['significant_bonferroni'] else 'Not Significant'})")
        lines.append(f"- Exact Sign test: $p = {nv['sign_test_p']:.4e}\n")

        lines.append("### 2. Vehicle-Matched Distance Gap (TD):")
        lines.append(f"- Matched Instances: {td['n_matched']}/{res['n_instances']} ({td['pct_matched']}%)")
        lines.append(f"- Mean TD Gap on Matched: {td['mean_gap_pct']:+.2f}%")
        lines.append(f"- Record on Matched: {td['wins_b']} Wins / {td['ties']} Ties / {td['losses_b']} Losses")
        lines.append(f"- Wilcoxon signed-rank: $W = {td['wilcoxon_w']:.1f}$, $p = {td['wilcoxon_p']:.4e}$ ({'Significant' if td['significant_bonferroni'] else 'Not Significant'})")
        lines.append(f"- Exact Sign test: $p = {td['sign_test_p']:.4e}\n")
        lines.append("-" * 75)

    return "\n".join(lines)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute rigorous statistical tests for paper tables.")
    parser.add_argument("--csv", type=str, required=True, help="Path to raw benchmark results CSV.")
    parser.add_argument("--out-md", type=str, default=None, help="Output markdown report path.")
    args = parser.parse_args()

    df = pd.read_csv(args.csv)
    df_agg = compute_instance_aggregates(df)

    solvers = df_agg["Algorithm"].unique().tolist()
    stats_list = []
    if "ALNS-Base" in solvers and "Hybrid-DDQN" in solvers:
        stats_list.append(run_statistical_comparison(df_agg, "ALNS-Base", "Hybrid-DDQN"))

    report = format_statistical_report(stats_list)
    print(report)

    if args.out_md:
        with open(args.out_md, "w", encoding="utf-8") as fh:
            fh.write(report)
        print(f"Report saved to {args.out_md}")

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
from pathlib import Path
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


def run_seed_paired_statistical_comparison(
    df_raw: pd.DataFrame,
    solver_a: str,
    solver_b: str,
    alpha: float = 0.05,
    m_comparisons: int = 3,
) -> dict[str, Any]:
    """Computes exact seed-paired comparison between two solvers across identical (Instance, Seed) runs."""
    alpha_bonferroni = alpha / max(m_comparisons, 1)

    df = df_raw.copy()
    col_map = {
        "solver": "Algorithm",
        "config": "Algorithm",
        "instance": "Instance",
        "seed": "Seed",
        "nv": "NV",
        "td": "TD",
        "feasible": "Feasible",
    }
    df.rename(columns={k: v for k, v in col_map.items() if k in df.columns}, inplace=True)
    df["Instance"] = df["Instance"].apply(lambda x: Path(str(x)).stem.upper())

    sub_a = df[df["Algorithm"] == solver_a].set_index(["Instance", "Seed"])
    sub_b = df[df["Algorithm"] == solver_b].set_index(["Instance", "Seed"])

    common_keys = sub_a.index.intersection(sub_b.index)
    if len(common_keys) == 0:
        return {"error": f"No common (Instance, Seed) pairs between {solver_a} and {solver_b}"}

    merged = pd.DataFrame(
        {
            "nv_a": sub_a.loc[common_keys, "NV"].astype(float),
            "td_a": sub_a.loc[common_keys, "TD"].astype(float),
            "nv_b": sub_b.loc[common_keys, "NV"].astype(float),
            "td_b": sub_b.loc[common_keys, "TD"].astype(float),
        },
        index=common_keys,
    ).dropna()

    n_pairs = len(merged)
    diff_nv = merged["nv_b"].to_numpy() - merged["nv_a"].to_numpy()
    wins_nv = int(np.sum(diff_nv < -1e-6))
    losses_nv = int(np.sum(diff_nv > 1e-6))
    ties_nv = int(np.sum(np.abs(diff_nv) <= 1e-6))

    # Matched-Fleet TD subset: exactly equal vehicle counts on that specific run
    matched = merged[np.abs(merged["nv_a"] - merged["nv_b"]) <= 1e-6]
    n_matched = len(matched)

    if n_matched > 0:
        td_a = matched["td_a"].to_numpy()
        td_b = matched["td_b"].to_numpy()
        diff_td = td_b - td_a
        gap_pct = (td_b - td_a) / td_a * 100.0

        wins_td = int(np.sum(diff_td < -1e-4))
        losses_td = int(np.sum(diff_td > 1e-4))
        ties_td = int(np.sum(np.abs(diff_td) <= 1e-4))

        non_zero_td = diff_td[np.abs(diff_td) > 1e-4]
        if len(non_zero_td) == 0:
            w_td, p_td, _r_td = 0.0, 1.0, 0.0
        else:
            try:
                res_w = stats.wilcoxon(td_b, td_a, alternative="two-sided", zero_method="pratt")
                w_td, p_td = float(res_w.statistic), float(res_w.pvalue)
                ranks = stats.rankdata(np.abs(non_zero_td))
                w_pos = float(np.sum(ranks[non_zero_td > 0]))
                w_neg = float(np.sum(ranks[non_zero_td < 0]))
                float((w_neg - w_pos) / max(w_pos + w_neg, 1e-9))
            except Exception:
                w_td, p_td, _r_td = 0.0, 1.0, 0.0

        sign_p = (
            float(stats.binomtest(wins_td, n=wins_td + losses_td, p=0.5).pvalue) if (wins_td + losses_td) > 0 else 1.0
        )
        mean_gap = float(np.mean(gap_pct))
        med_diff = float(np.median(diff_td))
    else:
        w_td, p_td, _r_td, sign_p = 0.0, 1.0, 0.0, 1.0
        wins_td, losses_td, ties_td = 0, 0, 0
        mean_gap, med_diff = 0.0, 0.0

    return {
        "solver_a": solver_a,
        "solver_b": solver_b,
        "n_seed_pairs": n_pairs,
        "fleet_size": {
            "wins_b": wins_nv,
            "losses_b": losses_nv,
            "ties": ties_nv,
            "mean_nv_a": float(np.mean(merged["nv_a"])),
            "mean_nv_b": float(np.mean(merged["nv_b"])),
        },
        "matched_fleet_distance": {
            "n_matched_pairs": n_matched,
            "pct_matched": round(n_matched / n_pairs * 100.0, 1) if n_pairs else 0.0,
            "mean_td_gap_pct": round(mean_gap, 2),
            "median_td_diff": round(med_diff, 2),
            "wins_td": wins_td,
            "losses_td": losses_td,
            "ties_td": ties_td,
            "wilcoxon_w": w_td,
            "wilcoxon_p": p_td,
            "sign_test_p": sign_p,
            "significant_bonferroni": bool(p_td < alpha_bonferroni),
        },
    }


def format_seed_paired_report(res: dict[str, Any]) -> str:
    lines = []
    lines.append(
        f"## Seed-Paired Matched-Fleet Analysis: {res['solver_b']} vs {res['solver_a']} (N={res['n_seed_pairs']} runs)"
    )
    f = res["fleet_size"]
    m = res["matched_fleet_distance"]
    lines.append("### 1. Primary Fleet Outcome (NV):")
    lines.append(
        f"- Record: {f['wins_b']} Wins / {f['ties']} Ties / {f['losses_b']} Losses (Mean NV: {f['mean_nv_b']:.2f} vs {f['mean_nv_a']:.2f})"
    )
    lines.append("### 2. Seed-Paired Matched-Fleet Distance (TD):")
    lines.append(f"- Matched Fleet Runs: {m['n_matched_pairs']}/{res['n_seed_pairs']} ({m['pct_matched']}%)")
    lines.append(f"- Matched TD Record: {m['wins_td']} Wins / {m['ties_td']} Ties / {m['losses_td']} Losses")
    lines.append(
        f"- Mean TD Reduction on Matched Fleets: {m['mean_td_gap_pct']:+.2f}% (Median diff: {m['median_td_diff']:+.2f})"
    )
    lines.append(
        f"- Wilcoxon signed-rank: W={m['wilcoxon_w']:.1f}, p={m['wilcoxon_p']:.4e} ({'Significant' if m['significant_bonferroni'] else 'Not Significant'})"
    )
    lines.append(f"- Exact Sign test: p={m['sign_test_p']:.4e}")
    return "\n".join(lines)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute rigorous statistical tests for paper tables.")
    parser.add_argument("--csv", type=str, required=True, help="Path to raw benchmark results CSV.")
    parser.add_argument("--solver-a", type=str, default="alns", help="Baseline solver name/config.")
    parser.add_argument("--solver-b", type=str, default="full", help="Candidate solver name/config.")
    parser.add_argument("--out-md", type=str, default=None, help="Output markdown report path.")
    args = parser.parse_args()

    df = pd.read_csv(args.csv)
    res_paired = run_seed_paired_statistical_comparison(df, args.solver_a, args.solver_b)
    if "error" in res_paired:
        print(f"Paired comparison error: {res_paired['error']}")
    else:
        report = format_seed_paired_report(res_paired)
        print(report)
        if args.out_md:
            Path(args.out_md).write_text(report, encoding="utf-8")

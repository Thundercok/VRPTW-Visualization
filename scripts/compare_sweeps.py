"""Compare two benchmark sweep CSVs (old vs new) per (Instance, Algorithm).

    python scripts/compare_sweeps.py OLD.csv NEW.csv [--algorithms Hybrid-DDQN ...]

This is an UNPAIRED comparison: the two sweeps were run with different seeds,
possibly on different machines. It is a sanity cross-check, not causal evidence
of improvement — cite the paired A/B harness (scripts/ab_compare.py) for that.

Reports, overall and per instance family (C1/C2/R1/R2/RC1/RC2):
  * mean NV and TD deltas with Wilcoxon signed-rank over per-instance means
  * TD compared only at matched vehicle count (the test that reversed the SISR
    conclusion — TD deltas at different NV are meaningless under a
    lexicographic objective)
  * BKS vehicle-count floor hits (the paper's NV-flattening argument)
"""

from __future__ import annotations

import argparse
import os
import re
import sys

import numpy as np
import pandas as pd

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_REPO, "src"))

from vrptw.config import BKS  # noqa: E402


def _family(instance: str) -> str:
    """R101 -> R1, rc207 -> RC2, r1_2_1 -> R1, c2_4_1 -> C2."""
    m = re.match(r"([a-zA-Z]+)(\d)", instance.strip())
    if not m:
        return "?"
    return (m.group(1) + m.group(2)).upper()


def _load(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    need = {"Instance", "Algorithm", "NV_mean", "TD_mean"}
    missing = need - set(df.columns)
    if missing:
        raise SystemExit(f"{path}: missing columns {sorted(missing)}")
    df = df[df["NV_mean"].notna()].copy()
    df["key_inst"] = df["Instance"].str.strip().str.lower()
    df["family"] = df["Instance"].map(_family)
    return df


def _bks_for(instance: str) -> dict | None:
    return BKS.get(instance) or BKS.get(instance.upper()) or BKS.get(instance.lower())


def _fmt_p(p: float | None) -> str:
    return "n/a" if p is None else f"{p:.4f}"


def _wilcoxon(before: np.ndarray, after: np.ndarray) -> float | None:
    if len(before) < 5 or np.allclose(before, after):
        return None
    try:
        from scipy.stats import wilcoxon

        _stat, p = wilcoxon(before, after)
        return float(p)
    except Exception:
        return None


def compare_block(merged: pd.DataFrame, label: str) -> None:
    nv_o = merged["NV_mean_old"].to_numpy(float)
    nv_n = merged["NV_mean_new"].to_numpy(float)
    td_o = merged["TD_mean_old"].to_numpy(float)
    td_n = merged["TD_mean_new"].to_numpy(float)

    p_nv = _wilcoxon(nv_o, nv_n)
    p_td = _wilcoxon(td_o, td_n)

    print(f"\n-- {label}  ({len(merged)} instance rows) " + "-" * max(1, 40 - len(label)))
    print(
        f"  NV mean : {nv_o.mean():8.3f} -> {nv_n.mean():8.3f}  ({nv_n.mean() - nv_o.mean():+.3f})"
        f"   Wilcoxon p={_fmt_p(p_nv)}"
    )
    print(
        f"  TD mean : {td_o.mean():8.1f} -> {td_n.mean():8.1f}  ({td_n.mean() - td_o.mean():+.1f})"
        f"   Wilcoxon p={_fmt_p(p_td)}"
    )
    print(
        f"  NV: better {int((nv_n < nv_o - 1e-9).sum())}, worse {int((nv_n > nv_o + 1e-9).sum())}, "
        f"tie {int((np.abs(nv_n - nv_o) <= 1e-9).sum())}"
    )

    # ── TD at matched vehicle count only ─────────────────────────────────────
    matched = merged[np.abs(merged["NV_mean_old"] - merged["NV_mean_new"]) <= 1e-9]
    if len(matched):
        gaps_o, gaps_n = [], []
        for _, r in matched.iterrows():
            bks = _bks_for(r["Instance_old"])
            if bks:
                gaps_o.append((r["TD_mean_old"] - bks["td"]) / bks["td"] * 100)
                gaps_n.append((r["TD_mean_new"] - bks["td"]) / bks["td"] * 100)
        if gaps_o:
            go, gn = float(np.mean(gaps_o)), float(np.mean(gaps_n))
            p_m = _wilcoxon(np.array(gaps_o), np.array(gaps_n))
            print(
                f"  TD at matched NV ({len(matched)} rows, {len(gaps_o)} with BKS): "
                f"gap {go:.2f}% -> {gn:.2f}% ({gn - go:+.2f} pp)  Wilcoxon p={_fmt_p(p_m)}"
            )
    else:
        print("  TD at matched NV: no rows with identical NV_mean")

    # ── BKS NV floor hits ────────────────────────────────────────────────────
    floor_o = floor_n = with_bks = 0
    for _, r in merged.iterrows():
        bks = _bks_for(r["Instance_old"])
        if not bks:
            continue
        with_bks += 1
        if round(float(r["NV_mean_old"]), 2) <= bks["nv"]:
            floor_o += 1
        if round(float(r["NV_mean_new"]), 2) <= bks["nv"]:
            floor_n += 1
    if with_bks:
        print(f"  BKS NV floor hit (NV_mean at/below BKS): {floor_o}/{with_bks} -> {floor_n}/{with_bks}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("old_csv")
    ap.add_argument("new_csv")
    ap.add_argument(
        "--algorithms", nargs="+", default=None, help="restrict to these Algorithm labels (default: all shared)"
    )
    args = ap.parse_args()

    old = _load(args.old_csv)
    new = _load(args.new_csv)
    if args.algorithms:
        old = old[old["Algorithm"].isin(args.algorithms)]
        new = new[new["Algorithm"].isin(args.algorithms)]

    merged = old.merge(new, on=["key_inst", "Algorithm"], suffixes=("_old", "_new"))
    if merged.empty:
        raise SystemExit("No overlapping (Instance, Algorithm) rows between the two files.")

    print("=" * 70)
    print("UNPAIRED sweep comparison — different seeds/machine/date.")
    print("Sanity cross-check only; cite the paired A/B for improvement claims.")
    print("=" * 70)
    print(f"old: {args.old_csv}\nnew: {args.new_csv}")

    for algo, g_algo in merged.groupby("Algorithm", observed=True):
        if not len(g_algo):
            continue
        print(f"\n{'=' * 70}\nAlgorithm: {algo}\n{'=' * 70}")
        compare_block(g_algo, "ALL")
        for fam in sorted(g_algo["family_old"].unique()):
            g_fam = g_algo[g_algo["family_old"] == fam]
            if len(g_fam) >= 2:
                compare_block(g_fam, f"family {fam}")

    # Wall-time overview (informational only — different machines!)
    if "Time_s_old" in merged.columns and merged["Time_s_old"].notna().all():
        to, tn = merged["Time_s_old"].sum(), merged["Time_s_new"].sum()
        print(f"\nTotal Time_s (informational, machines may differ): {to:.0f}s -> {tn:.0f}s")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Create the six-row Solomon summary table used in presentation outputs."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

FAMILIES = ("C1", "C2", "R1", "R2", "RC1", "RC2")


def main() -> None:
    parser = argparse.ArgumentParser(description="Format a Solomon benchmark summary as Markdown.")
    parser.add_argument("input", type=Path, help="Benchmark CSV (for example benchmark_clean.csv)")
    parser.add_argument("--algorithm", default="Hybrid-DDQN", help="Algorithm to summarize")
    parser.add_argument("--output", type=Path, help="Optional Markdown output path")
    args = parser.parse_args()

    df = pd.read_csv(args.input)
    required = {"Dataset", "Algorithm", "NV_mean", "TD_mean", "Gap%", "Time_s"}
    missing = required.difference(df.columns)
    if missing:
        raise SystemExit(f"Missing required columns: {', '.join(sorted(missing))}")

    rows = []
    data = df[df["Algorithm"] == args.algorithm]
    for family in FAMILIES:
        group = data[data["Dataset"].str.upper() == family]
        if group.empty:
            continue
        rows.append({
            "Lớp dữ liệu": f"{family} ({group['Instance'].nunique()} bài)",
            "Thuật toán": args.algorithm,
            "Mean NV": group["NV_mean"].mean(),
            "Mean TD": group["TD_mean"].mean(),
            "Gap% vs BKS": group["Gap%"].mean(),
            "Thời gian": group["Time_s"].mean(),
        })

    if len(rows) != len(FAMILIES):
        found = ", ".join(r["Lớp dữ liệu"].split()[0] for r in rows) or "none"
        raise SystemExit(f"Expected all six Solomon families; found: {found}")

    out = pd.DataFrame(rows)
    formatted = out.copy()
    formatted["Mean NV"] = formatted["Mean NV"].map("{:.2f}".format)
    formatted["Mean TD"] = formatted["Mean TD"].map("{:.2f}".format)
    formatted["Gap% vs BKS"] = formatted["Gap% vs BKS"].map("{:+.2f}%".format)
    formatted["Thời gian"] = formatted["Thời gian"].map("{:.1f}s".format)
    table = formatted.to_markdown(index=False)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(table + "\n", encoding="utf-8")
    print(table)


if __name__ == "__main__":
    main()

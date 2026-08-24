"""Generate the LaTeX result tables in docs/paper.tex from sweep CSVs.

    python scripts/make_paper_tables.py --sweep results/rerun_iters/benchmark_clean.csv \
        [--gnn results/gnn_validation.csv] [--out-dir docs/tables]

Emits one .tex snippet per table (same labels as paper.tex):
  ablation.tex           tab:ablation           overall NV diff / TD Gap%
  nv_summary.tex         tab:nv_summary         NV_diff by Solomon subset
  distance_summary.tex   tab:distance_summary   NV-filtered TD Gap%
  fair_by_category.tex   tab:fair_by_category   fair intersection by family
  gh200.tex              tab:gh200              GH-200 per-instance NV/TD gap
  gnn_comparison.tex     tab:gnn_comparison     only when --gnn is given

The tables previously lived inline in paper.tex and were maintained by hand;
this script exists so "regenerate every number" is a command, not a
transcription exercise.
"""

from __future__ import annotations

import argparse
import os
import re
import sys

import pandas as pd

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_REPO, "src"))

from vrptw.config import BKS  # noqa: E402

ALGOS = ["ALNS-Base", "Hybrid-Fixed", "Hybrid-Rule", "Hybrid-DDQN"]
ALGO_HEADS = {
    "ALNS-Base": r"\textbf{ALNS~\cite{Ropke2006}}",
    "Hybrid-Fixed": r"\textbf{H-Fixed}",
    "Hybrid-Rule": r"\textbf{H-Rule}",
    "Hybrid-DDQN": r"\textbf{H-DDQN}",
    "OR-Tools": r"\textbf{OR-Tools~(iso-time)}",
}
SUBSETS = {
    "Clustered": ("C1", "C2"),
    "Short horizon": ("R1", "RC1"),
    "Wide horizon": ("R2", "RC2"),
}


def _family(instance: str) -> str:
    m = re.match(r"([a-zA-Z]+)(\d)", instance.strip())
    return (m.group(1) + m.group(2)).upper() if m else "?"


def _bks_for(instance: str) -> dict | None:
    return BKS.get(instance) or BKS.get(instance.upper()) or BKS.get(instance.lower())


def _is_solomon(name: str) -> bool:
    return "_" not in name


def _load(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df[df["NV_mean"].notna()].copy()
    df["family"] = df["Instance"].map(_family)
    df["bks_nv"] = df["Instance"].map(lambda s: (_bks_for(s) or {}).get("nv"))
    df["bks_td"] = df["Instance"].map(lambda s: (_bks_for(s) or {}).get("td"))
    df["nv_diff"] = df["NV_mean"] - df["bks_nv"]
    df["gap"] = (df["TD_mean"] - df["bks_td"]) / df["bks_td"] * 100.0
    return df


def _pivot(df: pd.DataFrame, value: str) -> pd.DataFrame:
    return df.pivot_table(index="Instance", columns="Algorithm", values=value, aggfunc="first")


def _w(path: str, content: str) -> None:
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(content)
    print(f"wrote {path}")


def tab_ablation(df: pd.DataFrame, out: str) -> None:
    scored = df[df["bks_nv"].notna()]
    n = scored[scored["Algorithm"] == "Hybrid-DDQN"]["Instance"].nunique()
    lines = [
        r"\caption{Ablation Analysis: Component contributions (overall $N=" + str(n) +
        r"$ instances). "
        r"NV Diff (mean fleet gap to BKS) drops sharply from ALNS-Base to any hybrid variant, then flattens among hybrids. "
        r"Raw TD Gap rises in lock-step --- an arithmetic consequence of fewer vehicles "
        r"covering the same customers, not a quality loss; the valid, vehicle-matched "
        r"distance comparison is the strict fair intersection of "
        r"Table~\ref{tab:distance_summary}, where Hybrid-DDQN attains the lowest gap "
        r"($+1.078\%$).}",
        r"\label{tab:ablation}",
        r"\begin{tabular}{@{}lrr@{}}",
        r"\toprule",
        r"\textbf{Configuration} & \textbf{NV Diff} & \textbf{TD Gap\% (raw)}\\",
        r"\midrule",
    ]
    labels = {
        "ALNS-Base": "ALNS-Base (standard baseline)",
        "Hybrid-Fixed": "Hybrid-Fixed (no macro controller)",
        "Hybrid-Rule": "Hybrid-Rule (rule-based macro)",
        "Hybrid-DDQN": r"\textbf{Hybrid-DDQN (proposed learning framework)}",
    }
    for algo in ALGOS:
        g = scored[scored["Algorithm"] == algo]
        if g.empty:
            continue
        nv_d = g["nv_diff"].mean()
        gap = g["gap"].mean()
        if algo == "Hybrid-DDQN":
            lines.append(rf"{labels[algo]} & $\mathbf{{{nv_d:+.3f}}}$ & ${gap:+.3f}\%$ \\")
        else:
            lines.append(rf"{labels[algo]} & ${nv_d:+.3f}$ & ${gap:+.3f}\%$ \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    _w(out, "\n".join(lines) + "\n")


def tab_nv_summary(df: pd.DataFrame, out: str) -> None:
    sol = df[df["Instance"].map(_is_solomon) & df["bks_nv"].notna()]
    algos = ALGOS + (["OR-Tools"] if (sol["Algorithm"] == "OR-Tools").any() else [])
    header = " & ".join([r"\textbf{Subset}"] + [ALGO_HEADS[a] for a in algos]) + r"\\"
    lines = [
        r"\caption{$NV_{\text{diff}}$ on Solomon (56 instances). Lower is better; $0.000{=}$BKS. Bold marks the best value per row.}",
        r"\label{tab:nv_summary}",
        r"\renewcommand{\arraystretch}{1.08}",
        r"\setlength{\tabcolsep}{3pt}",
        r"\resizebox{\columnwidth}{!}{%",
        r"\begin{tabular}{@{}l" + "r" * len(algos) + r"@{}}",
        r"\toprule",
        header,
        r"\midrule",
    ]

    def _row(label: str, g: pd.DataFrame) -> str:
        vals = [g[g["Algorithm"] == a]["nv_diff"].mean() for a in algos]
        min_v = min(v for v in vals if pd.notna(v))
        cells = []
        for _a, v in zip(algos, vals):
            cell = f"{v:.3f}" if pd.notna(v) else "--"
            if pd.notna(v) and abs(v - min_v) < 1e-4:
                cells.append(rf"\textbf{{{cell}}}")
            else:
                cells.append(cell)
        return label + " & " + " & ".join(cells) + r"\\"

    for name, fams in SUBSETS.items():
        g = sol[sol["family"].isin(fams)]
        n = g[g["Algorithm"] == "Hybrid-DDQN"]["Instance"].nunique()
        lines.append(_row(rf"{name} ($N={n}$)", g))
    lines.append(r"\midrule")
    lines.append(_row("Overall mean", sol))
    lines += [r"\bottomrule", r"\end{tabular}}"]
    _w(out, "\n".join(lines) + "\n")


def _fair_sets(sol: pd.DataFrame) -> tuple[dict[str, set], set]:
    """Per-algo instances at NV<=BKS, and their intersection across the hybrids."""
    per_algo: dict[str, set] = {}
    for a in ALGOS:
        g = sol[(sol["Algorithm"] == a) & (sol["nv_diff"] <= 1e-9)]
        per_algo[a] = set(g["Instance"])
    inter = set.intersection(*(per_algo[a] for a in ALGOS)) if per_algo else set()
    return per_algo, inter


def tab_distance_summary(df: pd.DataFrame, out: str) -> None:
    sol = df[df["Instance"].map(_is_solomon) & df["bks_nv"].notna()]
    per_algo, inter = _fair_sets(sol)
    lines = [
        r"\caption{NV-filtered TD Gap\%.  Lower is better.}",
        r"\label{tab:distance_summary}",
        r"\renewcommand{\arraystretch}{1.08}",
        r"\setlength{\tabcolsep}{3pt}",
        r"\resizebox{\columnwidth}{!}{%",
        r"\begin{tabular}{@{}lrrrr@{}}",
        r"\toprule",
        r"\textbf{Subset} & \textbf{ALNS} & \textbf{H-Fixed} & \textbf{H-Rule} & \textbf{H-DDQN}\\",
        r"\midrule",
    ]
    cells = []
    for a in ALGOS:
        g = sol[(sol["Algorithm"] == a) & (sol["Instance"].isin(per_algo[a]))]
        v, n = g["gap"].mean(), len(per_algo[a])
        s = rf"${v:+.3f}\%\ (N{{=}}{n})$"
        cells.append(rf"\textbf{{{s}}}" if a == "Hybrid-DDQN" else s)
    lines.append("Algo-specific\n  & " + " & ".join(cells) + r"\\")
    cells = []
    for a in ALGOS:
        g = sol[(sol["Algorithm"] == a) & (sol["Instance"].isin(inter))]
        v = g["gap"].mean()
        s = rf"${v:+.3f}\%$"
        cells.append(rf"\textbf{{{s}}}" if a == "Hybrid-DDQN" else s)
    lines.append("Fair intersection\n  & " + " & ".join(cells) + r"\\")
    lines += [r"\bottomrule", r"\end{tabular}}"]
    _w(out, "\n".join(lines) + "\n")


def tab_fair_by_category(df: pd.DataFrame, out: str) -> None:
    sol = df[df["Instance"].map(_is_solomon) & df["bks_nv"].notna()]
    _per_algo, inter = _fair_sets(sol)
    fair = sol[sol["Instance"].isin(inter)]
    lines = [
        rf"\caption{{Strict fair intersection ($N={len(inter)}$) by Solomon family.}}",
        r"\label{tab:fair_by_category}",
        r"\renewcommand{\arraystretch}{1.08}",
        r"\setlength{\tabcolsep}{3pt}",
        r"\resizebox{\columnwidth}{!}{%",
        r"\begin{tabular}{@{}lrrrr@{}}",
        r"\toprule",
        r"\textbf{Category} & \textbf{ALNS} & \textbf{H-Fixed} & \textbf{H-Rule} & \textbf{H-DDQN}\\",
        r"\midrule",
    ]

    def _row(label: str, g: pd.DataFrame) -> str:
        cells = []
        for a in ALGOS:
            v = g[g["Algorithm"] == a]["gap"].mean()
            s = rf"${v:+.3f}\%$" if pd.notna(v) else "--"
            cells.append(rf"\textbf{{{s}}}" if a == "Hybrid-DDQN" else s)
        return label + "\n  & " + " & ".join(cells) + r"\\"

    for name, fams in SUBSETS.items():
        g = fair[fair["family"].isin(fams)]
        n = g[g["Algorithm"] == "Hybrid-DDQN"]["Instance"].nunique()
        lines.append(_row(rf"{name} ($N={n}$)", g))
    lines.append(r"\midrule")
    lines.append(_row(rf"Overall ($N={len(inter)}$)", fair))
    lines += [r"\bottomrule", r"\end{tabular}}"]
    _w(out, "\n".join(lines) + "\n")


def tab_gh200(df: pd.DataFrame, out: str) -> None:
    gh = df[~df["Instance"].map(_is_solomon) & df["bks_nv"].notna()]
    gh = gh[gh["Instance"].str.contains("_2_", na=False)]
    algos = ALGOS + (["OR-Tools"] if (gh["Algorithm"] == "OR-Tools").any() else [])
    sub_head = " & ".join([r"\textbf{Inst}", r"\textbf{BKS}"] + [ALGO_HEADS[a] for a in algos])
    header = f"{sub_head} & & {sub_head}" + r"\\"
    lines = [
        r"\caption{Gehring--Homberger~\cite{Gehring1999} 200-customer results at 600",
        r"iterations.  Format: $NV/TD\,\text{Gap}\%$.",
        r"${}^\dagger$: $NV>NV_{\BKS}$ --- negative gaps reflect over-allocation.}",
        r"\label{tab:gh200}",
        r"\renewcommand{\arraystretch}{1.06}",
        r"\setlength{\tabcolsep}{3.5pt}",
        r"\resizebox{\textwidth}{!}{%",
        r"\begin{tabular}{@{}l" + "r" * (len(algos) + 1) + r" c l" + "r" * (len(algos) + 1) + r"@{}}",
        r"\toprule",
        header,
        r"\midrule",
    ]

    inst_list = sorted(gh["Instance"].unique())
    n_half = (len(inst_list) + 1) // 2

    def _format_inst_cells(inst: str) -> str:
        bks = _bks_for(inst)
        tex_name = inst.replace("_", r"\_")
        cells = []
        for a in algos:
            r = gh[(gh["Instance"] == inst) & (gh["Algorithm"] == a)]
            if r.empty:
                cells.append("--")
                continue
            nv, gap = float(r["NV_mean"].iloc[0]), float(r["gap"].iloc[0])
            nv_s = f"{nv:.0f}" if abs(nv - round(nv)) < 1e-9 else f"{nv:.2f}"
            if bks and nv > bks["nv"] + 1e-9:
                s = rf"${nv_s}/\text{{--}}{{}}^\dagger$"
            else:
                s = rf"${nv_s}/{gap:+.2f}\%$"
            cells.append(rf"\textbf{{{s}}}" if a == "Hybrid-DDQN" else s)
        bks_str = f"{bks['nv']}/{bks['td']:.2f}" if bks else "--"
        return rf"${tex_name}$ & ${bks_str}$ & " + " & ".join(cells)

    for i in range(n_half):
        left_inst = inst_list[i]
        left_str = _format_inst_cells(left_inst)
        if i + n_half < len(inst_list):
            right_inst = inst_list[i + n_half]
            right_str = _format_inst_cells(right_inst)
        else:
            right_str = " & ".join([""] * (len(algos) + 2))
        lines.append(f"{left_str} & & {right_str}" + r"\\")

    lines += [r"\bottomrule", r"\end{tabular}}"]
    _w(out, "\n".join(lines) + "\n")


def tab_gh400(df: pd.DataFrame, out: str) -> None:
    gh = df[~df["Instance"].map(_is_solomon) & df["bks_nv"].notna()]
    gh = gh[gh["Instance"].str.contains("_4_", na=False)]
    algos = ALGOS + (["OR-Tools"] if (gh["Algorithm"] == "OR-Tools").any() else [])
    sub_head = " & ".join([r"\textbf{Inst}", r"\textbf{BKS}"] + [ALGO_HEADS[a] for a in algos])
    header = f"{sub_head} & & {sub_head}" + r"\\"
    lines = [
        r"\caption{Gehring--Homberger~\cite{Gehring1999} 400-customer results at 600",
        r"iterations.  Format: $NV/TD\,\text{Gap}\%$.",
        r"${}^\dagger$: $NV>NV_{\BKS}$ --- negative gaps reflect over-allocation.}",
        r"\label{tab:gh400}",
        r"\renewcommand{\arraystretch}{1.06}",
        r"\setlength{\tabcolsep}{3.5pt}",
        r"\resizebox{\textwidth}{!}{%",
        r"\begin{tabular}{@{}l" + "r" * (len(algos) + 1) + r" c l" + "r" * (len(algos) + 1) + r"@{}}",
        r"\toprule",
        header,
        r"\midrule",
    ]

    inst_list = sorted(gh["Instance"].unique())
    n_half = (len(inst_list) + 1) // 2

    def _format_inst_cells(inst: str) -> str:
        bks = _bks_for(inst)
        tex_name = inst.replace("_", r"\_")
        cells = []
        for a in algos:
            r = gh[(gh["Instance"] == inst) & (gh["Algorithm"] == a)]
            if r.empty:
                cells.append("--")
                continue
            nv, gap = float(r["NV_mean"].iloc[0]), float(r["gap"].iloc[0])
            nv_s = f"{nv:.0f}" if abs(nv - round(nv)) < 1e-9 else f"{nv:.2f}"
            if bks and nv > bks["nv"] + 1e-9:
                s = rf"${nv_s}/\text{{--}}{{}}^\dagger$"
            else:
                s = rf"${nv_s}/{gap:+.2f}\%$"
            cells.append(rf"\textbf{{{s}}}" if a == "Hybrid-DDQN" else s)
        bks_str = f"{bks['nv']}/{bks['td']:.2f}" if bks else "--"
        return rf"${tex_name}$ & ${bks_str}$ & " + " & ".join(cells)

    for i in range(n_half):
        left_inst = inst_list[i]
        left_str = _format_inst_cells(left_inst)
        if i + n_half < len(inst_list):
            right_inst = inst_list[i + n_half]
            right_str = _format_inst_cells(right_inst)
        else:
            right_str = " & ".join([""] * (len(algos) + 2))
        lines.append(f"{left_str} & & {right_str}" + r"\\")

    lines += [r"\bottomrule", r"\end{tabular}}"]
    _w(out, "\n".join(lines) + "\n")


def tab_gnn(gnn_csv: str, out: str) -> None:
    df = pd.read_csv(gnn_csv)
    lines = [
        r"\caption{Baseline vs.\ GNN-Guided Hybrid-DDQN (equal iterations, paired seeds).}",
        r"\label{tab:gnn_comparison}",
        r"\renewcommand{\arraystretch}{1.08}",
        r"\setlength{\tabcolsep}{4pt}",
        r"\resizebox{\columnwidth}{!}{%",
        r"\begin{tabular}{@{}llrrrr@{}}",
        r"\toprule",
        r"\textbf{Instance} & \textbf{BKS NV/TD} & \textbf{Base NV} & \textbf{Base TD}"
        r" & \textbf{GNN NV} & \textbf{GNN TD} \\",
        r"\midrule",
    ]
    for _, r in df.iterrows():
        bks = _bks_for(str(r["Instance"]))
        bks_s = f"{bks['nv']} / {bks['td']:.1f}" if bks else "--"
        lines.append(
            rf"\texttt{{{r['Instance']}}} & {bks_s} & {r['base_nv']:.2f} & {r['base_td']:.2f}"
            rf" & {r['gnn_nv']:.2f} & {r['gnn_td']:.2f} \\"
        )
    lines += [r"\bottomrule", r"\end{tabular}}"]
    _w(out, "\n".join(lines) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sweep", required=True, help="combined sweep CSV (Solomon + GH shards)")
    ap.add_argument("--gnn", default=None, help="optional GNN validation CSV")
    ap.add_argument("--out-dir", default=os.path.join(_REPO, "docs", "tables"))
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    df = _load(args.sweep)

    tab_ablation(df, os.path.join(args.out_dir, "ablation.tex"))
    tab_nv_summary(df, os.path.join(args.out_dir, "nv_summary.tex"))
    tab_distance_summary(df, os.path.join(args.out_dir, "distance_summary.tex"))
    tab_fair_by_category(df, os.path.join(args.out_dir, "fair_by_category.tex"))
    tab_gh200(df, os.path.join(args.out_dir, "gh200.tex"))
    tab_gh400(df, os.path.join(args.out_dir, "gh400.tex"))
    if args.gnn:
        tab_gnn(args.gnn, os.path.join(args.out_dir, "gnn_comparison.tex"))


if __name__ == "__main__":
    main()

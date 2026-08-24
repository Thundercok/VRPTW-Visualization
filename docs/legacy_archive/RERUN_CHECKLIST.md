# Re-run checklist for `paper.tex` — V2 upgrade (2026-07-24)

> **STATUS 2026-07-26: COMPLETE.** All 8 shards ran clean (984 rows, 0 null,
> 0 sleep-corrupted after the keep-awake fix). Tables regenerated from
> `results/rerun_combined.csv` via `scripts/make_paper_tables.py` into
> `docs/tables/*.tex` and `\input`-ed by `paper.tex`. All prose numbers, the
> abstract, the GNN section (rewritten to a scalability result + honest negative
> guidance ablation), the ablation/distance narrative, and the OR-Tools/GH prose
> were updated to the new data. `paper.tex` compiles clean (MiKTeX, 10 pages,
> 0 undefined refs). The headline change: NV improved (DDQN $+0.089$ vs the old
> $+0.143$), distance-at-matched-NV is best-in-class within the sweep ($+0.575\%$
> fair intersection) but the vehicle-distance trade means it is not a Pareto win
> over the previous version — reported honestly.

The V2 solver changes below alter search trajectories, so **every measured
number in the paper must be regenerated**. The regeneration sweeps are defined
in `scripts/run_rerun_sweeps.sh` (two protocols):

| Protocol | Shards | Output |
|---|---|---|
| Iteration-bounded (`--no-time-limit`) | Solomon ×3, H200, H400 | `results/rerun_iters/` |
| Time-bounded (anytime, 0.6 s × n) | H600, H800, H1000 | `results/rerun_time/` |

The split is deliberate: the paper's existing tables cover Solomon/H200/H400,
so that branch stays iteration-bounded for continuity with
`results/ultimate-publication-suite/`; H600–1000 are new scalability results
where iso-time against OR-Tools is the natural protocol.

## What changed in the solver (V2)

### Behaviour-preserving (verified bit-identical against regenerated goldens)

| Change | Where |
|---|---|
| Incremental `_PlanCache` — per-route data reused via content keys; Python sets/dict replaced by arrays | `local_search.py` |
| Exact pair-scan memoization (`scan_memo`) — kernel results reused across moves that don't touch the pair. Chosen over don't-look bits deliberately: DLB changes trajectories, this doesn't | `local_search.py` |
| Greedy repair via matrix + column refresh (same pattern as `_regret`) | `operators.py::_sequential_cheapest_insert` |
| RL structural state features cached per `cur` identity | `solvers.py::_StructuralFeatureCache` |
| PER priorities\*\*alpha maintained incrementally (sum-tree rejected: not bit-identical) | `rl.py` |

### Trajectory-changing (measured on the 90-run paired A/B, 400 iters)

| Change | Evidence |
|---|---|
| **B1 Guided Ejection Search** (`_guided_ejection_search`): LIFO ejection pool, per-customer penalty counters, min-Σp ejections, perturbation phase; fires where the beam search gave up | RC105 reached the **BKS floor of 13 vehicles on 4/5 seeds** (baseline: 0/5). Overall NV 5 better / 1 worse / 84 tie, mean 13.400→13.356, Wilcoxon p=0.102 — while total wall time *dropped* 594→545 s |
| **B2 SREX crossover**: old crossover produced feasible but fragmented offspring — **0/600 measured offspring could pass the `alt.nv < best.nv` gate**, so it never contributed. SREX: 491/600 pass | `rl.py::EliteArchive.crossover` |
| **B3 surgical SP column rescue + MILP memo**: plain `[:400]` truncation could drop every column of a customer (`row_sums == 0` → recombination dead). Rescue swaps in coverage columns only in that case. A broader reshuffle (≥3 columns/customer) was tried and **reverted — it cost a vehicle on rc1_2_1** | `pool.py::_select_milp_columns` |
| **A5 deadline-aware tail**: every tail phase now checks `_out_of_time()`; `td_converge_polish` takes a deadline. Measured: solve() finishes at **29.76 s on a 30 s budget (−0.8%)** vs +8% overrun before. Benchmark `Time_s` additionally includes ~2 s solver construction outside `solve()` — state this when claiming iso-time | `solvers.py`, `local_search.py` |

### Speed (paired A/B, bit-identical set only)

Baseline HEAD (already 2.62× over the original) → +Tier A: **594.1 s → 500.4 s
(~1.16× excluding the cold-JIT first run; 75/90 runs bit-identical)**. With GES
enabled the suite still runs faster than baseline (545.2 s) while winning
vehicles.

### Iso-time verdict for GES (the gate that killed the sawtooth)

GES consumed ~9% extra time and bought −0.044 mean NV. The measured
iterations-route to the same budget (+100% time → −0.111 NV) prices 9% of time
at ≈ −0.010 NV. **GES buys ~4× more NV per unit time than spending the same
time on more ALNS iterations.** It passes.

## Protocol notes for the paper

- **OR-Tools budget is iso-time, not a flat 120 s.** `benchmark.py` sets the
  OR-Tools time limit to `max(30, 0.95 × mean DDQN Time_s)`, overriding the
  `--ortools-time-limit` CLI value. On easy instances (DDQN ~35–90 s) OR-Tools
  runs well under 120 s; on hard ones it runs longer. The `120\,s` label in the
  tables should be corrected to state that OR-Tools is matched to DDQN wall time.
  This behaviour is unchanged from the previous suite, so old/new remain
  comparable — but any single anomalous DDQN run inflates the OR-Tools budget
  for that instance (observed: an overnight machine-sleep on R211 pushed its
  OR-Tools budget to ~3 h). The `--time-limit` fuses below bound this.
- **Safety fuses on the sweep** (`scripts/run_rerun_sweeps.sh`): solomon-wide
  1200 s, H200 1500 s, H400 3600 s — each ≥2.5× the healthy run time, so all
  iterations complete before they fire; they only cap runaways. After the sweep,
  verify no run's `Time_s` sits near its fuse; flag any that did.
- Anytime budget: deadline applies to `solve()`; construction (~2 s, torch
  init) is outside it. Deadline adherence after A5: within ±2%.
- Thread pinning: `run_benchmark.py` pins NUMBA/OMP/MKL to 1 thread per worker
  when parallel (was ~4× oversubscribed → noisy `Time_s`).
- GNN guidance stays **off** in production sweeps (validated: no quality gain,
  gap +0.22 pp, p=0.683). Report GNN separately as a scalability result
  (1517 MB → 1.3 MB, 3.96 s → 0.049 s at n=1000).

## Regeneration procedure

```bash
# 1. Full two-protocol sweep (~16 h wall on 12 logical cores)
bash scripts/run_rerun_sweeps.sh

# 2. Cross-check against the previous suite (UNPAIRED — sanity only)
python scripts/compare_sweeps.py <old.csv> <new.csv> --algorithms Hybrid-DDQN

# 3. Regenerate the six LaTeX tables
python scripts/make_paper_tables.py --sweep <combined_new.csv> --out-dir docs/tables

# 4. Recompile (MiKTeX 25.12 now installed at
#    C:\Users\han\AppData\Local\Programs\MiKTeX\miktex\bin\x64\pdflatex.exe,
#    user scope, auto-install on). Run 3 passes to resolve \cite/\Cref refs:
pdflatex -interaction=nonstopmode docs/paper.tex   # x3
```
The current `paper.tex` compiles clean to a 9-page PDF (all citations resolve,
including the new GES/SREX refs). PDF recompilation is no longer a blocker.

## Still stale in `paper.tex` until the sweep finishes

| Location | Claim |
|---|---|
| Abstract, ~line 80 | `+0.139` NV inflation, `13.3%` vs ALNS-Base, `92.7%` vs OR-Tools, `5.8%` TD gain |
| Setup, ~line 640 | Runtimes `31.5/47.1/58.9` s; `861.3 s -> 60.3 s (14x)` Numba claim |
| Tables ~672, 694, 729, 755, 780, 862 | All result tables |
| Throughout | All Wilcoxon p-values |

When quoting improvement *deltas*, cite the paired A/B numbers above — the
old-vs-new sweep comparison is unpaired (different seeds/machine) and is a
sanity check, not evidence.

## Rejected in V2 (do not re-add without an iso-time win)

| Idea | Why rejected |
|---|---|
| Don't-look bits | Trajectory-changing; exact memoization achieves the same reuse bit-identically |
| Sum-tree PER | Different RNG stream → not bit-identical; incremental `_pri_alpha` captures the dominant saving |
| SP column reshuffle (≥3 cols/customer) | Cost a vehicle on rc1_2_1 in the paired A/B |
| (V1 list) kNN candidate filter, FTS slack bonus, SISR, sawtooth schedule | See git history — each lost on measurement |

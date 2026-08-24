# Paper 1 Handoff — VRPTW 3-Tier MARL

> **Status:** All technical work DONE. Ready for paper writing.  
> **Date:** 2026-08-17  
> **Repo:** `VRPTW-Research-Optimization`

---

## What Was Built

A 3-tier DDQN+ALNS hybrid solver for VRPTW with:
- **Tier 1 — Macro Controller**: Dueling DDQN selects among 7 search modes (default, intensify, diversify, tw_rescue, pool_recombine, route_reduce, infeasible_descent)
- **Tier 2 — Operator Selection**: Thompson Bandit over 13 destroy × 5 repair operators, per-mode
- **Tier 3 — Set Covering Supercharger**: HiGHS MILP recombination over RoutePool of ~1000 candidate routes

## Benchmark Results (Locked)

**176 instances × 6 algorithms × 5 seeds = 5,280 runs, 0 failures.**  
Data: `results/ultimate-publication-suite/combined_clean.csv`

### Headline Numbers (Wilcoxon over N=176 instance-means)
- **NV**: Hybrid-DDQN vs ALNS-Base, p = 3.44×10⁻¹³
- **TD**: Hybrid-DDQN vs ALNS-Base, p = 5.02×10⁻¹⁴
- **Fair TD Gap** (N=33 matched-NV Solomon): DDQN +1.078% vs ALNS +2.151%

### Ablation (N=176)
| Config | NV Diff to BKS | Raw TD Gap% |
|---|:---:|:---:|
| ALNS-Base | +0.535 | +9.483% |
| Hybrid-Fixed | +0.230 | +4.930% |
| Hybrid-Rule | +0.232 | +4.880% |
| **Hybrid-DDQN** | **+0.231** | +4.995% |

> NV Diff drops monotonically. Raw TD Gap rises because fewer vehicles cover the same customers — use Fair Intersection table for valid TD comparison.

---

## Scope Boundaries

| | Paper 1 | Paper 2 |
|---|---|---|
| **Algorithms** | ALNS-Base, Hybrid-Fixed, Hybrid-Rule, Hybrid-DDQN, OR-Tools | GNN-Hybrid-DDQN |
| **Tables** | `docs/tables/*.tex` (GNN excluded) | Separate |

---

## Verified Findings & Agreed Framing

### 1. Long-Horizon vs Short-Horizon Split
- On C2/R2/RC2: Both Hybrid-Rule and DDQN hit same NV floor. Rule-based intensification yields 1.0–3.1% lower TD.
- On C1/R1/RC1: DDQN retains edge in both NV and TD.
- **Mechanism tested and rejected**: MODE_TW_RESCUE triggers <6% in ALL categories. cap_util is actually higher in Short-Horizon (smaller trucks pack tighter). The driver of this split is an **open question**.
- Script: `scripts/diagnose_mode_distribution.py`

### 2. Baseline Provenance
- **Hybrid-Fixed** (`solvers.py:2365`): Disables DDQN, only MODE_DEFAULT + route_reduce trigger. Fixed ModeSpec biases.
- **Hybrid-Rule** (`solvers.py:2407`): Hand-crafted if/elif tree with domain-expert thresholds (`fleet_fill>=0.66`, `progress<0.45`, `slack<0.16`, `tw_tight_frac>=0.18`). Frame as *intuitive expert baseline*, not tuned-optimal.
- **OR-Tools** (`solvers.py:2445`): Google OR-Tools RoutingModel, GLS, iso-time budget = max(30s, 95% of DDQN runtime).

### 3. Threats to Validity: Runtime Asymmetry (H400)
- ALNS-Base: 15.6s (premature early-stop after ~300 iters, stagnates in local optima)
- Hybrid-DDQN: 178.2s (sustains productive search via MIP recombine + mode switching)
- Frame as: *quality maximizer, not search accelerator*. Saving 1 vehicle justifies 3 min offline compute.

### 4. Slot A Eviction: Verified Non-Issue
- Stress test >1400 trims/run: absolute tie between `cost/len` and `(cost/len, -len)`. Kept simple.

### 5. Cold-Start Integrity
- All runs use `init=None` (greedy cold-start). No cross-seed archive leakage.
- TD comparisons excluded when NV doesn't match BKS (marked with `†`).

---

## File Map

### Data & Tables
- `results/ultimate-publication-suite/combined_clean.csv` — master data (1,056 rows)
- `docs/tables/ablation.tex` — component ablation (N=176)
- `docs/tables/nv_summary.tex` — Solomon NV breakdown by category
- `docs/tables/distance_summary.tex` — fair vehicle-matched TD comparison
- `docs/tables/fair_by_category.tex` — category-wise fair intersection
- `docs/tables/gh200.tex` — full 60-instance Homberger-200
- `docs/tables/gh400.tex` — full 60-instance Homberger-400

### Core Code
- `src/vrptw/solvers.py` — all solver classes (ALNS, DDQN, Fixed, Rule, OR-Tools)
- `src/vrptw/pool.py` — RoutePool + HiGHS set covering
- `src/vrptw/numba_kernels.py` — Numba O(1) forward slack + pruned LS
- `src/vrptw/config.py` — MODES, ModeSpec definitions, all constants
- `src/vrptw/dynamic_insertion.py` — Pillar 3: online insertion engine

### Scripts
- `scripts/make_paper_tables.py` — regenerates all LaTeX tables from CSV
- `scripts/benchmark.py` — benchmark runner & Wilcoxon analysis
- `scripts/diagnose_mode_distribution.py` — mode trace & cap_util diagnostic
- `docs/run_benchmark.py` — full benchmark launcher with CLI args

### Tests
- 66/66 pass: `uv run pytest --ignore=tests/e2e -v` (62.38s)
- Golden regression: `tests/test_regression_golden.py` (R101, RC207, C101, r1_2_1 × 2 algos × 2 seeds)

---

## Instrumentation Added (Behaviorally Neutral)

`self.mode_trace = Counter()` added to `HybridDDQNSolver.__init__` and `solve()`, incremented in `_select_action()` of all 3 solver variants. Golden regression tests confirmed unchanged output.

---

## What's Left (Paper Writing)

1. **Results & Discussion section** using agreed framing above
2. **Threats to Validity subsection** (runtime asymmetry text drafted in conversation)
3. **Method section**: Hybrid-Rule threshold provenance paragraph (expert-designed, not grid-searched)
4. **Figures**: Convergence curves, NV boxplots (not yet generated)
5. **Paper 2 prep**: GNN experiments via `scripts/validate_gnn.py` (separate scope)

---

## Rules (from .agents/AGENTS.md)

1. Never report warm-started results as standalone performance
2. Exclude TD comparisons when NV mismatched (use `†` marker)
3. Confirm iteration budgets match across parallel workers
4. Ponytail Minimalist Engineering — minimal code, reuse existing
5. All code must pass `uv run ruff check`

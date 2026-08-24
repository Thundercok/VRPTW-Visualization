# IEEE Access Paper Writing Workspace & Collaboration Plan

## 1. Project Directory Architecture

```
docs/
├── main.tex                       # Master LaTeX document (IEEE Access format)
├── ieeeaccess.cls                 # Patched & verified IEEE Access class file
├── IEEEtran.cls                   # Base IEEEtran class dependency
├── Logo.png, bullet.png...        # Journal branding assets
├── overleaf_ieee_access.zip       # Ready-to-upload ZIP for Overleaf
├── sections/                      # Modular writing sections
│   ├── 01_introduction.tex        # Introduction & Contributions
│   ├── 02_related_work.tex        # Literature review
│   ├── 03_formulation.tex         # VRPTW MILP & Lexicographic formulation
│   ├── 04_methodology.tex         # Hybrid DDQN-ALNS architecture & MDP
│   ├── 05_experiments.tex         # Benchmark results & ablation studies
│   ├── 06_discussion.tex          # Search dynamics, quality trade-offs, limitations
│   ├── 07_conclusion.tex          # Conclusion & future extensions
│   └── 99_bibliography.tex        # References
└── figures/                       # Convergence, route visualizer, and boxplot graphics
```

---

## 2. Collaboration Protocol: Step-by-Step

| Step | Section | Content Focus | Status |
| :--- | :--- | :--- | :--- |
| **0** | **Environment & Setup** | Clean IEEE Access class, modular structure, Overleaf bundle | **Done** |
| **1** | **Data & Benchmark Verification** | Strict cold-start runs on Solomon-100, Homberger-200, 400 | **Pending Benchmark** |
| **2** | **Section 3: Formulation** | Mathematical rigor: MILP, variables, time propagation, Lexicographic obj | **Ready** |
| **3** | **Section 4: Methodology** | Dual-agent MDP, features, LAC, Welford normalizer, training curriculum | **Ready** |
| **4** | **Section 5: Experiments** | Clean tables, fair subset TD gap comparisons, Wilcoxon statistical tests | **Waiting on Data** |
| **5** | **Section 1 & 2: Intro & Related** | Framing the narrative, contributions, positioning against literature | **Ready** |
| **6** | **Section 6 & 7: Discussion & Conclusion** | Deep insights, why state-conditioning works, limitations, future work | **Ready** |
| **7** | **Final Polish & Overleaf Build** | Compile verification, grammar, spacing, Overleaf export | **Pending** |

---

## 3. Academic Integrity & Benchmark Ground Rules
1. **Strict Independent Cold-Starts**: Standalone solver evaluations start from `build_greedy` in a fresh session without warm-start archive cache cross-talk.
2. **Fair Subset Distance Comparisons**: Distance comparisons are computed strictly on subsets where vehicle counts are matched ($NV_{\text{DDQN}} = NV_{\text{ALNS}}$).
3. **Reproducibility**: All experiments seed-locked (5 seeds) with statistical significance reported via two-tailed Wilcoxon signed-rank tests.

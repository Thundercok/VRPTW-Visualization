# Statistical Rigor Report (Two-Tailed Wilcoxon & Exact Sign Test)
Significance Thresholds: Nominal $\alpha = 0.05$, Bonferroni $\alpha_{\text{adj}} = 0.0167$ (m=3 families).

## Comparison: Hybrid-DDQN vs ALNS-Base (N=74 instances)
### 1. Fleet Size Reduction (NV):
- Mean NV: ALNS-Base = 9.85 vs Hybrid-DDQN = 9.84
- Record: 10 Wins / 54 Ties / 10 Losses
- Wilcoxon signed-rank: $W = 636.0$, $p = 9.5044e-01$ (Not Significant)
- Exact Sign test: $p = 1.0000e+00

### 2. Vehicle-Matched Distance Gap (TD):
- Matched Instances: 54/74 (73.0%)
- Mean TD Gap on Matched: -1.40%
- Record on Matched: 39 Wins / 12 Ties / 3 Losses
- Wilcoxon signed-rank: $W = 55.0$, $p = 1.9355e-08$ (Significant)
- Exact Sign test: $p = 5.6316e-09

---------------------------------------------------------------------------
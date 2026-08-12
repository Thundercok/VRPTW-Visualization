# CLAUDE.md — Handoff & Technical Briefing Guide for VRPTW Optimizer

This guide details the exact state of the codebase, recent benchmark findings, architectural design, and open improvement vectors for future development.

---

## 1. Project Overview & Current Baseline Status

We have implemented and empirically verified a SOTA Hybrid RL-ALNS VRPTW solver under **strict independent cold-starts** (cleared archive directories, zero cross-seeding).

### Verified Clean V7 Benchmark Findings (116 Instances, 580 Runs):
* **Feasibility Mandate**: **100% Feasible across all 580 runs (`Violations = 0`)**.
* **Homberger-200 Scale Breakout ($N=60$ instances)**:
  - `ALNS-Base`: Mean Fleet (NV) = **11.87**, Mean Distance (TD) = **3009.91 km**
  - `Hybrid-DDQN` (3-Tier MARL): Mean Fleet = **11.58**, Mean Distance = **2844.73 km** (**$-165.18$ km drop**, Wilcoxon $p = 0.00000$)
  - `GNN-Hybrid-DDQN` (InfoNCE Trained GNN): Mean Fleet = **11.58**, Mean Distance = **2870.82 km** (**$-139.09$ km drop**, Wilcoxon $p = 0.00027$)
* **Solomon-100 Benchmark ($N=56$ instances)**:
  - `ALNS-Base`: Mean Fleet = **7.54**, Mean Distance = **1017.31 km**
  - `Hybrid-DDQN`: Mean Fleet = **7.34**, Mean Distance = **1017.82 km** (Wilcoxon $p = 0.00127$)
  - `GNN-Hybrid-DDQN`: Mean Fleet = **7.32**, Mean Distance = **1020.14 km** (Wilcoxon $p = 0.00088$)
* **Tight Time-Window Breakthrough (RC1 & R1 Families)**:
  - RC1 Fleet: ALNS-Base = 15.87 $\to$ **15.11 vehicles** (Hybrid-DDQN & GNN-Hybrid-DDQN).
  - RC1 Distance: ALNS-Base = 2597.51 km $\to$ **2450.75 km** (Hybrid-DDQN, **$-146.76$ km drop**).

---

## 2. Core Architectural Components

1. **Instance-Adaptive Reheat Scheduler** ([`src/vrptw/solvers.py`](file:///Users/thundercock2/Documents/Github/VRPTW-Research-Optimization/VRPTW-Research-Optimization/src/vrptw/solvers.py)):
   - Dynamically scales temperature boost $\alpha_{\text{reheat}} = 1.5 + 2.5 \times (1 - \text{tw\_tight\_frac})$.
   - Protects tight TW feasibility on R1/RC1 while granting R2/RC2 wide TW instances $4.0\times$ exploration.
2. **3-Tier Hierarchical MARL (`MILPColumnController` DDQN-3)** ([`src/vrptw/rl.py`](file:///Users/thundercock2/Documents/Github/VRPTW-Research-Optimization/VRPTW-Research-Optimization/src/vrptw/rl.py)):
   - Tier 1: Destroy/Repair Operator Selector.
   - Tier 2: Learned Acceptance Criterion.
   - Tier 3: DDQN-3 Set-Partitioning Column Generation Manager (dynamically selects vehicle penalty $P_{\text{vehicle}}$ and MILP time limit).
3. **Contrastive Graph RL Edge Predictor (`GNNEdgePredictor`)** ([`src/vrptw/gnn.py`](file:///Users/thundercock2/Documents/Github/VRPTW-Research-Optimization/VRPTW-Research-Optimization/src/vrptw/gnn.py)):
   - Bilinear edge predictor with contrastive projection heads ($\mathbb{R}^H \to \mathbb{R}^{32}$).
   - Trained via Joint BCE + InfoNCE Loss ($\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{BCE}} + 0.25 \mathcal{L}_{\text{InfoNCE}}$) on elite positive edges vs constraint-violating negative edges.

---

## 3. Potential Improvement Vectors for Next Steps

* **Vector 1: Dynamic Soft-Edge Masking (Logit Bias vs Hard Pruning)**:
  - Currently, GNN applies a hard threshold to candidate edges, which can occasionally prune rare, long-range shortcut edges needed for distance minimization.
  - *Fix*: Replace hard pruning with continuous logit biasing $\beta_{\text{gnn}} \cdot \text{sigmoid}(e_{ij})$ in candidate selection.
* **Vector 2: Integrated Joint DDQN-3 + GNN Edge Weight Controller**:
  - Let DDQN-3 (Tier 3 MARL) output a joint continuous/discrete action controlling both MILP Column Generation penalty ($P_{\text{vehicle}}$) AND GNN edge heat bias weight $\beta_{\text{gnn}}(t)$ based on search stagnation.

---

## 4. Command Reference

### Environment & Test Suite
- **Run Pytest Regression & Golden Suite**:
  ```bash
  PYTHONPATH=src python3 -m pytest tests/
  ```
- **Re-Train InfoNCE GNN Model**:
  ```bash
  PYTHONPATH=src python3 -m vrptw.train_gnn 100
  ```
- **Execute Clean V7 Benchmark Sweep**:
  ```bash
  python3 scratch/run_clean_v7_marl_sweep.py
  ```
- **Analyze Clean V7 Results**:
  ```bash
  PYTHONPATH=src python3 scratch/analyze_clean_v7_marl.py
  ```

---

## 5. Engineering Principles (Ponytail Rules)

1. **Academic Integrity**: Standalone solver performance must be generated under strict cold-starts (empty archive).
2. **No Data Cross-Splicing**: Never mix NV from one algorithm with TD from another in summary tables or prose.
3. **Verified Feasibility**: Always log and report `Violations` column (`Violations = 0` mandatory).

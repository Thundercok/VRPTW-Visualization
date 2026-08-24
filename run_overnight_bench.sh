#!/usr/bin/env bash
# ==============================================================================
# Overnight Robust Benchmark Runner for VRPTW Neural Hybrid Optimizer
# ==============================================================================
set -uo pipefail

mkdir -p logs results/option_a_ablation

SUMMARY_LOG="logs/_summary.log"
echo "======================================================================" | tee "$SUMMARY_LOG"
echo "  VRPTW OVERNIGHT BENCHMARK SUITE - STARTED AT $(date)" | tee -a "$SUMMARY_LOG"
echo "======================================================================" | tee -a "$SUMMARY_LOG"

# ------------------------------------------------------------------------------
# STAGE 0: Sanity Check (Abort early if required scripts are missing)
# ------------------------------------------------------------------------------
echo -n "[Stage 0] Verifying script prerequisites... " | tee -a "$SUMMARY_LOG"

REQUIRED_FILES=(
    "scratch/wilcoxon_significance.py"
    "scratch/sweep_diversity_impact.py"
    "scratch/sweep_routepool_impact.py"
    "results/option_a_ablation/master_ablation_clean.csv"
)

MISSING=0
for f in "${REQUIRED_FILES[@]}"; do
    if [ ! -f "$f" ]; then
        echo -e "\n  ❌ MISSING REQUIRED FILE: $f" | tee -a "$SUMMARY_LOG"
        MISSING=1
    fi
done

if [ "$MISSING" -eq 1 ]; then
    echo "❌ Stage 0 FAILED: Missing prerequisite files. Aborting benchmark." | tee -a "$SUMMARY_LOG"
    exit 1
fi
echo "✓ PASSED (All files verified)." | tee -a "$SUMMARY_LOG"

# ------------------------------------------------------------------------------
# STAGE 1: Wilcoxon Significance Tests (Fast, uses existing verified CSV)
# ------------------------------------------------------------------------------
echo -e "\n----------------------------------------------------------------------" | tee -a "$SUMMARY_LOG"
echo "[Stage 1] Running Wilcoxon Significance Hypothesis Testing..." | tee -a "$SUMMARY_LOG"
STAGE1_LOG="logs/overnight_stage1_wilcoxon.log"

if uv run python scratch/wilcoxon_significance.py 2>&1 | tee "$STAGE1_LOG"; then
    echo "✓ Stage 1 COMPLETED successfully. Log: $STAGE1_LOG" | tee -a "$SUMMARY_LOG"
else
    echo "❌ Stage 1 ENCOUNTERED ERRORS. Log: $STAGE1_LOG" | tee -a "$SUMMARY_LOG"
fi

# ------------------------------------------------------------------------------
# STAGE 2: EliteArchive Edge-Jaccard Diversity Guard Sweep
# ------------------------------------------------------------------------------
echo -e "\n----------------------------------------------------------------------" | tee -a "$SUMMARY_LOG"
echo "[Stage 2] Running EliteArchive Diversity Guard Ablation Sweep..." | tee -a "$SUMMARY_LOG"
STAGE2_LOG="logs/overnight_stage2_diversity.log"

if uv run python scratch/sweep_diversity_impact.py 2>&1 | tee "$STAGE2_LOG"; then
    echo "✓ Stage 2 COMPLETED successfully. Log: $STAGE2_LOG" | tee -a "$SUMMARY_LOG"
else
    echo "❌ Stage 2 ENCOUNTERED ERRORS. Log: $STAGE2_LOG" | tee -a "$SUMMARY_LOG"
fi

# ------------------------------------------------------------------------------
# STAGE 3: RoutePool HiGHS Set Partitioning Recombination Sweep
# ------------------------------------------------------------------------------
echo -e "\n----------------------------------------------------------------------" | tee -a "$SUMMARY_LOG"
echo "[Stage 3] Running RoutePool Recombination Ablation Sweep..." | tee -a "$SUMMARY_LOG"
STAGE3_LOG="logs/overnight_stage3_routepool.log"

if uv run python scratch/sweep_routepool_impact.py 2>&1 | tee "$STAGE3_LOG"; then
    echo "✓ Stage 3 COMPLETED successfully. Log: $STAGE3_LOG" | tee -a "$SUMMARY_LOG"
else
    echo "❌ Stage 3 ENCOUNTERED ERRORS. Log: $STAGE3_LOG" | tee -a "$SUMMARY_LOG"
fi

echo -e "\n======================================================================" | tee -a "$SUMMARY_LOG"
echo "  VRPTW OVERNIGHT BENCHMARK SUITE - FINISHED AT $(date)" | tee -a "$SUMMARY_LOG"
echo "======================================================================" | tee -a "$SUMMARY_LOG"

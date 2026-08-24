from __future__ import annotations

import multiprocessing as mp
import os
import random
import time
from collections.abc import Iterable
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd
import torch

from .config import (
    ALGO_ALNS_BASE,
    ALGO_ALNS_BASE_PLUS,
    ALGO_DQN,
    ALGO_HYBRID_DDQN,
    ALGO_HYBRID_DDQN_TRANSFER,
    ALGO_HYBRID_DDQN_TRANSFER_DR,
    ALGO_HYBRID_DDQN_TRANSFER_RC2,
    ALGO_HYBRID_FIXED,
    ALGO_HYBRID_RULE,
    ALGO_ORTOOLS,
    BKS,
    Config,
    canonical_algo_label,
    normalize_algorithm_frame,
)
from .core import Inst, Plan
from .generators import SyntheticVRPTWGenerator
from .operators import op_regret_2, op_shaw
from .rl import EliteArchive, WelfordRewardNormalizer
from .solvers import ALNSSolver, DQNSolver, HybridDDQNSolver, HybridFixedSolver, HybridRuleSolver, run_ortools

try:
    from safetensors.torch import load_file as _st_load
    from safetensors.torch import save_file as _st_save

    SAFETENSORS_OK = True
except Exception:
    SAFETENSORS_OK = False
    _st_load = _st_save = None


def _save_weights(weights: dict, stem: str) -> None:
    if SAFETENSORS_OK and _st_save is not None:
        path = stem + ".safetensors"
        _st_save(weights, path)
    else:
        path = stem + ".pt"
        torch.save({k: v.cpu() for k, v in weights.items()}, path)
    print(f"Weights saved → {path}")


def _load_weights(stem: str) -> dict | None:
    for suffix, loader in (
        (".safetensors", _st_load if SAFETENSORS_OK else None),
        (".pt", lambda f: torch.load(f, map_location="cpu")),
    ):
        p = stem + suffix
        if os.path.exists(p) and loader is not None:
            print(f"Weights loaded → {p}")
            return loader(p)
    return None


SOLVER_REGISTRY = {
    ALGO_ALNS_BASE: ALNSSolver,
    ALGO_ALNS_BASE_PLUS: ALNSSolver,
    ALGO_HYBRID_FIXED: HybridFixedSolver,
    ALGO_HYBRID_RULE: HybridRuleSolver,
    ALGO_HYBRID_DDQN: HybridDDQNSolver,
    ALGO_HYBRID_DDQN_TRANSFER: HybridDDQNSolver,
    ALGO_HYBRID_DDQN_TRANSFER_RC2: HybridDDQNSolver,
    ALGO_HYBRID_DDQN_TRANSFER_DR: HybridDDQNSolver,
    ALGO_DQN: DQNSolver,
}


def run_instance(
    inst: Inst,
    algo: str,
    cfg: Config,
    seed: int,
    transfer_weights: dict | None = None,
    init_plan: Plan | None = None,
    ddqn_time: float | None = None,
) -> tuple[dict, Plan | None]:
    import copy
    import inspect
    start = time.time()
    algo_canonical = canonical_algo_label(algo)
    use_gnn = algo_canonical.startswith("GNN-")
    target_algo = algo_canonical[4:] if use_gnn else algo_canonical

    plan: Plan | None = None
    history: list[float] = []

    if target_algo == ALGO_ORTOOLS:
        cfg_matched = copy.copy(cfg)
        if ddqn_time is not None:
            cfg_matched.ortools_time_limit = max(30.0, float(ddqn_time) * 0.95)
        plan, elapsed = run_ortools(inst, cfg_matched)
        if plan is None:
            return {
                "algo": algo_canonical,
                "nv": None,
                "cost": None,
                "time": time.time() - start,
                "td_gap": None,
                "nv_diff": None,
                "on_time": None,
                "hist": [],
            }, None
        history = [plan.cost]
    elif target_algo in SOLVER_REGISTRY:
        solver_cls = SOLVER_REGISTRY[target_algo]
        solver = solver_cls(inst, cfg)
        if use_gnn and getattr(cfg, "gnn_model_path", None) is not None and hasattr(solver, "load_gnn_model"):
            solver.load_gnn_model(cfg.gnn_model_path)

        solve_kwargs = {"seed": seed}
        sig = inspect.signature(solver.solve)
        if "init" in sig.parameters and init_plan is not None:
            solve_kwargs["init"] = init_plan

        is_transfer = target_algo in (ALGO_HYBRID_DDQN_TRANSFER, ALGO_HYBRID_DDQN_TRANSFER_RC2, ALGO_HYBRID_DDQN_TRANSFER_DR)
        if is_transfer:
            if transfer_weights is not None and hasattr(solver, "load_weights"):
                solver.load_weights(transfer_weights)
            if "frozen" in sig.parameters:
                solve_kwargs["frozen"] = True

        plan, history = solver.solve(**solve_kwargs)
        plan.algo = algo_canonical
    else:
        raise ValueError(f"Unsupported algorithm: {algo_canonical}")
    bks = BKS.get(inst.name)
    return {
        "algo": plan.algo,
        "nv": plan.nv,
        "cost": plan.cost,
        "time": time.time() - start,
        "td_gap": (plan.cost - bks["td"]) / bks["td"] * 100 if bks else None,
        "nv_diff": plan.nv - bks["nv"] if bks else None,
        "on_time": plan.on_time_rate,
        "hist": history,
    }, plan


# ---------------------------------------------------------------------------
# Top-level workers  — must be module-level for ProcessPoolExecutor pickling
# ---------------------------------------------------------------------------
def _benchmark_worker(packed: tuple) -> tuple[dict, Plan | None]:
    inst, algo, cfg, seed, transfer_weights, init_plan = packed
    return run_instance(inst, algo, cfg, seed, transfer_weights, init_plan)


def _benchmark_instance_worker(packed: tuple) -> list[dict]:
    # Restrict thread count to 1 to avoid thread contention across parallel processes
    import os

    import torch

    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
    os.environ["NUMEXPR_NUM_THREADS"] = "1"
    os.environ["NUMBA_NUM_THREADS"] = "1"
    try:
        torch.set_num_threads(1)
    except Exception:
        pass

    inst, algorithms, cfg, transfer_weights, plans_folder, completed, completed_times, wall_start = packed
    torch.set_num_threads(1)
    print(f"    [PROCESSING] {inst.name}...", flush=True)
    t0 = time.time()
    try:
    # Archive is isolated per-algorithm inside the loop below

        dataset = (
            "RC1"
            if inst.name.upper().startswith("RC1")
            else "RC2"
            if inst.name.upper().startswith("RC2")
            else inst.name[:2].upper()
        )

        inst_rows = []
        log_lines = []

        for algo in algorithms:
            algo_label = canonical_algo_label(algo)
            if (inst.name, algo_label) in completed:
                continue

            # Isolate plans folder for this algorithm under output_dir/algo_name/elite_plans/
            algo_plans_folder = os.path.join(os.path.dirname(plans_folder), algo_label, "elite_plans")
            os.makedirs(algo_plans_folder, exist_ok=True)

            archive = EliteArchive(k=cfg.elite_archive_k)
            if algo_label != ALGO_ALNS_BASE:
                archive.load_plans(algo_plans_folder, {inst.name: inst})

            nv_v, cost_v, time_v, gap_v, nvd_v, ot_v = [], [], [], [], [], []
            n_runs_eff = 1 if algo_label == ALGO_ORTOOLS else cfg.n_runs

            best_overall: Plan | None = None
            run_results = []
            weights = (
                transfer_weights
                if algo_label
                in (ALGO_HYBRID_DDQN_TRANSFER, ALGO_HYBRID_DDQN_TRANSFER_RC2, ALGO_HYBRID_DDQN_TRANSFER_DR)
                else None
            )

            ddqn_time = None
            if algo_label == ALGO_ORTOOLS:
                ddqn_time = completed_times.get((inst.name, ALGO_HYBRID_DDQN))
                if ddqn_time is None:
                    ddqn_time = completed_times.get((inst.name, f"GNN-{ALGO_HYBRID_DDQN}"))
                if ddqn_time is None:
                    for row in inst_rows:
                        if row["Algorithm"] in (ALGO_HYBRID_DDQN, f"GNN-{ALGO_HYBRID_DDQN}"):
                            ddqn_time = row["Time_s"]
                            break

            for i in range(n_runs_eff):
                seed = cfg.seed + i
                # Enforce strict independent cold-start across all runs/seeds (no cross-seed carryover)
                init = None

                res, plan = run_instance(inst, algo_label, cfg, seed, weights, init, ddqn_time=ddqn_time)
                elapsed_h = (time.time() - wall_start) / 3600
                if res["nv"] is not None:
                    print(
                        f"    [RUN] {inst.name} | {algo_label} | run {i + 1}/{n_runs_eff}: "
                        f"nv={res['nv']} cost={res['cost']:.1f} ({res['time']:.1f}s) | wall {elapsed_h:.2f}h",
                        flush=True
                    )
                else:
                    print(f"    [RUN] {inst.name} | {algo_label} | run {i + 1}/{n_runs_eff}: FAILED ({res['time']:.1f}s)", flush=True)

                if (
                    algo_label
                    in (ALGO_HYBRID_DDQN_TRANSFER, ALGO_HYBRID_DDQN_TRANSFER_RC2, ALGO_HYBRID_DDQN_TRANSFER_DR)
                    and plan is not None
                ):
                    plan.algo = algo_label
                if plan is not None:
                    if best_overall is None or plan.dominates(best_overall) or plan.nv < best_overall.nv:
                        best_overall = plan.copy()
                run_results.append((res, plan))

            for res, plan in run_results:
                if plan is not None:
                    archive.update_and_save(plan, algo_plans_folder)
                time_v.append(res["time"])
                if res["nv"] is not None:
                    nv_v.append(res["nv"])
                    cost_v.append(res["cost"])
                    gap_v.append(res["td_gap"])
                    nvd_v.append(res["nv_diff"])
                    ot_v.append(res["on_time"])

            if not nv_v:
                # Store a dummy/failed entry in checkpoint to prevent infinite re-runs
                row = {
                    "Dataset": dataset,
                    "Instance": inst.name,
                    "Algorithm": algo_label,
                    "NV_mean": None,
                    "NV_std": None,
                    "NV_diff": None,
                    "TD_mean": None,
                    "TD_std": None,
                    "Gap%": None,
                    "OnTime": None,
                    "Time_s": round(float(np.mean(time_v)), 1) if time_v else 0.0,
                    "NV_cv": None,
                    "TD_cv": None,
                    "NV_inflated": False,
                    "raw_costs": "",
                    "raw_nv": "",
                }
                inst_rows.append(row)
                log_lines.append(f"\n[{inst.name}] {algo_label}")
                log_lines.append("  -> FAILED/INFEASIBLE (no solution found)")
                continue

            bks = BKS.get(inst.name)
            nv_inflated = (
                bks is not None
                and float(np.mean(nv_v)) > bks["nv"] + 0.4
                and all(g is not None for g in gap_v)
                and float(np.mean(gap_v)) < 0
            )

            row = {
                "Dataset": dataset,
                "Instance": inst.name,
                "Algorithm": algo_label,
                "NV_mean": round(float(np.mean(nv_v)), 2),
                "NV_std": round(float(np.std(nv_v)), 2),
                "NV_diff": round(float(np.mean(nvd_v)), 2) if nvd_v and nvd_v[0] is not None else None,
                "TD_mean": round(float(np.mean(cost_v)), 2),
                "TD_std": round(float(np.std(cost_v)), 2),
                "Gap%": round(float(np.mean(gap_v)), 2) if gap_v and gap_v[0] is not None else None,
                "OnTime": round(float(np.mean(ot_v)) * 100, 1),
                "Time_s": round(float(np.mean(time_v)), 1),
                "NV_cv": round(float(np.std(nv_v)) / max(float(np.mean(nv_v)), 1) * 100, 2),
                "TD_cv": round(float(np.std(cost_v)) / max(float(np.mean(cost_v)), 1) * 100, 2),
                "NV_inflated": nv_inflated,
                "raw_costs": ";".join(f"{c:.4f}" for c in cost_v),
                "raw_nv": ";".join(str(n) for n in nv_v),
            }
            inst_rows.append(row)

            # Format the output block for this algorithm
            log_lines.append(f"\n[{inst.name}] {algo_label}")
            for i, (res, _) in enumerate(run_results):
                elapsed_h = (time.time() - wall_start) / 3600
                if res["nv"] is not None:
                    log_lines.append(
                        f"  run {i + 1}/{n_runs_eff}: nv={res['nv']} cost={res['cost']:.1f} "
                        f"({res['time']:.1f}s) | wall {elapsed_h:.2f}h"
                    )
                else:
                    log_lines.append(f"  run {i + 1}/{n_runs_eff}: FAILED ({res['time']:.1f}s)")

            gap_text = f"{row['Gap%']:+.1f}%" if row["Gap%"] is not None else "--"
            log_lines.append(
                f"  -> nv={row['NV_mean']:.1f}±{row['NV_std']:.1f}  "
                f"td={row['TD_mean']:.1f}±{row['TD_std']:.1f}  gap={gap_text}"
            )

        elapsed = time.time() - t0
        print(f"    [SUCCESS] {inst.name} finished in {int(elapsed)}s", flush=True)
        if log_lines:
            print("\n".join(log_lines), flush=True)
        print(f"[{inst.name}] Completed all requested algorithms.", flush=True)
        return inst_rows
    except Exception as exc:
        print(f"    [!!! CRITICAL ERROR !!!] {inst.name} failed", flush=True)
        raise exc


def _diversified_init(run_idx: int, inst: Inst, archive: EliteArchive, cfg: Config) -> Plan | None:
    """
    run 0 → archive best       (exploitation of known-good solution)
    run 1 → shaw-perturbed     (exploration from good neighbourhood)
    run 2 → fresh greedy build (independent trajectory)
    Avoids 3 seeds converging to the same local optimum.
    """
    base = archive.best(inst.name)
    if run_idx == 0:
        return base
    if run_idx == 1 and base is not None:
        inst_hash = hash(inst.name) % 100_000
        random.seed(cfg.seed + 7919 + inst_hash)
        np.random.seed(cfg.seed + 7919 + inst_hash)
        size = max(4, int(0.20 * inst.n))
        dest, removed = op_shaw(base.copy(), size)
        cand = op_regret_2(dest, removed)
        return cand if (cand.feasible and cand.nv <= base.nv + 1) else base
    return None  # run_instance will call build_greedy


# ---------------------------------------------------------------------------
# run_benchmark  (ProcessPoolExecutor for true GIL bypass)
# ---------------------------------------------------------------------------
def run_benchmark(
    instances: Iterable[Inst],
    algorithms: list[str],
    cfg: Config,
    result_path: str | None = None,
    transfer_weights: dict | None = None,
    archive: EliteArchive | None = None,
    checkpoint_path: str | None = None,
    no_checkpoint: bool = False,
    max_workers: int | None = None,
) -> pd.DataFrame:
    cfg.validate()
    instances = list(instances)
    result_path = result_path or os.path.join(cfg.output_dir, "benchmark_clean.csv")
    ckpt_path = checkpoint_path or os.path.join(cfg.output_dir, "benchmark_checkpoint.csv")
    if archive is None:
        archive = EliteArchive(k=cfg.elite_archive_k)

    # Load existing plans for cross-seeding
    plans_folder = os.path.join(cfg.output_dir, "elite_plans")
    insts_dict = {inst.name: inst for inst in instances}
    archive.load_plans(plans_folder, insts_dict)

    rows: list[dict] = []
    completed: set = set()
    completed_times: dict = {}
    if not no_checkpoint and os.path.exists(ckpt_path):
        try:
            ckpt_df = pd.read_csv(ckpt_path)
            rows = ckpt_df.to_dict("records")
            for row in rows:
                key = (row["Instance"], canonical_algo_label(str(row["Algorithm"])))
                completed.add(key)
                if "Time_s" in row and not pd.isna(row["Time_s"]):
                    completed_times[key] = float(row["Time_s"])
            print(f"Resumed from checkpoint: {len(completed)} combo(s) already done")
        except Exception as exc:
            print(f"Checkpoint read failed ({exc}), starting fresh")

    total = len(instances) * len(algorithms)
    print(f"Total: {total} combos × {cfg.n_runs} runs  |  wall limit: {cfg.max_wall_hours:.1f}h")
    wall_start = time.time()

    # Prepare worker arguments for instance-level parallelization
    worker_args = []
    for inst in instances:
        # Check if any algorithm for this instance needs to be run
        needs_run = False
        for algo in algorithms:
            if (inst.name, canonical_algo_label(algo)) not in completed:
                needs_run = True
                break
        if needs_run:
            worker_args.append((inst, algorithms, cfg, transfer_weights, plans_folder, completed, completed_times, wall_start))
        else:
            # Already completed in checkpoint — print status for progress monitor
            print(f"    [PROCESSING] {inst.name}...", flush=True)
            print(f"    [SUCCESS] {inst.name} finished in 0s", flush=True)

    if worker_args:
        workers_count = max_workers if max_workers is not None else min(len(worker_args), max(1, os.cpu_count() - 1))
        print(f"Parallelizing benchmark at the INSTANCE level across {workers_count} worker(s)...")
        ctx = mp.get_context("spawn")
        with ProcessPoolExecutor(max_workers=workers_count, mp_context=ctx) as ex:
            futures = {ex.submit(_benchmark_instance_worker, arg): arg[0] for arg in worker_args}
            try:
                for future in as_completed(futures):
                    try:
                        inst_rows = future.result()
                    except Exception as exc:
                        inst_name = futures[future].name
                        print(f"    [!!! CRITICAL ERROR !!!] Instance {inst_name} failed with: {exc}", flush=True)
                        raise exc
                    for row in inst_rows:
                        rows.append(row)
                        key = (row["Instance"], row["Algorithm"])
                        completed.add(key)
                        if "Time_s" in row and row["Time_s"] is not None:
                            completed_times[key] = float(row["Time_s"])

                    # Check for wall time limit
                    elapsed_h = (time.time() - wall_start) / 3600
                    if elapsed_h >= cfg.max_wall_hours:
                        print(f"\n⚠️  Wall-clock limit {cfg.max_wall_hours:.1f}h reached — stopping early.")
                        for f in futures:
                            f.cancel()
                        pd.DataFrame(rows).to_csv(ckpt_path, index=False)
                        pd.DataFrame(rows).to_csv(result_path, index=False)
                        ex.shutdown(wait=False, cancel_futures=True)
                        return normalize_algorithm_frame(pd.DataFrame(rows))

                    # Save checkpoint and print progress
                    pd.DataFrame(rows).to_csv(ckpt_path, index=False)
                    print(
                        f"  ✓ Checkpoint saved ({len(completed)}/{total} combos complete, {elapsed_h:.2f}h) → {ckpt_path}"
                    )
            except (KeyboardInterrupt, SystemExit) as interrupt:
                print("\n[!] Interruption/Cancellation signal received. Cleaning up workers immediately...", flush=True)
                for f in futures:
                    f.cancel()
                pd.DataFrame(rows).to_csv(ckpt_path, index=False)
                ex.shutdown(wait=False, cancel_futures=True)
                raise interrupt

    df = normalize_algorithm_frame(pd.DataFrame(rows))
    df.to_csv(result_path, index=False)
    print(f"\nBenchmark complete in {(time.time() - wall_start) / 3600:.2f}h → {result_path}")
    return df


# ---------------------------------------------------------------------------
# Summary table  (with completeness + NV-inflation guards)
# ---------------------------------------------------------------------------
def print_summary_table(df: pd.DataFrame) -> None:
    df = normalize_algorithm_frame(df)
    expected = {"RC1": 8, "RC2": 8}
    for ds, exp_n in expected.items():
        for algo in df[df["Dataset"] == ds]["Algorithm"].dropna().unique():
            n = len(df[(df["Dataset"] == ds) & (df["Algorithm"] == algo)])
            if n < exp_n:
                print(f"  ⚠️  {ds}/{algo}: {n}/{exp_n} instances — summary is PARTIAL")

    # NV-inflation warning
    if "NV_inflated" in df.columns:
        flagged = df[df["NV_inflated"]][["Instance", "Algorithm", "Gap%", "NV_mean"]]
        for _, r in flagged.iterrows():
            print(
                f"  ⚠️  {r['Instance']} {r['Algorithm']}: Gap%={r['Gap%']:+.1f}% "
                f"with NV={r['NV_mean']:.1f} > BKS_NV — not a fair comparison"
            )

    summary = (
        df.groupby(["Dataset", "Algorithm"], observed=True)
        .agg(
            NV=("NV_mean", "mean"),
            NV_std=("NV_std", "mean"),
            NV_diff=("NV_diff", "mean"),
            TD=("TD_mean", "mean"),
            TD_std=("TD_std", "mean"),
            Gap=("Gap%", "mean"),
            OnTime=("OnTime", "mean"),
            Time=("Time_s", "mean"),
        )
        .round(2)
        .reset_index()
    )
    print("\n" + "-" * 96)
    print(
        f"{'DS':<4}{'Algorithm':<28}{'NV':>6}{'+/-':>6}{'vsBKS':>8}{'TD':>10}{'+/-':>8}{'Gap%':>8}{'OT%':>7}{'Time':>8}"
    )
    print("-" * 96)
    for _, row in summary.iterrows():
        gap = f"{row['Gap']:+.2f}%" if pd.notna(row["Gap"]) else "--"
        nv_diff = f"{row['NV_diff']:+.2f}" if pd.notna(row["NV_diff"]) else "--"
        print(
            f"{row['Dataset']:<4}{row['Algorithm']:<28}"
            f"{row['NV']:>6.2f}{row['NV_std']:>6.2f}{nv_diff:>8}"
            f"{row['TD']:>10.2f}{row['TD_std']:>8.2f}{gap:>8}"
            f"{row['OnTime']:>7.1f}{row['Time']:>7.1f}s"
        )
    print("-" * 96)


# ---------------------------------------------------------------------------
# Transfer / domain randomization
# ---------------------------------------------------------------------------
def train_transfer_model(instances: list[Inst], cfg: Config, seed: int = 42, label: str = "RC1") -> dict:
    print(f"Training transfer model on {label} ({len(instances)} instances)...")
    if not instances:
        raise ValueError("No source instances provided.")
    weights = None
    for epoch in range(cfg.transfer_epochs):
        order = list(instances)
        if cfg.transfer_shuffle:
            random.Random(seed + epoch).shuffle(order)
        print(f"  Epoch {epoch + 1}/{cfg.transfer_epochs}")
        for idx, inst in enumerate(order):
            solver = HybridDDQNSolver(inst, cfg)
            if weights is not None:
                solver.load_weights(weights)
            plan, _ = solver.solve(seed=seed + epoch * 100 + idx)
            weights = solver.clone_weights()
            td_gap, _ = plan.gap()
            print(f"    [{epoch + 1}:{idx + 1}] {inst.name}: nv={plan.nv} gap={td_gap:+.1f}%")
    stem = os.path.join(cfg.output_dir, f"rl_alns_transfer_{label.lower()}_v15")
    _save_weights(weights, stem)
    return weights


def train_domain_randomization(cfg: Config, seed: int = 42) -> dict:
    """
    3-phase curriculum:
      Phase 1 (0-40%):   Easy   — 20-40 nodes
      Phase 2 (40-80%):  Target — 40-80 nodes
      Phase 3 (80-100%): Chaos  — 20-100 nodes
    """
    total_epochs = int(cfg.domain_randomization_epochs)
    batch_size = int(cfg.domain_randomization_batch)
    distributions = ("C", "R", "RC")
    rng = random.Random(seed)
    weights: dict | None = None
    shared_norm = WelfordRewardNormalizer(clip_sigma=8.0, warmup=256)
    print(f"Domain-randomization curriculum: {total_epochs} epochs × {batch_size} instances/epoch")

    for epoch in range(total_epochs):
        frac = (epoch + 1) / max(total_epochs, 1)
        if frac <= 0.40:
            phase, n_min, n_max = "Easy  ", 20, 40
        elif frac <= 0.80:
            phase, n_min, n_max = "Target", 40, 80
        else:
            phase, n_min, n_max = "Chaos ", 20, 100
        print(f"  Epoch {epoch + 1:>2}/{total_epochs}  [{phase}  N={n_min}-{n_max}]")
        batch: list[Inst] = []
        for idx in range(batch_size):
            n_nodes = rng.randint(n_min, n_max)
            dist = distributions[(epoch + idx) % len(distributions)]
            gen_seed = seed + epoch * 10_000 + idx
            batch.append(SyntheticVRPTWGenerator(n_nodes, dist, seed=gen_seed).generate())
        for idx, inst in enumerate(batch):
            solver = HybridDDQNSolver(inst, cfg)
            if weights is not None:
                solver.load_weights(weights)
            plan, _ = solver.solve(seed=seed + epoch * 1_000 + idx, shared_norm=shared_norm)
            weights = solver.clone_weights()
            print(
                f"    [{idx + 1:>2}/{batch_size}] {inst.name}: "
                f"n={inst.n}  nv={plan.nv}  cost={plan.cost:.1f}  feasible={plan.feasible}"
            )

    if weights is None:
        raise RuntimeError("Domain randomization produced no weights.")
    os.makedirs(cfg.output_dir, exist_ok=True)
    stem = os.path.join(cfg.output_dir, "rl_alns_dr_v15")
    _save_weights(weights, stem)
    return weights


def train_transfer_model_within_rc2(rc2_instances: list[Inst], cfg: Config, seed: int = 42) -> dict:
    source = rc2_instances[: cfg.rc2_transfer_split]
    print(f"RC2-within transfer: train on {[i.name for i in source]}")
    return train_transfer_model(source, cfg, seed=seed, label="RC2-src")


def load_transfer_model(cfg: Config, label: str = "rc1") -> dict | None:
    stem = os.path.join(cfg.output_dir, f"rl_alns_transfer_{label}_v15")
    return _load_weights(stem)


# ---------------------------------------------------------------------------
# Smoke test  — per-algo timing for accurate wall estimate
# ---------------------------------------------------------------------------
def smoke_test(inst: Inst, seed: int = 42) -> dict[str, tuple[float, float]]:
    """
    Returns {algo_name: (gap_pct, elapsed_s)} for the 4 main solvers.
    Uses short iterations so it completes in < 5 min on any hardware.
    """
    short_cfg = Config(
        alns_iterations=300,
        hybrid_iterations=400,
        early_stop_patience=100,
        polish_iterations=50,
        polish_patience=30,
        n_runs=1,
    )
    results: dict[str, tuple[float, float]] = {}
    for algo_name, solver_cls in (
        (ALGO_ALNS_BASE, ALNSSolver),
        (ALGO_HYBRID_FIXED, HybridFixedSolver),
        (ALGO_HYBRID_RULE, HybridRuleSolver),
        (ALGO_HYBRID_DDQN, HybridDDQNSolver),
    ):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        t0 = time.time()
        plan, _ = solver_cls(inst, short_cfg).solve(seed=seed)
        elapsed = time.time() - t0
        td_gap, nv_gap = plan.gap()
        gap_str = f"{td_gap:+.1f}%" if td_gap is not None else "N/A"
        nv_str = f"{nv_gap:+d}" if nv_gap is not None else "N/A"
        print(f"{algo_name:<24} nv={plan.nv:>3} cost={plan.cost:>8.1f} BKS TD {gap_str} NV {nv_str} ({elapsed:.1f}s)")
        results[algo_name] = (float(td_gap) if td_gap is not None else 0.0, elapsed)
    return results


def _fts_debug(inst):
    max_fts_bonus = (0.15 + 0.45 * inst.tw_tight_frac + 0.25 * 1.0) * 1.0 * inst.max_dist
    print(
        f"{inst.name}: max_dist={inst.max_dist:.1f}  "
        f"tw_tight_frac={inst.tw_tight_frac:.3f}  "
        f"max_fts_bonus={max_fts_bonus:.1f}"
    )

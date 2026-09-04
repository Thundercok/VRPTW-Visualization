#!/usr/bin/env python3
"""Configuration-driven 6-Tier Ablation Runner.

Executes:
  1. standard ALNS (alns)
  2. ALNS + Granular Neighborhoods (gns)
  3. ALNS + GNS + Route-Pool Recombination (gns_pool)
  4. ALNS + GNS + Route-Pool + LAC (gns_pool_lac)
  5. Micro-only Hybrid (micro)
  6. Full Hierarchical Hybrid (full)

Example:
python scripts/run_rule_ablations.py \
  --instances data/Solomon/c101.txt \
  --solver "python scripts/benchmark.py" \
  --seeds 42 43 44 45 46 \
  --time-limit 60.0 \
  --output results/ablation_suite
"""

from __future__ import annotations

import argparse
import csv
import json
import shlex
import subprocess
import time
from pathlib import Path

CUMULATIVE_CONFIGS = {
    "alns": "--ablation alns",
    "gns": "--ablation gns",
    "gns_pool": "--ablation gns_pool",
    "gns_pool_lac": "--ablation gns_pool_lac",
    "micro": "--ablation micro",
    "full": "--ablation full",
}

LOCO_CONFIGS = {
    "Full": "--ablation full",
    "Full_no_Macro": "--ablation full_no_macro",
    "Full_no_Micro": "--ablation full_no_micro",
    "Full_no_LAC": "--ablation full_no_lac",
    "Full_no_Pool": "--ablation full_no_pool",
    "Full_no_GNN": "--ablation full_no_gnn",
    "Full_no_Gate": "--ablation full_no_gate",
}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--instances", nargs="+", required=True, help="Instance paths or text file with paths")
    p.add_argument("--solver", required=True, help="Base solver command")
    p.add_argument("--mode", choices=["cumulative", "loco"], default="cumulative", help="Ablation mode")
    p.add_argument("--configs", default=None, help="Optional JSON file mapping config name -> CLI flags")
    p.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44, 45, 46])
    p.add_argument("--workers", type=int, default=4, help="Number of parallel solver workers")
    p.add_argument("--time-limit", type=float, default=30.0)
    p.add_argument("--output", required=True)
    p.add_argument("--timeout-slack", type=float, default=20.0)
    return p.parse_args()


def load_instances(inputs: list[str]) -> list[str]:
    instances = []
    for item in inputs:
        path = Path(item)
        if path.is_file() and not path.stem.lower().startswith(("c1", "c2", "r1", "r2", "rc1", "rc2")):
            for line in path.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if line and not line.startswith("#"):
                    instances.append(line)
        else:
            instances.append(str(path))
    return instances


def execute_single_run(job: tuple[str, str, str, int, str, float, float, Path]) -> dict:
    config_name, extra, instance, seed, solver_cmd, time_limit, timeout_slack, out = job
    extra_args = shlex.split(extra or "")
    token = f"{config_name}__{Path(instance).stem}__seed{seed}"
    result_json = out / f"{token}.json"
    stdout = out / f"{token}.stdout"
    stderr = out / f"{token}.stderr"

    cmd = shlex.split(solver_cmd)
    cmd += [
        "--instance",
        instance,
        "--seed",
        str(seed),
        "--time-limit",
        str(time_limit),
        "--output-json",
        str(result_json),
    ]
    cmd += extra_args

    start = time.perf_counter()
    timed_out = False

    try:
        proc = subprocess.run(
            cmd,
            stdout=stdout.open("w", encoding="utf-8"),
            stderr=stderr.open("w", encoding="utf-8"),
            timeout=time_limit + timeout_slack,
            check=False,
        )
        returncode = proc.returncode
    except subprocess.TimeoutExpired:
        timed_out = True
        returncode = None

    elapsed = time.perf_counter() - start

    payload = {}
    if result_json.exists():
        try:
            payload = json.loads(result_json.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            payload = {"parse_error": str(exc)}

    row = {
        "config": config_name,
        "instance": instance,
        "seed": seed,
        "time_limit_sec": time_limit,
        "wallclock_sec": elapsed,
        "returncode": returncode,
        "timed_out": timed_out,
        "feasible": payload.get("feasible"),
        "nv": payload.get("nv"),
        "td": payload.get("td"),
        "iterations": payload.get("iterations"),
    }
    print(f"  [{config_name}] {Path(instance).name} (seed {seed}) -> NV={row['nv']}, TD={row['td']} ({elapsed:.1f}s)")
    return row


def main():
    args = parse_args()
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    raw = out / "ablation_raw.csv"

    if args.configs:
        configs = json.loads(Path(args.configs).read_text(encoding="utf-8"))
    elif args.mode == "loco":
        configs = LOCO_CONFIGS
    else:
        configs = CUMULATIVE_CONFIGS

    instances = load_instances(args.instances)
    jobs = []

    for config_name, extra in configs.items():
        for instance in instances:
            for seed in args.seeds:
                jobs.append((config_name, extra, instance, seed, args.solver, args.time_limit, args.timeout_slack, out))

    rows = []
    import concurrent.futures

    print(f"Executing {len(jobs)} ablation runs across {args.workers} workers...")
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = [executor.submit(execute_single_run, job) for job in jobs]
        for fut in concurrent.futures.as_completed(futures):
            row = fut.result()
            rows.append(row)
            with raw.open("w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=rows[0].keys())
                writer.writeheader()
                writer.writerows(rows)

    (out / "manifest.json").write_text(
        json.dumps(
            {
                "instances": instances,
                "seeds": args.seeds,
                "time_limit_sec": args.time_limit,
                "configs": configs,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"\n✓ Ablation completed ({len(rows)} executions recorded in {raw})")


if __name__ == "__main__":
    main()

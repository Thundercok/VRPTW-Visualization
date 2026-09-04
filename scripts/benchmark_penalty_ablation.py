#!/usr/bin/env python3
"""Anytime wall-clock benchmark harness for VRPTW solver comparisons.

Design goals:
- equal wall-clock budgets for every solver;
- isolated process per run;
- deterministic seed propagation;
- checkpoint-quality recording;
- no assumptions about the solver's internal implementation.

Example:
python scripts/benchmark_penalty_ablation.py \
  --instances data/Solomon/c101.txt \
  --solver "python scripts/benchmark.py" \
  --seeds 42 43 44 45 46 \
  --cutoffs 1 5 10 30 60 120 300 \
  --output results/anytime
"""

from __future__ import annotations

import argparse
import csv
import json
import shlex
import subprocess
import time
from pathlib import Path
from typing import Any

DEFAULT_CUTOFFS = [1, 5, 10, 30, 60, 120, 300]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--instances", nargs="+", required=True, help="Instance paths or text file with paths")
    p.add_argument("--solver", required=True, help="Base solver command")
    p.add_argument("--solver-name", default="solver")
    p.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44, 45, 46])
    p.add_argument("--cutoffs", nargs="+", type=float, default=DEFAULT_CUTOFFS)
    p.add_argument("--output", required=True)
    p.add_argument("--extra-args", default="")
    p.add_argument("--timeout-slack", type=float, default=20.0)
    p.add_argument("--resume", action="store_true")
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


def lex_key(row: dict[str, Any]) -> tuple[int, float]:
    return (int(row["nv"]), float(row["td"]))


def run_one(
    base_cmd: str,
    solver_name: str,
    instance: str,
    seed: int,
    cutoff: float,
    output_dir: Path,
    extra_args: str,
    timeout_slack: float,
) -> dict[str, Any]:

    output_dir.mkdir(parents=True, exist_ok=True)
    token = f"{solver_name}__{Path(instance).stem}__seed{seed}__t{cutoff:g}"
    json_path = output_dir / f"{token}.json"
    stdout_path = output_dir / f"{token}.stdout"
    stderr_path = output_dir / f"{token}.stderr"

    cmd = shlex.split(base_cmd)
    cmd += [
        "--instance",
        instance,
        "--seed",
        str(seed),
        "--time-limit",
        str(cutoff),
        "--output-json",
        str(json_path),
    ]
    if extra_args:
        cmd += shlex.split(extra_args)

    start = time.perf_counter()
    timed_out = False
    returncode: int | None = None

    try:
        proc = subprocess.run(
            cmd,
            stdout=stdout_path.open("w", encoding="utf-8"),
            stderr=stderr_path.open("w", encoding="utf-8"),
            timeout=cutoff + timeout_slack,
            check=False,
        )
        returncode = proc.returncode
    except subprocess.TimeoutExpired:
        timed_out = True

    wall = time.perf_counter() - start

    payload: dict[str, Any] = {}
    if json_path.exists():
        try:
            payload = json.loads(json_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            payload = {"parse_error": str(exc)}

    row: dict[str, Any] = {
        "solver": solver_name,
        "instance": instance,
        "seed": seed,
        "cutoff_sec": cutoff,
        "wallclock_sec": wall,
        "returncode": returncode,
        "timed_out": timed_out,
        "feasible": payload.get("feasible"),
        "nv": payload.get("nv"),
        "td": payload.get("td"),
        "iterations": payload.get("iterations"),
        "solver_runtime_sec": payload.get("runtime_sec"),
        "metadata_json": json.dumps(payload.get("metadata", {}), sort_keys=True),
    }

    if row["feasible"] is True and row["nv"] is not None and row["td"] is not None:
        row["lex_key"] = list(lex_key(row))
    else:
        row["lex_key"] = None

    return row


def aggregate(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    groups: dict[tuple[str, float], list[dict[str, Any]]] = {}
    for row in rows:
        groups.setdefault((row["solver"], row["cutoff_sec"]), []).append(row)

    for (solver, cutoff), items in sorted(groups.items()):
        successful = [
            x for x in items if x.get("feasible") is True and x.get("nv") is not None and x.get("td") is not None
        ]
        runtimes = [float(x["wallclock_sec"]) for x in items if x.get("wallclock_sec") is not None]
        out.append(
            {
                "solver": solver,
                "cutoff_sec": cutoff,
                "runs": len(items),
                "successful_runs": len(successful),
                "mean_wallclock_sec": sum(runtimes) / len(runtimes) if runtimes else None,
            }
        )
    return out


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    fields = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    args = parse_args()

    instances = load_instances(args.instances)
    output = Path(args.output)
    raw_dir = output / "raw"
    output.mkdir(parents=True, exist_ok=True)

    manifest = {
        "solver": args.solver,
        "solver_name": args.solver_name,
        "instances": instances,
        "seeds": args.seeds,
        "cutoffs_sec": args.cutoffs,
        "protocol": {
            "equal_wall_clock": True,
            "isolated_process_per_run": True,
            "parent_timeout_slack_sec": args.timeout_slack,
        },
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    rows: list[dict[str, Any]] = []
    completed = set()

    raw_csv = output / "anytime_raw.csv"
    if args.resume and raw_csv.exists():
        with raw_csv.open(newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                key = (row["solver"], row["instance"], int(row["seed"]), float(row["cutoff_sec"]))
                completed.add(key)
                rows.append(row)

    total = len(instances) * len(args.seeds) * len(args.cutoffs)
    done = len(completed)

    for instance in instances:
        for seed in args.seeds:
            for cutoff in args.cutoffs:
                key = (args.solver_name, instance, seed, cutoff)
                if key in completed:
                    continue

                done += 1
                print(
                    f"[{done}/{total}] {args.solver_name} | {Path(instance).name} | seed={seed} | t={cutoff:g}s",
                    flush=True,
                )

                row = run_one(
                    args.solver,
                    args.solver_name,
                    instance,
                    seed,
                    cutoff,
                    raw_dir,
                    args.extra_args,
                    args.timeout_slack,
                )
                rows.append(row)
                write_csv(raw_csv, rows)

    rows.sort(key=lambda r: (r["solver"], r["instance"], int(r["seed"]), float(r["cutoff_sec"])))
    write_csv(raw_csv, rows)
    write_csv(output / "computational_summary.csv", aggregate(rows))

    (output / "run_complete.json").write_text(
        json.dumps(
            {
                "completed_rows": len(rows),
                "expected_rows": total,
                "complete": len(rows) == total,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""
Solver fingerprint regression test.

Phase 1 and Phase 2 of the optimisation work are behaviour-preserving by design:
they remove Python/Numba marshalling overhead and redundant recomputation without
changing which moves the search makes. This test enforces that — if a "pure speed"
change shifts (nv, cost) at all, it changed the search, and that is a bug.

Regenerate the baseline with:
    python scripts/capture_golden.py
"""

from __future__ import annotations

import json
import os
import sys

import pytest

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_REPO, "scripts"))

from capture_golden import _make_cfg, load_instance  # noqa: E402

from vrptw.solvers import ALNSSolver, HybridDDQNSolver  # noqa: E402

GOLDEN_PATH = os.path.join(_REPO, "tests", "golden", "baseline.json")

SOLVERS = {"Hybrid-DDQN": HybridDDQNSolver, "ALNS-Base": ALNSSolver}

INSTANCE_PATHS = {
    "R101": "data/Solomon/r101.txt",
    "RC207": "data/Solomon/rc207.txt",
    "C101": "data/Solomon/c101.txt",
    "r1_2_1": "data/Gehring_Homberger/homberger_200_customer_instances/R1_2_1.TXT",
}


def _load_golden() -> list[dict]:
    if not os.path.exists(GOLDEN_PATH):
        pytest.skip(f"No golden baseline at {GOLDEN_PATH}; run scripts/capture_golden.py first")
    with open(GOLDEN_PATH, encoding="utf-8") as fh:
        return json.load(fh)["records"]


def _record_id(rec: dict) -> str:
    return f"{rec['instance']}-{rec['solver']}-seed{rec['seed']}"


@pytest.mark.parametrize("record", _load_golden(), ids=_record_id)
def test_solver_fingerprint_unchanged(record: dict) -> None:
    path = os.path.join(_REPO, INSTANCE_PATHS[record["instance"]])
    if not os.path.exists(path):
        pytest.skip(f"instance file missing: {path}")

    inst = load_instance(path)
    solver = SOLVERS[record["solver"]](inst, _make_cfg())
    best, _ = solver.solve(seed=record["seed"])

    assert best.feasible == record["feasible"], f"{_record_id(record)}: feasibility changed"
    assert best.nv == record["nv"], (
        f"{_record_id(record)}: NV changed {record['nv']} -> {best.nv}. "
        "A behaviour-preserving optimisation must not alter the search trajectory."
    )
    # Exact equality would be brittle against floating-point summation order, but
    # any real trajectory change moves the cost by far more than this tolerance.
    assert best.cost == pytest.approx(record["cost"], rel=1e-6, abs=1e-4), (
        f"{_record_id(record)}: cost changed {record['cost']:.6f} -> {best.cost:.6f}"
    )

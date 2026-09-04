import os
import random

import numpy as np

from vrptw.core import Inst, Plan
from vrptw.heuristics import build_greedy
from vrptw.rl import EliteArchive


def load_inst_rc202() -> Inst:
    file_path = "data/Solomon/RC202.txt"
    assert os.path.exists(file_path), f"Solomon file not found at {file_path}"
    with open(file_path, encoding="utf-8") as fh:
        lines = fh.readlines()
    name = lines[0].strip()
    capacity = float(lines[4].strip().split()[1])
    rows = [list(map(float, ln.split())) for ln in lines[9:] if ln.strip()]
    return Inst({"name": name, "capacity": capacity, "data": np.array(rows)})


def test_crossover_complete_plans():
    inst = load_inst_rc202()
    arch = EliteArchive(k=5)

    # Generate two distinct plans
    p1 = build_greedy(inst, heatmap=None, gnn_strength=0.0)
    # Generate a second plan by slightly shuffling/shifting routes to make it distinct
    routes2 = p1.routes[::-1]
    p2 = Plan(routes2, inst, "synth")

    key = inst.name
    bucket = arch._plans.setdefault(key, [])
    bucket.append(p1.copy())
    bucket.append(p2.copy())

    child = arch.crossover(inst.name)
    assert child is not None
    assert child.feasible
    assert len(child.routes) > 0


def test_crossover_offspring_feasible_exact_cover():
    """SREX offspring must be a feasible exact cover. The old implementation
    produced feasible but heavily fragmented offspring (inflated route count),
    so 0/600 measured offspring could pass the ``alt.nv < best.nv`` acceptance
    gate — crossover never contributed a usable plan to the search."""
    inst = load_inst_rc202()
    arch = EliteArchive(k=5)

    p1 = build_greedy(inst)
    assert p1.feasible

    # A second parent with a different route topology: reversed route order and
    # 4 dropped customers repaired back in by the crossover itself.
    second_half_customers = [c for r in p1.routes[len(p1.routes) // 2 :] for c in r]
    assert len(second_half_customers) >= 4
    drop_set = set(second_half_customers[:4])
    routes2 = [[c for c in r if c not in drop_set] for r in p1.routes[::-1]]
    p2 = Plan([r for r in routes2 if r], inst, "synth")

    bucket = arch._plans.setdefault(inst.name, [])
    bucket.append(p1.copy())
    bucket.append(p2.copy())

    random.seed(7)
    produced = 0
    for _ in range(20):
        child = arch.crossover(inst.name)
        if child is None:
            continue  # the exchange may be judged too destructive — allowed
        produced += 1
        assert child.feasible, "SREX offspring must be feasible by construction"
        served = [c for r in child.routes for c in r]
        assert len(served) == inst.n and len(set(served)) == inst.n, "offspring must serve every customer exactly once"
    assert produced > 0, "crossover must produce at least one feasible offspring in 20 tries"

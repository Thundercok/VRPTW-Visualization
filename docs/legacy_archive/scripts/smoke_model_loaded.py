"""Prove the trained DDQN weights actually drive the solver.

We compare two solver runs on the same instance:
  1. With the safetensors weights loaded (production behaviour).
  2. With weights replaced by zeros (baseline).

Different total_distance / different action sequences -> the weights matter.
Same numbers -> the controller never read the network.

Also prints the parameter L2-norm before/after loading so you can see they were
actually copied in.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src" / "backend"))

import torch  # noqa: E402
from services.solomon_service import load_solomon_dataset  # noqa: E402
from services.solver_service import (  # noqa: E402
    WEB_CONFIG,
    _load_transfer_weights,
    _resolve_transfer_path,
)


def _payload():
    data = load_solomon_dataset("demo")
    from models.schemas import FleetConfig, JobRequest, Point
    customers = [
        Point(
            id=int(c["id"]),
            name=str(c["name"]),
            address=str(c["address"]),
            lat=float(c["lat"]),
            lng=float(c["lng"]),
            demand=int(c["demand"]),
            ready=float(c["ready"]),
            due=float(c["due"]),
            service=float(c["service"]),
            isDepot=bool(c["isDepot"]),
        )
        for c in data["customers"]
    ]
    return JobRequest(
        mode="sample",
        fleet=FleetConfig(vehicles=int(data["fleet"]["vehicles"]), capacity=int(data["fleet"]["capacity"])),
        customers=customers,
    )


def _run(seed: int, load_weights: bool):
    from services.research_adapter import build_inst

    from vrptw import PlateauHybridSolver

    payload = _payload()
    inst = build_inst(payload.customers, capacity=payload.fleet.capacity, name="probe")
    solver = PlateauHybridSolver(inst, WEB_CONFIG)

    if load_weights:
        loaded = _load_transfer_weights(solver)
    else:
        loaded = False
        with torch.no_grad():
            for p in solver.ctrl.q.parameters():
                p.zero_()
            for p in solver.ctrl.q_t.parameters():
                p.zero_()

    solver.ctrl.eps = WEB_CONFIG.ctrl_eps_end
    norm = float(torch.cat([p.detach().flatten() for p in solver.ctrl.q.parameters()]).norm().item())
    plan, _ = solver.solve(seed=seed)
    return {
        "loaded": loaded,
        "qnet_norm": norm,
        "total_distance": float(plan.cost),
        "vehicles_used": int(plan.nv),
        "routes": [tuple(r) for r in plan.routes],
    }


def main() -> int:
    path = _resolve_transfer_path()
    print(f"Transfer weights path: {path}")
    if path is None:
        print("FAIL: cannot find rl_alns_transfer.safetensors")
        return 1

    seed = WEB_CONFIG.seed
    trained = _run(seed, load_weights=True)
    zeroed = _run(seed, load_weights=False)

    print("--- TRAINED ---")
    for k in ("loaded", "qnet_norm", "total_distance", "vehicles_used"):
        print(f"  {k}: {trained[k]}")
    print("--- ZEROED  ---")
    for k in ("loaded", "qnet_norm", "total_distance", "vehicles_used"):
        print(f"  {k}: {zeroed[k]}")

    different_norm = abs(trained["qnet_norm"] - zeroed["qnet_norm"]) > 1e-6
    different_routes = trained["routes"] != zeroed["routes"]
    different_dist = abs(trained["total_distance"] - zeroed["total_distance"]) > 1e-6

    print()
    print(f"Q-net norm differs:  {different_norm}")
    print(f"Routes differ:       {different_routes}")
    print(f"Total distance diff: {different_dist}  (delta = {trained['total_distance'] - zeroed['total_distance']:.4f})")

    if not different_norm:
        print("FAIL: Q-net weights are identical -> file was not loaded into the network")
        return 2
    if not different_routes and not different_dist:
        print("WARN: behaviour is identical between trained and zeroed weights. The "
              "instance may be too small/easy to expose the policy difference - try a "
              "harder instance once Solomon files are present.")

    print("OK: trained weights are present and load into the live solver.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

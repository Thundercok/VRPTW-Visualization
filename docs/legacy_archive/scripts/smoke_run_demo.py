"""Reproduce the 'Run Model' flow against the built-in demo dataset.

If the solver explodes, this prints the full traceback - much faster than
clicking through the UI.
"""

from __future__ import annotations

import asyncio
import os
import sys
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src" / "backend"))

os.environ.setdefault("DEMO_AUTH_BYPASS", "true")

from models.schemas import FleetConfig, JobRequest, Point  # noqa: E402
from services.solomon_service import load_solomon_dataset  # noqa: E402
from services.solver_service import solve_model  # noqa: E402


def build_payload() -> JobRequest:
    data = load_solomon_dataset("demo")
    customers: list[Point] = []
    for c in data["customers"]:
        customers.append(
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
        )
    fleet_dict = data["fleet"]
    return JobRequest(
        mode="sample",
        fleet=FleetConfig(vehicles=int(fleet_dict["vehicles"]), capacity=int(fleet_dict["capacity"])),
        customers=customers,
    )


async def main() -> int:
    payload = build_payload()
    print(f"Built payload: vehicles={payload.fleet.vehicles}, capacity={payload.fleet.capacity}, customers={len(payload.customers)}")
    try:
        result = await solve_model(payload)
    except Exception:
        print("--- FAILURE ---")
        traceback.print_exc()
        return 1
    ddqn = result["ddqn"]
    alns = result["alns"]
    print("DDQN:", {k: ddqn[k] for k in ("runtime_sec", "total_distance_km", "vehicles_used")})
    print("ALNS:", {k: alns[k] for k in ("runtime_sec", "total_distance_km", "vehicles_used")})
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))

"""Hit the full /api/jobs HTTP path to flush out any 500 the user is seeing."""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src" / "backend"))

os.environ.setdefault("DEMO_AUTH_BYPASS", "true")

from fastapi.testclient import TestClient  # noqa: E402
from services.solomon_service import load_solomon_dataset  # noqa: E402

from main import app  # noqa: E402


def main() -> int:
    data = load_solomon_dataset("demo")
    payload = {
        "mode": "sample",
        "fleet": data["fleet"],
        "customers": data["customers"],
    }

    with TestClient(app) as client:
        r = client.post("/api/jobs", json=payload)
        print("submit status:", r.status_code, r.text[:300])
        if r.status_code != 200:
            return 1

        job_id = r.json()["job_id"]
        print("job_id:", job_id)

        for i in range(80):
            time.sleep(0.5)
            rj = client.get(f"/api/jobs/{job_id}")
            body = rj.json()
            if i % 4 == 0 or body["status"] in {"done", "failed"}:
                print(f"  poll[{i}] status={body.get('status')} error={body.get('error')}")
            if body["status"] in {"done", "failed"}:
                if body["status"] == "failed":
                    print("DEBUG:", body.get("debug"))
                    return 2
                res = body["result"]
                for algo in ("ddqn", "alns"):
                    d = res[algo]
                    print(f"  {algo}: runtime={d['runtime_sec']:.2f}s dist={d['total_distance_km']:.1f}km vehicles={d['vehicles_used']}")
                return 0

    print("Timeout waiting for job")
    return 3


if __name__ == "__main__":
    sys.exit(main())

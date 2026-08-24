"""Smoke test: rate limit, /api/config, /api/health, /api/solomon?name=demo."""

from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src" / "backend"))

os.environ.setdefault("DEMO_AUTH_BYPASS", "true")
os.environ.setdefault("RATE_LIMIT_AUTH_TOKEN", "3/minute")
os.environ.setdefault("PLAUSIBLE_DOMAIN", "vrptw.test")
os.environ.setdefault("SENTRY_FRONTEND_DSN", "https://abc@sentry.io/123")

from fastapi.testclient import TestClient  # noqa: E402

from main import app  # noqa: E402


def _check(label: str, ok: bool, extra: str = "") -> None:
    mark = "ok" if ok else "FAIL"
    print(f"  [{mark}] {label} {extra}")


def main() -> int:
    client = TestClient(app)
    failures = 0

    r = client.get("/api/health")
    _check("/api/health 200", r.status_code == 200, str(r.json()))
    failures += 0 if r.status_code == 200 else 1

    r = client.get("/api/config")
    cfg = r.json()
    sentry_ok = cfg["sentry"]["enabled"] and cfg["sentry"]["dsn"].startswith("https://")
    plausible_ok = cfg["plausible"]["enabled"] and cfg["plausible"]["domain"] == "vrptw.test"
    _check("/api/config sentry+plausible", sentry_ok and plausible_ok, str(cfg))
    failures += 0 if (sentry_ok and plausible_ok) else 1

    r = client.get("/api/solomon", params={"name": "demo"})
    if r.status_code == 200:
        body = r.json()
        ok = body["dataset"] == "demo" and len(body["customers"]) >= 12 and body["customers"][0]["isDepot"]
    else:
        ok = False
    _check("/api/solomon?name=demo built-in", ok, f"status={r.status_code}")
    failures += 0 if ok else 1

    bad_count = 0
    statuses: list[int] = []
    for _ in range(5):
        rr = client.post(
            "/api/auth/token",
            json={"email": "no-such@user.local", "password": "x" * 8},
        )
        statuses.append(rr.status_code)
        if rr.status_code == 429:
            bad_count += 1
    _check(
        "POST /api/auth/token rate-limited after burst",
        bad_count >= 1,
        f"statuses={statuses}",
    )
    failures += 0 if bad_count >= 1 else 1

    r = client.get("/")
    csp = r.headers.get("content-security-policy", "")
    has_sentry = "sentry-cdn.com" in csp
    has_plausible = "plausible.io" in csp
    _check(
        "CSP allows sentry-cdn + plausible",
        has_sentry and has_plausible,
        f"sentry={has_sentry} plausible={has_plausible}",
    )
    failures += 0 if (has_sentry and has_plausible) else 1

    print()
    print(f"Failures: {failures}")
    return failures


if __name__ == "__main__":
    sys.exit(main())

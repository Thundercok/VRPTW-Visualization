"""Remote solver service for the VRPTW optimizer, deployed to Google Cloud Run.

The public API lives on Render, which is memory-capped well below what torch,
numba and the DDQN weights need. This service carries the research stack and
answers the three calls that actually have to touch it: a full multi-algorithm
solve, a local-search re-optimize, and a dynamic customer insertion.

Everything else (auth, geocoding, job bookkeeping, static frontend) stays on the
API service. See ``src/backend/services/compute_gateway.py`` for the caller.

Transfer weights ship in the image but can be overridden at boot from a Hugging
Face model repo (``VRPTW_HF_MODEL_REPO``), so retraining does not require an
image rebuild.
"""

from __future__ import annotations

import logging
import os
import secrets
import sys
from pathlib import Path
from typing import Any

from fastapi import Depends, FastAPI, Header, HTTPException
from pydantic import BaseModel

_ROOT = Path(__file__).resolve().parent
for candidate in (_ROOT / "src", _ROOT / "src" / "backend", _ROOT):
    if candidate.exists() and str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("vrptw.solver_space")

from models.schemas import JobRequest, ReoptimizeRequest  # noqa: E402

app = FastAPI(title="VRPTW Solver (Hugging Face Space)")


# ── Auth ──────────────────────────────────────────────────────────────────────


def _expected_token() -> str:
    return os.getenv("SOLVER_API_TOKEN", "").strip()


async def require_token(x_solver_token: str = Header(default="")) -> None:
    """Shared-secret gate.

    A Space is world-reachable, and a solve is minutes of CPU, so an open
    endpoint is a free denial-of-service. When no token is configured the
    service stays open on purpose — that is the local/dev case — but it says so
    loudly at startup.
    """
    expected = _expected_token()
    if not expected:
        return
    if not secrets.compare_digest(x_solver_token or "", expected):
        raise HTTPException(status_code=401, detail="Invalid or missing X-Solver-Token.")


# ── Model weights ─────────────────────────────────────────────────────────────

_HUB_WEIGHTS_STATUS: dict[str, Any] = {"source": "bundled"}


def _sync_weights_from_hub() -> None:
    """Pull transfer weights from the Hugging Face Hub when configured.

    The image already carries a copy, so a Hub failure is never fatal — it just
    leaves the bundled weights in place. Setting ``VRPTW_TRANSFER_WEIGHTS`` is
    what makes ``solver_service._resolve_transfer_path`` prefer the download.
    """
    repo = os.getenv("VRPTW_HF_MODEL_REPO", "").strip()
    if not repo:
        return

    filename = os.getenv("VRPTW_HF_MODEL_FILE", "rl_alns_dr_v15.safetensors").strip()
    try:
        from huggingface_hub import hf_hub_download

        path = hf_hub_download(
            repo_id=repo,
            filename=filename,
            revision=os.getenv("VRPTW_HF_MODEL_REVISION", "main").strip() or "main",
            token=os.getenv("HF_TOKEN") or None,
            cache_dir=os.getenv("HF_HOME", "/tmp/huggingface"),
        )
    except Exception as exc:
        logger.warning("Could not fetch weights from %s (%s); using the bundled copy.", repo, exc)
        _HUB_WEIGHTS_STATUS.update({"source": "bundled", "repo": repo, "error": str(exc)})
        return

    os.environ["VRPTW_TRANSFER_WEIGHTS"] = path
    _HUB_WEIGHTS_STATUS.update({"source": "huggingface", "repo": repo, "file": filename, "path": path})
    logger.info("Loaded transfer weights from Hugging Face repo %s (%s).", repo, filename)


@app.on_event("startup")
async def _startup() -> None:
    if not _expected_token():
        logger.warning("SOLVER_API_TOKEN is unset - this service accepts unauthenticated solve requests.")

    _sync_weights_from_hub()

    # Importing torch takes seconds; do it now so the first real request is not
    # the one that pays for it.
    try:
        from services.solver_service import _load_solver_runtime

        _load_solver_runtime()
        logger.info("Solver runtime preloaded.")
    except Exception as exc:
        logger.error("Solver runtime failed to preload: %s", exc)


# ── Schemas ───────────────────────────────────────────────────────────────────


class SolveRequest(BaseModel):
    payload: JobRequest
    matrix: list[list[float]] | None = None


class DynamicInsertRequest(BaseModel):
    dataset: str = "C101"
    customer_id: int
    existing_routes: list[list[int]]


# ── Routes ────────────────────────────────────────────────────────────────────


@app.get("/health")
async def health() -> dict[str, Any]:
    from services.solver_service import device_summary, transfer_weights_summary

    return {
        "status": "ok",
        "service": "vrptw-solver",
        "torch": device_summary(),
        "model": transfer_weights_summary(),
        "weights_origin": _HUB_WEIGHTS_STATUS,
        "authenticated": bool(_expected_token()),
    }


@app.post("/solve", dependencies=[Depends(require_token)])
async def solve(body: SolveRequest) -> dict[str, Any]:
    from services.solver_service import solve_model

    return await solve_model(body.payload, body.matrix)


@app.post("/reoptimize", dependencies=[Depends(require_token)])
async def reoptimize(body: ReoptimizeRequest) -> dict[str, Any]:
    from services.research_adapter import build_inst, plan_to_payload

    from vrptw import Plan
    from vrptw.local_search import td_converge_polish

    try:
        inst = build_inst(body.customers, capacity=body.fleet.capacity, name="Reoptimize")
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    id_to_idx = {c.id: idx for idx, c in enumerate(body.customers) if c.id is not None}

    mapped_routes = []
    for route in body.routes:
        mapped_route = [id_to_idx[cid] for cid in route if cid in id_to_idx and id_to_idx[cid] != 0]
        if mapped_route:
            mapped_routes.append(mapped_route)

    plan = Plan(mapped_routes, inst, algo="manual")
    polished = td_converge_polish(plan, max_passes=25)
    result = plan_to_payload(polished, body.customers, 0.0)
    result["feasible"] = polished.feasible
    return result


@app.post("/dynamic_insert", dependencies=[Depends(require_token)])
async def dynamic_insert(body: DynamicInsertRequest) -> dict[str, Any]:
    from services.solomon_service import load_solomon_dataset, to_inst_payload

    from vrptw.config import Config
    from vrptw.core import Inst, Plan
    from vrptw.solvers import HybridDDQNSolver

    try:
        inst = Inst(to_inst_payload(load_solomon_dataset(body.dataset)))
    except (ValueError, FileNotFoundError) as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    if not 1 <= body.customer_id <= inst.n:
        raise HTTPException(
            status_code=422,
            detail=f"customer_id must be between 1 and {inst.n} for dataset {body.dataset}; got {body.customer_id}.",
        )

    routed = {c for route in body.existing_routes for c in route}
    out_of_range = sorted(c for c in routed if not 1 <= c <= inst.n)
    if out_of_range:
        raise HTTPException(
            status_code=422,
            detail=(
                f"existing_routes contains ids outside 1..{inst.n} for dataset {body.dataset}: {out_of_range[:10]}."
            ),
        )
    if body.customer_id in routed:
        raise HTTPException(
            status_code=409,
            detail=f"customer_id {body.customer_id} is already served by existing_routes.",
        )

    plan = Plan(body.existing_routes, inst)
    solver = HybridDDQNSolver(inst, Config())
    updated = solver.insert_dynamic_customer(plan, body.customer_id)
    return {
        "dataset": body.dataset,
        "inserted_customer": body.customer_id,
        "routes": updated.routes,
        "nv": updated.nv,
        "td": updated.cost,
        "pareto_metrics": updated.calculate_pareto_metrics(),
    }

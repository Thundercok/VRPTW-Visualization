# ruff: noqa: E402
from __future__ import annotations

import asyncio
import json
import multiprocessing as mp
import os
import sys
import threading
import time
from concurrent.futures import ProcessPoolExecutor
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

from fastapi import APIRouter, Depends, File, HTTPException, Query, Request, UploadFile
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

# Ensure src is in sys.path for importing the vrptw package.
_ROOT_PATH = Path(__file__).resolve().parents[4]
_SRC_PATH = _ROOT_PATH / "src"
if str(_SRC_PATH) not in sys.path:
    sys.path.insert(0, str(_SRC_PATH))

if TYPE_CHECKING:
    import vrptw

from api.dependencies import require_user
from core.config import demo_auth_bypass_enabled
from core.firebase import is_firebase_enabled
from core.rate_limit import GEOCODE_LIMIT, JOBS_LIMIT, limiter
from models.schemas import JobRequest, MatrixRequest, ReoptimizeRequest
from services.compute_gateway import call_remote, remote_enabled, remote_health
from services.geocode_service import bulk_geocode_addresses, geocode_address, reverse_geocode_address
from services.job_service import job_service
from services.matrix_service import calculate_matrix, fetch_route_geometry
from services.solomon_service import list_solomon_datasets, load_solomon_dataset
from services.solver_service import device_summary, transfer_weights_summary
from services.text_block_parser import is_vietnamese_text_block, parse_vietnamese_text_block

router = APIRouter(tags=["ops"])

_LOGS_PATH = _ROOT_PATH / "docs" / "logs"


def _parse_result_version(folder_name: str) -> str | None:
    if not folder_name.startswith("results-v"):
        return None
    version = folder_name.replace("results-", "", 1)
    if not version.startswith("v"):
        return None
    if not all(ch.isdigit() or ch == "." or ch == "v" for ch in version):
        return None
    return version


def _version_key(version: str) -> tuple[int, ...]:
    core = version.lstrip("v")
    parts = []
    for token in core.split("."):
        try:
            parts.append(int(token))
        except ValueError:
            parts.append(0)
    return tuple(parts)


@router.get("/health")
async def health() -> dict[str, object]:
    fb = is_firebase_enabled()
    bypass = demo_auth_bypass_enabled()
    payload: dict[str, object] = {
        "status": "ok",
        "firebase_enabled": fb,
        "demo_auth_bypass": bypass,
        "demo_mode": (not fb) and bypass,
        "torch": device_summary(),
        "model": transfer_weights_summary(),
    }
    if remote_enabled():
        # In the Render deployment the local torch/model fields are both empty
        # by design; the Space is what actually answers a solve.
        payload["remote_solver"] = await remote_health()
    return payload


@router.get("/geocode")
@limiter.limit(GEOCODE_LIMIT)
async def geocode(
    request: Request,
    q: str = Query(min_length=2),
    limit: int = Query(default=5, ge=1, le=10),
) -> dict[str, Any]:
    return await geocode_address(q, limit)


@router.get("/reverse-geocode")
@limiter.limit(GEOCODE_LIMIT)
async def reverse_geocode(request: Request, lat: float = Query(), lng: float = Query()) -> dict[str, Any]:
    return await reverse_geocode_address(lat, lng)


@router.get("/solomon/list")
async def solomon_list(
    _: dict[str, str] = Depends(require_user),
) -> dict[str, Any]:
    return {"datasets": list_solomon_datasets()}


@router.get("/solomon")
async def solomon_dataset(
    name: str = Query(default="demo", min_length=2, max_length=10),
    _: dict[str, str] = Depends(require_user),
) -> dict[str, Any]:
    try:
        return load_solomon_dataset(name)
    except FileNotFoundError as exc:
        raise HTTPException(
            status_code=404,
            detail=(
                f"{exc} Solomon benchmark files are not bundled with this repo. "
                "Run `python scripts/fetch_solomon.py` to download them, use 'demo' "
                "for the built-in mini sample, or upload your own customers via "
                "Real Data import."
            ),
        ) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.post("/solomon/import-csv")
async def import_csv_file(
    file: UploadFile = File(...),
    _: dict[str, str] = Depends(require_user),
) -> dict[str, Any]:
    content = await file.read()
    try:
        text = content.decode("utf-8")
    except UnicodeDecodeError:
        try:
            text = content.decode("latin1")
        except Exception as exc:
            raise HTTPException(status_code=400, detail="Invalid file encoding.") from exc

    import csv
    import io
    import unicodedata

    def normalize_header(h: str) -> str:
        s = str(h or "").strip().lower()
        s = "".join(c for c in unicodedata.normalize("NFD", s) if not unicodedata.combining(c))
        return "".join(c for c in s if c.isalnum())

    def find_col_index(headers: list[str], aliases: list[str]) -> int:
        norm_aliases = [normalize_header(a) for a in aliases]
        for idx, h in enumerate(headers):
            if normalize_header(h) in norm_aliases:
                return idx
        return -1

    text_io = io.StringIO(text)
    sample = text[:2048]
    delimiter = ","
    if "\t" in sample:
        delimiter = "\t"
    elif ";" in sample:
        delimiter = ";"

    reader = csv.reader(text_io, delimiter=delimiter)
    rows = [r for r in reader if r]

    if not rows:
        raise HTTPException(status_code=400, detail="Empty CSV file.")

    # ── Auto-detect Vietnamese text-block format ─────────────────────
    if is_vietnamese_text_block(text):
        return await _geocode_text_block(text)

    headers = [str(cell).strip() for cell in rows[0]]

    name_idx = find_col_index(headers, ["name", "customer name", "customer", "client", "store", "shop"])
    addr_idx = find_col_index(headers, ["address", "addr", "location", "customer address", "full address"])
    lat_idx = find_col_index(headers, ["lat", "latitude", "y", "geo lat"])
    lng_idx = find_col_index(headers, ["lng", "lon", "long", "longitude", "x", "geo lng", "geo lon"])
    demand_idx = find_col_index(headers, ["demand", "qty", "quantity", "load", "order size", "weight"])
    ready_idx = find_col_index(headers, ["ready", "readytime", "open", "tw start", "twstart", "earliest", "start"])
    due_idx = find_col_index(
        headers, ["due", "duedate", "duetime", "close", "tw end", "twend", "latest", "end", "deadline"]
    )
    service_idx = find_col_index(headers, ["service", "servicetime", "svc", "dwell", "stoptime"])

    if (lat_idx >= 0 or lng_idx >= 0 or addr_idx >= 0) and (name_idx >= 0 or demand_idx >= 0):
        data_rows = rows[1:]
    else:
        name_idx, addr_idx, lat_idx, lng_idx, demand_idx, ready_idx, due_idx, service_idx = 0, 1, 2, 3, 4, 5, 6, 7
        data_rows = rows

    customers = []
    for idx, row in enumerate(data_rows):
        row_len = len(row)

        def val_at(col_idx: int, default: str = "", row=row, row_len=row_len) -> str:
            if 0 <= col_idx < row_len:
                return str(row[col_idx]).strip()
            return default

        name = val_at(name_idx) or f"Cust-{idx + 1}"
        address = val_at(addr_idx)
        lat_str = val_at(lat_idx)
        lng_str = val_at(lng_idx)
        demand_str = val_at(demand_idx, "10")
        ready_str = val_at(ready_idx, "0")
        due_str = val_at(due_idx, "1000")
        service_str = val_at(service_idx, "10")

        try:
            lat = float(lat_str) if lat_str else None
            lng = float(lng_str) if lng_str else None
        except ValueError:
            lat, lng = None, None

        try:
            demand = int(float(demand_str))
        except ValueError:
            demand = 10

        try:
            ready = float(ready_str)
        except ValueError:
            ready = 0.0

        try:
            due = float(due_str)
        except ValueError:
            due = 1000.0

        try:
            service = float(service_str)
        except ValueError:
            service = 10.0

        is_depot = idx == 0 and demand == 0
        customers.append(
            {
                "id": idx,
                "name": name,
                "address": address or f"Point {idx}",
                "lat": lat,
                "lng": lng,
                "demand": 0 if is_depot else demand,
                "ready": ready,
                "due": due,
                "service": service,
                "isDepot": is_depot,
                "priority": "Normal",
                "skill": "None",
            }
        )

    # Bulk geocode missing coords
    to_geocode = [c for c in customers if c["lat"] is None and c["address"]]
    if to_geocode:
        addresses = [c["address"] for c in to_geocode]
        results = await bulk_geocode_addresses(addresses)
        for cust, geo in zip(to_geocode, results):
            if geo.get("items"):
                cust["lat"] = float(geo["items"][0]["lat"])
                cust["lng"] = float(geo["items"][0]["lng"])

    # Filter out any that still don't have coords
    customers = [c for c in customers if c["lat"] is not None and c["lng"] is not None]

    return {"customers": customers}


async def _geocode_text_block(text: str) -> dict[str, Any]:
    """Shared helper: parse Vietnamese text-block, geocode, return customers."""
    customers = parse_vietnamese_text_block(text)
    total = len(customers)
    geocoded = 0

    to_geocode = []
    for cust in customers:
        if cust["lat"] is None and cust.get("address"):
            to_geocode.append(cust)

    if to_geocode:
        addresses = [c["address"] for c in to_geocode]
        results = await bulk_geocode_addresses(addresses)
        for cust, geo in zip(to_geocode, results):
            if geo.get("items"):
                cust["lat"] = float(geo["items"][0]["lat"])
                cust["lng"] = float(geo["items"][0]["lng"])
                geocoded += 1

    customers = [c for c in customers if c["lat"] is not None and c["lng"] is not None]
    return {"customers": customers, "geocoded_count": geocoded, "total_count": total}


@router.post("/solomon/import-text")
async def import_text_file(
    file: UploadFile = File(...),
    _: dict[str, str] = Depends(require_user),
) -> dict[str, Any]:
    """Parse a Vietnamese text-block file and auto-geocode addresses."""
    content = await file.read()
    try:
        text = content.decode("utf-8")
    except UnicodeDecodeError:
        try:
            text = content.decode("latin1")
        except Exception as exc:
            raise HTTPException(status_code=400, detail="Invalid file encoding.") from exc

    if not is_vietnamese_text_block(text):
        raise HTTPException(
            status_code=400,
            detail="File does not appear to be in Vietnamese text-block format. Use /solomon/import-csv for CSV files.",
        )

    return await _geocode_text_block(text)


@router.get("/analysis/versions")
async def analysis_versions(_: dict[str, str] = Depends(require_user)) -> dict[str, Any]:
    items: list[dict[str, Any]] = []
    if not _LOGS_PATH.exists():
        return {"items": [], "default": None}

    # New flat layout: logs/nexus_demo.json
    flat_nexus = _LOGS_PATH / "nexus_demo.json"
    if flat_nexus.exists():
        version = "v9.5"
        try:
            with flat_nexus.open("r", encoding="utf-8") as handle:
                payload = json.load(handle)
            meta_version = str(payload.get("meta", {}).get("version", "")).strip()
            if meta_version:
                version = meta_version
        except Exception:
            pass
        modified = datetime.fromtimestamp(flat_nexus.stat().st_mtime, tz=UTC).isoformat()
        items.append(
            {
                "version": version.lower(),
                "folder": "logs",
                "nexus_file": flat_nexus.name,
                "updated_at": modified,
            }
        )

    # Legacy layout: logs/results-vX/nexus_demo.json
    for child in _LOGS_PATH.iterdir():
        if not child.is_dir():
            continue
        version = _parse_result_version(child.name)
        if not version:
            continue
        nexus_path = child / "nexus_demo.json"
        if not nexus_path.exists():
            continue

        modified = datetime.fromtimestamp(nexus_path.stat().st_mtime, tz=UTC).isoformat()
        items.append(
            {
                "version": version,
                "folder": child.name,
                "nexus_file": str(nexus_path.name),
                "updated_at": modified,
            }
        )

    items.sort(key=lambda item: _version_key(item["version"]), reverse=True)
    default = items[0]["version"] if items else None
    return {"items": items, "default": default}


@router.get("/analysis/nexus")
async def analysis_nexus(
    version: str = Query(min_length=2, max_length=20),
    _: dict[str, str] = Depends(require_user),
) -> dict[str, Any]:
    normalized = version.strip().lower()
    if not normalized.startswith("v"):
        raise HTTPException(status_code=400, detail="Version must start with v, for example v9.5")
    if not all(ch.isdigit() or ch == "." or ch == "v" for ch in normalized):
        raise HTTPException(status_code=400, detail="Version format is invalid")

    file_path = _LOGS_PATH / "nexus_demo.json"
    folder = _LOGS_PATH
    if not file_path.exists():
        folder = _LOGS_PATH / f"results-{normalized}"
        file_path = folder / "nexus_demo.json"
    if not file_path.exists():
        raise HTTPException(status_code=404, detail=f"Cannot find nexus_demo.json for version {normalized}")

    with file_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)

    payload["_source"] = {
        "version": normalized,
        "folder": folder.name,
        "file": file_path.name,
    }
    return payload


@router.get("/analysis/activity")
async def analysis_activity(
    hours: int = Query(default=24, ge=1, le=168),
    _: dict[str, str] = Depends(require_user),
) -> dict[str, Any]:
    return job_service.get_activity(hours)


@router.post("/matrix")
async def matrix(body: MatrixRequest, _: dict[str, str] = Depends(require_user)) -> dict[str, Any]:
    return await calculate_matrix(body.points)


@router.get("/route-geometry")
async def route_geometry(coords: str = Query(..., description="Semicolon-separated lng,lat points")) -> dict[str, Any]:
    try:
        return await fetch_route_geometry(coords)
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"OSRM geometry fetch failed: {exc}") from exc


@router.post("/reoptimize")
async def reoptimize(
    body: ReoptimizeRequest,
    _: dict[str, str] = Depends(require_user),
) -> dict[str, Any]:
    if remote_enabled():
        return await call_remote("/reoptimize", body.model_dump())

    try:
        from services.research_adapter import build_inst, plan_to_payload

        from vrptw import Plan
        from vrptw.local_search import td_converge_polish
    except ImportError as exc:
        raise HTTPException(
            status_code=503,
            detail=f"Solver is unavailable because research dependencies are missing: {exc}",
        ) from exc

    try:
        inst = build_inst(body.customers, capacity=body.fleet.capacity, name="Reoptimize", dataset=body.dataset)
    except ValueError as val_err:
        raise HTTPException(status_code=400, detail=str(val_err)) from val_err

    id_to_idx = {c.id: idx for idx, c in enumerate(body.customers) if c.id is not None}

    mapped_routes = []
    for r in body.routes:
        mapped_route = []
        for cid in r:
            if cid in id_to_idx:
                node_idx = id_to_idx[cid]
                if node_idx != 0:
                    mapped_route.append(node_idx)
        if mapped_route:
            mapped_routes.append(mapped_route)

    plan = Plan(mapped_routes, inst, algo="manual")
    polished_plan = td_converge_polish(plan, max_passes=25)
    res = plan_to_payload(polished_plan, body.customers, 0.0)
    res["feasible"] = polished_plan.feasible
    return res


@router.post("/jobs")
@limiter.limit(JOBS_LIMIT)
async def submit_job(
    request: Request,
    body: JobRequest,
    _: dict[str, str] = Depends(require_user),
) -> dict[str, str]:
    return await job_service.submit(body)


@router.get("/jobs/{job_id}")
async def get_job(job_id: str, _: dict[str, str] = Depends(require_user)) -> dict[str, Any]:
    return job_service.get(job_id)


@router.get("/jobs/{job_id}/debug")
async def get_job_debug(job_id: str, _: dict[str, str] = Depends(require_user)) -> dict[str, Any]:
    return job_service.get_debug(job_id)


# ── BENCHMARK, TRAINING, & SMOKE TEST ENDPOINTS ──────────────────────────────────


class BenchmarkSubmitRequest(BaseModel):
    dataset: str
    n_runs: int
    max_wall_hours: float
    algorithms: list[str]


class DRTrainSubmitRequest(BaseModel):
    epochs: int


class TransferTrainSubmitRequest(BaseModel):
    dataset: str
    epochs: int


class TaskManager:
    def __init__(self):
        self.benchmark_state = {"status": "idle", "progress": 0.0, "error": None, "results": []}
        self.training_state = {
            "status": "idle",
            "type": None,
            "progress": 0.0,
            "error": None,
            "logs": [],
            "epochs_completed": 0,
            "total_epochs": 0,
        }
        self.smoke_test_state = {"status": "idle", "error": None, "results": None}
        self.lock = threading.Lock()


task_manager = TaskManager()


class LogCapture:
    def __init__(self, log_list):
        self.log_list = log_list
        self.stdout = sys.stdout

    def write(self, text):
        self.stdout.write(text)
        if text.strip():
            self.log_list.append(text.strip())

    def flush(self):
        self.stdout.flush()


def load_weights_for_algo(algo: str, cfg: vrptw.Config) -> dict | None:
    import vrptw

    if algo == "hybrid_ddqn_transfer_rc1":
        label = "rc1"
    elif algo == "hybrid_ddqn_transfer_rc2":
        label = "rc2"
    elif algo == "hybrid_ddqn_transfer_dr":
        label = "dr"
    else:
        return None

    candidates = []
    if label == "dr":
        candidates = [
            os.path.join(cfg.output_dir, "rl_alns_dr_v15"),
            os.path.join(cfg.output_dir, "rl_alns_dr_v15.safetensors"),
            os.path.join(cfg.output_dir, "rl_alns_dr_v15.pt"),
            os.path.join(str(_ROOT_PATH), "rl_alns_dr_v15"),
            os.path.join(str(_ROOT_PATH), "docs", "rl_alns_dr_v15"),
        ]
    else:
        candidates = [
            os.path.join(cfg.output_dir, f"rl_alns_transfer_{label}_v15"),
            os.path.join(cfg.output_dir, f"rl_alns_transfer_{label}_v15.safetensors"),
            os.path.join(str(_ROOT_PATH), "docs", "model", "rl_alns_transfer"),
            os.path.join(str(_ROOT_PATH), "logs", "results-v9.7", "rl_alns_transfer"),
            os.path.join(str(_ROOT_PATH), "logs", "results-v9.8", "rl_alns_transfer"),
        ]

    for cand in candidates:
        stem = cand
        if stem.endswith(".safetensors"):
            stem = stem[:-12]
        elif stem.endswith(".pt"):
            stem = stem[:-3]

        weights = vrptw._load_weights(stem)
        if weights is not None:
            print(f"Loaded weights for {algo} from {stem}")
            return weights

    print(f"No weights found for {algo}. Running with default weights.")
    return None


def run_benchmark_thread(dataset_key: str, algorithms: list[str], n_runs: int, max_wall_hours: float):
    import numpy as np

    import vrptw

    global task_manager
    try:
        cfg = vrptw.Config()
        cfg.data_path = str(_ROOT_PATH / "data" / "Solomon")
        cfg.output_dir = str(_LOGS_PATH)
        cfg.n_runs = n_runs
        cfg.max_wall_hours = max_wall_hours

        datasets = vrptw.load_datasets(cfg.data_path)
        ds_key = dataset_key.lower()
        if ds_key == "c":
            instances = datasets.get("c1", []) + datasets.get("c2", [])
        elif ds_key == "r":
            instances = datasets.get("r1", []) + datasets.get("r2", [])
        elif ds_key in datasets:
            instances = datasets[ds_key]
        else:
            instances = datasets.get(ds_key, [])

        if not instances:
            raise ValueError(f"No instances found for dataset key: {dataset_key}")

        weights_by_algo = {}
        for algo in algorithms:
            if algo in ("hybrid_ddqn_transfer_rc1", "hybrid_ddqn_transfer_rc2", "hybrid_ddqn_transfer_dr"):
                weights_by_algo[algo] = load_weights_for_algo(algo, cfg)

        archive = vrptw.EliteArchive(k=cfg.elite_archive_k)
        total_combos = len(instances) * len(algorithms)
        completed_combos = 0

        with task_manager.lock:
            task_manager.benchmark_state["status"] = "running"
            task_manager.benchmark_state["progress"] = 0.0
            task_manager.benchmark_state["error"] = None
            task_manager.benchmark_state["results"] = []

        wall_start = time.time()
        results_rows = []
        n_workers = min(cfg.n_runs, max(1, os.cpu_count() // 2))

        for inst in instances:
            elapsed_h = (time.time() - wall_start) / 3600
            if elapsed_h >= cfg.max_wall_hours:
                break

            dataset_name = "RC1" if inst.name[2] == "1" else "RC2"
            if inst.name.startswith("C"):
                dataset_name = "C"
            elif inst.name.startswith("R"):
                dataset_name = "R"

            for algo in algorithms:
                algo_label = algo
                if algo == "ortools":
                    algo_label = vrptw.ALGO_ORTOOLS
                elif algo == "alns_base":
                    algo_label = vrptw.ALGO_ALNS_BASE
                elif algo == "hybrid_fixed":
                    algo_label = vrptw.ALGO_HYBRID_FIXED
                elif algo == "hybrid_ddqn":
                    algo_label = vrptw.ALGO_HYBRID_DDQN
                elif algo == "hybrid_ddqn_transfer_rc1":
                    algo_label = vrptw.ALGO_HYBRID_DDQN_TRANSFER
                elif algo == "hybrid_ddqn_transfer_dr":
                    algo_label = vrptw.ALGO_HYBRID_DDQN_TRANSFER_DR

                algo_canonical = vrptw.canonical_algo_label(algo_label)
                nv_v, cost_v, time_v, gap_v, nvd_v, ot_v = [], [], [], [], [], []
                n_runs_eff = 1 if algo_canonical == vrptw.ALGO_ORTOOLS else cfg.n_runs

                transfer_weights = weights_by_algo.get(algo)
                worker_args = [
                    (
                        inst,
                        algo_canonical,
                        cfg,
                        cfg.seed + i,
                        transfer_weights,
                        vrptw._diversified_init(i, inst, archive, cfg),
                    )
                    for i in range(n_runs_eff)
                ]

                _n_workers = 1 if algo_canonical == vrptw.ALGO_ORTOOLS else n_workers
                ctx = mp.get_context("spawn")
                with ProcessPoolExecutor(max_workers=_n_workers, mp_context=ctx) as ex:
                    run_results = list(ex.map(vrptw._benchmark_worker, worker_args))

                for res, plan in run_results:
                    if plan is not None:
                        archive.update(plan)
                    time_v.append(res["time"])
                    if res["nv"] is not None:
                        nv_v.append(res["nv"])
                        cost_v.append(res["cost"])
                        gap_v.append(res["td_gap"])
                        nvd_v.append(res["nv_diff"])
                        ot_v.append(res["on_time"])

                if nv_v:
                    bks = vrptw.BKS.get(inst.name)
                    nv_inflated = (
                        bks is not None
                        and float(np.mean(nv_v)) > bks["nv"] + 0.4
                        and gap_v[0] is not None
                        and float(np.mean(gap_v)) < 0
                    )

                    row = {
                        "dataset": dataset_name,
                        "instance": inst.name,
                        "algorithm": algo_canonical,
                        "nv": round(float(np.mean(nv_v)), 2),
                        "nv_std": round(float(np.std(nv_v)), 2),
                        "td": round(float(np.mean(cost_v)), 2),
                        "td_std": round(float(np.std(cost_v)), 2),
                        "gap": round(float(np.mean(gap_v)), 2) if gap_v[0] is not None else 0.0,
                        "time": round(float(np.mean(time_v)), 1),
                        "nv_inflated": nv_inflated,
                    }
                    results_rows.append(row)

                completed_combos += 1
                with task_manager.lock:
                    task_manager.benchmark_state["progress"] = round((completed_combos / total_combos) * 100, 1)
                    task_manager.benchmark_state["results"] = results_rows

        with task_manager.lock:
            task_manager.benchmark_state["status"] = "done"
            task_manager.benchmark_state["progress"] = 100.0
    except Exception as e:
        import traceback

        print(f"Error in benchmark execution: {e}\n{traceback.format_exc()}")
        with task_manager.lock:
            task_manager.benchmark_state["status"] = "failed"
            task_manager.benchmark_state["error"] = str(e)


def run_training_thread(train_type: str, dataset_key: str | None = None, epochs: int = 1):
    import vrptw

    global task_manager
    try:
        cfg = vrptw.Config()
        cfg.output_dir = str(_LOGS_PATH)

        logs = []
        old_stdout = sys.stdout

        class LogCaptureWithProgress(LogCapture):
            def write(self, text):
                super().write(text)
                stripped = text.strip()
                if not stripped:
                    return
                import re

                match = re.search(r"Epoch\s+(\d+)/(\d+)", stripped, re.IGNORECASE)
                if match:
                    completed = int(match.group(1))
                    total = int(match.group(2))
                    with task_manager.lock:
                        task_manager.training_state["epochs_completed"] = completed
                        task_manager.training_state["total_epochs"] = total
                        task_manager.training_state["progress"] = round((completed / total) * 100, 1)

        capturer = LogCaptureWithProgress(logs)
        sys.stdout = capturer

        try:
            if train_type == "dr":
                cfg.domain_randomization_epochs = epochs
                cfg.domain_randomization_batch = 4  # moderate batch for testing
                with task_manager.lock:
                    task_manager.training_state["status"] = "running"
                    task_manager.training_state["type"] = "dr"
                    task_manager.training_state["total_epochs"] = epochs
                    task_manager.training_state["epochs_completed"] = 0
                    task_manager.training_state["progress"] = 0.0
                    task_manager.training_state["logs"] = logs
                    task_manager.training_state["error"] = None

                vrptw.train_domain_randomization(cfg, seed=42)

            elif train_type == "transfer":
                cfg.transfer_epochs = epochs
                with task_manager.lock:
                    task_manager.training_state["status"] = "running"
                    task_manager.training_state["type"] = "transfer"
                    task_manager.training_state["total_epochs"] = epochs
                    task_manager.training_state["epochs_completed"] = 0
                    task_manager.training_state["progress"] = 0.0
                    task_manager.training_state["logs"] = logs
                    task_manager.training_state["error"] = None

                datasets = vrptw.load_datasets(cfg.data_path)
                instances = datasets.get(dataset_key.lower() if dataset_key else "rc1", [])
                if not instances:
                    raise ValueError(f"No instances found for dataset {dataset_key}")

                vrptw.train_transfer_model(instances, cfg, seed=42, label=dataset_key.upper())

            with task_manager.lock:
                task_manager.training_state["status"] = "done"
                task_manager.training_state["progress"] = 100.0
        except Exception as e:
            import traceback

            err_str = f"Error in training: {e}\n{traceback.format_exc()}"
            print(err_str)
            with task_manager.lock:
                task_manager.training_state["status"] = "failed"
                task_manager.training_state["error"] = str(e)
        finally:
            sys.stdout = old_stdout
    except Exception as e:
        with task_manager.lock:
            task_manager.training_state["status"] = "failed"
            task_manager.training_state["error"] = str(e)


def run_smoke_test_thread():
    import vrptw

    global task_manager
    try:
        cfg = vrptw.Config()
        cfg.data_path = str(_ROOT_PATH / "data" / "Solomon")

        datasets = vrptw.load_datasets(cfg.data_path)
        rc1 = datasets.get("rc1", [])
        if not rc1:
            raise ValueError("No Solomon instances found to run smoke test.")
        inst = rc1[0]

        with task_manager.lock:
            task_manager.smoke_test_state["status"] = "running"
            task_manager.smoke_test_state["error"] = None
            task_manager.smoke_test_state["results"] = None

        results = vrptw.smoke_test(inst, seed=42)
        serializable_results = []
        for algo, (gap, elapsed) in results.items():
            serializable_results.append({"algorithm": algo, "gap": gap, "time": round(elapsed, 2)})

        with task_manager.lock:
            task_manager.smoke_test_state["status"] = "done"
            task_manager.smoke_test_state["results"] = serializable_results
    except Exception as e:
        import traceback

        print(f"Error in smoke test: {e}\n{traceback.format_exc()}")
        with task_manager.lock:
            task_manager.smoke_test_state["status"] = "failed"
            task_manager.smoke_test_state["error"] = str(e)


def _reject_if_research_offloaded(operation: str) -> None:
    """Block the long-running research jobs when the solver lives off-box.

    Benchmarks and training runs take hours and write into ``docs/logs``. They
    are local research operations, not something the slim API container or a
    request-scoped Space call can carry, so fail loudly instead of starting a
    thread that dies on ``import vrptw``.
    """
    if remote_enabled():
        raise HTTPException(
            status_code=503,
            detail=(
                f"{operation} is disabled in this deployment: the solver runs on a remote "
                "compute service and this operation must be run locally against the research stack."
            ),
        )


@router.post("/benchmark")
async def start_benchmark(
    body: BenchmarkSubmitRequest,
    _: dict[str, str] = Depends(require_user),
):
    _reject_if_research_offloaded("Benchmarking")
    global task_manager
    with task_manager.lock:
        if task_manager.benchmark_state["status"] == "running":
            raise HTTPException(status_code=400, detail="Benchmark is already running")

    thread = threading.Thread(
        target=run_benchmark_thread, args=(body.dataset, body.algorithms, body.n_runs, body.max_wall_hours), daemon=True
    )
    thread.start()
    return {"message": "Benchmark started successfully"}


@router.get("/benchmark/status")
async def get_benchmark_status(
    _: dict[str, str] = Depends(require_user),
):
    global task_manager
    with task_manager.lock:
        return task_manager.benchmark_state


@router.post("/train/dr")
async def start_train_dr(
    body: DRTrainSubmitRequest,
    _: dict[str, str] = Depends(require_user),
):
    _reject_if_research_offloaded("Domain randomization training")
    global task_manager
    with task_manager.lock:
        if task_manager.training_state["status"] == "running":
            raise HTTPException(status_code=400, detail="Training is already running")

    thread = threading.Thread(target=run_training_thread, args=("dr", None, body.epochs), daemon=True)
    thread.start()
    return {"message": "Domain randomization training started"}


@router.post("/train/transfer")
async def start_train_transfer(
    body: TransferTrainSubmitRequest,
    _: dict[str, str] = Depends(require_user),
):
    _reject_if_research_offloaded("Transfer learning training")
    global task_manager
    with task_manager.lock:
        if task_manager.training_state["status"] == "running":
            raise HTTPException(status_code=400, detail="Training is already running")

    thread = threading.Thread(target=run_training_thread, args=("transfer", body.dataset, body.epochs), daemon=True)
    thread.start()
    return {"message": "Transfer learning training started"}


@router.get("/train/status")
async def get_train_status(
    _: dict[str, str] = Depends(require_user),
):
    global task_manager
    with task_manager.lock:
        return task_manager.training_state


@router.post("/smoke-test")
async def start_smoke_test(
    _: dict[str, str] = Depends(require_user),
):
    _reject_if_research_offloaded("Smoke test")
    global task_manager
    with task_manager.lock:
        if task_manager.smoke_test_state["status"] == "running":
            raise HTTPException(status_code=400, detail="Smoke test is already running")

    thread = threading.Thread(target=run_smoke_test_thread, daemon=True)
    thread.start()
    return {"message": "Smoke test started"}


@router.get("/smoke-test/status")
async def get_smoke_test_status(
    _: dict[str, str] = Depends(require_user),
):
    global task_manager
    with task_manager.lock:
        return task_manager.smoke_test_state


class DynamicInsertRequest(BaseModel):
    dataset: str = "C101"
    customer_id: int
    existing_routes: list[list[int]]


@router.post("/solve/dynamic_insert")
async def solve_dynamic_insert(
    body: DynamicInsertRequest,
    _: dict[str, str] = Depends(require_user),
):
    """
    Real-time dynamic order insertion endpoint without full solver restart.
    """
    if remote_enabled():
        return await call_remote("/dynamic_insert", body.model_dump())

    from services.solomon_service import load_solomon_dataset, to_inst_payload

    from vrptw.config import Config
    from vrptw.core import Inst, Plan
    from vrptw.solvers import HybridDDQNSolver

    try:
        inst = Inst(to_inst_payload(load_solomon_dataset(body.dataset)))
    except (ValueError, FileNotFoundError) as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    # Customers are nodes 1..inst.n (node 0 is the depot). Without this check an
    # out-of-range id raises IndexError out of the handler as a 500, and a
    # negative id silently wraps under numpy indexing and inserts the wrong
    # customer while echoing the requested id back to the caller.
    if not 1 <= body.customer_id <= inst.n:
        raise HTTPException(
            status_code=422,
            detail=(f"customer_id must be between 1 and {inst.n} for dataset {body.dataset}; got {body.customer_id}."),
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
    # Re-inserting an already-routed customer would serve it twice and report
    # nv/td/pareto_metrics over that invalid plan without complaint.
    if body.customer_id in routed:
        raise HTTPException(
            status_code=409,
            detail=f"customer_id {body.customer_id} is already served by existing_routes.",
        )

    plan = Plan(body.existing_routes, inst)
    solver = HybridDDQNSolver(inst, Config())

    updated_plan = solver.insert_dynamic_customer(plan, body.customer_id)
    return {
        "dataset": body.dataset,
        "inserted_customer": body.customer_id,
        "routes": updated_plan.routes,
        "nv": updated_plan.nv,
        "td": updated_plan.cost,
        "pareto_metrics": updated_plan.calculate_pareto_metrics(),
    }


@router.get("/solve/stream")
async def solve_stream(
    dataset: str = Query(default="C101"),
    iterations: int = Query(default=100, ge=10, le=1000),
    _: dict[str, str] = Depends(require_user),
):
    """
    Server-Sent Events (SSE) transport demo for solver progress.

    NOTE: this endpoint does **not** run the solver. The ``nv``/``td``/``mode``
    values are a fixed synthetic ramp used to exercise the SSE wiring, and every
    payload carries ``"simulated": true`` to say so. Streaming real progress
    needs a per-iteration callback on ``HybridDDQNSolver.solve()`` (which does
    not exist yet) plus running the solve off the event loop; until then, do not
    present these numbers as solver output.
    """

    async def event_generator():
        from services.solomon_service import load_solomon_dataset

        # Parsing the .txt blocks for a moment on a large dataset, so keep it off
        # the event loop. Only the customer count is needed for the synthetic
        # ramp below, so this deliberately avoids building a vrptw ``Inst`` —
        # that would drag the research stack into the slim Render image.
        def _customer_count() -> int:
            return max(0, len(load_solomon_dataset(dataset).get("customers", [])) - 1)

        n_customers = await asyncio.to_thread(_customer_count)

        yield f"data: {json.dumps({'event': 'start', 'dataset': dataset, 'max_iters': iterations, 'simulated': True})}\n\n"

        for step in range(1, 6):
            # await, not time.sleep: this generator runs on the ASGI event loop.
            await asyncio.sleep(0.05)
            payload = {
                "event": "progress",
                "simulated": True,
                "progress_pct": step * 20,
                "current_it": int(step * (iterations / 5)),
                "nv": int(max(1, n_customers // 10)),
                "td": float(round(1000.0 - step * 20.0, 2)),
                "mode": "Default" if step < 3 else "Intensify",
            }
            yield f"data: {json.dumps(payload)}\n\n"

        yield f"data: {json.dumps({'event': 'complete', 'status': 'finished', 'simulated': True})}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")

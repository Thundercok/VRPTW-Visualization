from __future__ import annotations

from typing import Any

import httpx
from models.schemas import MatrixPoint
from services.distance_service import distance_km

# In-memory cache for OSRM route geometries (max 500 entries)
_GEOMETRY_CACHE: dict[str, dict[str, Any]] = {}
_MAX_CACHE_SIZE = 500


async def calculate_matrix(points: list[MatrixPoint]) -> dict[str, Any]:
    if len(points) < 2:
        return {"matrix": [[0.0]], "provider": "none"}

    coords = ";".join(f"{p.lng},{p.lat}" for p in points)
    osrm = f"https://router.project-osrm.org/table/v1/driving/{coords}?annotations=distance"
    headers = {"User-Agent": "NAMI-VRPTW-Research/1.0"}

    try:
        async with httpx.AsyncClient(timeout=10.0, headers=headers) as client:
            response = await client.get(osrm)
            response.raise_for_status()
            data = response.json()
        matrix_km = [[(v or 0.0) / 1000 for v in row] for row in data.get("distances", [])]
        return {"matrix": matrix_km, "provider": "osrm"}
    except Exception:
        geo_points = [(p.lat, p.lng) for p in points]
        fallback: list[list[float]] = []
        for i in geo_points:
            row: list[float] = []
            for j in geo_points:
                row.append(distance_km(i, j))
            fallback.append(row)
        return {"matrix": fallback, "provider": "haversine"}


async def fetch_route_geometry(coords: str) -> dict[str, Any]:
    """Proxy & cache OSRM route geometry for given semicolon-separated lng,lat coordinates."""
    if coords in _GEOMETRY_CACHE:
        return _GEOMETRY_CACHE[coords]

    osrm = f"https://router.project-osrm.org/route/v1/driving/{coords}?overview=full&geometries=geojson"
    headers = {"User-Agent": "NAMI-VRPTW-Research/1.0"}

    async with httpx.AsyncClient(timeout=10.0, headers=headers) as client:
        response = await client.get(osrm)
        response.raise_for_status()
        data = response.json()

    if len(_GEOMETRY_CACHE) >= _MAX_CACHE_SIZE:
        first_key = next(iter(_GEOMETRY_CACHE))
        _GEOMETRY_CACHE.pop(first_key, None)

    _GEOMETRY_CACHE[coords] = data
    return data

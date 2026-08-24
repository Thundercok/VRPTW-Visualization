from __future__ import annotations

from typing import Any

import httpx

REVERSE_GEOCODE_CACHE: dict[tuple[float, float], dict[str, Any]] = {}
GEOCODE_CACHE: dict[str, dict[str, Any]] = {}


def _extract_short_address(data: dict[str, Any]) -> str:
    parts = data.get("address", {}) or {}
    house_no = str(parts.get("house_number", "")).strip()
    road = (
        parts.get("road")
        or parts.get("pedestrian")
        or parts.get("residential")
        or parts.get("hamlet")
        or parts.get("suburb")
        or ""
    )
    road = str(road).strip()
    if house_no and road:
        return f"{house_no} {road}"
    if road:
        return road
    return ""


async def geocode_address(q: str, limit: int) -> dict[str, Any]:
    cache_key = q.strip().lower()
    if limit == 1 and cache_key in GEOCODE_CACHE:
        return GEOCODE_CACHE[cache_key]

    headers = {"User-Agent": "vrptw-dashboard/1.0"}

    async with httpx.AsyncClient(timeout=8.0) as client:
        data = []
        try:
            nominatim_resp = await client.get(
                "https://nominatim.openstreetmap.org/search",
                params={
                    "q": q,
                    "format": "json",
                    "limit": str(limit),
                    "accept-language": "vi,en",
                    "countrycodes": "vn",
                },
                headers=headers,
            )
            nominatim_resp.raise_for_status()
            data = nominatim_resp.json()
        except httpx.HTTPError:
            try:
                mapsco_resp = await client.get(
                    "https://photon.komoot.io/api/",
                    params={"q": f"{q}, Vietnam", "limit": str(limit)},
                    headers=headers,
                )
                mapsco_resp.raise_for_status()
                photon_data = mapsco_resp.json().get("features", [])

                # Convert photon format to the expected format
                data = []
                for feat in photon_data:
                    coords = feat.get("geometry", {}).get("coordinates", [0, 0])
                    props = feat.get("properties", {})
                    name = props.get("name", "")
                    street = props.get("street", "")
                    city = props.get("city", "")
                    display = ", ".join(filter(bool, [name, street, city]))
                    data.append({
                        "display_name": display,
                        "lat": coords[1],
                        "lon": coords[0]
                    })
                data = data[: max(1, int(limit))]
            except httpx.HTTPError:
                data = []

    items = [
        {
            "address": it.get("display_name", ""),
            "lat": float(it.get("lat", 0.0)),
            "lng": float(it.get("lon", 0.0)),
        }
        for it in data
    ]
    result = {"items": items}
    if limit == 1 and items:
        GEOCODE_CACHE[cache_key] = result
    return result


async def bulk_geocode_addresses(addresses: list[str]) -> list[dict[str, Any]]:
    import asyncio
    sem = asyncio.Semaphore(15)
    headers = {"User-Agent": "vrptw-dashboard/1.0"}

    async def fetch_one(addr: str, client: httpx.AsyncClient) -> dict[str, Any]:
        cache_key = addr.strip().lower()
        if cache_key in GEOCODE_CACHE:
            return GEOCODE_CACHE[cache_key]
        async with sem:
            try:
                resp = await client.get(
                    "https://photon.komoot.io/api/",
                    params={"q": f"{addr}, Vietnam", "limit": "1"},
                    headers=headers
                )
                resp.raise_for_status()
                photon_data = resp.json().get("features", [])
                if photon_data:
                    coords = photon_data[0].get("geometry", {}).get("coordinates", [0, 0])
                    res = {"items": [{"lat": coords[1], "lng": coords[0]}]}
                    GEOCODE_CACHE[cache_key] = res
                    return res
            except Exception:
                pass
            return {"items": []}

    async with httpx.AsyncClient(timeout=8.0) as client:
        tasks = [fetch_one(addr, client) for addr in addresses]
        return await asyncio.gather(*tasks)


async def reverse_geocode_address(lat: float, lng: float) -> dict[str, Any]:
    cache_key = (round(float(lat), 6), round(float(lng), 6))
    cached = REVERSE_GEOCODE_CACHE.get(cache_key)
    if cached:
        return cached

    headers = {"User-Agent": "vrptw-dashboard/1.0"}

    async def fetch_nominatim(client: httpx.AsyncClient) -> dict[str, Any]:
        url = "https://nominatim.openstreetmap.org/reverse"
        params = {
            "lat": str(lat),
            "lon": str(lng),
            "format": "jsonv2",
            "addressdetails": "1",
            "accept-language": "vi,en",
        }
        response = await client.get(url, params=params, headers=headers)
        response.raise_for_status()
        return response.json()

    async def fetch_photon(client: httpx.AsyncClient) -> dict[str, Any]:
        url = "https://photon.komoot.io/reverse"
        params = {
            "lat": str(lat),
            "lon": str(lng),
        }
        response = await client.get(url, params=params, headers=headers)
        response.raise_for_status()
        photon_data = response.json().get("features", [])
        if photon_data:
            props = photon_data[0].get("properties", {})
            name = props.get("name", "")
            street = props.get("street", "")
            city = props.get("city", "")
            display = ", ".join(filter(bool, [name, street, city]))
            return {"display_name": display, "address": props}
        return {}

    async def fetch_bigdatacloud(client: httpx.AsyncClient) -> dict[str, Any]:
        url = "https://api.bigdatacloud.net/data/reverse-geocode-client"
        params = {
            "latitude": str(lat),
            "longitude": str(lng),
            "localityLanguage": "vi",
        }
        response = await client.get(url, params=params, headers=headers)
        response.raise_for_status()
        data = response.json()

        locality = str(data.get("locality", "")).strip()
        city = str(data.get("city", "")).strip()
        region = str(data.get("principalSubdivision", "")).strip()
        country = str(data.get("countryName", "")).strip()
        pieces = [p for p in [locality, city, region, country] if p]
        display_name = ", ".join(pieces)

        return {
            "display_name": display_name,
            "address": {
                "suburb": locality,
            },
            "lat": data.get("latitude", lat),
            "lon": data.get("longitude", lng),
        }

    data: dict[str, Any] = {}
    async with httpx.AsyncClient(timeout=8.0) as client:
        try:
            data = await fetch_nominatim(client)
        except httpx.HTTPError:
            try:
                data = await fetch_photon(client)
            except httpx.HTTPError:
                try:
                    data = await fetch_bigdatacloud(client)
                except httpx.HTTPError:
                    data = {}

    short_address = _extract_short_address(data)

    payload = {
        "address": str(data.get("display_name", "")).strip(),
        "short_address": short_address,
        "lat": float(data.get("lat", lat) or lat),
        "lng": float(data.get("lon", lng) or lng),
    }
    REVERSE_GEOCODE_CACHE[cache_key] = payload
    return payload

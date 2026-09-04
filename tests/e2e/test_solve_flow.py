from pathlib import Path

from playwright.sync_api import APIRequestContext, Page, expect

SCREENSHOTS = Path("tests/e2e/screenshots")
SCREENSHOTS.mkdir(parents=True, exist_ok=True)

# FastAPI backend (API endpoints)
API_BASE = "http://127.0.0.1:8000"
# Firebase Hosting emulator (SPA frontend)
HOSTING_BASE = "http://127.0.0.1:5050"


def shot(page: Page, name: str) -> None:
    page.screenshot(path=str(SCREENSHOTS / f"{name}.png"), full_page=True)


def test_landing_page(page: Page):
    """Verify the landing page loads via the hosting emulator."""
    page.goto(HOSTING_BASE)
    page.wait_for_load_state("networkidle")
    shot(page, "01_landing")
    expect(page).not_to_have_title("")


def test_app_loads(page: Page):
    """Verify the app shell renders (auth screen visible)."""
    page.goto(f"{HOSTING_BASE}/app.html")
    page.wait_for_load_state("networkidle")
    shot(page, "02_app_initial")
    expect(page.locator("#auth-screen")).to_be_visible()


def test_api_health(page: Page):
    """Verify the FastAPI backend is reachable."""
    resp = page.request.get(f"{API_BASE}/api/health")
    assert resp.ok


def test_firebase_login(authed_page: Page):
    """Verify Firebase Auth login via emulator succeeds and enters the app."""
    shot(authed_page, "03_logged_in")
    expect(authed_page.locator("#app-shell")).to_be_visible()


def test_solomon_list(authed_api: APIRequestContext):
    """Verify Solomon dataset API returns data with auth token."""
    resp = authed_api.get("/api/solomon/list")
    assert resp.ok, f"Expected 200, got {resp.status}: {resp.text()}"
    data = resp.json()
    datasets = data.get("datasets", data) if isinstance(data, dict) else data
    assert isinstance(datasets, list) and len(datasets) > 0


def test_nexus_v95(authed_api: APIRequestContext):
    """Verify training analysis endpoint returns data with auth token."""
    resp = authed_api.get("/api/analysis/nexus?version=v9.5")
    assert resp.ok, f"Expected 200, got {resp.status}: {resp.text()}"


def test_reoptimize(authed_api: APIRequestContext):
    """Verify manual re-optimization endpoint works."""
    payload = {
        "fleet": {"capacity": 100, "vehicles": 2},
        "customers": [
            {
                "id": 0,
                "name": "Depot",
                "lat": 10.73193,
                "lng": 106.69934,
                "demand": 0,
                "ready": 0,
                "due": 1000,
                "service": 0,
                "isDepot": True,
            },
            {
                "id": 1,
                "name": "Cust1",
                "lat": 10.73293,
                "lng": 106.70034,
                "demand": 10,
                "ready": 0,
                "due": 1000,
                "service": 10,
                "isDepot": False,
            },
            {
                "id": 2,
                "name": "Cust2",
                "lat": 10.73393,
                "lng": 106.70134,
                "demand": 15,
                "ready": 0,
                "due": 1000,
                "service": 10,
                "isDepot": False,
            },
        ],
        "routes": [[1, 2]],
    }
    resp = authed_api.post("/api/reoptimize", data=payload)
    assert resp.ok, f"Expected 200, got {resp.status}: {resp.text()}"
    data = resp.json()
    assert "routes" in data
    assert len(data["routes"]) == 1
    assert data["feasible"] is True

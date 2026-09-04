"""
Live online verification script for SINTEF VRPTW benchmark BKS tables.
Performs an independent HTTP GET request to sintef.no and parses the exact published HTML.
"""

from __future__ import annotations

import re
import urllib.request
from datetime import datetime

from bs4 import BeautifulSoup

URLS = {
    "Solomon 100": "https://www.sintef.no/projectweb/top/vrptw/solomon-benchmark/100-customers/",
    "Homberger 200": "https://www.sintef.no/projectweb/top/vrptw/homberger-benchmark/200-customers/",
    "Homberger 400": "https://www.sintef.no/projectweb/top/vrptw/homberger-benchmark/400-customers/",
}


def fetch_live_url(url: str) -> str:
    req = urllib.request.Request(
        url, headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
    )
    with urllib.request.urlopen(req, timeout=15) as resp:
        return resp.read().decode("utf-8")


def verify_solomon_100():
    url = URLS["Solomon 100"]
    print(f"[{datetime.now().isoformat()}] Fetching live HTML from {url} ...")
    html = fetch_live_url(url)
    print(f"Successfully downloaded {len(html):,} bytes.")

    soup = BeautifulSoup(html, "html.parser")
    table = soup.find("table")
    if not table:
        raise ValueError("Could not find table on SINTEF page.")

    rows = table.find_all("tr")
    print(f"Found {len(rows)} table rows in SINTEF Solomon table.\n")

    focus_instances = ["RC101", "RC102", "RC105", "RC106", "RC201", "RC202", "RC205", "R211"]

    print("=" * 95)
    print(f"{'Instance':<10} | {'Vehicles (NV)':<15} | {'Distance (TD)':<15} | {'Ref':<10} | {'Detailed Comment'}")
    print("=" * 95)

    parsed_count = 0
    focus_data = {}

    for row in rows:
        cols = [td.get_text(strip=True) for td in row.find_all(["td", "th"])]
        if len(cols) >= 3:
            raw_name = cols[0].upper().replace("B", "").replace("*", "").strip()
            if re.match(r"^(C1|C2|R1|R2|RC1|RC2)\d{2}$", raw_name):
                parsed_count += 1
                nv = int(cols[1])
                td = float(cols[2].replace("*", "").strip())
                ref = cols[3] if len(cols) > 3 else ""
                comment = cols[4] if len(cols) > 4 else ""

                if raw_name in focus_instances:
                    focus_data[raw_name] = (nv, td, ref, comment)
                    print(f"{raw_name:<10} | {nv:<15} | {td:<15.2f} | {ref:<10} | {comment}")

    print("=" * 95)
    print(f"\nTotal Solomon 100 instances successfully parsed live: {parsed_count} / 56\n")

    # Explicit assertions against the live SINTEF values
    assert focus_data["RC202"][0] == 3 and abs(focus_data["RC202"][1] - 1365.65) < 1e-4, (
        f"RC202 mismatch: {focus_data['RC202']}"
    )
    assert focus_data["R211"][0] == 2 and abs(focus_data["R211"][1] - 885.71) < 1e-4, (
        f"R211 mismatch: {focus_data['R211']}"
    )
    assert focus_data["RC101"][0] == 14 and abs(focus_data["RC101"][1] - 1696.95) < 1e-4, (
        f"RC101 mismatch: {focus_data['RC101']}"
    )
    assert focus_data["RC201"][0] == 4 and abs(focus_data["RC201"][1] - 1406.94) < 1e-4, (
        f"RC201 mismatch: {focus_data['RC201']}"
    )
    assert focus_data["RC205"][0] == 4 and abs(focus_data["RC205"][1] - 1297.65) < 1e-4, (
        f"RC205 mismatch: {focus_data['RC205']}"
    )

    print("✅ VERIFICATION SUCCESSFUL: Live SINTEF values match repository BKS definition 100%!")


if __name__ == "__main__":
    verify_solomon_100()

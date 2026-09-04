"""
Parse Vietnamese delivery-order text blocks into customer dicts.

Supported format::

    Kho:
    <depot name>
    <depot address>

    Khách hàng

    1.
    Địa chỉ:
    <address>
    Khối lượng:
    <N> kg
    Thời gian:
    HH:MM - HH:MM

    2.
    ...
"""

from __future__ import annotations

import re
from typing import Any


def _time_to_minutes(time_str: str) -> float:
    """Convert ``'HH:MM'`` to minutes since midnight."""
    parts = time_str.strip().split(":")
    try:
        h = int(parts[0])
        m = int(parts[1]) if len(parts) > 1 else 0
        return float(h * 60 + m)
    except (ValueError, IndexError):
        return 0.0


def _parse_demand(text: str) -> int:
    """Extract integer demand from strings like ``'20 kg'``, ``'12kg'``."""
    m = re.search(r"(\d+(?:\.\d+)?)", text)
    if m:
        return int(float(m.group(1)))
    return 10  # default


def parse_vietnamese_text_block(text: str) -> list[dict[str, Any]]:
    """Return a list of customer dicts (with ``lat``/``lng`` set to ``None``).

    The caller is responsible for geocoding addresses afterward.
    """
    lines = text.splitlines()
    customers: list[dict[str, Any]] = []

    # ── 1. Locate and parse the depot ────────────────────────────────
    depot_name = "Kho"
    depot_address = ""

    for i, line in enumerate(lines):
        stripped = line.strip()
        if re.match(r"^Kho\s*:", stripped, re.IGNORECASE):
            # Depot name is on the next non-blank line, address on the one after
            remaining = [line_str.strip() for line_str in lines[i + 1 :] if line_str.strip()]
            if remaining:
                depot_name = remaining[0]
            if len(remaining) > 1:
                depot_address = remaining[1]
            break

    if depot_address:
        customers.append(
            {
                "id": 0,
                "name": depot_name,
                "address": depot_address,
                "lat": None,
                "lng": None,
                "demand": 0,
                "ready": 0.0,
                "due": 1440.0,
                "service": 0.0,
                "isDepot": True,
                "priority": "Normal",
                "skill": "None",
            }
        )

    # ── 2. Split by numbered markers (``1.``, ``2.``, …) ────────────
    # Build a list of (customer_number, start_line_index) tuples.
    markers: list[tuple[int, int]] = []
    for i, line in enumerate(lines):
        m = re.match(r"^\s*(\d+)\.\s*$", line)
        if m:
            markers.append((int(m.group(1)), i))

    # ── 3. Parse each customer block ────────────────────────────────
    for idx, (cust_num, start) in enumerate(markers):
        # Determine end of this block
        end = markers[idx + 1][1] if idx + 1 < len(markers) else len(lines)
        block = lines[start:end]

        address = ""
        demand = 10
        ready = 0.0
        due = 1440.0
        service = 10.0

        j = 0
        while j < len(block):
            stripped = block[j].strip()

            # Địa chỉ: (the address is on the NEXT line)
            if re.match(r"^[ĐĐđ][ịi]a\s*ch[ỉi]\s*:", stripped, re.IGNORECASE):
                if j + 1 < len(block):
                    address = block[j + 1].strip()
                    j += 2
                    continue

            # Khối lượng: (demand is on the NEXT line)
            if re.match(r"^Kh[ốo]i\s*l[ưu][ợo]ng\s*:", stripped, re.IGNORECASE):
                if j + 1 < len(block):
                    demand = _parse_demand(block[j + 1])
                    j += 2
                    continue

            # Thời gian: (time window is on the NEXT line, format HH:MM - HH:MM)
            if re.match(r"^Th[ờo]i\s*gian\s*:", stripped, re.IGNORECASE):
                if j + 1 < len(block):
                    tw_line = block[j + 1].strip()
                    tw_match = re.match(r"(\d{1,2}:\d{2})\s*[-–—]\s*(\d{1,2}:\d{2})", tw_line)
                    if tw_match:
                        ready = _time_to_minutes(tw_match.group(1))
                        due = _time_to_minutes(tw_match.group(2))
                    j += 2
                    continue

            j += 1

        if not address:
            continue

        cust_id = len(customers)
        customers.append(
            {
                "id": cust_id,
                "name": f"Khách hàng {cust_num}",
                "address": address,
                "lat": None,
                "lng": None,
                "demand": demand,
                "ready": ready,
                "due": due,
                "service": service,
                "isDepot": False,
                "priority": "Normal",
                "skill": "None",
            }
        )

    return customers


def is_vietnamese_text_block(text: str) -> bool:
    """Heuristic: does *text* look like the Vietnamese block format?

    Returns ``True`` if the first ~50 lines contain ``Kho:`` and a numbered
    marker like ``1.`` followed by ``Địa chỉ:``.
    """
    head = "\n".join(text.splitlines()[:60])
    has_kho = bool(re.search(r"Kho\s*:", head, re.IGNORECASE))
    has_diachi = bool(re.search(r"[ĐĐđ][ịi]a\s*ch[ỉi]\s*:", head, re.IGNORECASE))
    has_number = bool(re.search(r"^\s*\d+\.\s*$", head, re.MULTILINE))
    return has_kho and has_diachi and has_number

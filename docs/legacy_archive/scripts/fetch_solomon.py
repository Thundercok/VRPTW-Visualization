"""Download Solomon VRPTW benchmark instances (RC1, RC2 by default).

Usage:
    python scripts/fetch_solomon.py                # download default sets
    python scripts/fetch_solomon.py rc101 rc102    # download specific instances
    python scripts/fetch_solomon.py --all          # download every Solomon instance

Files are written to data/solomon/<name>.txt (or VRPTW_DATA_DIR if set).
"""

from __future__ import annotations

import argparse
import os
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = Path(os.environ.get("VRPTW_DATA_DIR") or ROOT / "data" / "solomon")

MIRRORS: tuple[str, ...] = (
    "https://web.cba.neu.edu/~msolomon/problems/{name}.txt",
    "https://raw.githubusercontent.com/sashakh/Solomon/master/Solomon/{name}.txt",
    "https://raw.githubusercontent.com/IceLemon/VRPTW-Solomon-Benchmark/master/{upper}.txt",
)

DEFAULT_INSTANCES: tuple[str, ...] = tuple(
    f"{prefix}{idx:03d}"
    for prefix in ("rc1", "rc2")
    for idx in range(1, 11)
)
ALL_INSTANCES: tuple[str, ...] = tuple(
    f"{prefix}{idx:03d}"
    for prefix in ("c1", "c2", "r1", "r2", "rc1", "rc2")
    for idx in range(1, 11)
)


def _try_fetch(name: str, dest: Path) -> bool:
    for template in MIRRORS:
        url = template.format(name=name, upper=name.upper())
        try:
            with urllib.request.urlopen(url, timeout=20) as resp:
                content = resp.read()
        except (urllib.error.URLError, TimeoutError, ConnectionError):
            continue
        if not content or b"CUST NO." not in content:
            continue
        dest.write_bytes(content)
        print(f"  ok  -> {dest.relative_to(ROOT)} ({len(content)} bytes from {url})")
        return True
    print(f"  FAIL: could not download {name} from any mirror")
    return False


def main() -> int:
    parser = argparse.ArgumentParser(description="Download Solomon VRPTW benchmark instances")
    parser.add_argument("instances", nargs="*", help="Instance names (e.g. rc101 r205). Defaults to RC1+RC2.")
    parser.add_argument("--all", action="store_true", help="Download every Solomon instance (60 files).")
    parser.add_argument("--force", action="store_true", help="Re-download even if file already exists.")
    args = parser.parse_args()

    if args.all:
        names = ALL_INSTANCES
    elif args.instances:
        names = tuple(name.lower() for name in args.instances)
    else:
        names = DEFAULT_INSTANCES

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Saving Solomon files to {DATA_DIR}")

    failures: list[str] = []
    for name in names:
        dest = DATA_DIR / f"{name}.txt"
        if dest.exists() and not args.force:
            print(f"  skip {name} (already present)")
            continue
        print(f"  fetching {name} ...")
        ok = _try_fetch(name, dest)
        if not ok:
            failures.append(name)
        time.sleep(0.4)

    if failures:
        print(f"\nDone with {len(failures)} failures: {', '.join(failures)}")
        print("Mirror sites change over time - retry later or download manually.")
        return 1

    print("\nAll requested instances downloaded.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

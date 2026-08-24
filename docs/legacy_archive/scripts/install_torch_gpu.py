"""Replace the CPU-only PyTorch wheel with a CUDA build that matches your driver.

Usage:
    python scripts/install_torch_gpu.py             # auto-detect CUDA via nvidia-smi
    python scripts/install_torch_gpu.py --cuda 124  # force a specific build (cu124, cu121, cu118, cu126)
    python scripts/install_torch_gpu.py --dry-run   # print the pip command without running it

The DDQN-ALNS solver auto-detects ``torch.cuda.is_available()``; once the wheel
is installed and you restart ``python main.py`` the backend logs::

    Torch device: GPU (NVIDIA GeForce RTX 3050, CUDA 12.4, 2.5.0+cu124)
"""

from __future__ import annotations

import argparse
import re
import shutil
import subprocess
import sys

# Map driver CUDA versions -> wheel tag PyTorch publishes. We only list builds
# that actually exist on https://download.pytorch.org/whl/.
_WHEEL_TAGS = ("cu118", "cu121", "cu124", "cu126")


def _detect_cuda_version() -> str | None:
    """Return the CUDA version reported by nvidia-smi (e.g. ``12.4``)."""
    if shutil.which("nvidia-smi") is None:
        return None
    try:
        out = subprocess.check_output(["nvidia-smi"], text=True, timeout=15)
    except (subprocess.SubprocessError, OSError):
        return None
    m = re.search(r"CUDA Version:\s*(\d+\.\d+)", out)
    return m.group(1) if m else None


def _pick_wheel_tag(cuda_version: str | None) -> str:
    """Pick the closest available cuXYZ tag <= driver's CUDA version."""
    if not cuda_version:
        return "cu124"
    try:
        major, minor = (int(p) for p in cuda_version.split(".")[:2])
    except ValueError:
        return "cu124"
    target = major * 10 + minor
    best = "cu118"
    for tag in _WHEEL_TAGS:
        n = int(tag.removeprefix("cu"))
        if n <= target:
            best = tag
    return best


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cuda", help="Force a wheel tag (e.g. 124 for cu124).")
    parser.add_argument("--dry-run", action="store_true", help="Print the command without running it.")
    args = parser.parse_args()

    if args.cuda:
        wheel_tag = f"cu{args.cuda}" if not args.cuda.startswith("cu") else args.cuda
        if wheel_tag not in _WHEEL_TAGS:
            print(f"Unknown wheel tag {wheel_tag}. Known: {', '.join(_WHEEL_TAGS)}")
            return 2
    else:
        cuda_version = _detect_cuda_version()
        if cuda_version is None:
            print(
                "Could not find nvidia-smi - no NVIDIA driver detected. "
                "If you actually have a GPU, install the driver first or pass --cuda <version>."
            )
            return 1
        print(f"Detected CUDA driver: {cuda_version}")
        wheel_tag = _pick_wheel_tag(cuda_version)

    index_url = f"https://download.pytorch.org/whl/{wheel_tag}"
    cmd = [
        sys.executable, "-m", "pip", "install", "--upgrade",
        "--index-url", index_url,
        "torch", "torchvision",
    ]
    print(f"Using wheel tag: {wheel_tag}")
    print(f"Index URL:       {index_url}")
    print(f"Command:         {' '.join(cmd)}")
    if args.dry_run:
        print("--dry-run: not executing.")
        return 0

    print()
    print("This will download ~2-3 GB the first time. Press Ctrl+C to abort.")
    return subprocess.call(cmd)


if __name__ == "__main__":
    sys.exit(main())

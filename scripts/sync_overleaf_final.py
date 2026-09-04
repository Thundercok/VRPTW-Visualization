#!/usr/bin/env python3
"""Synchronize and compile monolithic manuscript.tex,
and generate a clean Overleaf package with zero section dependencies.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs"
OVERLEAF_DIR = DOCS / "overleaf_bundle"
ZIP_PATH = DOCS / "overleaf_ieee_access.zip"


def sync_and_build() -> bool:
    print("[1/3] Compiling monolithic manuscript.tex with pdflatex (2-pass) ...")
    subprocess.run(
        ["pdflatex", "-interaction=nonstopmode", "manuscript.tex"],
        cwd=str(DOCS),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    res = subprocess.run(
        ["pdflatex", "-interaction=nonstopmode", "manuscript.tex"], cwd=str(DOCS), capture_output=True, text=True
    )
    if res.returncode != 0:
        print("Compilation of manuscript.tex FAILED:")
        print(res.stdout[-1500:])
        return False
    print("  -> docs/manuscript.pdf compiled successfully!")

    print("[2/3] Syncing Overleaf bundle directory (monolithic structure, manuscript.tex only) ...")
    if OVERLEAF_DIR.exists():
        shutil.rmtree(OVERLEAF_DIR)
    OVERLEAF_DIR.mkdir(parents=True, exist_ok=True)

    # Copy root files
    extensions = (".cls", ".bst", ".bib", ".bbl", ".sty", ".pfb", ".tfm", ".map", ".fd", ".jpg", ".png", ".tex")
    for f in os.listdir(DOCS):
        if (
            f.endswith(extensions)
            and not f.startswith("paper.")
            and not f.startswith("thesis.")
            and not f.startswith("test_")
            and not f.startswith("main.")
        ):
            src = DOCS / f
            if src.is_file():
                shutil.copy2(src, OVERLEAF_DIR / f)

    # Copy architecture.pdf
    arch_pdf = DOCS / "architecture.pdf"
    if arch_pdf.is_file():
        shutil.copy2(arch_pdf, OVERLEAF_DIR / "architecture.pdf")

    # Copy figures if present
    fig_src = DOCS / "figures"
    fig_dst = OVERLEAF_DIR / "figures"
    if fig_src.exists():
        fig_dst.mkdir(parents=True, exist_ok=True)
        for f in os.listdir(fig_src):
            shutil.copy2(fig_src / f, fig_dst / f)

    # Copy supplementary proofs
    if (DOCS / "supplementary_proofs.tex").exists():
        shutil.copy2(DOCS / "supplementary_proofs.tex", OVERLEAF_DIR / "supplementary_proofs.tex")

    print("[3/3] Packaging Overleaf ZIP ...")
    with zipfile.ZipFile(ZIP_PATH, "w", zipfile.ZIP_DEFLATED) as z:
        for root, _dirs, files in os.walk(OVERLEAF_DIR):
            for f in files:
                full = os.path.join(root, f)
                rel = os.path.relpath(full, OVERLEAF_DIR)
                z.write(full, rel)
    print(f"  -> Generated Overleaf ZIP: {ZIP_PATH} ({ZIP_PATH.stat().st_size / (1024 * 1024):.2f} MB)")
    return True


if __name__ == "__main__":
    success = sync_and_build()
    if not success:
        sys.exit(1)

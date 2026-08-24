#!/usr/bin/env python3
"""Regenerate monolithic manuscript.tex cleanly from docs/sections/*.tex and main.tex."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs"
SECTIONS = DOCS / "sections"
OVERLEAF = DOCS / "overleaf_bundle"

SECTION_FILES = [
    "01_introduction.tex",
    "02_related_work.tex",
    "03_formulation.tex",
    "04_methodology.tex",
    "05_experiments.tex",
    "06_discussion.tex",
    "07_conclusion.tex",
]

def build_monolithic_manuscript(target_path: Path) -> None:
    main_tex = (DOCS / "main.tex").read_text(encoding="utf-8")
    
    # Split main_tex into header (before 01_introduction) and footer (after 07_conclusion)
    header_marker = r"\input{sections/01_introduction.tex}"
    footer_marker = r"\input{sections/07_conclusion.tex}"
    
    if header_marker not in main_tex or footer_marker not in main_tex:
        raise ValueError("Could not find section input markers in main.tex")
        
    header = main_tex.split(header_marker)[0]
    footer = main_tex.split(footer_marker)[1].lstrip()
    
    # Read and concatenate all sections
    body_parts = []
    for sf in SECTION_FILES:
        sec_path = SECTIONS / sf
        if not sec_path.exists():
            raise FileNotFoundError(f"Missing section file: {sec_path}")
        sec_content = sec_path.read_text(encoding="utf-8").strip()
        body_parts.append(sec_content)
        
    body = "\n\n".join(body_parts)
    
    # Construct complete monolithic file
    full_content = f"{header.strip()}\n\n{body}\n\n{footer.strip()}\n"
    
    target_path.parent.mkdir(parents=True, exist_ok=True)
    target_path.write_text(full_content, encoding="utf-8")
    print(f"Successfully generated monolithic manuscript -> {target_path}")

def sync_bundle() -> None:
    import shutil
    OVERLEAF.mkdir(parents=True, exist_ok=True)
    if (DOCS / "refs.bib").exists():
        shutil.copy2(DOCS / "refs.bib", OVERLEAF / "refs.bib")
        print(f"Synced refs.bib -> {OVERLEAF / 'refs.bib'}")
    
    overleaf_sections = OVERLEAF / "sections"
    overleaf_sections.mkdir(parents=True, exist_ok=True)
    for sf in SECTION_FILES:
        sec_path = SECTIONS / sf
        if sec_path.exists():
            shutil.copy2(sec_path, overleaf_sections / sf)
    print(f"Synced sections -> {overleaf_sections}")

if __name__ == "__main__":
    p1 = DOCS / "manuscript.tex"
    p2 = OVERLEAF / "manuscript.tex"
    build_monolithic_manuscript(p1)
    build_monolithic_manuscript(p2)
    sync_bundle()

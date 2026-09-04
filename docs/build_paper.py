#!/usr/bin/env python3
import os
import shutil
import subprocess
import zipfile

DOCS_DIR = os.path.dirname(os.path.abspath(__file__))
BUNDLE_DIR = os.path.join(DOCS_DIR, "overleaf_bundle")
ZIP_PATH = os.path.join(DOCS_DIR, "overleaf_ieee_access.zip")

def compile_pdf():
    print("[1/2] Compiling docs/manuscript.tex with pdflatex (2-pass)...")
    subprocess.run(["pdflatex", "-interaction=nonstopmode", "manuscript.tex"], cwd=DOCS_DIR, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    result = subprocess.run(["pdflatex", "-interaction=nonstopmode", "manuscript.tex"], cwd=DOCS_DIR, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    if result.returncode != 0:
        print("Compilation FAILED! Log output:")
        print(result.stdout[-1500:])
        return False
    print("  -> manuscript.pdf built successfully!")
    return True

def package_bundle():
    print("[2/2] Packaging Overleaf bundle (manuscript.tex only)...")
    if os.path.exists(BUNDLE_DIR):
        shutil.rmtree(BUNDLE_DIR)
    os.makedirs(BUNDLE_DIR, exist_ok=True)

    # Copy root template, bibliography, font, and asset files
    extensions = (".cls", ".bst", ".bib", ".sty", ".pfb", ".tfm", ".map", ".fd", ".jpg", ".png", ".tex")
    for f in os.listdir(DOCS_DIR):
        if f.endswith(extensions) and not f.startswith("paper.") and not f.startswith("thesis.") and not f.startswith("main."):
            src = os.path.join(DOCS_DIR, f)
            if os.path.isfile(src):
                shutil.copy2(src, os.path.join(BUNDLE_DIR, f))

    # Also copy architecture.pdf to bundle root
    arch_pdf = os.path.join(DOCS_DIR, "architecture.pdf")
    if os.path.isfile(arch_pdf):
        shutil.copy2(arch_pdf, os.path.join(BUNDLE_DIR, "architecture.pdf"))

    # Copy sections (if exists)
    sec_src = os.path.join(DOCS_DIR, "sections")
    sec_dst = os.path.join(BUNDLE_DIR, "sections")
    if os.path.exists(sec_src):
        os.makedirs(sec_dst, exist_ok=True)
        for f in os.listdir(sec_src):
            if f.endswith(".tex"):
                shutil.copy2(os.path.join(sec_src, f), os.path.join(sec_dst, f))

    # Copy figures
    fig_src = os.path.join(DOCS_DIR, "figures")
    fig_dst = os.path.join(BUNDLE_DIR, "figures")
    os.makedirs(fig_dst, exist_ok=True)
    if os.path.exists(fig_src):
        for f in os.listdir(fig_src):
            shutil.copy2(os.path.join(fig_src, f), os.path.join(fig_dst, f))

    # Zip everything
    with zipfile.ZipFile(ZIP_PATH, "w", zipfile.ZIP_DEFLATED) as z:
        for root, _dirs, files in os.walk(BUNDLE_DIR):
            for f in files:
                full = os.path.join(root, f)
                rel = os.path.relpath(full, BUNDLE_DIR)
                z.write(full, rel)
    print(f"  -> Generated: {ZIP_PATH} (ready for Overleaf upload)")

if __name__ == "__main__":
    success = compile_pdf()
    package_bundle()
    print("\nAll done! docs/manuscript.pdf and docs/overleaf_ieee_access.zip are up to date.")

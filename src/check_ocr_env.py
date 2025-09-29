# -*- coding: utf-8 -*-
"""
Enhanced OCR environment checker for Windows (also works on Mac/Linux).
- Finds Tesseract and Poppler (pdftoppm/pdftotext)
- Prints actual executable paths and versions
- Shows which paths pytesseract/pdf2image will use
- Provides actionable suggestions if something is missing

Usage:
    python src/check_ocr_env.py
"""

import os
import sys
import shutil
import subprocess
from pathlib import Path

PRINT_PREFIX = {"ok": "[✓]", "warn": "[!]", "err": "[✗]"}


def run_version(cmd, version_args=("--version",)):
    try:
        out = subprocess.run([cmd, *version_args],
                             capture_output=True, text=True, check=False)
        # Prefer stdout; some tools print version to stderr
        txt = (out.stdout or out.stderr).strip()
        first = txt.splitlines()[0] if txt else "unknown"
        return True, first
    except Exception as e:
        return False, str(e)


def find_executable(name, extra_candidates=None):
    """
    Try PATH first, then try extra candidate absolute paths (files or dirs).
    If a candidate is a directory, we try `<dir>/<name>` and `<dir>/<name>.exe` on Windows.
    Returns absolute path or None.
    """
    path = shutil.which(name)
    if path:
        return Path(path).resolve()

    extra_candidates = extra_candidates or []
    exts = [""] if os.name != "nt" else ["", ".exe"]

    for cand in extra_candidates:
        p = Path(cand)
        if p.is_file():
            return p.resolve()
        if p.is_dir():
            for ext in exts:
                q = p / f"{name}{ext}"
                if q.exists():
                    return q.resolve()
    return None


def print_status(ok: bool, title: str, path: Path | None, version: str):
    if ok:
        print(f"{PRINT_PREFIX['ok']} {title}")
        print(f"    path   : {path}")
        print(f"    version: {version}")
    else:
        print(f"{PRINT_PREFIX['err']} {title} NOT found")
        if path:
            print(f"    tried  : {path}")
        print("    hint   : Add its bin folder to PATH, or set an env var as shown below.")


def main():
    print("=== OCR Environment Check (Enhanced) ===\n")

    # -------- Common install locations (Windows) --------
    # You can tweak these to your actual paths if needed.
    TESS_ENV = os.getenv("TESSERACT_CMD")
    POP_ENV = os.getenv("POPPLER_PATH")

    tess_candidates = [
        TESS_ENV or "",
        r"C:\Program Files\Tesseract-OCR",
        r"C:\Program Files (x86)\Tesseract-OCR",
        r"C:\Program Files\Tesseract-OCR\tesseract.exe",
        r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
        # UB Mannheim builds (popular on Windows)
        r"C:\Program Files\Tesseract-OCR\tesseract.exe",
    ]

    poppler_bin_candidates = [
        POP_ENV or "",
        r"C:\poppler\Library\bin",
        r"C:\poppler-25.07.0\Library\bin",
        r"C:\poppler-24.08.0\Library\bin",
        r"C:\Program Files\poppler\Library\bin",
    ]

    # -------- Tesseract --------
    tess_path = find_executable("tesseract", tess_candidates)
    tess_ok, tess_ver = (False, "not found")
    if tess_path:
        tess_ok, tess_ver = run_version(str(tess_path))
    print_status(tess_ok, "Tesseract OCR", tess_path, tess_ver)
    if not tess_ok:
        print("    add to PATH: e.g. C:\\Program Files (x86)\\Tesseract-OCR\\")
        print("    or set env : TESSERACT_CMD=C:\\...\\Tesseract-OCR\\tesseract.exe\n")

    # -------- Poppler: pdftoppm --------
    pdftoppm_path = find_executable("pdftoppm", poppler_bin_candidates)
    ppm_ok, ppm_ver = (False, "not found")
    if pdftoppm_path:
        ppm_ok, ppm_ver = run_version(str(pdftoppm_path), version_args=("-v",))
    print_status(ppm_ok, "Poppler (pdftoppm)", pdftoppm_path, ppm_ver)

    # -------- Poppler: pdftotext --------
    pdftotext_path = find_executable("pdftotext", poppler_bin_candidates)
    ptt_ok, ptt_ver = (False, "not found")
    if pdftotext_path:
        ptt_ok, ptt_ver = run_version(str(pdftotext_path), version_args=("-v",))
    print_status(ptt_ok, "Poppler (pdftotext)", pdftotext_path, ptt_ver)
    if not ppm_ok or not ptt_ok:
        print("    add to PATH: e.g. C:\\poppler-25.07.0\\Library\\bin\\")
        print("    or set env : POPPLER_PATH=C:\\...\\poppler\\Library\\bin\\\n")

    # -------- Python wrappers --------
    # pytesseract
    try:
        import pytesseract
        print(f"{PRINT_PREFIX['ok']} pytesseract Python package")
        print(f"    pytesseract.pytesseract.tesseract_cmd = {pytesseract.pytesseract.tesseract_cmd}")
    except Exception as e:
        print(f"{PRINT_PREFIX['err']} pytesseract import failed: {e}")
        print("    pip install pytesseract\n")

    # pdf2image
    try:
        import pdf2image
        print(f"{PRINT_PREFIX['ok']} pdf2image Python package")
        print("    pdf2image will use POPPLER_PATH if provided, else relies on pdftoppm in PATH.")
        print(f"    POPPLER_PATH env = {os.getenv('POPPLER_PATH')}")
    except Exception as e:
        print(f"{PRINT_PREFIX['err']} pdf2image import failed: {e}")
        print("    pip install pdf2image\n")

    # -------- Final verdict --------
    all_ok = tess_ok and ppm_ok and ptt_ok
    print("\n=== Result ===")
    if all_ok:
        print("✅ All OCR dependencies are correctly installed and discoverable.")
    else:
        print("⚠ Some dependencies are missing or misconfigured.")
        print("  - After editing PATH/env vars, close and reopen your terminal/IDE before retrying.")

    # Exit code (0 = success; 1 = missing deps)
    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()

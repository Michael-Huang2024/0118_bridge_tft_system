# src/ingest_pdf.py
from __future__ import annotations

import os
import re
import json
from pathlib import Path
from typing import List, Dict, Optional

import pdfplumber
import fitz  # PyMuPDF
from pdf2image import convert_from_path
import pytesseract


# ----------------------
# Config / env detection
# ----------------------

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PDF_DIR = ROOT / "data" / "pdf"
DEFAULT_OUT_DIR = ROOT / "artifacts"

# If you want to pin paths explicitly, set envs:
#   TESSERACT_CMD=C:\Program Files\Tesseract-OCR\tesseract.exe
#   POPPLER_PATH=C:\poppler-25.07.0\Library\bin
TESSERACT_CMD = os.getenv("TESSERACT_CMD")
if TESSERACT_CMD:
    pytesseract.pytesseract.tesseract_cmd = TESSERACT_CMD

POPPLER_PATH = os.getenv("POPPLER_PATH")  # directory containing pdftoppm/pdftotext


# ----------------------
# Helpers
# ----------------------

def extract_year_from_name(name: str) -> str:
    """
    Try to find a 4-digit year in the filename (e.g., ..._2023.pdf).
    Returns 'unknown' if not found.
    """
    m = re.search(r'(19|20)\d{2}', name)
    return m.group(0) if m else "unknown"


def chunk_text(text: str, max_chars: int = 1500, overlap: int = 150, min_len: int = 40) -> List[str]:
    """
    Simple fixed-length chunking with overlap. Keeps only non-trivial chunks.
    """
    text = text.strip()
    if not text:
        return []
    out = []
    i = 0
    n = len(text)
    while i < n:
        j = min(i + max_chars, n)
        chunk = text[i:j].strip()
        if len(chunk) >= min_len:
            out.append(chunk)
        if j == n:
            break
        i = j - overlap
        if i < 0:
            i = 0
    return out


# -----------------------------------
# Extraction (no OCR) and OCR fallback
# -----------------------------------

def extract_text_pdfplumber(pdf_path: Path) -> List[str]:
    """
    Extract per-page text with pdfplumber. Returns list of page strings.
    """
    pages: List[str] = []
    with pdfplumber.open(str(pdf_path)) as pdf:
        for p in pdf.pages:
            txt = (p.extract_text() or "").strip()
            pages.append(txt)
    return pages


def extract_text_pymupdf(pdf_path: Path) -> List[str]:
    """
    Extract per-page text with PyMuPDF as a secondary non-OCR strategy.
    """
    pages: List[str] = []
    with fitz.open(str(pdf_path)) as doc:
        for page in doc:
            txt = (page.get_text("text") or "").strip()
            pages.append(txt)
    return pages


def ocr_pdf(pdf_path: Path, dpi: int = 300, lang: str = "eng") -> List[str]:
    """
    OCR the entire PDF using pdf2image + Tesseract. Returns per-page text list.
    Requires Poppler (pdftoppm) available on PATH or via POPPLER_PATH.
    """
    images = convert_from_path(str(pdf_path), dpi=dpi, poppler_path=POPPLER_PATH)
    out: List[str] = []
    for img in images:
        txt = pytesseract.image_to_string(img, lang=lang, config="--psm 6")
        out.append((txt or "").strip())
    return out


# ----------------------
# Pipeline per PDF
# ----------------------

def process_one_pdf(
    pdf_path: Path,
    out_dir: Path,
    lang: str = "eng",
    max_chars: int = 1500,
    overlap: int = 150,
    min_len: int = 40,
    force_ocr: bool = False,
) -> int:
    """
    Process a single PDF: try text extraction; if empty, fallback to OCR.
    Saves JSONL chunks to artifacts/<year>/<filename>_chunks.jsonl
    Returns number of chunks written.
    """
    year = extract_year_from_name(pdf_path.stem)
    target_dir = out_dir / year
    target_dir.mkdir(parents=True, exist_ok=True)

    out_file = target_dir / f"{pdf_path.stem}_chunks.jsonl"

    # Try non-OCR first (unless forced)
    page_texts: List[str] = []
    if not force_ocr:
        # 1) pdfplumber
        page_texts = extract_text_pdfplumber(pdf_path)
        if not any(t.strip() for t in page_texts):
            # 2) PyMuPDF
            page_texts = extract_text_pymupdf(pdf_path)

    # If still empty or OCR forced -> OCR
    if force_ocr or not any(t.strip() for t in page_texts):
        print(f"[info] OCR fallback: {pdf_path.name}")
        page_texts = ocr_pdf(pdf_path, dpi=300, lang=lang)

    # Build chunks and write JSONL
    total_chunks = 0
    with open(out_file, "w", encoding="utf-8") as f:
        for page_idx, page_text in enumerate(page_texts, start=1):
            if not page_text.strip():
                continue
            chunks = chunk_text(page_text, max_chars=max_chars, overlap=overlap, min_len=min_len)
            for ci, ch in enumerate(chunks, start=1):
                rec: Dict[str, object] = {
                    "text": ch,
                    "metadata": {
                        "bridge": pdf_path.stem.replace(".pdf", ""),
                        "page": page_idx,
                        "chunk": ci,
                        "year": year,
                    },
                    "source": pdf_path.name,
                }
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                total_chunks += 1

    print(f"[ok] {pdf_path.name}: {total_chunks} chunks -> {out_file}")
    return total_chunks


# ----------------------
# Batch runner / CLI
# ----------------------

def find_pdfs(pdf_dir: Path) -> List[Path]:
    return [p for p in pdf_dir.glob("*.pdf") if p.is_file()]


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Ingest PDFs to JSONL chunks with optional OCR fallback.")
    parser.add_argument("--pdf_dir", type=str, default=str(DEFAULT_PDF_DIR), help="Directory with PDF files")
    parser.add_argument("--out_dir", type=str, default=str(DEFAULT_OUT_DIR), help="Output artifacts directory")
    parser.add_argument("--lang", type=str, default="eng", help="Tesseract language for OCR (e.g., eng)")
    parser.add_argument("--max_chars", type=int, default=1500, help="Chunk max characters")
    parser.add_argument("--overlap", type=int, default=150, help="Chunk overlap characters")
    parser.add_argument("--min_len", type=int, default=40, help="Minimum characters to keep a chunk")
    parser.add_argument("--force_ocr", action="store_true", help="Force OCR even if text extraction succeeds")
    args = parser.parse_args()

    pdf_dir = Path(args.pdf_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    files = find_pdfs(pdf_dir)
    if not files:
        print(f"[warn] No PDFs found in {pdf_dir}")
        return

    print(f"[info] Found {len(files)} PDF(s) in {pdf_dir}")
    total = 0
    for p in files:
        try:
            total += process_one_pdf(
                pdf_path=p,
                out_dir=out_dir,
                lang=args.lang,
                max_chars=args.max_chars,
                overlap=args.overlap,
                min_len=args.min_len,
                force_ocr=args.force_ocr,
            )
        except Exception as e:
            print(f"[err] Failed on {p.name}: {e}")

    print(f"\nDone. Total chunks: {total}")


if __name__ == "__main__":
    main()

# src/extract_fields_all.py
import os
import re
import json
import glob
import csv
from pathlib import Path
from typing import Dict, Any, List, Optional

# Try to import local ingest function (optional)
INGEST_AVAILABLE = False
try:
    # Your ingest_pdf.py should define process_one(pdf_path: Path, out_dir: Path, use_ocr=True) -> int
    from ingest_pdf import process_one as ingest_one  # type: ignore
    INGEST_AVAILABLE = True
except Exception:
    INGEST_AVAILABLE = False


ROOT = Path(__file__).resolve().parents[1]
PDF_DIR = ROOT / "data" / "pdf"
ARTIFACTS = ROOT / "artifacts"
EXTRACT_DIR = ARTIFACTS / "extracted"
INDEX_DIR = ARTIFACTS / "index" / "faiss"  # not used here, kept for reference

OUT_CSV = EXTRACT_DIR / "bridge_summary_all_years.csv"
OUT_JSON = EXTRACT_DIR / "bridge_summary_all_years.json"


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def to_int_or_none(x: Optional[str]) -> Optional[int]:
    if not x:
        return None
    s = re.sub(r"[^\d]", "", x)
    if not s:
        return None
    try:
        return int(s)
    except Exception:
        return None


def extract_year_from_filename(pdf_path: Path) -> Optional[int]:
    # Expect filenames like "..._2019.pdf", "..._2023.pdf"
    m = re.search(r"(\d{4})(?=\.pdf$)", pdf_path.name)
    if m:
        try:
            return int(m.group(1))
        except Exception:
            return None
    return None


def chunks_path_for(pdf_path: Path, year: Optional[int]) -> Path:
    # artifacts/<year>/<bridge>_<year>_chunks.jsonl
    bridge = pdf_path.stem
    y = str(year) if year else "unknown"
    return ARTIFACTS / y / f"{bridge}_chunks.jsonl"


def find_or_build_chunks(pdf_path: Path) -> Optional[Path]:
    year = extract_year_from_filename(pdf_path)
    out_jsonl = chunks_path_for(pdf_path, year)
    if out_jsonl.exists() and out_jsonl.stat().st_size > 0:
        return out_jsonl

    if not INGEST_AVAILABLE:
        print(f"[warn] Missing chunks and ingest function not available -> skip: {pdf_path.name}")
        return None

    ensure_dir(out_jsonl.parent)
    try:
        # ingest_one should return number of chunks extracted
        n = ingest_one(pdf_path, out_jsonl.parent, use_ocr=True)  # type: ignore
        if out_jsonl.exists() and out_jsonl.stat().st_size > 0 and n > 0:
            print(f"[ok] Built chunks -> {out_jsonl}")
            return out_jsonl
        print(f"[warn] Ingest returned {n} but chunks not found or empty: {pdf_path.name}")
    except Exception as e:
        print(f"[err] Ingest failed for {pdf_path.name}: {e}")
    return None


# ----------- parsers over a jsonl of OCR/text chunks -----------

RE_DECK = re.compile(r"Deck\s*Condition\s*Rating\s*\(58\)\s*:\s*([0-9])", re.I)
RE_SUPER = re.compile(r"Superstructure\s*Condition\s*Rating\s*\(59\)\s*:\s*([0-9])", re.I)
RE_SUB = re.compile(r"Substructure\s*Condition\s*Rating\s*\(60\)\s*:\s*([0-9])", re.I)
RE_ADT = re.compile(r"Average\s+Daily\s+Traffic\s*\(29\)\s*:\s*([\d,]+)", re.I)
RE_INSP = re.compile(r"Inspection\s*Date\s*\(90\)\s*:\s*([A-Za-z]+\s+\d{4}|\d{4})", re.I)
RE_LOADDESC = re.compile(r"Design\s*Load\s*Descriptor\s*\(31\)\s*:\s*(.+)", re.I)

# Try a fallback Inspection Year if Date is not found
RE_YEAR_ANY = re.compile(r"(20\d{2})")


def parse_one_jsonl(jsonl_path: Path) -> Dict[str, Any]:
    """Parse a jsonl of chunks and extract fields."""
    deck, sup, sub = None, None, None
    adt, insp_date, design_load_desc = None, None, None

    with open(jsonl_path, "r", encoding="utf-8") as fh:
        for line in fh:
            try:
                d = json.loads(line)
            except Exception:
                continue
            text = d.get("text", "") or ""

            if deck is None:
                m = RE_DECK.search(text)
                if m:
                    deck = to_int_or_none(m.group(1))

            if sup is None:
                m = RE_SUPER.search(text)
                if m:
                    sup = to_int_or_none(m.group(1))

            if sub is None:
                m = RE_SUB.search(text)
                if m:
                    sub = to_int_or_none(m.group(1))

            if adt is None:
                m = RE_ADT.search(text)
                if m:
                    adt = to_int_or_none(m.group(1))

            if insp_date is None:
                m = RE_INSP.search(text)
                if m:
                    insp_date = m.group(1).strip()

            if design_load_desc is None:
                m = RE_LOADDESC.search(text)
                if m:
                    # Take only the first line; strip decorations
                    design_load_desc = m.group(1).strip().splitlines()[0]

    return {
        "deck_cond": deck,
        "super_cond": sup,
        "sub_cond": sub,
        "adt": adt,
        "inspection_date": insp_date,
        "design_load_desc": design_load_desc,
    }


def decide_year(pdf_path: Path, parsed: Dict[str, Any]) -> Optional[int]:
    # Priority: filename year -> year in inspection_date -> any 20xx found
    y = extract_year_from_filename(pdf_path)
    if y:
        return y
    insp = parsed.get("inspection_date") or ""
    m = RE_YEAR_ANY.search(str(insp))
    if m:
        try:
            return int(m.group(1))
        except Exception:
            pass
    return None


def process_pdf(pdf_path: Path) -> Optional[Dict[str, Any]]:
    chunks = find_or_build_chunks(pdf_path)
    if not chunks:
        return None
    parsed = parse_one_jsonl(chunks)
    year = decide_year(pdf_path, parsed)

    row = {
        "year": year,
        "inspection_date": parsed.get("inspection_date"),
        "deck_cond": parsed.get("deck_cond"),
        "super_cond": parsed.get("super_cond"),
        "sub_cond": parsed.get("sub_cond"),
        "adt": parsed.get("adt"),
        "design_load_desc": parsed.get("design_load_desc"),
        "source_pdf": str(pdf_path.name),
        "source_chunks": str(chunks.relative_to(ROOT)),
    }
    return row


def write_outputs(rows: List[Dict[str, Any]]) -> None:
    ensure_dir(EXTRACT_DIR)

    # Write CSV
    fields = [
        "year",
        "inspection_date",
        "deck_cond",
        "super_cond",
        "sub_cond",
        "adt",
        "design_load_desc",
        "source_pdf",
        "source_chunks",
    ]
    with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in sorted(rows, key=lambda x: (x.get("year") or 0, x["source_pdf"])):
            w.writerow({k: r.get(k) for k in fields})

    # Write JSON
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(sorted(rows, key=lambda x: (x.get("year") or 0, x["source_pdf"])), f, indent=2, ensure_ascii=False)

    print(f"[ok] CSV  -> {OUT_CSV}")
    print(f"[ok] JSON -> {OUT_JSON}")


def main() -> None:
    ensure_dir(EXTRACT_DIR)

    pdfs = sorted(glob.glob(str(PDF_DIR / "*.pdf")))
    if not pdfs:
        print(f"[warn] No PDFs found in {PDF_DIR}")
        return

    rows: List[Dict[str, Any]] = []
    for p in pdfs:
        pdf_path = Path(p)
        print(f"[info] Processing: {pdf_path.name}")
        row = process_pdf(pdf_path)
        if row:
            rows.append(row)
        else:
            print(f"[warn] Skipped: {pdf_path.name}")

    if not rows:
        print("[warn] No rows parsed.")
        return

    write_outputs(rows)
    print(f"[ok] Parsed {len(rows)} rows from {len(pdfs)} PDFs.")


if __name__ == "__main__":
    main()

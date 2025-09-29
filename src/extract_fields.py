# src/extract_fields.py
import os, re, json, glob
from pathlib import Path
from collections import defaultdict

ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = ROOT / "artifacts"
OUT_DIR = ARTIFACTS / "extracted"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ---------- helpers ----------
def _num(s: str):
    if not s:
        return None
    s = s.replace(",", "").strip()
    try:
        return int(s)
    except:
        try:
            return float(s)
        except:
            return s

def _clean(s: str):
    return s.strip().replace("\x00", "")

# Regex patterns for typical fields
PATTERNS = {
    "structure_number": re.compile(r"Structure\s*Number\(\s*8\s*\)\s*:\s*([A-Z0-9\-]+)"),
    "state_name": re.compile(r"State\s*Name\(\s*1\s*\)\s*:\s*(.+)"),
    "county_name": re.compile(r"CountyName\(\s*3\s*\)\s*:\s*(.+)"),
    "year_built": re.compile(r"Year\s*Built\(\s*27\s*\)\s*:\s*(\d{4})"),
    "inspection_date": re.compile(r"InspectionDate\(\s*90\s*\)\s*:\s*([^\n]+)"),
    "deck_condition_rating": re.compile(r"Deck\s*ConditionRating\(\s*58\s*\)\s*:\s*(\d+)\s*-\s*([A-Za-z ]+)"),
    "superstructure_condition_rating": re.compile(r"SuperstructureConditionRating\(\s*59\s*\)\s*:\s*(\d+)"),
    "substructure_condition_rating": re.compile(r"SubstructureConditionRating\(\s*60\s*\)\s*:\s*(\d+)"),
    "bridge_condition_cat10": re.compile(r"BridgeCondition\(CAT10\)\s*:\s*([A-Z])\s*-\s*([A-Za-z ]+)"),
    "adt": re.compile(r"Average\s*Daily\s*Traffic\s*\(\s*29\s*\)\s*:\s*([\d,]+)"),
    "adt_year": re.compile(r"Year\s*of\s*Average\s*Daily\s*Traffic\s*\(\s*30\s*\)\s*:\s*(\d{4})"),
    "future_adt": re.compile(r"Future\s*Average\s*Daily\s*Traffic\s*\(\s*114\s*\)\s*:\s*([\d,]+)"),
    "future_adt_year": re.compile(r"Year\s*of\s*Future\s*Average\s*Daily\s*Traffic\s*\(\s*115\s*\)\s*:\s*(\d{4})"),
    "route_number": re.compile(r"Route\s*Number\(\s*5D\s*\)\s*:\s*([0-9A-Za-z\-]+)"),
}

FIELDS = [
    "source_file",
    "structure_number",
    "state_name",
    "county_name",
    "year_built",
    "inspection_date",
    "deck_condition_rating",         # number
    "deck_condition_desc",           # text
    "superstructure_condition_rating",
    "substructure_condition_rating",
    "bridge_condition_cat10_code",   # e.g., G
    "bridge_condition_cat10_desc",   # e.g., Good
    "adt", "adt_year",
    "future_adt", "future_adt_year",
    "route_number",
]

def parse_one_chunk(text: str, rec: dict):
    t = _clean(text)

    m = PATTERNS["structure_number"].search(t)
    if m and not rec.get("structure_number"):
        rec["structure_number"] = _clean(m.group(1))

    m = PATTERNS["state_name"].search(t)
    if m and not rec.get("state_name"):
        rec["state_name"] = _clean(m.group(1))

    m = PATTERNS["county_name"].search(t)
    if m and not rec.get("county_name"):
        rec["county_name"] = _clean(m.group(1))

    m = PATTERNS["year_built"].search(t)
    if m and not rec.get("year_built"):
        rec["year_built"] = _num(m.group(1))

    m = PATTERNS["inspection_date"].search(t)
    if m and not rec.get("inspection_date"):
        rec["inspection_date"] = _clean(m.group(1))

    m = PATTERNS["deck_condition_rating"].search(t)
    if m and not rec.get("deck_condition_rating"):
        rec["deck_condition_rating"] = _num(m.group(1))
        rec["deck_condition_desc"]   = _clean(m.group(2))

    m = PATTERNS["superstructure_condition_rating"].search(t)
    if m and not rec.get("superstructure_condition_rating"):
        rec["superstructure_condition_rating"] = _num(m.group(1))

    m = PATTERNS["substructure_condition_rating"].search(t)
    if m and not rec.get("substructure_condition_rating"):
        rec["substructure_condition_rating"] = _num(m.group(1))

    m = PATTERNS["bridge_condition_cat10"].search(t)
    if m and not rec.get("bridge_condition_cat10_code"):
        rec["bridge_condition_cat10_code"] = _clean(m.group(1))
        rec["bridge_condition_cat10_desc"] = _clean(m.group(2))

    m = PATTERNS["adt"].search(t)
    if m and not rec.get("adt"):
        rec["adt"] = _num(m.group(1))

    m = PATTERNS["adt_year"].search(t)
    if m and not rec.get("adt_year"):
        rec["adt_year"] = _num(m.group(1))

    m = PATTERNS["future_adt"].search(t)
    if m and not rec.get("future_adt"):
        rec["future_adt"] = _num(m.group(1))

    m = PATTERNS["future_adt_year"].search(t)
    if m and not rec.get("future_adt_year"):
        rec["future_adt_year"] = _num(m.group(1))

    m = PATTERNS["route_number"].search(t)
    if m and not rec.get("route_number"):
        rec["route_number"] = _clean(m.group(1))

def main():
    files = glob.glob(str(ARTIFACTS / "*" / "*_chunks.jsonl"))  # artifacts/2023/.. 等
    if not files:
        print("[warn] No chunks found. Did you run ingest_pdf.py ?")
        return

    # aggregate by source_file (bridge)
    aggr = defaultdict(dict)

    total_lines = 0
    for f in files:
        with open(f, "r", encoding="utf-8") as fh:
            for line in fh:
                total_lines += 1
                try:
                    d = json.loads(line)
                except:
                    continue
                text = d.get("text", "")
                meta = d.get("metadata", {}) or {}
                src  = meta.get("bridge") or meta.get("source") or meta.get("source_file") or Path(f).name

                rec = aggr[src]
                rec.setdefault("source_file", src)
                parse_one_chunk(text, rec)

    rows = []
    for src, rec in aggr.items():
        row = {k: rec.get(k) for k in FIELDS}
        rows.append(row)

    # write JSON
    out_json = OUT_DIR / "bridge_fields.json"
    with open(out_json, "w", encoding="utf-8") as fh:
        json.dump(rows, fh, ensure_ascii=False, indent=2)
    print(f"[ok] JSON saved -> {out_json}")

    # write CSV
    out_csv = OUT_DIR / "bridge_fields.csv"
    import csv
    with open(out_csv, "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"[ok] CSV  saved -> {out_csv}")
    print(f"[ok] Parsed {len(rows)} bridges from {len(files)} files, {total_lines} chunks.")

if __name__ == "__main__":
    main()

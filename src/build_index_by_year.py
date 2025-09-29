# src/build_index_by_year.py
import os, glob, json
from pathlib import Path
from typing import List
from dotenv import load_dotenv

from langchain.docstore.document import Document
from langchain_community.vectorstores import FAISS
from langchain.embeddings.base import Embeddings
from langchain_openai import OpenAIEmbeddings
from langchain_community.embeddings import HuggingFaceEmbeddings

ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = ROOT / "artifacts"
INDEX_ROOT = ARTIFACTS / "index"

def load_json_docs(files: List[str]) -> List[Document]:
    docs: List[Document] = []
    for f in files:
        with open(f, "r", encoding="utf-8") as fh:
            for line in fh:
                d = json.loads(line)
                text = (d.get("text") or "").strip()
                meta = d.get("metadata", {}) or {}
                if text:
                    docs.append(Document(page_content=text, metadata=meta))
    return docs

def get_embeddings() -> Embeddings:
    # 与你的 build_index.py 相同的策略：有 OPENAI_API_KEY 用 openai，否则本地
    key = os.getenv("OPENAI_API_KEY", "").strip()
    backend = os.getenv("EMBED_BACKEND", "auto").lower()
    if backend == "openai" or (backend == "auto" and key):
        print("[embed] Using OpenAI text-embedding-3-small")
        return OpenAIEmbeddings(model="text-embedding-3-small")
    else:
        print("[embed] Using local sentence-transformers/all-MiniLM-L6-v2")
        return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

def build_for_year(year_dir: Path, emb: Embeddings) -> int:
    files = glob.glob(str(year_dir / "*_chunks.jsonl"))
    if not files:
        return 0
    docs = load_json_docs(files)
    if not docs:
        return 0
    db = FAISS.from_documents(docs, emb)
    outdir = INDEX_ROOT / str(year_dir.name) / "faiss"
    outdir.mkdir(parents=True, exist_ok=True)
    db.save_local(str(outdir))
    print(f"[ok] Year {year_dir.name} index saved -> {outdir}")
    return len(docs)

def build_merged_all(emb: Embeddings) -> int:
    files = glob.glob(str(ARTIFACTS / "*" / "*_chunks.jsonl"))
    docs = load_json_docs(files)
    if not docs:
        return 0
    db = FAISS.from_documents(docs, emb)
    outdir = INDEX_ROOT / "all" / "faiss"
    outdir.mkdir(parents=True, exist_ok=True)
    db.save_local(str(outdir))
    print(f"[ok] Merged 'all' index saved -> {outdir}")
    return len(docs)

def main():
    load_dotenv()
    emb = get_embeddings()
    years = sorted([p for p in ARTIFACTS.iterdir() if p.is_dir() and p.name.isdigit()])

    total_docs = 0
    for yd in years:
        total_docs += build_for_year(yd, emb)
    merged_docs = build_merged_all(emb)
    print(f"[done] per-year total docs={total_docs}, merged(all) docs={merged_docs}")

if __name__ == "__main__":
    main()

# src/build_index.py
import os, json, glob
from pathlib import Path
from typing import List

from dotenv import load_dotenv
from langchain.docstore.document import Document
from langchain_community.vectorstores import FAISS
from langchain.embeddings.base import Embeddings
from langchain_openai import OpenAIEmbeddings
from langchain_community.embeddings import HuggingFaceEmbeddings

load_dotenv()

ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = ROOT / "artifacts"
INDEX_DIR = ARTIFACTS / "index" / "faiss"
INDEX_DIR.mkdir(parents=True, exist_ok=True)

def load_jsonl_docs() -> List[Document]:
    files = glob.glob(str(ARTIFACTS / "*" / "*_chunks.jsonl"))
    docs: List[Document] = []
    for f in files:
        with open(f, "r", encoding="utf-8") as fh:
            for line in fh:
                d = json.loads(line)
                text = d.get("text", "").strip()
                meta = d.get("metadata", {}) or {}
                meta["source_file"] = Path(f).name
                if text:
                    docs.append(Document(page_content=text, metadata=meta))
    print(f"[ok] Loaded {len(docs)} docs from {len(files)} files.")
    return docs

def get_embeddings() -> Embeddings:
    key = os.getenv("OPENAI_API_KEY", "").strip()
    backend = os.getenv("EMBED_BACKEND", "auto").lower()
    if backend == "openai" or (backend == "auto" and key):
        print("[embed] Using OpenAI text-embedding-3-small")
        return OpenAIEmbeddings(model="text-embedding-3-small")
    else:
        print("[embed] Using local sentence-transformers/all-MiniLM-L6-v2")
        return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

def main():
    docs = load_jsonl_docs()
    if not docs:
        print("[warn] No documents found. Did you run ingest_pdf.py?")
        return
    embeddings = get_embeddings()
    db = FAISS.from_documents(docs, embeddings)
    db.save_local(str(INDEX_DIR))
    print(f"[ok] FAISS index saved -> {INDEX_DIR}")

if __name__ == "__main__":
    main()

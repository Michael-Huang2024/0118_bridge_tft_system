# src/rag_query.py
import os
from pathlib import Path
from typing import List

from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage

load_dotenv()

ROOT = Path(__file__).resolve().parents[1]
INDEX_DIR = ROOT / "artifacts" / "index" / "faiss"

def load_db():
    if not INDEX_DIR.exists():
        raise FileNotFoundError(f"Index not found: {INDEX_DIR}. Run build_index.py first.")
    # NOTE: use the same embedding backend for loading FAISS (metadata only)
    # FAISS can load without embeddings, but for safety pass dummy OpenAIEmbeddings if available.
    from langchain_openai import OpenAIEmbeddings
    from langchain_community.embeddings import HuggingFaceEmbeddings

    key = os.getenv("OPENAI_API_KEY", "").strip()
    backend = os.getenv("EMBED_BACKEND", "auto").lower()
    if backend == "openai" or (backend == "auto" and key):
        emb = OpenAIEmbeddings(model="text-embedding-3-small")
    else:
        emb = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    return FAISS.load_local(str(INDEX_DIR), emb, allow_dangerous_deserialization=True)

def pretty_sources(docs: List):
    out = []
    for i, d in enumerate(docs, 1):
        md = d.metadata or {}
        where = f"{md.get('source_file','')} | page={md.get('page','?')} | chunk={md.get('chunk','?')}"
        out.append(f"[{i}] {where}\n{d.page_content[:500]}...")
    return "\n\n".join(out)

def answer_with_llm(query: str, docs: List):
    system = (
        "You are a helpful assistant for bridge maintenance. "
        "Answer strictly using the provided context. "
        "If you are not sure, say you don't know."
    )
    context = "\n\n---\n\n".join([d.page_content for d in docs])
    user = f"Question: {query}\n\nContext:\n{context}"
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    msg = llm([SystemMessage(content=system), HumanMessage(content=user)])
    return msg.content

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-q", "--query", required=True, help="Your question")
    parser.add_argument("-k", "--topk", type=int, default=4, help="retrieval top-k")
    args = parser.parse_args()

    db = load_db()
    retriever = db.as_retriever(search_type="mmr", search_kwargs={"k": args.topk})
    docs = retriever.get_relevant_documents(args.query)

    if not docs:
        print("No relevant chunks found.")
        return

    if os.getenv("OPENAI_API_KEY", "").strip():
        ans = answer_with_llm(args.query, docs)
        print("\n=== Answer ===\n", ans)
        print("\n=== Sources ===\n", pretty_sources(docs))
    else:
        # fallback: extractive (no LLM)
        print("\n[no OPENAI_API_KEY] Showing top context chunks instead:\n")
        print(pretty_sources(docs))

if __name__ == "__main__":
    main()

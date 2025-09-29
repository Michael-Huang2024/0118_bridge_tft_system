# src/check_deps.py
import importlib

required = [
    "pdfplumber",
    "fitz",               # PyMuPDF
    "langchain",
    "langchain_community",
    "langchain_openai",
    "faiss",
    "tiktoken",
    "dotenv",
    "regex",
    "unidecode",
    "pandas",
    "matplotlib"
]

print("🔍 Checking dependencies...\n")
for pkg in required:
    try:
        importlib.import_module(pkg)
        print(f"✅ {pkg} 已安装")
    except ImportError:
        print(f"❌ {pkg} 未安装，请运行: pip install {pkg}")

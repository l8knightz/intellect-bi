# etl/index_docs.py
import os, glob, math
from pathlib import Path
from typing import List, Tuple
import chromadb
from chromadb.config import Settings
from pypdf import PdfReader
import docx2txt

CHROMA_DIR = os.environ.get("CHROMA_PERSIST_DIR", "./vector")
DATA_DIR = os.environ.get("DATA_DIR", "./data")
DOCS_DIR = os.path.join(DATA_DIR, "docs")
COLLECTION_NAME = "docs"

def read_pdf(path: str) -> Tuple[str, List[Tuple[int, str]]]:
    reader = PdfReader(path)
    chunks = []
    pages = []
    for i, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        pages.append((i, text))
    return os.path.basename(path), pages

def read_docx(path: str) -> Tuple[str, List[Tuple[int, str]]]:
    text = docx2txt.process(path) or ""
    # treat as single “page”
    return os.path.basename(path), [(1, text)]

def read_text(path: str) -> Tuple[str, List[Tuple[int, str]]]:
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        text = f.read()
    return os.path.basename(path), [(1, text)]

def chunk_text(text: str, size: int = 1500, overlap: int = 200) -> List[str]:
    text = " ".join(text.split())
    if not text:
        return []
    chunks = []
    start = 0
    while start < len(text):
        end = start + size
        chunks.append(text[start:end])
        start = end - overlap
        if start < 0:
            start = 0
    return chunks

def main():
    Path(CHROMA_DIR).mkdir(parents=True, exist_ok=True)
    Path(DOCS_DIR).mkdir(parents=True, exist_ok=True)

    # Client + embedding function
    from api.ollama_embedder import OllamaEmbeddingFunction  # type: ignore
    embedder = OllamaEmbeddingFunction()

    client = chromadb.PersistentClient(path=CHROMA_DIR, settings=Settings(anonymized_telemetry=False))

    # get_or_create covers both cases across chromadb 0.5/1.x lines
    try:
        collection = client.get_or_create_collection(name=COLLECTION_NAME, embedding_function=embedder, metadata={"hnsw:space": "cosine"})
    except Exception:
        # fallback path (older apis)
        try:
            collection = client.get_collection(COLLECTION_NAME, embedding_function=embedder)
        except Exception:
            collection = client.create_collection(COLLECTION_NAME, embedding_function=embedder, metadata={"hnsw:space": "cosine"})

    files = []
    files += glob.glob(os.path.join(DOCS_DIR, "*.pdf"))
    files += glob.glob(os.path.join(DOCS_DIR, "*.docx"))
    files += glob.glob(os.path.join(DOCS_DIR, "*.txt"))
    files += glob.glob(os.path.join(DOCS_DIR, "*.md"))

    if not files:
        print(f"No docs found in {DOCS_DIR}. Place .pdf/.docx/.txt/.md files there.")
        return

    ids, documents, metadatas = [], [], []
    doc_id_counter = 0

    for f in files:
        ext = os.path.splitext(f)[1].lower()
        if ext == ".pdf":
            fname, pages = read_pdf(f)
        elif ext == ".docx":
            fname, pages = read_docx(f)
        else:
            fname, pages = read_text(f)

        for page_num, page_text in pages:
            for j, chunk in enumerate(chunk_text(page_text)):
                doc_id_counter += 1
                ids.append(f"{fname}::p{page_num}::c{j}")
                documents.append(chunk)
                metadatas.append({"source": fname, "page": page_num, "chunk": j})

    if not ids:
        print("Parsed docs but no non-empty chunks were produced.")
        return

    # Upsert in manageable batches
    BATCH = 128
    for i in range(0, len(ids), BATCH):
        collection.upsert(
            ids=ids[i:i+BATCH],
            documents=documents[i:i+BATCH],
            metadatas=metadatas[i:i+BATCH],
        )
    print(f"Indexed {len(ids)} chunks from {len(files)} files into collection '{COLLECTION_NAME}' at {CHROMA_DIR}")

if __name__ == "__main__":
    main()

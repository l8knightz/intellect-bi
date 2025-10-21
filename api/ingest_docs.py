# ingest_docs.py (safe low-memory ingestion)
import os, gc, uuid, pathlib, itertools, time
from typing import Iterable, List, Dict
from chromadb.config import Settings
import chromadb

from ollama_embedder import OllamaEmbeddingFunction

# ---- env knobs ----
CHROMA_DIR = os.getenv("CHROMA_PERSIST_DIR", "./vector")
DATA_DIR = os.getenv("DATA_DIR", "./data/docs")
CHUNK_SIZE = int(os.getenv("INGEST_CHUNK_SIZE", "800"))
CHUNK_OVERLAP = int(os.getenv("INGEST_CHUNK_OVERLAP", "120"))
INGEST_BATCH_CHUNKS = int(os.getenv("INGEST_BATCH_CHUNKS", "32"))
EMBED_BATCH = int(os.getenv("EMBED_BATCH", "16"))

# ---- init ----
client = chromadb.PersistentClient(path=CHROMA_DIR, settings=Settings(anonymized_telemetry=False))
try:
    coll = client.get_collection("docs")
except Exception:
    coll = client.create_collection(name="docs", metadata={"hnsw:space":"cosine"})

embedder = OllamaEmbeddingFunction()

# ---- simple PDF/text loader + splitter (use your existing if you have one) ----
def load_texts(path: str) -> List[Dict]:
    """Return [{'text': <chunk>, 'meta': {...}} ...] for a single file."""
    p = pathlib.Path(path)
    if p.suffix.lower() in {".pdf"}:
        # very light PDF load (no page images)
        import pypdf
        reader = pypdf.PdfReader(str(p))
        pages = []
        for i, page in enumerate(reader.pages):
            try:
                pages.append((i+1, page.extract_text() or ""))
            except Exception:
                pages.append((i+1, ""))
    else:
        txt = p.read_text(errors="ignore")
        pages = [(1, txt)]

    # recursive-like splitter, but simple and light
    chunks = []
    for page_num, text in pages:
        text = " ".join((text or "").split())
        if not text:
            continue
        start = 0
        while start < len(text):
            end = min(start + CHUNK_SIZE, len(text))
            chunk = text[start:end]
            chunks.append({
                "text": chunk,
                "meta": {
                    "source": p.name,
                    "page": page_num,
                }
            })
            start = end - CHUNK_OVERLAP if end - CHUNK_OVERLAP > start else end
    return chunks

def batched(iterable: Iterable, n: int):
    it = iter(iterable)
    while True:
        batch = list(itertools.islice(it, n))
        if not batch:
            return
        yield batch

def _retry_embed_one(text: str, attempts: int = 3, delay: float = 0.2):
    """
    Call embedder.embed_one with small retries to survive transient Ollama hiccups.
    """
    for i in range(attempts):
        try:
            return embedder.embed_one(text)
        except Exception:
            if i == attempts - 1:
                raise
            time.sleep(delay)
            delay = min(delay * 2, 1.0)

def embed_texts(texts: List[str]) -> List[List[float]]:
    """
    Low-memory batching that uses embed_one() per text.
    """
    vecs: List[List[float]] = []
    for sub in batched(texts, EMBED_BATCH):
        for t in sub:
            vecs.append(_retry_embed_one(t))
            # tiny pause keeps memory/CPU flatter on some hosts
            time.sleep(0.005)
    return vecs

def upsert_batch(batch: List[Dict]):
    ids = [str(uuid.uuid4()) for _ in batch]
    docs = [b["text"] for b in batch]
    metas = [b["meta"] for b in batch]
    embs = embed_texts(docs)
    coll.add(ids=ids, metadatas=metas, documents=docs, embeddings=embs)

def iter_files(root: str):
    exts = {".pdf", ".txt", ".md"}
    rootp = pathlib.Path(root)
    # recurse (rglob) so /app/data/docs works even if DATA_DIR=/app/data
    for p in sorted(rootp.rglob("*")):
        if p.is_file() and p.suffix.lower() in exts:
            yield str(p)

def main():
    files = list(iter_files(DATA_DIR))
    if not files:
        print(f"[ingest] no files found in {DATA_DIR}")
        return

    print(f"[ingest] start, files={len(files)} | chunk={CHUNK_SIZE}/{CHUNK_OVERLAP} | batch={INGEST_BATCH_CHUNKS} | embed_batch={EMBED_BATCH}")
    total_chunks = 0
    for f in files:
        chunks = load_texts(f)
        print(f"[ingest] {pathlib.Path(f).name}: chunks={len(chunks)}")
        for batch in batched(chunks, INGEST_BATCH_CHUNKS):
            upsert_batch(batch)
            total_chunks += len(batch)
            # free memory aggressively
            del batch[:]
            gc.collect()
        # one more GC after each file
        gc.collect()
    print(f"[ingest] done. total_chunks={total_chunks}")

if __name__ == "__main__":
    main()

import os, time, requests
from typing import List, Sequence, Optional

DEFAULT_TIMEOUT = float(os.environ.get("EMBED_TIMEOUT_S", "180"))   # was 60
RETRIES = int(os.environ.get("EMBED_RETRIES", "4"))
BACKOFF = float(os.environ.get("EMBED_BACKOFF_S", "1.5"))

class OllamaEmbeddingFunction:
    """
    Chroma 1.x-compatible embedding function using Ollama /api/embeddings.
    Retries on timeouts/transient errors. Also exposes embed_one().
    """
    def __init__(self, model: Optional[str] = None, base_url: Optional[str] = None, timeout: float = DEFAULT_TIMEOUT):
        self.model = model or os.environ.get("EMBEDDING_MODEL", "nomic-embed-text")
        self.base_url = (base_url or os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")).rstrip("/")
        self.timeout = timeout
        self._session = requests.Session()  # keep-alive between calls

    def name(self) -> str:
        return f"ollama:{self.model}"

    def to_dict(self):
        return {"provider": "ollama", "model": self.model, "base_url": self.base_url}

    def __call__(self, input: Sequence[str]) -> List[List[float]]:
        return [self.embed_one(t) for t in input]

    def embed_one(self, text: str) -> List[float]:
        last_err = None
        for attempt in range(1, RETRIES + 1):
            try:
                r = self._session.post(
                    f"{self.base_url}/api/embeddings",
                    json={"model": self.model, "prompt": text},
                    timeout=self.timeout,  # applies to both connect & read
                )
                r.raise_for_status()
                return r.json()["embedding"]
            except Exception as e:
                last_err = e
                # backoff with jitter
                sleep_s = BACKOFF * attempt
                time.sleep(sleep_s)
        raise RuntimeError(f"embedding failed after {RETRIES} retries: {last_err}")

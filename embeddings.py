from typing import List

import numpy as np

try:
    from sentence_transformers import SentenceTransformer
except Exception:  # pragma: no cover - user may not have package installed yet
    SentenceTransformer = None


class Embedder:
    """Embedder using sentence-transformers. Falls back to a simple random embeddings
    if the package is not available (useful for quick demos).
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        if SentenceTransformer is not None:
            self.model = SentenceTransformer(model_name)
        else:
            self.model = None

    def embed(self, texts: List[str]) -> List[List[float]]:
        if self.model is not None:
            arr = self.model.encode(texts, show_progress_bar=False, convert_to_numpy=True)
            return arr.tolist()

        # Fallback: deterministic hashing-based vectors (not semantically meaningful,
        # but lets the demo run without heavy deps).
        vectors = []
        for t in texts:
            h = abs(hash(t))
            rng = np.random.RandomState(h % (2 ** 32))
            vectors.append(rng.rand(384).tolist())
        return vectors


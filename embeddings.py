from typing import List

import numpy as np

from utils.errors import EmbeddingError

try:
    from sentence_transformers import SentenceTransformer
except ImportError:  # pragma: no cover - user may not have package installed yet
    SentenceTransformer = None


class Embedder:
    """Embedder using sentence-transformers. Falls back to a simple random embeddings
    if the package is not available (useful for quick demos).
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """Initialize embedder with specified model.
        
        Args:
            model_name: Name of sentence-transformers model to use.
            
        Raises:
            EmbeddingError: If model cannot be loaded (when sentence-transformers available).
        """
        self.model_name = model_name
        self.model = None
        
        if SentenceTransformer is not None:
            try:
                self.model = SentenceTransformer(model_name)
            except Exception as e:
                raise EmbeddingError(
                    f"Failed to load embedding model '{model_name}': {e}. "
                    f"Ensure the model name is valid or install sentence-transformers."
                ) from e

    def embed(self, texts: List[str]) -> List[List[float]]:
        """Embed a list of texts.
        
        Args:
            texts: List of text strings to embed.
            
        Returns:
            List of embedding vectors (each vector is a list of floats).
            
        Raises:
            EmbeddingError: If embedding fails.
        """
        if not texts:
            raise EmbeddingError("Cannot embed empty text list.")
        
        if self.model is not None:
            try:
                arr = self.model.encode(texts, show_progress_bar=False, convert_to_numpy=True)
                return arr.tolist()
            except Exception as e:
                raise EmbeddingError(f"Embedding failed: {e}") from e

        # Fallback: deterministic hashing-based vectors (not semantically meaningful,
        # but lets the demo run without heavy deps).
        vectors = []
        try:
            for t in texts:
                h = abs(hash(t))
                rng = np.random.RandomState(h % (2 ** 32))
                vectors.append(rng.rand(384).tolist())
            return vectors
        except Exception as e:
            raise EmbeddingError(f"Fallback embedding failed: {e}") from e


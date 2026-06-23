from typing import List, Tuple

import numpy as np

try:
	from sklearn.neighbors import NearestNeighbors
except Exception:  # pragma: no cover - fallback if scikit-learn not installed
	NearestNeighbors = None


class VectorStore:
	"""A lightweight vector store using scikit-learn NearestNeighbors with cosine distance."""

	def __init__(self, vectors: List[List[float]], chunks: List[str]):
		self.vectors = np.array(vectors, dtype=float)
		self.chunks = list(chunks)
		if len(self.vectors) > 0 and NearestNeighbors is not None:
			self.nn = NearestNeighbors(n_neighbors=min(10, len(self.vectors)), metric="cosine")
			self.nn.fit(self.vectors)
		else:
			self.nn = None

	def search(self, query_vector: List[float], k: int = 3) -> List[Tuple[float, str]]:
		"""Return a list of (distance, chunk) pairs for the k nearest neighbors."""
		query_vector = np.array(query_vector, dtype=float)
		k = min(k, len(self.chunks))

		# Preferred: use scikit-learn NearestNeighbors if available
		if self.nn is not None:
			dists, idxs = self.nn.kneighbors(query_vector.reshape(1, -1), n_neighbors=k)
			results = []
			for dist, idx in zip(dists[0], idxs[0]):
				results.append((float(dist), self.chunks[int(idx)]))
			return results

		# Fallback: brute-force cosine distances
		# cosine distance = 1 - cosine_similarity
		norms = np.linalg.norm(self.vectors, axis=1) * np.linalg.norm(query_vector)
		sims = np.dot(self.vectors, query_vector)
		# avoid division by zero
		with np.errstate(invalid="ignore", divide="ignore"):
			sims = sims / np.where(norms == 0, 1e-8, norms)
		dists = 1.0 - sims
		idxs = np.argsort(dists)[:k]
		return [(float(dists[i]), self.chunks[int(i)]) for i in idxs]



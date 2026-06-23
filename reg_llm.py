from typing import List

from embeddings import Embedder
from vector_store import VectorStore
from chunking import chunk_text
from ingest import load_documents

_embedder = Embedder()
_store: VectorStore | None = None


def build_index(docs: List[str], chunk_size: int = 200, overlap: int = 50) -> VectorStore:
    """Chunk documents, embed chunks, and build a VectorStore."""
    chunks = []
    for d in docs:
        chunks.extend(chunk_text(d, chunk_size=chunk_size, overlap=overlap))

    vectors = _embedder.embed(chunks)
    global _store
    _store = VectorStore(vectors, chunks)
    return _store


def retrieve(query: str, k: int = 3) -> List[str]:
    """Return the top-k retrieved chunks for `query`."""
    if _store is None:
        raise RuntimeError("Vector store not built. Call build_index(...) first.")
    qv = _embedder.embed([query])[0]
    hits = _store.search(qv, k=k)
    return [chunk for _, chunk in hits]


def generate_answer(query: str, contexts: List[str]) -> str:
    """Simple answer synthesis: score sentences by token overlap with query.

    This is intentionally lightweight (no external LLM). It demonstrates how
    retrieved context can be combined to produce an answer.
    """
    if not contexts:
        return "No context found."

    qtokens = set(q.lower() for q in query.split())
    best_sentence = None
    best_score = -1

    for ctx in contexts:
        for sent in ctx.split("."):
            tokens = set(tok.lower().strip(".,;:()[]\"'") for tok in sent.split())
            score = len(qtokens & tokens)
            if score > best_score and sent.strip():
                best_score = score
                best_sentence = sent.strip()

    if best_sentence and best_score > 0:
        return f"Answer (extracted): {best_sentence}.\n\n---\nRetrieved context:\n\n" + "\n\n".join(contexts)

    # Fallback: return concatenated contexts
    return "\n\n".join(contexts)


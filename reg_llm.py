from typing import List

import config
from chunking import chunk_text
from embeddings import Embedder
from ingest import load_documents
from vector_store import VectorStore
from utils.errors import ChunkingError, IndexNotBuiltError

_embedder = Embedder()
_store: VectorStore | None = None


def build_index(docs: List[str], chunk_size: int = 200, overlap: int = 50) -> VectorStore:
    """Chunk documents, embed chunks, and build a VectorStore.
    
    Args:
        docs: List of document strings to index.
        chunk_size: Size of each chunk in characters.
        overlap: Overlap between consecutive chunks.
        
    Returns:
        The built VectorStore object.
        
    Raises:
        ChunkingError: If chunking fails.
        EmbeddingError: If embedding fails.
        VectorStoreError: If vector store creation fails.
    """
    try:
        if not docs:
            raise ChunkingError("Cannot build index from empty document list.")
        
        chunks = []
        for i, d in enumerate(docs):
            try:
                doc_chunks = chunk_text(d, chunk_size=chunk_size, overlap=overlap)
                chunks.extend(doc_chunks)
            except ChunkingError as e:
                raise ChunkingError(f"Failed to chunk document {i}: {e}") from e

        if not chunks:
            raise ChunkingError("No chunks produced from documents.")

        vectors = _embedder.embed(chunks)
        global _store
        _store = VectorStore(vectors, chunks)
        return _store
    except Exception:
        raise


def retrieve(query: str, k: int = 3) -> List[str]:
    """Return the top-k retrieved chunks for `query`.
    
    Args:
        query: The search query string.
        k: Number of chunks to retrieve.
        
    Returns:
        List of top-k most relevant chunks.
        
    Raises:
        IndexNotBuiltError: If index hasn't been built yet.
        EmbeddingError: If query embedding fails.
    """
    if _store is None:
        raise IndexNotBuiltError(
            "Vector store not built. Call build_index(...) first to create the index."
        )
    
    if not query.strip():
        raise ValueError("Query cannot be empty.")
    
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


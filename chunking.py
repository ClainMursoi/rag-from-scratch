from typing import List


def chunk_text(text: str, chunk_size: int = 200, overlap: int = 50) -> List[str]:
    """Split `text` into fixed-size character chunks with `overlap`.

    This is a simple, explicit chunker suitable for demonstration and
    experimentation.
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be > 0")

    chunks: List[str] = []
    start = 0
    text_len = len(text)
    while start < text_len:
        end = start + chunk_size
        chunks.append(text[start:end])
        start = max(end - overlap, end)
    return chunks


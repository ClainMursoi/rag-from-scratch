"""Small interactive demo for the RAG-from-scratch pipeline.

Usage:
    python demo.py                                   # Use default config
    python demo.py --chunk-size 300 --overlap 100   # Override chunk settings
    python demo.py --k 5                             # Retrieve 5 chunks instead of 3

It builds an index from `data/sample.txt` and then accepts queries on the command
line, printing retrieved context and a lightweight synthesized answer.
"""
import argparse
from pathlib import Path
from typing import Optional

import config
from ingest import load_documents
from reg_llm import build_index, retrieve, generate_answer


def main(chunk_size: Optional[int] = None, overlap: Optional[int] = None, 
         k: Optional[int] = None) -> None:
    """Run the interactive RAG demo.
    
    Args:
        chunk_size: Override default chunk size.
        overlap: Override default chunk overlap.
        k: Override default retrieval k value.
    """
    # Update config if overrides provided
    overrides = {}
    if chunk_size is not None:
        overrides["chunk_size"] = chunk_size
    if overlap is not None:
        overrides["chunk_overlap"] = overlap
    if k is not None:
        overrides["retrieval_k"] = k
    
    if overrides:
        config.update_from_dict(overrides)
    
    data_path = Path(__file__).parent / config.DATA_PATH
    if not data_path.exists():
        print(f"Missing sample data at {data_path}.")
        return

    docs = load_documents(str(data_path))
    print(f"Loaded {len(docs)} documents. Building index...")
    print(f"  Chunk size: {config.CHUNK_SIZE}, Overlap: {config.CHUNK_OVERLAP}")
    build_index(docs, chunk_size=config.CHUNK_SIZE, overlap=config.CHUNK_OVERLAP)
    print("Index built. You can now enter queries (empty line to quit).\n")

    while True:
        q = input("Query> ").strip()
        if not q:
            break
        contexts = retrieve(q, k=config.RETRIEVAL_K)
        answer = generate_answer(q, contexts)
        print("\n" + answer + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Interactive RAG demo with configurable parameters"
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=None,
        help=f"Chunk size in characters (default: {config.CHUNK_SIZE})"
    )
    parser.add_argument(
        "--overlap",
        type=int,
        default=None,
        help=f"Chunk overlap in characters (default: {config.CHUNK_OVERLAP})"
    )
    parser.add_argument(
        "--k",
        type=int,
        default=None,
        help=f"Number of chunks to retrieve (default: {config.RETRIEVAL_K})"
    )
    
    args = parser.parse_args()
    main(chunk_size=args.chunk_size, overlap=args.overlap, k=args.k)

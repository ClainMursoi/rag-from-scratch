"""Small interactive demo for the RAG-from-scratch pipeline.

Usage:
    python demo.py

It builds an index from `data/sample.txt` (created with the repo) and then
accepts queries on the command line, printing retrieved context and a
lightweight synthesized answer.
"""
from pathlib import Path

from ingest import load_documents
from reg_llm import build_index, retrieve, generate_answer


DATA = Path(__file__).parent / "data" / "sample.txt"


def main():
    if not DATA.exists():
        print(f"Missing sample data at {DATA}.")
        return

    docs = load_documents(str(DATA))
    print(f"Loaded {len(docs)} documents. Building index...")
    build_index(docs)
    print("Index built. You can now enter queries (empty line to quit).\n")

    while True:
        q = input("Query> ").strip()
        if not q:
            break
        contexts = retrieve(q, k=3)
        answer = generate_answer(q, contexts)
        print("\n" + answer + "\n")


if __name__ == "__main__":
    main()

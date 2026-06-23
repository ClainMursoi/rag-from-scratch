# RAG From Scratch

A minimal implementation of Retrieval-Augmented Generation without frameworks.

## Why this project
Most RAG tutorials hide key decisions behind abstractions.  
This project exposes chunking, retrieval, and evaluation choices explicitly.

## Architecture
Document → Chunk → Embed → Retrieve → Generate

## Experiments
- Fixed vs overlapping chunking
- Retrieval recall@k

## Key Takeaways
(TBD)
## Quick demo

This repository contains a minimal RAG pipeline. Run the interactive demo:

1. Install dependencies:

```bash
pip install -r requirements.txt
```

2. Run the demo:

```bash
python demo.py
```

The demo builds an index from `data/sample.txt`, then accepts queries and
prints retrieved context plus a lightweight synthesized answer (no external
LLM required).

## What the implementation does

- `ingest.py` — loads documents (splits on double-newline).
- `chunking.py` — splits documents into fixed-size, overlapping character chunks.
- `embeddings.py` — wraps `sentence-transformers` (`all-MiniLM-L6-v2`) and
	falls back to a deterministic random vector generator if the model isn't available.
- `vector_store.py` — nearest-neighbor search using `scikit-learn`'s
	`NearestNeighbors` with cosine distance.
- `reg_llm.py` — ties components together: `build_index`, `retrieve`, and
	`generate_answer` (lightweight sentence-extraction synthesis).
- `demo.py` — interactive demo that runs the full pipeline.

This is intentionally minimal so you can experiment with chunk sizes, overlap,
and retrieval parameters.

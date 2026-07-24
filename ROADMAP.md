# RAG From Scratch - Development Roadmap

A structured guide to expanding this project with daily contributions.

## Priority Levels
- 🟢 **Quick Wins** — 30-60 min, single file changes
- 🟡 **Core Features** — 1-2 hours, impacts core pipeline
- 🔴 **Advanced** — 2+ hours, significant refactoring
- 🔵 **Infrastructure** — Testing, CI/CD, DevOps

---

## Phase 1: Quick Wins (This Week)
Easy daily contributions to build momentum.

### 🟢 Better Error Handling
- [ ] Add try-catch blocks to `ingest.py`, `embeddings.py`, `vector_store.py`
- [ ] Improve error messages with helpful debugging info
- [ ] File: Create `utils/errors.py` with custom exceptions

### 🟢 Logging System
- [ ] Add structured logging to all modules
- [ ] Log chunk counts, embedding times, retrieval metrics
- [ ] File: Add logging config to `utils/logging.py`

### 🟢 Type Hints
- [ ] Add type hints to all functions in `ingest.py`
- [ ] Add type hints to `chunking.py`
- [ ] Add type hints to `embeddings.py`
- [ ] Add type hints to `vector_store.py`

### 🟢 Unit Tests - Part 1
- [ ] Create `tests/test_chunking.py` — test fixed chunking, overlaps
- [ ] Create `tests/test_ingest.py` — test document loading
- [ ] Create `tests/test_embeddings.py` — test vector generation

### 🟢 Documentation
- [ ] Add module docstrings to all `.py` files
- [ ] Create `ARCHITECTURE.md` with detailed pipeline diagram
- [ ] Create `CONTRIBUTING.md` for future contributors

### 🟢 Config System - Basic
- [ ] Create `config.py` with default chunk size, overlap, k-value
- [ ] Allow `demo.py` to accept command-line args for these values
- [ ] Files: `config.py`

---

## Phase 2: Core Features (Weeks 2-3)
Meaningful enhancements to the pipeline.

### 🟡 Multiple Embedding Models
- [ ] Refactor `embeddings.py` to support model selection
- [ ] Add support for: `all-MiniLM-L6-v2`, `all-mpnet-base-v2`, `distiluse-base-multilingual`
- [ ] Add environment variable to choose model
- [ ] Files: Modify `embeddings.py`, `config.py`

### 🟡 Semantic Chunking
- [ ] Implement sentence-based splitting (use `sentence_splitter` or NLTK)
- [ ] Create `chunking_semantic.py` with semantic-aware chunking
- [ ] Compare fixed vs semantic chunking in `demo.py`
- [ ] Files: `chunking_semantic.py`, update `demo.py`

### 🟡 Query Expansion
- [ ] Create `query_expansion.py` with synonym expansion
- [ ] Use NLTK WordNet or simple keyword extraction
- [ ] Integrate into retrieval pipeline
- [ ] Files: `query_expansion.py`, modify `reg_llm.py`

### 🟡 Hybrid Search (BM25 + Semantic)
- [ ] Implement BM25 keyword search in `vector_store.py`
- [ ] Add hybrid search that combines BM25 + cosine similarity scores
- [ ] Files: Modify `vector_store.py`, update `reg_llm.py`

### 🟡 Performance Metrics
- [ ] Create `metrics.py` with retrieval latency, index size, precision/recall
- [ ] Log metrics during indexing and retrieval
- [ ] Add `benchmark.py` script to compare configurations
- [ ] Files: `metrics.py`, `benchmark.py`

### 🟡 Batch Processing
- [ ] Update `ingest.py` to load multiple files from a directory
- [ ] Create `demo_batch.py` for multi-document indexing
- [ ] Files: Modify `ingest.py`, create `demo_batch.py`

---

## Phase 3: Advanced Features (Weeks 4+)
Significant enhancements.

### 🔴 Re-ranking Layer
- [ ] Implement cross-encoder re-ranker (optional: use `sentence-transformers` cross-encoder)
- [ ] Add `re_ranker.py` module
- [ ] Integrate into retrieval pipeline for top-k refinement
- [ ] Files: `re_ranker.py`, modify `reg_llm.py`

### 🔴 LLM Integration
- [ ] Add support for OpenAI API
- [ ] Add fallback to Ollama for local LLM
- [ ] Create `llm.py` with LLM backends
- [ ] Replace lightweight synthesis with actual LLM-based generation
- [ ] Files: `llm.py`, modify `reg_llm.py`

### 🔴 Web UI (FastAPI)
- [ ] Create `api.py` with FastAPI endpoints
- [ ] `/index` — POST endpoint to build index
- [ ] `/retrieve` — POST endpoint to retrieve contexts
- [ ] `/query` — POST endpoint for full RAG query
- [ ] Create `frontend/` directory with simple HTML/JS UI
- [ ] Files: `api.py`, `frontend/index.html`, `frontend/app.js`

### 🔴 Advanced Evaluation
- [ ] Create `evaluation.py` with ROUGE, BLEU, BERTScore
- [ ] Build evaluation dataset from `data/sample.txt`
- [ ] Compare different chunk sizes, models, retrieval methods
- [ ] Generate comparison report
- [ ] Files: `evaluation.py`, `data/eval_set.json`

### 🔴 Database Backend
- [ ] Add support for persistent vector DB (Weaviate, Milvus, or SQLite)
- [ ] Create abstraction layer in `vector_store.py`
- [ ] Allow saving/loading indexes without re-indexing
- [ ] Files: Modify `vector_store.py`, add `db_backends/`

---

## Phase 4: Infrastructure (Ongoing)
Testing, CI/CD, deployment.

### 🔵 Comprehensive Tests
- [ ] Create `tests/test_vector_store.py`
- [ ] Create `tests/test_reg_llm.py`
- [ ] Create integration tests in `tests/test_integration.py`
- [ ] Aim for 80%+ code coverage
- [ ] Files: `tests/test_*.py`

### 🔵 GitHub Actions CI/CD
- [ ] Create `.github/workflows/test.yml` — run tests on push
- [ ] Create `.github/workflows/lint.yml` — lint with `pylint`/`flake8`
- [ ] Create `.github/workflows/coverage.yml` — track coverage
- [ ] Files: `.github/workflows/*.yml`

### 🔵 Docker Support
- [ ] Create `Dockerfile` for containerized RAG service
- [ ] Create `docker-compose.yml` for development
- [ ] Files: `Dockerfile`, `docker-compose.yml`, `.dockerignore`

### 🔵 Performance Optimization
- [ ] Profile code with `cProfile`
- [ ] Optimize hot paths (embedding, retrieval)
- [ ] Add caching layer (LRU cache for embeddings)
- [ ] Create `PERFORMANCE.md` with benchmarks
- [ ] Files: Modify bottleneck modules, add `PERFORMANCE.md`

---

## How to Use This Roadmap

**Daily Contribution Strategy:**
1. Pick one task from **Phase 1: Quick Wins** (30-60 min)
2. Create a feature branch: `git checkout -b add/[feature-name]`
3. Implement the feature
4. Test locally
5. Commit: `git commit -m "Add [feature]"`
6. Push: `git push origin add/[feature-name]`
7. Open a PR (or merge to main if solo)

**Week-to-Week:**
- Week 1: Complete Phase 1 (easy wins, build confidence)
- Week 2-3: Move to Phase 2 (core features, meaningful additions)
- Week 4+: Advanced features & infrastructure

**Green Streak Tips:**
- Even small commits count (docs, tests, type hints)
- Batch related tasks into one PR per feature
- Combine multiple small tasks into one commit if they're cohesive

---

## Notes
- Each feature has suggested files to modify/create
- Features can be reordered based on interest
- Some features are independent; others build on previous work
- Document trade-offs in commit messages and PRs

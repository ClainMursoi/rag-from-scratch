"""Run the RAG demo with example prompts to showcase functionality."""

from ingest import load_documents
from reg_llm import build_index, retrieve, generate_answer
from pathlib import Path

DATA = Path(__file__).parent / "data" / "sample.txt"

# Example queries that demonstrate different aspects of the RAG system
EXAMPLE_QUERIES = [
    "What is retrieval-augmented generation?",
    "Tell me about chunking and embeddings",
    "How does the system retrieve documents?",
    "What is the quick brown fox?",
    "What does the demo use for vector search?",
]

def main():
    if not DATA.exists():
        print(f"Missing sample data at {DATA}.")
        return

    print("=" * 80)
    print("RAG FROM SCRATCH - DEMO WITH EXAMPLE PROMPTS")
    print("=" * 80)
    
    docs = load_documents(str(DATA))
    print(f"\n✓ Loaded {len(docs)} documents from {DATA}")
    
    print("Building index (chunking, embedding, indexing)...")
    build_index(docs, chunk_size=200, overlap=50)
    print("✓ Index built successfully!\n")
    
    print("=" * 80)
    
    for i, query in enumerate(EXAMPLE_QUERIES, 1):
        print(f"\n[Example {i}]")
        print(f"Query: {query}\n")
        
        try:
            contexts = retrieve(query, k=3)
            print(f"Retrieved {len(contexts)} chunks:")
            for j, chunk in enumerate(contexts, 1):
                preview = chunk[:80].replace('\n', ' ') + ("..." if len(chunk) > 80 else "")
                print(f"  [{j}] {preview}")
            
            print("\nSynthesized Answer:")
            answer = generate_answer(query, contexts)
            print(answer)
        except Exception as e:
            print(f"Error: {e}")
        
        print("\n" + "-" * 80)

if __name__ == "__main__":
    main()

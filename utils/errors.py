"""Custom exception classes for the RAG pipeline.

These exceptions provide clear, actionable error messages for common failure scenarios.
"""


class RAGError(Exception):
    """Base exception for all RAG-related errors."""
    pass


class DataLoadError(RAGError):
    """Raised when document loading fails."""
    pass


class ChunkingError(RAGError):
    """Raised when text chunking fails."""
    pass


class EmbeddingError(RAGError):
    """Raised when text embedding fails."""
    pass


class VectorStoreError(RAGError):
    """Raised when vector store operations fail."""
    pass


class IndexNotBuiltError(RAGError):
    """Raised when attempting to retrieve without building index first."""
    pass


class ConfigError(RAGError):
    """Raised when configuration is invalid."""
    pass

"""Configuration settings for the RAG-from-scratch pipeline.

This module defines default parameters used throughout the system.
Values can be overridden programmatically or via command-line arguments.
"""

from typing import Dict, Any


# ============================================================================
# CHUNKING CONFIGURATION
# ============================================================================

# Size of each text chunk in characters
CHUNK_SIZE: int = 200

# Overlap between consecutive chunks in characters (for context continuity)
CHUNK_OVERLAP: int = 50


# ============================================================================
# RETRIEVAL CONFIGURATION
# ============================================================================

# Number of chunks to retrieve for a query
RETRIEVAL_K: int = 3

# Embedding model to use (from sentence-transformers)
# Options: "all-MiniLM-L6-v2", "all-mpnet-base-v2", "distiluse-base-multilingual-v1.5"
EMBEDDING_MODEL: str = "all-MiniLM-L6-v2"


# ============================================================================
# DEMO CONFIGURATION
# ============================================================================

# Path to sample data (relative to project root)
DATA_PATH: str = "data/sample.txt"


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def get_defaults() -> Dict[str, Any]:
    """Return a dictionary of all default configuration values.
    
    Returns:
        Dictionary mapping config keys to their default values.
    """
    return {
        "chunk_size": CHUNK_SIZE,
        "chunk_overlap": CHUNK_OVERLAP,
        "retrieval_k": RETRIEVAL_K,
        "embedding_model": EMBEDDING_MODEL,
        "data_path": DATA_PATH,
    }


def update_from_dict(config_dict: Dict[str, Any]) -> None:
    """Update module-level config from a dictionary.
    
    Args:
        config_dict: Dictionary with keys matching config names.
                     Only provided keys are updated; others remain unchanged.
    
    Example:
        update_from_dict({"chunk_size": 300, "retrieval_k": 5})
    """
    global CHUNK_SIZE, CHUNK_OVERLAP, RETRIEVAL_K, EMBEDDING_MODEL, DATA_PATH
    
    if "chunk_size" in config_dict:
        CHUNK_SIZE = config_dict["chunk_size"]
    if "chunk_overlap" in config_dict:
        CHUNK_OVERLAP = config_dict["chunk_overlap"]
    if "retrieval_k" in config_dict:
        RETRIEVAL_K = config_dict["retrieval_k"]
    if "embedding_model" in config_dict:
        EMBEDDING_MODEL = config_dict["embedding_model"]
    if "data_path" in config_dict:
        DATA_PATH = config_dict["data_path"]

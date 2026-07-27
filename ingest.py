from typing import List


def load_documents(path: str) -> List[str]:
    """Load documents from a file, splitting on double newlines.
    
    Args:
        path: Path to the text file to load.
        
    Returns:
        A list of document strings.
    """
    with open(path, "r") as f:
        return f.read().split("\n\n")

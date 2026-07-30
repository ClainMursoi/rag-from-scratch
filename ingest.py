from typing import List

from utils.errors import DataLoadError


def load_documents(path: str) -> List[str]:
    """Load documents from a file, splitting on double newlines.
    
    Args:
        path: Path to the text file to load.
        
    Returns:
        A list of document strings.
        
    Raises:
        DataLoadError: If file cannot be read or is invalid.
    """
    try:
        with open(path, "r", encoding="utf-8") as f:
            content = f.read()
            if not content.strip():
                raise DataLoadError(f"File '{path}' is empty.")
            docs = content.split("\n\n")
            return [doc for doc in docs if doc.strip()]
    except FileNotFoundError as e:
        raise DataLoadError(f"Cannot find data file: {path}") from e
    except (IOError, UnicodeDecodeError) as e:
        raise DataLoadError(f"Error reading file '{path}': {e}") from e

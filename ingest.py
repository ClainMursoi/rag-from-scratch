def load_documents(path: str) -> list[str]:
    with open(path, "r") as f:
        return f.read().split("\n\n")

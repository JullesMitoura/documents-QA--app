from typing import List

def chunker(text: str, chunk_size: int = 1200, overlap: int = 200) -> List[str]:
    """
    Split a text into chunks of `chunk_size` characters, with `overlap` characters overlapping
    between consecutive chunks.

    Args:
        text (str): The input text to split.
        chunk_size (int): Maximum size of each chunk.
        overlap (int): Number of characters to overlap between consecutive chunks.

    Returns:
        List[str]: A list of text chunks.
    """
    if chunk_size <= overlap:
        raise ValueError("chunk_size must be greater than overlap")

    chunks = []
    start = 0
    text_length = len(text)

    while start < text_length:
        end = min(start + chunk_size, text_length)
        chunks.append(text[start:end])
        start += chunk_size - overlap  # move start forward but keep overlap

    return chunks
"""Cut contracts into chunks. Every chunk remembers where it came from.

Two strategies:
- chunk_fixed: windows of about the same size, with some overlap.
- chunk_by_paragraph: keep paragraphs together, merge small ones, cut very long ones.
"""

from contract_rag.schemas import Chunk, Contract


def fixed_windows(text: str, begin: int, finish: int, size: int, overlap: int) -> list:
    """Cut text[begin:finish] into windows of about `size` characters.

    Returns a list of (start, end) positions. Neighbouring windows overlap by about
    `overlap` characters, and window ends are moved back to a space so that words
    are not cut in half.
    """
    windows = []
    start = begin
    while start < finish:
        end = min(start + size, finish)
        if end < finish:
            space = text.rfind(" ", start + size // 2, end)
            if space != -1:
                end = space
        windows.append((start, end))
        if end >= finish:
            break

        # The next window starts `overlap` characters before this one ended,
        # moved forward to the beginning of a word
        next_start = end - overlap
        space = text.find(" ", next_start, end)
        if space != -1:
            next_start = space + 1
        start = max(next_start, start + 1)
    return windows


def find_paragraphs(text: str) -> list:
    """Return the (start, end) position of every paragraph.

    Paragraphs are separated by blank lines. Parts with only whitespace are skipped.
    """
    paragraphs = []
    position = 0
    for part in text.split("\n\n"):
        start = position
        end = position + len(part)
        if part.strip():
            paragraphs.append((start, end))
        position = end + 2
    return paragraphs


def make_chunks(contract: Contract, pieces: list) -> list[Chunk]:
    """Turn (start, end) positions into Chunk objects."""
    chunks = []
    for start, end in pieces:
        text = contract.text[start:end]
        if not text.strip():
            continue
        chunk_id = f"{contract.id}::{len(chunks)}"
        chunks.append(Chunk(id=chunk_id, contract_id=contract.id, start=start, end=end, text=text))
    return chunks


def chunk_fixed(contract: Contract, size: int = 1000, overlap: int = 200) -> list[Chunk]:
    """Cut the whole contract into overlapping windows of about `size` characters."""
    pieces = fixed_windows(contract.text, 0, len(contract.text), size, overlap)
    return make_chunks(contract, pieces)


def chunk_by_paragraph(contract: Contract, max_size: int = 1500, overlap: int = 200) -> list[Chunk]:
    """Keep paragraphs together.

    Small neighbouring paragraphs are merged until the chunk would be longer than
    `max_size`. A single paragraph longer than `max_size` is cut into windows.
    """
    pieces = []
    current = None

    for start, end in find_paragraphs(contract.text):
        if end - start > max_size:
            if current is not None:
                pieces.append(current)
                current = None
            pieces.extend(fixed_windows(contract.text, start, end, max_size, overlap))
        elif current is None:
            current = (start, end)
        elif end - current[0] <= max_size:
            current = (current[0], end)
        else:
            pieces.append(current)
            current = (start, end)

    if current is not None:
        pieces.append(current)
    return make_chunks(contract, pieces)

from pathlib import Path

import pytest

from contract_rag.chunking import chunk_by_paragraph, chunk_fixed, find_paragraphs
from contract_rag.data.download import CUAD_JSON_NAME
from contract_rag.data.loader import load_contracts
from contract_rag.schemas import Chunk, Contract, Span

RAW_DIR = Path("data/raw")
HAVE_DATA = (RAW_DIR / CUAD_JSON_NAME).exists()

# A text with three paragraphs, separated by blank lines.
PARAGRAPHS = (
    "First paragraph about payment terms. " * 8
    + "\n\n"
    + "Second paragraph about termination. " * 8
    + "\n\n\n"
    + "Third one is short."
)


def both_chunkers(contract: Contract) -> list:
    """Run both chunkers with small sizes, so a short test text gets several chunks."""
    return [
        chunk_fixed(contract, size=200, overlap=50),
        chunk_by_paragraph(contract, max_size=300, overlap=50),
    ]


def test_chunk_text_matches_its_position() -> None:
    contract = Contract(id="c", text=PARAGRAPHS)
    for chunks in both_chunkers(contract):
        assert len(chunks) > 1
        for chunk in chunks:
            assert contract.text[chunk.start : chunk.end] == chunk.text


def test_no_text_is_lost() -> None:
    contract = Contract(id="c", text=PARAGRAPHS)
    for chunks in both_chunkers(contract):
        covered = [False] * len(contract.text)
        for chunk in chunks:
            for i in range(chunk.start, chunk.end):
                covered[i] = True
        for i, character in enumerate(contract.text):
            if not character.isspace():
                assert covered[i], f"character {i} is not in any chunk"


def test_chunk_ids_are_unique_and_chunks_not_empty() -> None:
    contract = Contract(id="c", text=PARAGRAPHS)
    for chunks in both_chunkers(contract):
        ids = [chunk.id for chunk in chunks]
        assert len(set(ids)) == len(ids)
        assert all(chunk.text.strip() for chunk in chunks)


def test_fixed_chunks_respect_size_and_overlap() -> None:
    text = " ".join(f"word{i}" for i in range(300))
    chunks = chunk_fixed(Contract(id="c", text=text), size=200, overlap=50)
    for chunk in chunks:
        assert len(chunk.text) <= 200
    for i in range(len(chunks) - 1):
        first, second = chunks[i], chunks[i + 1]
        assert second.start < first.end  # neighbours share some text
        assert second.start > first.start


def test_find_paragraphs_positions() -> None:
    text = "aaa\n\nbbb\n\n\n\nccc"
    found = [text[start:end] for start, end in find_paragraphs(text)]
    assert found == ["aaa", "bbb", "ccc"]


def test_small_paragraphs_are_merged() -> None:
    contract = Contract(id="c", text="aaa\n\nbbb\n\nccc")
    chunks = chunk_by_paragraph(contract, max_size=100)
    assert len(chunks) == 1
    assert chunks[0].text == "aaa\n\nbbb\n\nccc"


def test_long_paragraph_is_cut() -> None:
    text = " ".join(f"word{i}" for i in range(500))  # one paragraph, no blank line
    chunks = chunk_by_paragraph(Contract(id="c", text=text), max_size=300, overlap=50)
    assert len(chunks) > 1
    assert all(len(chunk.text) <= 300 for chunk in chunks)


def test_chunk_contains_and_overlaps_span() -> None:
    chunk = Chunk(id="c::0", contract_id="c", start=100, end=200, text="x" * 100)
    assert chunk.contains(Span(start=120, end=150, text="x" * 30))
    assert not chunk.contains(Span(start=150, end=250, text="x" * 100))
    assert chunk.overlaps(Span(start=150, end=250, text="x" * 100))
    assert not chunk.overlaps(Span(start=200, end=250, text="x" * 50))


@pytest.mark.skipif(not HAVE_DATA, reason="CUAD not downloaded: run scripts/explore_cuad.py")
def test_real_contracts_chunk_correctly() -> None:
    contracts = load_contracts(RAW_DIR / CUAD_JSON_NAME)[:20]
    for contract in contracts:
        for chunks in [chunk_fixed(contract), chunk_by_paragraph(contract)]:
            assert len(chunks) > 0
            for chunk in chunks:
                assert contract.text[chunk.start : chunk.end] == chunk.text

import math
from pathlib import Path

import pytest

from contract_rag.index import ChunkIndex
from contract_rag.retrieval import KeywordIndex, Retriever, tokenize
from contract_rag.schemas import Chunk
from fakes import FakeEmbedder


def make_chunk(contract_id: str, number: int, start: int, text: str) -> Chunk:
    return Chunk(
        id=f"{contract_id}::{number}",
        contract_id=contract_id,
        start=start,
        end=start + len(text),
        text=text,
    )


def make_index(tmp_path: Path) -> ChunkIndex:
    index = ChunkIndex(tmp_path / "chroma", "test", FakeEmbedder())
    index.add_chunks(
        [
            make_chunk("c1", 2, 100, "unrelated words here"),  # added out of order on purpose
            make_chunk("c1", 0, 0, "governed by the laws of England"),
            make_chunk("c1", 1, 50, "payment within thirty days"),
            make_chunk("c2", 0, 0, "governed by the laws of France"),
        ]
    )
    return index


def test_tokenize_lowercases_and_cuts_words() -> None:
    assert tokenize("Competing, COMPETE; the 12 laws!") == ["compet", "compet", "the", "12", "laws"]


def test_bm25_score_matches_a_hand_calculation() -> None:
    chunks = [make_chunk("c", 0, 0, "alpha beta"), make_chunk("c", 1, 20, "beta gamma")]
    hits = KeywordIndex(chunks).search("alpha")
    assert len(hits) == 1
    assert hits[0].chunk.id == "c::0"
    # idf = ln(1 + (2 - 1 + 0.5) / (1 + 0.5)) = ln 2, and the chunk has exactly the average
    # length, so the rest of the formula is 1 * (k1 + 1) / (1 + k1) = 1.
    assert abs(hits[0].score - math.log(2)) < 1e-9


def test_bm25_prefers_the_chunk_with_the_rare_word() -> None:
    chunks = [
        make_chunk("c", 0, 0, "the party shall pay"),
        make_chunk("c", 1, 30, "exclusive license granted"),
        make_chunk("c", 2, 60, "the party may terminate"),
    ]
    hits = KeywordIndex(chunks).search("the exclusive license")
    assert hits[0].chunk.id == "c::1"


def test_bm25_leaves_out_chunks_without_any_query_word() -> None:
    chunks = [make_chunk("c", 0, 0, "alpha beta"), make_chunk("c", 1, 20, "gamma delta")]
    index = KeywordIndex(chunks)
    assert [hit.chunk.id for hit in index.search("alpha")] == ["c::0"]
    assert index.search("zebra") == []


def test_bm25_without_chunks_returns_nothing() -> None:
    assert KeywordIndex([]).search("anything") == []


def test_bm25_matches_different_forms_of_a_word() -> None:
    chunks = [make_chunk("c", 0, 0, "neither party shall compete"), make_chunk("c", 1, 40, "other")]
    hits = KeywordIndex(chunks).search("restricted from competing")
    assert hits[0].chunk.id == "c::0"


def test_get_chunks_returns_the_stored_chunks_in_order(tmp_path: Path) -> None:
    index = make_index(tmp_path)
    chunks = index.get_chunks("c1")
    assert [chunk.id for chunk in chunks] == ["c1::0", "c1::1", "c1::2"]
    assert chunks[0].text == "governed by the laws of England"
    assert (chunks[1].start, chunks[1].end) == (50, 50 + len("payment within thirty days"))


def test_every_retriever_mode_finds_the_obvious_chunk(tmp_path: Path) -> None:
    index = make_index(tmp_path)
    for mode in ["dense", "bm25", "hybrid"]:
        hits = Retriever(index, mode).search("laws of England", "c1", k=3)
        assert hits[0].chunk.id == "c1::0", mode
        assert all(hit.chunk.contract_id == "c1" for hit in hits), mode


def test_retriever_builds_each_keyword_index_only_once(tmp_path: Path) -> None:
    retriever = Retriever(make_index(tmp_path), "bm25")
    assert retriever.keyword_index("c1") is retriever.keyword_index("c1")


def test_retriever_rejects_an_unknown_mode(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        Retriever(make_index(tmp_path), "magic")

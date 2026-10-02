from pathlib import Path

from contract_rag.data.loader import sample_contracts
from contract_rag.index import ChunkIndex
from contract_rag.schemas import Chunk, Contract
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
    return ChunkIndex(tmp_path / "chroma", "test", FakeEmbedder())


def test_search_finds_the_matching_chunk(tmp_path: Path) -> None:
    index = make_index(tmp_path)
    law = make_chunk("c1", 0, 0, "governed by the laws of the State of New York")
    pay = make_chunk("c1", 1, 100, "invoices must be paid within thirty days")
    index.add_chunks([law, pay])

    hits = index.search("which laws govern New York", contract_id="c1", k=2)
    assert hits[0].chunk.id == "c1::0"
    assert hits[0].score > hits[1].score


def test_search_keeps_the_chunk_position(tmp_path: Path) -> None:
    index = make_index(tmp_path)
    index.add_chunks([make_chunk("c1", 0, 250, "exclusive license granted")])

    hit = index.search("exclusive license", contract_id="c1", k=1)[0]
    assert (hit.chunk.start, hit.chunk.end) == (250, 250 + len("exclusive license granted"))


def test_search_stays_inside_one_contract(tmp_path: Path) -> None:
    index = make_index(tmp_path)
    text = "governed by the laws of England"
    index.add_chunks([make_chunk("c1", 0, 0, text), make_chunk("c2", 0, 0, text)])

    hits = index.search("laws of England", contract_id="c2", k=5)
    assert len(hits) == 1
    assert hits[0].chunk.contract_id == "c2"


def test_adding_twice_does_not_duplicate(tmp_path: Path) -> None:
    index = make_index(tmp_path)
    chunks = [make_chunk("c1", 0, 0, "some text"), make_chunk("c1", 1, 50, "more text")]
    index.add_chunks(chunks)
    index.add_chunks(chunks)
    assert index.count() == 2


def test_chunk_count_per_contract(tmp_path: Path) -> None:
    index = make_index(tmp_path)
    index.add_chunks([make_chunk("c1", 0, 0, "some text"), make_chunk("c1", 1, 50, "more text")])
    assert index.chunk_count("c1") == 2
    assert index.chunk_count("c2") == 0


def test_sample_contracts_is_repeatable() -> None:
    contracts = [Contract(id=f"c{i}", text="x") for i in range(50)]
    first = sample_contracts(contracts, 10)
    second = sample_contracts(contracts, 10)
    assert [c.id for c in first] == [c.id for c in second]
    assert len(first) == 10
    assert len(sample_contracts(contracts, 100)) == 50

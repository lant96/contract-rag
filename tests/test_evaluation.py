from pathlib import Path

from contract_rag.clauses import ClauseSpec
from contract_rag.evaluation import average, evaluate_retrieval, score_query
from contract_rag.index import ChunkIndex
from contract_rag.schemas import Chunk, GoldLabel, Hit, Span
from fakes import FakeEmbedder


def make_hit(start: int, end: int, score: float = 0.9) -> Hit:
    chunk = Chunk(id=f"c::{start}", contract_id="c", start=start, end=end, text="x" * (end - start))
    return Hit(chunk=chunk, score=score)


def make_label(*ranges: tuple) -> GoldLabel:
    spans = [Span(start=s, end=e, text="x" * (e - s)) for s, e in ranges]
    return GoldLabel(contract_id="c", clause="governing_law", spans=spans)


def test_hit_at_k_and_mrr() -> None:
    label = make_label((100, 200))
    hits = [make_hit(0, 50), make_hit(90, 300), make_hit(300, 400)]
    scores = score_query(hits, label)
    assert scores["hit@1"] == 0.0
    assert scores["hit@3"] == 1.0
    assert scores["mrr"] == 0.5


def test_a_chunk_with_too_little_overlap_does_not_count() -> None:
    label = make_label((100, 200))
    hits = [make_hit(0, 140)]  # covers only 40% of the answer
    assert score_query(hits, label, min_overlap=0.5)["hit@5"] == 0.0
    assert score_query(hits, label, min_overlap=0.3)["hit@5"] == 1.0


def test_recall_counts_found_spans() -> None:
    label = make_label((100, 200), (1000, 1100))  # two gold spans
    hits = [make_hit(90, 210)]  # finds only the first one
    scores = score_query(hits, label)
    assert scores["hit@1"] == 1.0
    assert scores["recall@5"] == 0.5


def test_nothing_found_gives_zeros() -> None:
    scores = score_query([make_hit(0, 50)], make_label((100, 200)))
    assert scores["hit@5"] == 0.0
    assert scores["mrr"] == 0.0


def test_overlap_fraction() -> None:
    chunk = Chunk(id="c::0", contract_id="c", start=100, end=200, text="x" * 100)
    assert chunk.overlap_fraction(Span(start=100, end=200, text="x" * 100)) == 1.0
    assert chunk.overlap_fraction(Span(start=150, end=250, text="x" * 100)) == 0.5
    assert chunk.overlap_fraction(Span(start=300, end=400, text="x" * 100)) == 0.0


def test_evaluate_retrieval_end_to_end(tmp_path: Path) -> None:
    index = ChunkIndex(tmp_path / "chroma", "test", FakeEmbedder())
    answer = "governed by the laws of England"
    index.add_chunks(
        [
            Chunk(id="c1::0", contract_id="c1", start=0, end=31, text=answer),
            Chunk(
                id="c1::1", contract_id="c1", start=40, end=80, text="payment within thirty days"
            ),
        ]
    )
    clauses = [ClauseSpec(key="governing_law", cuad_category="x", questions=["laws of England"])]
    labels = [
        GoldLabel(
            contract_id="c1",
            clause="governing_law",
            spans=[Span(start=0, end=31, text=answer)],
        ),
        GoldLabel(contract_id="c2", clause="governing_law", spans=[]),  # absent: skipped
    ]

    rows = evaluate_retrieval(index, ["c1", "c2"], labels, clauses)
    assert len(rows) == 1
    assert rows[0]["hit@1"] == 1.0
    assert average(rows)["queries"] == 1

from pathlib import Path

from contract_rag.clauses import ClauseSpec
from contract_rag.evaluation import average, evaluate_retrieval, fuse_hits, score_query
from contract_rag.index import ChunkIndex
from contract_rag.schemas import Chunk, GoldLabel, Hit, Span
from fakes import FakeEmbedder

ANSWER = "governed by the laws of England"


def make_hit(start: int, end: int, score: float = 0.9) -> Hit:
    chunk = Chunk(id=f"c::{start}", contract_id="c", start=start, end=end, text="x" * (end - start))
    return Hit(chunk=chunk, score=score)


def make_label(*ranges: tuple) -> GoldLabel:
    spans = [Span(start=s, end=e, text="x" * (e - s)) for s, e in ranges]
    return GoldLabel(contract_id="c", clause="governing_law", spans=spans)


def make_small_index(tmp_path: Path) -> ChunkIndex:
    """An index with one contract that has an answer chunk and an unrelated chunk."""
    index = ChunkIndex(tmp_path / "chroma", "test", FakeEmbedder())
    index.add_chunks(
        [
            Chunk(id="c1::0", contract_id="c1", start=0, end=31, text=ANSWER),
            Chunk(
                id="c1::1", contract_id="c1", start=40, end=80, text="payment within thirty days"
            ),
        ]
    )
    return index


def make_labels() -> list[GoldLabel]:
    return [
        GoldLabel(
            contract_id="c1",
            clause="governing_law",
            spans=[Span(start=0, end=31, text=ANSWER)],
        ),
        GoldLabel(contract_id="c2", clause="governing_law", spans=[]),  # absent: skipped
    ]


def test_hit_at_k_and_mrr() -> None:
    label = make_label((100, 200))
    hits = [make_hit(0, 50), make_hit(90, 300), make_hit(300, 400)]  # the answer is in the 2nd
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
    label = make_label((100, 200), (1000, 1100))
    hits = [make_hit(90, 210)]
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


def test_fuse_hits_rewards_chunks_found_by_several_queries() -> None:
    first = [make_hit(0, 10), make_hit(20, 30), make_hit(40, 50)]
    second = [make_hit(40, 50), make_hit(60, 70)]
    fused = fuse_hits([first, second])
    ids = [hit.chunk.id for hit in fused]
    assert ids[0] == "c::40"
    assert sorted(ids) == ["c::0", "c::20", "c::40", "c::60"]


def test_fuse_hits_with_one_list_keeps_its_order() -> None:
    hits = [make_hit(0, 10), make_hit(20, 30)]
    assert [hit.chunk.id for hit in fuse_hits([hits])] == ["c::0", "c::20"]


def test_evaluate_retrieval_end_to_end(tmp_path: Path) -> None:
    index = make_small_index(tmp_path)
    clauses = [ClauseSpec(key="governing_law", cuad_category="x", questions=["laws of England"])]

    rows = evaluate_retrieval(index, ["c1", "c2"], make_labels(), clauses)
    assert len(rows) == 1
    assert rows[0]["hit@1"] == 1.0
    assert average(rows)["queries"] == 1


def test_fused_evaluation_gives_one_row_per_clause(tmp_path: Path) -> None:
    index = make_small_index(tmp_path)
    questions = ["laws of England", "governed by England"]
    clauses = [ClauseSpec(key="governing_law", cuad_category="x", questions=questions)]

    separate = evaluate_retrieval(index, ["c1"], make_labels(), clauses)
    fused = evaluate_retrieval(index, ["c1"], make_labels(), clauses, fuse=True)
    assert len(separate) == 2
    assert len(fused) == 1
    assert fused[0]["hit@1"] == 1.0

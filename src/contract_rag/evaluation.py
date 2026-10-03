"""Measure how well retrieval finds the text that the experts marked as the answer."""

from contract_rag.clauses import ClauseSpec
from contract_rag.index import ChunkIndex
from contract_rag.schemas import GoldLabel, Hit, Span

K_VALUES = [1, 3, 5]
METRICS = ["hit@1", "hit@3", "hit@5", "recall@5", "mrr"]
FUSION_DEPTH = 10  # how many chunks are taken from each query before combining them
RRF_K = 60  # the usual constant of Reciprocal Rank Fusion


def span_is_found(hits: list[Hit], span: Span, min_overlap: float) -> bool:
    """True if one of the hits contains at least `min_overlap` of the span's characters."""
    return any(hit.chunk.overlap_fraction(span) >= min_overlap for hit in hits)


def score_query(hits: list[Hit], label: GoldLabel, min_overlap: float = 0.5) -> dict:
    """Score the hits of one query against the gold spans of one (contract, clause).

    hit@k:    1 if at least one gold span is found in the top k chunks, else 0
    recall@k: share of the gold spans that are found in the top k chunks
    mrr:      1 / rank of the first chunk that finds a gold span (0 if none does)
    """
    scores = {}
    for k in K_VALUES:
        found = [span_is_found(hits[:k], span, min_overlap) for span in label.spans]
        scores[f"hit@{k}"] = 1.0 if any(found) else 0.0
        scores[f"recall@{k}"] = sum(found) / len(found)

    scores["mrr"] = 0.0
    for rank, hit in enumerate(hits, start=1):
        if any(span_is_found([hit], span, min_overlap) for span in label.spans):
            scores["mrr"] = 1 / rank
            break
    return scores


def fuse_hits(hit_lists: list[list[Hit]]) -> list[Hit]:
    """Combine several rankings of the same contract into one (Reciprocal Rank Fusion).

    Every chunk gets 1 / (RRF_K + rank) from each list it appears in, and the scores are
    added up. A chunk that several queries rank highly ends up on top.
    """
    scores = {}  # chunk id -> fused score
    chunks = {}  # chunk id -> the chunk itself
    for hits in hit_lists:
        for rank, hit in enumerate(hits, start=1):
            chunk_id = hit.chunk.id
            scores[chunk_id] = scores.get(chunk_id, 0.0) + 1 / (RRF_K + rank)
            chunks[chunk_id] = hit.chunk

    ordered_ids = sorted(scores, key=lambda chunk_id: scores[chunk_id], reverse=True)
    return [Hit(chunk=chunks[chunk_id], score=scores[chunk_id]) for chunk_id in ordered_ids]


def make_row(label: GoldLabel, question: str, hits: list[Hit], min_overlap: float) -> dict:
    """Score one query and add the information we want to keep about it."""
    row = score_query(hits, label, min_overlap)
    row["contract_id"] = label.contract_id
    row["clause"] = label.clause
    row["question"] = question
    return row


def evaluate_retrieval(
    index: ChunkIndex,
    contract_ids: list[str],
    labels: list[GoldLabel],
    clauses: list[ClauseSpec],
    min_overlap: float = 0.5,
    fuse: bool = False,
) -> list[dict]:
    """Ask the questions of every clause on every contract where that clause exists.

    By default every question is scored on its own (one row per question). With
    fuse=True all questions of a clause are searched together and their rankings are
    combined with fuse_hits (one row per contract and clause).

    Contracts where the clause is absent are skipped, because there is nothing to find.
    """
    wanted = set(contract_ids)
    clause_for_key = {clause.key: clause for clause in clauses}

    rows = []
    for label in labels:
        if label.contract_id not in wanted or not label.is_present():
            continue
        questions = clause_for_key[label.clause].questions

        if fuse:
            hit_lists = [index.search(q, label.contract_id, k=FUSION_DEPTH) for q in questions]
            hits = fuse_hits(hit_lists)
            row = make_row(label, f"fused: {len(questions)} questions", hits, min_overlap)
            rows.append(row)
        else:
            for question in questions:
                hits = index.search(question, label.contract_id, k=max(K_VALUES))
                rows.append(make_row(label, question, hits, min_overlap))
    return rows


def average(rows: list[dict]) -> dict:
    """Mean of every metric over the rows, plus the number of queries."""
    result: dict[str, float] = {"queries": len(rows)}
    for name in METRICS:
        result[name] = sum(row[name] for row in rows) / len(rows)
    return result


def average_per_clause(rows: list[dict]) -> dict:
    """Same as average(), but separately for each clause."""
    per_clause = {}
    for key in sorted({row["clause"] for row in rows}):
        clause_rows = [row for row in rows if row["clause"] == key]
        per_clause[key] = average(clause_rows)
    return per_clause

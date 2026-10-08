"""Score the answers of the LLM step against the expert labels. No LLM is used as a judge."""

from collections.abc import Callable

from contract_rag.answering import ClauseResult
from contract_rag.schemas import Chunk, GoldLabel, Span
from contract_rag.stats import bootstrap_mean

MIN_OVERLAP = 0.5


def spans_agree(quote: Span, gold: Span, min_overlap: float = MIN_OVERLAP) -> bool:
    """True if two spans overlap by at least `min_overlap` of the shorter one."""
    overlap = min(quote.end, gold.end) - max(quote.start, gold.start)
    if overlap <= 0:
        return False
    shorter = min(quote.end - quote.start, gold.end - gold.start)
    return overlap >= min_overlap * shorter


def evidence_hits(result: ClauseResult, gold: GoldLabel) -> int:
    """How many of the verified quotes agree with an expert span."""
    hits = 0
    for item in result.evidence:
        if any(spans_agree(item.span, span) for span in gold.spans):
            hits += 1
    return hits


def passages_contain_answer(passages: list[Chunk], gold: GoldLabel) -> bool:
    """True if a shown passage holds at least half of an expert span.

    This is the best the LLM step could do: if the answer was never shown, it cannot find it.
    """
    return any(
        chunk.overlap_fraction(span) >= MIN_OVERLAP for chunk in passages for span in gold.spans
    )


def score_result(result: ClauseResult, gold: GoldLabel, passages: list[Chunk]) -> dict:
    """One row of numbers for one (contract, clause) pair."""
    present = gold.is_present()
    row = {
        "contract_id": result.contract_id,
        "clause": result.clause,
        "gold_present": present,
        "found": result.found,
        "correct": result.found == present,
        "status": result.status,
        "n_quotes": len(result.evidence),
        "prompt_tokens": result.prompt_tokens,
        "completion_tokens": result.completion_tokens,
        "retrieval_hit": passages_contain_answer(passages, gold) if present else None,
        "evidence_hit": None,
        "evidence_precision": None,
    }
    if present and result.found:
        hits = evidence_hits(result, gold)
        row["evidence_hit"] = hits > 0
        row["evidence_precision"] = hits / len(result.evidence) if result.evidence else 0.0
    return row


def rate(part: float, whole: float) -> float | None:
    """part / whole, or None when there is nothing to divide by."""
    return part / whole if whole else None


def summarize(rows: list[dict]) -> dict:
    """The headline numbers for a list of rows."""
    present = [row for row in rows if row["gold_present"]]
    absent = [row for row in rows if not row["gold_present"]]
    found_present = [row for row in present if row["found"]]

    tp = len(found_present)
    fn = len(present) - tp
    fp = sum(1 for row in absent if row["found"])
    tn = len(absent) - fp

    precision = rate(tp, tp + fp)
    recall = rate(tp, tp + fn)
    f1 = None
    if precision is not None and recall is not None and precision + recall > 0:
        f1 = 2 * precision * recall / (precision + recall)

    quote_scores = [row["evidence_precision"] for row in found_present]
    return {
        "pairs": len(rows),
        "present": len(present),
        "absent": len(absent),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "accuracy": rate(tp + tn, len(rows)),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "false_found_rate": rate(fp, fp + tn),
        "retrieval_ceiling": rate(sum(1 for row in present if row["retrieval_hit"]), len(present)),
        "evidence_hit_rate": rate(sum(1 for row in found_present if row["evidence_hit"]), tp),
        "evidence_precision": rate(sum(quote_scores), len(quote_scores)),
        "failed_validation_rate": rate(
            sum(r["status"] == "failed_validation" for r in rows), len(rows)
        ),
        "retried_rate": rate(sum(r["status"] == "retried" for r in rows), len(rows)),
        "mean_prompt_tokens": rate(sum(r["prompt_tokens"] for r in rows), len(rows)),
        "mean_completion_tokens": rate(sum(r["completion_tokens"] for r in rows), len(rows)),
    }


def metric_with_interval(
    rows: list[dict], select: Callable[[dict], bool], value: Callable[[dict], float]
) -> tuple[int, float, float, float] | None:
    """Mean of `value` over the selected rows, with a 95% interval from resampling contracts.

    Returns (number of rows, mean, low, high), or None when no row is selected.
    """
    by_contract: dict[str, list[float]] = {}
    count = 0
    for row in rows:
        if select(row):
            by_contract.setdefault(row["contract_id"], []).append(value(row))
            count += 1
    if count == 0:
        return None
    mean, low, high = bootstrap_mean(list(by_contract.values()))
    return count, mean, low, high

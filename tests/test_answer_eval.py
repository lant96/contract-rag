from contract_rag.answer_eval import (
    evidence_hits,
    metric_with_interval,
    passages_contain_answer,
    score_result,
    spans_agree,
    summarize,
)
from contract_rag.answering import ClauseResult, VerifiedEvidence
from contract_rag.schemas import Chunk, GoldLabel, Span


def span(start: int, end: int) -> Span:
    return Span(start=start, end=end, text="x" * (end - start))


def chunk(start: int, end: int) -> Chunk:
    return Chunk(id=f"c::{start}", contract_id="c", start=start, end=end, text="x" * (end - start))


def gold(*ranges: tuple) -> GoldLabel:
    return GoldLabel(
        contract_id="c", clause="governing_law", spans=[span(start, end) for start, end in ranges]
    )


def make_result(found: bool, *ranges: tuple, status: str = "ok") -> ClauseResult:
    evidence = [
        VerifiedEvidence(chunk_id="c::0", quote="some quote", span=span(start, end))
        for start, end in ranges
    ]
    return ClauseResult(
        contract_id="c",
        clause="governing_law",
        found=found,
        summary="s" if found else "",
        evidence=evidence,
        status=status,
        problems=[],
        first_problems=[],
        passage_ids=["c::0"],
        prompt_tokens=1500,
        completion_tokens=100,
        api_calls=1,
    )


# when do two spans agree


def test_identical_spans_agree_and_touching_spans_do_not() -> None:
    assert spans_agree(span(100, 200), span(100, 200))
    assert not spans_agree(span(100, 200), span(200, 300))


def test_a_short_quote_inside_a_long_expert_span_agrees() -> None:
    assert spans_agree(span(500, 600), span(100, 1100))  # the whole quote lies inside


def test_agreement_needs_half_of_the_shorter_span() -> None:
    assert spans_agree(span(100, 200), span(140, 300))  # 60 of 100 characters overlap
    assert not spans_agree(span(100, 200), span(170, 300))  # only 30 of 100 overlap


def test_evidence_hits_counts_the_quotes_that_agree() -> None:
    result = make_result(True, (100, 200), (210, 260), (900, 950))
    assert evidence_hits(result, gold((100, 200))) == 1


def test_the_ceiling_checks_whether_the_answer_was_ever_shown() -> None:
    assert passages_contain_answer([chunk(0, 150)], gold((100, 200))) is True  # 50% inside
    assert passages_contain_answer([chunk(0, 130)], gold((100, 200))) is False  # 30% inside


# scoring one pair


def test_a_correct_finding_scores_evidence_hit_and_precision() -> None:
    result = make_result(True, (100, 200), (210, 260), (900, 950))  # one good, two padding
    row = score_result(result, gold((100, 200)), [chunk(0, 300)])
    assert row["correct"] is True
    assert row["evidence_hit"] is True
    assert abs(row["evidence_precision"] - 1 / 3) < 1e-9
    assert row["retrieval_hit"] is True


def test_a_missed_clause_is_incorrect_and_has_no_evidence_scores() -> None:
    row = score_result(make_result(False), gold((100, 200)), [chunk(0, 300)])
    assert row["correct"] is False
    assert row["evidence_hit"] is None
    assert row["retrieval_hit"] is True  # it was shown but not recognised: an LLM miss


def test_claiming_an_absent_clause_is_incorrect() -> None:
    row = score_result(make_result(True, (100, 200)), gold(), [chunk(0, 300)])
    assert row["correct"] is False
    assert row["retrieval_hit"] is None


def test_correctly_saying_not_found_for_an_absent_clause() -> None:
    row = score_result(make_result(False), gold(), [chunk(0, 300)])
    assert row["correct"] is True


# summaries


def row(present: bool, found: bool, status: str = "ok", contract: str = "c") -> dict:
    return {
        "contract_id": contract,
        "gold_present": present,
        "found": found,
        "correct": present == found,
        "status": status,
        "prompt_tokens": 1000,
        "completion_tokens": 100,
        "retrieval_hit": True if present else None,
        "evidence_hit": True if present and found else None,
        "evidence_precision": 0.5 if present and found else None,
    }


def test_summarize_computes_the_confusion_counts_and_rates() -> None:
    rows = (
        [row(True, True)] * 3  # found when present
        + [row(True, False)]  # missed
        + [row(False, True)]  # claimed an absent clause
        + [row(False, False)] * 5  # correctly not found
    )
    summary = summarize(rows)
    assert (summary["tp"], summary["fn"], summary["fp"], summary["tn"]) == (3, 1, 1, 5)
    assert summary["accuracy"] == 0.8
    assert summary["precision"] == 0.75
    assert summary["recall"] == 0.75
    assert summary["f1"] == 0.75
    assert abs(summary["false_found_rate"] - 1 / 6) < 1e-9
    assert summary["retrieval_ceiling"] == 1.0
    assert summary["evidence_precision"] == 0.5
    assert summary["mean_prompt_tokens"] == 1000


def test_summarize_counts_validation_outcomes() -> None:
    rows = [
        row(False, False, "ok"),
        row(False, False, "retried"),
        row(False, False, "failed_validation"),
    ]
    summary = summarize(rows)
    assert abs(summary["retried_rate"] - 1 / 3) < 1e-9
    assert abs(summary["failed_validation_rate"] - 1 / 3) < 1e-9


def test_summarize_leaves_undefined_rates_empty() -> None:
    summary = summarize([row(False, False)])  # no present clause at all
    assert summary["recall"] is None
    assert summary["f1"] is None
    assert summary["evidence_hit_rate"] is None


def test_metric_with_interval_resamples_contracts() -> None:
    rows = [
        row(True, True, contract="a"),
        row(True, False, contract="b"),
        row(True, True, contract="c"),
    ]
    result = metric_with_interval(rows, lambda r: r["gold_present"], lambda r: float(r["found"]))
    assert result is not None
    count, mean, low, high = result
    assert count == 3
    assert abs(mean - 2 / 3) < 1e-9
    assert low <= mean <= high


def test_metric_with_interval_returns_none_without_rows() -> None:
    assert (
        metric_with_interval([row(False, False)], lambda r: r["gold_present"], lambda r: 1.0)
        is None
    )

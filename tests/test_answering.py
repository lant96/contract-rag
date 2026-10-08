import json

from contract_rag.answering import (
    ANSWER_SCHEMA,
    ClauseAnswer,
    Evidence,
    answer_clause,
    build_messages,
    check_answer,
    locate_quote,
    parse_answer,
    retrieve_passages,
)
from contract_rag.clauses import CLAUSES
from contract_rag.llm import LLMReply
from contract_rag.schemas import Chunk, Hit

CLAUSE = [clause for clause in CLAUSES if clause.key == "governing_law"][0]
LAW_TEXT = (
    "13. Governing Law. This Agreement shall be governed by the laws of the State of New York."
)
QUOTE = "This Agreement shall be governed by the laws of the State of New York."


def make_chunk(number: int, start: int, text: str) -> Chunk:
    return Chunk(id=f"c::{number}", contract_id="c", start=start, end=start + len(text), text=text)


class StubRetriever:
    """Always returns the same chunks, best first."""

    def __init__(self, chunks: list[Chunk]):
        self.hits = [Hit(chunk=chunk, score=1.0 - 0.1 * i) for i, chunk in enumerate(chunks)]

    def search(self, query: str, contract_id: str, k: int = 5) -> list[Hit]:
        return self.hits[:k]


class FakeLLM:
    """Plays back prepared answers (JSON text), one per call, and records the messages."""

    def __init__(self, *answers: str):
        self.answers = list(answers)
        self.calls = []

    def complete_json(self, messages: list, schema: dict, schema_name: str) -> LLMReply:
        self.calls.append(messages)
        return LLMReply(text=self.answers.pop(0), prompt_tokens=100, completion_tokens=20)


def answer_json(found: bool = True, summary: str = "New York law applies.", quotes=None) -> str:
    evidence = [{"passage": "P1", "quote": quote} for quote in (quotes or [])]
    return json.dumps({"found": found, "summary": summary, "evidence": evidence})


def make_retriever() -> StubRetriever:
    return StubRetriever([make_chunk(0, 1000, LAW_TEXT), make_chunk(1, 2000, "Payment is due.")])


# locating quotes


def test_locate_quote_ignores_spacing_and_capital_letters() -> None:
    chunk = make_chunk(0, 100, LAW_TEXT)
    span = locate_quote("THIS AGREEMENT  shall be governed by the laws", chunk)
    assert span is not None
    assert span.text == "This Agreement shall be governed by the laws"


def test_locate_quote_copes_with_broken_spacing_in_the_contract() -> None:
    chunk = make_chunk(0, 0, "The parties are bound by this Agreementare subject to the rules.")
    span = locate_quote("this Agreement are subject to the rules", chunk)
    assert span is not None
    assert span.text == "this Agreementare subject to the rules"  # the original spacing is kept


def test_locate_quote_gives_the_position_in_the_whole_contract() -> None:
    chunk = make_chunk(0, 1000, LAW_TEXT)
    span = locate_quote("shall be governed by the laws", chunk)
    assert span is not None
    assert span.start == 1000 + LAW_TEXT.index("shall be governed")


def test_locate_quote_returns_none_for_text_that_is_not_there() -> None:
    chunk = make_chunk(0, 0, LAW_TEXT)
    assert locate_quote("governed by the laws of France", chunk) is None
    assert locate_quote("   ", chunk) is None


# checking answers


def passages() -> dict:
    return {"P1": make_chunk(0, 1000, LAW_TEXT)}


def test_a_correct_answer_has_no_problems() -> None:
    answer = ClauseAnswer(
        found=True, summary="NY law.", evidence=[Evidence(passage="P1", quote=QUOTE)]
    )
    assert check_answer(answer, passages()) == []


def test_not_found_without_evidence_is_fine() -> None:
    assert check_answer(ClauseAnswer(found=False, summary="", evidence=[]), passages()) == []


def test_each_kind_of_bad_answer_is_reported() -> None:
    def problems_of(**changes) -> list[str]:
        values = {
            "found": True,
            "summary": "ok",
            "evidence": [Evidence(passage="P1", quote=QUOTE)],
        } | changes
        return check_answer(ClauseAnswer(**values), passages())

    assert "must be empty" in problems_of(found=False)[0]
    assert "summary must not be empty" in problems_of(summary="  ")[0]
    assert "at least one evidence" in problems_of(evidence=[])[0]
    assert "does not exist" in problems_of(evidence=[Evidence(passage="P9", quote=QUOTE)])[0]
    assert "too short" in problems_of(evidence=[Evidence(passage="P1", quote="New York")])[0]
    invented = Evidence(passage="P1", quote="This Agreement is governed by Greek law entirely.")
    assert "word for word" in problems_of(evidence=[invented])[0]


def test_a_quote_from_the_wrong_passage_is_rejected() -> None:
    two = {"P1": make_chunk(0, 0, "Payment is due in thirty days."), "P2": passages()["P1"]}
    answer = ClauseAnswer(found=True, summary="x", evidence=[Evidence(passage="P1", quote=QUOTE)])
    assert "word for word" in check_answer(answer, two)[0]


def test_parse_answer_rejects_the_wrong_format() -> None:
    answer, problems = parse_answer('{"found": "maybe"}')
    assert answer is None
    assert problems
    assert parse_answer("not json at all")[0] is None
    good, no_problems = parse_answer(answer_json(found=False, summary=""))
    assert good is not None
    assert no_problems == []


# prompt and schema


def test_the_prompt_contains_the_definition_and_labelled_passages() -> None:
    messages = build_messages(CLAUSE, passages())
    assert messages[0]["role"] == "system"
    assert "copied exactly" in messages[0]["content"]
    user_text = messages[1]["content"]
    assert "Governing Law" in user_text
    assert f"[P1] {LAW_TEXT}" in user_text
    assert "law" in user_text.lower()


def test_the_json_schema_matches_the_pydantic_models_and_is_strict() -> None:
    assert set(ANSWER_SCHEMA["properties"]) == set(ClauseAnswer.model_fields)
    item = ANSWER_SCHEMA["properties"]["evidence"]["items"]
    assert set(item["properties"]) == set(Evidence.model_fields)
    assert set(ANSWER_SCHEMA["required"]) == set(ANSWER_SCHEMA["properties"])
    assert ANSWER_SCHEMA["additionalProperties"] is False
    assert set(item["required"]) == set(item["properties"])
    assert item["additionalProperties"] is False


def test_retrieve_passages_labels_and_limits_the_chunks() -> None:
    chunks = [make_chunk(number, number * 100, f"text {number}") for number in range(7)]
    found = retrieve_passages("c", CLAUSE, StubRetriever(chunks), max_passages=3)
    assert list(found) == ["P1", "P2", "P3"]
    assert found["P1"].id == "c::0"


# the whole pipeline


def test_a_good_answer_is_accepted_and_its_quote_located() -> None:
    llm = FakeLLM(answer_json(quotes=[QUOTE]))
    result = answer_clause("c", CLAUSE, make_retriever(), llm)

    assert result.status == "ok"
    assert result.found
    assert not result.needs_review()
    assert result.evidence[0].chunk_id == "c::0"
    assert result.evidence[0].span.text == QUOTE
    assert result.evidence[0].span.start == 1000 + LAW_TEXT.index("This Agreement")
    assert (result.prompt_tokens, result.completion_tokens) == (100, 20)
    assert result.api_calls == 1
    assert result.first_problems == []
    assert result.passage_ids == ["c::0", "c::1"]


def test_not_found_is_a_valid_answer() -> None:
    llm = FakeLLM(answer_json(found=False, summary=""))
    result = answer_clause("c", CLAUSE, make_retriever(), llm)
    assert result.status == "ok"
    assert not result.found
    assert result.evidence == []


def test_a_wrong_quote_triggers_one_retry_with_feedback() -> None:
    invented = "This Agreement is governed by the laws of Greece."
    llm = FakeLLM(answer_json(quotes=[invented]), answer_json(quotes=[QUOTE]))
    result = answer_clause("c", CLAUSE, make_retriever(), llm)

    assert result.status == "retried"
    assert result.found
    assert len(llm.calls) == 2
    feedback = llm.calls[1][-1]["content"]
    assert "word for word" in feedback
    assert (result.prompt_tokens, result.completion_tokens) == (200, 40)
    assert result.api_calls == 2
    assert "word for word" in result.first_problems[0]
    assert result.problems == []


def test_a_broken_format_is_also_retried() -> None:
    llm = FakeLLM("this is not json", answer_json(quotes=[QUOTE]))
    result = answer_clause("c", CLAUSE, make_retriever(), llm)
    assert result.status == "retried"
    assert result.found


def test_two_bad_answers_end_in_a_flagged_not_found_result() -> None:
    invented = "This Agreement is governed by the laws of Greece."
    llm = FakeLLM(answer_json(quotes=[invented]), answer_json(quotes=[invented]))
    result = answer_clause("c", CLAUSE, make_retriever(), llm)

    assert result.status == "failed_validation"
    assert result.needs_review()
    assert not result.found
    assert result.evidence == []
    assert result.problems
    assert len(llm.calls) == 2

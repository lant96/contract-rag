"""Answer "does this contract contain clause X?" with quotes that are checked against the text.

The steps for one clause of one contract:
1. retrieve the best passages (hybrid search, several phrasings, fused)
2. ask the LLM for a structured answer: found or not, a summary, and exact quotes
3. check the answer ourselves: every quote must appear word for word in a passage
4. if the check fails, tell the LLM what was wrong and ask once more
"""

from pydantic import BaseModel, ValidationError

from contract_rag.clauses import ClauseSpec
from contract_rag.descriptions import CLAUSE_DEFINITIONS
from contract_rag.retrieval import FUSION_DEPTH, fuse_hits
from contract_rag.schemas import Chunk, Span

MIN_QUOTE_LENGTH = 20
MAX_ATTEMPTS = 2

SYSTEM_PROMPT = """You are a careful contract analyst. You get numbered passages from ONE \
contract, plus the name and definition of a clause type. Decide whether the passages contain \
that clause.

Rules:
1. Use only the passages. Do not guess and do not use outside knowledge.
2. If a passage contains the clause: set found to true, write a one or two sentence \
plain-language summary, and give one to three evidence quotes. Each quote must be copied \
exactly from one passage (same words, nothing added, nothing changed) and must say who is \
bound or allowed to do what. Give the label of the passage each quote comes from.
3. If no passage contains the clause: set found to false, leave the summary empty and give no \
evidence.
4. You do not give legal advice."""


class Evidence(BaseModel):
    passage: str
    quote: str


class ClauseAnswer(BaseModel):
    """What we ask the LLM to produce."""

    found: bool
    summary: str
    evidence: list[Evidence]


# The same structure as a JSON schema, in the strict form the provider requires: every
# property listed as required, and no extra properties allowed.
ANSWER_SCHEMA = {
    "type": "object",
    "properties": {
        "found": {"type": "boolean"},
        "summary": {"type": "string"},
        "evidence": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {"passage": {"type": "string"}, "quote": {"type": "string"}},
                "required": ["passage", "quote"],
                "additionalProperties": False,
            },
        },
    },
    "required": ["found", "summary", "evidence"],
    "additionalProperties": False,
}


class VerifiedEvidence(BaseModel):
    chunk_id: str
    quote: str
    span: Span


class ClauseResult(BaseModel):
    """The final, checked result for one clause of one contract."""

    contract_id: str
    clause: str
    found: bool
    summary: str
    evidence: list[VerifiedEvidence]
    status: str
    problems: list[str]
    first_problems: list[str]
    passage_ids: list[str]
    prompt_tokens: int
    completion_tokens: int
    api_calls: int

    def needs_review(self) -> bool:
        """True when a human should look at this result."""
        return self.status == "failed_validation"


def fold(character: str) -> str:
    """Lowercase one character, unless that would change the text length."""
    lowered = character.lower()
    return lowered if len(lowered) == 1 else character


def locate_quote(quote: str, chunk: Chunk) -> Span | None:
    """Find a quote inside a chunk, ignoring spaces and capital letters.

    Contracts often have broken spacing ("Agreementare"), and an LLM may repair it.
    Returns the quote's position in the contract, or None.
    """
    wanted = "".join(fold(c) for c in quote if not c.isspace())
    if not wanted:
        return None

    kept = []
    positions = []
    for index, character in enumerate(chunk.text):
        if not character.isspace():
            kept.append(fold(character))
            positions.append(index)

    found_at = "".join(kept).find(wanted)
    if found_at == -1:
        return None
    start = positions[found_at]
    end = positions[found_at + len(wanted) - 1] + 1
    return Span(start=chunk.start + start, end=chunk.start + end, text=chunk.text[start:end])


def retrieve_passages(contract_id: str, clause: ClauseSpec, retriever, max_passages: int = 5):
    """The best chunks for a clause, as a dict {label: chunk}, labels being P1, P2, ...

    `retriever` is any object with search(query, contract_id, k), such as retrieval.Retriever.
    Search with every phrasing of the clause and fuse the rankings.
    """
    queries = clause.questions + clause.questions_v2
    hit_lists = [retriever.search(query, contract_id, k=FUSION_DEPTH) for query in queries]
    hits = fuse_hits(hit_lists)[:max_passages]
    return {f"P{number}": hit.chunk for number, hit in enumerate(hits, start=1)}


def build_messages(clause: ClauseSpec, passages: dict[str, Chunk]) -> list[dict]:
    """The conversation we send to the LLM."""
    definition = CLAUSE_DEFINITIONS[clause.key]
    passage_text = "\n\n".join(f"[{label}] {chunk.text}" for label, chunk in passages.items())
    user_text = (
        f"Clause type: {clause.cuad_category}\n"
        f"Definition: {definition}\n\n"
        f"Passages:\n\n{passage_text}\n\n"
        "Answer in the required JSON format."
    )
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_text},
    ]


def check_answer(answer: ClauseAnswer, passages: dict[str, Chunk]) -> list[str]:
    """The problems found in an answer. An empty list means the answer is fine."""
    problems = []
    if not answer.found:
        if answer.evidence:
            problems.append("found is false, so the evidence list must be empty")
        return problems

    if not answer.summary.strip():
        problems.append("found is true, so the summary must not be empty")
    if not answer.evidence:
        problems.append("found is true, so at least one evidence quote is required")

    for number, item in enumerate(answer.evidence, start=1):
        chunk = passages.get(item.passage)
        if chunk is None:
            labels = ", ".join(passages)
            problems.append(
                f"evidence {number}: passage '{item.passage}' does not exist ({labels})"
            )
        elif len(item.quote.strip()) < MIN_QUOTE_LENGTH:
            problems.append(f"evidence {number}: the quote is too short to prove anything")
        elif locate_quote(item.quote, chunk) is None:
            problems.append(
                f"evidence {number}: the quote does not appear word for word in {item.passage}"
            )
    return problems


def parse_answer(text: str) -> tuple[ClauseAnswer | None, list[str]]:
    """Turn the LLM's JSON text into a ClauseAnswer, or explain why that is not possible."""
    try:
        return ClauseAnswer.model_validate_json(text), []
    except ValidationError as error:
        return None, [
            f"the answer does not match the required format: {error.error_count()} errors"
        ]


def answer_clause(
    contract_id: str, clause: ClauseSpec, retriever, llm, max_passages: int = 5
) -> ClauseResult:
    """Retrieve, ask the LLM, check the answer, and retry once if the check fails.

    `llm` is any object with complete_json(messages, schema, name), such as llm.LLMClient.
    """
    passages = retrieve_passages(contract_id, clause, retriever, max_passages)
    messages = build_messages(clause, passages)

    prompt_tokens = 0
    completion_tokens = 0
    api_calls = 0
    answer = None
    problems: list[str] = []
    first_problems: list[str] = []
    status = "failed_validation"

    for attempt in range(1, MAX_ATTEMPTS + 1):
        reply = llm.complete_json(messages, ANSWER_SCHEMA, "clause_answer")
        prompt_tokens += reply.prompt_tokens
        completion_tokens += reply.completion_tokens

        if not reply.cached:
            api_calls += 1

        answer, problems = parse_answer(reply.text)
        if answer is not None:
            problems = check_answer(answer, passages)

        if attempt == 1:
            first_problems = list(problems)

        if not problems:
            status = "ok" if attempt == 1 else "retried"
            break

        # Tell the LLM what was wrong and let it try again.
        feedback = "Your answer has these problems:\n- " + "\n- ".join(problems)
        feedback += "\nAnswer again and fix them. Copy each quote exactly from one passage."
        messages = messages + [
            {"role": "assistant", "content": reply.text},
            {"role": "user", "content": feedback},
        ]

    evidence = []
    if status != "failed_validation" and answer is not None and answer.found:
        for item in answer.evidence:
            chunk = passages[item.passage]
            span = locate_quote(item.quote, chunk)
            if span is not None:
                evidence.append(VerifiedEvidence(chunk_id=chunk.id, quote=item.quote, span=span))

    ok = status != "failed_validation" and answer is not None
    return ClauseResult(
        contract_id=contract_id,
        clause=clause.key,
        found=bool(ok and answer is not None and answer.found),
        summary=answer.summary if ok and answer is not None else "",
        evidence=evidence,
        status=status,
        problems=problems,
        first_problems=first_problems,
        passage_ids=[chunk.id for chunk in passages.values()],
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        api_calls=api_calls,
    )

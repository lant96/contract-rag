"""Data models shared across the project."""

from pydantic import BaseModel


class Span(BaseModel):
    """A piece of a contract's text, given as character positions: text[start:end]."""

    start: int
    end: int
    text: str

    def overlaps(self, other: "Span") -> bool:
        return self.start < other.end and other.start < self.end


class Contract(BaseModel):
    """One contract. The title is used as the id."""

    id: str
    text: str


class GoldLabel(BaseModel):
    """If the clause is not in the contract, `spans` is empty."""

    contract_id: str
    clause: str
    spans: list[Span]

    def is_present(self) -> bool:
        return len(self.spans) > 0


class Chunk(BaseModel):
    id: str
    contract_id: str
    start: int
    end: int
    text: str

    def overlaps(self, span: Span) -> bool:
        """True if the chunk shares at least one character with the span"""
        return self.start < span.end and span.start < self.end

    def contains(self, span: Span) -> bool:
        """True if the whole span is into the chunk"""
        return self.start <= span.start and span.end <= self.end

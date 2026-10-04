"""Find the best chunks of one contract: vector search, keyword search (BM25), or both."""

import math
import re
from collections import Counter

from contract_rag.index import ChunkIndex
from contract_rag.schemas import Chunk, Hit

FUSION_DEPTH = 10
RRF_K = 60
PREFIX_LENGTH = 6
MODES = ["dense", "bm25", "hybrid"]


def fuse_hits(hit_lists: list[list[Hit]]) -> list[Hit]:
    """Combine several rankings of the same contract into one (Reciprocal Rank Fusion).

    Every chunk gets 1 / (RRF_K + rank) from each list it appears in, and the scores are
    added up. A chunk that several rankings put near the top ends up on top.
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


def tokenize(text: str) -> list[str]:
    """Split a text into lowercase words and keep the first six letters of each.

    Cutting words short is a crude stemmer: "compete", "competing" and "competition"
    all become "compet", so they match each other.
    """
    words = re.findall(r"[a-z0-9]+", text.lower())
    return [word[:PREFIX_LENGTH] for word in words]


class KeywordIndex:
    """BM25 keyword search over the chunks of one contract.

    BM25 scores a chunk by the query words it contains. Rare words count more than
    common ones (idf), repeating a word helps only a little, and long chunks are
    slightly penalised.
    """

    def __init__(self, chunks: list[Chunk], k1: float = 1.5, b: float = 0.75):
        self.chunks = chunks
        self.k1 = k1
        self.b = b
        self.term_counts = [Counter(tokenize(chunk.text)) for chunk in chunks]
        self.lengths = [sum(counts.values()) for counts in self.term_counts]

        total_length = sum(self.lengths)
        self.average_length = total_length / len(chunks) if total_length > 0 else 1.0

        # In how many chunks does each word appear?
        self.document_frequency: Counter[str] = Counter()
        for counts in self.term_counts:
            self.document_frequency.update(counts.keys())

    def idf(self, term: str) -> float:
        """How rare a word is in this contract (always positive)."""
        n = len(self.chunks)
        df = self.document_frequency[term]
        return math.log(1 + (n - df + 0.5) / (df + 0.5))

    def score(self, query_terms: set[str], position: int) -> float:
        """BM25 score of the chunk at `position` for a set of query words."""
        counts = self.term_counts[position]
        length_part = 1 - self.b + self.b * self.lengths[position] / self.average_length

        total = 0.0
        for term in query_terms:
            frequency = counts[term]
            if frequency == 0:
                continue
            total += (
                self.idf(term) * frequency * (self.k1 + 1) / (frequency + self.k1 * length_part)
            )
        return total

    def search(self, query: str, k: int = 5) -> list[Hit]:
        """The k chunks with the best score. Chunks without any query word are left out."""
        query_terms = set(tokenize(query))
        scored = []
        for position in range(len(self.chunks)):
            score = self.score(query_terms, position)
            if score > 0:
                scored.append((score, position))

        scored.sort(key=lambda item: (-item[0], item[1]))  # best score first, then earliest
        return [Hit(chunk=self.chunks[position], score=score) for score, position in scored[:k]]


class Retriever:
    """Searches inside one contract with the chosen method.

    mode "dense":  vector search only
    mode "bm25":   keyword search only
    mode "hybrid": both rankings, combined with Reciprocal Rank Fusion
    """

    def __init__(self, index: ChunkIndex, mode: str = "dense"):
        if mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}, not '{mode}'")
        self.index = index
        self.mode = mode
        self.keyword_indexes: dict[str, KeywordIndex] = {}

    def keyword_index(self, contract_id: str) -> KeywordIndex:
        if contract_id not in self.keyword_indexes:
            chunks = self.index.get_chunks(contract_id)
            self.keyword_indexes[contract_id] = KeywordIndex(chunks)
        return self.keyword_indexes[contract_id]

    def search(self, query: str, contract_id: str, k: int = 5) -> list[Hit]:
        if self.mode == "dense":
            return self.index.search(query, contract_id, k=k)

        keyword_hits = self.keyword_index(contract_id).search(query, k=FUSION_DEPTH)
        if self.mode == "bm25":
            return keyword_hits[:k]

        dense_hits = self.index.search(query, contract_id, k=FUSION_DEPTH)
        return fuse_hits([dense_hits, keyword_hits])[:k]

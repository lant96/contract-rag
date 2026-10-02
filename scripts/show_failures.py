"""Show the queries where retrieval missed, for one clause, so we can see why.

Example
    uv run python scripts/show_failures.py --config fixed_1500_300 --clause exclusivity --limit 50
"""

import argparse
from pathlib import Path

from contract_rag.chunking import CHUNKER_CONFIGS
from contract_rag.clauses import CLAUSES
from contract_rag.data.download import TEST_JSON_NAME, download_cuad
from contract_rag.data.loader import (
    load_contracts,
    load_gold_labels,
    load_test_ids,
    sample_contracts,
    split_contracts,
)
from contract_rag.embeddings import Embedder
from contract_rag.evaluation import span_is_found
from contract_rag.index import ChunkIndex

INDEX_DIR = Path("data/index")
PREVIEW_LENGTH = 260


def preview(text: str) -> str:
    """The first characters of a text, on one line."""
    return text[:PREVIEW_LENGTH].replace("\n", " ")


def main() -> None:
    parser = argparse.ArgumentParser(description="Show retrieval misses for one clause.")
    parser.add_argument("--config", required=True, choices=list(CHUNKER_CONFIGS))
    parser.add_argument("--clause", required=True, choices=[c.key for c in CLAUSES])
    parser.add_argument("--split", default="dev", choices=["dev", "test"])
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--show", type=int, default=5, help="how many misses to print")
    args = parser.parse_args()

    path = download_cuad()
    contracts = load_contracts(path)
    labels = load_gold_labels(path)
    test_ids = load_test_ids(path.parent / TEST_JSON_NAME)
    dev, test = split_contracts(contracts, test_ids)

    chosen = dev if args.split == "dev" else test
    if args.limit is not None:
        chosen = sample_contracts(chosen, args.limit)
    wanted = {c.id for c in chosen}

    clause = [c for c in CLAUSES if c.key == args.clause][0]
    index = ChunkIndex(INDEX_DIR, args.config, Embedder())

    queries = 0
    misses = 0
    printed = 0
    for label in labels:
        if label.contract_id not in wanted or label.clause != args.clause:
            continue
        if not label.is_present():
            continue
        for question in clause.questions:
            queries += 1
            hits = index.search(question, label.contract_id, k=5)
            if not hits:
                print("No chunks found. Build the index for these contracts first.")
                return
            if any(span_is_found(hits, span, 0.5) for span in label.spans):
                continue

            misses += 1
            if printed >= args.show:
                continue
            printed += 1

            # How close did the best chunk get to the gold answer?
            best_overlap = 0.0
            for hit in hits:
                for span in label.spans:
                    best_overlap = max(best_overlap, hit.chunk.overlap_fraction(span))

            gold = label.spans[0]
            print(f"--- miss {printed} " + "-" * 50)
            print(f"contract: {label.contract_id[:70]}")
            print(f"question: {question}")
            print(f"gold answer ({gold.end - gold.start} chars): {preview(gold.text)}")
            print(f"best overlap of any top-5 chunk with the gold answer: {best_overlap:.0%}")
            print(f"top hit (score {hits[0].score:.3f}): {preview(hits[0].chunk.text)}")
            print()

    print(f"{misses} of {queries} queries missed in the top 5 ({args.clause}, {args.config}).")


if __name__ == "__main__":
    main()

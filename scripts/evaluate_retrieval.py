"""Measure retrieval quality for one index and save the results.

Examples (run from the project root; the contracts must already be in the index):
    uv run python scripts/evaluate_retrieval.py --config paragraph_1500 --split dev --limit 50
    uv run python scripts/evaluate_retrieval.py --config paragraph_1500 --split test

Use the same --split and --limit that you used in build_index.py.
"""

import argparse
import json
from pathlib import Path

from contract_rag.chunking import CHUNKER_CONFIGS, chunk_contract
from contract_rag.clauses import CLAUSES
from contract_rag.data.download import TEST_JSON_NAME, download_cuad
from contract_rag.data.loader import (
    load_contracts,
    load_gold_labels,
    load_test_ids,
    sample_contracts,
    split_contracts,
)
from contract_rag.embeddings import MODEL_NAME, Embedder
from contract_rag.evaluation import (
    METRICS,
    average,
    average_per_clause,
    evaluate_retrieval,
)
from contract_rag.index import ChunkIndex

INDEX_DIR = Path("data/index")
RESULTS_DIR = Path("results")


def print_row(name: str, scores: dict) -> None:
    numbers = "  ".join(f"{scores[m]:7.3f}" for m in METRICS)
    print(f"{name:30s} {scores['queries']:5d}  {numbers}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate retrieval for one index.")
    parser.add_argument("--config", required=True, choices=list(CHUNKER_CONFIGS))
    parser.add_argument("--split", required=True, choices=["dev", "test"])
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    path = download_cuad()
    contracts = load_contracts(path)
    labels = load_gold_labels(path)
    test_ids = load_test_ids(path.parent / TEST_JSON_NAME)
    dev, test = split_contracts(contracts, test_ids)

    chosen = dev if args.split == "dev" else test
    if args.limit is not None:
        chosen = sample_contracts(chosen, args.limit)
    contract_ids = [c.id for c in chosen]

    index = ChunkIndex(INDEX_DIR, args.config, Embedder())
    missing = []
    for contract in chosen:
        expected = len(chunk_contract(contract, args.config))
        if index.chunk_count(contract.id) != expected:
            missing.append(contract.id)
    if missing:
        print(f"{len(missing)} of {len(contract_ids)} contracts are not in the index yet.")
        print("Run build_index.py with the same --config, --split and --limit first.")
        return

    rows = evaluate_retrieval(index, contract_ids, labels, CLAUSES)
    overall = average(rows)
    per_clause = average_per_clause(rows)

    print(f"\nindex: {args.config} | split: {args.split} | contracts: {len(contract_ids)}")
    print(f"{'':30s} {'n':>5s}  " + "  ".join(f"{m:>7s}" for m in METRICS))
    print_row("ALL", overall)
    for clause_key, scores in per_clause.items():
        print_row(clause_key, scores)

    RESULTS_DIR.mkdir(exist_ok=True)
    out_path = RESULTS_DIR / f"retrieval_{args.split}_{args.config}.json"
    report = {
        "config": args.config,
        "split": args.split,
        "contracts": len(contract_ids),
        "embedding_model": MODEL_NAME,
        "overall": overall,
        "per_clause": per_clause,
        "rows": rows,
    }
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()

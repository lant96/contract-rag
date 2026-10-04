"""Measure retrieval quality for one index and save the results.

Examples
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
from contract_rag.retrieval import MODES

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
    parser.add_argument(
        "--questions",
        default="v1",
        choices=["v1", "v2", "both"],
        help="which queries to use: v1, v2, or both sets together",
    )
    parser.add_argument(
        "--retriever",
        default="dense",
        choices=MODES,
        help="dense = vector search, bm25 = keyword search, hybrid = both combined",
    )
    parser.add_argument(
        "--fuse",
        action="store_true",
        help="search all queries of a clause together and combine their rankings",
    )
    parser.add_argument(
        "--no-prefix",
        action="store_true",
        help="do not add the bge query instruction in front of the queries",
    )
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

    embedder = Embedder(query_prefix="") if args.no_prefix else Embedder()
    index = ChunkIndex(INDEX_DIR, args.config, embedder)
    missing = []
    for contract in chosen:
        expected = len(chunk_contract(contract, args.config))
        if index.chunk_count(contract.id) != expected:
            missing.append(contract.id)
    if missing:
        print(f"{len(missing)} of {len(contract_ids)} contracts are not in the index yet.")
        print("Run build_index.py with the same --config, --split and --limit first.")
        return

    clauses = CLAUSES
    if args.questions == "v2":
        clauses = [c.model_copy(update={"questions": c.questions_v2}) for c in CLAUSES]
    elif args.questions == "both":
        both = [c.model_copy(update={"questions": c.questions + c.questions_v2}) for c in CLAUSES]
        clauses = both
    rows = evaluate_retrieval(
        index, contract_ids, labels, clauses, fuse=args.fuse, mode=args.retriever
    )
    overall = average(rows)
    per_clause = average_per_clause(rows)

    prefix_text = "off" if args.no_prefix else "on"
    print(f"\nindex: {args.config} | split: {args.split} | contracts: {len(contract_ids)}")
    print(f"retriever: {args.retriever} | questions: {args.questions} | fused: {args.fuse}")
    print(f"query prefix: {prefix_text}")
    print(f"{'':30s} {'n':>5s}  " + "  ".join(f"{m:>7s}" for m in METRICS))
    print_row("ALL", overall)
    for clause_key, scores in per_clause.items():
        print_row(clause_key, scores)

    RESULTS_DIR.mkdir(exist_ok=True)
    suffix = ""
    if args.questions != "v1":
        suffix += f"_{args.questions}"
    if args.no_prefix:
        suffix += "_noprefix"
    if args.retriever != "dense":
        suffix += f"_{args.retriever}"
    if args.fuse:
        suffix += "_fused"
    out_path = RESULTS_DIR / f"retrieval_{args.split}_{args.config}{suffix}.json"
    report = {
        "config": args.config,
        "split": args.split,
        "contracts": len(contract_ids),
        "embedding_model": MODEL_NAME,
        "questions": args.questions,
        "query_prefix": not args.no_prefix,
        "retriever": args.retriever,
        "fused": args.fuse,
        "overall": overall,
        "per_clause": per_clause,
        "rows": rows,
    }
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()

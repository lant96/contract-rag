"""Embed contracts and store their chunks in a Chroma index.

Examples (run from the project root):
    uv run python scripts/build_index.py --config fixed_1000_200 --split dev --limit 100
    uv run python scripts/build_index.py --config fixed_1000_200 --split test

You can stop the script and run it again: contracts already in the index are skipped.
"""

import argparse
import time
from pathlib import Path

from contract_rag.chunking import CHUNKER_CONFIGS, chunk_contract
from contract_rag.data.download import TEST_JSON_NAME, download_cuad
from contract_rag.data.loader import (
    load_contracts,
    load_test_ids,
    sample_contracts,
    split_contracts,
)
from contract_rag.embeddings import Embedder
from contract_rag.index import ChunkIndex

INDEX_DIR = Path("data/index")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a chunk index for retrieval experiments.")
    parser.add_argument("--config", required=True, choices=list(CHUNKER_CONFIGS))
    parser.add_argument("--split", required=True, choices=["dev", "test"])
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="only use this many randomly chosen contracts (always the same ones)",
    )
    args = parser.parse_args()

    path = download_cuad()
    contracts = load_contracts(path)
    test_ids = load_test_ids(path.parent / TEST_JSON_NAME)
    dev, test = split_contracts(contracts, test_ids)

    chosen = dev if args.split == "dev" else test
    if args.limit is not None:
        chosen = sample_contracts(chosen, args.limit)

    print("Loading the embedding model (the first time it is downloaded)...")
    index = ChunkIndex(INDEX_DIR, args.config, Embedder())
    print(f"Index '{args.config}' already holds {index.count()} chunks.\n")

    start_time = time.time()
    new_chunks = 0
    for number, contract in enumerate(chosen, start=1):
        chunks = chunk_contract(contract, args.config)
        if index.chunk_count(contract.id) == len(chunks):
            print(f"[{number}/{len(chosen)}] skipped (already indexed): {contract.id[:60]}")
            continue
        index.add_chunks(chunks)
        new_chunks += len(chunks)
        print(f"[{number}/{len(chosen)}] {len(chunks):4d} chunks  {contract.id[:60]}")

    seconds = time.time() - start_time
    if new_chunks > 0:
        print(
            f"\nAdded {new_chunks} chunks in {seconds:.0f} s ({new_chunks / seconds:.1f} chunks/s)"
        )
    print(f"Index '{args.config}' now holds {index.count()} chunks.")


if __name__ == "__main__":
    main()

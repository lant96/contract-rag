"""Ask one question about one contract and print the best matching chunks.

Example (the contract must already be in the index):
    uv run python scripts/search_demo.py --config paragraph_1500 --contract Remarketing \
        --question "Which law governs this agreement?"
"""

import argparse
from pathlib import Path

from contract_rag.data.download import download_cuad
from contract_rag.data.loader import load_contracts
from contract_rag.embeddings import Embedder
from contract_rag.index import ChunkIndex

INDEX_DIR = Path("data/index")


def main() -> None:
    parser = argparse.ArgumentParser(description="Search inside one contract.")
    parser.add_argument("--config", required=True, help="name of the index, e.g. paragraph_1500")
    parser.add_argument("--contract", required=True, help="part of the contract title")
    parser.add_argument("--question", required=True)
    parser.add_argument("-k", type=int, default=3, help="how many chunks to show")
    args = parser.parse_args()

    contracts = load_contracts(download_cuad())
    matches = [c for c in contracts if args.contract.lower() in c.id.lower()]
    if len(matches) != 1:
        print(f"'{args.contract}' matches {len(matches)} contracts. Use a more specific part.")
        for contract in matches[:10]:
            print(f"  {contract.id}")
        return

    contract = matches[0]
    index = ChunkIndex(INDEX_DIR, args.config, Embedder())
    hits = index.search(args.question, contract.id, k=args.k)
    if not hits:
        print("No chunks found. Is this contract in the index? Run build_index.py first.")
        return

    print(f"Contract: {contract.id}")
    print(f"Question: {args.question}\n")
    for rank, hit in enumerate(hits, start=1):
        chunk = hit.chunk
        print(f"#{rank}  score {hit.score:.3f}  characters {chunk.start}-{chunk.end}")
        print(chunk.text[:400].replace("\n", " "))
        print()


if __name__ == "__main__":
    main()

"""Compare chunking strategies on all contracts, before using any embedding model.

For every labeled answer (gold span) we ask: does at least one chunk contain the whole
answer? If an answer is cut between two chunks, no single chunk holds all of it.
"""

import statistics

from contract_rag.chunking import chunk_by_paragraph, chunk_fixed
from contract_rag.data.download import download_cuad
from contract_rag.data.loader import load_contracts, load_gold_labels

# (name, chunking function, settings)
CONFIGS = [
    ("fixed 500/100", chunk_fixed, {"size": 500, "overlap": 100}),
    ("fixed 1000/200", chunk_fixed, {"size": 1000, "overlap": 200}),
    ("fixed 1500/300", chunk_fixed, {"size": 1500, "overlap": 300}),
    ("paragraph 1500", chunk_by_paragraph, {"max_size": 1500}),
    ("paragraph 2500", chunk_by_paragraph, {"max_size": 2500}),
]


def main() -> None:
    path = download_cuad()
    contracts = load_contracts(path)
    labels = load_gold_labels(path)
    n_spans = sum(len(label.spans) for label in labels)

    print(f"{len(contracts)} contracts, {n_spans} gold spans\n")
    print(f"{'strategy':16s} {'chunks':>7s} {'median len':>11s} {'spans kept whole':>17s}")

    for name, chunker, settings in CONFIGS:
        chunks_of = {}
        lengths = []
        for contract in contracts:
            chunks = chunker(contract, **settings)
            chunks_of[contract.id] = chunks
            lengths.extend(len(c.text) for c in chunks)

        whole = 0
        for label in labels:
            for span in label.spans:
                if any(chunk.contains(span) for chunk in chunks_of[label.contract_id]):
                    whole += 1

        print(
            f"{name:16s} {len(lengths):7d} {int(statistics.median(lengths)):11d} "
            f"{whole / n_spans:16.1%}"
        )


if __name__ == "__main__":
    main()

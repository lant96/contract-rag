"""Compare two saved retrieval results and say whether the difference could be noise.

Each file is reduced to one number per (contract, clause) pair. If a run asked several
questions for a pair, their scores are averaged. Then resample the contracts many times
to see how much the difference could change by chance (a "bootstrap").

Example:
    uv run python scripts/compare_runs.py results/retrieval_dev_paragraph_1500.json \
        results/retrieval_dev_paragraph_1500_fused.json
"""

import json
import sys
from pathlib import Path

from contract_rag.stats import bootstrap_difference

METRICS = ["hit@1", "hit@5", "mrr"]


def load_pair_scores(path: Path, metric: str) -> dict:
    """Map (contract id, clause) -> the metric, averaged over the questions of that pair."""
    report = json.loads(path.read_text(encoding="utf-8"))
    values = {}
    for row in report["rows"]:
        key = (row["contract_id"], row["clause"])
        values.setdefault(key, []).append(row[metric])
    return {key: sum(numbers) / len(numbers) for key, numbers in values.items()}


def main() -> None:
    if len(sys.argv) != 3:
        print("Usage: uv run python scripts/compare_runs.py <results A> <results B>")
        return

    path_a = Path(sys.argv[1])
    path_b = Path(sys.argv[2])
    print(f"A: {path_a.stem}")
    print(f"B: {path_b.stem}\n")
    print(f"{'metric':8s} {'A':>7s} {'B':>7s} {'B - A':>8s}   95% interval          verdict")

    n_pairs = 0
    n_contracts = 0
    for metric in METRICS:
        scores_a = load_pair_scores(path_a, metric)
        scores_b = load_pair_scores(path_b, metric)
        shared = sorted(set(scores_a) & set(scores_b))
        if len(shared) != len(scores_a) or len(shared) != len(scores_b):
            print(f"Warning: only {len(shared)} pairs are in both files.")

        # Group the differences by contract, so the bootstrap can resample whole contracts.
        by_contract = {}
        for contract_id, clause in shared:
            difference = scores_b[(contract_id, clause)] - scores_a[(contract_id, clause)]
            by_contract.setdefault(contract_id, []).append(difference)

        n_pairs = len(shared)
        n_contracts = len(by_contract)
        mean_a = sum(scores_a[key] for key in shared) / len(shared)
        mean_b = sum(scores_b[key] for key in shared) / len(shared)
        mean, low, high = bootstrap_difference(list(by_contract.values()))
        verdict = "clear difference" if low > 0 or high < 0 else "could be noise"
        print(
            f"{metric:8s} {mean_a:7.3f} {mean_b:7.3f} {mean:+8.3f}   "
            f"[{low:+.3f}, {high:+.3f}]   {verdict}"
        )

    print(f"\n{n_pairs} pairs, {n_contracts} contracts.")


if __name__ == "__main__":
    main()

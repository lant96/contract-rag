"""Show hit@1 and hit@5 for every single query, from a saved results file.

Example
    uv run python scripts/analyze_results.py results/retrieval_dev_fixed_1500_300.json
"""

import json
import sys
from collections import defaultdict
from pathlib import Path


def main() -> None:
    if len(sys.argv) != 2:
        print("Usage: uv run python scripts/analyze_results.py <results file>")
        return

    path = Path(sys.argv[1])
    report = json.loads(path.read_text(encoding="utf-8"))
    if "rows" not in report:
        print("This file has no per-query rows. Re-run evaluate_retrieval.py to create them.")
        return

    # (clause, question) -> list of hit@1 values and list of hit@5 values
    hit1 = defaultdict(list)
    hit5 = defaultdict(list)
    for row in report["rows"]:
        key = (row["clause"], row["question"])
        hit1[key].append(row["hit@1"])
        hit5[key].append(row["hit@5"])

    print(f"{path.name}: questions={report.get('questions', 'v1')}")
    print(f"{'clause':28s} {'n':>3s} {'hit@1':>6s} {'hit@5':>6s}  query")
    for key in sorted(hit1):
        clause, question = key
        n = len(hit1[key])
        h1 = sum(hit1[key]) / n
        h5 = sum(hit5[key]) / n
        print(f"{clause:28s} {n:3d} {h1:6.2f} {h5:6.2f}  {question[:70]}")


if __name__ == "__main__":
    main()

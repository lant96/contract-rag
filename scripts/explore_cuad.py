import json
import statistics
from collections import Counter

from contract_rag.data.download import download_cuad


def category_of(qa_id: str) -> str:
    """Extract the category from a CUAD question ID."""
    return qa_id.rsplit("__", 1)[1]


def main() -> None:
    path = download_cuad()
    contracts = json.loads(path.read_text(encoding="utf-8"))["data"]

    # Lengths for chunks
    lengths = [len(c["paragraphs"][0]["context"]) for c in contracts]

    present: Counter[str] = Counter()
    total: Counter[str] = Counter()
    n_spans = 0

    for contract in contracts:
        for qa in contract["paragraphs"][0]["qas"]:
            category = category_of(qa["id"])
            total[category] += 1

            if qa["answers"]:
                present[category] += 1
                n_spans += len(qa["answers"])

    print(f"contracts: {len(contracts)}")
    print(f"categories: {len(total)}")
    print(f"labeled answer spans: {n_spans}")

    print(
        "contract length (chars): "
        f"min={min(lengths)}, "
        f"median={int(statistics.median(lengths))}, "
        f"max={max(lengths)}"
    )

    print("\nCategory: contracts where the clause is present")

    for category, count in present.most_common():
        print(f"  {category:40s} {count:3d}/{total[category]}")


if __name__ == "__main__":
    main()

import json
from pathlib import Path

METRICS = ["hit@1", "hit@3", "hit@5", "recall@5", "mrr"]


def main() -> None:
    files = sorted(Path("results").glob("retrieval_*.json"))
    if not files:
        print("No results found in the results folder.")
        return

    print(f"{'file':52s} {'n':>4s}  " + "  ".join(f"{m:>8s}" for m in METRICS))
    for path in files:
        report = json.loads(path.read_text(encoding="utf-8"))
        overall = report["overall"]
        numbers = "  ".join(f"{overall[m]:8.3f}" for m in METRICS)
        print(f"{path.stem:52s} {overall['queries']:4d}  {numbers}")


if __name__ == "__main__":
    main()

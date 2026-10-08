"""List the pairs where the answer pipeline went wrong, from a saved results file.

Examples
    uv run python scripts/show_answer_errors.py results/answers_FILE.json
    uv run python scripts/show_answer_errors.py results/answers_FILE.json --only-wrong
"""

import json
import sys
from pathlib import Path

from contract_rag.data.download import download_cuad
from contract_rag.data.loader import load_gold_labels

PREVIEW_LENGTH = 230


def preview(text: str) -> str:
    return text[:PREVIEW_LENGTH].replace("\n", " ")


def main() -> None:
    arguments = [a for a in sys.argv[1:] if not a.startswith("--")]
    only_wrong = "--only-wrong" in sys.argv
    if len(arguments) != 1:
        print("Usage: uv run python scripts/show_answer_errors.py <results file> [--only-wrong]")
        return

    report = json.loads(Path(arguments[0]).read_text(encoding="utf-8"))
    gold_for = {(g.contract_id, g.clause): g for g in load_gold_labels(download_cuad())}

    shown = 0
    for number in range(len(report["rows"])):
        row = report["rows"][number]
        result = report["results"][number]
        wrong = not row["correct"]
        had_trouble = row["status"] != "ok"
        if not wrong and not (had_trouble and not only_wrong):
            continue
        shown += 1

        expert = "present" if row["gold_present"] else "absent"
        model = "found" if row["found"] else "not found"
        print(f"--- pair {number + 1}: {row['clause']} | expert: {expert} | model: {model}")
        print(f"    contract: {row['contract_id'][:60]}")
        shown_to_model = row["retrieval_hit"]
        print(f"    status: {row['status']} | expert span was in the passages: {shown_to_model}")
        for problem in result["first_problems"]:
            print(f"    first attempt problem: {problem}")
        for problem in result["problems"]:
            print(f"    final problem: {problem}")
        if result["summary"]:
            print(f"    model summary: {preview(result['summary'])}")
        for item in result["evidence"]:
            print(f"    model quote: {preview(item['quote'])}")
        if row["gold_present"] and not row["found"]:
            gold = gold_for[(row["contract_id"], row["clause"])]
            for span in gold.spans[:2]:
                print(f"    EXPERT text: {preview(span.text)}")
        print()

    print(f"{shown} pairs listed.")


if __name__ == "__main__":
    main()

"""Evaluate the whole answering pipeline (retrieval + LLM + checks) against the expert labels.

Examples
    uv run python scripts/evaluate_answers.py --split dev --limit 5
    uv run python scripts/evaluate_answers.py --split test --limit 12

Everything the LLM says is cached, so you can stop and run the same command again: finished
pairs cost nothing. The script pauses between real calls to respect the per-minute limit, and
stops after --max-calls real calls to protect the daily limit.
"""

import argparse
import hashlib
import json
import time
from pathlib import Path

from contract_rag.answer_eval import metric_with_interval, score_result, summarize
from contract_rag.answering import SYSTEM_PROMPT, answer_clause, retrieve_passages
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
from contract_rag.embeddings import Embedder
from contract_rag.index import ChunkIndex
from contract_rag.llm import LLMClient, LLMError, load_settings
from contract_rag.retrieval import Retriever

INDEX_DIR = Path("data/index")
RESULTS_DIR = Path("results")


def fmt(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.3f}"


def is_present(row: dict) -> bool:
    return row["gold_present"]


def is_absent(row: dict) -> bool:
    return not row["gold_present"]


def is_found_present(row: dict) -> bool:
    """The clause is there and the system said so."""
    return row["gold_present"] and row["found"]


def print_interval(name: str, rows: list[dict], select, value) -> None:
    result = metric_with_interval(rows, select, value)
    if result is None:
        print(f"  {name:34s} n/a")
        return
    count, mean, low, high = result
    print(f"  {name:34s} {mean:.3f}  [{low:.3f}, {high:.3f}]  (n={count})")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate the answering pipeline.")
    parser.add_argument("--split", required=True, choices=["dev", "test"])
    parser.add_argument("--limit", required=True, type=int, help="how many contracts to use")
    parser.add_argument("--config", default="paragraph_1500", choices=list(CHUNKER_CONFIGS))
    parser.add_argument("--pause", type=float, default=15.0, help="seconds between real calls")
    parser.add_argument(
        "--max-calls", type=int, default=100, help="stop after this many real calls"
    )
    args = parser.parse_args()

    path = download_cuad()
    contracts = load_contracts(path)
    labels = load_gold_labels(path)
    test_ids = load_test_ids(path.parent / TEST_JSON_NAME)
    dev, test = split_contracts(contracts, test_ids)
    chosen = sample_contracts(dev if args.split == "dev" else test, args.limit)

    index = ChunkIndex(INDEX_DIR, args.config, Embedder())
    missing = [
        c.id for c in chosen if index.chunk_count(c.id) != len(chunk_contract(c, args.config))
    ]
    if missing:
        print(f"{len(missing)} of {len(chosen)} contracts are not in the index yet.")
        print("Run build_index.py with the same --config and --split first.")
        return

    retriever = Retriever(index, "hybrid")
    settings = load_settings()
    llm = LLMClient(settings)
    gold_for = {(g.contract_id, g.clause): g for g in labels}
    pairs = [(contract, clause) for contract in chosen for clause in CLAUSES]
    print(
        f"{len(chosen)} contracts x {len(CLAUSES)} clauses = {len(pairs)} pairs | {settings.model}"
    )
    print(f"A real call costs roughly 2,000 tokens; at most {args.max_calls} real calls now.\n")

    rows = []
    results = []
    api_calls = 0
    complete = True
    for number, (contract, clause) in enumerate(pairs, start=1):
        if api_calls >= args.max_calls:
            print(f"\nStopping after {api_calls} real calls (--max-calls).")
            print("Run the same command later: finished pairs are cached and cost nothing.")
            complete = False
            break
        try:
            result = answer_clause(contract.id, clause, retriever, llm)
        except LLMError as error:
            print(f"\nFAILED: {error}")
            complete = False
            break

        api_calls += result.api_calls
        passages = list(retrieve_passages(contract.id, clause, retriever).values())
        row = score_result(result, gold_for[(contract.id, clause.key)], passages)
        rows.append(row)
        results.append(result.model_dump())

        expert = "present" if row["gold_present"] else "absent "
        verdict = "found    " if result.found else "not found"
        mark = "ok " if row["correct"] else "WRONG"
        print(
            f"[{number}/{len(pairs)}] {clause.key:27s} expert {expert} -> {verdict} "
            f"{mark} | {result.status} | {result.prompt_tokens}+{result.completion_tokens} tokens"
            f"{' (cached)' if result.api_calls == 0 else ''}"
        )
        if result.api_calls > 0 and args.pause > 0:
            time.sleep(args.pause * result.api_calls)

    if not rows:
        return

    summary = summarize(rows)
    print(f"\n=== {len(rows)} of {len(pairs)} pairs{'' if complete else ' (INCOMPLETE)'} ===")
    print(f"clause present in {summary['present']} pairs, absent in {summary['absent']}")
    print(
        f"confusion: found+present {summary['tp']}, missed {summary['fn']}, "
        f"false claims {summary['fp']}, correct 'not found' {summary['tn']}"
    )
    print("\nWith 95% intervals (resampling contracts):")
    print_interval("decision accuracy", rows, lambda r: True, lambda r: float(r["correct"]))
    print_interval("recall (present clauses found)", rows, is_present, lambda r: float(r["found"]))
    print_interval(
        "false-found rate (absent claimed)", rows, is_absent, lambda r: float(r["found"])
    )
    print_interval("retrieval ceiling", rows, is_present, lambda r: float(r["retrieval_hit"]))
    print_interval("evidence hit rate", rows, is_found_present, lambda r: float(r["evidence_hit"]))
    print_interval(
        "evidence precision", rows, is_found_present, lambda r: float(r["evidence_precision"])
    )
    print(
        f"\nF1 {fmt(summary['f1'])} | retried {fmt(summary['retried_rate'])} | "
        f"failed validation {fmt(summary['failed_validation_rate'])}"
    )
    print(
        f"mean tokens per pair: {fmt(summary['mean_prompt_tokens'])} in, "
        f"{fmt(summary['mean_completion_tokens'])} out"
    )

    print("\nPer clause:   present absent  found-when-present  false-claims  evidence-hit")
    for clause in CLAUSES:
        part = summarize([r for r in rows if r["clause"] == clause.key])
        if part["pairs"] == 0:
            continue
        line = f"  {clause.key:27s} {part['present']:4d} {part['absent']:6d}  "
        line += f"{part['tp']:3d}/{part['present']:<3d}          {part['fp']:7d}  "
        line += f"{fmt(part['evidence_hit_rate']):>10s}"
        print(line)

    RESULTS_DIR.mkdir(exist_ok=True)
    prompt_id = hashlib.sha256(SYSTEM_PROMPT.encode("utf-8")).hexdigest()[:8]
    out_path = RESULTS_DIR / f"answers_{args.split}_{args.limit}contracts_{prompt_id}.json"
    report = {
        "split": args.split,
        "contracts": [c.id for c in chosen],
        "model": settings.model,
        "prompt_id": prompt_id,
        "complete": complete,
        "summary": summary,
        "rows": rows,
        "results": results,
    }
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()

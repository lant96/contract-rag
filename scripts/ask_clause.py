"""Ask about one clause of one contract and show the whole chain of evidence.

Example:
    uv run python scripts/ask_clause.py --contract EmmisCommunicationsCorp --clause governing_law

One run costs about 2,000 tokens of the free daily limit.
"""

import argparse
from pathlib import Path

from contract_rag.answering import answer_clause, retrieve_passages
from contract_rag.chunking import CHUNKER_CONFIGS
from contract_rag.clauses import CLAUSES
from contract_rag.data.download import download_cuad
from contract_rag.data.loader import load_contracts, load_gold_labels
from contract_rag.embeddings import Embedder
from contract_rag.index import ChunkIndex
from contract_rag.llm import LLMClient, LLMError, load_settings
from contract_rag.retrieval import Retriever

INDEX_DIR = Path("data/index")


def preview(text: str, length: int = 300) -> str:
    return text[:length].replace("\n", " ")


def main() -> None:
    parser = argparse.ArgumentParser(description="Ask about one clause of one contract.")
    parser.add_argument("--config", default="paragraph_1500", choices=list(CHUNKER_CONFIGS))
    parser.add_argument("--contract", required=True, help="part of the contract title")
    parser.add_argument("--clause", required=True, choices=[c.key for c in CLAUSES])
    args = parser.parse_args()

    path = download_cuad()
    contracts = load_contracts(path)
    matches = [c for c in contracts if args.contract.lower() in c.id.lower()]
    if len(matches) != 1:
        print(f"'{args.contract}' matches {len(matches)} contracts. Use a more specific part.")
        for match in matches[:10]:
            print(f"  {match.id}")
        return
    contract = matches[0]
    clause = [c for c in CLAUSES if c.key == args.clause][0]

    index = ChunkIndex(INDEX_DIR, args.config, Embedder())
    retriever = Retriever(index, "hybrid")

    print(f"Contract: {contract.id}")
    print(f"Clause:   {clause.cuad_category}\n")
    passages = retrieve_passages(contract.id, clause, retriever)
    if not passages:
        print("No chunks found. Is this contract in the index? Run build_index.py first.")
        return
    print("Passages shown to the LLM:")
    for label, chunk in passages.items():
        print(f"  [{label}] characters {chunk.start}-{chunk.end}: {preview(chunk.text, 90)}")

    try:
        result = answer_clause(contract.id, clause, retriever, LLMClient(load_settings()))
    except LLMError as error:
        print(f"\nFAILED: {error}")
        return

    print(
        f"\nstatus: {result.status} | found: {result.found} | needs review: {result.needs_review()}"
    )
    print(f"tokens: {result.prompt_tokens} in, {result.completion_tokens} out")
    if result.summary:
        print(f"summary: {result.summary}")
    for number, item in enumerate(result.evidence, start=1):
        print(f"evidence {number} (verified, characters {item.span.start}-{item.span.end}):")
        print(f"  {preview(item.span.text)}")
    for problem in result.problems:
        print(f"problem: {problem}")

    gold = [g for g in load_gold_labels(path) if g.contract_id == contract.id]
    gold = [g for g in gold if g.clause == clause.key][0]
    print("\nExpert label (CUAD):")
    if not gold.is_present():
        print("  the clause is NOT in this contract")
    for span in gold.spans[:3]:
        print(f"  characters {span.start}-{span.end}: {preview(span.text)}")
    if gold.is_present() and result.evidence:
        overlap = any(e.span.overlaps(s) for e in result.evidence for s in gold.spans)
        print(f"\nDoes the evidence overlap an expert span? {'yes' if overlap else 'no'}")


if __name__ == "__main__":
    main()

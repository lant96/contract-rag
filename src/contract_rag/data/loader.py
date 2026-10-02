"""Turn the raw CUAD JSON files into the models in schemas.py."""

import json
import random
from pathlib import Path

from contract_rag.clauses import CLAUSES
from contract_rag.schemas import Contract, GoldLabel, Span


def read_json(path: Path) -> list:
    """Read a CUAD file and return its list of contracts (the "data" part)."""
    with open(path, encoding="utf-8") as f:
        return json.load(f)["data"]


def load_contracts(path: Path) -> list[Contract]:
    """Load every contract: id = title, text = full contract text."""
    contracts = []
    for item in read_json(path):
        text = item["paragraphs"][0]["context"]
        contracts.append(Contract(id=item["title"], text=text))
    return contracts


def load_gold_labels(path: Path) -> list[GoldLabel]:
    """Load the expert labels for the clauses listed in clauses.py.

    Returns one GoldLabel per (contract, clause) pair.
    """
    # Map the CUAD category name to the clause key
    key_for_category = {}
    for clause in CLAUSES:
        key_for_category[clause.cuad_category] = clause.key

    labels = []
    for item in read_json(path):
        for qa in item["paragraphs"][0]["qas"]:
            # The id looks like "<contract title>__<Category>", so the category comes last.
            category = qa["id"].split("__")[-1]
            if category not in key_for_category:
                continue

            spans = []
            for answer in qa["answers"]:
                start = answer["answer_start"]
                text = answer["text"]
                spans.append(Span(start=start, end=start + len(text), text=text))

            label = GoldLabel(
                contract_id=item["title"],
                clause=key_for_category[category],
                spans=spans,
            )
            labels.append(label)
    return labels


def load_test_ids(path: Path) -> set:
    """Titles of the contracts in the official CUAD test split."""
    return {item["title"] for item in read_json(path)}


def split_contracts(contracts: list[Contract], test_ids: set):
    """Split the contracts into (dev, test). Tune on dev, report final results on test."""
    dev = [c for c in contracts if c.id not in test_ids]
    test = [c for c in contracts if c.id in test_ids]
    return dev, test


def sample_contracts(contracts: list[Contract], n: int, seed: int = 42) -> list[Contract]:
    """Pick n contracts at random. The same seed always gives the same contracts."""
    rng = random.Random(seed)
    return rng.sample(contracts, min(n, len(contracts)))

import json
from pathlib import Path

import pytest

from contract_rag.clauses import CLAUSES
from contract_rag.data.download import CUAD_JSON_NAME, TEST_JSON_NAME
from contract_rag.data.loader import (
    load_contracts,
    load_gold_labels,
    load_test_ids,
    split_contracts,
)
from contract_rag.schemas import Span

RAW_DIR = Path("data/raw")
HAVE_DATA = (RAW_DIR / CUAD_JSON_NAME).exists() and (RAW_DIR / TEST_JSON_NAME).exists()


def write_tiny_cuad(folder: Path) -> None:
    """Write two tiny fake contracts in the same format as the real CUAD files."""
    text_a = "This Agreement is governed by the laws of England. Nothing else."
    answer = "governed by the laws of England"
    contract_a = {
        "title": "contract_a",
        "paragraphs": [
            {
                "context": text_a,
                "qas": [
                    {
                        "id": "contract_a__Governing Law",
                        "answers": [{"text": answer, "answer_start": text_a.index(answer)}],
                    },
                    {"id": "contract_a__Exclusivity", "answers": []},
                ],
            }
        ],
    }
    contract_b = {
        "title": "contract_b",
        "paragraphs": [
            {
                "context": "Short text.",
                "qas": [{"id": "contract_b__Governing Law", "answers": []}],
            }
        ],
    }
    # contract_b is the only "test" contract
    (folder / CUAD_JSON_NAME).write_text(json.dumps({"data": [contract_a, contract_b]}))
    (folder / TEST_JSON_NAME).write_text(json.dumps({"data": [contract_b]}))


def test_span_overlap() -> None:
    a = Span(start=0, end=10, text="x" * 10)
    assert a.overlaps(Span(start=9, end=20, text="x" * 11))
    assert not a.overlaps(Span(start=10, end=20, text="x" * 10))  # touching is not overlap


def test_gold_labels_present_and_absent(tmp_path: Path) -> None:
    write_tiny_cuad(tmp_path)
    labels = load_gold_labels(tmp_path / CUAD_JSON_NAME)

    # contract_a has governing_law + exclusivity, contract_b has governing_law
    assert len(labels) == 3

    first = labels[0]
    assert first.contract_id == "contract_a"
    assert first.clause == "governing_law"
    assert first.is_present()
    assert first.spans[0].text == "governed by the laws of England"

    second = labels[1]
    assert second.clause == "exclusivity"
    assert not second.is_present()


def test_split_has_no_overlap(tmp_path: Path) -> None:
    write_tiny_cuad(tmp_path)
    contracts = load_contracts(tmp_path / CUAD_JSON_NAME)
    test_ids = load_test_ids(tmp_path / TEST_JSON_NAME)

    dev, test = split_contracts(contracts, test_ids)
    assert [c.id for c in dev] == ["contract_a"]
    assert [c.id for c in test] == ["contract_b"]


@pytest.mark.skipif(not HAVE_DATA, reason="CUAD not downloaded: run scripts/explore_cuad.py")
def test_real_cuad_counts_and_offsets() -> None:
    """Sanity check on the real data (skipped if it has not been downloaded)."""
    contracts = load_contracts(RAW_DIR / CUAD_JSON_NAME)
    ids = [c.id for c in contracts]
    assert len(contracts) == 510
    assert len(set(ids)) == 510

    test_ids = load_test_ids(RAW_DIR / TEST_JSON_NAME)
    dev, test = split_contracts(contracts, test_ids)
    assert len(dev) == 408
    assert len(test) == 102

    labels = load_gold_labels(RAW_DIR / CUAD_JSON_NAME)
    assert len(labels) == 510 * len(CLAUSES)

    # Every labeled span must match the contract text at its position
    text_of = {c.id: c.text for c in contracts}
    for label in labels:
        for span in label.spans:
            assert text_of[label.contract_id][span.start : span.end] == span.text

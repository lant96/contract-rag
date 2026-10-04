from contract_rag.clauses import CLAUSES
from contract_rag.paraphrases import PARAPHRASES


def test_every_clause_has_the_same_number_of_v1_and_v2_questions() -> None:
    # Both sets must have the same size, so that comparing them is fair.
    for clause in CLAUSES:
        assert len(clause.questions) > 0
        assert len(clause.questions_v2) == len(clause.questions), clause.key


def test_clause_keys_are_unique() -> None:
    keys = [clause.key for clause in CLAUSES]
    assert len(set(keys)) == len(keys)


def test_paraphrases_cover_every_clause_with_the_same_number_of_questions() -> None:
    assert set(PARAPHRASES) == {clause.key for clause in CLAUSES}
    for clause in CLAUSES:
        assert len(PARAPHRASES[clause.key]) == len(clause.questions), clause.key

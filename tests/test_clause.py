from contract_rag.clauses import CLAUSES


def test_every_clause_has_the_same_number_of_v1_and_v2_questions() -> None:
    # Both sets must have the same size, so that comparing them is fair
    for clause in CLAUSES:
        assert len(clause.questions) > 0
        assert len(clause.questions_v2) == len(clause.questions), clause.key


def test_clause_keys_are_unique() -> None:
    keys = [clause.key for clause in CLAUSES]
    assert len(set(keys)) == len(keys)

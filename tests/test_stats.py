from contract_rag.stats import bootstrap_difference


def test_no_difference_gives_a_zero_interval() -> None:
    groups = [[0.0, 0.0], [0.0], [0.0, 0.0, 0.0]]
    mean, low, high = bootstrap_difference(groups)
    assert (mean, low, high) == (0.0, 0.0, 0.0)


def test_a_steady_improvement_is_clearly_positive() -> None:
    groups = [[0.1, 0.1] for _ in range(30)]
    mean, low, high = bootstrap_difference(groups)
    assert abs(mean - 0.1) < 1e-9
    assert low > 0


def test_random_ups_and_downs_include_zero() -> None:
    groups = [[1.0] for _ in range(20)] + [[-1.0] for _ in range(20)]
    mean, low, high = bootstrap_difference(groups)
    assert mean == 0.0
    assert low < 0 < high


def test_same_seed_gives_the_same_interval() -> None:
    groups = [[0.5, -0.2], [0.1], [0.3, 0.3], [-0.4]]
    assert bootstrap_difference(groups, seed=1) == bootstrap_difference(groups, seed=1)

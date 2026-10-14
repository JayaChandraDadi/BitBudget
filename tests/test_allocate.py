import random

import pytest

from bitbudget.allocate import solve_bruteforce, solve_dp, solve_lagrangian


def random_instance(rng, n_layers=5, bits=(2, 3, 4, 8)):
    errors, costs = [], []
    for _ in range(n_layers):
        n = rng.choice([64, 128, 256, 512])
        # convex-ish, decreasing error in bits, plus noise
        scale = rng.uniform(0.1, 10)
        errors.append([scale * 4.0 ** (-b) * rng.uniform(0.7, 1.3) for b in bits])
        costs.append([n * (b + 0.5) for b in bits])
    return errors, costs


@pytest.mark.parametrize("seed", range(25))
def test_dp_matches_bruteforce(seed):
    rng = random.Random(seed)
    errors, costs = random_instance(rng)
    lo = sum(min(c) for c in costs)
    hi = sum(max(c) for c in costs)
    budget = rng.uniform(lo, hi)
    bf = solve_bruteforce(errors, costs, budget)
    # integer costs: a unit of 1 bit makes the DP exact
    dp = solve_dp(errors, costs, budget, max_units=int(budget) + 1)
    assert dp.cost <= budget
    assert dp.error == pytest.approx(bf.error, rel=1e-9)


@pytest.mark.parametrize("seed", range(10))
def test_dp_coarse_discretization_never_exceeds_budget(seed):
    rng = random.Random(100 + seed)
    errors, costs = random_instance(rng, n_layers=8)
    lo, hi = sum(min(c) for c in costs), sum(max(c) for c in costs)
    budget = rng.uniform(lo * 1.05, hi)
    dp = solve_dp(errors, costs, budget, max_units=50)  # very coarse
    assert dp.cost <= budget


@pytest.mark.parametrize("seed", range(10))
def test_lagrangian_feasible_and_not_better_than_optimum(seed):
    rng = random.Random(200 + seed)
    errors, costs = random_instance(rng)
    lo, hi = sum(min(c) for c in costs), sum(max(c) for c in costs)
    budget = rng.uniform(lo, hi)
    bf = solve_bruteforce(errors, costs, budget)
    lg = solve_lagrangian(errors, costs, budget)
    assert lg.cost <= budget + 1e-6
    assert lg.error >= bf.error - 1e-12


def test_more_budget_never_hurts():
    rng = random.Random(7)
    errors, costs = random_instance(rng, n_layers=10)
    lo, hi = sum(min(c) for c in costs), sum(max(c) for c in costs)
    prev = float("inf")
    for k in range(11):
        budget = lo + (hi - lo) * k / 10
        e = solve_dp(errors, costs, budget).error
        assert e <= prev + 1e-12
        prev = e


def test_infeasible_budget_raises():
    errors = [[1.0, 0.5], [1.0, 0.5]]
    costs = [[10, 20], [10, 20]]
    with pytest.raises(ValueError):
        solve_dp(errors, costs, 19)
    with pytest.raises(ValueError):
        solve_lagrangian(errors, costs, 19)


def test_full_budget_picks_lowest_error():
    errors = [[3.0, 2.0, 1.0], [5.0, 1.0, 0.1]]
    costs = [[1, 2, 3], [1, 2, 3]]
    assert solve_dp(errors, costs, 6, max_units=6).choice == [2, 2]

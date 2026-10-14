"""Bit allocation under a memory budget.

Each quantizable layer ``l`` must pick one bit-width ``b`` from a candidate set.
Picking ``b`` costs ``n_l * (b + overhead/group_size)`` bits of storage and adds
an (estimated) loss ``e_l(b)``. Assuming the losses of different layers add up,
the best allocation for a budget ``C`` is the *multiple-choice knapsack*

    minimize  sum_l e_l(b_l)   subject to   sum_l cost_l(b_l) <= C.

``solve_dp`` solves it exactly (up to cost discretization, which is rounded
*up* so the budget is never exceeded). ``solve_lagrangian`` is the classic
fast relaxation: for a multiplier ``lam`` each layer independently minimizes
``e + lam * cost``; ``lam`` is bisected to meet the budget. It only reaches
allocations on the lower convex hull, so it can be slightly suboptimal.
"""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass

import numpy as np


@dataclass
class Allocation:
    choice: list[int]  # index into each layer's option list
    cost: float  # total cost in bits
    error: float  # total predicted error


def _check(errors, costs):
    if len(errors) != len(costs):
        raise ValueError("errors and costs must have one entry per layer")
    for e, c in zip(errors, costs):
        if len(e) != len(c) or len(e) == 0:
            raise ValueError("each layer needs the same non-zero number of options")


def _total(errors, costs, choice):
    e = sum(errors[l][j] for l, j in enumerate(choice))
    c = sum(costs[l][j] for l, j in enumerate(choice))
    return Allocation(list(choice), float(c), float(e))


def solve_dp(errors, costs, budget: float, max_units: int = 200_000) -> Allocation:
    """Exact multiple-choice knapsack by dynamic programming over cost."""
    _check(errors, costs)
    min_cost = sum(min(c) for c in costs)
    if min_cost > budget * (1 + 1e-12):
        raise ValueError(
            f"budget infeasible: the cheapest allocation needs {min_cost:.0f} bits "
            f"but the budget is {budget:.0f}"
        )
    # Discretize: unit chosen so the capacity has at most max_units cells.
    unit = max(budget / max_units, 1e-12)
    int_costs = [[int(math.ceil(c / unit - 1e-9)) for c in row] for row in costs]
    cap = int(math.floor(budget / unit + 1e-9))

    n_layers = len(errors)
    dp = np.full(cap + 1, np.inf)
    dp[0] = 0.0
    choice = np.zeros((n_layers, cap + 1), dtype=np.int16)
    for l in range(n_layers):
        new = np.full(cap + 1, np.inf)
        new_choice = np.zeros(cap + 1, dtype=np.int16)
        for j, (c, e) in enumerate(zip(int_costs[l], errors[l])):
            if c > cap:
                continue
            cand = np.full(cap + 1, np.inf)
            cand[c:] = dp[: cap + 1 - c] + e
            better = cand < new
            new[better] = cand[better]
            new_choice[better] = j
        dp = new
        choice[l] = new_choice

    if not np.isfinite(dp).any():
        # Rounding costs up pushed a barely-feasible budget over the edge:
        # retry on a finer grid, and at the limit take the cheapest options
        # (feasible by the check above, and optimal when budget == min cost).
        if max_units * 8 <= 2_000_000:
            return solve_dp(errors, costs, budget, max_units * 8)
        cheapest = [
            min(range(len(c)), key=lambda j: (c[j], e[j])) for e, c in zip(errors, costs)
        ]
        return _total(errors, costs, cheapest)
    c = int(np.argmin(dp))
    picks = [0] * n_layers
    for l in range(n_layers - 1, -1, -1):
        j = int(choice[l, c])
        picks[l] = j
        c -= int_costs[l][j]
    assert c == 0
    return _total(errors, costs, picks)


def solve_lagrangian(errors, costs, budget: float, iters: int = 100) -> Allocation:
    """Lagrangian relaxation with bisection on the multiplier."""
    _check(errors, costs)
    E = [np.asarray(e, dtype=np.float64) for e in errors]
    C = [np.asarray(c, dtype=np.float64) for c in costs]

    def pick(lam):
        return [int(np.argmin(e + lam * c + 1e-30 * c)) for e, c in zip(E, C)]

    def cost(choice):
        return sum(C[l][j] for l, j in enumerate(choice))

    min_choice = [int(np.argmin(c)) for c in C]
    if cost(min_choice) > budget * (1 + 1e-12):
        raise ValueError("budget infeasible")
    lo, hi = 0.0, 1.0
    while cost(pick(hi)) > budget:
        hi *= 2.0
        if hi > 1e300:
            return _total(errors, costs, min_choice)
    if cost(pick(lo)) <= budget:
        return _total(errors, costs, pick(lo))
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        if cost(pick(mid)) > budget:
            lo = mid
        else:
            hi = mid
    return _total(errors, costs, pick(hi))


def solve_bruteforce(errors, costs, budget: float) -> Allocation:
    """Exhaustive search (for testing on tiny instances)."""
    _check(errors, costs)
    best = None
    for choice in itertools.product(*[range(len(e)) for e in errors]):
        a = _total(errors, costs, choice)
        if a.cost <= budget and (best is None or a.error < best.error):
            best = a
    if best is None:
        raise ValueError("budget infeasible")
    return best


SOLVERS = {"dp": solve_dp, "lagrangian": solve_lagrangian}

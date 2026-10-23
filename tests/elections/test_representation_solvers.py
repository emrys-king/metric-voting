"""Check solver migration against exhaustive assignment enumeration."""

from collections import Counter
from itertools import combinations, product

import numpy as np
import pulp
import pytest

from metric_voting import (
    ChamberlinCourant,
    ChamberlinCourantTiebreak,
    Monroe,
    MonroeTiebreak,
)
from metric_voting.elections import scip_tiebreak


def enumerate_optimal_assignments(profile, k, balanced):
    """Compute a small election's optimum independently of the solver models."""
    m, n = profile.shape
    scores = m - 1 - np.argsort(profile, axis=0)
    best_score = -1
    solutions = set()
    for committee in combinations(range(m), k):
        for assignment in product(committee, repeat=n):
            counts = Counter(assignment)
            if balanced and any(
                not n // k <= counts[j] <= (n + k - 1) // k for j in committee
            ):
                continue

            score = sum(
                scores[candidate, voter] for voter, candidate in enumerate(assignment)
            )
            if score > best_score:
                best_score, solutions = score, set()
            if score == best_score:
                solutions.add((committee, assignment))

    return best_score, solutions


@pytest.mark.parametrize(
    "election_type,balanced",
    [(MonroeTiebreak, True), (ChamberlinCourantTiebreak, False)],
)
@pytest.mark.parametrize(
    "n,m,k,seed",
    [
        (5, 3, 2, 0),
        (4, 3, 1, 0),
        (4, 3, 3, 1),
        (4, 3, 2, 2),
        (1, 1, 1, 0),
        (5, 3, 2, None),
    ],
)
def test_scip_samples_each_distinct_optimal_committee(
    election_type, balanced, n, m, k, seed, monkeypatch
):
    rng = np.random.default_rng(seed)
    profile = np.column_stack(
        [np.arange(m) if seed is None else rng.permutation(m) for _ in range(n)]
    )
    best_score, assignments = enumerate_optimal_assignments(profile, k, balanced)
    expected = {committee for committee, _ in assignments}
    actual = []
    for index in range(len(expected)):

        def choose_index(high, selected_index=index):
            assert high == len(expected)
            return selected_index

        monkeypatch.setattr(np.random, "randint", choose_index)
        election = election_type()
        actual.append(tuple(election.elect(profile, k)))
        assert election.objective == best_score

    assert set(actual) == expected
    assert len(actual) == len(set(actual))


def test_scip_gives_equal_weight_to_committees_with_unequal_assignment_counts(
    monkeypatch,
):
    profile = np.array(
        [[1, 2, 3, 0, 0], [0, 0, 0, 1, 1], [2, 3, 2, 2, 2], [3, 1, 1, 3, 3]]
    )
    best_score, assignments = enumerate_optimal_assignments(profile, 2, balanced=True)
    assert Counter(committee for committee, _ in assignments) == {
        (0, 1): 2,
        (0, 2): 2,
        (0, 3): 1,
    }
    winners = []
    for index in range(3):

        def choose_index(high, selected_index=index):
            assert high == 3
            return selected_index

        monkeypatch.setattr(np.random, "randint", choose_index)
        election = MonroeTiebreak()
        winners.append(tuple(election.elect(profile, 2)))
        assert election.objective == best_score

    assert set(winners) == {(0, 1), (0, 2), (0, 3)}


@pytest.mark.parametrize("election_type", [MonroeTiebreak, ChamberlinCourantTiebreak])
def test_scip_rejects_incomplete_committee_enumeration(
    election_type, tmp_path, monkeypatch
):
    settings = tmp_path / "limit.set"
    settings.write_text("limits/time = 10\n")
    clock = iter([0.0, 11.0])
    monkeypatch.setattr(scip_tiebreak, "monotonic", lambda: next(clock))

    def reject_sampling(high):
        pytest.fail("An incomplete set of committees must not be sampled")

    monkeypatch.setattr(np.random, "randint", reject_sampling)
    election = election_type(tuning_file_path=str(settings))
    with pytest.raises(RuntimeError, match="committee enumeration stopped: timelimit"):
        election.elect(np.array([[0, 1, 2], [1, 2, 0], [2, 0, 1]]), 2)


@pytest.mark.parametrize("election_type", [MonroeTiebreak, ChamberlinCourantTiebreak])
def test_scip_rejects_unproven_optimum(election_type, tmp_path):
    settings = tmp_path / "limit.set"
    settings.write_text("limits/time = 0\n")
    election = election_type(tuning_file_path=str(settings), n_threads=2)
    with pytest.raises(RuntimeError, match="did not prove an optimum"):
        election.elect(np.array([[0, 1, 2], [1, 2, 0], [2, 0, 1]]), 2)


@pytest.mark.parametrize(
    "election_type,balanced", [(Monroe, True), (ChamberlinCourant, False)]
)
def test_highs_default_returns_optimal_committee(election_type, balanced, tmp_path):
    rng = np.random.default_rng(0)
    profile = np.column_stack([rng.permutation(3) for _ in range(5)])
    best_score, expected = enumerate_optimal_assignments(profile, 2, balanced)
    log_file = tmp_path / "highs.log"
    election = election_type(log_path=str(log_file))
    assert isinstance(election.solver, pulp.HiGHS)
    assert tuple(election.elect(profile, 2)) in {committee for committee, _ in expected}
    assert election.objective == best_score
    assert log_file.stat().st_size > 0


@pytest.mark.parametrize("election_type", [Monroe, ChamberlinCourant])
def test_highs_rejects_feasible_but_unproven_result(election_type, monkeypatch):
    def return_feasible(problem, solver):
        problem.assignStatus(pulp.LpStatusOptimal, pulp.LpSolutionIntegerFeasible)

    monkeypatch.setattr(pulp.LpProblem, "solve", return_feasible)
    with pytest.raises(pulp.PulpSolverError, match="did not prove an optimal"):
        election_type().elect(np.array([[0, 1], [1, 0]]), 1)


@pytest.mark.parametrize("election_type", [Monroe, ChamberlinCourant])
def test_highs_accepts_near_integer_winner_values(election_type, monkeypatch):
    solve = pulp.LpProblem.solve

    def solve_with_roundoff(problem, solver):
        result = solve(problem, solver)
        for variable in problem.variables():
            if variable.name.startswith("y_") and variable.varValue > 0.5:
                variable.varValue = 1.0000000000000002
        return result

    monkeypatch.setattr(pulp.LpProblem, "solve", solve_with_roundoff)
    election = election_type()
    assert election.elect(np.array([[0], [1]]), 1).tolist() == [0]


@pytest.mark.parametrize("election_type", [Monroe, ChamberlinCourant])
def test_highs_accepts_zero_objective(election_type):
    election = election_type()
    assert election.elect(np.array([[0]]), 1).tolist() == [0]
    assert election.objective == 0

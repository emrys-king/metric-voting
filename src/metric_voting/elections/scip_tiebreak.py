"""Enumerate distinct optimal committees for representation elections."""

from time import monotonic

import numpy as np
from numpy.typing import NDArray
from pyscipopt import Expr, Model, Variable, quicksum


def enumerate_optimal_committees(
    model: Model, objective: Expr, candidate_variables: list[Variable]
) -> tuple[list[NDArray[np.int64]], float]:
    """Find every distinct committee attaining the maximum integer Borda score.

    Args:
        model (Model): Unsolved maximization model requiring a fixed committee size.
            The caller owns and frees it. Enumeration changes its objective, constraints,
            and time limit. The initial time limit covers optimization and enumeration.
        objective (Expr): Integer-valued Borda objective already installed on the model.
        candidate_variables (list[Variable]): Binary election variables in candidate-index
            order. Voter-assignment variables do not distinguish committees.

    Returns:
        tuple[list[NDArray[np.int64]], float]: Distinct optimal committees and their Borda
            score. Each committee contains sorted candidate indices; committee order is
            unspecified. Every committee occurs exactly once.

    Raises:
        RuntimeError: SCIP did not prove optimality or finish enumerating committees.
    """
    deadline = monotonic() + model.getParam("limits/time")
    optimal_score = _solve_optimal_borda_score(model)

    committee = np.flatnonzero([model.getVal(variable) > 0.5 for variable in candidate_variables])
    committees = [committee]

    model.freeTransform()
    model.addCons(objective == optimal_score)
    model.setObjective(0)

    while True:
        # Fixed committee size makes this exclude every assignment to this committee alone.
        model.addCons(quicksum(candidate_variables[j] for j in committee) <= len(committee) - 1)
        model.setParam("limits/time", max(0.0, deadline - monotonic()))
        model.optimize()
        status = model.getStatus()

        if status == "infeasible":
            return committees, optimal_score

        if status != "optimal":
            raise RuntimeError(f"SCIP committee enumeration stopped: {status}")

        committee = np.flatnonzero([model.getVal(variable) > 0.5 for variable in candidate_variables])
        committees.append(committee)

        model.freeTransform()


def _solve_optimal_borda_score(model: Model) -> float:
    """Prove the integer Borda optimum, leaving its solution available for extraction.

    Override gap tolerances so a merely near-optimal committee cannot enter the lottery.
    Raise RuntimeError if SCIP stops without proving optimality.
    """
    model.setParam("limits/gap", 0)
    model.setParam("limits/absgap", 0)
    model.optimize()
    status = model.getStatus()

    if status != "optimal":
        raise RuntimeError(f"SCIP did not prove an optimum: {status}")

    return float(round(model.getObjVal()))

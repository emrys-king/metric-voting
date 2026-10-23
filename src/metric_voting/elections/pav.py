from collections.abc import Callable

from .base import Election


class PAV(Election):
    """
    NOTE: This is not working as expected right now ... I will plan to fix it as soon as possible.

    Elect k candidates with the harmonic comittee scoring mechanism..
    This function uses an integer linear program to find a winner set which
    maximizes the sum of harmonic adjusted committee scores.
    With an approval scoring scheme, this is equivalent to the PAV rule.
    With a borda scoring scheme, this is equivalent to the Harmonic Borda rule.

    For more information, please see:
    "What Do Multiwinner Voting Rules Do? An Experiment Over the Two-Dimensional Euclidean Domain"
    Elkind et al (2019)
    (https://arxiv.org/abs/1901.09217)
    NOTE: See section E in the appendix (it is labeled as harmonic borda there, but
        is the same linear program, just with a different scoring mechanism).

    NOTE: Define linear program here once this is fixed.

    Args:
        solver (str, optional): Solver for the integer linear program. These are taken
            from PuLP's available solvers, for more information please see
            (https://coin-or.github.io/pulp/guides/how_to_configure_solvers.html).

            Some options 'PULP_CBC_CMD' and 'GUROBI_CMD' (requires licesnse).
            Defaults to 'PULP_CBC_CMD', which uses PuLPs default coin and branch bound solver.

        log_path (str, optional): Path to log file for solver output. Defaults to None.

    Attributes:
        objective (float): Objective value of the last solved problem.
    """

    def __init__(
        self,
        scoring_scheme: Callable = lambda x, y, z: 1 if z <= y else 0,
        solver: str = "PULP_CBC_CMD",
        log_path: str | None = None,
    ):
        raise NotImplementedError(
            "This class is not working currently. Check for an update in a future release."
        )

    #     self.scoring_scheme = scoring_scheme
    #     self.log_path = log_path
    #     self.solver = pulp.getSolver(solver, msg=False, logPath=log_path)
    #
    # def elect(self, profile: NDArray, k: int) -> NDArray:
    #     """
    #
    #     Args:
    #         profile (np.ndarray): (candidates x voters) Preference Profile.
    #         k (int): Number of candidates to elect
    #
    #     Returns:
    #         elected (np.ndarray): Winning candidates
    #     """
    #     # Creates a new log file for each election
    #     if self.log_path is not None:
    #         with open(self.log_path, "w") as f:
    #             f.write("")
    #
    #     self._approve_profile(profile, k)
    #     m, n = profile.shape
    #     B = borda_matrix(
    #         profile, k, self.scoring_scheme
    #     ).T  # n x m matrix after transpose
    #
    #     # Randomly permute the candidates to break ties
    #     # candidate_permutation = np.random.permutation(m)
    #     # B = B[:, candidate_permutation]
    #
    #     problem = pulp.LpProblem("PAV", pulp.LpMaximize)
    #
    #     # Voter assignment variable
    #     x = pulp.LpVariable.dicts(
    #         "x",
    #         ((i, j, l) for i in range(n) for j in range(m) for l in range(k)),
    #         cat="Binary",
    #     )
    #     # Candidate 'elected' variable
    #     y = pulp.LpVariable.dicts("y", range(m), cat="Binary")
    #
    #     # Objective function:
    #     problem += pulp.lpSum(
    #         (1 / (l + 1)) * B[i, j] * x[i, j, l]
    #         for i in range(n)
    #         for j in range(m)
    #         for l in range(k)
    #     )
    #
    #     # Each candidate may only occupy at most one ranked position
    #     for i in range(n):
    #         for j in range(m):
    #             problem += pulp.lpSum(x[i, j, l] for l in range(k)) <= 1
    #
    #     # Each position l occupied exactly one candidate
    #     for i in range(n):
    #         for l in range(k):
    #             problem += pulp.lpSum(x[i, j, l] for j in range(m)) == 1
    #
    #     # A voter can only be assigned to a candidate if that candidate is elected
    #     for i in range(n):
    #         for j in range(m):
    #             # Voter  i's jth ranked candidate:
    #             # c = np.where(candidate_permutation == profile[j, i])[0][0]
    #             c = profile[j, i]
    #             for l in range(k):
    #                 problem += x[i, j, l] <= y[c]
    #
    #     # Elect exactly k candidates
    #     problem += pulp.lpSum(y[j] for j in range(m)) == k
    #
    #     problem.solve(self.solver)
    #     self.objective = pulp.value(problem.objective)
    #     elected = np.array([j for j in range(m) if pulp.value(y[j]) == 1])
    #     # Translate back to original indices:
    #     # elected = candidate_permutation[elected]
    #     return elected

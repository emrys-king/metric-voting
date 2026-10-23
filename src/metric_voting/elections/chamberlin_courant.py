import gurobipy as gp
import numpy as np
import pulp
from gurobipy import GRB
from numpy.typing import NDArray

from ..utils import borda_matrix, tiebreak
from .base import Election


class ChamberlinCourant(Election):
    """
    Elect k candidates with the Chamberlain Courant Mechanism.
    This function uses an integer linear program to compute an optimal
    assignment of voters to candidates to maximize the assignment scores
    (where assignment scores are calculated with the borda score).

    NOTE: This is a basic implementation of the linear program and it will not randomly
    break ties between optimal solutions. The LP will choose the solution deterministically
    using its own tiebreaking mechanisms. See ChamberlinCourantTiebreak for controlled
    randomization.

    This is based on work from the following papers:
    "Achieving Fully Proportional Representation: Approximability Results"
    Skowron et al (2013)
    (https://arxiv.org/abs/1312.4026)
    NOTE: See section 4.7 starting on page 32.

    "What Do Multiwinner Voting Rules Do? An Experiment Over the Two-Dimensional Euclidean Domain"
    Elkind et al (2019)
    (https://arxiv.org/abs/1901.09217)

    LP definition:
    - Let B be the Borda matrix of the profile, where B(i, j) is the Borda score of candidate j
        in voter i's ranking.
    - Let x(i, j) be a binary representation variable indicating if voter i is assigned
        candidate j as its representative.
    - Let y(j) be a binary variable indicating if candidate j is elected.

    Objective:
        max: sum_{over all voters i and candidates j} Borda-Sore(i, j) * x(i, j)

    subject to:
        1) sum_{over all candidates j} x(i, j) = 1 for all i
            (each voter must get exactly one representative)

        2)  x(i, j) <= y(j) for all i, j
            (a voter can only be assigned to a candidate if that candidate is elected)

        3) sum_{over all candidates j} y(j) = k
            (exactly k candidates are elected)


    Args:
        solver (str, optional): Solver for the integer linear program. These are taken
            from PuLP's available solvers, for more information please see
            (https://coin-or.github.io/pulp/guides/how_to_configure_solvers.html).

            Some options 'PULP_CBC_CMD' and 'GUROBI_CMD' (requires licesnse).
            Defaults to 'PULP_CBC_CMD', which uses PuLPs default coin and branch bound solver.

        log_path (str, optional): Path to log file for solver output. Defaults to 'cc.log'.

    Attributes:
        objective (float): Objective value of the last solved problem.
    """

    def __init__(self, solver: str = "PULP_CBC_CMD", log_path: str | None = None):
        self.log_path = log_path
        self.solver = pulp.getSolver(solver, msg=False, logPath=log_path)
        self.objective = None

    def elect(self, profile: NDArray, k: int) -> NDArray:
        """
        Elect k candidates with the Chamberlain Courant Mechanism.

        Args:
            profile (np.ndarray): (candidates x voters) Preference Profile.
            k (int): Number of candidates to elect

        Returns:
            elected (np.ndarray): Winning candidates
        """
        # Creates a new log file for each election
        if self.log_path is not None:
            with open(self.log_path, "w") as f:
                f.write("")

        self._approve_profile(profile, k)
        m, n = profile.shape
        B = borda_matrix(profile, k).T  # n x m matrix after transpose

        # Randomly permute the candidates to break ties
        # candidate_permutation = np.random.permutation(m)
        # B = B[:, candidate_permutation]

        problem = pulp.LpProblem("Chamberlin-Courant", pulp.LpMaximize)

        # Voter assignment variable (voter i gets assigned to candidate j)
        x = pulp.LpVariable.dicts(
            "x", ((i, j) for i in range(n) for j in range(m)), cat="Binary"
        )
        # Candidate 'elected' variable
        y = pulp.LpVariable.dicts("y", range(m), cat="Binary")

        # Objective function:
        problem += pulp.lpSum(B[i, j] * x[i, j] for i in range(n) for j in range(m))

        # Each voter is assigned to exactly one candidate
        for i in range(n):
            problem += pulp.lpSum(x[i, j] for j in range(m)) == 1

        # A voter can only be assigned to a candidate if that candidate is elected
        for i in range(n):
            for j in range(m):
                problem += x[i, j] <= y[j]

        # Elect exactly k candidates
        problem += pulp.lpSum(y[j] for j in range(m)) == k

        problem.solve(self.solver)
        self.objective = pulp.value(problem.objective)
        elected = np.array([j for j in range(m) if pulp.value(y[j]) == 1])
        # Translate back to original indices:
        # elected = candidate_permutation[elected]
        return elected


####################################################################################################


class ChamberlinCourantTiebreak(Election):
    """
    Elect k candidates with the Chamberlain Courant Mechanism.
    This function uses an integer linear program to compute an optimal
    assignment of voters to candidates to maximize the assignment scores
    (where assignment scores are calculated with the borda score).

    NOTE: This version of the function is specifically designed to randomly break ties.

    A large portion of this code is written by and attributed to: Peter Rock, peter@mggg.org

    NOTE: Since this model uses Gurobi as an LP solver,
    you will need a license to run larger elections.

    This is based on work from the following papers:
    "Achieving Fully Proportional Representation: Approximability Results"
    Skowron et al (2013)
    (https://arxiv.org/abs/1312.4026)
    NOTE: See section 4.7 starting on page 32.

    "What Do Multiwinner Voting Rules Do? An Experiment Over the Two-Dimensional Euclidean Domain"
    Elkind et al (2019)
    (https://arxiv.org/abs/1901.09217)

    LP definition:
    - Let B be the Borda matrix of the profile, where B(i, j) is the Borda score of candidate j
        in voter i's ranking.
    - Let x(i, j) be a binary representation variable indicating if voter i is assigned
        candidate j as its representative.
    - Let y(j) be a binary variable indicating if candidate j is elected.

    Objective:
        max: sum_{over all voters i and candidates j} Borda-Sore(i, j) * x(i, j)

    subject to:
        1) sum_{over all candidates j} x(i, j) = 1 for all i
            (each voter must get exactly one representative)

        2)  x(i, j) <= y(j) for all i, j
            (a voter can only be assigned to a candidate if that candidate is elected)

        3) sum_{over all candidates j} y(j) = k
            (exactly k candidates are elected)


    Args:
        tuning_file_path (str, optional) : Filepath for .prm file giving tuning parameters
            for Gurobi.

    Attributes:
        objective (float): Objective value of the last solved problem.
    """

    def __init__(self, tuning_file_path: str | None = None, n_threads: int = 1):
        self.tuning_file_path = tuning_file_path
        self.n_threads = n_threads
        self.objective = None

    def elect(self, profile: NDArray, k: int) -> NDArray:
        """
        Elect k candidates with the Chamberlain Courant Mechanism.

        Args:
            profile (np.ndarray): (candidates x voters) Preference Profile.
            k (int): Number of candidates to elect

        Returns:
            elected (np.ndarray): Winning candidates
        """
        self._approve_profile(profile, k)
        m, n = profile.shape
        borda_scores = borda_matrix(profile, k).T  # n x m matrix after transpose

        env = gp.Env(params={"OutputFlag": 0})  # Silent mode
        model = gp.Model("ChamberlinCourant", env=env)
        model.setParam("Threads", self.n_threads)

        if self.tuning_file_path is not None:
            model.read(self.tuning_file_path)

        # Define binary variables
        x = model.addVars(n, m, vtype=GRB.BINARY, name="x")
        y = model.addVars(m, vtype=GRB.BINARY, name="y")

        # Objective function: maximize sum of assigned Borda scores
        model.setObjective(
            gp.quicksum(
                borda_scores[i, j] * x[i, j] for i in range(n) for j in range(m)
            ),
            GRB.MAXIMIZE,
        )

        # Constraint 1: Each voter is assigned to exactly 1 candidate
        for i in range(n):
            model.addConstr(
                gp.quicksum(x[i, j] for j in range(m)) == 1, f"AssignVoter_{i}"
            )

        # Constraint 2: A voter can only be assigned to an elected candidate
        for i in range(n):
            for j in range(m):
                model.addConstr(x[i, j] <= y[j], f"OnlyElected_{i}_{j}")

        # Constraint 3: Exactly k candidates are elected
        model.addConstr(gp.quicksum(y[j] for j in range(m)) == k, "ElectExactlyK")

        # Allow for multiple solutions
        model.setParam(GRB.Param.PoolSearchMode, 2)
        # Store mutliple solutions (up at most 100,000)
        n_store = 100000
        model.setParam(GRB.Param.PoolSolutions, n_store)
        # Only store the global optima
        model.setParam(GRB.Param.PoolGap, 0.0)

        model.optimize()

        assert model.status == GRB.OPTIMAL, "OPTIMAL solution not found."

        n_solutions = model.SolCount
        best_solutions = []
        best_values = []
        for sol_idx in range(n_solutions):
            model.setParam(GRB.Param.SolutionNumber, sol_idx)
            model.update()
            obj_value = model.PoolObjVal
            elected_candidates = np.array([j for j in range(m) if y[j].Xn > 0.5])
            best_solutions.append(elected_candidates)
            best_values.append(obj_value)

        # This handles numerical errors -- solutions should be integer values anyways
        best_values = np.round(best_values, decimals=5)

        # Choose randomly among the best solutions available.
        rand_solution_idx = tiebreak(best_values)[0]
        self.objective = best_values[rand_solution_idx]
        return best_solutions[rand_solution_idx]


####################################################################################################


class GreedyCC(Election):
    """
    Elect k candidates using a greedy approximation to the
    Chamberlain Courant rule. At every iteration, this rule
    selects a candidate to add to a growing winner set by greedily selecting
    the candidate which gives the best increase to the current assignment scores.

    For more information, please see the papers:
    "Budgeted Social Choice: From Consensus to Personalized Decision Making"
    - Lu and Boutilier (2011)
    (https://www.cs.toronto.edu/~tl/papers/LuBoutilier_budgeted_IJCAI11.pdf)

    "What Do Multiwinner Voting Rules Do? An Experiment Over the Two-Dimensional Euclidean Domain"
    Elkind et al (2019)
    (https://arxiv.org/abs/1901.09217)


    Attributes:
        objective (float): Objective value of the last solved problem.
    """

    def __init__(self):
        self.objective = None

    def elect(self, profile: NDArray, k: int) -> NDArray:
        """
        Elect k candidates using a greedy approximation to the
        Chamberlain Courant rule.

        Args:
            profile (np.ndarray): (candidates x voters) Preference Profile.
            k (int): Number of candidates to elect

        Returns:
            elected (np.ndarray): Winning candidates
        """
        self._approve_profile(profile, k)
        m, n = profile.shape
        B = borda_matrix(profile, k)

        is_elected = np.zeros(m, dtype=bool)
        voter_assign_scores = np.zeros(n) - 1

        for _ in range(k):
            scores = np.zeros(m)
            for i in range(m):
                if not is_elected[i]:
                    score_gain = np.sum(np.maximum(voter_assign_scores, B[i, :]))
                    scores[i] = score_gain

            # Break ties randomly
            ranking = tiebreak(scores)
            max_cand = ranking[-1]
            is_elected[max_cand] = True
            voter_assign_scores = np.maximum(voter_assign_scores, B[max_cand, :])

        self.objective = np.sum(voter_assign_scores)
        return np.where(is_elected)[0]

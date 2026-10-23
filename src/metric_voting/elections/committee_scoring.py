from collections.abc import Callable

import numpy as np
from numpy.typing import NDArray

from ..utils import tiebreak
from .base import Election


class CommitteeScoring(Election):
    """
    General parent class for electing k candidates via simple Committee Scoring Mechanisms.
    This class uses a scoring scheme to give scores to candidates based upon
    their position in the preference profile. This is a function
    which takes as input the number of candidates m, the number of winners k,
    and the position i in {1,...,m} of the candidate in a given voter's ranking.

    Scores for a candidate are summed over all voters, and the k candidates with the
    largest scores are elected.

    Args:
        scoring_scheme (Callable[[int, int, int], float]): Scheme for candidate scoring.
            Function of the form f(m, k, i) where m is the total number of candidates,
            k is the voter's ranking of the candidate, and
            i is a voter's ranking in [1,...,m] of the candidate.
    """

    def __init__(self, scoring_scheme: Callable[[int, int, int], float]):
        self.scoring_scheme = scoring_scheme

    def elect(self, profile: NDArray, k: int) -> NDArray:
        """
        Elect k candidates with the largest scores.

        Args:
            profile (NDArray): (m x n) Preference Profile.
            k (int): Number of candidates to elect

        Returns:
            elected (NDArray): Winning candidates
        """

        self._approve_profile(profile, k)
        m, n = profile.shape
        candidate_scores = np.zeros(m)

        for i in range(n):
            for j in range(m):
                c = profile[j, i]
                candidate_scores[c] += self.scoring_scheme(m, k, j + 1)

        ranking = tiebreak(candidate_scores)[::-1]
        elected = ranking[:k]
        return elected


####################################################################################################


class Bloc(CommitteeScoring):
    """
    Bloc approval election method. This is a committee scoring method
    which uses the scoring scheme f(m, k, i) = 1 for i <= k and 0 otherwise. In other words,
    all candidates in the top k positions of a voter's ranking receive a score of 1, and all
    other candidates receive a score of 0.
    """

    def __init__(self):
        scoring_scheme = lambda x, y, z: 1 if z <= y else 0
        super().__init__(scoring_scheme)


####################################################################################################


class Borda(CommitteeScoring):
    """
    Elect k candidates with the Borda Scoring Mechanism. This is a committee scoring method
    which uses the scoring scheme f(m, k, i) = m - (i + 1). In other words, candidates receive
    scores of m - 1 for being ranked first, m - 2 for being ranked second, and so on.
    """

    def __init__(self):
        scoring_scheme = lambda x, y, z: x - z
        super().__init__(scoring_scheme)

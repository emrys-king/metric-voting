import numpy as np
from numpy.typing import NDArray

from ..utils import tiebreak
from .base import Election


class SNTV(Election):
    """
    Single Non-Transferable Vote (Plurality) election.

    NOTE: Should this elect random candidates if it can't find k winners??
    """

    def __init__(self):
        pass

    def elect(self, profile: NDArray, k: int) -> NDArray:
        """
        Elect k candidates with the largest plurality scores. If k candidates are not
        named among the first choices, the election moves on to consider the second choices,
        and so on until k candidates are elected.

        Args:
            profile (np.ndarray): (candidates x voters) Preference Profile.
            k (int): Number of candidates to elect

        Returns:
            elected (np.ndarray): Winning candidates
        """
        m, _ = profile.shape
        self._approve_profile(profile, k)
        elected_mask = np.zeros(m, dtype=bool)
        elected_count = 0
        counts = np.zeros(m)

        current_rank = 0
        while elected_count < k:
            votes = profile[current_rank, :]
            for c in votes:
                counts[c] += 1

            ranking = tiebreak(counts)[::-1]
            for c in ranking:
                if (elected_count < k) and (not elected_mask[c]) and (counts[c] > 0):
                    elected_mask[c] = True
                    elected_count += 1

            current_rank += 1

        return np.where(elected_mask)[0]

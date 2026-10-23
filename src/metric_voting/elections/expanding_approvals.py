import numpy as np
from numpy.typing import NDArray

from .base import Election


class ExpandingApprovals(Election):
    """
    Elect k candidates using the expanding approvals rule.
    Candidates are elected through a sequential approval process. At each step i,
    voters approve their ith ranked candidate in a random order. Once candidates
    have passed an approval threshold, they are elected.

    For more information please see the paper:
    "Proportional Representation in Metric Spaces and Low-Distortion Committee Selection"
    -Kalayci, Kempe, Kher (2024)
    (https://arxiv.org/abs/2312.10369)
    """

    def __init__(self):
        self.quota = None
        self.uncovered_mask = None
        self.elected_mask = None
        self.neighborhood = None
        self.random_order = None

    def candidate_check_elect(self, c: int):
        """
        Check if a candidate has passed the quota and elect them if they have.

        Args:
            c (int): Candidate index.
        """
        assert self.quota is not None, "Quota has not been set."
        assert self.uncovered_mask is not None, "Uncovered mask has not been set."
        assert self.elected_mask is not None, "Elected mask has not been set."
        assert self.neighborhood is not None, "Neighborhood has not been set."

        if np.sum(self.neighborhood[c, :]) >= self.quota:
            self.elected_mask[c] = True
            c_voters = np.where(self.neighborhood[c, :])[0]
            self.neighborhood[:, c_voters] = 0
            self.uncovered_mask[c_voters] = False

    def approval_round(self, t: int):
        """
        Conduct an approval round of the expanding approvals rule.

        Args:
            t (int): Round number.
        """
        assert self.profile is not None, "Profile has not been set."
        assert self.uncovered_mask is not None, "Uncovered mask has not been set."
        assert self.elected_mask is not None, "Elected mask has not been set."
        assert self.random_order is not None, "Random order has not been set."
        assert self.neighborhood is not None, "Neighborhood has not been set."

        for v in self.random_order:
            if self.uncovered_mask[v]:
                c = self.profile[t, v]
                if not self.elected_mask[c]:
                    self.neighborhood[c, v] = 1
                    self.candidate_check_elect(c)

    def elect(self, profile: NDArray, k: int) -> NDArray:
        """
        Elect k candidates using the expanding approvals rule.

        Args:
            profile (np.ndarray): (candidates x voters) Preference Profile.
            k (int): Number of candidates to elect

        Returns:
            elected (np.ndarray): Winning candidates
        """
        self._approve_profile(profile, k)
        self.profile = profile
        m, n = profile.shape
        self.quota = np.ceil(n / k)
        self.uncovered_mask = np.ones(n, dtype=bool)
        self.elected_mask = np.zeros(m, dtype=bool)
        self.neighborhood = np.zeros(profile.shape)

        # Main election loop
        for t in range(m):
            self.random_order = np.random.permutation(n)
            self.approval_round(t)

        # Elect remaining candidates if needed
        if np.sum(self.elected_mask) < k:
            remaining = k - np.sum(self.elected_mask)
            non_elected = np.where(self.elected_mask == False)[0]
            new_elects = np.random.choice(non_elected, remaining, replace=False)
            self.elected_mask[new_elects] = True

        elected = np.where(self.elected_mask)[0]
        return elected

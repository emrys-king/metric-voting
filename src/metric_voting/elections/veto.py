import numpy as np
from numpy.typing import NDArray

from .base import Election


class PluralityVeto(Election):
    """
    Elect k candidates with the Plurality Veto mechanism. Counts
    initial plurality scores for every candidate then, in a randomized order,
    lets voters veto candidates until there are only k left.

    For more information please see the paper:
    "Plurality Veto: A Simple Voting Rule Achieving Optimal Metric Distortion"
    - Kizilkaya, Kempe (2023)
    (https://arxiv.org/abs/2206.07098)

    NOTE: That this is an extension for what is really designed to be a single winner
    voting rule. To really use this as a multiwinner rule, need to construct a special
    preference profile. Please see the example notebooks for more detail.
    """

    def __init__(self):
        self.profile = None
        self.m, self.n = None, None
        self.k = None
        self.candidate_scores = None
        self.eliminated = None
        self.last_place_indices = None

    def update_least_preffered(self):
        """
        Updates the last place indices of voters in the preference profile.
        """
        assert self.profile is not None, "Profile has not been set."
        assert self.n is not None, "Number of voters has not been set."
        assert self.eliminated is not None, "Eliminated candidates have not been set."
        assert self.last_place_indices is not None, (
            "Last place indices have not been set."
        )

        for i in range(self.n):
            while self.eliminated[self.profile[self.last_place_indices[i], i]] == 1:
                self.last_place_indices[i] -= 1

    def initialize(self, k):
        """
        Counts first place votes and eliminates any candidates with 0.

        Args:
            k (int): Number of candidates to elect.
        """
        assert self.profile is not None, "Profile has not been set."
        assert self.m is not None, "Number of candidates has not been set."
        assert self.k is not None, "Number of candidates to elect has not been set."
        assert self.candidate_scores is not None, "Candidate scores have not been set."
        assert self.eliminated is not None, "Eliminated candidates have not been set."

        # Count initial plurality scores
        first_choice_votes = self.profile[:k, :]
        mentions, counts = np.unique(first_choice_votes, return_counts=True)
        for i, c in enumerate(mentions):
            self.candidate_scores[c] += counts[i]

        # Find candidates with 0 plurality score
        zero_scores = np.where(self.candidate_scores == 0)[0]
        if len(zero_scores) > self.m - self.k:
            np.random.shuffle(zero_scores)
            zero_scores = zero_scores[: (self.m - self.k)]

        # And remove them from the election
        self.eliminated[zero_scores] = 1
        self.update_least_preffered()

    def elect(self, profile: NDArray, k: int) -> NDArray:
        """
        Elect k candidates with the multiwinner Plurality Veto mechanism.
        Args:
            profile (np.ndarray): (candidates x voters) Preference Profile.
            k (int): Number of candidates to elect

        Returns:
            elected (np.ndarray): Winning candidates
        """

        self._approve_profile(profile, k)
        self.profile = profile
        self.k = k
        self.m, self.n = profile.shape
        self.candidate_scores = np.zeros(self.m)
        self.eliminated = np.zeros(self.m)
        self.last_place_indices = (
            np.zeros(self.n, dtype=int) + self.m - 1
        )  # Last place index

        self.initialize(k)
        eliminated_count = len(np.where(self.eliminated == 1)[0])

        # Veto in a randomize order
        random_order = list(range(self.n))
        np.random.shuffle(random_order)
        while eliminated_count < (self.m - self.k):
            for i, v in enumerate(random_order):
                least_preferred = profile[self.last_place_indices[v], v]
                # A veto decrements the candidates score by 1
                self.candidate_scores[least_preferred] -= 1
                if self.candidate_scores[least_preferred] <= 0:
                    self.eliminated[least_preferred] = 1
                    self.update_least_preffered()
                    eliminated_count += 1
                    random_order = random_order[i + 1 :] + random_order[: i + 1]
                    break

        elected = np.where(self.eliminated == 0)[0]
        return elected


####################################################################################################


class CommitteeVeto(PluralityVeto):
    """
    Elect k candidates with the Committee Veto mechanism. This is really identical to PluralityVeto,
    but should be run with a special preference profile constructed
    over all possible comittees (Just helpful to have another class with a different name).
    """

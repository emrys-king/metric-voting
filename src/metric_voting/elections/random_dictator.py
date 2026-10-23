import numpy as np
from numpy.typing import NDArray

from .base import Election


class SMRD(Election):
    """
    Sequential Multiwinner Random Dictator (SMRD) election method.
    Elect k candidates from k randomly chosen 'dictators'.
    From each dictator elect their first non-elected candidate.
    """

    def __init__(self):
        pass

    def elect(self, profile: NDArray, k: int) -> NDArray:
        """
        Elect k candidates with the SMRD method.

        Args:
            profile (np.ndarray): (candidates x voters) Preference Profile.
            k (int): Number of candidates to elect

        Returns:
            elected (np.ndarray): Winning candidates
        """
        self._approve_profile(profile, k)
        m, n = profile.shape
        elected = np.zeros(k, dtype=int) - 1
        elected_mask = np.zeros(m, dtype=bool)
        dictators = np.random.choice(range(n), size=k, replace=True)

        for i in range(k):
            dictator = dictators[i]
            # find next available candidate:
            for j in range(m):
                choice = profile[j, dictator]
                if not elected_mask[choice]:
                    elected[i] = choice
                    elected_mask[choice] = True
                    break

        return elected


####################################################################################################


class OMRD(Election):
    """
    One-Shot Multiwinner Random Dictator (OMRD) election method.
    Chooses a single random dictator and lets them elect their top k
    preferences.
    """

    def __init__(self):
        pass

    def elect(self, profile: NDArray, k: int) -> NDArray:
        """
        Elect k winners with the OMRD method.

        Args:
            profile (np.ndarray): (candidates x voters) Preference Profile.
            k (int): Number of candidates to elect

        Returns:
            elected (np.ndarray): Winning candidates
        """
        self._approve_profile(profile, k)
        _, n = profile.shape
        dictator = np.random.choice(range(n))
        elected = profile[:k, dictator]
        return elected


####################################################################################################


class DMRD(Election):
    """
    Discounted Multiwinner Random Dictator (DMRD) election method.
    Elect k candidates with k iterations of Random Dictator.
    At each iteration, randomly choose a voter and elect its first choice.
    Then discount or reweight the voting power of voters who voted for that candidate by
    a factor of rho.

    Args:
        rho (float, optional): Reweighting factor. Default is 1.
    """

    def __init__(self, rho: int = 1):
        self.rho = rho

    def update_current_ballot_positions(self):
        """
        Updates the first place vote indices of voters in the preference profile, to account
        for candidates who have already been elected.
        """
        for i in range(self.n):
            while (
                self.elected_mask[
                    self.profile[self.voter_current_ballot_position[i], i]
                ]
                == 1
            ):
                self.voter_current_ballot_position[i] += 1

    def elect(self, profile: NDArray, k: int) -> NDArray:
        """
        Elect k candidates with the DMRD method.

        Args:
            profile (np.ndarray): (candidates x voters) Preference Profile.
            k (int): Number of candidates to elect

        Returns:
            elected (np.ndarray): Winning candidates
        """
        self._approve_profile(profile, k)
        self.m, self.n = profile.shape
        self.profile = profile
        self.voter_current_ballot_position = np.zeros(self.n, dtype=int)
        self.elected_mask = np.zeros(self.m, dtype=bool)
        elected = np.zeros(k, dtype=int) - 1
        voter_probability = np.ones(self.n) / self.n

        for i in range(k):
            dictator = np.random.choice(range(self.n), p=voter_probability)
            winner = self.profile[
                self.voter_current_ballot_position[dictator], dictator
            ]
            elected[i] = winner
            self.elected_mask[winner] = True

            # Find who voted for the winner
            first_choice_votes = self.profile[
                self.voter_current_ballot_position, np.arange(self.n)
            ]
            support = first_choice_votes == winner
            others = ~support

            if np.sum(others) > 0:
                renorm_factor = np.sum(voter_probability[support]) / np.sum(
                    voter_probability[others]
                )
                renorm_factor = 1 + (1 - self.rho) * renorm_factor

                # Adjusts voter probability for the next round
                voter_probability[support] *= self.rho
                voter_probability[others] *= renorm_factor
                # voter_probability /= np.sum(voter_probability)

            # Effectively removes winning candidate from profile
            self.update_current_ballot_positions()

        return elected

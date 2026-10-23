import numpy as np
from numpy.typing import NDArray

####################################################################################################


class Election:
    """
    Base class for election types.
    """

    def __init__(self):
        pass

    def _is_complete_ranking(self, ranking: NDArray) -> bool:
        """
        Checks if the input ranking is a complete ranking of the same m candidates.

        Args:
            ranking (np.ndarray): (m) Array of candidate indices.

        Returns:
            (bool): True if the ranking is complete, False otherwise.
        """
        return bool(np.all(np.isin(np.arange(len(ranking)), np.unique(ranking))))

    def _approve_profile(self, profile: NDArray, k: int) -> None:
        """
        Checks if the input profile is an ranked preference profile in
        correct form. Specifically, for our simplified models, this means
        a complete ranking of the same m candidates for each voter.

        Args:
            profile (np.ndarray): (candidates x voters) Preference profile matrix.

            k (int): Number of candidates to elect.

        Returns:
            (bool): True if the profile is approved, False otherwise.
        """
        m, n = profile.shape
        if n < k:
            raise ValueError("Requested more elected seats than there are voters!")
        if m < k:
            raise ValueError("Requested more elected seats than there are candidates!")

        if not np.all(
            np.apply_along_axis(self._is_complete_ranking, axis=0, arr=profile)
        ):
            raise ValueError("Profile not in correct form.")

    def elect(self, profile: NDArray, k: int) -> NDArray:
        """
        Elect k candidates using an input preference profile.

        Args:
            profile (np.ndarray): (candidates x voters) Preference Profile.
            k (int): Number of candidates to elect

        Returns:
            elected (np.ndarray): Winning candidates
        """

        raise NotImplementedError

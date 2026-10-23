import numpy as np
from numpy.typing import NDArray

from ..utils import geq_with_tol, tiebreak
from .base import Election


class STV(Election):
    """
    Elect k candidates with the Single Transferrable Vote election.
    Uses the droop quota as an election threshold, and breaks ties randomly.
    To transfer votes, one may select either 'fractional', 'weighted-fractional',
    or 'cambridge' transfer types. The 'fractional' type transfers an
    equivalent fraction of surplus votes back to voters. The
    'weighted-fractional' type transfers a fraction of surplus votes back
    to voters proportional to their original vote weight. The 'cambridge'
    type transfers whole surplus votes to a random set of voters who voted for the
    elected candidate.

    Args:
        transfer_type (str): Type of vote transfer.
            Options: 'fractional', 'weighted-fractional', 'cambridge.' Defaults to
            'weighted-fractional'.
        tiebreak_type (str) : Method for breaking ties. Options: 'fpv_random', 'random'.
            Defaults to 'fpv_random', which breaks ties by comparing first round scores,
            and if that fails, breaks ties randomly. Otherwise, the 'random' option breaks
            ties randomly.

    Attributes:
        transfer_type (str): Type of vote transfer.
        tiebreak_type (str): Method for breaking ties.
        verbose (bool): Print election details.
        m (int): Number of candidates.
        n (int): Number of voters.
        droop (int): Droop quota.
        elected_mask (NDArray): Tracks elected candidates.
        eliminated_mask (NDArray): Tracks eliminated candidates.
        voter_current_ballot_position (NDArray): Tracks voter's next available choices.
        voter_weights (NDArray): Weights for each voter's votes.
        elected_count (int): Number of candidates currently elected.
        eliminated_count (int): Number of candidates currently eliminated.
        first_round_scores (NDArray): Scores for the first round of candidates.
    """

    def __init__(
        self,
        transfer_type: str = "weighted-fractional",
        tiebreak_type: str = "fpv_random",
        verbose: bool = False,
    ):
        if transfer_type not in ["fractional", "weighted-fractional", "cambridge"]:
            raise ValueError("Invalid transfer type.")
        self.transfer_type = transfer_type

        self.tiebreak_type = tiebreak_type
        if tiebreak_type not in ["fpv_random", "random"]:
            raise ValueError("Invalid tiebreak type.")

        self.verbose = verbose
        self.m = None
        self.n = None
        self.droop = None
        self.elected_mask = None
        self.eliminated_mask = None
        self.voter_current_ballot_position = None
        self.voter_weights = None
        self.elected_count = None
        self.eliminated_count = None
        self.first_round_scores = None

    def elect(self, profile: NDArray, k: int) -> NDArray:
        """
        Elect k candidates with the Single Transferrable Vote election.

        Args:
            profile (NDArray): (m x n) Preference Profile.
            k (int): Number of candidates to elect

        Returns:
            elected (NDArray): Winning candidates
        """
        self._approve_profile(profile, k)
        self.profile = profile
        self.k = k
        self.m, self.n = profile.shape
        if self.m < k:
            raise ValueError("Requested more elected seats than there are candidates!")

        # Initialize tracking variables
        self.droop = int(self.n / (self.k + 1)) + 1
        self.elected_mask = np.zeros(self.m)
        self.eliminated_mask = np.zeros(self.m)
        self.voter_current_ballot_position = np.zeros(self.n, dtype=int)
        self.voter_weights = np.ones(self.n, dtype=np.float64)
        self.elected_count = 0
        self.eliminated_count = 0

        # Main election loop
        if self.verbose:
            print("Starting Election: ")

        while self.elected_count < self.k and self.eliminated_count < self.m - self.k:
            candidate_scores, candidate_voters, satisfies_quota = self.count()

            if self.elected_count + self.eliminated_count == 0:
                self.first_round_scores = candidate_scores

            if self.verbose:
                print("Round: " + str(self.elected_count + self.eliminated_count))
                print("voter weights: " + str(self.voter_weights))
                print("voter indices: " + str(self.voter_current_ballot_position))
                print("scores: " + str(candidate_scores))
                print("sum of scores: " + str(np.sum(candidate_scores)))
                print("sum of weights: " + str(np.sum(self.voter_weights)))

            if len(satisfies_quota) > 0:
                self.add_candidates(satisfies_quota)
                elected = satisfies_quota
                for c in elected:
                    self.transfer(float(candidate_scores[c]), candidate_voters[c])

                if self.verbose:
                    print("elected: " + str(elected))

            else:
                elim = self.eliminate(candidate_scores)
                self.transfer(candidate_scores[elim], candidate_voters[elim])

                if self.verbose:
                    print("eliminated: " + str(elim))

            if self.verbose:
                print()

        # Final check: Elect any remaining candidates if needed
        if self.elected_count < self.k:
            remaining_candidates = np.where(
                (self.elected_mask == 0) & (self.eliminated_mask == 0)
            )[0]
            self.elected_mask[remaining_candidates] = 1
            self.elected_count += len(remaining_candidates)

        return np.where(self.elected_mask == 1)[0]

    def count(self) -> tuple[NDArray, dict[int, list[int]], NDArray]:
        """
        Count votes in the current round and determine if any
        candidates satisfy the droop quota.

        Returns:
            candidate_scores (NDArray): Length m array containing the
                number of votes for each candidate.
            candidate_voters (dict): Dictionary with candidate indices as keys
                and lists of their voters as values.
            satisfies_quota (NDArray): Array of candidate indices that
                satisfy the droop quota.
        """
        assert self.m is not None, "Number of candidates has not been set."
        assert self.n is not None, "Number of voters has not been set."
        assert self.voter_current_ballot_position is not None, (
            "Voter indices have not been set."
        )
        assert self.voter_weights is not None, "Voter weights have not been set."
        assert self.elected_mask is not None, "Elected mask has not been set."
        assert self.eliminated_mask is not None, "Eliminated mask has not been set."
        assert self.droop is not None, "Droop quota has not been set."

        candidate_scores = np.zeros(self.m, dtype=np.float64)
        candidate_voters = {c: [] for c in range(self.m)}
        for i in range(self.n):
            if self.voter_current_ballot_position[i] != -1:
                voter_choice = self.profile[self.voter_current_ballot_position[i], i]

                if self.voter_weights[i] > 0:
                    # candidate_scores[voter_choice] += self.voter_weights[i]
                    candidate_voters[voter_choice].append(i)

        for c, voters in candidate_voters.items():
            candidate_scores[c] = np.sum(self.voter_weights[voters])

        # Finds cands that satisfy the droop quota that are not elected or eliminated.
        satisfies_quota = np.where(
            (geq_with_tol(candidate_scores, self.droop, tol=1e-10))
            & (self.elected_mask != 1)
            & (self.eliminated_mask != 1)
        )[0]

        return candidate_scores, candidate_voters, satisfies_quota

    def add_candidates(self, satisfies_quota: NDArray):
        """
        Elect all candidates that satisfy the droop quota.

        Args:
            satisfies_quota (NDArray): Array of candidate indices that
                satisfy the droop quota.
        """
        assert self.elected_mask is not None, "Elected mask has not been set."
        assert self.elected_count is not None, "Elected count has not been set."

        self.elected_mask[satisfies_quota] = 1
        self.elected_count += len(satisfies_quota)

    def eliminate(self, candidate_scores: NDArray) -> int:
        """
        Eliminate the candidate with the lowest score, breaking
        ties randomly.

        Args:
            candidate_scores (NDArray): Length m array containing the
                number of votes for each candidate.

        Returns:
            eliminated (int): Index of the eliminated candidate.
        """
        assert self.first_round_scores is not None, (
            "First round scores have not been set."
        )
        assert self.elected_mask is not None, "Elected mask has not been set."
        assert self.eliminated_mask is not None, "Eliminated mask has not been set."
        assert self.eliminated_count is not None, "Eliminated count has not been set."

        if self.tiebreak_type == "fpv_random":
            score_sort = tiebreak(candidate_scores, self.first_round_scores)
        else:
            score_sort = tiebreak(candidate_scores)

        eliminated = None
        for e in score_sort:
            if (self.elected_mask[e] == 0) and (self.eliminated_mask[e] == 0):
                self.eliminated_mask[e] = 1
                self.eliminated_count += 1
                eliminated = e

                # only eliminate one candidate per round
                break

        assert eliminated is not None, "No candidate was eliminated."
        return eliminated

    def transfer(self, total_votes: float, candidate_voters: list[int]):
        """
        Transfer votes from an elected candidate to the next available
        candidate on each voter's ballot. Unless there the ballot
        has been exhausted, in which case the voter is removed
        from any further voting process.

        Args:
            total_votes (float): Total number of votes for the candidate.
            candidate_voters (List[int]): List of indices for voters
                who voted for the elected candidate.
        """
        assert self.droop is not None, "Droop quota has not been set."
        assert self.voter_weights is not None, "Voter weights have not been set."
        assert self.voter_current_ballot_position is not None, (
            "Voter indices have not been set."
        )

        surplus_votes = total_votes - self.droop

        if self.verbose:
            print("surplus: " + str(surplus_votes))

        # This is done with tolerance to account for floating point issues,
        # which are a result of fractional surplus votes.
        if surplus_votes >= -1e-10:
            # We adjust the weight of the voters whose top remaining choice
            # was the elected candidate so that their next candidate gets
            # votes equal to some fraction of their initial weight,
            # where the fraction depends on a function of the surplus votes.
            if self.transfer_type == "fractional":
                self.voter_weights[candidate_voters] = np.maximum(
                    (surplus_votes / len(candidate_voters)), 0.0
                )

            elif self.transfer_type == "weighted-fractional":
                weights_normalized = self.voter_weights[candidate_voters] / total_votes
                self.voter_weights[candidate_voters] = np.maximum(
                    surplus_votes * weights_normalized, 0.0
                )

            elif self.transfer_type == "cambridge":
                quota_voters = np.random.choice(
                    candidate_voters, size=self.droop, replace=False
                )
                self.voter_weights[quota_voters] = 0

            else:
                raise ValueError("Invalid transfer type.")

        # Once the weights have been adjusted, we move the voters'
        # ballot positions to complete the transfer.
        self.adjust_current_ballot_positions(candidate_voters)

    def adjust_current_ballot_positions(self, candidate_voters: list[int]):
        """
        Adjust voter indices to the next available candidate on their ballot.
        If a voter's ballot has been exhausted, remove them from any further
        voting process by setting their voter index to -1.

        Args:
            candidate_voters (List[int]): List of voter indices for
                voters who voted for the candidate to transfer from.
        """
        assert self.voter_current_ballot_position is not None, (
            "Voter indices have not been set."
        )
        assert self.elected_mask is not None, "Elected mask has not been set."
        assert self.eliminated_mask is not None, "Eliminated mask has not been set."
        assert self.profile is not None, "Profile has not been set."

        for v in candidate_voters:
            while self.voter_current_ballot_position[v] != -1 and (
                (
                    self.elected_mask[
                        self.profile[self.voter_current_ballot_position[v], v]
                    ]
                    == 1
                )
                or (
                    self.eliminated_mask[
                        self.profile[self.voter_current_ballot_position[v], v]
                    ]
                    == 1
                )
            ):
                self.voter_current_ballot_position[v] += 1

                # ballot fully exhausted
                if self.voter_current_ballot_position[v] >= self.m:
                    self.voter_current_ballot_position[v] = -1
                    break

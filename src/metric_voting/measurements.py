import numpy as np
import math
from itertools import combinations
from numpy.typing import NDArray
from typing import Callable, Tuple, Any, Optional, Union, List, Set
from .utils import euclidean_distance, tiebreak, random_voter_bloc


def cost_array(
        voter_positions: NDArray, 
        candidate_positions: NDArray, 
        distance_fn: Callable = euclidean_distance
) -> NDArray:
    """
    Given a set of voter and candidate positions along with a distance function,
    returns an (m x n) array with each entry i,j storing the
    distance from candidate i to voter j.

    Args:
        voter_positions (np.ndarray): (n x d) Array of voter positions in a metric space.
        candidate_positions (np.ndarray): (m x d) Array of candidate positions in a metric space.
        distance_fn (callable, optional): Callable distance function which should
            take as input two d dimensional vectors and output a real number,
            defaults to euclidean_distance (see euclidean_distance() above for a reference format).

    Returns:
        (np.ndarray): Size (m x n) array of distances from voters to candidates.
    """

    dists = np.zeros((len(candidate_positions), len(voter_positions)))
    for i in range(len(candidate_positions)):
        for j in range(len(voter_positions)):
            dists[i, j] = distance_fn(voter_positions[j], candidate_positions[i])
    return dists


def euclidean_cost_array(
    voter_positions: NDArray, 
    candidate_positions: NDArray,
) -> NDArray:
    """
    Given a set of voter and candidate positions, returns an
    (m x n) array with each entry i,j storing the
    distance from candidate i to voter j.

    NOTE: This is an optimized version of cost_array()
        which assumes distance is euclidean.

    Args:
        voter_positions (np.ndarray): (n x d) Array of voter positions in a metric space.
        candidate_positions (np.ndarray): (m x d) Array of candidate positions in a metric space.

    Returns:
        (np.ndarray): Size (m x n) array of distances from voters to candidates.
    """
    diffs = voter_positions[np.newaxis, :, :] - candidate_positions[:, np.newaxis, :]
    dists = np.sqrt(np.sum(diffs**2, axis=-1))
    return dists


def cost(cst_array: NDArray) -> float:
    """
    Given an (m x n) cost array storing all distances from each voter to
    each candidate, finds the total cost by summing all entries.

    Args:
        cst_array (np.ndarray): (m x n) Array of costs with 
            each entry i,j computed as the distance from candidate i to voter j. 

    Returns:
        float: Sum of distances (cost).
    """
    return np.sum(cst_array)


def candidate_costs(cst_array: NDArray) -> NDArray:
    """
    Given an (m x n) cost array storing all distances from each voter to
    each candidate, finds the total cost for each candidate
    summed along all voters.

    Args:
        cst_array (np.ndarray): (m x n) Array of costs with 
            each entry i,j computed as the distance from candidate i to voter j. 

    Returns:
        (np.ndarray): Length m array of distance sums for candidates.
    """
    candidate_csts = np.sum(cst_array, axis=1)
    return candidate_csts


def voter_costs(cst_array: NDArray) -> NDArray:
    """
    Given an (m x n) cost array storing all distances from each voter to
    each candidate, finds the total cost for each voter
    summed across all candidates.

    Args:
        cst_array (np.ndarray): (m x n) Array of costs with 
            each entry i,j computed as the distance from candidate i to voter j. 

    Returns:
        (np.ndarray): Length n array of distance sums for voters.
    """
    voter_csts = np.sum(cst_array, axis=0)
    return voter_csts


def q_costs(
    q : int,
    cst_array : NDArray
):
    """
    Given a set of voter and candidate positions along with a distance function,
    returns a length n array with each entry i storing the
    distance from voter i to its qth closest candidate.

    Args:
        q (int): qth closest candidate to compute distance for. 
        cst_array (np.ndarray): (m x n) Array of costs with 
            each entry i,j computed as the distance from candidate i to voter j. 

    Returns:
        (np.ndarray): Size n array of distances from voters to candidates.
    """
    return np.sort(cst_array, axis = 0)[q - 1, :]


def q_cost_array(
    q : int,
    cst_array : NDArray,
    candidate_subsets : List[Set[int]]
) -> NDArray:
    """
    Given a list of candidate subsets, computes the q cost array for each subset.

    Args:
        q (int): qth closest candidate to compute distance for. 
        cst_array (np.ndarray): (m x n) Array of costs with 
            each entry i,j computed as the distance from candidate i to voter j. 
        candidate_subsets (List[Set[int]]): List of candidate subsets to compute q costs for.

    Returns:
        (np.ndarray): Size subsets x n array of q-costs from voters to subsets of candidates.
    """
    q_cst_array = np.zeros((len(candidate_subsets), cst_array.shape[1]))
    for i, subset in enumerate(candidate_subsets):
        q_cst_array[i, :] = q_costs(q, cst_array[list(subset), :])
    return q_cst_array


def min_assignment(cst_array: NDArray, size: int) -> NDArray:
    """
    Given voters and candidates in an input cost array, finds a
    representative subset of candidates of a given size, with minimum sum of 
    costs to voters. 

    Args:
        cst_array (np.ndarray): (m x n) Array of costs with 
            each entry i,j computed as the distance from candidate i to voter j. 
        size (int): Size required for the selected subset of candidates.

    Returns:
        NDArray : Length size array of candidate indices
    """
    if size > len(cst_array):
        raise ValueError("Requested size is too large!")

    candidate_csts = candidate_costs(cst_array)
    return tiebreak(candidate_csts)[:size]


def min_assignment_cost(cst_array: NDArray, size: int) -> float:
    """
    Given voters and candidates in an input cost array, finds the lowest cost 
    assignment of voters to a subset of candidates with a given size. 

    Args:
        cst_array (np.ndarray): (m x n) Array of costs with 
            each entry i,j computed as the distance from candidate i to voter j. 
        size (int): Size required for the selected subset of candidates.

    Returns:
        float: Sum of distances (cost).
    """
    if size > len(cst_array):
        raise ValueError("Requested size is too large!")

    candidate_csts = candidate_costs(cst_array)
    min_cands = tiebreak(candidate_csts)[:size]
    return np.sum(candidate_csts[min_cands])


def proportional_assignment(
    cst_array: NDArray,  
    voter_labels: NDArray, 
    bloc_label: int,
    k: int,
    floor=True,
) -> NDArray:  
    """
    Given voters and candidates in an input cost array, finds the lowest cost 
    representative subset of candidates with proportionally chosen size. 
    
    Args:
        cst_array (np.ndarray): (m x n) Array of costs with 
            each entry i,j computed as the distance from candidate i to voter j. 
        voter_labels (np.ndarray[int]): Integer array where index i gives
                                        the bloc label of voter i.
        bloc_label (int): Selected bloc label to compute score for.
        k (int): Number of winners.
        floor (bool): whether or not to round the number of deserved seats to the nearest integer.
                    if True (default), the number of deserved seats will be rounded down (i.e 2.8 becomes 2);
                    if False, the number of deserved seats will be rounded to the nearest integer
                    (i.e. 2.8 becomes 3).
    
    Returns:
        NDArray : Length size array of candidate indices
    """
    _, n = cst_array.shape
    bloc_size = np.sum(voter_labels == bloc_label)
    bloc_array = cst_array[:, voter_labels == bloc_label]
    if floor:
        size = int(bloc_size / n * k)
    else:
        size = round(bloc_size / n * k)
    return min_assignment(bloc_array, size)


def proportional_assignment_cost(
    cst_array: NDArray,  
    voter_labels: NDArray, 
    bloc_label: int,
    k : int,
    floor=True,
) -> float:  
    """
    Given voters and candidates in an input cost array, finds the lowest cost 
    associated with the representative subset of candidates with proportionally chosen size. 
    
    Args:
        cst_array (np.ndarray): (m x n) Array of costs with 
            each entry i,j computed as the distance from candidate i to voter j. 
        voter_labels (np.ndarray[int]): Integer array where index i gives
                                        the bloc label of voter i.
        bloc_label (int): Selected bloc label to compute score for.
        k (int): Number of winners.
        floor (bool): whether or not to round the number of deserved seats to the nearest integer.
                    if True (default), the number of deserved seats will be rounded down (i.e 2.8 becomes 2);
                    if False, the number of deserved seats will be rounded to the nearest integer
                    (i.e. 2.8 becomes 3).
            
    
    Returns:
        float: Sum of distances (cost).
    """
    _, n = cst_array.shape
    bloc_size = np.sum(voter_labels == bloc_label)
    bloc_array = cst_array[:, voter_labels == bloc_label]
    if floor:
        size = int(bloc_size / n * k)
    else:
        size = round(bloc_size / n * k)
    return min_assignment_cost(bloc_array, size)


def group_inefficiency(
    cst_array: NDArray, 
    winner_indices: NDArray, 
    voter_labels: NDArray, 
    bloc_label: int,
    floor=True
) -> float:
    """
    Computes the group inefficiency score as the cost ratio between
    best subset of winners, and the best subset of all candidates.

    Optional: Instead of using proportionally sized representative sets, set
            the size parameter to enforce that voters are represented by a constant size set.

    Args:
        cst_array (np.ndarray): (m x n) Array of costs with 
            each entry i,j computed as the distance from candidate i to voter j. 
        winner_indices (np.ndarray[int]): Length k array of winning candidate indices.
        voter_labels (np.ndarray[int]): Integer array where index i gives
                                        the bloc label of voter i.
        bloc_label (int): Selected bloc label to compute score for.
        floor (bool): whether or not to round the number of deserved seats to the nearest integer.
                    if True (default), the number of deserved seats will be rounded down (i.e 2.8 becomes 2);
                    if False, the number of deserved seats will be rounded to the nearest integer
                    (i.e. 2.8 becomes 3).
            
        
    Returns:
        float: Group inefficiency score.
    """
    k = len(winner_indices)
    cost1 = proportional_assignment_cost(
        cst_array[winner_indices, :],
        voter_labels,
        bloc_label,
        k,
        floor=floor
    )
    cost2 = proportional_assignment_cost(
        cst_array,
        voter_labels,
        bloc_label,
        k,
        floor=floor
    )
    
    if cost1 == 0 and cost2 == 0:
        return 1
    
    return cost1 / cost2


def random_group_inefficiency(
    cost_arr: NDArray, 
    winner_indices: NDArray, 
    t: Optional[int] = 1,
    weights : Optional[NDArray] = None
) -> Tuple[float, NDArray]:
    """
    For a randomly selected bloc of voters with t representatives, computes the group
    inefficiency score. Each voter is selected with probabilities proportional to their
    weights. By default, the weights are computed using a greedy heuristic. Otherwise, 
    they may be passed in as input.
    
    Args:
        cost_arr (np.ndarray): (m x n) Array of costs with 
            each entry i,j computed as the distance from candidate i to voter j. 
        winner_indices (np.ndarray[int]): Length k array of winning candidate indices.
        t (int, optional): Number of representatives that the voter bloc 'deserves.'
            Defaults to 1.
        weights (np.ndarray, optional): Voter probabilities of selection. Defaults to None, 
            in which case a greedy heuristic is used. 
    """
    m, n = cost_arr.shape
    k = len(winner_indices)
    
    if weights is None:
        # Get cost to winners for all voters
        winner_set_cost_arr = np.sum(
            np.sort(cost_arr[winner_indices, :], axis = 0)[:t, :], axis = 0
        )

        # Cost of best k candidates for each voter
        candidate_set_cost_arr = np.sum(
            np.sort(cost_arr, axis = 0)[:t, :], axis = 0
        )

        # Greedy estimate / inefficiency heuristic for voters
        greedy_scores = np.array([winner_set_cost_arr[i]/candidate_set_cost_arr[i] 
                                  if candidate_set_cost_arr[i] != 0 else 0
                                  for i in range(n)])
        weights = greedy_scores / np.sum(greedy_scores)
    
    random_bloc = random_voter_bloc(n, k, t = t, weights = weights)
    random_bloc_labels = np.zeros(n)
    random_bloc_labels[random_bloc] = 1
    return (
        group_inefficiency(cost_arr, winner_indices, random_bloc_labels, bloc_label=1), 
        random_bloc
    )
    
    
def worst_random_group_inefficiency(
    n_samples : int,
    cost_arr: NDArray, 
    winner_indices: NDArray, 
    weights : Optional[NDArray] = None
) -> Tuple[float, NDArray]:
    """
    Over a series of samples, computes the group inefficiency score for a random group, 
    and outputs the sample that gave the worst ineffciency. 
    
    Args:
        n_samples (int): Number of samples to take.
        cost_arr (np.ndarray): (m x n) Array of costs with 
            each entry i,j computed as the distance from candidate i to voter j. 
        winner_indices (np.ndarray[int]): Length k array of winning candidate indices.
        weights (np.ndarray, optional): Voter probabilities of selection. Defaults to None, 
            in which case a greedy heuristic is used. 
    """
    worst_score = 0
    worst_bloc = None 
    
    for _ in range(n_samples):
        rand_t = np.random.randint(1, len(winner_indices) + 1)
        score, bloc = random_group_inefficiency(
            cost_arr, 
            winner_indices, 
            t = rand_t, 
            weights = weights
        )
        if score > worst_score:
            worst_score = score
            worst_bloc = bloc
    
    return worst_score, worst_bloc    

   

def heuristic_worst_bloc(
    cst_array: NDArray,
    winner_indices: NDArray,
    max_size : int = None
) -> NDArray:
    """
    Given a cost array and a set of winning candidates, finds a cohesive 
    voter bloc with large inefficiency score.
    
    Args:
        cst_array (np.ndarray): (m x n) Array of costs with 
            each entry i,j computed as the distance from candidate i to voter j. 
        winner_indices (np.ndarray[int]): Length k array of winning candidate indices.
        size (int): Maximum number of deserved candidates for the bloc. Defaults to None, in which 
            case the maximum size is taken to be k, the number of winners.
    
    Returns:
        heuristic_bloc (np.ndarray) : Indices of voters in the worst bloc found.
    """
    m,n = cst_array.shape
    k = len(winner_indices)
    if max_size is None:
        max_size = k
    
    heuristic_bloc = None
    heuristic_size = None
    heuristic_score = -1
    
    for s in range(1, max_size + 1):
        size = math.ceil(s * n/k)
        if s == k:
            # all combinations will produce the same result
            combs = [range(s)]
        else:
            combs = combinations(range(m), s)
        
        for comb in combs:
            sum_of_distances = np.sum(cst_array[list(comb), :], axis = 0)
            closest = tiebreak(sum_of_distances)[:size]
            closest_mask = np.zeros(n, dtype = int)
            closest_mask[closest] = 1
            score = group_inefficiency(cst_array, winner_indices, closest_mask, bloc_label=1)
            
            if np.isclose(score,heuristic_score, atol = 1e-8):
                # If scores are close enough that differences can be chalked up to numerical error,
                # Take the bloc with the larger size.
                if size > heuristic_size:
                    heuristic_score = score
                    heuristic_size = size
                    heuristic_bloc = closest
                # If sizes are equal, keep both and break ties randomly later.
                else:
                    if len(heuristic_bloc.shape) == 1:
                        heuristic_bloc = np.array([heuristic_bloc]) 
                    heuristic_bloc = np.vstack((heuristic_bloc, closest))
                        
            elif score > heuristic_score:
                heuristic_score = score
                heuristic_size = size
                heuristic_bloc = closest

    if len(heuristic_bloc.shape) > 1:
        random_idx = np.random.randint(0, heuristic_bloc.shape[0])
        heuristic_bloc = heuristic_bloc[random_idx]
        
    return heuristic_bloc
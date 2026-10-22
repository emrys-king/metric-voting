import numpy as np
import pulp
from joblib import Parallel, delayed, parallel_config
from typing import Dict, Callable, Any, List, Tuple
from numpy.typing import NDArray

from .elections import *
from .measurements import euclidean_cost_array, q_cost_array
from .utils import cost_array_to_ranking, ensure_dir


def generate_election_input(
    generator : Callable,
    generator_input : Dict[str, Any]
) -> Tuple[NDArray, NDArray, NDArray, NDArray, NDArray]:
    """
    Randomly creates a single preference profile, along with candidate positions/labels, and
    voter positions/labels. 
    
    Args:
        generator (Callable, Spatial): Object for generating 
            random preference profiles in metric space.
        generator_input (Dict[str, Any]): keyword arguments for input to generator.generate()

    Returns:
        profile (np.ndarray): Preference profile.
        candidate_positions (np.ndarray): Numpy matrix where each row encodes a candidate's
            position in the metric space.
        voter_positions (np.ndarray): Numpy matrix where each row encodes a voters position
            in the metric space.
        candidate_labels (NDArray): Length m array of group labels for voters.
        voter_labels (NDArray): Length n array of group labels for voters.
        
    Example Usage and Extra information:
    - A generator object is taken as one of the classes from spatial_generation.py, 
        please see the full documentation there for more specific parameter info.
        
    - Its purpose is to generate random voter/candidate positions in metric space, 
        and will generally take parameters which define the distributions to sample from.
        
    - generator_input, on the other hand, is used specifically for the .generate() method of 
        a generator object. Here it is given as a *single* dictionary containing
        the input voter and candidate group sizes to pass to the .generate() method.
    ```
    generator = GroupSpatial(...group spatial params ...)
    generator_input = {'voter_group_sizes': group_sizes, 'candidate_group_sizes': [m]}
    profile, candidate_positions, voter_positions, candidate_labels, voter_labels = 
        generate_election_input(generator, generator_input)
    ```
    """
    (profile,
     candidate_positions,
     voter_positions,
     candidate_labels,
     voter_labels) = generator.generate(
        **generator_input
    )
    return profile, candidate_positions, voter_positions, candidate_labels, voter_labels


####################################################################################################


def election_sample(
    profile : NDArray,
    candidate_positions : NDArray,
    voter_positions : NDArray,
    elections_dict : Dict[Callable, Dict[str, Any]],
    k : int
) -> Dict[str, NDArray]:
    """
    Conducts elections on a single input setting and records the results.
    
    Args:
        profile (np.ndarray): Preference profile.
        candidate_positions (np.ndarray): Numpy matrix where each row encodes a candidate's
            position in the metric space.
        voter_positions (np.ndarray): Numpy matrix where each row encodes a voters position
            in the metric space.
        elections_dict (dict[Callable Election, dict[str, Any]]): Election mechanism dictionary 
            where keys are election mechanisms and their values are dictionaries with any
            additional keyword arguments.
        k (int): Number of candidates to elect.

    Returns:
        winner_dict (dict[str, np.ndarray]): Dictionary with election names as keys and their 
            corresponding winners as values. Winners are shown by a length m boolean array 
            with True values representing winning candidates. 
            Querying candidate_positions[winners['STV'],:] for example gives
            the winning candidates mask.
            
        
    Example Usage and Extra information:        
    - The elections_dict is a dictionary where the keys are election mechanisms objects 
        from elections.py, and values are dictionaries with any parameters to use 
        when initializing that election class.
    ```
    elections_dict = {Borda : {}, STV : {'transfer_type' : 'weighted-fractional'}}
    winners = election_sample(
            profile,
            voter_positions,
            candidate_positions,
            elections_dict,
            k = 3
        )
    )
    ```
    """
    winner_dict = {}

    for election, params in elections_dict.items():
        if election.__name__ == "CommitteeVeto":
            # Form the multi-winner plurality veto profile.
            # Assuming Euclidean Distance here!!
            if 'q' in params:
                q = params['q']
            else:
                q = k
            
            cost_arr = euclidean_cost_array(voter_positions, candidate_positions)
            candidate_subsets = [set(top_k_cands) for top_k_cands in profile[:k,:].T]
            q_cost_arr = q_cost_array(q, cost_arr, candidate_subsets)
            q_profile = cost_array_to_ranking(q_cost_arr)
            elect_subset = election().elect(profile=q_profile, k=1)[0]
            elects = np.array(list(candidate_subsets[elect_subset]))
            
        else:
            try:
                elects = election(**params).elect(profile=profile, k=k)
            except pulp.apis.core.PulpSolverError:
                elects = np.zeros(k, dtype=int) - 1
                np.save("error_profile.npy", profile)
        
        winner_dict[election.__name__] = elects

    return winner_dict


####################################################################################################


def samples(
    s : int,
    generator : Callable, 
    elections_dict : Dict[Callable, Dict[str, Any]],
    generator_input : List[Dict[str, Any]],
    k : int,
    dim : int = 2,
    filename : str = None
):
    """
    For a given number of samples, generate election input, run elections on them,
    and record the results.

    Args:
        s (int): Number of samples.
        generator (Spatial): Spatial object for creating random preference profiles.
        elections_dict (dict[Callable Election, dict[str, Any]]): Election mechanism dictionary 
            where keys are election mechanisms and their values are dictionaries with any additional
            keyword arguments.
        generator_input (List[dict[str, Any]]): Dictionary for input settings to 
            generator.generate().
        k (int): Number of candidates to elect.
        dim (int, optional): Number of dimensions in a voter or candidate position 
            in the metric space, defaults to 2d.
        filename (str, optional): Filename to save results to, optional but if None results
            will not be saved.

    Returns:
        results_list (List[Dict[str, np.ndarray]]): List of dictionaries where each dictionary
            contains the results of the election sampling
            
            
    Example Usage and Extra information:
    - A generator object is taken as one of the classes from spatial_generation.py, 
        please see the full documentation there for more specific parameter info.
        
    - Its purpose is to generate random voter/candidate positions in metric space, 
        and will generally take parameters which define the distributions to sample from.
        
    - generator_input, on the other hand, is used specifically for the .generate() method of 
        a generator object. It is a *list* of dictionaries where each dictionary contains
        the input voter and candidate group sizes to pass to the .generate() method.
        For every set of input parameters in the list, a new experiment is performed --
        this is useful for experiments in which the sizes of voter/candidate groups are varied.
        
    - The elections_dict is a dictionary where the keys are election mechanisms objects 
        from elections.py, and values are dictionaries with any parameters to use 
        when initializing that election class.
    ```
    generator = GroupSpatial(...group spatial params ...)
    elections_dict = {Borda : {}, STV : {'transfer_type' : 'weighted-fractional'}}
    # Input for the generator.generate() method:
    generator_input = [{'voter_group_sizes': group_sizes, 'candidate_group_sizes': [m]}]
    sample_result_list = 
        samples(1000, generator, elections_dict, generator_input, k = 3, dim = 2, filename = f)
    )
    ```
    """
    results_list = []
    for gidx, gen_input in enumerate(generator_input):
        n = gen_input.get("n")
        m = gen_input.get("m")
        if n is None:
            n = sum(gen_input.get("voter_group_sizes"))
        if m is None:
            m = sum(gen_input.get("candidate_group_sizes"))
        

        result_dict = {election.__name__: np.zeros((s, m), dtype = bool)
                       for election in elections_dict.keys()}
        result_dict["voters"] = [np.zeros((s, n, dim))] * s
        result_dict["candidates"] = [np.zeros((s, m, dim))] * s
        result_dict["voter_labels"] = [np.zeros((s, n), dtype = int)] * s
        result_dict["candidate_labels"] = [np.zeros((s, m), dtype = int)] * s

        for i in range(s):
            (prof,
             cand_pos,
             voter_pos,
             c_labels,
             v_labels
             ) = generate_election_input(generator, gen_input)
            winner_dict = election_sample(prof, cand_pos, voter_pos, elections_dict, k)
            
            result_dict["voters"][i] = voter_pos
            result_dict["candidates"][i] = cand_pos
            result_dict["voter_labels"][i] = v_labels
            result_dict["candidate_labels"][i] = c_labels
            for name, idxs in winner_dict.items():
                winner_mask = np.zeros(m, dtype=bool)
                winner_mask[idxs.astype(np.int32)] = True
                result_dict[name][i] = winner_mask
                

        # Save results for this set of generator input parameters.
        if not filename is None:
            if len(generator_input) > 1:
                np.savez_compressed(ensure_dir(filename[:-4] + str(gidx) + filename[-4:]), **result_dict)
            else:
                np.savez_compressed(ensure_dir(filename), **result_dict)

        results_list.append(result_dict)

    return results_list


####################################################################################################


def parallel_samples(
    s : int,
    generator : Callable, 
    elections_dict : Dict[Callable, Dict[str, Any]],
    generator_input : List[Dict[str, Any]],
    k : int,
    dim : int = 2,
    cpu_count : int = 8,
    filename : str = None
):
    """
    For a given number of samples, sample elections with election_sample()
    and record the results. This version of the sampling function is indentical 
    to the previous aside from the fact that results are computed in parallel.

    Args:
        s (int): Number of samples.
        generator_list (Spatial): List of spatial objects for creating 
            random preference profiles.
        elections_dict (dict[Callable Election, dict[str, Any]]): Election mechanism dictionary 
            where keys are election mechanisms and their values are dictionaries with any additional
            keyword arguments.
        generator_input (List[dict[str, Any]]): Dictionary for input settings to 
            generator.generate().
        k (int): Number of candidates to elect.
        dim (int, optional): Number of dimensions in a voter or candidate position 
            in the metric space, defaults to 2d.
	    cpu_count (int, optional): Number of available cpus to use for processing. Defaults to 8.
        filename (str, optional): Filename to save results to, optional but if None results
            will not be saved.

    Returns:
        results_list (List[Dict[str, np.ndarray]]): List of dictionaries where each dictionary
            contains the results of the election sampling
            
            
    Example Usage and Extra information:
    - A generator object is taken as one of the classes from spatial_generation.py, 
        please see the full documentation there for more specific parameter info.
        
    - Its purpose is to generate random voter/candidate positions in metric space, 
        and will generally take parameters which define the distributions to sample from.
        
    - generator_input, on the other hand, is used specifically for the .generate() method of 
        a generator object. It is a *list* of dictionaries where each dictionary contains
        the input voter and candidate group sizes to pass to the .generate() method.
        For every set of input parameters in the list, a new experiment is performed --
        this is useful for experiments in which the sizes of voter/candidate groups are varied.
        
    - The elections_dict is a dictionary where the keys are election mechanisms objects 
        from elections.py, and values are dictionaries with any parameters to use 
        when initializing that election class.
    ```
    generator = GroupSpatial(...group spatial params ...)
    elections_dict = {Borda : {}, STV : {'transfer_type' : 'weighted-fractional'}}
    # Input for the generator.generate() method:
    generator_input = [{'voter_group_sizes': group_sizes, 'candidate_group_sizes': [m]}]
    sample_result_list = 
        samples(1000, generator, elections_dict, generator_input, k = 3, dim = 2, cpu_count = 8,
        filename = f)
    )
    ```
    """
    results_list = []
    for gidx, gen_input in enumerate(generator_input):
        n = gen_input.get("n")
        m = gen_input.get("m")
        if n is None:
            n = sum(gen_input.get("voter_group_sizes"))
        if m is None:
            m = sum(gen_input.get("candidate_group_sizes"))
        

        result_dict = {election.__name__: np.zeros((s, m), dtype = bool)
                       for election in elections_dict.keys()}
        result_dict["voters"] = [np.zeros((s, n, dim))] * s
        result_dict["candidates"] = [np.zeros((s, m, dim))] * s
        result_dict["voter_labels"] = [np.zeros((s, n), dtype = int)] * s
        result_dict["candidate_labels"] = [np.zeros((s, m), dtype = int)] * s
        
        # Generate input samples for all s samples.
        input_sample_list = []
        for i in range(s):
            (prof,
             cand_pos,
             voter_pos,
             c_labels,
             v_labels
             ) = generate_election_input(generator, gen_input)
            
            input_dict = {}
            input_dict['profile'] = prof
            input_dict['candidate_positions'] = cand_pos
            input_dict['voter_positions'] = voter_pos
            input_sample_list.append(input_dict)

            
            result_dict["candidates"][i] = cand_pos
            result_dict["voters"][i] = voter_pos
            result_dict["candidate_labels"][i] = c_labels
            result_dict["voter_labels"][i] = v_labels
        
        # Then run elections in parallel.
        parallel_results = Parallel(n_jobs=cpu_count, backend = 'loky')(
            delayed(election_sample)(**input_sample, elections_dict = elections_dict, k = k)
            for input_sample in input_sample_list
        )
         
        for i, winner_dict in enumerate(parallel_results):               
            for name, idxs in winner_dict.items():
                winner_mask = np.zeros(m, dtype=bool)
                winner_mask[idxs.astype(np.int32)] = True
                result_dict[name][i] = winner_mask
                

        if not filename is None:
            if len(generator_input) > 1:
                np.savez_compressed(ensure_dir(filename[:-4] + str(gidx) + filename[-4:]), **result_dict)
            else:
                np.savez_compressed(ensure_dir(filename), **result_dict)

        results_list.append(result_dict)

    return results_list




####################################################################################################


def parallel_with_precomputed_samples(
    precomputed_sample_list : List[Dict[str, Any]],
    elections_dict : Dict[Callable, Dict[str, Any]],
    n : int,
    m : int,
    k : int,
    cpu_count : int = 8,
    filename : str = None
):
    """
    For a set of precomputed election settings, run elections in parallel and 
    store the winning results.

    Args:
        precomputed_sample_list (List[Dict[str, Any]]): List of dictionaries where each dictionary
            has keys 'voters' giving voter positions, 'candidates' giving candidate positions, 
            and 'profile' giving the corresponding preference profile. 
        elections_dict (dict[Callable Election, dict[str, Any]]): Election mechanism dictionary 
            where keys are election mechanisms and their values are dictionaries with any additional
            keyword arguments.
        n (int): Number of voters
        m (int) : Number of candidates
        k (int): Number of candidates to elect.
	    cpu_count (int, optional): Number of available cpus to use for processing. Defaults to 8.
        filename (str, optional): Filename to save results to, optional but if None results
            will not be saved.

    Returns:
        results_dict ([Dict[str, np.ndarray]): Dictionary which
            contains the results of the election sampling

    """
    s = len(precomputed_sample_list)
    result_dict = {election.__name__: np.zeros((s, m), dtype = bool)
                       for election in elections_dict.keys()}
    
    parallel_results = None
    with parallel_config(backend = 'loky', n_jobs=cpu_count, inner_max_num_threads=1):
        parallel_results = Parallel()(
            delayed(election_sample)(**input_sample, elections_dict = elections_dict, k = k)
            for input_sample in precomputed_sample_list
        )
        
    for i, winner_dict in enumerate(parallel_results):               
        for name, idxs in winner_dict.items():
            winner_mask = np.zeros(m, dtype=bool)
            winner_mask[idxs.astype(np.int32)] = True
            result_dict[name][i] = winner_mask
            
    if not filename is None:
        np.savez_compressed(ensure_dir(filename), **result_dict)

    return result_dict
    
    

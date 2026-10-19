import numpy as np
import pandas as pd
import math
from metric_voting import *

np.random.seed(918717)
filename = 'data/four_bloc/four_winner/example_ineff.npz'

# Choose number of voters n and the number of candidates m
n = 1000
m = 20

# Choose the number of winners for the election
k = 4

# Means for each of the 4 Normal distributions
means = [[-2, 0], [2, 0], [0, 2], [0, -2]]
stds = [1/3, 1/3, 1/3, 1/3]  # Standard deviations for each Normal
voter_group_sizes = [250, 250, 250, 250]  # Group Sizes

# Create a list of voter parameters -- with each set of parameters being a dict
voter_params = [{'loc': None, 'scale': None, 'size': 2} for _ in range(len(voter_group_sizes))]
for i,mean in enumerate(means):
    voter_params[i]['loc'] = mean

for i,std in enumerate(stds):
    voter_params[i]['scale'] = std
    
# define the single set of candidate paramters
candidate_params = [{'low': -3, 'high': 3, 'size': 2}]

# define a distance function between voters and candidates
distance = lambda point1, point2: np.linalg.norm(point1 - point2)

# Create the group spatial generator object!
generator = GroupSpatial(
    n_voter_groups = 4,
    n_candidate_groups = 1,
    voter_dist_fns = [np.random.normal]*len(voter_group_sizes),
    voter_dist_fn_params = voter_params,
    candidate_dist_fns = [np.random.uniform],
    candidate_dist_fn_params = candidate_params,
    distance_fn = distance
)


# Now each call .generate() will make a profile with random candidate and voter positions
profile, \
candidate_positions, \
voter_positions, \
candidate_labels, \
voter_labels = generator.generate(voter_group_sizes = voter_group_sizes, 
                                            candidate_group_sizes = [m])



# Collect results for different blocs and their representatives
result_dict = {'voters' : voter_positions, 'candidates' : candidate_positions}
cst_array = euclidean_cost_array(voter_positions, candidate_positions)
elections_dict = {Bloc:{}, Borda:{}, STV:{'transfer_type' : 'weighted-fractional'}}

for E, params in elections_dict.items():
    e_dict = {}
    name = E.__name__
    winners = E(**params).elect(profile,k)
    winner_mask = np.zeros(m, dtype=bool)
    winner_mask[winners] = True
    e_dict['winners'] = winner_mask
    
    bloc_label = 1
    
    # single group ineff
    group_mask = voter_labels
    group_rep_mask = None
    group_ineff = -1
    
    for lab in np.unique(group_mask):
        lab_group_ineff = group_inefficiency(cst_array, winners, group_mask, lab)
        if lab_group_ineff > group_ineff:
            group_ineff = lab_group_ineff
            group_reps = proportional_assignment(cst_array[winners, :], group_mask, lab, k)
            group_rep_mask = np.zeros(m, dtype=bool)
            group_rep_mask[winners[group_reps]] = True
    
    
    e_dict['group'] = {
            'labels' : group_mask,
            'reps' : group_rep_mask,
            'ineff' : group_ineff
    }

    # overall group ineff
    overall_mask = np.ones(n, dtype = int)
    overall_reps = proportional_assignment(cst_array[winners, :], overall_mask, bloc_label, k)
    overall_rep_mask = np.zeros(m, dtype=bool)
    overall_rep_mask[winners[overall_reps]] = True
    overall_ineff = group_inefficiency(cst_array, winners, overall_mask, bloc_label)
    e_dict['overall'] = {
        'labels' : overall_mask,
        'reps' : overall_rep_mask,
        'ineff' : overall_ineff
    }
        
    heuristic_bloc = heuristic_worst_bloc(cst_array, winners, max_size = None)
    heuristic_mask = np.zeros(n, dtype = int)
    heuristic_mask[heuristic_bloc] = 1
        
    heuristic_reps = proportional_assignment(cst_array[winners, :], heuristic_mask, bloc_label, k)
    heuristic_rep_mask = np.zeros(m, dtype=bool)
    heuristic_rep_mask[winners[heuristic_reps]] = True
    heuristic_ineff = group_inefficiency(cst_array, winners, heuristic_mask, bloc_label)
    e_dict['heuristic'] = {
        'labels' : heuristic_mask,
        'reps' : heuristic_rep_mask,
        'ineff' : heuristic_ineff
    }
    
    result_dict[name] = e_dict
    
    
np.savez_compressed(ensure_dir(filename), **result_dict)
    

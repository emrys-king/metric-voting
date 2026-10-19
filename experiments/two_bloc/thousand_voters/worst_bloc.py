import numpy as np
from metric_voting.measurements import *


np.random.seed(918717)
input_file = 'data/two_bloc/thousand_voters/samples.npz'
output_file = 'data/two_bloc/thousand_voters/worst_bloc.npz'

loaded_data = np.load(input_file)
result_dict = {key: loaded_data[key] for key in loaded_data.files}
voters = result_dict['voters']
candidates = result_dict['candidates']

worst_bloc_dict = {
    'voters' : voters,
    'candidates' : candidates
}

elections = [_ for _ in result_dict.keys() if _ not in 
             ['voters', 'candidates', 'voter_labels', 'candidate_labels']]
n_samples = result_dict[elections[0]].shape[0]

for e in elections:
    worst_bloc_samples = []
    for j in range(n_samples):
        V = voters[j]
        C = candidates[j]
        cst_array = euclidean_cost_array(V,C)
        winner_indices = np.where(result_dict[e][j])[0]
        worst_bloc = heuristic_worst_bloc(cst_array, winner_indices, max_size = None)
        worst_mask = np.zeros(V.shape[0], dtype = bool)
        worst_mask[worst_bloc] = True
        worst_bloc_samples.append(worst_mask)
        
    worst_bloc_dict[e] = np.array(worst_bloc_samples)
    
    
np.savez_compressed(ensure_dir(output_file), **worst_bloc_dict)

import os
import numpy as np
from metric_voting.spatial_generation import *
from metric_voting.election_sampling import (samples,
                                             parallel_samples,
                                             parallel_with_precomputed_samples)
from metric_voting.elections import *
from metric_voting.measurements import euclidean_cost_array
from metric_voting.utils import cost_array_to_ranking, ensure_dir


# Prevent extra threading:
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

# Job number
job_number = int(os.getenv("SGE_TASK_ID"))
sample_number = job_number - 1

# Choose number of voters n
# And the number of candidates m
n = 1000
m = 20

# And the number of winners for the election
k = 5

# NOTE that this requires precomputed samples that may be obtained by running sample_collect.py
# Load-precomputed samples:
f = 'data/four_bloc/five_winner/samples.npz'
loaded_data = np.load(f)
result_dict = {key: loaded_data[key] for key in loaded_data.files}
n_samples = result_dict['voters'].shape[0]

precomputed_sample_list = []
for i in range(n_samples):
    voter_positions = result_dict['voters'][i]
    candidate_positions = result_dict['candidates'][i]
    cost_arr = euclidean_cost_array(voter_positions, candidate_positions)
    profile = cost_array_to_ranking(cost_arr)
    sample_dict = {'profile' : profile,
                   'candidate_positions' : candidate_positions,
                   'voter_positions' : voter_positions
                }
    precomputed_sample_list.append(sample_dict)

# Define elections
elections_dict = {
    ChamberlinCourantTiebreak:{}
}

# set the seed for deterministic results:
np.random.seed(918717)

# and sample from them
f = 'data/four_bloc/five_winner/tiebreak/samples_chamberlin' + str(sample_number) + '.npz'


start = sample_number * 2500
end = start + 2500

result_list = parallel_with_precomputed_samples(
    precomputed_sample_list[start:end],
    elections_dict,
    n,
    m,
    k,
    cpu_count = 16,
    filename = f
)
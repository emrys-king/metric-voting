import numpy as np
from metric_voting.spatial_generation import *
from metric_voting.election_sampling import samples, parallel_samples
from metric_voting.elections import *
from metric_voting.utils import ensure_dir


# Choose number of voters n
# And the number of candidates m
n = 1000
m = 20

# And the number of winners for the election
k = 4 

# Means for each of the 2 Gaussian distributions
means = [[0, -2], [0, 2]]
stds = [1, 1]  # Standard deviations for each Gaussian
group_sizes = [500,500]  # Group Sizes

voter_params = [{'loc': None, 'scale': None, 'size': 2} for _ in range(len(group_sizes))]
for i,mean in enumerate(means):
    voter_params[i]['loc'] = mean

for i,std in enumerate(stds):
    voter_params[i]['scale'] = std
    
candidate_params = [{'low': -3, 'high': 3, 'size': 2}]

distance = lambda point1, point2: np.linalg.norm(point1 - point2)

generator = GroupSpatial(
    n_voter_groups = 2,
    n_candidate_groups = 1, 
    voter_dist_fns = [np.random.normal]*len(group_sizes),
    voter_dist_fn_params = voter_params,
    candidate_dist_fns = [np.random.uniform],
    candidate_dist_fn_params = candidate_params,
    distance_fn = distance
)

# Define elections
# NOTE: Here the Chamberlin and Monroe elections do not use random tiebreaking -- that is
# implemented and run in a separate tiebreak experiment (in the tiebreak folder)
elections_dict = {
    SNTV:{},
    Bloc:{},
    Borda:{},
    STV:{'transfer_type' : 'weighted-fractional'},
    ChamberlinCourant:{'solver' : 'HiGHS', 'log_path' : 'experiments/overlapping/cc.log'},
    GreedyCC:{},
    Monroe:{'solver' : 'HiGHS', 'log_path' : 'experiments/overlapping/monroe.log'},
    GreedyMonroe:{}, 
    PluralityVeto:{},
    CommitteeVeto:{'q':k}, 
    ExpandingApprovals: {},
    SMRD:{},
    OMRD:{},
    DMRD:{'rho': 0.5}
}


# Number of samples to use
n_samples = 100

# set the seed for deterministic results:
np.random.seed(918717)

# and sample from them
f = 'data/overlapping/samples.npz'

generator_input = [
    {'voter_group_sizes': group_sizes,
     'candidate_group_sizes': [m]}
]

result_list = parallel_samples(
    n_samples,
    generator,
    elections_dict,
    generator_input,
    k,
    dim = 2,
    cpu_count = 8,
    filename = f
)
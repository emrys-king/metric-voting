import numpy as np
from metric_voting.spatial_generation import *
from metric_voting.election_sampling import samples
from metric_voting.elections import *
from metric_voting.utils import ensure_dir


# Choose number of voters n
# And the number of candidates m
n = 1000
m = 20

# And the number of winners for the election
k = 4 

def sample_triangle(loc, base, height, direction):
    x = np.random.uniform(low = -base/2, high = base/2)
    if x > 0:
        y = direction * np.random.uniform(low = 0, high = -2 * height/base * x + height) + loc
    if x <= 0:
        y = direction * np.random.uniform(low = 0, high = 2 * height/base * x + height) + loc
    return np.array([x,y])

# Means and standard deviations for each of the two voter distributions
voter_group_sizes = [500,500]  # Group Sizes
voter_params = [
    {'loc' : 3, 'base' : 3, 'height' : 3, 'direction' : -1},
    {'loc' : -3, 'base' : 3, 'height' : 3, 'direction' : 1}
]
    
# Define the single set of candidate paramters
candidate_params = [{'low': -3, 'high': 3, 'size': 2}]

# define a distance function between voters and candidates
distance = lambda point1, point2: np.linalg.norm(point1 - point2)

# Create the group spatial generator object!
generator = GroupSpatial(n_voter_groups = 2, n_candidate_groups = 1,
                       voter_dist_fns = [sample_triangle, sample_triangle],
                       voter_dist_fn_params = voter_params,
                       candidate_dist_fns = [np.random.uniform],
                       candidate_dist_fn_params = candidate_params,
                       distance_fn = distance)

# Define elections
# Define elections
elections_dict = {
    SNTV:{},
    Bloc:{},
    Borda:{},
    STV:{'transfer_type' : 'weighted-fractional'},
    ChamberlinCourant:{'solver' : 'GUROBI_CMD', 'log_path' : 'experiments/hourglass/cc.log'},
    GreedyCC:{},
    Monroe:{'solver' : 'GUROBI_CMD', 'log_path' : 'experiments/hourglass/monroe.log'},
    GreedyMonroe:{}, 
    #PAV:{'solver' : 'GUROBI_CMD', 'log_path' : 'experiments/hourglass/pav.log'},
    PluralityVeto:{},
    CommitteeVeto:{'q':k}, 
    ExpandingApprovals: {},
    SMRD:{},
    OMRD:{},
    DMRD:{'rho': 0.5}
}


# Number of samples to use
n_samples = 10000

# set the seed for deterministic results:
np.random.seed(918717)

# and sample from them
f = 'data/hourglass/samples.npz'

generator_input = [
    {'voter_group_sizes': voter_group_sizes,
     'candidate_group_sizes': [m]}
]

result_list = samples(
    n_samples,
    generator,
    elections_dict,
    generator_input,
    k,
    dim = 2,
    filename = f
)
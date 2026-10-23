# Metric Voting :ballot_box:

A computational implementation of multi-winner social choice mechanisms in a metric space setting. 
We assume that voters and candidates are embedded in some latent metric space and provide
code for randomly generating voter and candidate positions given input distributions. 
We also implement various multi-winner election mechanisms and
provide tools for measuring voter representation in their output. 
Please see [VoteKit](https://github.com/mggg/VoteKit) as well, especially if metric assumptions aren't needed.

## Social Choice Mechanisms Supported:
We provide implementations of the following multi-winner election mechansisms. For 
detailed information and definitions please refer to [1, 2].

1. SNTV (Plurality)
2. Bloc
3. Borda
4. STV
5. Chamberlin Courant
6. A greedy Chamberlin Courant approximation algorithm
8. Monroe
9. A greedy Monroe approximation algorithm
10. Two multi-winner extensions of Plurality Veto [3]
11. Expanding Approvals [4]
12. Multiple variations on multi-winner Random Dictator

## Getting Started
This repository is written in python and uses uv to manage its dependencies.
If you don't currently have uv on your system instructions for installing it 
may be found [here](https://docs.astral.sh/uv/getting-started/installation/).

Once uv is installed, run the following command to set up your virtual environment
and install all dependencies:
```
uv sync
```
From there, any script may be run safely with the `uv run` prefix. As an example:
```
uv run ./experiments/four_bloc/five_winner/sample_collect.py
```
will generate the data for Experiment 6, four blocs of equal size in an election with five winners.

Chamberlin-Courant and Monroe use PuLP with HiGHS by default. Their tiebreak versions
use repeated SCIP solves to enumerate all distinct optimal winning committees and select
one uniformly. Each committee receives equal probability regardless of its number of
optimal voter assignments, changing the weighting of the original Gurobi implementation.
Enumeration has no committee-count cap and can be expensive when many committees tie.
A solver limit raises an error instead of sampling an incomplete set; the time limit
covers optimization and enumeration together. No commercial solver license is needed.
SCIP tuning files use its `.set` format, rather than Gurobi's `.prm` format.

## Examples
Examples for generating metric space settings, performing elections, computing measurements, 
and visualizing results can be found within jupyter notebooks supplied in the `examples` folder.
These show how to interface with the spatial generation tools in `spatial_generation.py`, 
the mechanisms in `elections.py`, and the measures of representation in `measurements.py`.

## Experimentation
All code used to run and collect data for our experiments may be found nested within the `experiments` folder. 
Each specific experiment contains files:
  1. `example_ineff.py` for running and collecting data on a single simple example setting.
  2. `sample_collect.py` for sampling and recording results from generated elections.
  3. `plot_distribution.py` for plotting KDE's visualizing the samples in metric space.

These may be run to re-create our experimental results. To run a specific script, use `uv run path/to/script.py` as detailed in [Getting Started](#getting-started).

Once scripts have been run to generate necessary data, additional plotting may be created within `examples/4-experiment-plotting.ipynb`. We note that our standard versions of Chamberlin Courant and Monroe deterministically choose winners in the case of ties. The specialized ChamberlinCourantTiebreak and MonroeTiebreak election classes randomly break ties using SCIP. These are therefore presented as separate experiments within nested `tiebreak/` folders for each experiment respectively.

## References
[1] Faliszewski, Piotr, et al. "Multiwinner voting: A new challenge for social choice theory." Trends in computational social choice 74.2017 (2017): 27-47.

[2] Elkind, Edith, et al. "What do multiwinner voting rules do? An experiment over the two-dimensional euclidean domain." Proceedings of the AAAI Conference on Artificial Intelligence. Vol. 31. No. 1. 2017.

[3] Kizilkaya, Fatih Erdem, and David Kempe. "Plurality veto: A simple voting rule achieving optimal metric distortion." arXiv preprint arXiv:2206.07098 (2022).

[4] Kalayci, Yusuf, David Kempe, and Vikram Kher. "Proportional representation in metric spaces and low-distortion committee selection." Proceedings of the AAAI Conference on Artificial Intelligence. Vol. 38. No. 9. 2024.

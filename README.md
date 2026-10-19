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

## Getting Started (need to fix!)
This repository is written in python and uses poetry to manage its dependencies.
If you don't currently have poetry installed on your system instructions for downloading it 
may be found [here](https://python-poetry.org/docs/). Once poetry has been installed, 
you may download the required dependencies by simply running the following commands:

```
poetry install
```
Then to activate the virtual environment it creates:
```
poetry shell
```

## Examples
Examples for generating metric space settings, performing elections, computing measurements, 
and visualizing results can be found within jupyter notebooks supplied in the `examples` folder.
These show how to interface with the spatial generation tools in `spatial_generation.py`, 
the mechanisms in `elections.py`,
and the measures of representation in `measurements.py`.

## Experimentation
All code used to run and collect data for our experiments may be found nested within the `experiments` folder. 
Each specific experiment contains files:
  1. `example_ineff.py` for running and collecting data on a single simple example setting.
  2. `sample_collect.py` for sampling and recording results from generated elections.
  3. `plot_distribution.py` for plotting KDE's visualizing the samples in metric space.

These may be run to re-create our experimental results. To run, you must call the script from the top level working-directory, using `poetry run python` before. For example, to run Experiment 3, use:
```
poetry run python experiments/two_bloc/10k_voters_200_cands/example_ineff.py
poetry run python experiments/two_bloc/10k_voters_200_cands/sample_collect.py
poetry run python experiments/two_bloc/10k_voters_200_cands/plot_distribution.py
```

Once they have been run, additional plotting may be created within `examples/4-experiment-plotting.ipynb`. We note that 
our standard versions of Chamberlin Courant and Monroe deterministically choose winners in the case
of ties. We also have specialized ChamberlinCourantTiebreak and MonroeTiebreak election classes 
which randomly break ties, however, running experiments for these elections requires a 
license with gurobi. These are therefore presented as separate experiments within nested 
`tiebreak/` folders for each experiment respectively.

## References
[1] Faliszewski, Piotr, et al. "Multiwinner voting: A new challenge for social choice theory." Trends in computational social choice 74.2017 (2017): 27-47.

[2] Elkind, Edith, et al. "What do multiwinner voting rules do? An experiment over the two-dimensional euclidean domain." Proceedings of the AAAI Conference on Artificial Intelligence. Vol. 31. No. 1. 2017.

[3] Kizilkaya, Fatih Erdem, and David Kempe. "Plurality veto: A simple voting rule achieving optimal metric distortion." arXiv preprint arXiv:2206.07098 (2022).

[4] Kalayci, Yusuf, David Kempe, and Vikram Kher. "Proportional representation in metric spaces and low-distortion committee selection." Proceedings of the AAAI Conference on Artificial Intelligence. Vol. 38. No. 9. 2024.

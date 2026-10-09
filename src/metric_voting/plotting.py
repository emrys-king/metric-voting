import sys
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import seaborn as sns
from numpy.typing import NDArray
from typing import Callable, Dict, Tuple, Any, List, Optional, Union

from metric_voting.elections import *
from metric_voting.measurements import *


####################################################################################################


def plot_winner_distribution(
    results : Dict[str, NDArray],
    fig_params : Dict[str, Any],
    xlim : List[float],
    ylim : List[float],
    colors : List[str],
    sample_fraction : float = 0.1,
    plot_kde : bool = True,
    kde_sample_fraction : float = 1,
    random_seed : int = None,
    output_file : str = None
):
    """
    Plots the distribution of winners across election methods.
    
    Args:
        results (dict[str, np.ndarray]): Dictionary with strings as keys and their corresponding
            np array datasets as values. In every result dictionary, 
            the voters' data points should be given for the 
            key 'voters' as an array of shape (s, n, dim) where s is the number of random samples,
            n is the number of voters in each sample, and dim is the number of dimensions in the 
            metric space. Likewise, 'candidates' should specify a (s, m, dim) array of candidate 
            data points and 'labels' should specify a (s, m) array of group labels for 
            each candidate sample. 
            
            Then, For each election type in the data, the result dictionary should have a 
            (key, value) pairs as (election name, subset of the candidate array) plot.
            
        fig_params (dict[str, Any]): Dictionary with figure parameters.    
        
        xlim (list[float]): List of two floats specifying the x-axis limits for the plot.
        
        ylim (list[float]): List of two floats specifying the y-axis limits for the plot.
            
        colors (list[str]): Length 3 list of colors to use for voters, candidates, 
            and winners respectively. Colors should be specified in hex format. 

        plot_kde (bool, default=True): whether or not to generate KDE plots, which can take a while to run.
            
        sample_fraction (float, optional): Fraction of the data to use for KDE plots.
            Otherwise plotting all the data takes a while to run without much added benefit.
            
        kde_sample_fraction (float, optional): Portion of the sample fraction to devote to 
            kde plots.
            
        random_seed (int, optional): Seed for deterministic sampling results.
            
        output_file (str, optional): Filepath to save the plot to. If None, the plot will
            be displayed but not saved.
    """
    elections = [_ for _ in results.keys() if _ not in 
                 ['voters', 'candidates', 'voter_labels', 'candidate_labels']]
    voter_color = colors[0]
    candidate_color = colors[1]
    winner_color = colors[2]
    
    fig, axes = plt.subplots(len(elections) + 1, 3, **fig_params)
    for i, ax in enumerate(axes.flat):
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_xlabel('')
        ax.set_ylabel('')
    
    # Index for the sample used in the example plots:
    s = results['voters'].shape[0]
    np.random.seed(random_seed)
    example_idx = np.random.randint(s)
    
    # Gather data for plotting:
    voter_example = results['voters'][example_idx]
    voter_stack = pd.DataFrame(np.vstack(results['voters']), columns = ['x','y'])
    # Use a smaller sample for KDE plots (otherwise takes a while to run without much added benefit)
    voter_stack_sample = voter_stack.sample(frac=sample_fraction, random_state=random_seed)
    if plot_kde:
        kde_voter_stack_sample = voter_stack_sample.sample(frac = kde_sample_fraction,
                                                           random_state = random_seed)

    candidate_example = results['candidates'][example_idx]
    candidate_stack = pd.DataFrame(np.vstack(results['candidates']), columns = ['x','y'])
    # Use a smaller sample for KDE plots (otherwise takes a while to run without much added benefit)
    candidate_stack_sample = candidate_stack.sample(frac=sample_fraction, random_state=random_seed)
    if plot_kde:
        kde_candidate_stack_sample = candidate_stack_sample.sample(frac = kde_sample_fraction,
                                                                   random_state = random_seed)

    # Set x and y limits for scatter and example plots:
    '''
    epsilon = 0.5
    ymin = np.min((voter_stack.loc[:,'y'].min(), candidate_stack.loc[:,'y'].min()))
    ymax = np.max((voter_stack.loc[:,'y'].max(), candidate_stack.loc[:,'y'].max()))
    xmin = np.min((voter_stack.loc[:,'x'].min(), candidate_stack.loc[:,'x'].min()))
    xmax = np.max((voter_stack.loc[:,'x'].max(), candidate_stack.loc[:,'x'].max()))

    scatter_xlim = [xmin - epsilon,xmax + epsilon]
    scatter_ylim = [ymin - epsilon,ymax + epsilon]
    scatter_xlim = [-3 - epsilon,3 + epsilon]
    scatter_ylim = [-3 - epsilon,3 + epsilon]

    example_xlim = [xmin - epsilon,xmax + epsilon]
    example_ylim = [ymin - epsilon,ymax + epsilon]
    example_xlim = [-3 - epsilon,3 + epsilon]
    example_ylim = [-3 - epsilon, 3 + epsilon]
    '''
    
    # Plot the voter and candidate KDE distributions:
    if plot_kde:
        sns.kdeplot(data=kde_voter_stack_sample, x='x', y='y', color = voter_color, fill=False,
                    thresh=0.1, levels=10, alpha = 1, ax = axes[0][0])
        sns.kdeplot(data=kde_candidate_stack_sample, x='x', y='y', color = candidate_color, fill=False,
                    thresh=0.1, levels=10, alpha = 0.9, ax = axes[0][0])
        axes[0][0].set_xlim(xlim)
        axes[0][0].set_ylim(ylim)
        axes[0][0].set_title('KDE')
        axes[0][0].set_ylabel('')
        axes[0][0].set_xlabel('')

    # Plot the voter and candidate scatter distributions:
    axes[0][1].scatter(voter_stack_sample.iloc[:,0], voter_stack_sample.iloc[:,1],
                    facecolors = voter_color, edgecolors = 'none', alpha = 0.3, s = 5)
    axes[0][1].scatter(candidate_stack_sample.iloc[:,0], candidate_stack_sample.iloc[:,1],
                    facecolors = candidate_color, edgecolors = 'none', alpha = 0.025, s = 5)
    axes[0][1].set_xlim(xlim)
    axes[0][1].set_ylim(ylim)
    axes[0][1].set_title('Scatter')

    # Plot an example voter, candidate setting.
    axes[0][2].scatter(voter_example[:,0], voter_example[:,1],
                    facecolors = voter_color, edgecolors = 'none', alpha = 0.9, s = 30)
    axes[0][2].scatter(candidate_example[:,0], candidate_example[:,1],
                    facecolors = candidate_color, edgecolors = 'none', alpha = 0.9, s = 30)
    axes[0][2].set_xlim(xlim)
    axes[0][2].set_ylim(ylim)
    axes[0][2].set_title('Example')

    # Plot the winner distributions for each election method:
    Candidates = results['candidates']
    for i,name in enumerate(elections):     
        ax_idx = i + 1
        
        # Gather data:
        winners = results[name]
        winner_example = Candidates[example_idx][winners[example_idx], :]
        winner_stack = pd.DataFrame(Candidates[winners], columns = ['x','y'])
        winner_stack_sample = winner_stack.sample(frac=sample_fraction, random_state=random_seed)
        if plot_kde:
            kde_winner_stack_sample = winner_stack_sample.sample(frac = kde_sample_fraction,
                                                                 random_state = random_seed)
        
        # Plot the KDE:
        if plot_kde:
            sns.kdeplot(data=kde_voter_stack_sample, x='x', y='y', color = voter_color, fill=False,
                        thresh=0.1, levels=10, alpha = 1, ax = axes[ax_idx][0])
            sns.kdeplot(data=kde_winner_stack_sample, x='x', y='y', color = winner_color, fill=False,
                        thresh=0.1, levels=10, alpha = 0.7, ax = axes[ax_idx][0])
        
        if name == 'ChamberlinCourant':
            display_name = 'CC'
        elif name == 'ExpandingApprovals':
            display_name = 'Expanding'
        elif name == 'Bloc':
            display_name = 'Block'
        else:
            display_name = name
            
        axes[ax_idx][0].set_ylabel(display_name, rotation = 0, ha='right', labelpad = 5)
        axes[ax_idx][0].set_xlabel('')
        
        # Plot the scatter:
        axes[ax_idx][1].scatter(voter_stack_sample.iloc[:,0], voter_stack_sample.iloc[:,1],
                    facecolors = voter_color, edgecolors = 'none', alpha = 0.3, s = 5)
        axes[ax_idx][1].scatter(winner_stack_sample.iloc[:,0], winner_stack_sample.iloc[:,1],
                        facecolors = winner_color, edgecolors = 'none', alpha = 0.1, s = 5)
        axes[ax_idx][1].set_xlim(xlim)
        axes[ax_idx][1].set_ylim(ylim)

        # Plot the example:
        axes[ax_idx][2].scatter(voter_example[:,0], voter_example[:,1],
                        facecolors = voter_color, edgecolors = 'none', alpha = 0.9, s = 30)
        axes[ax_idx][2].scatter(winner_example[:,0], winner_example[:,1],
                        facecolors = winner_color, edgecolors = 'none', alpha = 0.9, s = 30)
        axes[ax_idx][2].set_xlim(xlim)
        axes[ax_idx][2].set_ylim(ylim)
        

    legend_elements = [
        Line2D([0], [0], marker = 'o', color=voter_color, linestyle='None', label='voters'),
        Line2D([0], [0], marker = 'o', color=candidate_color, linestyle='None', label='candidates'),
        Line2D([0], [0], marker = 'o', color=winner_color, linestyle='None', label='winners')
    ]

    fig.legend(handles=legend_elements, loc='lower center', bbox_to_anchor=(0.5, 0.075), ncol=3)
    plt.subplots_adjust(wspace=0.1, hspace=0.1)
    if output_file is not None:
        plt.savefig(output_file, bbox_inches='tight')
        
    plt.show()


####################################################################################################


def plot_ineff_example(
    results : Dict[str, NDArray],
    fig_params : Dict[str, Any],
    colors : List[str],
    output_file : str = None
):
    """
    Plots an example inefficiencies for different voter blocs in a single election setting. 
    
    Args:
        results (dict[str, np.ndarray]): Dictionary with strings as keys and their corresponding
            np array datasets as values. In every result dictionary, the voters' data points should
            be given for the key 'voters' as an array of shape (n, dim) where
            n is the number of voters in each sample, and dim is the number of dimensions in the 
            metric space. Likewise, 'candidates' should specify a (m, dim) array of candidate 
            data points.
            
            Then, For each election type being used, the result dictionary should have a 
            (key, value) pairs as (election name, election-data), where the election-data
            given is a dictionary with information on specific voter blocs and their 
            representatives.
            
        fig_params (dict[str, Any]): Dictionary with figure parameters.    
            
        colors (list[str]): Length 3 list of colors to use for voters, candidates, 
            and winners respectively. Colors should be specified in hex format.
            
        output_file (str, optional): Filepath to save the plot to. If None, the plot will
            be displayed but not saved.
    """
    voter_color = colors[0]
    candidate_color = colors[1]
    winner_color = colors[2]
    bloc_color = colors[3]
    reps_color = colors[4]
    
    voter_pos = results['voters']
    candidate_pos = results['candidates']
    elections = [_ for _ in results.keys() if _ not in 
                 ['voters', 'candidates', 'voter_labels', 'candidate_labels']]
    methods = [method for method in results[elections[0]].keys() if method != 'winners']
    
    fig, axes = plt.subplots(len(methods), len(elections), **fig_params)
    for i, ax in enumerate(axes.flat):
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_xlabel('')
        ax.set_ylabel('')
    
    # Set x and y limits for scatter and example plots:
    epsilon = 0.1
    ymin = np.min((np.min(voter_pos[:,1]), np.min(candidate_pos[:,1])))
    ymax = np.max((np.max(voter_pos[:,1]), np.max(candidate_pos[:,1])))
    xmin = np.min((np.min(voter_pos[:,0]), np.min(candidate_pos[:,0])))
    xmax = np.max((np.max(voter_pos[:,0]), np.max(candidate_pos[:,0])))
    xlim = [xmin - epsilon,xmax + epsilon]
    ylim = [ymin - epsilon,ymax + epsilon]
    xlim = [-3 - epsilon,3 + epsilon]
    ylim = [-3 - epsilon,3 + epsilon]

    # Plot the winner distributions for each election method:
    for i,name in enumerate(elections):
        e_dict = results[name]
        winners = e_dict['winners']
        other_candidates = ~winners
        
        if name == 'Bloc':
            axes[0][i].set_title('Block')
        else:
            axes[0][i].set_title(name)

        methods = [method for method in e_dict.keys() if method != 'winners']
        for j, method in enumerate(methods):
            method_results = e_dict[method]
            
            # The specific bloc to show results for should be given as input in the result dict.
            bloc_label = method_results.get('bloc_label')
            # If they aren't, the default label is 1
            if bloc_label is None:
                bloc_label = 1
            
            bloc = np.where(method_results['labels'] == bloc_label)[0]
            other_voters = np.where(method_results['labels'] != bloc_label)[0]
            reps = np.where(method_results['reps'] == True)[0]
            other_winners = np.array([w for w in np.where(winners)[0] if w not in reps], 
                                     dtype = np.int32)
            
            axes[j][i].scatter(voter_pos[other_voters,0], voter_pos[other_voters,1],
                            facecolors = voter_color, edgecolors = 'none', alpha = 0.9, s = 20)
            axes[j][i].scatter(voter_pos[bloc,0], voter_pos[bloc,1],
                        facecolors = bloc_color, edgecolors = 'none', alpha = 1, s = 20)
            axes[j][i].scatter(candidate_pos[other_candidates,0], candidate_pos[other_candidates,1],
                        facecolors = candidate_color, edgecolors = 'black', alpha = 0.9, s = 50)
            axes[j][i].scatter(candidate_pos[other_winners,0], candidate_pos[other_winners,1],
                            facecolors = winner_color, edgecolors = 'black', alpha = 1, s = 70)
            axes[j][i].scatter(candidate_pos[reps,0], candidate_pos[reps,1],
                        facecolors = reps_color, edgecolors = 'black', alpha = 1, s = 100, 
                        marker = '^')
            axes[j][i].set_xlim(xlim)
            axes[j][i].set_ylim(ylim)

            if i == 0:
                if method == 'heuristic':
                    method = 'Worst-Bloc'
                axes[j][i].set_ylabel(method.title())
                #axes[j][i].set_xlabel('')
                
            axes[j][i].text(1, 1,
                            str(np.round(method_results['ineff'], 2)),
                            transform = axes[j][i].transAxes,
                            verticalalignment = 'top',
                            horizontalalignment = 'right',    
                        )
        

    legend_elements = [
        Line2D([0], [0], marker = 'o', color=voter_color, linestyle = 'None', label='voters'),
        Line2D([0], [0], marker = 'o', color=bloc_color, linestyle = 'None', label='voter bloc'),
        Line2D([0], [0], marker = 'o', color=candidate_color, linestyle = 'None', label='candidates',
               markeredgecolor = 'black'),
        Line2D([0], [0], marker = 'o', color=winner_color, linestyle = 'None', label='winners', 
               markeredgecolor = 'black'),
        Line2D([0], [0], marker = '^', color=reps_color, markeredgecolor = 'black', 
               linestyle = 'None', label='representatives'),
        ]

    fig.legend(handles=legend_elements, loc='lower center', bbox_to_anchor=(0.5, 0.014), ncol=3)
    plt.subplots_adjust(wspace=0.1, hspace=0.1)
    if output_file is not None:
        plt.savefig(output_file, bbox_inches='tight')
        
    plt.show()
    
    
####################################################################################################
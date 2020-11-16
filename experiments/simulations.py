#!/usr/bin/env python
# coding: utf-8

# In[1]:


"""
This script runs simulations reported in our paper Confidence Intervals for Policy Evaluation in Adaptive Experiments (https://arxiv.org/abs/1911.02768)
"""

import sys
# sys.path.insert(0, "/home/rhzhan/adaptive-confidence-intervals/")
from time import time
from sys import argv
from random import choice
import pickle
import os
import numpy as np
import pandas as pd
from adaptive_CI.experiments import run_mab_experiment
from adaptive_CI.compute import stick_breaking
from adaptive_CI.saving import *
from adaptive_CI.inference import *
from adaptive_CI.weights import *

import os
import subprocess
from time import time
from os.path import dirname, realpath, join, exists
from os import makedirs, chmod
from getpass import getuser


# magics removed
# magics removed


# In[2]:


def on_sherlock():
    """ Checks if running locally or on sherlock """
    return 'GROUP_SCRATCH' in os.environ


def get_sherlock_dir(project, *tail, create=True):
    """
    Output consistent folder name in Sherlock.
    If create=True and on Sherlock, also makes folder with group permissions.
    If create=True and not on Sherlock, does not create anything.

    '/scratch/groups/athey/username/project/tail1/tail2/.../tailn'.

    >>> get_sherlock_dir('adaptive-inference')
    '/scratch/groups/athey/adaptive-inference/vitorh'

    >>> get_sherlock_dir('toronto')
    '/scratch/groups/athey/toronto/vitorh/'

    >>> get_sherlock_dir('adaptive-inference', 'experiments', 'exp_out')
    '/scratch/groups/athey/adaptive-inference/vitorh/experiments/exp_out'
    """
    base = join("/", "scratch", "groups", "athey", project, getuser())
    path = join(base, *tail)
    if not exists(path) and create and on_sherlock():
        makedirs(path, exist_ok=True)
        # Correct permissions for the whole directory branch
        chmod_path = base
        chmod(base, 0o775)
        for child in tail:
            chmod_path = join(chmod_path, child)
            chmod(chmod_path, 0o775)
    return path


# In[3]:


dfs = []


# In[8]:


# ----------------------------------------------------
num_sims = 50 if on_sherlock() else 5

# Read DGP specification
noise_func = 'uniform'
truths = {
    'nosignal': np.array([1., 1., 1.]),
    'lowSNR': np.array([.9, 1., 1.1]),
    'highSNR': np.array([.5, 1., 1.5])
}

results_list = []
start_time = time()


# ----------------------------------------------------
# Run simulations
for s in range(num_sims):
    if (s+1) % 10 == 0:
        print(f'Running simulation {s+1}/{num_sims}')

    """ Experiment configuration """
    T = choice([1000, 5000, 10000, 50000, 100_000])  # number of samples
    experiment = choice(list(truths.keys()))
    truth = truths[experiment]
    K = len(truth)  # number of arms
    initial = 5  # initial number of samples of each arm to do pure exploration
    floor_start = 1/K
    floor_decay = choice([.25, .5, .6, .7, .8, .9, .99])
    exploration = 'TS'
    noise_scale = 1.0

    """ Generate data """
    if noise_func == 'uniform':
        noise = np.random.uniform(-noise_scale, noise_scale, size=(T, K))
        R = noise_scale * 2
    else:
        noise = np.random.exponential(noise_scale, size=(T, K)) - noise_scale
        R = -np.log(0.001) * noise_scale
    ys = truth + noise

    """ Run experiment """
    data = run_mab_experiment(
        ys,
        initial=initial,
        floor_start=floor_start,
        floor_decay=floor_decay,
        exploration=exploration)

    probs = data['probs']
    rewards = data['rewards']
    arms = data['arms']

    """ Compute AIPW scores """
    muhat = np.row_stack([np.zeros(K), sample_mean(rewards, arms, K)[:-1]])
    scores = aw_scores(rewards, arms, probs, muhat)

    """ Compute weights """
    # Two-point allocation rate
    twopoint_ratio = twopoint_stable_var_ratio(e=probs, alpha=floor_decay)
    twopoint_ratio_old = twopoint_stable_var_ratio_old(probs, floor_start, floor_decay)
    twopoint_h2es = stick_breaking(twopoint_ratio)
    twopoint_h2es_old = stick_breaking(twopoint_ratio_old)
    wts_twopoint = np.sqrt(np.maximum(0., twopoint_h2es * probs))
    wts_twopoint_old = np.sqrt(np.maximum(0., twopoint_h2es_old * probs))

    # Other weights: lvdl(constant allocation rate), propscore and uniform
    wts_lvdl = np.sqrt(probs)
    wts_propscore = probs
    wts_uniform = np.ones_like(probs)

    """ Estimate arm values """
    # for each weighting scheme, return [estimate, S.E, bias, 95%-coverage, t-stat, mse, truth]
    stats = dict(
        uniform=aw_stats(scores, wts_uniform, truth),
        propscore=aw_stats(scores, wts_propscore, truth),
        lvdl=aw_stats(scores, wts_lvdl, truth),
        two_point=aw_stats(scores, wts_twopoint, truth),
        two_point_old=aw_stats(scores, wts_twopoint_old, truth),
        bernstein=population_bernstein_stats(rewards, arms, truth, K),
        #empirical_bernstein=empirical_bernstein_stats(rewards, arms, truth, K, R),
    )

    # # add estimates of W_decorrelation
    # W_names = f'W_lambdas_{experiment}-{noise_func}-{T}.npz'
    # W_save = np.load(W_names)  # load presaved W-lambdas
    # for percentile, W_lambda in zip(W_save['percentiles'], W_save['W_lambdas']):
    #     stats[f'W-decorrelation_{percentile}'] = wdecorr_stats(
    #         arms, rewards, K, W_lambda, truth)

    """ Estimate contrasts """
    contrasts = dict(
        uniform=aw_contrasts(scores, wts_uniform, truth),
        propscore=aw_contrasts(scores, wts_propscore, truth),
        lvdl=aw_contrasts(scores, wts_lvdl, truth),
        two_point=aw_contrasts(scores, wts_twopoint, truth),
        two_point_old=aw_contrasts(scores, wts_twopoint_old, truth),
        bernstein=population_bernstein_contrast(rewards, arms, truth, K)
    )

    """ Save results """
    weights = dict(
        uniform=wts_uniform,
        propscore=wts_propscore,
        lvdl=wts_lvdl,
        two_point=wts_twopoint,
    )

    ratios = dict(
        lvdl=np.ones((T, K)) / np.arange(T, 0, -1)[:, np.newaxis],
        two_point=twopoint_ratio,
    )

    config = dict(
        T=T,
        K=K,
        noise_func=noise_func,
        noise_scale=noise_scale,
        floor_start=floor_start,
        floor_decay=floor_decay,
        initial=initial,
        dgp=experiment,
    )

#     # only save at saved_timepoints for assigmnment probabilities, conditional variance, weights, ratios(lambdas)
#     saved_timepoints = list(range(0, T, T//100))
#     condVars = dict()
#     for method, weight in weights.items():
#         condVar = weight ** 2 / probs / np.sum(weight, 0) ** 2 * T
#         weight = weight / np.sum(weight, 0) * T
#         condVars[method] = condVar[saved_timepoints, :]
#         weights[method] = weight[saved_timepoints, :]
#     for ratio in ratios:
#         ratios[ratio] = ratios[ratio][saved_timepoints, :]
#     probs = probs[saved_timepoints, :]

    results = dict(
        config=config,
        probs=probs,
        stats=stats,
        contrasts=contrasts,
        weights=weights,
#         ratios=ratios,
#         condVars=condVars
    )
    
    r = results
    timepoints = np.arange(0, T, T//100)

    # get statistics table
    tabs_stats = []
    for method, stat in r['stats'].items():
        stat = np.row_stack([stat, np.abs(stat[2])])
        tab_stats = pd.DataFrame({"statistic": ["estimate", "stderr", "bias", "90% coverage of t-stat", "t-stat", "mse", "CI_width", "truth", 'abserr'] * stat.shape[1],
                                  "policy": np.repeat(np.arange(K), stat.shape[0]),
                                  "value":  stat.flatten(order='F'),
                                  "method": method,
                                 **r['config']})
        tabs_stats.append(tab_stats)


    # get contrast table
    tabs_contrasts = []
    for method, contrast in r['contrasts'].items():
        tabs_contrast = pd.DataFrame({"statistic": ["truth",
                                                    "estimate", "bias", "mse",
                                                    "stderr", "t-stat", "90% coverage of t-stat", "CI_width"] * contrast.shape[1],
                                      "policy": np.repeat([f"(0,{k})" for k in np.arange(1, K)], contrast.shape[0]),
                                      "value": contrast.flatten(order='F'),
                                      "method": method,
                                     **r['config']})
        tabs_contrasts.append(tabs_contrast)

    df = pd.concat(tabs_stats + tabs_contrasts)
    
    dfs.append(df)


# In[7]:


df = pd.concat(dfs)

filename = compose_filename(f'weight_experiment_{experiment}_{noise_func}', 'pkl')

if on_sherlock():
    write_dir = get_sherlock_dir('adaptive-confidence-intervals', 'simulations', create=True)
    print(f"saving at {write_dir}")
else:
     write_dir = join(os.getcwd(), 'results')
write_path = os.path.join(write_dir, filename)
df.to_pickle(write_path)
    
print(f"Time passed {time()-start_time}s")


# In[ ]:





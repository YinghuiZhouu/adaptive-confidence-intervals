#!/usr/bin/env python
# coding: utf-8

# In[27]:


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


# In[28]:


start_time = time()


# In[29]:


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


# In[30]:


num_sims = 20 if on_sherlock() else 1

# DGP specification
# ----------------------------------------------------
noise_func = 'uniform'
truths = {
    'nosignal': np.array([1., 1., 1.]),
    'lowSNR': np.array([.9, 1., 1.1]),
    'highSNR': np.array([.5, 1., 1.5])
}
if on_sherlock():
    Ts = [1_000, 5_000, 10_000, 50_000, 100_000]
else:
    Ts = [100_000]
floor_decays = [.7] #[.25, .5, .6, .7, .8, .9, .99]
initial = 5  # initial number of samples of each arm to do pure exploration
exploration = 'TS'
noise_scale = 1.


# In[12]:


df_stats = []
df_lambdas = []


# In[32]:


# Run simulations
for s in range(num_sims):
    if (s+1) % 10 == 0:
        print(f'Running simulation {s+1}/{num_sims}')

    """ Experiment configuration """
    T = choice(Ts)  # number of samples
    experiment = choice(list(truths.keys()))
    truth = truths[experiment]
    K = len(truth)  # number of arms
    floor_start = 1/K
    floor_decay = choice(floor_decays)

    """ Generate data """
    noise = np.random.uniform(-noise_scale, noise_scale, size=(T, K))
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

    # Other weights: lvdl(constant allocation rate), propscore and uniform
    wts_lvdl = np.sqrt(probs)
    wts_propscore = probs
    wts_uniform = np.ones_like(probs)

    """ Estimate arm values """
    # for each weighting scheme, return [estimate, S.E, bias, 90%-coverage, t-stat, mse, truth]
    stats = dict(
        uniform=evaluate_aipw_stats(scores, wts_uniform, truth),
        propscore=evaluate_aipw_stats(scores, wts_propscore, truth),
        lvdl=evaluate_aipw_stats(scores, wts_lvdl, truth),
        two_point=evaluate_aipw_stats(scores, wts_twopoint, truth),
        beta_bernoulli=evaluate_beta_bernoulli_stats(rewards, arms, truth, K, floor_decay, alpha=.1),
        gamma_exponential=evaluate_gamma_exponential_stats(rewards, arms, truth, K, floor_decay, c=2, expected_noise_variance=1/3, alpha=.1),
    )
    
    # # add estimates of W_decorrelation
    W_name = f'wdecorr_results/W_lambdas_{experiment}-{noise_func}-{T}-{floor_decay}.npz'
    try:
        W_save = np.load(W_name)  # load presaved W-lambdas
        for percentile, W_lambda in zip(W_save['percentiles'], W_save['W_lambdas']):
            stats[f'W-decorrelation_{percentile}'] = wdecorr_stats(arms, rewards, K, W_lambda, truth)
    except FileNotFoundError:
        print(f'Could not find relevant w-decorrelation file {W_name}.')
        
    
    """ Estimate contrasts """
    contrasts = dict(
        uniform=evaluate_aipw_contrasts(scores, wts_uniform, truth),
        propscore=evaluate_aipw_contrasts(scores, wts_propscore, truth),
        lvdl=evaluate_aipw_contrasts(scores, wts_lvdl, truth),
        two_point=evaluate_aipw_contrasts(scores, wts_twopoint, truth),
        beta_bernoulli=evaluate_beta_bernoulli_contrasts(rewards, arms, truth, K, floor_decay, alpha=.1),
        gamma_exponential=evaluate_gamma_exponential_contrasts(rewards, arms, truth, K, floor_decay, c=2, expected_noise_variance=1/3, alpha=.1),
    )

    
    """ Save results """
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

    ratios = dict(
        lvdl=np.ones((T, K)) / np.arange(T, 0, -1)[:, np.newaxis],
        two_point=twopoint_ratio,
    )
    
    # save lambda values at selected timepoints
    saved_timepoints = list(range(0, T, T // 250)) + [T-1]
    for ratio in ratios:
        ratios[ratio] = ratios[ratio][saved_timepoints, :]
    
    # tabulate arm values
    tabs_stats = []
    for method, stat in stats.items():
        tab_stats = pd.DataFrame({"statistic": ["estimate", "stderr", "bias", "90% coverage of t-stat", "t-stat", "mse", "CI_width", "truth"] * stat.shape[1],
                                  "policy": np.repeat(np.arange(K), stat.shape[0]),
                                  "value":  stat.flatten(order='F'),
                                  "method": method,
                                 **config})
        tabs_stats.append(tab_stats)


    # tabulate arm contrasts
    tabs_contrasts = []
    for method, contrast in contrasts.items():
        tabs_contrast = pd.DataFrame({"statistic": ["estimate", "stderr", "bias", "90% coverage of t-stat", "t-stat", "mse", "CI_width", "truth"] * contrast.shape[1],
                                      "policy": np.repeat([f"(0,{k})" for k in np.arange(1, K)], contrast.shape[0]),
                                      "value": contrast.flatten(order='F'),
                                      "method": method,
                                     **config})
        tabs_contrasts.append(tabs_contrast)

    
    df_stats.extend(tabs_stats)
    df_stats.extend(tabs_contrasts)
    
    
    """ Save relevant lambda weights, if applicable """
    if T == max(Ts):
        saved_timepoints = list(range(0, T, T // 500))
        lambdas = twopoint_ratio[saved_timepoints] * (T - np.array(saved_timepoints)[:,np.newaxis])
        lambdas = {key: value for key, value in enumerate(lambdas.T)}
        dfl = pd.DataFrame({**lambdas, **config, 'time': saved_timepoints})
        dfl = pd.melt(dfl, id_vars=list(config.keys()) + ['time'], var_name='policy', value_vars=list(range(K)))
        df_lambdas.append(dfl)
        
    print(f"Time passed {time()-start_time}s")


# In[ ]:


df_stats = pd.concat(df_stats)
if len(df_lambdas) > 0:
    df_lambdas = pd.concat(df_lambdas)


# In[ ]:


filename1 = compose_filename(f'stats', 'pkl')
filename2 = compose_filename(f'lambdas', 'pkl')

if on_sherlock():
    write_dir = get_sherlock_dir('adaptive-confidence-intervals', 'simulations', create=True)
    print(f"saving at {write_dir}")
else:
     write_dir = join(os.getcwd(), 'results')
write_path1 = os.path.join(write_dir, filename1)
write_path2 = os.path.join(write_dir, filename2)

df_stats.to_pickle(write_path1)
if len(df_lambdas) > 0:
    df_lambdas.to_pickle(write_path2)


# In[ ]:


print("All done.")


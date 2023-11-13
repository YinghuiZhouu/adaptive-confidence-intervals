#!/usr/bin/env python
# coding: utf-8

# In[7]:


import os
from adaptive.inference import analyze, aw_scores
from adaptive.experiment import *
from adaptive.ridge import *
from adaptive.datagen import *
from adaptive.saving import *
import random
random.seed(60637) # check which of these we actually need for fixed state
np.random.seed(60637)
seed = random.randrange(99999)


# In[3]:


K = 4 # Number of arms
p = 3 # Number of features
T = 7000 # Sample size
batch_sizes = [200] + [100] * 68 # Batch sizes
signal_strength = 0.5
config = dict(T=T, K=K, p=p, noise_form='normal', noise_std=1, noise_scale=0.5, floor_start=1/K,
      bandit_model = 'RegionModel', floor_decay=0.8, dgp='synthetic_signal')

# Collect data from environment, run experiment
data_exp, mus = simple_tree_data(T=T, K=K, p=p, noise_std=1,
    split=0.5, signal_strength=signal_strength, noise_form='normal', seed = seed)
xs, ys = data_exp['xs'], data_exp['ys']
data = run_experiment(xs, ys, config, batch_sizes=batch_sizes)
yobs, ws, probs = data['yobs'], data['ws'], data['probs']


# In[4]:


# Estimate muhat and gammahat
muhat = ridge_muhat_lfo_pai(xs, ws, yobs, K, batch_sizes)
gammahat = aw_scores(yobs=yobs, ws=ws, balwts=1 / collect(collect3(probs), ws),
                     K=K, muhat=collect3(muhat))
optimal_mtx = expand(np.ones(T), np.argmax(data_exp['muxs'], axis=1), K)
analysis = analyze(
                probs=probs,
                gammahat=gammahat,
                policy=optimal_mtx,
                policy_value=0.5,
            )


# In[ ]:


analysis


# In[ ]:


seed


# In[5]:


np.savetxt("results/yobs.csv", data['yobs'], delimiter=",")
np.savetxt("results/ws.csv", data['ws'], delimiter=",")
np.savetxt("results/xs.csv", data['xs'], delimiter=",")
np.savetxt("results/ys.csv", data['ys'], delimiter=",")
np.savetxt("results/muxs.csv", data_exp['muxs'], delimiter=",")
np.savetxt("results/muhat.csv", collect3(muhat), delimiter=",")
np.savetxt("results/gammahat.csv", gammahat, delimiter=",")


# In[6]:


np.save("results/probs", probs)


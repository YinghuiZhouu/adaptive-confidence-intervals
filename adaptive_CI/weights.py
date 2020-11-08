"""
This script contains helper functions to compute evaluation weights. 
"""

from adaptive_CI.compute import *
import numpy as np


# def twopoint_stable_var_ratio(probs, floor_decay):
#     """
#     Compute lambda of two-point allocation rate weights
# 
#     INPUT:
#         - probs: arm assignment probabilities of shape [T, K]
#         - floor_start: assignment probability floor starting value
#         - floor_decay: assignment probability floor decaying rate
#         assignment probability floor = floor start * t ^ {-floor_decay}
# 
#     OUTPUT:
#         - ratio: lambda of size [T]
#     """
#     # analytical expectation is E[e_t/(e_t+...+e_T)]
#     T, K = probs.shape
#     floor_start = 1/K
#     t = np.arange(1, T + 1)[:, np.newaxis]
#     l_t = floor_start * (((t + 1) ** (1-floor_decay) - t **
#                           (1-floor_decay))) / (1 - floor_decay)
#     L_t = floor_start * (((T + 1) ** (1-floor_decay) - t **
#                           (1-floor_decay))) / (1 - floor_decay)
# 
#     ratio_best = (1 - (K-1) * l_t) / ((T + 1) - t - (K-1) * L_t)
#     ratio_not_best = l_t / L_t
#     ratio = probs * ratio_best + (1 - probs) * ratio_not_best
#     return ratio
    
    

def twopoint_stable_var_ratio(e, alpha):
    T, K = e.shape
    t = np.arange(1, T + 1)[:, np.newaxis]

    bad_denom  = 1 + T * (t/T)**alpha - t   # used when e small
    good_denom = 1 + T - t                  # used when e large 

    lamb = (1 - e) / bad_denom + e * e / good_denom
    
    assert np.all(lamb >= 0)
    assert np.all(lamb <= 1)
    return lamb


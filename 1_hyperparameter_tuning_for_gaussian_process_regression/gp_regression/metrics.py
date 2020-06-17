"""
The module contains functions for performing pair-wise distance calculations.
"""

import numpy as np


def euclidean_dist(X, X_prime):
     return np.sqrt(np.power(X[:, np.newaxis] - X_prime[np.newaxis, :], 2), dtype=np.float)

def squared_euclidean_dist(X, X_prime):
    return np.power(X[:, np.newaxis] - X_prime[np.newaxis, :], 2, dtype=np.float)

def manhattan_dist(X, X_prime):
    return np.abs(X[:, np.newaxis] - X_prime[np.newaxis, :], dtype = np.float)

# dictionary for selecting a distance metric
PAIRWISE_DISTANCE_FUNCS = {'euclidean': euclidean_dist, 
                           'manhattan': manhattan_dist,
                           'l1': manhattan_dist,
                           'l2': euclidean_dist,
                           'seuclidean': squared_euclidean_dist,
                          }

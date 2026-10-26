# SPDX-License-Identifier: GPL-3.0-only
# Copyright (C) 2026 Jorge Marco Blanco
# Distributed under the GNU GPL version 3; see LICENSE.

import numpy as np
from sklearn.decomposition import PCA
from scipy.spatial.distance import pdist, squareform



def calculate_effective_dimensionality(features):
    """
    Calculates Effective Dimensionality (ED) using Participation Ratio.
    Returns a scalar ED score, or 0.0 for empty or zero-variance data.
    """
    N, D = features.shape
    
    if N == 0: return 0.0

    X = features - np.mean(features, axis=0)

    if N < D:
        cov = (X @ X.T) / (N - 1)
    else:
        cov = np.cov(X, rowvar=False)
        
    eigvals = np.linalg.eigvalsh(cov)
    eigvals = eigvals[eigvals > 1e-9]
    
    if len(eigvals) == 0: return 0.0
    
    numerator = np.sum(eigvals) ** 2
    denominator = np.sum(eigvals ** 2)
    
    ed = numerator / denominator
    return float(ed)

def calculate_distance_correlation(X, Y):
    """
    Calculates the Distance Correlation (dCor) between X and Y.
    """
    X = X.reshape(X.shape[0], -1)
    Y = Y.reshape(Y.shape[0], -1)
    n = X.shape[0]
    if n < 2: return 0.0

    # Pairwise Euclidean Distance
    a = squareform(pdist(X, metric='euclidean'))
    b = squareform(pdist(Y, metric='euclidean'))

    # Double Center
    A = a - a.mean(axis=0)[None, :] - a.mean(axis=1)[:, None] + a.mean()
    B = b - b.mean(axis=0)[None, :] - b.mean(axis=1)[:, None] + b.mean()

    dcov2_xy = np.mean(A * B)
    dcov2_xx = np.mean(A * A)
    dcov2_yy = np.mean(B * B)

    if dcov2_xx <= 0 or dcov2_yy <= 0:
        return 0.0
        
    dcor = np.sqrt(dcov2_xy) / np.sqrt(np.sqrt(dcov2_xx) * np.sqrt(dcov2_yy))
    return dcor

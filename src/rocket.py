# Angus Dempster, Francois Petitjean, Geoff Webb
#
# @article{dempster_etal_2020,
#   author  = {Dempster, Angus and Petitjean, Fran\c{c}ois and Webb, Geoffrey I},
#   title   = {ROCKET: Exceptionally fast and accurate time classification using random convolutional kernels},
#   year    = {2020},
#   journal = {Data Mining and Knowledge Discovery},
#   doi     = {https://doi.org/10.1007/s10618-020-00701-z}
# }
#
# https://arxiv.org/abs/1910.13051 (preprint)

# SPDX-License-Identifier: GPL-3.0-only
# Adaptation and modifications: Copyright (C) 2026 Jorge Marco Blanco.
# Modified version prepared for publication on 2026-10-26.
# Changes: PPV-only features, configurable lengths, unit-energy kernels,
# input checks and a scikit-learn-style wrapper. See LICENSE and NOTICE.md.

# Adapted from the original ROCKET implementation by
# Angus Dempster, François Petitjean and Geoffrey I. Webb.
#
# Original code:
# https://github.com/angus924/rocket
#
# Reference:
# Dempster, A., Petitjean, F. and Webb, G. I. (2020).
# ROCKET: exceptionally fast and accurate time series classification
# using random convolutional kernels.
# Data Mining and Knowledge Discovery, 34, 1454–1495.
# https://doi.org/10.1007/s10618-020-00701-z
#
# This adaptation retains PPV features only, allows configurable kernel
# lengths, and provides a scikit-learn-style interface.
#
# Retain the original licence and any applicable copyright notices.

import numpy as np
from numba import njit, prange

@njit
def generate_kernels(input_length, num_kernels, kernel_lengths):
    """
    Generates random weights, biases, dilations, and paddings for ROCKET.
    Returns a flattened weights array to remain Numba-compatible.
    """
    if input_length < 2:
        raise ValueError("Input series must contain at least two observations.")
    if len(kernel_lengths) == 0:
        raise ValueError("At least one kernel length is required.")
    for length in kernel_lengths:
        if length < 2:
            raise ValueError("Kernel lengths must be at least 2.")

    weights = np.zeros(num_kernels, dtype=np.float64) # Placeholder, overwritten below
    lengths = np.zeros(num_kernels, dtype=np.int32)
    biases = np.zeros(num_kernels, dtype=np.float64)
    dilations = np.zeros(num_kernels, dtype=np.int32)
    paddings = np.zeros(num_kernels, dtype=np.int32)
    
    num_lengths = len(kernel_lengths)
    
    # Randomly select lengths for each kernel and compute total weight size
    total_weights = 0
    for i in range(num_kernels):
        length = kernel_lengths[np.random.randint(num_lengths)]
        lengths[i] = length
        total_weights += length
        
    # Standard normal weights
    weights_flat = np.random.normal(0.0, 1.0, total_weights)
    
    a1 = 0

    # Assign dilations, paddings, and biases
    for i in range(num_kernels):
        length = lengths[i]

        b1 = a1 + length
        _weights = weights_flat[a1:b1]
        _weights = _weights - _weights.mean()

        norm = np.sqrt(np.sum(_weights ** 2))
        if norm == 0.0:
            raise ValueError("Cannot normalise a zero-energy kernel")

        weights_flat[a1:b1] = _weights / norm
        a1 = b1
        
        # Dilation scale
        max_exponent = np.log2((input_length - 1) / (length - 1))
        dilation = int(2 ** np.random.uniform(0, max_exponent))
        dilations[i] = dilation
        
        # Padding (50% chance of padding)
        padding = ((length - 1) * dilation) // 2 if np.random.randint(2) == 1 else 0
        paddings[i] = padding
        
        # Bias drawn from Uniform(-1, 1)
        biases[i] = np.random.uniform(-1.0, 1.0)
        
    return weights_flat, lengths, biases, dilations, paddings

@njit(parallel=True, fastmath=True)
def apply_kernels_ppv_only(X, weights_flat, lengths, biases, dilations, paddings):
    """
    Applies kernels to dataset X and computes ONLY the Proportion of Positive Values (PPV).
    Adapted from the original ROCKET implementation.
    """
    n_instances, _, input_length = X.shape
    num_kernels = len(lengths)
    
    # Feature matrix will only hold PPV (n_instances, num_kernels)
    features = np.zeros((n_instances, num_kernels), dtype=np.float64)
    
    for i in prange(n_instances):
        w_idx = 0
        for k in range(num_kernels):
            length = lengths[k]
            dilation = dilations[k]
            padding = paddings[k]
            bias = biases[k]
            
            # Extract weights for this specific kernel
            w = weights_flat[w_idx : w_idx + length]
            w_idx += length
            
            ppv_count = 0.0
            n_windows = 0
            
            # Sliding window convolution
            for start in range(-padding, input_length - ((length - 1) * dilation) + padding):
                n_windows += 1
                dot_prod = 0.0
                
                for step in range(length):
                    idx = start + step * dilation
                    # Zero padding bounds check
                    if 0 <= idx < input_length:
                        dot_prod += X[i, 0, idx] * w[step]
                
                # PPV: Check if dot product + bias is strictly positive
                if (dot_prod + bias) > 0:
                    ppv_count += 1.0
                    
            features[i, k] = ppv_count / n_windows if n_windows > 0 else 0.0
            
    return features

class RocketPPV:
    """
    Minimalist scikit-learn compatible wrapper for the original ROCKET algorithm,
    stripped down to generate only PPV features.
    """
    def __init__(self, num_kernels=10000, kernel_lengths=None):
        self.num_kernels = num_kernels
        self.kernel_lengths = kernel_lengths if kernel_lengths is not None else [7, 9, 11]
        self.params = None
        
    def fit(self, X, y=None):
        """Generates kernels based on the time series input length."""
        _, _, input_length = X.shape
        self.params = generate_kernels(input_length, self.num_kernels, np.array(self.kernel_lengths, dtype=np.int32))
        return self
        
    def transform(self, X):
        """Applies the generated kernels to the input data."""
        if self.params is None:
            raise ValueError("Model must be fit before calling transform.")
        return apply_kernels_ppv_only(X, *self.params)

    def fit_transform(self, X, y=None):
        """Fits the model and transforms the data in one step."""
        self.fit(X)
        return self.transform(X)

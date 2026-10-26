# SPDX-License-Identifier: GPL-3.0-only
# Copyright (C) 2026 Jorge Marco Blanco
# Distributed under the GNU GPL version 3; see LICENSE.

import os
import sys
import yaml
import argparse
import itertools
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.linear_model import RidgeClassifierCV
from sklearn.preprocessing import StandardScaler

# Setup paths
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.append(project_root)

from src.data_loader import DataManager
from src.rocket import RocketPPV
from src.metrics import calculate_effective_dimensionality, calculate_distance_correlation


def main():
    parser = argparse.ArgumentParser(description="Run Effective Dimensionality Experiment")
    
    parser.add_argument(
        "--config", 
        type=str, 
        default="experiments/configs/experiment_kernel_length.yaml", 
        help="Path to YAML config file"
    )
    args = parser.parse_args()

    # Load Config
    with open(args.config, "r") as file:
        config = yaml.safe_load(file)

    # Extract parameters
    requested_datasets = config.get("datasets", ["all"])
    param_grid = config.get("param_grid", {})
    
    raw_lengths = param_grid.get("kernel_lengths", [7, 9, 11])
    num_kernels_list = param_grid.get("num_kernels", [10000])

    # Ensure lengths are safely formatted as lists of lists 
    kernel_lengths_list = []
    for l in raw_lengths:
        if isinstance(l, list):
            kernel_lengths_list.append(l)
        else:
            kernel_lengths_list.append([l])

    if not kernel_lengths_list or any(not lengths for lengths in kernel_lengths_list):
        raise ValueError("The experiment must specify nonempty kernel-length configurations.")

    # Output setup
    out_dir = Path(project_root) / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f"{config['experiment_name']}.csv"
    
    dm = DataManager()
    results = []

    print(f"--- Starting Experiment: {config['experiment_name']} ---")
    print(f"Using config file: {args.config}")
    
    # Using our new clean DataManager generator
    for dataset, X_train, y_train, X_test, y_test in dm.yield_datasets(requested_datasets):
        print(f"\nProcessing Dataset: {dataset}")

        # Create the grid of parameters to test 
        experiment_grid = list(itertools.product(num_kernels_list, kernel_lengths_list))

        for num_k, k_lengths in experiment_grid:
            print(f"  -> Testing num_kernels: {num_k}, kernel_lengths: {k_lengths}")
            
            # Initialize minimal ROCKET (PPV only)
            rocket = RocketPPV(num_kernels=num_k, kernel_lengths=k_lengths)
            
            
            # Extract Features
            features_train_raw = rocket.fit_transform(X_train)
            features_test_raw = rocket.transform(X_test)

            # Fit preprocessing on training data and reuse it for test data.
            scaler = StandardScaler()
            features_train_scaled = scaler.fit_transform(features_train_raw)
            features_test_scaled = scaler.transform(features_test_raw)

            # Select regularisation using training data, then score both splits.
            classifier = RidgeClassifierCV(
                alphas=np.logspace(-3, 3, 10),
                scoring="accuracy",
            )
            classifier.fit(features_train_scaled, y_train)
            train_accuracy = classifier.score(features_train_scaled, y_train)
            test_accuracy = classifier.score(features_test_scaled, y_test)
            print(f"     Accuracy: train={train_accuracy:.4f}, test={test_accuracy:.4f}")
            
            # Calculate ED
            ed_train = calculate_effective_dimensionality(features_train_scaled)
            
            # Measure dependence between the feature representation and input series.
            dcor_train = calculate_distance_correlation(features_train_scaled, X_train)

            # Append results
            results.append({
                "dataset": dataset,
                "num_kernels": num_k,
                "kernel_lengths": str(k_lengths),
                "ED_Train": ed_train,
                "DCor_Train": dcor_train,
                "Accuracy_Train": train_accuracy,
                "Accuracy_Test": test_accuracy,
                "Ridge_Alpha": float(classifier.alpha_),
            })

            # Save progress incrementally (safe against crashes)
            pd.DataFrame(results).to_csv(out_file, index=False)
            
    print(f"\n[Success] Results saved to {out_file}")

if __name__ == "__main__":
    main()

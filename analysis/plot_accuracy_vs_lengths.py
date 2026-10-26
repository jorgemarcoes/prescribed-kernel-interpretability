# SPDX-License-Identifier: GPL-3.0-only
# Copyright (C) 2026 Jorge Marco Blanco
# Distributed under the GNU GPL version 3; see LICENSE.

"""Reproduce the Chapter 3 accuracy table and training/test mean-rank figure."""

import argparse
import ast
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default="results/metrics_vs_lengths.csv")
    parser.add_argument("--metadata", default="data/metadata.csv")
    parser.add_argument("--output-dir", default="reports/accuracy_vs_lengths")
    args = parser.parse_args()

    df = pd.read_csv(ROOT / args.input)
    metadata = pd.read_csv(ROOT / args.metadata)
    if metadata["Dataset"].duplicated().any():
        raise ValueError("Dataset metadata must contain one row per dataset.")
    lengths = metadata.set_index("Dataset")["Length"]
    df = df.loc[df["num_kernels"] == 10000].copy()
    if df.empty:
        raise ValueError("No results found for 10,000 kernels.")
    configurations = df["kernel_lengths"].map(ast.literal_eval)
    if not configurations.map(lambda x: isinstance(x, list) and len(x) == 1).all():
        raise ValueError("Each configuration must specify one kernel length.")
    df["kernel_length"] = configurations.map(lambda x: int(x[0]))
    grid = sorted(df["kernel_length"].unique())
    if grid != list(range(3, 28, 2)):
        raise ValueError("Expected the 13 thesis kernel lengths from 3 to 27.")
    missing = set(df["dataset"]) - set(lengths.index)
    if missing:
        raise ValueError(f"Missing series lengths for {sorted(missing)}")
    df = df.loc[df["dataset"].map(lengths) >= max(grid)].copy()
    if df.empty or df.duplicated(["dataset", "kernel_length"]).any():
        raise ValueError("Expected a nonempty cohort with unique configurations.")
    if not df.groupby("dataset")["kernel_length"].nunique().eq(len(grid)).all():
        raise ValueError("Every dataset must have all kernel lengths.")
    accuracy_columns = ["Accuracy_Train", "Accuracy_Test"]
    values = df[accuracy_columns].to_numpy()
    if not np.isfinite(values).all() or ((values < 0) | (values > 1)).any():
        raise ValueError("Accuracies must be finite and within [0, 1].")

    for split in ["Train", "Test"]:
        df[f"{split}_rank"] = df.groupby("dataset")[f"Accuracy_{split}"].rank(
            ascending=False, method="average"
        )
    table = df.groupby("kernel_length")[[
        "Train_rank", "Test_rank", "Accuracy_Train", "Accuracy_Test"
    ]].mean().sort_index()
    output_dir = ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    table.to_csv(output_dir / "accuracy_vs_lengths.csv")
    print(f"Using {df['dataset'].nunique()} datasets with series length >= {max(grid)}.")
    print(table.round({"Train_rank": 2, "Test_rank": 2,
                       "Accuracy_Train": 4, "Accuracy_Test": 4}).to_string())

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(table.index, table["Train_rank"], marker="o", label="Training")
    ax.plot(table.index, table["Test_rank"], marker="s", label="Test")
    ax.set(xlabel="Kernel length", ylabel="Mean rank (lower is better)",
           xticks=grid)
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "train_vs_test_mean_rank.png", dpi=300)
    plt.close(fig)


if __name__ == "__main__":
    main()

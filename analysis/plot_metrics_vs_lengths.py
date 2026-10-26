# SPDX-License-Identifier: GPL-3.0-only
# Copyright (C) 2026 Jorge Marco Blanco
# Distributed under the GNU GPL version 3; see LICENSE.

import argparse
import os
import sys
import ast
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# --- PATH SETUP ---
current_dir = os.path.dirname(os.path.abspath(__file__)) 
project_root = os.path.dirname(current_dir)              

if project_root not in sys.path:
    sys.path.append(project_root)

# --- CONFIGURATION CONSTANTS ---
DEFAULT_INPUT_FILE = "results/metrics_vs_lengths.csv" 
DEFAULT_METADATA_PATH = "data/metadata.csv" 
DEFAULT_REPORT_DIR = "reports/metrics_vs_lengths"

# --- PLOTTING STYLE ---
sns.set_theme(style="whitegrid", context="paper", font_scale=1.3)

def extract_length(val):
    """Parses the kernel_length string (e.g., '[7]') back into an integer."""
    try:
        if isinstance(val, str) and '[' in val:
            parsed = ast.literal_eval(val)
            if isinstance(parsed, list) and len(parsed) > 0:
                return int(parsed[0])
        return int(val)
    except Exception:
        return np.nan

def bin_classes(n):
    if pd.isna(n): return 'Unknown'
    if n <= 2: return 'Binary (2)'
    elif n <= 4: return 'Low (3-4)'
    elif n <= 10: return 'Medium (5-10)'
    else: return 'High (>10)'

def get_num_classes(dataset_names, metadata_path):
    classes_dict = {}
    if os.path.exists(metadata_path):
        try:
            meta_df = pd.read_csv(metadata_path)
            ds_col = next((c for c in meta_df.columns if c.lower() in ['dataset', 'name']), None)
            cls_col = next((c for c in meta_df.columns if c.lower() in ['classes', 'num_classes', 'n_classes']), None)
            if ds_col and cls_col:
                mapping = dict(zip(meta_df[ds_col], meta_df[cls_col]))
                for ds in dataset_names:
                    if ds in mapping:
                        classes_dict[ds] = mapping[ds]
        except Exception as e:
            print(f"[Warning] Failed to parse metadata.csv: {e}")
    return classes_dict


def get_series_lengths(dataset_names, metadata_path):
    """Read series lengths exclusively from data/metadata.csv."""
    metadata = pd.read_csv(metadata_path)
    if not {'Dataset', 'Length'}.issubset(metadata.columns):
        raise ValueError("Metadata must contain 'Dataset' and 'Length' columns")
    metadata = metadata.loc[metadata['Dataset'].isin(dataset_names)].copy()
    if metadata['Dataset'].duplicated().any():
        raise ValueError('Metadata contains duplicate dataset rows')
    lengths = pd.to_numeric(metadata['Length'], errors='raise')
    if not np.isfinite(lengths).all() or (lengths <= 0).any():
        raise ValueError('Metadata series lengths must be positive finite numbers')
    return dict(zip(metadata['Dataset'], lengths))

# --- GENERIC PLOTTING FUNCTIONS ---

def plot_absolute(df, x_col, y_col, y_label, title, output_dir, filename, color):
    plt.figure(figsize=(10, 6))
    
    sns.lineplot(
        data=df, x=x_col, y=y_col, marker='o', linewidth=3,
        color=color, errorbar=('ci', 95)
    )
    
    plt.title(title)
    plt.ylabel(y_label)
    plt.xlabel("Kernel Length")
    
    x_vals = sorted(df[x_col].unique())
    plt.xticks(x_vals)
        
    plt.tight_layout()
    save_path = output_dir / filename
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"  Generated: {save_path}")

def plot_relative(df, x_col, y_col, y_label, title, output_dir, filename, color):
    plt.figure(figsize=(10, 6))
    
    df_rel = df.copy()
    max_vals = df_rel.groupby('dataset')[y_col].transform('max')
    rel_col = f'Relative_{y_col}'
    df_rel[rel_col] = df_rel[y_col] / max_vals.replace(0, np.nan)
    
    sns.lineplot(
        data=df_rel, x=x_col, y=rel_col, marker='o', linewidth=3,
        color=color, errorbar=('ci', 95)
    )
    
    plt.title(title)
    plt.ylabel(y_label)
    plt.xlabel("Kernel Length")
    
    x_vals = sorted(df_rel[x_col].unique())
    plt.xticks(x_vals)
 
    plt.tight_layout()

    save_path = output_dir / filename
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"  Generated: {save_path}")

def plot_relative_by_class(df, x_col, y_col, y_label, title, output_dir, filename):
    if 'class_bin' not in df.columns or df['class_bin'].nunique() <= 1:
        print(f"  [Skip] Class-binned plot skipped for {y_col} (metadata missing or homogeneous).")
        return

    plt.figure(figsize=(10, 6))
    
    df_rel = df.copy()
    max_vals = df_rel.groupby('dataset')[y_col].transform('max')
    rel_col = f'Relative_{y_col}'
    df_rel[rel_col] = df_rel[y_col] / max_vals.replace(0, np.nan)
    
    order = ['Binary (2)', 'Low (3-4)', 'Medium (5-10)', 'High (>10)']
    
    sns.lineplot(
        data=df_rel, x=x_col, y=rel_col, hue='class_bin',
        hue_order=[b for b in order if b in df_rel['class_bin'].values],
        style='class_bin', markers=True, dashes=False, linewidth=2.5,
        palette="viridis", errorbar=('ci', 95)
    )
    
    plt.title(title)
    plt.ylabel(y_label)
    plt.xlabel("Kernel Length")
    
    x_vals = sorted(df_rel[x_col].unique())
    plt.xticks(x_vals)
        
    
    
    plt.legend(title="Complexity", loc="lower right")
    plt.tight_layout()
    save_path = output_dir / filename
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"  Generated: {save_path}")

def main():
    parser = argparse.ArgumentParser(description="Visualize Metrics vs Kernel Length")
    parser.add_argument("--input", type=str, default=DEFAULT_INPUT_FILE, help="Path to the results CSV")
    parser.add_argument("--metadata", type=str, default=DEFAULT_METADATA_PATH, help="Path to metadata.csv")
    args = parser.parse_args()

    input_path = os.path.join(project_root, args.input)
    if not os.path.exists(input_path):
        print(f"[!] Error: Input file not found at {input_path}")
        sys.exit(1)

    print(f"\n--- Loading data from: {args.input} ---")
    df = pd.read_csv(input_path)
    
    if 'ED_Train' not in df.columns or 'kernel_lengths' not in df.columns:
        print("[!] Error: CSV must contain 'ED_Train' and 'kernel_lengths' columns.")
        sys.exit(1)

    df['kernel_length'] = df['kernel_lengths'].apply(extract_length)
    df = df.dropna(subset=['kernel_length'])
    
    unique_kernels = df['num_kernels'].unique()
    if len(unique_kernels) > 1:
        max_k = max(unique_kernels)
        print(f"[Info] Found multiple num_kernels {unique_kernels}. Filtering to highest capacity ({max_k}).")
        df = df[df['num_kernels'] == max_k]

    unique_datasets = df['dataset'].unique()
    series_lengths = get_series_lengths(
        unique_datasets, os.path.join(project_root, args.metadata)
    )
    missing = sorted(set(unique_datasets) - set(series_lengths))
    if missing:
        print(f"[Warning] Excluding datasets absent from metadata: {', '.join(missing)}")
        df = df.loc[df['dataset'].isin(series_lengths)].copy()
    # Use one dataset cohort across the entire plotted kernel-length range.
    max_kernel_length = df['kernel_length'].max()
    invalid = df['dataset'].map(series_lengths) < max_kernel_length
    for dataset in sorted(df.loc[invalid, 'dataset'].unique()):
        print(f"[Info] Excluding {dataset}: N={series_lengths[dataset]}, "
              f"shorter than maximum plotted kernel length {max_kernel_length:g}; "
              "removing all kernel-length results.")
    df = df.loc[~invalid].copy()
    if df.empty:
        raise ValueError('No datasets support the full plotted kernel-length range')
    print(f"[Info] Removed {invalid.sum()} rows; using {df['dataset'].nunique()} "
          f"datasets with N >= {max_kernel_length:g} at every kernel length.")
    # Invalid legacy rows may have made pandas infer text for the ED column.
    for metric in ['ED_Train', 'DCor_Train']:
        if metric in df.columns:
            df[metric] = pd.to_numeric(df[metric], errors='raise')

    unique_datasets = df['dataset'].unique()
    classes_dict = get_num_classes(unique_datasets, os.path.join(project_root, args.metadata))
    if classes_dict:
        df['num_classes'] = df['dataset'].map(classes_dict)
        df['class_bin'] = df['num_classes'].apply(bin_classes)
    
    report_base = Path(project_root) / DEFAULT_REPORT_DIR
    report_base.mkdir(parents=True, exist_ok=True)

    print("\n--- Generating Plots for ED ---")
    plot_absolute(df, 'kernel_length', 'ED_Train', "Absolute ED (Components)", 
                  "Absolute Effective Dimensionality vs. Kernel Length", report_base, "absolute_ed_vs_length.png", "#1f77b4")
    plot_relative(df, 'kernel_length', 'ED_Train', "Relative ED (Proportion of Max)", 
                  "Relative Effective Dimensionality vs. Kernel Length", report_base, "relative_ed_vs_length.png", "#2ca02c")
    plot_relative_by_class(df, 'kernel_length', 'ED_Train', "Relative ED (Proportion of Max)", 
                           "Relative ED vs. Kernel Length\n(Stratified by Complexity)", report_base, "relative_ed_vs_length_by_classes.png")

    if 'DCor_Train' in df.columns:
        print("\n--- Generating Plots for Distance Correlation ---")
        plot_absolute(df, 'kernel_length', 'DCor_Train', "Absolute DCor", 
                      "Absolute Distance Correlation vs. Kernel Length", report_base, "absolute_dcor_vs_length.png", "#d62728")
        plot_relative(df, 'kernel_length', 'DCor_Train', "Relative DCor (Proportion of Max)", 
                      "Relative Distance Correlation vs. Kernel Length", report_base, "relative_dcor_vs_length.png", "#ff7f0e")
        plot_relative_by_class(df, 'kernel_length', 'DCor_Train', "Relative DCor (Proportion of Max)", 
                               "Relative DCor vs. Kernel Length\n(Stratified by Complexity)", report_base, "relative_dcor_vs_length_by_classes.png")
    
    print("\n[Success] Analysis Complete.")

if __name__ == "__main__":
    main()

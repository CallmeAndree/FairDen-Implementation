#!/usr/bin/env python3
"""
Balance Visualization Script

This script generates a bar chart visualization of Balance metrics from experiment results.
It processes results from rw_experiment, student_experiment, and compas_experiment directories.

Output is saved to Report/fig directory.
"""

import os
import glob
import ast
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Define paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(BASE_DIR, 'results')
OUTPUT_DIR = os.path.join(BASE_DIR, 'Report', 'fig')

# Experiment directories to process
EXPERIMENT_DIRS = [
    'rw_experiment',
    'student_experiment',
    'compas_experiment'
]

# Algorithm colors (matching the reference image exactly)
# Order: DBSCAN, FairDEN, FairSC, FairSC (N), Fairlet (MCF), GroundTruth, Scalable
ALGORITHM_COLORS = {
    'GroundTruth_DB': '#9E9E9E',       # Gray (DBSCAN)
    'FairDen': '#FF9800',              # Orange (FairDEN)
    'FairSC': '#2196F3',               # Blue (FairSC)
    'FairSC_normalized': '#00BCD4',    # Cyan (FairSC (N))
    'Fairlet_MCF Fairlet': '#4CAF50',  # Green (Fairlet (MCF))
    'GroundTruth': '#E91E63',          # Pink (GroundTruth)
    'Scalable': '#9C27B0',             # Purple (Scalable)
}

# Algorithm display names (for legend)
ALGORITHM_DISPLAY_NAMES = {
    'GroundTruth_DB': 'DBSCAN',
    'FairDen': 'FairDEN',
    'FairSC': 'FairSC',
    'FairSC_normalized': 'FairSC (N)',
    'Fairlet_MCF Fairlet': 'Fairlet (MCF)',
    'GroundTruth': 'GroundTruth',
    'Scalable': 'Scalable',
}

# Dataset display names
DATASET_NAMES = {
    'adult': 'Adult (race)',
    'adult4': 'Adult (gender)',
    'bank': 'Bank',
    'communities': 'Communities',
    'diabetes': 'Diabetes',
    'student_address': 'Student (address)',
    'compas': 'COMPAS (race)',
}

# Datasets to include (filter out other variants)
INCLUDED_DATASETS = [
    'adult', 'adult4', 'bank', 'communities', 'diabetes',
    'student_address', 'compas'
]


def parse_balance_value(value):
    """
    Parse Balance value which can be either:
    - A float/int
    - A string representation of a dict (for multi-attribute datasets)
    
    For dict values, returns the average of all attribute balances.
    Returns -2.0 for invalid/failed experiments.
    """
    if pd.isna(value):
        return -2.0
    
    if isinstance(value, (int, float)):
        return float(value)
    
    if isinstance(value, str):
        # Try to parse as dict
        if value.startswith('{'):
            try:
                balance_dict = ast.literal_eval(value)
                if isinstance(balance_dict, dict):
                    # Return average of all attribute balances
                    valid_values = [v for v in balance_dict.values() if v != -2.0]
                    if valid_values:
                        return sum(valid_values) / len(valid_values)
                    return -2.0
            except (ValueError, SyntaxError):
                pass
        # Try to parse as float
        try:
            return float(value)
        except ValueError:
            return -2.0
    
    return -2.0


def load_results(experiment_dir: str) -> pd.DataFrame:
    """
    Load all *_results.csv files from an experiment directory.
    
    Note: compas comes from compas_experiment, student_address comes from student_experiment.
    These are filtered out from rw_experiment to avoid duplicates.
    """
    results = []
    result_files = glob.glob(os.path.join(RESULTS_DIR, experiment_dir, '*_results.csv'))
    
    for file_path in result_files:
        try:
            df = pd.read_csv(file_path)
            # Parse Balance column
            df['Balance'] = df['Balance'].apply(parse_balance_value)
            
            # Filter based on experiment directory
            # compas should only come from compas_experiment
            # student_address should only come from student_experiment
            if experiment_dir == 'rw_experiment':
                df = df[~df['Data'].isin(['compas', 'student_address'])]
            
            if len(df) > 0:
                results.append(df)
        except Exception as e:
            print(f"Warning: Could not load {file_path}: {e}")
    
    if results:
        return pd.concat(results, ignore_index=True)
    return pd.DataFrame()


def filter_valid_results(df: pd.DataFrame) -> pd.DataFrame:
    """Filter out invalid results (Balance = -2.0 indicates failed experiments)."""
    return df[df['Balance'] != -2.0].copy()


def get_best_balance_per_dataset_algorithm(df: pd.DataFrame) -> pd.DataFrame:
    """
    Get the best (highest) Balance value for each (Data, Algorithm) pair.
    This handles cases where there are multiple experiments with different parameters.
    For diabetes dataset, only use results with N_cluster=2.
    """
    # For diabetes, only use N_cluster=2
    if 'N_cluster' in df.columns:
        diabetes_mask = (df['Data'] == 'diabetes') & (df['N_cluster'] != 2)
        df = df[~diabetes_mask].copy()
    
    # Group by Data and Algorithm, take the maximum Balance
    grouped = df.groupby(['Data', 'Algorithm'])['Balance'].max().reset_index()
    return grouped


def create_balance_chart(df: pd.DataFrame, output_path: str):
    """Create a grouped bar chart of Balance values by dataset and algorithm."""
    
    # Get unique datasets and algorithms
    datasets = df['Data'].unique()
    algorithms = df['Algorithm'].unique()
    
    # Sort algorithms to maintain consistent order (matching reference image)
    # Order: DBSCAN, FairDEN, FairSC, FairSC (N), Fairlet (MCF), GroundTruth, Scalable
    algorithm_order = ['GroundTruth_DB', 'FairDen', 'FairSC', 'FairSC_normalized', 
                       'Fairlet_MCF Fairlet', 'GroundTruth', 'Scalable']
    algorithms = [a for a in algorithm_order if a in algorithms]
    
    # Sort datasets
    dataset_order = ['adult', 'adult4', 'bank', 'communities', 'diabetes', 
                     'student_address', 'compas']
    datasets = [d for d in dataset_order if d in datasets]
    
    # Set up the figure
    fig, ax = plt.subplots(figsize=(16, 5))
    
    # Calculate bar positions
    n_datasets = len(datasets)
    n_algorithms = len(algorithms)
    bar_width = 0.11
    group_width = n_algorithms * bar_width + 0.15
    
    # Create x positions for each dataset group
    x_positions = np.arange(n_datasets) * group_width
    
    # Plot bars for each algorithm
    for i, algo in enumerate(algorithms):
        values = []
        for dataset in datasets:
            mask = (df['Data'] == dataset) & (df['Algorithm'] == algo)
            if mask.any():
                values.append(df.loc[mask, 'Balance'].values[0])
            else:
                values.append(0)  # No data for this combination
        
        color = ALGORITHM_COLORS.get(algo, '#607D8B')
        display_name = ALGORITHM_DISPLAY_NAMES.get(algo, algo)
        positions = x_positions + i * bar_width
        ax.bar(positions, values, bar_width, label=display_name, color=color, edgecolor='none')
    
    # Customize the chart
    ax.set_ylabel('Balance', fontsize=12, fontweight='bold')
    ax.set_ylim(0, 1.05)
    ax.set_yticks([0.0, 0.25, 0.5, 0.75, 1.0])
    
    # Set x-axis labels
    dataset_labels = [DATASET_NAMES.get(d, d) for d in datasets]
    ax.set_xticks(x_positions + (n_algorithms - 1) * bar_width / 2)
    ax.set_xticklabels(dataset_labels, fontsize=10, rotation=0, ha='center')
    
    # Add legend
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.12), 
              ncol=min(7, n_algorithms), frameon=False, fontsize=9)
    
    # Remove top and right spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # Add grid lines
    ax.yaxis.grid(True, linestyle='-', alpha=0.3)
    ax.set_axisbelow(True)
    
    # Adjust layout
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.2)
    
    # Save figure
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.savefig(output_path.replace('.png', '.pdf'), bbox_inches='tight', facecolor='white')
    print(f"Saved: {output_path}")
    print(f"Saved: {output_path.replace('.png', '.pdf')}")
    plt.close()


def main():
    """Main function to generate balance visualization."""
    print("=" * 60)
    print("Balance Visualization Generator")
    print("=" * 60)
    
    # Load all results
    all_results = []
    for exp_dir in EXPERIMENT_DIRS:
        print(f"\nLoading results from: {exp_dir}")
        df = load_results(exp_dir)
        if not df.empty:
            print(f"  Found {len(df)} result entries")
            all_results.append(df)
        else:
            print(f"  No results found")
    
    if not all_results:
        print("\nError: No results found in any experiment directory!")
        return
    
    # Combine all results
    combined_df = pd.concat(all_results, ignore_index=True)
    print(f"\nTotal entries: {len(combined_df)}")
    
    # Filter invalid results
    valid_df = filter_valid_results(combined_df)
    print(f"Valid entries (Balance != -2.0): {len(valid_df)}")
    
    # Filter to only include specified datasets
    valid_df = valid_df[valid_df['Data'].isin(INCLUDED_DATASETS)]
    print(f"Filtered to included datasets: {len(valid_df)}")
    
    # Get best balance per dataset-algorithm pair
    best_df = get_best_balance_per_dataset_algorithm(valid_df)
    print(f"Unique dataset-algorithm pairs: {len(best_df)}")
    
    # Print summary
    print("\n" + "-" * 60)
    print("Balance Summary by Dataset and Algorithm:")
    print("-" * 60)
    pivot = best_df.pivot(index='Data', columns='Algorithm', values='Balance')
    print(pivot.round(4).to_string())
    
    # Create visualization
    output_path = os.path.join(OUTPUT_DIR, 'balance_comparison.png')
    print(f"\n" + "-" * 60)
    print(f"Generating visualization...")
    create_balance_chart(best_df, output_path)
    
    print("\n" + "=" * 60)
    print("Done!")
    print("=" * 60)


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""
Visualization script for K-Line experiments.
Creates line plots showing Balance and DCSI vs number of clusters (k).
Similar to the author's plots in the paper.
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Output directory
OUTPUT_DIR = Path('visualization')
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Results directories
RESULTS_DIRS = {
    'k_line_experiment': Path('results/k_line_experiment'),
    'compas_experiment': Path('results/compas_experiment')
}

# Algorithm colors and styles (matching the paper)
ALGORITHM_STYLES = {
    'FairDen': {'color': '#E91E63', 'linestyle': '-', 'linewidth': 2},      # Pink, solid
    'FairSC': {'color': '#FFC107', 'linestyle': '--', 'linewidth': 2},      # Yellow, dashed
    'FairSC_normalized': {'color': '#2196F3', 'linestyle': ':', 'linewidth': 2},  # Blue, dotted
    'Fairlet_MCF Fairlet': {'color': '#FF9800', 'linestyle': '-.', 'linewidth': 2},  # Orange, dash-dot
    'Scalable': {'color': '#9C27B0', 'linestyle': '--', 'linewidth': 2, 'dashes': [5, 2, 2, 2]},  # Purple, custom dash
}

# Display names for algorithms
ALGORITHM_NAMES = {
    'FairDen': 'FairDen',
    'FairSC': 'FairSC',
    'FairSC_normalized': 'FairSC (N)',
    'Fairlet_MCF Fairlet': 'Fairlet (MCF)',
    'Scalable': 'Scalable'
}

# Dataset display names
DATASET_NAMES = {
    'adult2': 'Adult (race)',
    'adult5': 'Adult (gender)',
    'compas': 'COMPAS (race)',
    'compas_sex': 'COMPAS (sex)'
}


def load_results(results_dir, filename):
    """Load results from CSV file."""
    filepath = results_dir / filename
    if filepath.exists():
        return pd.read_csv(filepath)
    return None


def extract_balance_value(balance):
    """Extract balance value from potentially dict-like string."""
    if isinstance(balance, str):
        try:
            import ast
            balance_dict = ast.literal_eval(balance)
            if isinstance(balance_dict, dict):
                # Average all values
                return np.mean(list(balance_dict.values()))
        except:
            pass
    if isinstance(balance, (int, float)):
        return float(balance)
    return np.nan


def prepare_kline_data(df, algorithms):
    """Prepare data for k-line plotting."""
    result = {}
    
    for algo in algorithms:
        algo_data = df[df['Algorithm'] == algo].copy()
        if algo_data.empty:
            continue
        
        # Get k values and metrics
        k_values = []
        balance_values = []
        dcsi_values = []
        
        for _, row in algo_data.iterrows():
            k = row.get('N_cluster', row.get('Actual_clusters', -1))
            if k <= 0 or k == -2:
                continue
            
            balance = extract_balance_value(row.get('Balance', -2))
            dcsi = row.get('DCSI', -2)
            
            if balance == -2 or dcsi == -2:
                continue
            
            k_values.append(k)
            balance_values.append(balance)
            dcsi_values.append(dcsi)
        
        if k_values:
            result[algo] = {
                'k': k_values,
                'balance': balance_values,
                'dcsi': dcsi_values
            }
    
    return result


def plot_kline(data_dict, dataset_name, output_name, figsize=(12, 8)):
    """
    Create k-line plot with Balance and DCSI subplots.
    
    Parameters:
    -----------
    data_dict: dict
        Dictionary with algorithm data {algo: {'k': [...], 'balance': [...], 'dcsi': [...]}}
    dataset_name: str
        Name of dataset for title
    output_name: str
        Output filename without extension
    """
    fig, axes = plt.subplots(2, 1, figsize=figsize, sharex=True)
    
    # Balance plot
    ax1 = axes[0]
    ax1.set_ylabel('Balance', fontsize=12)
    ax1.set_title(dataset_name, fontsize=14)
    ax1.set_ylim(0, 1.0)
    ax1.set_yticks([0, 0.5, 1.0])
    ax1.grid(True, alpha=0.3)
    
    # DCSI plot
    ax2 = axes[1]
    ax2.set_xlabel('k', fontsize=12)
    ax2.set_ylabel('DCSI', fontsize=12)
    ax2.set_ylim(0, 0.6)
    ax2.grid(True, alpha=0.3)
    
    # Plot each algorithm
    for algo, data in data_dict.items():
        if algo not in ALGORITHM_STYLES:
            continue
        
        style = ALGORITHM_STYLES[algo]
        label = ALGORITHM_NAMES.get(algo, algo)
        
        # Sort by k
        sorted_idx = np.argsort(data['k'])
        k_sorted = np.array(data['k'])[sorted_idx]
        balance_sorted = np.array(data['balance'])[sorted_idx]
        dcsi_sorted = np.array(data['dcsi'])[sorted_idx]
        
        # Plot Balance
        if 'dashes' in style:
            line1, = ax1.plot(k_sorted, balance_sorted, color=style['color'], 
                             linestyle=style['linestyle'], linewidth=style['linewidth'], 
                             label=label)
            line1.set_dashes(style['dashes'])
        else:
            ax1.plot(k_sorted, balance_sorted, color=style['color'], 
                    linestyle=style['linestyle'], linewidth=style['linewidth'], 
                    label=label)
        
        # Plot DCSI
        if 'dashes' in style:
            line2, = ax2.plot(k_sorted, dcsi_sorted, color=style['color'], 
                             linestyle=style['linestyle'], linewidth=style['linewidth'])
            line2.set_dashes(style['dashes'])
        else:
            ax2.plot(k_sorted, dcsi_sorted, color=style['color'], 
                    linestyle=style['linestyle'], linewidth=style['linewidth'])
    
    # Set x-axis
    all_k = []
    for data in data_dict.values():
        all_k.extend(data['k'])
    if all_k:
        ax2.set_xlim(min(all_k) - 0.5, max(all_k) + 0.5)
        ax2.set_xticks(sorted(set(all_k)))
    
    plt.tight_layout()
    
    # Save
    plt.savefig(OUTPUT_DIR / f'{output_name}.png', dpi=150, bbox_inches='tight')
    plt.savefig(OUTPUT_DIR / f'{output_name}.pdf', bbox_inches='tight')
    plt.close()
    
    print(f"Saved: {output_name}.png/.pdf")


def plot_kline_comparison(data_dict1, data_dict2, name1, name2, output_name, figsize=(12, 8)):
    """
    Create side-by-side k-line comparison plots (like the paper figure).
    
    Parameters:
    -----------
    data_dict1, data_dict2: dict
        Algorithm data for each dataset
    name1, name2: str
        Dataset display names
    output_name: str
        Output filename
    """
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    
    datasets = [(data_dict1, name1), (data_dict2, name2)]
    
    for col, (data_dict, dataset_name) in enumerate(datasets):
        # Balance plot
        ax1 = axes[0, col]
        ax1.set_title(dataset_name, fontsize=12)
        ax1.set_ylabel('Balance' if col == 0 else '', fontsize=11)
        ax1.set_ylim(0, 1.0)
        ax1.set_yticks([0, 0.5, 1.0])
        ax1.grid(True, alpha=0.3)
        
        # DCSI plot  
        ax2 = axes[1, col]
        ax2.set_xlabel('k', fontsize=11)
        ax2.set_ylabel('DCSI' if col == 0 else '', fontsize=11)
        ax2.set_ylim(0, 0.6)
        ax2.grid(True, alpha=0.3)
        
        # Plot each algorithm
        for algo, data in data_dict.items():
            if algo not in ALGORITHM_STYLES:
                continue
            
            style = ALGORITHM_STYLES[algo]
            label = ALGORITHM_NAMES.get(algo, algo)
            
            # Sort by k
            sorted_idx = np.argsort(data['k'])
            k_sorted = np.array(data['k'])[sorted_idx]
            balance_sorted = np.array(data['balance'])[sorted_idx]
            dcsi_sorted = np.array(data['dcsi'])[sorted_idx]
            
            # Plot Balance
            ax1.plot(k_sorted, balance_sorted, color=style['color'], 
                    linestyle=style['linestyle'], linewidth=style['linewidth'], 
                    label=label)
            
            # Plot DCSI
            ax2.plot(k_sorted, dcsi_sorted, color=style['color'], 
                    linestyle=style['linestyle'], linewidth=style['linewidth'])
        
        # Set x-axis
        all_k = []
        for data in data_dict.values():
            all_k.extend(data['k'])
        if all_k:
            ax1.set_xlim(min(all_k) - 0.5, max(all_k) + 0.5)
            ax2.set_xlim(min(all_k) - 0.5, max(all_k) + 0.5)
            ax2.set_xticks(sorted(set(all_k)))
    
    # Add legend at the bottom - collect from both columns
    all_handles = []
    all_labels = []
    seen_labels = set()
    for col in [0, 1]:
        handles, labels = axes[0, col].get_legend_handles_labels()
        for h, l in zip(handles, labels):
            if l and l not in seen_labels:
                all_handles.append(h)
                all_labels.append(l)
                seen_labels.add(l)
    
    fig.legend(all_handles, all_labels, loc='lower center', ncol=5, fontsize=10,
               bbox_to_anchor=(0.5, -0.02))
    
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.12)
    
    # Save
    plt.savefig(OUTPUT_DIR / f'{output_name}.png', dpi=150, bbox_inches='tight')
    plt.savefig(OUTPUT_DIR / f'{output_name}.pdf', bbox_inches='tight')
    plt.close()
    
    print(f"Saved: {output_name}.png/.pdf")


def main():
    """Main function to create all k-line visualizations."""
    
    # Adult dataset k-line plots (existing results from author)
    adult2_df = load_results(RESULTS_DIRS['k_line_experiment'], 'adult2_results.csv')
    adult5_df = load_results(RESULTS_DIRS['k_line_experiment'], 'adult5_results.csv')
    
    if adult2_df is not None and adult5_df is not None:
        # Prepare data
        adult2_data = prepare_kline_data(adult2_df, ALGORITHM_STYLES.keys())
        adult5_data = prepare_kline_data(adult5_df, ALGORITHM_STYLES.keys())
        
        # Create comparison plot
        plot_kline_comparison(
            adult2_data, adult5_data,
            DATASET_NAMES['adult2'], DATASET_NAMES['adult5'],
            'kline_adult_comparison'
        )
    
    # COMPAS dataset k-line plots
    compas_df = load_results(RESULTS_DIRS['compas_experiment'], 'compas_results.csv')
    compas_sex_df = load_results(RESULTS_DIRS['compas_experiment'], 'compas_sex_results.csv')
    
    if compas_df is not None and compas_sex_df is not None:
        # Prepare data
        compas_data = prepare_kline_data(compas_df, ALGORITHM_STYLES.keys())
        compas_sex_data = prepare_kline_data(compas_sex_df, ALGORITHM_STYLES.keys())
        
        # Create comparison plot
        plot_kline_comparison(
            compas_data, compas_sex_data,
            DATASET_NAMES['compas'], DATASET_NAMES['compas_sex'],
            'kline_compas_comparison'
        )
        
        # Individual plots
        if compas_data:
            plot_kline(compas_data, DATASET_NAMES['compas'], 'kline_compas')
        if compas_sex_data:
            plot_kline(compas_sex_data, DATASET_NAMES['compas_sex'], 'kline_compas_sex')
    
    print("K-Line visualization completed!")


if __name__ == "__main__":
    main()

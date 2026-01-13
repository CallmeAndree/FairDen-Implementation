"""
Visualization for Multi-Attribute Fair Clustering experiments
Creates comparison between Author's results and Our results
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


def load_adult_results():
    """Load adult multi-attribute experiment results"""
    data_dir = Path('results/adult_multi_exp')
    
    results = {}
    for f in sorted(data_dir.glob('experimental_adult_*.csv')):
        setting = f.stem.replace('experimental_adult_', '')
        df = pd.read_csv(f)
        fairden = df[df['Algorithm'].str.contains('FairDen', case=False, na=False)]
        if len(fairden) > 0:
            results[setting] = fairden.iloc[0]
    
    return results


def load_census_results():
    """Load census multi-attribute experiment results"""
    data_dir = Path('results/multi_attr')
    results = {}
    
    # Map file names to setting keys
    file_map = {
        'experimental_cens_gender.csv': 'g',
        'experimental_cens_marital.csv': 'm',
        'experimental_cens_race.csv': 'r',
        'experimental_cens_gender_marital.csv': 'gm',
        'experimental_cens_gender_race.csv': 'gr',
        'experimental_cens_race_marital.csv': 'mr',
        'experimental_cens_intersectional.csv': 'gmr',
    }
    
    for filename, setting in file_map.items():
        filepath = data_dir / filename
        if filepath.exists():
            df = pd.read_csv(filepath)
            fairden = df[df['Algorithm'].str.contains('FairDen', case=False, na=False)]
            if len(fairden) > 0:
                row = fairden.iloc[0]
                # Map column names to standard format
                results[setting] = {
                    'Balance_gender': row.get('Balance_gender', 0),
                    'Balance_marital_status': row.get('Balance_marital_status', 0),
                    'Balance_race': row.get('Balance_race', 0),
                }
    
    return results


def create_heatmap_panel(ax, data, row_labels, col_labels, title=None, highlight_cells=None):
    """Create a single heatmap panel"""
    cmap = plt.cm.Greens
    im = ax.imshow(data, cmap=cmap, vmin=0, vmax=1, aspect='equal')
    
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            val = data[i, j]
            color = 'white' if val > 0.6 else 'black'
            ax.text(j, i, f'{val:.2f}', ha='center', va='center', 
                   fontsize=11, fontweight='bold', color=color)
    
    ax.set_xticks(range(len(col_labels)))
    ax.set_xticklabels(col_labels)
    ax.set_yticks(range(len(row_labels)))
    ax.set_yticklabels(row_labels)
    ax.set_xlabel('Balance', fontsize=10)
    
    if title:
        ax.set_title(title, fontsize=11, pad=5)
    
    return im


def create_author_visualization():
    """Create visualization with author's exact data from Figure 3"""
    
    # Author's data from Figure 3
    # Single: G, M, R (rows) vs Balance_G, Balance_M, Balance_R (cols)
    author_single = np.array([
        [0.96, 0.66, 0.73],  # G
        [0.94, 0.76, 0.79],  # M
        [0.90, 0.60, 0.91],  # R
    ])
    
    # Double: G&M, G&R, M&R
    author_double = np.array([
        [0.97, 0.74, 0.82],  # G&M
        [1.00, 0.50, 0.82],  # G&R
        [0.93, 0.90, 0.85],  # M&R
    ])
    
    # Triple: G&M&R
    author_triple = np.array([[0.87, 0.83, 0.78]])
    
    # Create figure
    fig = plt.figure(figsize=(14, 4))
    fig.patch.set_facecolor('white')
    
    # Panel 1
    ax1 = fig.add_axes([0.05, 0.15, 0.22, 0.7])
    create_heatmap_panel(ax1, author_single, ['G', 'M', 'R'], ['G', 'M', 'R'])
    ax1.set_ylabel('Sensitive attribute(s)', fontsize=10)
    
    # Panel 2
    ax2 = fig.add_axes([0.35, 0.15, 0.22, 0.7])
    create_heatmap_panel(ax2, author_double, ['G&M', 'G&R', 'M&R'], ['G', 'M', 'R'])
    
    # Panel 3
    ax3 = fig.add_axes([0.68, 0.32, 0.22, 0.28])
    create_heatmap_panel(ax3, author_triple, ['G&M&R'], ['G', 'M', 'R'])
    
    output_dir = Path('visualization')
    output_dir.mkdir(exist_ok=True)
    plt.savefig(output_dir / 'multi_attr_author.png', dpi=150, bbox_inches='tight', facecolor='white')
    print(f"Saved: {output_dir}/multi_attr_author.png")
    plt.close()


def create_our_visualization(results, title, output_name):
    """Create our results visualization"""
    
    single_data = np.zeros((3, 3))
    double_data = np.zeros((3, 3))
    triple_data = np.zeros((1, 3))
    
    single_settings = ['g', 'm', 'r']
    for i, setting in enumerate(single_settings):
        if setting in results:
            row = results[setting]
            single_data[i] = [
                row.get('Balance_gender', 0),
                row.get('Balance_marital_status', 0),
                row.get('Balance_race', 0)
            ]
    
    double_settings = ['gm', 'gr', 'mr']
    for i, setting in enumerate(double_settings):
        if setting in results:
            row = results[setting]
            double_data[i] = [
                row.get('Balance_gender', 0),
                row.get('Balance_marital_status', 0),
                row.get('Balance_race', 0)
            ]
    
    if 'gmr' in results:
        row = results['gmr']
        triple_data[0] = [
            row.get('Balance_gender', 0),
            row.get('Balance_marital_status', 0),
            row.get('Balance_race', 0)
        ]
    
    fig = plt.figure(figsize=(14, 4))
    fig.patch.set_facecolor('white')
    
    ax1 = fig.add_axes([0.05, 0.15, 0.22, 0.7])
    create_heatmap_panel(ax1, single_data, ['G', 'M', 'R'], ['G', 'M', 'R'])
    ax1.set_ylabel('Sensitive attribute(s)', fontsize=10)
    
    ax2 = fig.add_axes([0.35, 0.15, 0.22, 0.7])
    create_heatmap_panel(ax2, double_data, ['G&M', 'G&R', 'M&R'], ['G', 'M', 'R'])
    
    ax3 = fig.add_axes([0.68, 0.32, 0.22, 0.28])
    create_heatmap_panel(ax3, triple_data, ['G&M&R'], ['G', 'M', 'R'])
    
    output_dir = Path('visualization')
    plt.savefig(output_dir / f'{output_name}.png', dpi=150, bbox_inches='tight', facecolor='white')
    print(f"Saved: {output_dir}/{output_name}.png")
    plt.close()


def create_comparison_figure():
    """Create side-by-side comparison of author vs our results"""
    
    adult_results = load_adult_results()
    
    # Author's data
    author_single = np.array([
        [0.96, 0.66, 0.73],
        [0.94, 0.76, 0.79],
        [0.90, 0.60, 0.91],
    ])
    author_double = np.array([
        [0.97, 0.74, 0.82],
        [1.00, 0.50, 0.82],
        [0.93, 0.90, 0.85],
    ])
    author_triple = np.array([[0.87, 0.83, 0.78]])
    
    # Our data
    our_single = np.zeros((3, 3))
    our_double = np.zeros((3, 3))
    our_triple = np.zeros((1, 3))
    
    for i, s in enumerate(['g', 'm', 'r']):
        if s in adult_results:
            row = adult_results[s]
            our_single[i] = [row.get('Balance_gender', 0), row.get('Balance_marital_status', 0), row.get('Balance_race', 0)]
    
    for i, s in enumerate(['gm', 'gr', 'mr']):
        if s in adult_results:
            row = adult_results[s]
            our_double[i] = [row.get('Balance_gender', 0), row.get('Balance_marital_status', 0), row.get('Balance_race', 0)]
    
    if 'gmr' in adult_results:
        row = adult_results['gmr']
        our_triple[0] = [row.get('Balance_gender', 0), row.get('Balance_marital_status', 0), row.get('Balance_race', 0)]
    
    # Create 2-row figure
    fig = plt.figure(figsize=(14, 8))
    fig.patch.set_facecolor('white')
    
    # Row 1: Author
    ax1 = fig.add_axes([0.05, 0.55, 0.20, 0.35])
    create_heatmap_panel(ax1, author_single, ['G', 'M', 'R'], ['G', 'M', 'R'])
    ax1.set_ylabel("Author's", fontsize=11, fontweight='bold')
    
    ax2 = fig.add_axes([0.32, 0.55, 0.20, 0.35])
    create_heatmap_panel(ax2, author_double, ['G&M', 'G&R', 'M&R'], ['G', 'M', 'R'])
    
    ax3 = fig.add_axes([0.62, 0.62, 0.20, 0.18])
    create_heatmap_panel(ax3, author_triple, ['G&M&R'], ['G', 'M', 'R'])
    
    # Row 2: Ours
    ax4 = fig.add_axes([0.05, 0.10, 0.20, 0.35])
    create_heatmap_panel(ax4, our_single, ['G', 'M', 'R'], ['G', 'M', 'R'])
    ax4.set_ylabel("Ours", fontsize=11, fontweight='bold')
    
    ax5 = fig.add_axes([0.32, 0.10, 0.20, 0.35])
    create_heatmap_panel(ax5, our_double, ['G&M', 'G&R', 'M&R'], ['G', 'M', 'R'])
    
    ax6 = fig.add_axes([0.62, 0.17, 0.20, 0.18])
    create_heatmap_panel(ax6, our_triple, ['G&M&R'], ['G', 'M', 'R'])
    
    output_dir = Path('visualization')
    plt.savefig(output_dir / 'multi_attr_comparison.png', dpi=150, bbox_inches='tight', facecolor='white')
    plt.savefig(output_dir / 'multi_attr_comparison.pdf', bbox_inches='tight', facecolor='white')
    print(f"Saved: {output_dir}/multi_attr_comparison.png")
    plt.close()


if __name__ == "__main__":
    # Create author's visualization
    create_author_visualization()
    
    # Create our visualization (Adult)
    adult_results = load_adult_results()
    if adult_results:
        create_our_visualization(adult_results, "Our Results (Adult Dataset)", 'multi_attr_adult')
    
    # Create Census visualization
    census_results = load_census_results()
    if census_results:
        create_our_visualization(census_results, "Our Results (Census Dataset)", 'multi_attr_census')
    
    # Create comparison
    create_comparison_figure()
    
    print("All visualizations created!")

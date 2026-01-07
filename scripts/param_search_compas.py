"""
Parameter optimization script for COMPAS dataset.
Searches for optimal (minPts, eps) parameters by maximizing DCSI score.
Similar to the approach described in Section 2 of the Report.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.cluster import DBSCAN
from tqdm import tqdm
import json

from src.utils.DataLoader import DataLoader
from src.evaluation.dcsi import dcsiscore
from src.evaluation.balance import balance_score
from src.evaluation.noise import noise_percent


def search_optimal_params(dataname: str, output_file: str = None):
    """
    Search for optimal DBSCAN parameters for a given dataset configuration.
    
    Parameters:
    -----------
    dataname: str
        Name of the dataset config (e.g., 'compas', 'compas2', 'compas_sex')
    output_file: str
        Optional path to save results CSV
    
    Returns:
    --------
    DataFrame with all parameter combinations and their scores
    """
    print(f"\n{'='*60}")
    print(f"Parameter Search for: {dataname}")
    print(f"{'='*60}")
    
    # Parameter space as defined in the paper
    minPts_values = [4, 5, 7, 10, 15]  # {4, 5, 2*d_n-1=7, 10, 15}
    eps_values = [0.01, 0.05] + [round(0.1 + i*0.1, 1) for i in range(5)] + \
                 [round(0.6 + i*0.1, 1) for i in range(20)] + \
                 [2.6, 2.8, 3.0, 3.25, 3.5, 3.75]
    
    # Remove duplicates and sort
    eps_values = sorted(list(set(eps_values)))
    
    print(f"minPts values: {minPts_values}")
    print(f"eps values: {len(eps_values)} values from {min(eps_values)} to {max(eps_values)}")
    print(f"Total combinations: {len(minPts_values) * len(eps_values)}")
    
    # Load data
    dataloader = DataLoader(dataname, categorical=False)
    dataloader.load()
    data = dataloader.get_data()
    sens_attr = dataloader.get_sens_attr()
    sensitive_columns = dataloader.get_sensitive_columns()
    
    print(f"Data shape: {data.shape}")
    print(f"Sensitive attributes: {sens_attr}")
    
    results = []
    dcsi_min_pts = 5  # Fixed for DCSI evaluation
    
    # Grid search
    total = len(minPts_values) * len(eps_values)
    pbar = tqdm(total=total, desc="Searching")
    
    for minPts in minPts_values:
        for eps in eps_values:
            try:
                # Run DBSCAN
                dbscan = DBSCAN(eps=eps, min_samples=minPts).fit(data)
                labels = dbscan.labels_
                
                # Count clusters (excluding noise)
                n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
                
                # Skip if no valid clusters
                if n_clusters < 2:
                    pbar.update(1)
                    continue
                
                # Calculate metrics
                dcsi = dcsiscore(data, labels, min_pts=dcsi_min_pts)
                noise = noise_percent(labels)
                
                # Calculate balance
                try:
                    if len(sens_attr) == 1:
                        balance, _, _ = balance_score(
                            dataname, sens_attr, np.array(labels),
                            sensitive_columns, per_cluster=True
                        )
                    else:
                        balance, _, _ = balance_score(
                            dataname, sens_attr, np.array(labels),
                            sensitive_columns, per_cluster=True
                        )
                except Exception as e:
                    balance = -1
                
                results.append({
                    'minPts': minPts,
                    'eps': eps,
                    'n_clusters': n_clusters,
                    'DCSI': dcsi,
                    'Balance': balance,
                    'Noise%': noise * 100
                })
                
            except Exception as e:
                pass
            
            pbar.update(1)
    
    pbar.close()
    
    # Create DataFrame and sort by DCSI
    df = pd.DataFrame(results)
    if len(df) > 0:
        df = df.sort_values('DCSI', ascending=False).reset_index(drop=True)
        
        print(f"\n{'='*60}")
        print(f"Results for {dataname}")
        print(f"{'='*60}")
        print(f"\nTop 10 configurations by DCSI:")
        print(df.head(10).to_string(index=False))
        
        print(f"\nBest configuration:")
        best = df.iloc[0]
        print(f"  minPts = {int(best['minPts'])}")
        print(f"  eps = {best['eps']}")
        print(f"  DCSI = {best['DCSI']:.4f}")
        print(f"  Balance = {best['Balance']}")
        print(f"  Clusters = {int(best['n_clusters'])}")
        print(f"  Noise% = {best['Noise%']:.2f}%")
        
        # Best eps for each minPts
        print(f"\nBest eps for each minPts:")
        for minPts in minPts_values:
            subset = df[df['minPts'] == minPts]
            if len(subset) > 0:
                best_row = subset.iloc[0]
                print(f"  minPts={minPts}: eps={best_row['eps']}, DCSI={best_row['DCSI']:.4f}, clusters={int(best_row['n_clusters'])}")
        
        # Save results
        if output_file:
            df.to_csv(output_file, index=False)
            print(f"\nResults saved to: {output_file}")
    else:
        print("No valid configurations found!")
    
    return df


def main():
    """Run parameter search for all COMPAS configurations."""
    Path('results/param_search').mkdir(parents=True, exist_ok=True)
    
    configs = [
        ('compas', 'race only'),
        ('compas_sex', 'sex only'),
        ('compas2', 'race + sex')
    ]
    
    all_results = {}
    
    for dataname, desc in configs:
        print(f"\n\n{'#'*70}")
        print(f"# Configuration: {dataname} ({desc})")
        print(f"{'#'*70}")
        
        output_file = f'results/param_search/{dataname}_param_search.csv'
        df = search_optimal_params(dataname, output_file)
        all_results[dataname] = df
    
    # Summary
    print(f"\n\n{'='*70}")
    print("SUMMARY: Optimal Parameters for Each Configuration")
    print(f"{'='*70}")
    print(f"{'Config':<15} {'Sensitive':<15} {'minPts':<8} {'eps':<8} {'DCSI':<10} {'Balance':<12} {'Clusters':<10}")
    print("-" * 80)
    
    for dataname, desc in configs:
        df = all_results[dataname]
        if len(df) > 0:
            best = df.iloc[0]
            balance_str = str(best['Balance'])[:10] if isinstance(best['Balance'], dict) else f"{best['Balance']:.4f}"
            print(f"{dataname:<15} {desc:<15} {int(best['minPts']):<8} {best['eps']:<8} {best['DCSI']:<10.4f} {balance_str:<12} {int(best['n_clusters']):<10}")
    
    print(f"\nDetailed results saved in: results/param_search/")


if __name__ == "__main__":
    main()

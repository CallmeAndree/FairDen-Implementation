# COMPAS Hyperparameter Optimization
# Following paper methodology:
# - minPts: {4, 5, 10, 15, 2d-1}
# - epsilon: {0.01, 0.05, 0.1, ..., 0.5, 0.6, ..., 2.5, 2.6, 2.8, 3, 3.25, 3.5, 3.75}
# - DCSI minPts = 5 (constant)
# Criterion: Maximize DCSI score

import sys
from pathlib import Path

# Add project root to path for direct execution
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))

import json
import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN
from tqdm import tqdm

from src.utils.DataLoader import DataLoader
from src.evaluation.dcsi import dcsiscore
from src.evaluation.balance import balance_score
from src.evaluation.noise import noise_percent


def get_balance_value(balance_result):
    """Extract balance value from result (handles dict for multiple sensitive attrs)."""
    if isinstance(balance_result, dict):
        return min(balance_result.values())
    return balance_result


def update_config(dataname: str, best_params: dict):
    """Update config file with optimal parameters."""
    config_path = f'config/realworld/{dataname}.json'
    
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    config['DBSCAN_min_pts'] = best_params['minPts']
    config['DBSCAN_eps'] = best_params['epsilon']
    
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=4)
    
    print(f"\n*** Config updated: {config_path} ***")
    print(f"  DBSCAN_min_pts: {best_params['minPts']}")
    print(f"  DBSCAN_eps: {best_params['epsilon']}")


def compas_hyperparameter_search(dataname: str, output_dir: str):
    """
    Hyperparameter optimization for COMPAS dataset.
    Find best minPts and epsilon based on DCSI score.
    """
    DCSI_MINPTS = 5  # Constant for DCSI evaluation as per paper
    
    # Create results directory
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Load data
    dataloader = DataLoader(dataname, categorical=False)
    dataloader.load()
    data = dataloader.get_data()
    
    d = data.shape[1]  # Number of dimensions
    n = data.shape[0]  # Number of samples
    
    print(f"\n{'='*60}")
    print(f"Dataset: {dataname}")
    print(f"{'='*60}")
    print(f"Samples (n): {n}")
    print(f"Dimensions (d): {d}")
    print(f"Sensitive attributes: {dataloader.get_sens_attr()}")
    
    # Parameter space as per paper
    minpts_candidates = [4, 5, 10, 15, 2*d - 1]
    minpts_candidates = sorted(list(set(minpts_candidates)))
    
    # Epsilon values as per paper
    eps_values = list(np.arange(0.01, 0.06, 0.04))  # 0.01, 0.05
    eps_values += list(np.arange(0.1, 0.6, 0.1))     # 0.1, 0.2, 0.3, 0.4, 0.5
    eps_values += list(np.arange(0.6, 2.6, 0.1))     # 0.6, 0.7, ..., 2.5
    eps_values += [2.6, 2.8, 3.0, 3.25, 3.5, 3.75]
    eps_values = sorted(list(set([round(e, 2) for e in eps_values])))
    
    print(f"\nParameter Space:")
    print(f"minPts candidates: {minpts_candidates}")
    print(f"Epsilon values: {len(eps_values)} values from {eps_values[0]} to {eps_values[-1]}")
    print(f"DCSI minPts (constant): {DCSI_MINPTS}")
    
    results = []
    best_dcsi = -1
    best_params = {}
    
    total_combos = len(minpts_candidates) * len(eps_values)
    print(f"\nTotal combinations to test: {total_combos}")
    
    for minpts in minpts_candidates:
        for eps in tqdm(eps_values, desc=f"minPts={minpts}"):
            # Run DBSCAN
            dbscan = DBSCAN(eps=eps, min_samples=minpts).fit(data)
            labels = dbscan.labels_
            
            # Count clusters (excluding noise)
            n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
            noise_pct = noise_percent(labels)
            
            # Calculate DCSI with constant minPts=5
            if n_clusters > 0:
                dcsi = dcsiscore(data, labels, min_pts=DCSI_MINPTS)
                
                # Calculate balance
                try:
                    balance_result = balance_score(
                        dataname, dataloader.get_sens_attr(), np.array(labels),
                        dataloader.get_sensitive_columns(), per_cluster=True
                    )
                    balance = get_balance_value(balance_result[0])
                except Exception as e:
                    balance = 0
            else:
                dcsi = 0
                balance = 0
            
            results.append({
                'minPts': minpts,
                'epsilon': eps,
                'n_clusters': n_clusters,
                'DCSI': round(dcsi, 4) if dcsi > 0 else 0,
                'Balance': round(balance, 4) if balance > 0 else 0,
                'Noise%': round(noise_pct * 100, 2)
            })
            
            # Track best: prioritize low noise (< 50%) with best DCSI
            noise_pct_100 = noise_pct * 100
            if dcsi > 0 and n_clusters > 1 and noise_pct_100 < 50:
                if dcsi > best_dcsi:
                    best_dcsi = dcsi
                    best_params = {
                        'minPts': minpts,
                        'epsilon': eps,
                        'DCSI': dcsi,
                        'n_clusters': n_clusters,
                        'Balance': balance,
                        'Noise%': noise_pct_100
                    }
    
    # Save all results
    df = pd.DataFrame(results)
    df.to_csv(f'{output_dir}/{dataname}_all_results.csv', index=False)
    
    # Print summary
    print("\n" + "="*80)
    print(f"HYPERPARAMETER OPTIMIZATION RESULTS - {dataname}")
    print("="*80)
    
    if best_params:
        print(f"\n*** BEST CONFIGURATION (DCSI max with Noise < 50%) ***")
        print(f"  minPts: {best_params['minPts']}")
        print(f"  epsilon: {best_params['epsilon']}")
        print(f"  DCSI: {best_params['DCSI']:.4f}")
        print(f"  Balance: {best_params['Balance']:.4f}")
        print(f"  Clusters: {best_params['n_clusters']}")
        print(f"  Noise%: {best_params['Noise%']:.2f}%")
    else:
        print("No valid clustering configuration found with Noise < 50%!")
        return None, df
    
    # Top 10 by DCSI
    print("\n*** TOP 10 Configurations by DCSI ***")
    top10 = df[df['DCSI'] > 0].nlargest(10, 'DCSI')
    print(top10.to_string(index=False))
    
    # Best for each minPts
    print("\n*** Best epsilon for each minPts ***")
    for minpts in minpts_candidates:
        subset = df[(df['minPts'] == minpts) & (df['DCSI'] > 0)]
        if len(subset) > 0:
            best = subset.loc[subset['DCSI'].idxmax()]
            print(f"  minPts={minpts}: eps={best['epsilon']}, DCSI={best['DCSI']:.4f}, clusters={int(best['n_clusters'])}")
        else:
            print(f"  minPts={minpts}: No valid clustering found")
    
    # Save best params
    with open(f'{output_dir}/{dataname}_best_params.txt', 'w') as f:
        f.write(f"COMPAS ({dataname}) - Best Hyperparameters (by DCSI)\n")
        f.write("="*50 + "\n")
        f.write(f"minPts: {best_params['minPts']}\n")
        f.write(f"epsilon: {best_params['epsilon']}\n")
        f.write(f"DCSI: {best_params['DCSI']:.4f}\n")
        f.write(f"Balance: {best_params['Balance']:.4f}\n")
        f.write(f"Clusters: {best_params['n_clusters']}\n")
        f.write(f"Noise%: {best_params['Noise%']:.2f}%\n")
    
    print(f"\nResults saved to: {output_dir}/")
    print("="*80)
    
    return best_params, df


def main():
    """Run hyperparameter search for all COMPAS configurations."""
    output_dir = 'results/compas_hyperparam'
    
    configs = [
        ('compas', 'race only'),
        ('compas_sex', 'sex only'),
        ('compas2', 'race + sex')
    ]
    
    all_best = {}
    
    for dataname, desc in configs:
        print(f"\n\n{'#'*70}")
        print(f"# Configuration: {dataname} (sensitive: {desc})")
        print(f"{'#'*70}")
        
        best_params, df = compas_hyperparameter_search(dataname, output_dir)
        all_best[dataname] = best_params
        
        # Auto-update config with optimal parameters
        if best_params:
            update_config(dataname, best_params)
    
    # Final summary
    print("\n\n" + "="*80)
    print("FINAL SUMMARY: Optimal Parameters for COMPAS Dataset")
    print("="*80)
    print(f"{'Config':<15} {'Sensitive':<15} {'minPts':<8} {'eps':<8} {'DCSI':<10} {'Balance':<10} {'Clusters':<10}")
    print("-" * 90)
    
    for dataname, desc in configs:
        best = all_best[dataname]
        if best:
            print(f"{dataname:<15} {desc:<15} {best['minPts']:<8} {best['epsilon']:<8} {best['DCSI']:<10.4f} {best['Balance']:<10.4f} {best['n_clusters']:<10}")
    
    print(f"\nDetailed results saved in: {output_dir}/")


if __name__ == "__main__":
    main()

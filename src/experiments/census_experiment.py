# Census Experiment - FairDen only
# Runs FairDen on all cens_xxx configs and calculates balance for gender, race, marital_status
# regardless of what's specified in the config

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))

import json
import numpy as np
import pandas as pd
from tqdm import tqdm

from src.utils.DataLoader import DataLoader
from src.FairDen import FairDen
from src.evaluation.balance import balance_score
from src.evaluation.dcsi import dcsiscore
from src.evaluation.noise import noise_percent
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score


def census_experiment():
    """
    Run FairDen on all cens_xxx configurations.
    Calculate balance for all 3 sensitive attributes: gender, race, marital_status.
    """
    
    # Output directory
    output_dir = Path('results/cens_experiment')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Configs to process - all 7 Census configs
    configs = [
        ('cens_gender', 'gender'),
        ('cens_race', 'race'),
        ('cens_marital', 'marital_status'),
        ('cens_gender_race', 'gender + race'),
        ('cens_gender_marital', 'gender + marital_status'),
        ('cens_race_marital', 'race + marital_status'),
        ('cens_intersectional', 'all 3 attrs'),
    ]
    
    # All sensitive attributes we want to calculate balance for
    ALL_SENS_ATTRS = ['gender', 'race', 'marital_status']
    
    results = []
    
    for dataname, config_sens in tqdm(configs, desc="Census Experiment"):
        print(f"\n{'='*60}")
        print(f"Processing: {dataname} (config sensitive: {config_sens})")
        print(f"{'='*60}")
        
        # Load config
        with open(f'config/realworld/{dataname}.json', 'r') as f:
            config = json.load(f)
        
        # Load data using DataLoader
        dataloader = DataLoader(dataname, categorical=False)
        dataloader.load()
        data = dataloader.get_data()
        
        # Get ground truth
        ground_truth = dataloader.get_target_columns()
        
        # Get DBSCAN params
        min_pts, eps = dataloader.get_dbscan_config()
        dcsi_min_pts = dataloader.get_dcsi_min_pts()
        n_clusters = dataloader.get_num_clusters()
        
        # Get all sensitive columns from DataLoader
        # DataLoader stores all 3 sensitive columns in __all_sensitive for adult/cens datasets
        all_sensitive_df = dataloader.get_all_sensitive()
        
        print(f"Data shape: {data.shape}")
        print(f"minPts: {min_pts}, eps: {eps}")
        print(f"n_clusters from config: {n_clusters}")
        
        # Run FairDen
        try:
            # Calculate heuristic min_pts for FairDen
            sens_count = len(config['sensitive_attrs']) if isinstance(config['sensitive_attrs'], list) else 1
            fairden_min_pts = max(2 * (data.shape[1] + sens_count) - 1, 20)
            
            print(f"Running FairDen with minPts={fairden_min_pts}")
            
            # Run FairDen - pass dataloader object, not raw data
            fairden = FairDen(
                dataloader,
                min_pts=fairden_min_pts,
            )
            
            labels = fairden.run(k=n_clusters)
            
            if labels is None:
                print(f"FairDen returned None for {dataname}")
                continue
            
            # Evaluate clustering quality
            n_clusters_found = len(set(labels)) - (1 if -1 in labels else 0)
            noise = noise_percent(labels)
            dcsi = dcsiscore(data, labels, min_pts=dcsi_min_pts)
            
            # ARI and NMI
            if ground_truth is not None:
                ari = adjusted_rand_score(ground_truth, labels)
                nmi = normalized_mutual_info_score(ground_truth, labels)
            else:
                ari, nmi = -1, -1
            
            print(f"Clusters found: {n_clusters_found}, Noise: {noise:.2%}, DCSI: {dcsi:.4f}")
            
            # Calculate balance for ALL three sensitive attributes
            balance_results = {}
            
            # The labels array length should match data length
            # Need to align all_sensitive_df with the actual data used
            n_labels = len(labels)
            
            for sens_attr in ALL_SENS_ATTRS:
                try:
                    # Get sensitive column from all_sensitive_df
                    # Reset index and take first n_labels rows to match labels length
                    sens_col = all_sensitive_df[sens_attr].reset_index(drop=True).iloc[:n_labels].values
                    
                    # Create DataFrame for balance calculation
                    sens_df = pd.DataFrame({sens_attr: sens_col})
                    
                    # Calculate balance
                    balance = balance_score(
                        dataname, [sens_attr], np.array(labels),
                        sens_df, per_cluster=True
                    )
                    
                    balance_results[sens_attr] = balance[0] if isinstance(balance, tuple) else balance
                    print(f"  Balance ({sens_attr}): {balance_results[sens_attr]:.4f}")
                except Exception as e:
                    print(f"  Error calculating balance for {sens_attr}: {e}")
                    balance_results[sens_attr] = -1
            
            # Save result
            results.append({
                'Config': dataname,
                'Config_Sensitive': config_sens,
                'N_Clusters': n_clusters_found,
                'Noise': noise,
                'DCSI': dcsi,
                'ARI': ari,
                'NMI': nmi,
                'Balance_Gender': balance_results.get('gender', -1),
                'Balance_Race': balance_results.get('race', -1),
                'Balance_Marital': balance_results.get('marital_status', -1),
            })
            
        except Exception as e:
            print(f"Error running FairDen on {dataname}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # Save results
    df_results = pd.DataFrame(results)
    df_results.to_csv(output_dir / 'census_fairden_results.csv', index=False)
    
    print("\n" + "="*80)
    print("CENSUS EXPERIMENT RESULTS - FairDen")
    print("="*80)
    print(df_results.to_string(index=False))
    print(f"\nResults saved to: {output_dir}/census_fairden_results.csv")
    
    return df_results


if __name__ == "__main__":
    census_experiment()

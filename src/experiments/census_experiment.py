# Census Multi-Attribute Experiment
# Similar structure to adult_experiment.py
# Runs FairDen on all cens_xxx configs and calculates balance for gender, race, marital_status

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))

import json
import numpy as np
import pandas as pd
from tqdm import tqdm

from sklearn.cluster import DBSCAN
from sklearn.metrics.cluster import normalized_mutual_info_score
from sklearn.metrics import adjusted_rand_score

from src.utils.DataLoader import DataLoader
from src.FairDen import FairDen
from src.evaluation.balance import balance_score, balance_mixed
from src.evaluation.dcsi import dcsiscore
from src.evaluation.noise import noise_percent


def census_experiment():
    """
    Run FairDen on all cens_xxx configurations.
    Calculate balance for all 3 sensitive attributes: gender, race, marital_status.
    Similar to adult_experiment() structure.
    """
    
    # Configs to process - all 7 Census configs with their settings
    configs = [
        ('cens_gender', 'G'),
        ('cens_race', 'R'),
        ('cens_marital', 'M'),
        ('cens_gender_race', 'G&R'),
        ('cens_gender_marital', 'G&M'),
        ('cens_race_marital', 'M&R'),
        ('cens_intersectional', 'G&M&R'),
    ]
    
    MIN_PTS = [15]
    attr1, attr2, attr3 = ['gender', 'race', 'marital_status']
    
    # Output directory
    output_dir = Path('results/multi_attr')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    for dataname, setting in tqdm(configs, desc="Census Multi-Attr Experiment"):
        print(f"\n{'='*60}")
        print(f"Processing: {dataname} (Setting: {setting})")
        print(f"{'='*60}")
        
        experimental = []
        
        # Load data
        dataloader = DataLoader(dataname, categorical=False)
        dataloader.load()
        
        ground_truth = dataloader.get_target_columns()
        degree = dataloader.get_num_clusters()
        data_sensitive = dataloader.get_data()
        data_sensitive = np.array(data_sensitive)
        min_pts, eps = dataloader.get_dbscan_config()
        
        # Calculate DBSCAN clustering for density-based ground-truth
        dbscan = DBSCAN(eps=eps, min_samples=min_pts).fit(data_sensitive)
        ground_truth_db = dbscan.labels_
        
        # Transform to numpy array - handle case when ground_truth is None or scalar
        if ground_truth is None or (hasattr(ground_truth, '__len__') == False):
            # Use DBSCAN labels as ground truth if no target available
            labels = ground_truth_db.copy()
            ground_truth = ground_truth_db.copy()
        else:
            labels = np.array(ground_truth)
            ground_truth = np.array(ground_truth)
        
        lab, count = np.unique(labels, return_counts=True)
        
        # Evaluate Ground Truth clustering
        balance, ari, nmi, dcsi, noise, ari_db, nmi_db, balance_per_cluster, balance_per_group_per_cluster = evaluate(
            labels, dataname, dataloader, ground_truth,
            ground_truth_db, data_sensitive)
        
        # Evaluate mixed balances
        balance_mix, balance_per_cluster_mix, balance_per_group_per_cluster_mix = evaluate_balance_mixed(
            labels, dataname, dataloader)
        
        # Save results for ground truth clustering
        experimental.append({
            "Setting": setting, "Degree": degree, "Algorithm": 'GroundTruth', "min_pts": min_pts,
            f"Balance_{attr1}": balance.get(attr1, -1),
            f"Balance_{attr2}": balance.get(attr2, -1),
            f"Balance_{attr3}": balance.get(attr3, -1),
            "Balance_Mixed": balance_mix,
            'Cluster_0': count[0] if len(count) > 0 else 0,
            "Cluster_1": count[1] if len(count) > 1 else 0,
        })
        
        # Evaluate DBSCAN clustering
        labels = ground_truth_db
        lab, count = np.unique(labels, return_counts=True)
        
        balance, ari, nmi, dcsi, noise, ari_db, nmi_db, balance_per_cluster, balance_per_group_per_cluster = evaluate(
            ground_truth_db, dataname, dataloader, ground_truth,
            ground_truth_db, data_sensitive)
        
        balance_mix, balance_per_cluster_mix, balance_per_group_per_cluster_mix = evaluate_balance_mixed(
            labels, dataname, dataloader)
        
        # Save results for DBSCAN clustering
        experimental.append({
            "Setting": setting, "Degree": degree, "Algorithm": 'DBSCAN', "min_pts": min_pts,
            f"Balance_{attr1}": balance.get(attr1, -1),
            f"Balance_{attr2}": balance.get(attr2, -1),
            f"Balance_{attr3}": balance.get(attr3, -1),
            "Balance_Mixed": balance_mix,
            'Cluster_0': count[0] if len(count) > 0 else 0,
            "Cluster_1": count[1] if len(count) > 1 else 0,
        })
        
        # Run FairDen
        for degree in [degree]:
            for min_pts in tqdm(MIN_PTS, desc=f"MinPts for {dataname}"):
                print(f'Checking minpts {min_pts}.')
                
                # Create FairDen object
                algorithm = FairDen(dataloader, min_pts=min_pts, alpha='0')
                labels = algorithm.run(degree)
                
                if labels is None:
                    print(f"FairDen returned None for {dataname}")
                    continue
                
                labels = np.array(labels)
                
                # Evaluate clusterings
                balance, ari, nmi, dcsi, noise, ari_db, nmi_db, balance_per_cluster, balance_per_group_per_cluster = evaluate(
                    labels, dataname, dataloader, ground_truth,
                    ground_truth_db, data_sensitive)
                
                # Evaluate clusterings regarding mixed sensitive attribute
                balance_mix, balance_per_cluster_mix, balance_per_group_per_cluster_mix = evaluate_balance_mixed(
                    labels, dataname, dataloader)
                
                lab, count = np.unique(labels, return_counts=True)
                
                algo_name = 'FairDen_v1' if -1 in labels else 'FairDen'
                
                result_dict = {
                    "Setting": setting, "Degree": degree, "Algorithm": algo_name, "min_pts": min_pts,
                    f"Balance_{attr1}": balance.get(attr1, -1),
                    f"Balance_{attr2}": balance.get(attr2, -1),
                    f"Balance_{attr3}": balance.get(attr3, -1),
                    "Balance_Mixed": balance_mix,
                    'Cluster_0': count[0] if len(count) > 0 else 0,
                    "Cluster_1": count[1] if len(count) > 1 else 0,
                }
                
                if len(count) > 2:
                    result_dict["Cluster_2"] = count[2]
                
                experimental.append(result_dict)
        
        # Save results
        df = pd.DataFrame(experimental)
        df.to_csv(output_dir / f"experimental_{dataname}.csv", index=False)
        print(f"Saved results to: {output_dir}/experimental_{dataname}.csv")
    
    print("\n" + "="*80)
    print("CENSUS MULTI-ATTR EXPERIMENT COMPLETED")
    print("="*80)


def evaluate(labels, dataname, dataloader, ground_truth, ground_truth_db, data):
    """
    Evaluate given clusterings.
    """
    min_pts = 5
    sensitive = dataloader.get_all_sensitive()
    
    # Align sensitive with labels length
    n_labels = len(labels)
    sensitive = sensitive.reset_index(drop=True).iloc[:n_labels]
    
    balance, balance_per_cluster, balance_per_group_per_cluster = balance_score(
        dataname, list(sensitive.columns), labels, sensitive, per_cluster=True)
    
    ari = adjusted_rand_score(labels, ground_truth[:n_labels])
    nmi = normalized_mutual_info_score(labels, ground_truth[:n_labels])
    ari_db = adjusted_rand_score(labels, ground_truth_db[:n_labels])
    nmi_db = normalized_mutual_info_score(labels, ground_truth_db[:n_labels])
    dcsi = dcsiscore(data[:n_labels], labels, min_pts=min_pts)
    noise = noise_percent(labels)
    
    return balance, ari, nmi, dcsi, noise, ari_db, nmi_db, balance_per_cluster, balance_per_group_per_cluster


def evaluate_balance_mixed(labels, dataname, dataloader):
    """
    Evaluate given clusterings regarding combined sensitive attributes.
    """
    try:
        sensitive = dataloader.get_sens_combi_mixed()
        
        # Align with labels length
        n_labels = len(labels)
        sensitive = sensitive.reset_index(drop=True).iloc[:n_labels]
        
        balance_mix, balance_per_cluster, balance_per_group_per_cluster = balance_mixed(
            dataname, ['combi'], labels, sensitive, per_cluster=True)
        return balance_mix, balance_per_cluster, balance_per_group_per_cluster
    except Exception as e:
        print(f"Error in evaluate_balance_mixed: {e}")
        return -1, {}, {}


if __name__ == "__main__":
    census_experiment()

# Copyright 2025 Forschungszentrum Juelich GmbH
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# COMPAS K-Line experiment implementation
# Similar to k_line_experiment.py but for COMPAS dataset

import numpy as np
import pandas as pd

from pathlib import Path

from sklearn.cluster import DBSCAN
from sklearn.metrics.cluster import normalized_mutual_info_score
from sklearn.metrics import adjusted_rand_score
from tqdm import tqdm

from src.utils.ClusteringAlgorithm import ClusteringAlgorithm
from src.utils.DataLoader import DataLoader
from src.evaluation.dcsi import dcsiscore
from src.evaluation.balance import balance_score
from src.evaluation.noise import noise_percent


def compas_experiment():
    """
    Run COMPAS k-line experiments with varying n_clusters (2, 4, 6, 8, 10).
    - compas: race as sensitive attribute (non-binary: 4 groups)
    - compas_sex: sex as sensitive attribute (binary: Male/Female)
    
    Results saved to results/compas_experiment/
    """
    # Create results directory if not exists
    Path('results/compas_experiment').mkdir(parents=True, exist_ok=True)
    
    # Run experiment for non-binary sensitive attribute (race)
    compas_kline_multi()
    
    # Run experiment for binary sensitive attribute (sex)
    compas_kline_binary()
    
    print("COMPAS k-line experiments completed!")


def compas_kline_multi():
    """
    K-line experiment for COMPAS with race (non-binary sensitive attribute).
    Algorithms: FairDen, FairSC_normalized, FairSC
    """
    DATANAMES = ["compas"]
    # Algorithms that can handle non-binary sensitive attributes
    ALGORITHMS = ['FairDen', 'FairSC_normalized', 'FairSC']
    # Number of clusters to test (following author's paper)
    N_CLUSTERS = [2, 4, 6, 8, 10]
    
    for dataname in tqdm(DATANAMES, desc="COMPAS (race)"):
        # result list
        results = []
        # create DataLoader object
        dataloader = DataLoader(dataname, categorical=False)
        dataloader.load()
        data = dataloader.get_data()
        min_pts, eps = dataloader.get_dbscan_config()
        
        # generate DBSCAN clustering for db ground truth
        dbscan = DBSCAN(eps=eps, min_samples=min_pts).fit(data)
        ground_truth_db = dbscan.labels_
        
        # Calculate min_pts using heuristic: 2 * (d + num_sens_attrs) - 1
        # But use at least 20 for numerical stability with COMPAS
        min_pts_calc = 2 * (data.shape[1] + len(dataloader.get_sens_attr())) - 1
        min_pts = max(min_pts_calc, 20)
        
        ground_truth = dataloader.get_target_columns()
        
        result_file = Path('results/compas_experiment/{}.csv'.format(dataname))
        # if the file exists load it
        if result_file.is_file():
            dataframe = pd.read_csv('results/compas_experiment/{}.csv'.format(dataname))
        else:
            dataframe = dataloader.get_data_frame()
            dataframe['GroundTruth'] = dataloader.get_target_columns()
        
        labels = np.array(ground_truth)
        # evaluate ground truth clustering
        balance, ari, nmi, dcsi, noise, degree, ari_db, nmi_db = evaluate(
            labels, dataname, dataloader, ground_truth, ground_truth_db, data
        )
        
        deg = len(set(labels)) - (1 if -1 in labels else 0)
        # save ground truth results
        results.append({
            "Data": dataname, 
            "Algorithm": 'GroundTruth', 
            'N_cluster': deg, 
            "min_pts": min_pts, 
            "DCSI": dcsi,
            "Balance": balance, 
            "ARI": ari,
            "NMI": nmi, 
            "Noise": noise, 
            "ARI_DB": ari_db, 
            "NMI_DB": nmi_db
        })
        dataframe['GroundTruth'] = labels
        
        # evaluate DBSCAN clustering
        labels = ground_truth_db
        balance, ari, nmi, dcsi, noise, degree, ari_db, nmi_db = evaluate(
            labels, dataname, dataloader, ground_truth, ground_truth_db, data
        )
        deg = len(set(labels)) - (1 if -1 in labels else 0)
        # save DBSCAN results
        results.append({
            "Data": dataname, 
            "Algorithm": 'GroundTruth_DB', 
            'N_cluster': deg, 
            "min_pts": min_pts, 
            "DCSI": dcsi,
            "Balance": balance, 
            "ARI": ari,
            "NMI": nmi, 
            "Noise": noise, 
            "ARI_DB": ari_db, 
            "NMI_DB": nmi_db
        })
        dataframe['GroundTruth_DB'] = labels
        
        # for multiple numbers of clusters
        for n_cluster in N_CLUSTERS:
            # for each algorithm
            for algo in tqdm(ALGORITHMS, desc=f"k={n_cluster}", leave=False):
                # create ClusteringAlgorithm object
                algorithm = ClusteringAlgorithm(algo, dataloader, min_pts, dataname)
                labels = algorithm.run(n_cluster)
                if labels is None:
                    balance, ari, nmi, dcsi, noise, degree, ari_db, nmi_db = -2, -2, -2, -2, -2, -2, -2, -2
                else:
                    dataframe[algo + str(n_cluster)] = labels
                    balance, ari, nmi, dcsi, noise, degree, ari_db, nmi_db = evaluate(
                        labels, dataname, dataloader, ground_truth, ground_truth_db, data
                    )
                # save results
                results.append({
                    "Data": dataname, 
                    "Algorithm": algo, 
                    "N_cluster": n_cluster,
                    "Actual_clusters": degree, 
                    "DCSI": dcsi, 
                    "Balance": balance,
                    "ARI": ari,
                    "NMI": nmi,
                    "Noise": noise, 
                    "ARI_DB": ari_db, 
                    "NMI_DB": nmi_db
                })
        
        # save the results to csvs
        dataframe.to_csv('results/compas_experiment/{}.csv'.format(dataname))
        df = pd.DataFrame(results)
        df.to_csv('results/compas_experiment/{}_results.csv'.format(dataname))


def compas_kline_binary():
    """
    K-line experiment for COMPAS with sex (binary sensitive attribute).
    Algorithms: Scalable, FairDen, FairSC_normalized, FairSC, Fairlet
    """
    DATANAMES = ["compas_sex"]
    # Algorithms that work with binary sensitive attributes
    ALGORITHMS = ['Scalable', 'FairDen', 'FairSC_normalized', 'FairSC', 'Fairlet']
    # Number of clusters to test
    N_CLUSTERS = [2, 4, 6, 8, 10]
    
    for dataname in tqdm(DATANAMES, desc="COMPAS (sex)"):
        # list for results
        results = []
        # create DataLoader objects
        dataloader = DataLoader(dataname, categorical=False)
        dataloader.load()
        data = dataloader.get_data()
        min_pts, eps = dataloader.get_dbscan_config()
        
        # generate DBSCAN clustering
        dbscan = DBSCAN(eps=eps, min_samples=min_pts).fit(data)
        ground_truth_db = dbscan.labels_
        
        # Calculate min_pts using heuristic
        min_pts_calc = 2 * (data.shape[1] + len(dataloader.get_sens_attr())) - 1
        min_pts = max(min_pts_calc, 20)
        
        ground_truth = dataloader.get_target_columns()
        
        result_file = Path('results/compas_experiment/{}.csv'.format(dataname))
        # if the file already exist load it
        if result_file.is_file():
            dataframe = pd.read_csv('results/compas_experiment/{}.csv'.format(dataname))
        else:
            dataframe = dataloader.get_data_frame()
            dataframe['GroundTruth'] = dataloader.get_target_columns()
        
        labels = np.array(ground_truth)
        # evaluate ground truth labels
        balance, ari, nmi, dcsi, noise, degree, ari_db, nmi_db = evaluate(
            labels, dataname, dataloader, ground_truth, ground_truth_db, data
        )
        
        deg = len(set(labels)) - (1 if -1 in labels else 0)
        # save evaluation results for ground truth labels
        results.append({
            "Data": dataname, 
            "Algorithm": 'GroundTruth', 
            'N_cluster': deg, 
            "min_pts": min_pts, 
            "DCSI": dcsi,
            "Balance": balance, 
            "ARI": ari,
            "NMI": nmi, 
            "Noise": noise, 
            "ARI_DB": ari_db, 
            "NMI_DB": nmi_db
        })
        dataframe['GroundTruth'] = labels
        
        labels = ground_truth_db
        deg = len(set(labels)) - (1 if -1 in labels else 0)
        # evaluate DBSCAN labels
        balance, ari, nmi, dcsi, noise, degree, ari_db, nmi_db = evaluate(
            labels, dataname, dataloader, ground_truth, ground_truth_db, data
        )
        
        # save results for DBSCAN
        results.append({
            "Data": dataname, 
            "Algorithm": 'GroundTruth_DB', 
            'N_cluster': deg, 
            "min_pts": min_pts, 
            "DCSI": dcsi,
            "Balance": balance, 
            "ARI": ari,
            "NMI": nmi, 
            "Noise": noise, 
            "ARI_DB": ari_db, 
            "NMI_DB": nmi_db
        })
        dataframe['GroundTruth_DB'] = labels
        
        # for each number of clusters
        for n_cluster in N_CLUSTERS:
            # for each algorithm
            for algo in tqdm(ALGORITHMS, desc=f"k={n_cluster}", leave=False):
                if algo == 'Fairlet':
                    try:
                        # create object for Fairlets (returns multiple versions)
                        algorithm = ClusteringAlgorithm(algo, dataloader, min_pts, dataname)
                        names, labelss = algorithm.run(n_cluster)
                        for name, labels in zip(names, labelss):
                            dataframe[name + str(n_cluster)] = labels
                            balance, ari, nmi, dcsi, noise, degree, ari_db, nmi_db = evaluate(
                                labels, dataname, dataloader, ground_truth, ground_truth_db, data
                            )
                            # save results for both Fairlet versions
                            results.append({
                                "Data": dataname, 
                                "Algorithm": name, 
                                "N_cluster": n_cluster,
                                "Actual_clusters": degree, 
                                "DCSI": dcsi, 
                                "Balance": balance,
                                "ARI": ari,
                                "NMI": nmi,
                                "Noise": noise, 
                                "ARI_DB": ari_db, 
                                "NMI_DB": nmi_db
                            })
                    except Exception as e:
                        print(f"Fairlet failed for k={n_cluster}: {e}")
                        # Record failure
                        for name in ['Fairlet_MCF Fairlet', 'Fairlet_VKC Fairlet']:
                            results.append({
                                "Data": dataname, 
                                "Algorithm": name, 
                                "N_cluster": n_cluster,
                                "Actual_clusters": -2, 
                                "DCSI": -2, 
                                "Balance": -2,
                                "ARI": -2,
                                "NMI": -2,
                                "Noise": -2, 
                                "ARI_DB": -2, 
                                "NMI_DB": -2
                            })
                else:
                    # create ClusteringAlgorithm object
                    algorithm = ClusteringAlgorithm(algo, dataloader, min_pts, dataname)
                    labels = algorithm.run(n_cluster)
                    if labels is None:
                        balance, ari, nmi, dcsi, noise, degree, ari_db, nmi_db = -2, -2, -2, -2, -2, -2, -2, -2
                    else:
                        dataframe[algo + str(n_cluster)] = labels
                        # evaluate clustering
                        balance, ari, nmi, dcsi, noise, degree, ari_db, nmi_db = evaluate(
                            labels, dataname, dataloader, ground_truth, ground_truth_db, data
                        )
                    # save results
                    results.append({
                        "Data": dataname, 
                        "Algorithm": algo, 
                        "N_cluster": n_cluster,
                        "Actual_clusters": degree, 
                        "DCSI": dcsi, 
                        "Balance": balance,
                        "ARI": ari,
                        "NMI": nmi,
                        "Noise": noise, 
                        "ARI_DB": ari_db, 
                        "NMI_DB": nmi_db
                    })
        
        # save dataframe results to csv
        dataframe.to_csv('results/compas_experiment/{}.csv'.format(dataname))
        df = pd.DataFrame(results)
        df.to_csv('results/compas_experiment/{}_results.csv'.format(dataname))


def evaluate(labels, dataname, dataloader, ground_truth, ground_truth_db, data):
    """
    Evaluate given clusterings.

    Parameters
    ----------
    labels: clustering labels.
    dataname: dataset name.
    dataloader: DataLoader object.
    ground_truth: Ground truth clustering labels.
    ground_truth_db: DBSCAN clustering labels.
    data: datapoints.

    Returns
    -------
        evaluation metrics for a given clustering regarding balance, external clustering validation,
        internal clustering validation and noise.
    """
    min_pts = 5
    degree = len(set(labels)) - (1 if -1 in labels else 0)
    # if clustering includes only noise points
    if degree != 0:
        balance, b1, b2 = balance_score(
            dataname, dataloader.get_sens_attr(), np.array(labels),
            dataloader.get_sensitive_columns(), per_cluster=True
        )
    else:
        balance = 0
    ari = adjusted_rand_score(labels, ground_truth)
    nmi = normalized_mutual_info_score(labels, ground_truth)
    ari_db = adjusted_rand_score(labels, ground_truth_db)
    nmi_db = normalized_mutual_info_score(labels, ground_truth_db)

    dcsi = dcsiscore(data, labels, min_pts=min_pts)
    noise = noise_percent(labels)
    return balance, ari, nmi, dcsi, noise, degree, ari_db, nmi_db


if __name__ == "__main__":
    compas_experiment()

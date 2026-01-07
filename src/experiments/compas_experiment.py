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
# COMPAS experiment implementation

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
    Run COMPAS experiments:
    - compas: race only as sensitive attribute
    - compas2: race + sex as combined sensitive attributes
    """
    # Experiment 1: race only, Experiment 2: race + sex, Experiment 3: sex only
    DATANAMES = ["compas", "compas2", "compas_sex"]
    # algorithms that can handle non-binary sensitive attributes
    ALGORITHMS = ['FairDen', 'FairSC_normalized', 'FairSC']
    
    # Create results directory if not exists
    Path('results/compas_experiment').mkdir(parents=True, exist_ok=True)
    
    # Delete old result files to ensure fresh results
    import os
    for f in Path('results/compas_experiment').glob('*.csv'):
        os.remove(f)
    
    # for each configuration
    for dataname in tqdm(DATANAMES):
        # result list
        results = []
        # construct DataLoader object
        dataloader = DataLoader(dataname, categorical=False)
        dataloader.load()
        degree_ = dataloader.get_num_clusters()
        degree_db = dataloader.get_n_clusters_db()
        data = dataloader.get_data()
        
        # generate DBSCAN clustering and labels
        min_pts, eps = dataloader.get_dbscan_config()
        dbscan = DBSCAN(eps=eps, min_samples=min_pts).fit(data)
        ground_truth_db = dbscan.labels_
        
        # define min pts - using 20 to avoid numerical noise in sqrtm()
        # Original heuristic: 2 * (data.shape[1] + len(sens_attr)) - 1
        # However, debug analysis shows min_pts=20 works reliably for COMPAS
        min_pts = 20
        ground_truth = dataloader.get_target_columns()
        
        result_file = Path('results/compas_experiment/{}.csv'.format(dataname))
        # if the result file already exists load content as dataframe
        if result_file.is_file():
            dataframe = pd.read_csv('results/compas_experiment/{}.csv'.format(dataname))
        # create a new dataframe
        else:
            dataframe = dataloader.get_data_frame()
            dataframe['GroundTruth'] = dataloader.get_target_columns()
        
        labels = np.array(ground_truth)
        # evaluate ground truth
        balance, ari, nmi, dcsi, noise, degree, ari_db, nmi_db = evaluate(
            labels, dataname, dataloader, ground_truth, ground_truth_db, data
        )
        
        attr1 = dataloader.get_sens_attr()
        # save results
        results.append({
            "Data": dataname, 
            "Algorithm": 'GroundTruth', 
            "min_pts": min_pts, 
            "DCSI": dcsi,
            "Balance": balance, 
            "ARI": ari,
            "NMI": nmi, 
            "Noise": noise, 
            "Categorical": "None", 
            "ARI_DB": ari_db, 
            "NMI_DB": nmi_db, 
            "N_cluster": degree
        })
        dataframe['GroundTruth'] = labels
        
        # evaluate DBSCAN clustering
        labels = ground_truth_db
        balance, ari, nmi, dcsi, noise, degree, ari_db, nmi_db = evaluate(
            labels, dataname, dataloader, ground_truth, ground_truth_db, data
        )
        
        # save results
        results.append({
            "Data": dataname, 
            "Algorithm": 'GroundTruth_DB', 
            "min_pts": min_pts, 
            "DCSI": dcsi,
            "Balance": balance, 
            "ARI": ari,
            "NMI": nmi, 
            "Noise": noise, 
            "Categorical": "None", 
            "ARI_DB": ari_db, 
            "NMI_DB": nmi_db, 
            "N_cluster": degree_db
        })
        dataframe['GroundTruth_DB'] = labels
        
        skip = False
        # for ground truth numbers of clusters and DBSCAN number of clusters
        for n_cluster in [degree_, degree_db]:
            # if both are the same the second run will be skipped
            if skip:
                continue
            # for each algorithm
            for algo in tqdm(ALGORITHMS):
                algorithm = ClusteringAlgorithm(algo, dataloader, min_pts, dataname)
                labels = algorithm.run(n_cluster)
                # if labels couldn't be generated add -2
                if labels is None:
                    balance, ari, nmi, dcsi, noise, degree, ari_db, nmi_db = -2, -2, -2, -2, -2, -2, -2, -2
                else:
                    dataframe[algo] = labels
                    balance, ari, nmi, dcsi, noise, degree, ari_db, nmi_db = evaluate(
                        labels, dataname, dataloader, ground_truth, ground_truth_db, data
                    )
                # save results
                results.append({
                    "Data": dataname, 
                    "Algorithm": algo, 
                    "N_cluster": degree, 
                    "DCSI": dcsi, 
                    "Balance": balance,
                    "ARI": ari,
                    "NMI": nmi,
                    "Noise": noise, 
                    "Categorical": "None", 
                    "ARI_DB": ari_db, 
                    "NMI_DB": nmi_db
                })
            if degree_ == degree_db:
                skip = True
        
        # save dataframe results to csv
        dataframe.to_csv('results/compas_experiment/{}.csv'.format(dataname))
        df = pd.DataFrame(results)
        df.to_csv('results/compas_experiment/{}_results.csv'.format(dataname))
    
    print("COMPAS experiments completed!")


def evaluate(labels, dataname, dataloader, ground_truth, ground_truth_db, data):
    """
    Evaluate given clustering.

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
    min_pts = dataloader.get_dcsi_min_pts()
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

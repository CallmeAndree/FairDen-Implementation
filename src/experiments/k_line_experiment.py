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
# File author: Lena Krieger

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


def k_line_multi():
    # experiment with multiple sensitive groups
    DATANAMES = ["adult2"]
    ALGORITHMS = ['FairDen', 'FairSC_normalized', 'FairSC']
    for dataname in tqdm(DATANAMES):
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
        min_pts = 2 * (data.shape[1] + len(dataloader.get_sens_attr())) - 1
        ground_truth = dataloader.get_target_columns()
        result_file = Path('results/k_line_experiment/{}.csv'.format(dataname))
        # if the file exists load it
        if result_file.is_file():
            dataframe = pd.read_csv(
                'results/k_line_experiment/{}.csv'.format(dataname))
        # otherwise create a dataframe and add data
        else:
            dataframe = dataloader.get_data_frame()
            dataframe['GroundTruth'] = dataloader.get_target_columns()
        labels = np.array(ground_truth)
        # evaluate ground truth clustering
        balance, ari, nmi, dcsi, noise, degree, ari_db, nmi_db = evaluate(labels, dataname, dataloader, ground_truth,
                                                                          ground_truth_db, data)

        deg = len(set(labels)) - (1 if -1 in labels else 0)
        # save ground truth results
        results.append(
            {"Data": dataname, "Algorithm": 'GroundTruth', 'N_cluster': deg, "min_pts": min_pts, "DCSI": dcsi,
             "Balance": balance, "ARI": ari,
             "NMI": nmi, "Noise": noise, "Categorical": "None", "ARI_DB": ari_db, "NMI_DB": nmi_db})
        dataframe['GroundTruth'] = labels
        # evaluate DBSCAN clustering
        balance, ari, nmi, dcsi, noise, degree, ari_db, nmi_db = evaluate(labels, dataname, dataloader, ground_truth,
                                                                          ground_truth_db, data)
        labels = ground_truth_db
        deg = len(set(labels)) - (1 if -1 in labels else 0)
        # save DBSCAN results
        results.append(
            {"Data": dataname, "Algorithm": 'GroundTruth_DB', 'N_cluster': deg, "min_pts": min_pts, "DCSI": dcsi,
             "Balance": balance, "ARI": ari,
             "NMI": nmi, "Noise": noise, "Categorical": "None", "ARI_DB": ari_db, "NMI_DB": nmi_db})
        dataframe['GroundTruth_DB'] = labels
        # for mulitple numbers of clusters
        for n_cluster in [2, 3, 4, 5, 6, 7, 8, 9, 10]:
            # for each algorithm
            for algo in tqdm(ALGORITHMS):
                # if Fairlet includes both versions
                if algo == 'Fairlet':
                    # create ClusteringAlgorithm object
                    algorithm = ClusteringAlgorithm(algo, dataloader, min_pts, dataname)
                    names, labelss = algorithm.run(n_cluster)
                    for name, labels in zip(names, labelss):
                        dataframe[name + str(n_cluster)] = labels
                        # evaluate clusterings
                        balance, ari, nmi, dcsi, noise, degree, ari_db, nmi_db = evaluate(labels, dataname, dataloader,
                                                                                          ground_truth, ground_truth_db,
                                                                                          data)
                        # save results
                        results.append(
                            {"Data": dataname, "Algorithm": name, "N_cluster": degree, "DCSI": dcsi, "Balance": balance,
                             "ARI": ari,
                             "NMI": nmi,
                             "Noise": noise, "Categorical": "None", "ARI_DB": ari_db, "NMI_DB": nmi_db})
                else:
                    # create ClusteringAlgorithm object
                    algorithm = ClusteringAlgorithm(algo, dataloader, min_pts, dataname)
                    labels = algorithm.run(n_cluster)
                    if labels is None:
                        balance, ari, nmi, dcsi, noise, degree, ari_db, nmi_db = -2, -2, -2, -2, -2, -2, -2, -2
                    else:
                        dataframe[algo + str(n_cluster)] = labels
                        balance, ari, nmi, dcsi, noise, degree, ari_db, nmi_db = evaluate(labels, dataname, dataloader,
                                                                                          ground_truth, ground_truth_db,
                                                                                          data)
                    # save results
                    results.append(
                        {"Data": dataname, "Algorithm": algo, "N_cluster": degree, "DCSI": dcsi, "Balance": balance,
                         "ARI": ari,
                         "NMI": nmi,
                         "Noise": noise, "Categorical": "None", "ARI_DB": ari_db, "NMI_DB": nmi_db})
        # save the results to csvs
        dataframe.to_csv('results/k_line_experiment/{}.csv'.format(dataname))
        df = pd.DataFrame(results)
        df.to_csv('results/k_line_experiment/{}_results.csv'.format(dataname))
    # call second experiment
    k_line_twosens()


def k_line_twosens():
    # experiment with two sensitive groups
    DATANAMES = ["adult5"]
    # list of algorithms
    ALGORITHMS = ['Scalable', 'FairDen', 'FairSC_normalized', 'FairSC', 'Fairlet']
    for dataname in tqdm(DATANAMES):
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
        min_pts = 2 * (data.shape[1] + len(dataloader.get_sens_attr())) - 1
        ground_truth = dataloader.get_target_columns()
        result_file = Path('results/k_line_experiment/{}.csv'.format(dataname))
        # if the file already exist load it
        if result_file.is_file():
            dataframe = pd.read_csv(
                'results/k_line_experiment/{}.csv'.format(dataname))
        else:
            # else create dataframe and add data to it
            dataframe = dataloader.get_data_frame()
            dataframe['GroundTruth'] = dataloader.get_target_columns()
        labels = np.array(ground_truth)
        # evaluate ground truth labels
        balance, ari, nmi, dcsi, noise, degree, ari_db, nmi_db = evaluate(labels, dataname, dataloader, ground_truth,
                                                                          ground_truth_db, data)

        deg = len(set(labels)) - (1 if -1 in labels else 0)
        # save evaluation results for ground truth labels
        results.append(
            {"Data": dataname, "Algorithm": 'GroundTruth', 'N_cluster': deg, "min_pts": min_pts, "DCSI": dcsi,
             "Balance": balance, "ARI": ari,
             "NMI": nmi, "Noise": noise, "Categorical": "None", "ARI_DB": ari_db, "NMI_DB": nmi_db})
        dataframe['GroundTruth'] = labels
        labels = ground_truth_db
        deg = len(set(labels)) - (1 if -1 in labels else 0)
        # evaluate ground truth labels
        balance, ari, nmi, dcsi, noise, degree, ari_db, nmi_db = evaluate(labels, dataname, dataloader, ground_truth,
                                                                          ground_truth_db, data)

        # save results for DBSCAN
        results.append(
            {"Data": dataname, "Algorithm": 'GroundTruth_DB', 'N_cluster': deg, "min_pts": min_pts, "DCSI": dcsi,
             "Balance": balance, "ARI": ari,
             "NMI": nmi, "Noise": noise, "Categorical": "None", "ARI_DB": ari_db, "NMI_DB": nmi_db})
        dataframe['GroundTruth_DB'] = labels
        # for each number of clusterings
        for n_cluster in [2, 3, 4, 5, 6, 7, 8, 9, 10]:
            # for each algorithm
            for algo in tqdm(ALGORITHMS):
                if algo == 'Fairlet':
                    # create object for Fairlets
                    algorithm = ClusteringAlgorithm(algo, dataloader, min_pts, dataname)
                    names, labelss = algorithm.run(n_cluster)
                    for name, labels in zip(names, labelss):
                        dataframe[name + str(n_cluster)] = labels
                        balance, ari, nmi, dcsi, noise, degree, ari_db, nmi_db = evaluate(labels, dataname, dataloader,
                                                                                          ground_truth, ground_truth_db,
                                                                                          data)
                        # save results for both Fairlet versions
                        results.append(
                            {"Data": dataname, "Algorithm": name, "N_cluster": degree, "DCSI": dcsi, "Balance": balance,
                             "ARI": ari,
                             "NMI": nmi,
                             "Noise": noise, "Categorical": "None", "ARI_DB": ari_db, "NMI_DB": nmi_db})
                else:
                    # create ClusteringAlgorithm object
                    algorithm = ClusteringAlgorithm(algo, dataloader, min_pts, dataname)
                    labels = algorithm.run(n_cluster)
                    if labels is None:
                        balance, ari, nmi, dcsi, noise, degree, ari_db, nmi_db = -2, -2, -2, -2, -2, -2, -2, -2
                    else:
                        dataframe[algo + str(n_cluster)] = labels
                        # evaluate clustering
                        balance, ari, nmi, dcsi, noise, degree, ari_db, nmi_db = evaluate(labels, dataname, dataloader,
                                                                                          ground_truth, ground_truth_db,
                                                                                          data)
                    # save results
                    results.append(
                        {"Data": dataname, "Algorithm": algo, "N_cluster": degree, "DCSI": dcsi, "Balance": balance,
                         "ARI": ari,
                         "NMI": nmi,
                         "Noise": noise, "Categorical": "None", "ARI_DB": ari_db, "NMI_DB": nmi_db})
        # save dataframe results to csv
        dataframe.to_csv('results/k_line_experiment/{}.csv'.format(dataname))
        df = pd.DataFrame(results)
        df.to_csv('results/k_line_experiment/{}_results.csv'.format(dataname))


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
        balance, b1, b2 = balance_score(dataname, dataloader.get_sens_attr(), np.array(labels),
                                        dataloader.get_sensitive_columns(), per_cluster=True)
    else:
        balance = 0
    ari = adjusted_rand_score(labels, ground_truth)
    nmi = normalized_mutual_info_score(labels, ground_truth)
    ari_db = adjusted_rand_score(labels, ground_truth_db)
    nmi_db = normalized_mutual_info_score(labels, ground_truth_db)

    dcsi = dcsiscore(data, labels, min_pts=min_pts)
    noise = noise_percent(labels)
    return balance, ari, nmi, dcsi, noise, degree, ari_db, nmi_db

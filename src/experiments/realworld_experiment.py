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


def realworld_experiment():
    # list of datasets
    DATANAMES = ["adult4", "diabetes", "communities"]
    # list of algorithms to include
    ALGORITHMS = ['FairDen', 'Scalable', 'FairSC_normalized', 'FairSC', 'Fairlet']
    # for each dataset
    for dataname in tqdm(DATANAMES):
        # list of results
        results = []
        # create a DataLoader object
        dataloader = DataLoader(dataname, categorical=False)
        dataloader.load()
        degree_ = dataloader.get_num_clusters()
        degree_db = dataloader.get_n_clusters_db()
        data = dataloader.get_data()
        min_pts, eps = dataloader.get_dbscan_config()
        # generate DBSCAN clustering
        dbscan = DBSCAN(eps=eps, min_samples=min_pts).fit(data)
        ground_truth_db = dbscan.labels_
        # define minpoints according to heuristic
        min_pts = 2 * (data.shape[1] + len(dataloader.get_sens_attr())) - 1
        ground_truth = dataloader.get_target_columns()

        result_file = Path('results/rw_experiment/{}.csv'.format(dataname))
        # if the file already exists load it to a DataFrame
        if result_file.is_file():
            dataframe = pd.read_csv(
                'results/rw_experiment/{}.csv'.format(dataname))
        # Else create a new Dataframe
        else:
            dataframe = dataloader.get_data_frame()
            dataframe['GroundTruth'] = dataloader.get_target_columns()

        labels = np.array(ground_truth)
        # evaluate ground truth clustering
        balance, ari, nmi, dcsi, noise, degree, ari_db, nmi_db = evaluate(labels, dataname, dataloader, ground_truth,
                                                                          ground_truth_db, data)
        attr1 = dataloader.get_sens_attr()
        # save results
        results.append(
            {"Data": dataname, "Algorithm": 'GroundTruth', "min_pts": min_pts, "DCSI": dcsi,
             "Balance": balance, "ARI": ari,
             "NMI": nmi, "Noise": noise, "Categorical": "None", "ARI_DB": ari_db, "NMI_DB": nmi_db,"N_cluster": degree})
        dataframe['GroundTruth'] = labels
        labels = ground_truth_db
        # evaluate DBSCAN clustering
        balance, ari, nmi, dcsi, noise, degree, ari_db, nmi_db = evaluate(labels, dataname, dataloader, ground_truth,
                                                                          ground_truth_db, data)
        attr1 = dataloader.get_sens_attr()
        # save results
        results.append(
            {"Data": dataname, "Algorithm": 'GroundTruth_DB', "min_pts": min_pts, "DCSI": dcsi,
             "Balance": balance, "ARI": ari,
             "NMI": nmi, "Noise": noise, "Categorical": "None", "ARI_DB": ari_db, "NMI_DB": nmi_db,"N_cluster": degree_db})
        dataframe['GroundTruth_DB'] = labels
        skip = False
        # for both numbers of clusters (ground truth and DBSCAN)
        for n_cluster in [degree_, degree_db]:
            # skip if both numbers of clusters are the same
            if skip:
                continue
            # for each algorithm
            for algo in tqdm(ALGORITHMS):
                if algo == 'Fairlet':
                    try:
                        algorithm = ClusteringAlgorithm(algo, dataloader, min_pts, dataname)
                        names, labelss = algorithm.run(n_cluster)
                        for name, labels in zip(names, labelss):
                            dataframe[name + str(n_cluster)] = labels
                            # evaluate Fairlet results for each version
                            balance, ari, nmi, dcsi, noise, degree, ari_db, nmi_db = evaluate(labels, dataname,
                                                                                              dataloader,
                                                                                              ground_truth,
                                                                                              ground_truth_db,
                                                                                              data)
                            # save results
                            results.append(
                                {"Data": dataname, "Algorithm": name, "N_cluster": degree, "DCSI": dcsi,
                                 "Balance": balance, "ARI": ari,
                                 "NMI": nmi,
                                 "Noise": noise, "Categorical": "None", "ARI_DB": ari_db, "NMI_DB": nmi_db})
                    except Exception:
                        # save results
                        results.append(
                            {"Data": dataname, "Algorithm": 'Fairlet', "N_cluster": '-', "DCSI": '-',
                             "Balance": '-', "ARI": '-',
                             "NMI": '-',
                             "Noise": '-', "Categorical": "None", "ARI_DB": '-', "NMI_DB": '-'})
                else:
                    # create ClusteringAlgorithm object
                    algorithm = ClusteringAlgorithm(algo, dataloader, min_pts, dataname)
                    labels = algorithm.run(n_cluster)
                    # if labels couldn't be generated add -2
                    if labels is None:
                        balance, ari, nmi, dcsi, noise, degree, ari_db, nmi_db = -2, -2, -2, -2, -2, -2, -2, -2

                    else:
                        dataframe[algo] = labels
                        # else evaluate as always
                        balance, ari, nmi, dcsi, noise, degree, ari_db, nmi_db = evaluate(labels, dataname, dataloader,
                                                                                          ground_truth, ground_truth_db,
                                                                                          data)
                    # save results
                    results.append(
                        {"Data": dataname, "Algorithm": algo, "N_cluster": degree, "DCSI": dcsi, "Balance": balance,
                         "ARI": ari,
                         "NMI": nmi,
                         "Noise": noise, "Categorical": "None", "ARI_DB": ari_db, "NMI_DB": nmi_db})
                # if both numbers of clusters are the same skip
                if degree_ == degree_db:
                    skip = True
        # save dataframe results to csv
        dataframe.to_csv('results/rw_experiment/{}.csv'.format(dataname))
        df = pd.DataFrame(results)
        df.to_csv('results/rw_experiment/{}_results.csv'.format(dataname))
    # call second part of the experiment
    realworld_experiment_multi()


def realworld_experiment_multi():
    # datasets with non-binary sensitive attributes
    DATANAMES = ["adult", "bank"]
    # algorithms that can handle non-binary
    ALGORITHMS = ['FairDen', 'FairSC_normalized', 'FairSC']
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
        # define min pts according to heuristic
        min_pts = 2 * (data.shape[1] + len(dataloader.get_sens_attr())) - 1
        ground_truth = dataloader.get_target_columns()
        result_file = Path('results/rw_experiment/{}.csv'.format(dataname))
        # if the result file already exists load content as dataframe
        if result_file.is_file():
            dataframe = pd.read_csv(
                'results/rw_experiment/{}.csv'.format(dataname))
        # create a new dataframe
        else:
            dataframe = dataloader.get_data_frame()
            dataframe['GroundTruth'] = dataloader.get_target_columns()
        labels = np.array(ground_truth)
        # evaluate ground truth
        balance, ari, nmi, dcsi, noise, degree, ari_db, nmi_db = evaluate(labels, dataname, dataloader, ground_truth,
                                                                          ground_truth_db, data)

        attr1 = dataloader.get_sens_attr()
        # save results
        results.append(
            {"Data": dataname, "Algorithm": 'GroundTruth', "min_pts": min_pts, "DCSI": dcsi,
             "Balance": balance, "ARI": ari,
             "NMI": nmi, "Noise": noise, "Categorical": "None", "ARI_DB": ari_db, "NMI_DB": nmi_db, "N_cluster": degree})
        dataframe['GroundTruth'] = labels
        # evaluate DBSCAN clustering
        labels = ground_truth_db
        balance, ari, nmi, dcsi, noise, degree, ari_db, nmi_db = evaluate(labels, dataname, dataloader, ground_truth,
                                                                          ground_truth_db, data)
        attr1 = dataloader.get_sens_attr()
        # save results
        results.append(
            {"Data": dataname, "Algorithm": 'GroundTruth_DB', "min_pts": min_pts, "DCSI": dcsi,
             "Balance": balance, "ARI": ari,
             "NMI": nmi, "Noise": noise, "Categorical": "None", "ARI_DB": ari_db, "NMI_DB": nmi_db, "N_cluster": degree_db})
        dataframe['GroundTruth_DB'] = labels
        skip = False
        # for ground truth numbers of clusters and DBSCAN number of clusters
        for n_cluster in [degree_, degree_db]:
            # if both are the same the second run will be skipped
            if skip:
                continue
            # for each algorithm
            for algo in tqdm(ALGORITHMS):
                if algo == 'Fairlet':
                    try:
                        algorithm = ClusteringAlgorithm(algo, dataloader, min_pts, dataname)
                        names, labelss = algorithm.run(n_cluster)
                        for name, labels in zip(names, labelss):
                            dataframe[name + str(n_cluster)] = labels
                            # evaluate clustering
                            balance, ari, nmi, dcsi, noise, degree, ari_db, nmi_db = evaluate(labels, dataname,
                                                                                              dataloader,
                                                                                              ground_truth,
                                                                                              ground_truth_db,
                                                                                              data)

                            # save results
                            results.append(
                                {"Data": dataname, "Algorithm": name, "N_cluster": degree, "DCSI": dcsi,
                                 "Balance": balance, "ARI": ari,
                                 "NMI": nmi,
                                 "Noise": noise, "Categorical": "None", "ARI_DB": ari_db, "NMI_DB": nmi_db})
                    except Exception:
                        # save results
                        results.append(
                            {"Data": dataname, "Algorithm": 'Fairlet', "N_cluster": '-', "DCSI": '-',
                             "Balance": '-', "ARI": '-',
                             "NMI": '-',
                             "Noise": '-', "Categorical": "None", "ARI_DB": '-', "NMI_DB": '-'})

                else:
                    algorithm = ClusteringAlgorithm(algo, dataloader, min_pts, dataname)
                    labels = algorithm.run(n_cluster)
                    # if labels couldn't be generated add -2
                    if labels is None:
                        balance, ari, nmi, dcsi, noise, degree, ari_db, nmi_db = -2, -2, -2, -2, -2, -2, -2, -2
                    else:
                        dataframe[algo] = labels
                        balance, ari, nmi, dcsi, noise, degree, ari_db, nmi_db = evaluate(labels, dataname, dataloader,
                                                                                          ground_truth, ground_truth_db,
                                                                                          data)
                    # save results
                    results.append(
                        {"Data": dataname, "Algorithm": algo, "N_cluster": degree, "DCSI": dcsi, "Balance": balance,
                         "ARI": ari,
                         "NMI": nmi,
                         "Noise": noise, "Categorical": "None", "ARI_DB": ari_db, "NMI_DB": nmi_db})
                if degree_ == degree_db:
                    skip = True
        # save dataframe results to csv
        dataframe.to_csv('results/rw_experiment/{}.csv'.format(dataname))
        df = pd.DataFrame(results)
        df.to_csv('results/rw_experiment/{}_results.csv'.format(dataname))


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

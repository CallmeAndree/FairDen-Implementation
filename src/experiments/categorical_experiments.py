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

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from tqdm import tqdm
from src.FairDen import FairDen

from src.evaluation.balance import balance_score
from src.evaluation.noise import noise_percent
from src.utils.DataLoader import DataLoader


def categorical_experiments():
    # list of datasets
    DATANAMES = ["bank", "adult2", "adult5", "student_address"]
    MIN_PTS = ['d']
    # name mapping
    NAME_MAP = {"bank": 'Bank (marital)', "adult2": 'Adult (race)', "adult5": 'Adult (gender)', "student_address": 'Student (address)'}
    # for each dataset
    for dataname in tqdm(DATANAMES):
        print('Run for {} started.'.format(dataname))
        name = NAME_MAP[dataname]
        # generate DataLoader
        dataloader = DataLoader(dataname, categorical=True)
        dataloader.load()
        ground_truth = dataloader.get_target_columns()
        degree_ = dataloader.get_num_clusters()
        degree_db = dataloader.get_n_clusters_db()
        data_wo_sensitive = dataloader.get_data()
        data_wo_sensitive = np.array(data_wo_sensitive)

        min_pts, eps = dataloader.get_dbscan_config()
        # generate DBSCAN clustering as 'density-based ground truth'
        dbscan = DBSCAN(eps=eps, min_samples=min_pts).fit(data_wo_sensitive)
        ground_truth_db = dbscan.labels_
        # if the file already exists load it
        result_file = Path('results/categorical_exp/combined.csv')
        if result_file.is_file():
            total = pd.read_csv('results/categorical_exp/combined.csv')
        # otherwise create a DataFrame
        else:
            total = pd.DataFrame([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]],
                                 columns=['Algorithm', 'min_pts', 'ARI', 'NMI', 'Noise', 'ARI_DB', 'NMI_DB', 'Degree',
                                          'Dataset', 'Balance', 'Stratified_Balance'])
        labels = np.array(ground_truth)
        # evaluate clustering ground truth clusters
        balance, ari, nmi, noise, ari_db, nmi_db = evaluate(labels, dataname, dataloader, ground_truth,
                                                            ground_truth_db)

        deg = len(set(labels)) - (1 if -1 in labels else 0)
        # save results to the DataFrame
        total.loc[len(total.index)] = {'Algorithm': 'Ground_Truth', 'min_pts': min_pts, 'ARI': ari, 'NMI': nmi,
                                       'Noise': noise, 'ARI_DB': ari_db, 'NMI_DB': nmi_db, 'Degree': deg,
                                       'Dataset': name, 'Balance': balance, 'Stratified_Balance': balance * (1 - noise)}
        # evaluate clustering DBSCAN clusters
        balance, ari, nmi, noise, ari_db, nmi_db = evaluate(ground_truth_db, dataname, dataloader, ground_truth,
                                                            ground_truth_db)
        labels = ground_truth_db

        deg = len(set(labels)) - (1 if -1 in labels else 0)
        # save results to the DataFrame
        total.loc[len(total.index)] = {'Algorithm': 'DBSCAN', 'min_pts': min_pts, 'ARI': ari, 'NMI': nmi,
                                       'Noise': noise, 'ARI_DB': ari_db, 'NMI_DB': nmi_db, 'Degree': deg,
                                       'Dataset': name, 'Balance': balance, 'Stratified_Balance': balance * (1 - noise)}
        skip = False
        for degree in [degree_, degree_db]:
            if skip:
                continue
            for min_pts in tqdm(MIN_PTS):
                print('Min PTS: {}'.format(min_pts))
                if min_pts == "d":
                    min_pts = 2 * (data_wo_sensitive.shape[1] + 1 + dataloader.get_num_categorical()) - 1
                print('Checking minpts {}.'.format(min_pts))
                # create FairDen object
                algorithm = FairDen(dataloader, min_pts=min_pts, alpha='avg')
                labels = algorithm.run(degree)
                if labels is None:
                    continue
                labels = np.array(labels)
                data_wo_sensitive = np.array(data_wo_sensitive)
                # evaluate clustering
                balance, ari, nmi, noise, ari_db, nmi_db = evaluate(labels, dataname, dataloader, ground_truth,
                                                                    ground_truth_db)
                deg = len(set(labels)) - (1 if -1 in labels else 0)
                # save results
                total.loc[len(total.index)] = {'Algorithm': 'FairDen', 'min_pts': min_pts, 'ARI': ari, 'NMI': nmi,
                                               'Noise': noise, 'ARI_DB': ari_db, 'NMI_DB': nmi_db, 'Degree': degree,
                                               'Dataset': name, 'Balance': balance,
                                               'Stratified_Balance': balance * (1 - noise)}
            if degree_ == degree_db:
                skip = True
        # save the DataFrame to csv
        total.to_csv('results/categorical_exp/combined.csv')
    not_categorical_experiments()


def not_categorical_experiments():
    # datasets
    DATANAMES = ["bank3", "adult", "adult4"]
    MIN_PTS = ['d']
    # mapping for DataFrame
    NAME_MAP = {"bank3": 'Bank (marital)', "adult": 'Adult (race)', "adult4": 'Adult (gender)'}
    for dataname in tqdm(DATANAMES):
        name = NAME_MAP[dataname]
        print('Run for {} started.'.format(dataname))
        # create DataLoader object
        dataloader = DataLoader(dataname, categorical=False)
        dataloader.load()
        ground_truth = dataloader.get_target_columns()
        degree_ = dataloader.get_num_clusters()
        degree_db = dataloader.get_n_clusters_db()
        data_wo_sensitive = dataloader.get_data()
        data_wo_sensitive = np.array(data_wo_sensitive)

        min_pts, eps = dataloader.get_dbscan_config()
        # create DBSCAN clusterings
        dbscan = DBSCAN(eps=eps, min_samples=min_pts).fit(data_wo_sensitive)
        ground_truth_db = dbscan.labels_

        result_file = Path('results/categorical_exp/combined.csv')
        # if the file already exist load it
        if result_file.is_file():
            total = pd.read_csv('results/categorical_exp/combined.csv')
        skip = False
        # for numbers of clusters as in ground truth and in DBSCAN clusterings
        for degree in [degree_, degree_db]:
            if skip:
                continue
            for min_pts in tqdm(MIN_PTS):
                print('Min PTS: {}'.format(min_pts))
                if min_pts == "d":
                    min_pts = 2 * (data_wo_sensitive.shape[1] + 1) - 1
                print('Checking minpts {}.'.format(min_pts))
                # create FairDen object
                algorithm = FairDen(dataloader, min_pts=min_pts, alpha='0')
                labels = algorithm.run(degree)
                if labels is None:
                    continue
                labels = np.array(labels)
                data_wo_sensitive = np.array(data_wo_sensitive)
                # evaluate the calculated clustering
                balance, ari, nmi, noise, ari_db, nmi_db = evaluate(labels, dataname, dataloader, ground_truth,
                                                                    ground_truth_db)
                deg = len(set(labels)) - (1 if -1 in labels else 0)
                # save results to the DataFrame
                total.loc[len(total.index)] = {'Algorithm': 'FairDen-', 'min_pts': min_pts, 'ARI': ari, 'NMI': nmi,
                                               'Noise': noise, 'ARI_DB': ari_db, 'NMI_DB': nmi_db, 'Degree': degree,
                                               'Dataset': name, 'Balance': balance,
                                               'Stratified_Balance': balance * (1 - noise)}

            if degree_ == degree_db:
                skip = True
        for name in total.columns.values:
            if 'Unnamed' in name:
                total = total.drop(name, axis=1)
        # save the DataFrame to a csv
        total.to_csv('results/categorical_exp/combined.csv')


def evaluate(labels, dataname, dataloader, ground_truth, ground_truth_db):
    """
        Evaluate given clusterings.

        Parameters
        ----------
        labels: clustering labels.
        dataname: dataset name.
        dataloader: DataLoader object.
        ground_truth: Ground truth clustering labels.
        ground_truth_db: DBSCAN clustering labels.

        Returns
        -------
            evaluation metrics for a given clustering regarding balance and external clustering validation.
    """
    balance, b1, b2 = balance_score(dataname, dataloader.get_sens_attr(), labels, dataloader.get_sensitive_columns(),
                                    per_cluster=True)
    ari = adjusted_rand_score(labels, ground_truth)
    nmi = normalized_mutual_info_score(labels, ground_truth)
    indices = np.asarray(ground_truth_db == -1).nonzero()[0]
    labels_wo_noise = np.delete(labels, indices)
    gt_wo_noise = np.delete(ground_truth_db, indices)
    ari_db = adjusted_rand_score(labels_wo_noise, gt_wo_noise)
    nmi_db = normalized_mutual_info_score(labels_wo_noise, gt_wo_noise)
    noise = noise_percent(labels)
    return balance, ari, nmi, noise, ari_db, nmi_db

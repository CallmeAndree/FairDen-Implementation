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

from sklearn.cluster import DBSCAN
from sklearn.metrics.cluster import normalized_mutual_info_score
from sklearn.metrics import adjusted_rand_score
from tqdm import tqdm

from src.utils.DataLoader import DataLoader
from src.evaluation.dcsi import dcsiscore
from src.evaluation.balance import balance_score, balance_mixed
from src.evaluation.noise import noise_percent
from src.FairDen import FairDen


def adult_experiment():
    # perform experiment with different combinations of sensitive attributes
    #DATANAMES = ['adult_g', "adult_m", "adult_r", "adult_gm", "adult_gr", "adult_mr","adult_gmr"]
    DATANAMES = ["adult_gmr"]
    MIN_PTS = [15]
    # mapping
    settings = {'g': 'G', 'm': 'M', 'r': 'R', 'gm': 'G&M', 'mr': 'M&R', 'gr': 'G&R', 'gmr': 'G&M&R'}
    # for each configuration
    for dataname in tqdm(DATANAMES):
        print('Optimization run for {} started.'.format(dataname))
        setting = settings[dataname.split('_')[-1]]
        # list to save results
        experimental = []
        # construct a DataLoader
        dataloader = DataLoader(dataname, categorical=False)
        dataloader.load()
        ground_truth = dataloader.get_target_columns()
        degree = dataloader.get_num_clusters()
        data_sensitive = dataloader.get_data()
        data_sensitive = np.array(data_sensitive)
        min_pts, eps = dataloader.get_dbscan_config()
        # calculate DBSCAN clustering for density-based ground-truth
        dbscan = DBSCAN(eps=eps, min_samples=min_pts).fit(data_sensitive)
        ground_truth_db = dbscan.labels_
        # transform to numpy array
        labels = np.array(ground_truth)
        ground_truth = np.array(ground_truth)
        # list of sensitive attributes
        attr1, attr2, attr3 = ['gender', 'race', 'marital_status']
        lab, count = np.unique(labels, return_counts=True)
        # evaluate clustering
        balance, ari, nmi, dcsi, noise, ari_db, nmi_db, balance_per_cluster, balance_per_group_per_cluster = evaluate(
            ground_truth_db, dataname, dataloader, ground_truth,
            ground_truth_db, data_sensitive)
        # evaluate mixed balances
        balance_mix, balance_per_cluster_mix, balance_per_group_per_cluster_mix = evaluate_balance_mixed(
            labels, dataname, dataloader)
        # save results for ground truth clustering
        experimental.append(
            {"Setting": setting, "Degree": degree, "Algorithm": 'GroundTruth', "min_pts": min_pts,
             "Balance_{}".format(attr1): balance[attr1], "Balance_{}".format(attr2): balance[attr2],
             "Balance_{}".format(attr3): balance[attr3], "Balance_Mixed": balance_mix,
             'Cluster_{}'.format(lab[0]): count[0],
             "Cluster_{}".format(lab[1]): count[1],
             'Balances_per_Cluster_{}'.format(attr1): balance_per_cluster[attr1],
             'Balances_per_Cluster_{}'.format(attr2): balance_per_cluster[attr2],
             'Balances_per_Cluster_{}'.format(attr3): balance_per_cluster[attr3],
             'Balances_per_Cluster_Mixed': balance_per_cluster_mix,
             'Balances_per_Group_per_Cluster_{}'.format(attr1): balance_per_group_per_cluster[attr1],
             'Balances_per_Group_per_Cluster_{}'.format(attr2): balance_per_group_per_cluster[attr2],
             'Balances_per_Group_per_Cluster_{}'.format(attr3): balance_per_group_per_cluster[attr3],
             'Balances_per_Group_per_Cluster_Mixed': balance_per_group_per_cluster_mix})

        labels = ground_truth_db
        lab, count = np.unique(labels, return_counts=True)
        balance, ari, nmi, dcsi, noise, ari_db, nmi_db, balance_per_cluster, balance_per_group_per_cluster = evaluate(
            ground_truth_db, dataname, dataloader, ground_truth,
            ground_truth_db, data_sensitive)

        balance_mix, balance_per_cluster_mix, balance_per_group_per_cluster_mix = evaluate_balance_mixed(
            labels, dataname, dataloader)
        # save results for DBSCAN clustering
        experimental.append(
            {"Setting": setting, "Degree": degree, "Algorithm": 'DBSCAN', "min_pts": min_pts,
             "Balance_{}".format(attr1): balance[attr1], "Balance_{}".format(attr2): balance[attr2],
             "Balance_{}".format(attr3): balance[attr3], "Balance_Mixed": balance_mix,
             'Cluster_{}'.format(lab[0]): count[0],
             "Cluster_{}".format(lab[1]): count[1],
             'Balances_per_Cluster_{}'.format(attr1): balance_per_cluster[attr1],
             'Balances_per_Cluster_{}'.format(attr2): balance_per_cluster[attr2],
             'Balances_per_Cluster_{}'.format(attr3): balance_per_cluster[attr3],
             'Balances_per_Cluster_Mixed': balance_per_cluster_mix,
             'Balances_per_Group_per_Cluster_{}'.format(attr1): balance_per_group_per_cluster[attr1],
             'Balances_per_Group_per_Cluster_{}'.format(attr2): balance_per_group_per_cluster[attr2],
             'Balances_per_Group_per_Cluster_{}'.format(attr3): balance_per_group_per_cluster[attr3],
             'Balances_per_Group_per_Cluster_Mixed': balance_per_group_per_cluster_mix})

        for degree in [degree]:
            for min_pts in tqdm(MIN_PTS):
                print('Min PTS: {}'.format(min_pts))
                if min_pts == "d1":
                    min_pts = 2 * (data_sensitive.shape[1] + len(dataloader.get_sens_attr())) - 1
                print('Checking minpts {}.'.format(min_pts))
                # create FairDen object
                algorithm = FairDen(dataloader, min_pts=min_pts, alpha='0')
                labels = algorithm.run(degree)
                if labels is None:
                    continue
                labels = np.array(labels)
                data_sensitive = np.array(data_sensitive)
                # evaluate clusterings
                balance, ari, nmi, dcsi, noise, ari_db, nmi_db, balance_per_cluster, balance_per_group_per_cluster = evaluate(
                    labels, dataname, dataloader, ground_truth,
                    ground_truth_db, data_sensitive)
                # evaluate clusterings regarding mixed sensitive attribute
                balance_mix, balance_per_cluster_mix, balance_per_group_per_cluster_mix = evaluate_balance_mixed(
                    labels, dataname, dataloader)
                lab, count = np.unique(labels, return_counts=True)
                if -1 in labels:
                    # save results for FairDen clustering including noise
                    experimental.append(
                        {"Setting": setting, "Degree": degree, "Algorithm": 'FairDen_v1', "min_pts": min_pts,
                         "Balance_{}".format(attr1): balance[attr1], "Balance_{}".format(attr2): balance[attr2],
                         "Balance_{}".format(attr3): balance[attr3], "Balance_Mixed": balance_mix,
                         'Cluster_{}'.format(lab[0]): count[0],
                         "Cluster_{}".format(lab[1]): count[1], "Cluster_{}".format(lab[2]): count[2],
                         'Balances_per_Cluster_{}'.format(attr1): balance_per_cluster[attr1],
                         'Balances_per_Cluster_{}'.format(attr2): balance_per_cluster[attr2],
                         'Balances_per_Cluster_{}'.format(attr3): balance_per_cluster[attr3],
                         'Balances_per_Cluster_Mixed': balance_per_cluster_mix,
                         'Balances_per_Group_per_Cluster_{}'.format(attr1): balance_per_group_per_cluster[attr1],
                         'Balances_per_Group_per_Cluster_{}'.format(attr2): balance_per_group_per_cluster[attr2],
                         'Balances_per_Group_per_Cluster_{}'.format(attr3): balance_per_group_per_cluster[attr3],
                         'Balances_per_Group_per_Cluster_Mixed': balance_per_group_per_cluster_mix})
                else:
                    # save results for FairDen clustering including noise
                    experimental.append(
                        {"Setting": setting, "Degree": degree, "Algorithm": 'FairDen', "min_pts": min_pts,
                         "Balance_{}".format(attr1): balance[attr1], "Balance_{}".format(attr2): balance[attr2],
                         "Balance_{}".format(attr3): balance[attr3], "Balance_Mixed": balance_mix,
                         'Cluster_{}'.format(lab[0]): count[0],
                         "Cluster_{}".format(lab[1]): count[1],
                         'Balances_per_Cluster_{}'.format(attr1): balance_per_cluster[attr1],
                         'Balances_per_Cluster_{}'.format(attr2): balance_per_cluster[attr2],
                         'Balances_per_Cluster_{}'.format(attr3): balance_per_cluster[attr3],
                         'Balances_per_Cluster_Mixed': balance_per_cluster_mix,
                         'Balances_per_Group_per_Cluster_{}'.format(attr1): balance_per_group_per_cluster[attr1],
                         'Balances_per_Group_per_Cluster_{}'.format(attr2): balance_per_group_per_cluster[attr2],
                         'Balances_per_Group_per_Cluster_{}'.format(attr3): balance_per_group_per_cluster[attr3],
                         'Balances_per_Group_per_Cluster_Mixed': balance_per_group_per_cluster_mix})
        df_2 = pd.DataFrame(experimental)
        df_2.to_csv("results/adult_multi_exp/experimental_{}.csv".format(dataname))
    adult_experiment_cat()


def adult_experiment_cat():
    # perform experiment with different combinations of sensitive attributes
    DATANAMES = ["adult_gmr"]
    DATANAMES = ['adult_g', "adult_m", "adult_r", "adult_gm", "adult_gr", "adult_mr"]
    MIN_PTS = [15]
    settings = {'g': 'G', 'm': 'M', 'r': 'R', 'gm': 'G&M', 'mr': 'M&R', 'gr': 'G&R', 'gmr': 'G&M&R'}
    # for each setting
    for dataname in tqdm(DATANAMES):
        setting = settings[dataname.split('_')[-1]]
        print('Optimization run for {} started.'.format(dataname))
        # result list
        experimental = []
        # DataLoader object
        dataloader = DataLoader(dataname, categorical=True)
        dataloader.load()
        ground_truth = dataloader.get_target_columns()
        degree = dataloader.get_num_clusters()
        data_sensitive = dataloader.get_data()
        data_sensitive = np.array(data_sensitive)
        min_pts, eps = dataloader.get_dbscan_config()
        # generate DBSCAN clustering
        dbscan = DBSCAN(eps=eps, min_samples=min_pts).fit(data_sensitive)
        ground_truth_db = dbscan.labels_
        labels = np.array(ground_truth)
        ground_truth = np.array(ground_truth)
        # evaluate clustering
        balance, ari, nmi, dcsi, noise, ari_db, nmi_db, balance_per_cluster, balance_per_group_per_cluster = evaluate(
            labels, dataname, dataloader, ground_truth,
            ground_truth_db, data_sensitive)
        # list of sensitive attributes
        attr1, attr2, attr3 = ['gender', 'race', 'marital_status']
        # evaluate clustering regarding mixed sensitive attribute
        balance_mix, balance_per_cluster_mix, balance_per_group_per_cluster_mix = evaluate_balance_mixed(
            labels, dataname, dataloader)

        lab, count = np.unique(labels, return_counts=True)
        # save results for ground truth clustering
        experimental.append(
            {"Setting": setting, "Degree": degree, "Algorithm": 'GroundTruth', "min_pts": min_pts,
             "Balance_{}".format(attr1): balance[attr1], "Balance_{}".format(attr2): balance[attr2],
             "Balance_{}".format(attr3): balance[attr3], "Balance_Mixed": balance_mix,
             'Cluster_{}'.format(lab[0]): count[0],
             "Cluster_{}".format(lab[1]): count[1],
             'Balances_per_Cluster_{}'.format(attr1): balance_per_cluster[attr1],
             'Balances_per_Cluster_{}'.format(attr2): balance_per_cluster[attr2],
             'Balances_per_Cluster_{}'.format(attr3): balance_per_cluster[attr3],
             'Balances_per_Cluster_Mixed': balance_per_cluster_mix,
             'Balances_per_Group_per_Cluster_{}'.format(attr1): balance_per_group_per_cluster[attr1],
             'Balances_per_Group_per_Cluster_{}'.format(attr2): balance_per_group_per_cluster[attr2],
             'Balances_per_Group_per_Cluster_{}'.format(attr3): balance_per_group_per_cluster[attr3],
             'Balances_per_Group_per_Cluster_Mixed': balance_per_group_per_cluster_mix})
        labels = ground_truth_db
        # save results for ground truth clustering
        balance, ari, nmi, dcsi, noise, ari_db, nmi_db, balance_per_cluster, balance_per_group_per_cluster = evaluate(
            ground_truth_db, dataname, dataloader, ground_truth,
            ground_truth_db, data_sensitive)
        # evaluate clustering regarding mixed sensitive attribute
        balance_mix, balance_per_cluster_mix, balance_per_group_per_cluster_mix = evaluate_balance_mixed(
            labels, dataname, dataloader)

        lab, count = np.unique(labels, return_counts=True)
        # save results for DBSCAN clustering
        experimental.append(
            {"Setting": setting, "Degree": degree, "Algorithm": 'DBSCAN', "min_pts": min_pts,
             "Balance_{}".format(attr1): balance[attr1], "Balance_{}".format(attr2): balance[attr2],
             "Balance_{}".format(attr3): balance[attr3], "Balance_Mixed": balance_mix,
             'Cluster_{}'.format(lab[0]): count[0],
             "Cluster_{}".format(lab[1]): count[1],
             'Balances_per_Cluster_{}'.format(attr1): balance_per_cluster[attr1],
             'Balances_per_Cluster_{}'.format(attr2): balance_per_cluster[attr2],
             'Balances_per_Cluster_{}'.format(attr3): balance_per_cluster[attr3],
             'Balances_per_Cluster_Mixed': balance_per_cluster_mix,
             'Balances_per_Group_per_Cluster_{}'.format(attr1): balance_per_group_per_cluster[attr1],
             'Balances_per_Group_per_Cluster_{}'.format(attr2): balance_per_group_per_cluster[attr2],
             'Balances_per_Group_per_Cluster_{}'.format(attr3): balance_per_group_per_cluster[attr3],
             'Balances_per_Group_per_Cluster_Mixed': balance_per_group_per_cluster_mix})

        for degree in [degree]:
            for min_pts in tqdm(MIN_PTS):
                print('Min PTS: {}'.format(min_pts))
                if min_pts == "d":
                    min_pts = 2 * (data_sensitive.shape[1] + len(dataloader.get_sens_attr())) - 1
                print('Checking minpts {}.'.format(min_pts))
                # create FairDen object
                algorithm = FairDen(dataloader, min_pts=min_pts, alpha='avg')
                labels = algorithm.run(degree)
                if labels is None:
                    continue
                labels = np.array(labels)
                data_sensitive = np.array(data_sensitive)
                # evaluate clustering
                balance, ari, nmi, dcsi, noise, ari_db, nmi_db, balance_per_cluster, balance_per_group_per_cluster = evaluate(
                    labels, dataname, dataloader, ground_truth,
                    ground_truth_db, data_sensitive)
                # evaluate clustering regarding mixed sensitive attributes
                balance_mix, balance_per_cluster_mix, balance_per_group_per_cluster_mix = evaluate_balance_mixed(
                    labels, dataname, dataloader)
                lab, count = np.unique(labels, return_counts=True)
                if -1 in labels:
                    # save results for FairDen clustering including noise
                    experimental.append(
                        {"Setting": setting, "Degree": degree, "Algorithm": 'FairDen', "min_pts": min_pts,
                         "Balance_{}".format(attr1): balance[attr1], "Balance_{}".format(attr2): balance[attr2],
                         "Balance_{}".format(attr3): balance[attr3], "Balance_Mixed": balance_mix,
                         'Cluster_{}'.format(lab[0]): count[0],
                         "Cluster_{}".format(lab[1]): count[1], "Cluster_{}".format(lab[2]): count[2],
                         'Balances_per_Cluster_{}'.format(attr1): balance_per_cluster[attr1],
                         'Balances_per_Cluster_{}'.format(attr2): balance_per_cluster[attr2],
                         'Balances_per_Cluster_{}'.format(attr3): balance_per_cluster[attr3],
                         'Balances_per_Cluster_Mixed': balance_per_cluster_mix,
                         'Balances_per_Group_per_Cluster_{}'.format(attr1): balance_per_group_per_cluster[attr1],
                         'Balances_per_Group_per_Cluster_{}'.format(attr2): balance_per_group_per_cluster[attr2],
                         'Balances_per_Group_per_Cluster_{}'.format(attr3): balance_per_group_per_cluster[attr3],
                         'Balances_per_Group_per_Cluster_Mixed': balance_per_group_per_cluster_mix})
                else:
                    # save results for FairDen clustering without noise
                    experimental.append(
                        {"Setting": setting, "Degree": degree, "Algorithm": 'FairDen_v1', "min_pts": min_pts,
                         "Balance_{}".format(attr1): balance[attr1], "Balance_{}".format(attr2): balance[attr2],
                         "Balance_{}".format(attr3): balance[attr3], "Balance_Mixed": balance_mix,
                         'Cluster_{}'.format(lab[0]): count[0],
                         "Cluster_{}".format(lab[1]): count[1],
                         'Balances_per_Cluster_{}'.format(attr1): balance_per_cluster[attr1],
                         'Balances_per_Cluster_{}'.format(attr2): balance_per_cluster[attr2],
                         'Balances_per_Cluster_{}'.format(attr3): balance_per_cluster[attr3],
                         'Balances_per_Cluster_Mixed': balance_per_cluster_mix,
                         'Balances_per_Group_per_Cluster_{}'.format(attr1): balance_per_group_per_cluster[attr1],
                         'Balances_per_Group_per_Cluster_{}'.format(attr2): balance_per_group_per_cluster[attr2],
                         'Balances_per_Group_per_Cluster_{}'.format(attr3): balance_per_group_per_cluster[attr3],
                         'Balances_per_Group_per_Cluster_Mixed': balance_per_group_per_cluster_mix})
        df_2 = pd.DataFrame(experimental)
        df_2.to_csv("results/adult_multi_exp/experimental_{}.csv".format(dataname))




def evaluate(labels, dataname, dataloader, ground_truth, ground_truth_db, data):
    """
        Evaluate given clusterings.

        Parameters
        ----------
        labels: clustering labels.
        dataname: name of dataset.
        dataloader: dataloader object.
        ground_truth: ground truth clustering labels.
        ground_truth_db: DBSCAN clustering labels.
        data: dataset.

        Returns
        -------
            evaluation metrics regarding balance and external clustering evaluation.
    """
    min_pts = 5
    sensitive = dataloader.get_all_sensitive()
    balance, balance_per_cluster, balance_per_group_per_cluster = balance_score(dataname, list(sensitive.columns),
                                                                                labels, sensitive, per_cluster=True)
    ari = adjusted_rand_score(labels, ground_truth)
    nmi = normalized_mutual_info_score(labels, ground_truth)
    ari_db = adjusted_rand_score(labels, ground_truth_db)
    nmi_db = normalized_mutual_info_score(labels, ground_truth_db)
    dcsi = dcsiscore(data, labels, min_pts=min_pts)
    noise = noise_percent(labels)
    return balance, ari, nmi, dcsi, noise, ari_db, nmi_db, balance_per_cluster, balance_per_group_per_cluster


def evaluate_balance_mixed(labels, dataname, dataloader):
    """
            Evaluate given clusterings regarding combined sensitive attributes.

            Parameters
            ----------
            labels: clustering labels.
            dataname: name of dataset.
            dataloader: dataloader object.

            Returns
            -------
                evaluation metrics regarding balance and external clustering evaluation.
        """
    sensitive = dataloader.get_sens_combi_mixed()
    balance_mix, balance_per_cluster, balance_per_group_per_cluster = balance_mixed(dataname, ['combi'],
                                                                                      labels, sensitive,
                                                                                      per_cluster=True)
    return balance_mix, balance_per_cluster, balance_per_group_per_cluster

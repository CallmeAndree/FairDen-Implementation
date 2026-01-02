import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN

from auxiliary.AuxExperiments.ClusteringAlgorithm import ClusteringAlgorithm
from src.evaluation.balance import balance_score
from src.evaluation.dcsi import dcsiscore
from src.evaluation.noise import noise_percent
from src.utils.DataLoader import DataLoader

def three_moons():
    results = []
    label_dict = {}
    dataset = "three_moons_mixed"
    dataloader = DataLoader(dataset, categorical=False)
    dataloader.load()
    gt = dataloader.get_target_columns()
    X = dataloader.get_data()
    label_dict['0'] = X[:,0]
    label_dict['1'] = X[:,1]
    label_dict['sensitive_attribute'] = np.array(dataloader.get_sensitive_columns()['sensitive_attribute'])
    label_dict['GroundTruth'] = gt
    min_pts = 5
    eps = 0.2
    dbscan = DBSCAN(eps=eps, min_samples=min_pts).fit(X)
    labels = dbscan.labels_
    labels = np.array(labels)
    balance, dcsi, noise = evaluate(labels, dataset, dataloader, X)
    row = {"Algorithm": 'DBSCAN', 'Noise': noise, 'DCSI': dcsi, 'Balance': balance}
    label_dict['DBSCAN'] = labels
    results.append(row)
    ALGORITHMS = ['FairDen', 'Scalable',  'FairSC','Fairlet']
    minpts=4
    for algo in ALGORITHMS:
        algorithm = ClusteringAlgorithm(algo, dataloader, minpts, 'runtime')
        labels = algorithm.run(2)
        if algo == 'Fairlet':
            names, labelss = algorithm.run(2)
            print(labelss)
            labels = labelss[0]
        label_dict[algo] = labels
        labels = np.array(labels)
        balance, dcsi, noise = evaluate(labels, dataset, dataloader, X)
        row = {"Algorithm": algo, 'Noise': noise, 'DCSI': dcsi, 'Balance': balance}
        results.append(row)
    label_df = pd.DataFrame(label_dict)
    label_df.to_csv('results/three_moons_labels.csv', index=False)
    result_df = pd.DataFrame(results)
    print(result_df)
    result_df.to_csv('results/three_moons_results.csv', index=False)


def append_row(df, row):
    return pd.concat([
                df,
                pd.DataFrame([row])], ignore_index=True
           ).reset_index(drop=True)


def evaluate(labels, dataname, dataloader, data):
    min_pts = 5
    balance = balance_score(dataname, dataloader.get_sens_attr(), labels, dataloader.get_sensitive_columns())

    dcsi = dcsiscore(data, labels, min_pts=min_pts)
    noise = noise_percent(labels)
    return balance, dcsi, noise


if __name__ == '__main__':
    three_moons()
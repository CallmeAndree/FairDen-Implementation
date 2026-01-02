
from auxiliary.AuxExperiments.ClusteringAlgorithm import ClusteringAlgorithm
from auxiliary.AuxExperiments.DataLoader import DataLoader
import pandas as pd
import numpy as np
import time
from tqdm import tqdm
from interruptingcow import timeout

# runtime experiments regarding different dimensions in the dataset for FairDen
# min points differs depending on the dimensions of the dataset
def dim_experiment():
    # to collect results
    results = []
    # different dimensions
    for dim in tqdm([5, 10, 50, 100]):
        # input synthetic dataset
        df = pd.read_csv('auxiliary/AuxExperiments/Data/dim/dataset_dim_{}.csv'.format(dim))
        # construct dataloader for the dataset
        dataloader = DataLoader(df)
        dataloader.load()
        data_wo_sensitive = dataloader.get_data()
        data = np.array(data_wo_sensitive)
        # min points as defined (varies depending on dataset dimensions)
        minpts = 2 * (data.shape[1] + 1) - 1
        # 5 runs for each configuration
        for i in range(5):
            try:
                # set a timeout
                with timeout(7200, exception=RuntimeError):
                    # starting time
                    st = time.time()
                    # construct FairDen object
                    algorithm = ClusteringAlgorithm('FairDen', dataloader, minpts, 'runtime')
                    labels = algorithm.run(2)
                    # end time
                    end = time.time()
                    # save results
                    results.append({'dim': dim, 'Time': end - st, 'Algorithm': 'FairDen'})
            except RuntimeError:
                print('RuntimeError')
                break
            except Exception:
                print('Exception')
    # save the results in dataframe and then as csv
    res_df = pd.DataFrame(results)
    res_df.to_csv('auxiliary/AuxExperiments/results/dim_experiment.csv')

# runtime experiments regarding different dimensions in the dataset for FairDen
# minpts stays 5 for every dataset
def dim_experiment2():
    # to collect results
    results = []
    # 5 runs for each configuration
    for i in range(5):
        # different dimensions
        for dim in tqdm([5, 10, 50, 100, 1000]):
            # read synthetic dataset
            df = pd.read_csv('auxiliary/AuxExperiments/Data/dim/dataset_dim_{}_{}.csv'.format(dim, i))
            # construct dataloader for the dataset
            dataloader = DataLoader(df)
            dataloader.load()
            data_wo_sensitive = dataloader.get_data()
            # minpts does not change in this experiment
            minpts = 5
            for j in range(1):
                try:
                    # set a timeout
                    with timeout(7200, exception=RuntimeError):
                        # starting time
                        st = time.time()
                        # algorithm object
                        algorithm = ClusteringAlgorithm('FairDen', dataloader, minpts, 'runtime')
                        labels = algorithm.run(2)
                        # end time
                        end = time.time()
                        # save results
                        results.append({'Dim': dim, 'Time': end - st, 'algorithm': 'FairDen', 'Dataset': i})
                except RuntimeError:
                    print('RuntimeError')
                    break
                except Exception:
                    print('Exception')
    # save results as dataframe and the also csv
    res_df = pd.DataFrame(results)
    res_df.to_csv('auxiliary/AuxExperiments/results/dim_experiment2.csv')

# runtime experiments regarding different numbers of ground truth clusters (k)
def k_experiment():
    # save results
    results = []
    # 5 runs forch each configuration
    for i in range(5):
        # different numbers of ground truth clusters
        for k in tqdm([2, 3, 4, 5, 6, 7, 8, 9]):
            # import dataset
            df = pd.read_csv('auxiliary/AuxExperiments/Data/k/dataset_k_{}_{}.csv'.format(k,i))
            # create corresponding dataloader object
            dataloader = DataLoader(df)
            dataloader.load()
            data_wo_sensitive = dataloader.get_data()
            data = np.array(data_wo_sensitive)
            # define varying minpts
            minpts = 2 * (data.shape[1] + 1) - 1
            for j in range(1):
                try:
                    # set runtime limit
                    with timeout(7200, exception=RuntimeError):
                        # starting time
                        st = time.time()
                        # FairDen object
                        algorithm = ClusteringAlgorithm('FairDen', dataloader, minpts, 'runtime')
                        labels = algorithm.run(k)
                        # ending time
                        end = time.time()
                        # save result
                        results.append({'k': k, 'Time': end - st, 'Algorithm': 'FairDen', 'Dataset': i})
                except RuntimeError:
                    print('exceeded time limit')
                    break
                except Exception:
                    print('Exception')
    # save results to df and csv
    res_df = pd.DataFrame(results)
    res_df.to_csv('auxiliary/AuxExperiments/results/k_experiment.csv')

# runtime experiments regarding different numbers of data points (n)
def n_experiment():
    # list to save results
    results = []
    # different dataset sizes
    for n in tqdm([100, 200, 500, 1000, 2000, 5000, 10000, 20000]):
        # import dataset
        df = pd.read_csv('auxiliary/AuxExperiments/Data/n/dataset_n_{}.csv'.format(n))
        # construct corresponding dataframe
        dataloader = DataLoader(df)
        dataloader.load()
        data_wo_sensitive = dataloader.get_data()
        data = np.array(data_wo_sensitive)
        # define minpts
        minpts = 2 * (data.shape[1] + 1) - 1
        # five runs
        for i in range(5):
            try:
                # set runtime limit
                with timeout(7200, exception=RuntimeError):
                    # starting time
                    st = time.time()
                    # construct clustering algorithm object
                    algorithm = ClusteringAlgorithm('FairDen', dataloader, minpts, 'runtime')
                    labels = algorithm.run(2)
                    # end time
                    end = time.time()
                    results.append({'n': n, 'Time': end - st, 'Algorithm': 'FairDen'})
            except RuntimeError:
                print('exceeded time limit')
                break
            except Exception:
                print('Exception')
    # save results to df and csv
    res_df = pd.DataFrame(results)
    res_df.to_csv('auxiliary/AuxExperiments/results/n_experiment.csv')

# perform the dimension runtime experiment for the other algorithms
def dim_experiment_other():
    # result list
    results = []
    # different dimensions
    for dim in tqdm([5, 10, 50, 100, 1000]):
        # read synthetic dataset
        df = pd.read_csv('auxiliary/AuxExperiments/Data/dim/dataset_dim_{}.csv'.format(dim))
        # create Dataloader object
        dataloader = DataLoader(df)
        dataloader.load()
        data_wo_sensitive = dataloader.get_data()
        data = np.array(data_wo_sensitive)
        # define minpts
        minpts = 2 * (data.shape[1] + 1) - 1
        # algorithms
        ALGORITHMS = ['Scalable', 'FairSC_normalized', 'FairSC']
        for algo in ALGORITHMS:
            print(algo)
            # 5 runs each
            for i in tqdm(range(5)):
                try:
                    with timeout(7200, exception=RuntimeError):
                        # starting time
                        st = time.time()
                        # construct clustering algorithm object
                        algorithm = ClusteringAlgorithm(algo, dataloader, minpts)
                        labels = algorithm.run(2)
                        # ending time
                        end = time.time()
                        # save results
                        results.append({'Dim': dim, 'Time': end - st, 'Algorithm': algo})
                except RuntimeError:
                    print('exceeded time limit')
                    break
                except Exception:
                    print('Exception')
    # save results to df and csv
    res_df = pd.DataFrame(results)
    res_df.to_csv('auxiliary/AuxExperiments/results/o_dim2_experiment.csv')

# perform the runtime experiment with varying ground truth clusters for the other algorithms
def k_experiment_other():
    # to collect results
    results = []
    # 5 runs each
    for i in range(5):
        # different number of ground truth clusters
        for k in tqdm([2, 3, 4, 5, 6, 7, 8, 9]):
            # read synthetic dataset
            df = pd.read_csv('auxiliary/AuxExperiments/Data/k/dataset_k_{}_{}.csv'.format(k, i))
            # create Dataloader object
            dataloader = DataLoader(df)
            dataloader.load()
            data_wo_sensitive = dataloader.get_data()
            data = np.array(data_wo_sensitive)
            # define minpts
            minpts = 2 * (data.shape[1] + 1) - 1
            # list algorithms
            ALGORITHMS = ['Scalable', 'FairSC_normalized', 'FairSC']  # , 'Fairlet']
            for algo in ALGORITHMS:
                print(algo)
                for j in tqdm(range(1)):
                    try:
                        with timeout(7200, exception=RuntimeError):
                            # starting time
                            st = time.time()
                            # construct clustering algorithm object
                            algorithm = ClusteringAlgorithm(algo, dataloader, minpts)
                            if algo == 'Fairlet':
                                names, labelss = algorithm.run(k)
                            else:
                                labels = algorithm.run(k)
                                # ending time
                                end = time.time()
                                # save results
                            results.append({'k': k, 'Time': end - st, 'Algorithm': algo, 'Dataset': i})
                    except RuntimeError:
                        print('exceeded time limit')
                        break
                    except Exception:
                        print('Exception')
    # save results to df and csv
    res_df = pd.DataFrame(results)
    res_df.to_csv('auxiliary/AuxExperiments/results/o_k_experiment.csv')

# perform the runtime experiment with varying dataset sizes for the other algorithms
def n_experiment_other():
    # to collect results
    results = []
    # different dataset sizes
    for n in tqdm([100, 200, 500, 1000, 2000, 5000, 10000, 20000]):
        # read synthetic dataset
        df = pd.read_csv('auxiliary/AuxExperiments/Data/n/dataset_n_{}.csv'.format(n))
        # create Dataloader object
        dataloader = DataLoader(df)
        dataloader.load()
        data_wo_sensitive = dataloader.get_data()
        data = np.array(data_wo_sensitive)
        # define minpts
        minpts = 2 * (data.shape[1] + 1) - 1
        ALGORITHMS = ['Scalable', 'FairSC_normalized', 'FairSC']
        for algo in ALGORITHMS:
            print(algo)
            # 5 runs each
            for i in tqdm(range(5)):
                try:
                    with timeout(7200, exception=RuntimeError):
                        # starting time
                        st = time.time()
                        # construct clustering algorithm object
                        algorithm = ClusteringAlgorithm(algo, dataloader, minpts)
                        labels = algorithm.run(2)
                        # ending time
                        end = time.time()
                        # save results
                        results.append({'n': n, 'Time': end - st, 'Algorithm': algo})
                except RuntimeError:
                    print('exceeded time limit')
                    break
                except Exception:
                    print('Exception')
    # save results to df and csv
    res_df = pd.DataFrame(results)
    res_df.to_csv('auxiliary/AuxExperiments/results/o_n_experiment.csv')

# perform the runtime experiment with varying dimensions for Fairlet
def dim_experiment_fairlet():
    # to collect results
    results = []
    # 5 runs each
    for i in range(5):
        # different dimensions
        for dim in tqdm([5, 10, 50, 100, 1000]):
            # read synthetic dataset
            df = pd.read_csv('auxiliary/AuxExperiments/Data/dim/dataset_dim_{}_{}.csv'.format(dim, i))
            # create Dataloader object
            dataloader = DataLoader(df)
            dataloader.load()
            data_wo_sensitive = dataloader.get_data()
            data = np.array(data_wo_sensitive)
            # define minpts
            minpts = 2 * (data.shape[1] + 1) - 1
            ALGORITHMS = ['Fairlet']
            for algo in ALGORITHMS:
                print(algo)
                for j in tqdm(range(1)):
                    try:
                        with timeout(7200, exception=RuntimeError):
                            # starting time
                            st = time.time()
                            # construct clustering algorithm object
                            algorithm = ClusteringAlgorithm(algo, dataloader, minpts)
                            names, labelss = algorithm.run(2)
                            # ending time
                            end = time.time()
                            # save results
                            results.append({'Dim': dim, 'Time': end - st, 'Algorithm': algo, 'Dataset': i})
                    except RuntimeError:
                        print('exceeded time limit')
                        break
                    except Exception:
                        print('Exception')
    # save results to df and csv
    res_df = pd.DataFrame(results)
    res_df.to_csv('auxiliary/AuxExperiments/results/fairlet_dim2_experiment.csv')

# perform the runtime experiment with varying ground truth clusters for Fairlet
def k_experiment_fairlet():
    # to collect results
    results = []
    # different numbers of ground truth clusters
    for k in tqdm([2, 3, 4, 5, 6, 7, 8, 9]):
        # read synthetic dataset
        df = pd.read_csv('auxiliary/AuxExperiments/Data/k/dataset_k_{}.csv'.format(k))
        # create Dataloader object
        dataloader = DataLoader(df)
        dataloader.load()
        data_wo_sensitive = dataloader.get_data()
        data = np.array(data_wo_sensitive)
        # define minpts
        minpts = 2 * (data.shape[1] + 1) - 1
        ALGORITHMS = ['Fairlet']
        for algo in ALGORITHMS:
            print(algo)
            # 5 runs each
            for i in tqdm(range(5)):
                try:
                    with timeout(7200, exception=RuntimeError):
                        # starting time
                        st = time.time()
                        # construct clustering algorithm object
                        algorithm = ClusteringAlgorithm(algo, dataloader, minpts)
                        names, labelss = algorithm.run(k)
                        # ending time
                        end = time.time()
                        # save results
                        results.append({'k': k, 'Time': end - st, 'Algorithm': algo})
                except RuntimeError:
                    print('exceeded time limit')
                    break
                except Exception:
                    print('Exception')
    # save results to df and csv
    res_df = pd.DataFrame(results)
    res_df.to_csv('auxiliary/AuxExperiments/results/fairlet_k_experiment.csv')

# perform the runtime experiment with varying dataset size for Fairlet
def n_experiment_fairlet():
    # to collect results
    results = []
    # different dataset sizes
    for n in tqdm([100, 200, 500, 1000, 2000, 5000, 10000]):  # , 20000]):
        # read synthetic dataset
        df = pd.read_csv('auxiliary/AuxExperiments/Data/n/dataset_n_{}.csv'.format(n))
        # create Dataloader object
        dataloader = DataLoader(df)
        dataloader.load()
        data_wo_sensitive = dataloader.get_data()
        data = np.array(data_wo_sensitive)
        # define minpts
        minpts = 2 * (data.shape[1] + 1) - 1
        ALGORITHMS = ['Fairlet']
        for algo in ALGORITHMS:
            print(algo)
            # 5 runs each
            for i in tqdm(range(5)):
                try:
                    with timeout(7200, exception=RuntimeError):
                        # starting time
                        st = time.time()
                        # construct clustering algorithm object
                        algorithm = ClusteringAlgorithm(algo, dataloader, minpts)
                        names, labelss = algorithm.run(2)
                        # ending time
                        end = time.time()
                        # save results
                        results.append({'n': n, 'Time': end - st, 'Algorithm': algo})

                except RuntimeError:
                    print('exceeded time limit')
                    break
                except Exception:
                    print('Exception')
    # save results to df and csv
    res_df = pd.DataFrame(results)
    res_df.to_csv('auxiliary/AuxExperiments/results/fairlet_n_experiment.csv')


# perform all runtime experiments
def experiments():
    print('Dimension Experiment FairDen variable minpts')
    dim_experiment()
    print('Dimension Experiment FairDen fixed minpts')
    dim_experiment2()
    print('k Experiment FairDen')
    k_experiment()
    print('n Experiment FairDen')
    n_experiment()

    print('Dimension Experiment other')
    dim_experiment_other()
    print('k Experiment other')
    k_experiment_other()
    print('n Experiment other')
    n_experiment_other()

    print('Dimension Experiment fairlet')
    dim_experiment_fairlet()
    print('k Experiment fairlet')
    k_experiment_fairlet()
    print('n Experiment fairlet')
    n_experiment_fairlet()

if __name__ == "__main__":
    experiments()
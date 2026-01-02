# This is just a wrapper for the sklearn implementation

from sklearn.cluster import DBSCAN

class VanillaDBSCAN:
    """
        DBSCAN-Clustering algorithm object.
    """
    def __init__(self, data):
        """
            Construct object for VanillaDBSCAN-Clustering.

                Parameters:
                        config (DictConfig): Dictionary with Hydra configuration.
                        data_loader : DataLoader-object encapsulating the configuration and data.

                Returns:
                        -
        """
        self.__data = data.get_data()
        self.__min_samples, self.__eps = data.get_dbscan_config()

    def run(self):
        """
            Run DBSCAN and return the labels.

                Parameters:
                        eps (float): defines the reachability threshold.

                Returns:
                        labels (np.ndarray): clustering labels.
        """
        clustering = DBSCAN(self.__eps, min_samples=self.__min_samples).fit(self.data)
        labels_ = clustering.labels_
        return labels_

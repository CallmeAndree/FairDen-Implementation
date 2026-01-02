import json

import pandas as pd
import numpy as np

from pandas.api.types import is_numeric_dtype
from sklearn.preprocessing import StandardScaler
from src.utils.DataEncoder import Goodall1


class DataLoader(object):

    def __init__(self, df):
        """
            Construct DataLoader object and setup according to the name.

            Parameters:
                dataname (str): name of the dataset. Defaults to 'default'.
                categorical (boolean): whether to include categorical data. Defaults to True.

        """
        self.__data_frame = None
        self.__points = None
        self.__data = df
        self.__dcsi_min_pts = 5
        # target column only used for classification
        self.__target_column = df['Label']
        # sensitive column
        self.__sensitive_columns = df[['Sensitive']].astype(int)
        # normalized data
        self.__normalized = None
        self.__data_wosensitive = None
        self.__n_clusters = len(np.unique(df['Label']))
        global RANDOM_STATE
        RANDOM_STATE = 42

    def get_dcsi_min_pts(self):
        """

            Returns:
                min_pts (int): minimum points parameters for DCSI.

        """
        return self.__dcsi_min_pts

    def get_sens_attr(self):
        """

            Returns:
                sensitive_names (list of str): list of names of sensitive attributes.

        """
        return self.__sensitive_columns

    def get_sens_mixed(self):
        """

            Returns:
                sensitive_mixed (array): mixed sensitive attributes (meta sensitive attribute).

        """
        return self.__sensitive_columns

    def get_num_clusters(self):
        """

            Returns:
                n_clusters (int): number of ground truth clusters.

        """
        return self.__n_clusters

    def get_num_categorical(self):
        """

            Returns:
                num_categorical (int): number of categorical features.

        """
        return 0

    def get_data(self, wo_sensitive=True):
        """

            Returns:
                normalized (array): Returns the data as numpy array, depending on if normalization is set or not.

        """
        if wo_sensitive:
            return self.__normalized_wosensitive
        return self.__normalized

    def get_data_as_list(self, wo_sensitive=True):
        """

            Returns:
                normalized (list): Returns the data as numpy array, depending on if normalization is set or not.

        """
        if wo_sensitive:
            return [list(i) for i in np.array(self.__normalized_wosensitive)]
        return [list(i) for i in np.array(self.__normalized)]

    def get_sensitive_columns(self):
        """

            Returns:
                sensitive_columns (pd.Series): Returns the sensitive columns.

        """
        return self.__sensitive_columns

    def get_target_columns(self):
        """

            Returns:
                target_column (array): Returns the labels for the ground truth clustering.

        """
        return self.__target_column

    def get_df(self):
        """

            Returns:
                dataframe (DataFrame): Returns the data as pd.DataFrame.

        """
        return self.__data_frame

    def load(self):
        self.__target_column = self.__data['Label']
        # make target column consist of numeric values only
        self.__target_column = self.__target_column.replace(self.__target_column.unique(),
                                                            list(range(0, len(self.__target_column.unique()))), )
        self.__target_column = self.__target_column.to_numpy()

        self.__points = self.__data.drop(['Sensitive'], axis=1).to_numpy()

        self.__data = self.__data.drop(['Label'], axis=1)

        self.__data_wosensitive = self.__data.drop(['Sensitive'], axis=1)
        self._normalize()
        self.__data_frame = self.__data
        self.__data = self.__data.to_numpy()

    def _normalize(self):
        """Standardize features by removing the mean and scaling to unit variance."""
        scaler = StandardScaler()
        self.__normalized = self.__data.copy()
        self.__normalized = scaler.fit_transform(self.__normalized)
        self.__normalized_wosensitive = self.__data_wosensitive.copy()
        self.__normalized_wosensitive = scaler.fit_transform(
            self.__normalized_wosensitive
        )

    def get_data_frame(self):
        return self.__data_frame

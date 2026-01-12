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

import json

import pandas as pd
import numpy as np

from pandas.api.types import is_numeric_dtype
from sklearn.preprocessing import StandardScaler
from src.utils.DataEncoder import Goodall1


class DataLoader(object):

    def __init__(self, dataname, categorical=True):
        """
            Construct DataLoader object and setup according to the name.

            Parameters:
                dataname (str): name of the dataset. Defaults to 'default'.
                categorical (boolean): whether to include categorical data. Defaults to True.

        """
        self.__name = dataname
        name = dataname
        print(dataname)
        if 'three' in dataname:
            with open('../../config/three_moons/{}.json'.format(name)) as json_file:
                self.__data_config = json.load(json_file)
        else:
            with open('config/realworld/{}.json'.format(name)) as json_file:
                self.__data_config = json.load(json_file)
        self.__dbscan_min_pts = self.__data_config['DBSCAN_min_pts']
        self.__dbscan_eps = self.__data_config['DBSCAN_eps']
        self.__dcsi_min_pts = self.__data_config['DCSI_min_pts']
        self.__n_clusters_db = self.__data_config['n_clusters_db']
        self.__categorical = categorical
        self.__data_to_encode = None
        self.__sensitive_combi_mixed = None
        # target column only used for classification
        self.__target_column = None
        # sensitive column
        self.__sensitive_columns = None
        # normalized data
        self.__normalized = None
        self.__data_wosensitive = None
        # combination of multiple sensitive attributes
        self.__sensitive_mixed = None
        self.__encoded_data = None
        self.__categories = None
        self.__all_sensitive = None
        self.__n_clusters = self.__data_config['n_clusters']
        global RANDOM_STATE
        if "random_state" in self.__data_config:
            RANDOM_STATE = self.__data_config['random_state']
        else:
            RANDOM_STATE = 42

    def get_dbscan_config(self):
        """

        Returns:
            min_pts (int): minimum points parameters for DBSCAN.
            eps (float): epsilon for DBSCAN.

        """
        return self.__dbscan_min_pts, self.__dbscan_eps

    def get_dcsi_min_pts(self):
        """

            Returns:
                min_pts (int): minimum points parameters for DCSI.

        """
        return self.__dcsi_min_pts

    def get_sens_combi_mixed(self):
        """

            Returns:
                sensitive_mixed (array): combination of mixed sensitive attributes.

        """
        return self.__sensitive_combi_mixed

    def get_n_clusters_db(self):
        """

            Returns:
                n_clusters_db (int): number of clusters detected by DBSCAN.

        """
        return self.__n_clusters_db

    def get_all_sensitive(self):
        """

            Returns:
                all_sensitive (array): columns of all sensitive attributes.

        """
        return self.__all_sensitive

    def get_sens_attr(self):
        """

            Returns:
                sensitive_names (list of str): list of names of sensitive attributes.

        """
        return self.__data_config['sensitive_attrs']

    def get_sens_mixed(self):
        """

            Returns:
                sensitive_mixed (array): mixed sensitive attributes (meta sensitive attribute).

        """
        return self.__sensitive_mixed

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
        return len(self.__data_config['categorical_features'])

    def get_encoded_data(self):
        """

            Returns:
                encoded_data (array): categorical data that has been encoded with Goodall1.

        """
        return self.__encoded_data

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

    def get_sensitive_mixed(self):
        """

            Returns:
                sensitive_mixed (array): Returns the column for the meta sensitive attribute.

        """
        return self.__sensitive_mixed

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
            self.__setup()
            if len(self.__data_config['categorical_features']) != 0:
                self.__data = self.__data[
                    self.__data_config['sensitive_attrs'] + [
                        self.__data_config['target']] + self.__data_config['columns'] + self.__data_config[
                        'categorical_features']
                    ]

            if len(self.__data_config['sensitive_attrs']) > 1:
                if len(self.__data_config['sensitive_attrs']) == 2:
                    self.__data["sensitive_attr"] = (
                            self.__data[str(self.__data_config['sensitive_attrs'][0])]
                            + self.__data[str(self.__data_config['sensitive_attrs'][1])]
                    )
                else:
                    self.__data["sensitive_attr"] = (
                            self.__data[str(self.__data_config['sensitive_attrs'][0])]
                            + self.__data[str(self.__data_config['sensitive_attrs'][1])]
                            + self.__data[str(self.__data_config['sensitive_attrs'][2])]
                    )
                if 'census' in self.__name or 'cens_' in self.__name:
                    self.__data = self.__data[
                        ["sensitive_attr"] + self.__data_config['sensitive_attrs'] + self.__data_config['columns'] +
                        self.__data_config['categorical_features']]

                else:
                    self.__data = self.__data[
                        ["sensitive_attr"]
                        + self.__data_config['sensitive_attrs']
                        + [self.__data_config['target']]
                        + self.__data_config['columns']
                        + self.__data_config['categorical_features']
                        ]
            if 'creditcard' in self.__name or 'adult_' in self.__name:
                self.__all_sensitive = self.__all_sensitive[self.__all_sensitive.index.isin(self.__data.index)]
                for column in self.__all_sensitive:
                    columnData = self.__all_sensitive[column]
                    labels, counts = np.unique(columnData, return_counts=True)
                    values = list(labels)
                    for label, count in zip(labels, counts):
                        if count < self.__data_config['n_clusters'] or count < self.__n_clusters_db:
                            values.remove(label)
                    self.__all_sensitive = self.__all_sensitive[
                        self.__all_sensitive[column].isin(values)
                    ]
                self.__data = self.__data[self.__data.index.isin(self.__all_sensitive.index)]
            # For synthetic data (cens_ configs need sensitive value encoding, unlike census)
            if "synthetic" not in self.__name and "three_moons" not in self.__name and 'census' not in self.__name:

                for sens_attr in self.__data_config['sensitive_attrs']:
                    values = self.__data_config['sensitive_values']

                    label, count = np.unique(self.__data[sens_attr], return_counts=True)
                    for label, count in zip(label, count):
                        if count < self.__data_config['n_clusters'] or count < self.__n_clusters_db:
                            if label in values:
                                values.remove(label)

                    if len(self.__data_config['sensitive_values']) != 0:
                        # Remove data that has different values than the defined ones in column sensitive_attr
                        self.__data = self.__data[
                            self.__data[sens_attr].isin(values)
                        ]

                    sensitive_values = np.unique(self.__data[sens_attr])
                    # replace sensitive values with numbers
                    self.__data[sens_attr] = self.__data[sens_attr].replace(
                        sensitive_values,
                        list(range(0, len(sensitive_values)))
                    )

                if len(self.__data_config['categorical_features']) != 0 and 'cens_' not in self.__name:
                    self.__data_to_encode = self.__data[self.__data_config['categorical_features']]
                    if len(self.__data_config['sensitive_attrs']) == 1:
                        self.__data = self.__data[
                            self.__data_config['sensitive_attrs'] + [self.__data_config['target']] + self.__data_config[
                                'columns']
                            ]
                    else:
                        self.__data = self.__data[["sensitive_attr"] +
                                                  self.__data_config['sensitive_attrs'] + [
                                                      self.__data_config['target']] + self.__data_config[
                                                      'columns']
                                                  ]
                if len(self.__data_config['sensitive_attrs']) > 1:
                    sensitive_values = np.unique(self.__data["sensitive_attr"])
                    # replace sensitive values with numbers
                    self.__data["sensitive_attr"] = self.__data["sensitive_attr"].replace(
                        sensitive_values,
                        list(range(0, len(sensitive_values))),
                    )
            if 'census' not in self.__name and 'cens_' not in self.__name:
                target_values = self.__data[self.__data_config['target']].unique()
                # Replace target values with numerical values
                # self.__data[self.__target_name].replace(
                #    target_values, list(range(0, len(target_values))), inplace=True
                # )
                self.__data[self.__data_config['target']] = self.__data[self.__data_config['target']].replace(
                    target_values, list(range(0, len(target_values))))

            # Remove duplicates from the dataset because this leads to mistakes otherwise
            # self.__data = self.__data.drop_duplicates(
            #    subset=self.__data_config['sensitive_attrs'] + self.__data_config['columns'], ignore_index=True
            # )
            if 'creditcard' in self.__name or 'adult_' in self.__name:
                self.__all_sensitive = self.__all_sensitive[self.__all_sensitive.index.isin(self.__data.index)]
                columns = list(self.__all_sensitive.columns)
                sensitive_mix = pd.DataFrame(self.__all_sensitive)
                sensitive_mix['combi'] = (
                        self.__all_sensitive[str(columns[0])]
                        + self.__all_sensitive[str(columns[1])]
                        + self.__all_sensitive[str(columns[2])]
                )
                labels, counts = np.unique(sensitive_mix, return_counts=True)
                values = list(labels)
                for label, count in zip(labels, counts):
                    if count < self.__data_config['n_clusters'] or count < self.__n_clusters_db:
                        values.remove(label)
                sensitive_mix = sensitive_mix[
                    sensitive_mix['combi'].isin(values)]
                self.__sensitive_combi_mixed = sensitive_mix['combi']
                self.__all_sensitive = self.__all_sensitive[
                    self.__all_sensitive.index.isin(sensitive_mix.index)]
                self.__data = self.__data[self.__data.index.isin(sensitive_mix.index)]

            # change datatype to numeric by replacing with a number
            for column in self.__data_config['columns']:
                if not is_numeric_dtype(self.__data[column]):
                    self.__data[column].replace(
                        self.__data[column].unique(),
                        list(range(0, len(self.__data[column].unique()))),
                        inplace=True,
                    )
            if 'census' not in self.__name and 'cens_' not in self.__name:
                self.__target_column = self.__data[self.__data_config['target']]
                # make target column consist of numeric values only
                self.__target_column = self.__target_column.replace(
                    self.__target_column.unique(),
                    list(range(0, len(self.__target_column.unique()))),
                )
                self.__target_column = self.__target_column.to_numpy()

            self.__points = self.__data.drop(self.__data_config['sensitive_attrs'], axis=1).to_numpy()
            self.__sensitive_columns = self.__data[self.__data_config['sensitive_attrs']]
            if len(self.__data_config['sensitive_attrs']) > 1:
                self.__data_fairden = self.__data[
                    ["sensitive_attr"] + self.__data_config['columns']
                    ].copy()
                self.__sensitive_mixed = self.__data[
                    ["sensitive_attr"]]

            else:
                self.__data_fairden = self.__data[
                    self.__data_config['sensitive_attrs'] + self.__data_config['columns']
                    ].copy()
                self.__data = self.__data[self.__data_config['sensitive_attrs'] + self.__data_config['columns']].copy()

            self.__data_wosensitive = self.__data[self.__data_config['columns']].copy()
            self._normalize()
            self.__data_frame = self.__data
            self.__data = self.__data.to_numpy()
            if len(self.__data_config['categorical_features']) > 0 and self.__categorical:
                self.encode_categorical()

    def _normalize(self):
        """Standardize features by removing the mean and scaling to unit variance."""
        scaler = StandardScaler()
        self.__normalized = self.__data.copy()

        self.__normalized = scaler.fit_transform(self.__normalized)

        self.__normalized_wosensitive = self.__data_wosensitive.copy()
        self.__normalized_wosensitive = scaler.fit_transform(
            self.__normalized_wosensitive
        )

    def __setup(self):
        if 'bank' in self.__name:
            self.__data = pd.read_csv(self.__data_config['file_name'], sep=";")
            self.__data = self.__data.drop_duplicates(
                subset=self.__data_config['sensitive_attrs'] + self.__data_config['columns'], ignore_index=True
            )
            if self.__data_config['n_samples'] != 'all':
                self.__data = self.__data.sample(
                    n=self.__data_config['n_samples'], random_state=RANDOM_STATE
                )

            if len(self.__data_config['categorical_features']) != 0:
                self.__data_to_encode = self.__data[self.__data_config['categorical_features']]
        elif 'census' in self.__name:
            self.__data = pd.read_csv("data/realworld/census1990.csv")
            self.__data = self.__data[
                ["dAncstry1", "dAncstry2", "iAvail", "iCitizen", "iClass", "dDepart", "iDisabl1", "iDisabl2",
                 "iEnglish", "iFeb55", "iFertil", "dHispanic", "dHour89", "dHours", "iImmigr", "dIncome1",
                 "dIncome2", "dIncome3", "dIncome4", "dIncome5", "dIncome6", "dIncome7", "dIncome8",
                 "dIndustry", "iKorean", "iLang1", "iLooking", "iMarital", "iMay75880", "iMeans", "iMilitary",
                 "iMobility", "iMobillim", "dOccup", "iOthrserv", "iPerscare", "dPOB", "dPoverty", "dPwgt1",
                 "iRagechld", "dRearning", "iRelat1", "iRelat2", "iRemplpar", "iRiders", "iRlabor", "iRownchld",
                 "dRpincome", "iRPOB", "iRrelchld", "iRspouse", "iRvetserv", "iSchool", "iSept80", "iSubfam1",
                 "iSubfam2", "iTmpabsnt", "dTravtime", "iVietnam", "dWeek89", "iWork89", "iWorklwk", "iWWII",
                 "iYearsch", "iYearwrk", "dYrsserv"] + self.__data_config['sensitive_attrs']]

            self.__data = self.__data.drop_duplicates(
                subset=self.__data_config['sensitive_attrs'] + self.__data_config['columns'], ignore_index=True
            )
            self.__data = self.__data.sample(n=self.__data_config['n_samples'], random_state=RANDOM_STATE).reset_index(
                drop=True)


        elif 'adult' in self.__name or 'cens_' in self.__name:
            self.__data = pd.read_csv(self.__data_config['file_name'])
            self.__data = self.__data.drop_duplicates(
                subset=self.__data_config['sensitive_attrs'] + self.__data_config['columns'], ignore_index=True
            )
            if self.__data_config['n_samples'] != 'all':
                self.__data = self.__data.sample(
                    n=self.__data_config['n_samples'], random_state=RANDOM_STATE
                )

            if len(self.__data_config['categorical_features']) != 0:
                self.__data_to_encode = self.__data[self.__data_config['categorical_features']]
            self.__all_sensitive = self.__data[['gender', 'marital_status', 'race']]
        elif 'creditcard' in self.__name:
            self.__data = pd.read_csv(self.__data_config['file_name'])
            self.__data['EDUCATION'] = self.__data['EDUCATION'].apply(str)
            self.__data['MARRIAGE'] = self.__data['MARRIAGE'].apply(str)
            self.__data['SEX'] = self.__data['SEX'].apply(str)
            self.__data = self.__data.drop_duplicates(
                subset=self.__data_config['sensitive_attrs'] + self.__data_config['columns'], ignore_index=True
            )
            if self.__data_config['n_samples'] != 'all':
                self.__data = self.__data.sample(
                    n=self.__data_config['n_samples'], random_state=RANDOM_STATE
                )

            self.__all_sensitive = self.__data[['EDUCATION', 'MARRIAGE', 'SEX']]
            if len(self.__data_config['categorical_features']) != 0:
                self.__data_to_encode = self.__data[self.__data_config['categorical_features']]
        elif self.__name == 'communities':
            attrib = pd.read_csv(self.__data_config['file_name'][0], delim_whitespace=True)
            data = pd.read_csv(
                self.__data_config['file_name'][1], names=attrib["attributes"]
            )
            # We create a new binary class label namely class
            # based on ViolentCrimesPerPop attribute
            # (the total number of violent crimes per 100,000 population).
            # As illustrated in the related work (Kearns et al., 2018),
            # a label “high-crime” is set if the crime rate of the communities (positive class)
            # is greater than 0.7, otherwise, “low-crime” is given.
            data["high-crime"] = np.where(data["ViolentCrimesPerPop"] > 0.7, 1, 0)
            # In the literature (Kamiran et al., 2013; Kamishima et al., 2012),
            # typically, researchers derive a new attribute, namely Black,
            # which is considered as the protected attribute, to divide
            # the communities according to race by thresholding the attribute
            # racepctblack (the percentage of the population that is African American) at 0.06.
            data["black"] = np.where(data["racepctblack"] > 0.06, "black", "else")
            # remove all columns including NAN values
            # Based on the suggestions from the literature
            # (Calders et al., 2013; Heidari et al., 2018),
            # we remove all columns containing missing values.
            data = data.dropna(axis="columns")
            # remove categorical
            data = data.drop(
                ["state", "communityname", "fold", "racePctWhite", "racePctAsian", "racePctHisp", "whitePerCap",
                 "blackPerCap", "indianPerCap", "AsianPerCap", "OtherPerCap", "HispPerCap", "NumIlleg", "PctIlleg",
                 "NumImmig", "PctImmigRecent", "PctImmigRec5", "PctImmigRec8", "PctImmigRec10", "PctRecentImmig",
                 "PctRecImmig5", "PctRecImmig8", "PctRecImmig10", "PctSpeakEnglOnly", "PctNotSpeakEnglWell"], axis=1)

            # read data from csv and remove 'Unnamed: 0' column
            self.__data = data
        elif "compas" in self.__name:
            self.__data = pd.read_csv(self.__data_config['file_name'])
            self.__data = self.__data.dropna(axis="columns")
            self.__data = self.__data.drop_duplicates(
                subset=self.__data_config['sensitive_attrs'] + self.__data_config['columns'], ignore_index=True
            )
            if self.__data_config['n_samples'] != 'all':
                self.__data = self.__data.sample(
                    n=self.__data_config['n_samples'], random_state=RANDOM_STATE
                )

            if len(self.__data_config['categorical_features']) != 0:
                self.__data_to_encode = self.__data[self.__data_config['categorical_features']]
        elif self.__name == 'diabetes':
            # read csv file
            self.__data = pd.read_csv(self.__data_config['file_name'])
            # make age into numerical numbers taking the mean of the interval
            age_buckets = {
                "[70-80)": 75,
                "[60-70)": 65,
                "[50-60)": 55,
                "[80-90)": 85,
                "[40-50)": 45,
                "[30-40)": 35,
                "[90-100)": 95,
                "[20-30)": 25,
                "[10-20)": 15,
                "[0-10)": 5,
            }
            self.__data["age"] = self.__data.apply(
                lambda x: age_buckets[x["age"]], axis=1
            )
            # make readmission into a binary column
            readmission = {
                "No": 0,
                "NO": 0,
                "<30": 0,
                ">30": 1,
            }
            self.__data["readmitted"] = self.__data.apply(
                lambda x: readmission[x["readmitted"]], axis=1
            )
            self.__data = self.__data.drop_duplicates(
                subset=self.__data_config['sensitive_attrs'] + self.__data_config['columns'], ignore_index=True
            )
            if self.__data_config['n_samples'] != 'all':
                self.__data = self.__data.sample(
                    n=self.__data_config['n_samples'], random_state=RANDOM_STATE
                )
            if len(self.__data_config['categorical_features']) != 0:
                self.__data_to_encode = self.__data[self.__data_config['categorical_features']]
        elif "student" in self.__name:
            # Student Performance dataset
            self.__data = pd.read_csv(self.__data_config['file_name'])
            self.__data = self.__data.dropna(axis="columns")
            self.__data = self.__data.drop_duplicates(
                subset=self.__data_config['sensitive_attrs'] + self.__data_config['columns'], ignore_index=True
            )
            if self.__data_config['n_samples'] != 'all':
                self.__data = self.__data.sample(
                    n=self.__data_config['n_samples'], random_state=RANDOM_STATE
                )

            if len(self.__data_config['categorical_features']) != 0:
                self.__data_to_encode = self.__data[self.__data_config['categorical_features']]
        elif "synthetic" in self.__name or "three_moons" in self.__name:
            # read data from csv and remove 'Unnamed: 0' column
            if 'three_moons' in self.__name:
                self.__data = pd.read_csv(self.__data_config['file_name'])
            else:
                self.__data = pd.read_csv(self.__data_config['file_name'] + '{}.csv'.format(self.__name))
            self.__n_clusters = len(np.unique(self.__data['cluster']))

    def encode_categorical(self):
        self.__data_to_encode = self.__data_to_encode[self.__data_to_encode.index.isin(self.__data_frame.index)]
        self.__encoded_data = Goodall1(self.__data_to_encode)

    def get_data_frame(self):
        return self.__data_frame

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

import networkx
import numpy as np

from src.comparative_methods.fairlets.Fairlet import Fairlet
from src.comparative_methods.fairsc.FairSC import FairSC
from src.comparative_methods.vanilladbscan.VanillaDBSCAN import VanillaDBSCAN
from src.comparative_methods.scalable_fair.Scala import Scala
from src.comparative_methods.FairSC_normalized.FairSC_normalized import FairSC_normalized
from src.FairDen import FairDen


class ClusteringAlgorithm(object):
    def __init__(self, name, dataloader, min_pts, dataname='default'):
        """
            Construct ClusteringAlgorithm object and set corresponding to algorithm name provided.

            Parameters:
                name (str): name of the algorithm to be instantiated.
                data_loader (DataLoader): DataLoader object encapsulating the dataset.
                min_pts (int): min_pts parameter for dc_distance.
                dataname (str): name of the dataset. Defaults to 'default'.

        """
        self.__name = name
        self.__algorithm = None
        self.__dataloader = dataloader
        self.__min_pts = min_pts
        self.__dataname = dataname
        self.__setup()

    def __setup(self):
        if self.__name == "Fairlet":
            if self.__dataname is None:
                distance_threshold = 225000
            else:
                distance_thresholds = {'bank3': 400, 'bank4': 400, 'adult': 225000, 'adult4': 225000,
                                       'communities': 225000, 'compas3': 225000, 'compas4': 225000,
                                       'diabetes': 40, 'default': 225000}
                if self.__dataname not in distance_thresholds.keys():
                    distance_threshold = distance_thresholds['default']
                else:
                    distance_threshold = distance_thresholds[self.__dataname]
            try:
                self.__algorithm = Fairlet(self.__dataloader, distance_threshold)
            except networkx.exception.NetworkXUnfeasible:
                raise Exception("NetworkX Unfeasible")
        elif self.__name == "FairSC":
            self.__algorithm = FairSC(self.__dataloader)
        elif self.__name == "FairDen":
            self.__algorithm = FairDen(self.__dataloader, self.__min_pts)
        elif self.__name == "VanillaDBSCAN":
            self.__algorithm = VanillaDBSCAN(self.__dataloader)
        elif self.__name == "Scalable":
            self.__algorithm = Scala(self.__dataloader)
        elif self.__name == "FairSC_normalized":
            self.__algorithm = FairSC_normalized(self.__dataloader)
        else:
            raise ValueError("Algorithm {} is not yet implemented.".format(self.__name))

    def run(self, n_clusters=None):
        """ for n_clusters number of clusters """
        if n_clusters is None:
            n_clusters = self.__dataloader.get_num_clusters()

        if (
                self.__name == "Scalable"
                or self.__name == "FairSC"
                or self.__name == "FairSC_normalized"
        ):
            mapping = self.__algorithm.run(n_clusters)
            return mapping

        elif "FairDen" in self.__name:
            name = "{}_MinPts_{}_Deg_{}".format(
                self.__name,
                self.__min_pts,
                n_clusters
            )
            mapping = self.__algorithm.run(n_clusters)
            if mapping is None:
                return None
            return np.array(mapping)

        elif self.__name == "VanillaDBSCAN":
            mapping = self.__algorithm.run()
            return mapping
        elif self.__name == "Fairlet":
            names = []
            mappings = []
            self.labels = ["MCF Fairlet"]
            # for both types of Fairlets
            for label in self.labels:
                algo = self.__name + "_" + label
                name = "{}".format(algo)
                mapping, centers = self.__algorithm.run_experiment(
                    n_clusters, label
                )
                predictions = mapping
                predictions.sort(key=lambda x: x[0])
                labels = [x for i, x in predictions]
                targets = np.unique(labels).tolist()
                mapping = np.array([targets.index(x) for i, x in predictions])
                names.append(name)
                mappings.append(mapping)
            return names, mappings
        else:
            raise ValueError("Algorithm {} is not yet implemented.".format(self.__name))

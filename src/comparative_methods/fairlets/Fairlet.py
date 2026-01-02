# Implementation of Fairlets by
# - Author: Akhil Gupta, Anunay Sharma, Ayush Rajput, Badrinarayanan Rajasekaran, and Vishnu Pratheek Challa
# - Source: https://github.com/guptakhil/fair-clustering-fairlets
# - License: MIT

# Paper: Fair Clustering Through Fairlets
# Authors: Flavio Chierichetti, Ravi Kumar, Silvio Lattanzi, Sergei Vassilvitskii
# Link: https://arxiv.org/abs/1802.05733



import time
import matplotlib.pyplot as plt
from src.comparative_methods.fairlets.KCenter import KCenters
from src.comparative_methods.fairlets.VanillaFairletDecomposition import (
    VanillaFairletDecomposition,
)
import numpy as np
from src.comparative_methods.fairlets.MCFFairletDecomposition import (
    MCFFairletDecomposition,
)


class Fairlet:
    def __init__(self, data, distance_threshold):
        self.__data = data.get_data()

        mask = data.get_sensitive_columns() == 0
        blues = np.where(mask == True)[0].tolist()
        reds = np.where(mask == False)[0].tolist()
        if len(blues) < len(reds):
            temp = blues
            blues = reds
            reds = temp

        ratio = len(reds) / len(blues)
        print(ratio)
        p = 1
        q = int(round(1 / ratio + 0.5))

        self.vfd = VanillaFairletDecomposition(
            p, q, blues, reds, self.__data.tolist(),
        )
        self.mcf = MCFFairletDecomposition(
            blues,
            reds,
            2,
            distance_threshold,
            self.__data.tolist(),
        )
        self.setup_vanilla()
        self.setup_mcf()

    def setup_vanilla(self):
        (
            vanilla_fairlets,
            vanilla_fairlet_centers,
            vanilla_fairlet_costs,
        ) = self.vfd.decompose()
        self.vanilla_fairlets = vanilla_fairlets
        self.vanilla_fairlet_centers = vanilla_fairlet_centers
        self.vanilla_fairlet_costs = vanilla_fairlet_costs

    def setup_mcf(self):
        self.mcf.compute_distances()
        self.mcf.build_graph(plot_graph=False)
        mcf_fairlets, mcf_fairlet_centers, mcf_fairlet_costs = self.mcf.decompose()
        self.mcf_fairlets = mcf_fairlets
        self.mcf_fairlet_centers = mcf_fairlet_centers
        self.mcf_fairlet_costs = mcf_fairlet_costs

    def run_experiment(self, degree, type):
        if type == 'MCF Fairlet':
            mcf_mapping, mcf_centers = self.run_mcf_fairlet(degree=degree)
            return mcf_mapping, mcf_centers
        elif type == 'Vanilla Fairlet':
            v_mapping, v_centers = self.run_vanilla_fairlet(degree=degree)
            return v_mapping, v_centers
        else:
            raise ValueError('Wrong type of fairlet')

    def run_vanilla_fairlet(self, degree):
        return self.run_fairlet(degree, self.vanilla_fairlets, self.vanilla_fairlet_centers)

    def run_mcf_fairlet(self, degree):
        return self.run_fairlet(degree, self.mcf_fairlets, self.mcf_fairlet_centers)

    def run_fairlet(self, degree, fairlets, fairlet_centers):
        data = self.__data
        kcenters = KCenters(k=degree)
        kcenters.fit([data[i] for i in fairlet_centers])
        mapping = kcenters.assign()

        final_clusters = []
        for fairlet_id, final_cluster in mapping:
            for point in fairlets[fairlet_id]:
                final_clusters.append((point, fairlet_centers[final_cluster]))

        centers = [fairlet_centers[i] for i in kcenters.centers]
        return final_clusters, centers

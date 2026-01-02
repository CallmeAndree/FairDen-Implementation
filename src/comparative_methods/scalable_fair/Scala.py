# Implementation of Scalable Fair Clustering by
# - Author: Arturs Backurs, Piotr Indyk, Krzysztof Onak, Baruch Schieber, Ali Vakilian, Tal Wagner
# - Source: https://github.com/talwagner/fair_clustering
# - License: MIT License

# Paper: Scalable Fair Clustering
# Authors: Arturs Backurs, Piotr Indyk, Krzysztof Onak, Baruch Schieber, Ali Vakilian, Tal Wagner
# Link: https://arxiv.org/abs/1902.03519

# Our modifications:
#    (1) reimplementation without matlab engine

import numpy as np
from collections import defaultdict
from sklearn_extra.cluster import KMedoids

# import matlab.engine
from scipy.spatial.distance import cdist
import math


class Scala:
    def __init__(self, data):
        self.__data = data.get_data()
        self.colors = data.get_sensitive_columns()
        self.points = self.__data
        mask = data.get_sensitive_columns() == 0
        blues = np.where(mask == True)[0].tolist()
        reds = np.where(mask == False)[0].tolist()
        if len(blues) < len(reds):
            temp = blues
            blues = reds
            reds = temp
        ratio = len(reds) / len(blues)
        self.colors = self.colors[self.colors.columns[0]].to_list()
        self.p = 1
        self.q = int(round(1 / ratio + 0.5))
        self.epsilon = 0.0001
        self.fairlet_centers = []
        self.fairlets = []
        root = self.build_quadtree(self.points)
        cost = self.tree_fairlet_decomposition(
            self.p, self.q, root, self.points, self.colors
        )
        # Fairlet Center Points
        self.fairlet_center_pt = np.array(
            [np.array(self.points[index]) for index in self.fairlet_centers]
        )

    def run(self, k):
        X = self.fairlet_center_pt

        # C: Cluster medoid locations, returned as a numeric matrix.
        # C is a k-by-d matrix, where row j is the medoid of cluster j
        #
        # midx: Index to mat_matrix, returned as a column vector of indices.
        # midx is a k-by-1 vector and the indices satisfy C = X(midx,:)
        kmedoids = KMedoids(
            n_clusters=k,
            random_state=0,
            metric="euclidean",
        ).fit(X)
        labels = kmedoids.labels_
        centers = kmedoids.cluster_centers_
        center_index = kmedoids.medoid_indices_
        # indices of center points in dataset
        centroids = [self.fairlet_centers[index] for index in center_index]
        # kmedian_cost = self.fair_kmedian_cost(centroids, self.points)
        center_points = [self.points[centroids[i]] for i in range(k)]
        dist_matr = cdist(center_points, self.points)
        labels_ = np.argmin(dist_matr, axis=0)
        return labels_

    # def run(self,k):

    # mat_matrix = matlab.double(self.fairlet_center_pt.tolist())

    # # Run k-mediod code in Matlab
    # eng = matlab.engine.start_matlab()

    # # C: Cluster medoid locations, returned as a numeric matrix.
    # # C is a k-by-d matrix, where row j is the medoid of cluster j
    # #
    # # midx: Index to mat_matrix, returned as a column vector of indices.
    # # midx is a k-by-1 vector and the indices satisfy C = X(midx,:)

    # idx,C,sumd,D,midx,info = eng.kmedoids(mat_matrix, k,'Distance','euclidean', nargout=6)

    # np_midx = (np.array(midx._data)).flatten()
    # c_idx_matrix = np_midx.astype(int)
    # #in matlab, arrays are numbered from 1
    # c_idx_matrix[:] = [index - 1 for index in c_idx_matrix]

    # # indices of center points in dataset
    # centroids = [self.fairlet_centers[index] for index in c_idx_matrix]

    # kmedian_cost = self.fair_kmedian_cost(centroids, self.points)

    # return centroids

    def kmedian_cost(self, points, centroids, dataset):
        "Computes and returns k-median cost for given dataset and centroids"
        return sum(
            np.amin(
                np.concatenate(
                    [
                        np.linalg.norm(
                            dataset[:, :] - dataset[centroid, :], axis=1
                        ).reshape((dataset.shape[0], 1))
                        for centroid in centroids
                    ],
                    axis=1,
                ),
                axis=1,
            )
        )

    def fair_kmedian_cost(self, centroids, dataset):
        "Return the fair k-median cost for given centroids and fairlet decomposition"
        total_cost = 0
        for i in range(len(self.fairlets)):
            # Choose index of centroid which is closest to the i-th fairlet center
            cost_list = [
                np.linalg.norm(
                    dataset[centroids[j], :] - dataset[self.fairlet_centers[i], :]
                )
                for j in range(len(centroids))
            ]
            cost, j = min((cost, j) for (j, cost) in enumerate(cost_list))
            # Assign all points in i-th fairlet to above centroid and compute cost
            total_cost += sum(
                [
                    np.linalg.norm(dataset[centroids[j], :] - dataset[point, :])
                    for point in self.fairlets[i]
                ]
            )
        return total_cost

    ### FAIRLET DECOMPOSITION CODE ###

    def balanced(self, p, q, r, b):
        if r == 0 and b == 0:
            return True
        if r == 0 or b == 0:
            return False
        return min(r * 1.0 / b, b * 1.0 / r) >= p * 1.0 / q

    def make_fairlet(self, points, dataset):
        "Adds fairlet to fairlet decomposition, returns median cost"
        self.fairlets.append(points)
        cost_list = [
            sum(
                [
                    np.linalg.norm(dataset[center, :] - dataset[point, :])
                    for point in points
                ]
            )
            for center in points
        ]
        cost, center = min((cost, center) for (center, cost) in enumerate(cost_list))
        self.fairlet_centers.append(points[center])
        return cost

    def basic_fairlet_decomposition(self, p, q, blues, reds, dataset):
        """
        Computes vanilla (p,q)-fairlet decomposition of given points (Lemma 3 in NIPS17 paper).
        Returns cost.
        Input: Balance parameters p,q which are non-negative integers satisfying p<=q and gcd(p,q)=1.
        "blues" and "reds" are sets of points indices with balance at least p/q.
        """

        assert p <= q, "Please use balance parameters in the correct order"
        if len(reds) < len(blues):
            temp = blues
            blues = reds
            reds = temp
        R = len(reds)
        B = len(blues)
        assert self.balanced(p, q, R, B), (
                "Input sets are unbalanced: " + str(R) + "," + str(B)
        )

        if R == 0 and B == 0:
            return 0

        b0 = 0
        r0 = 0
        cost = 0
        while (R - r0) - (B - b0) >= q - p and R - r0 >= q and B - b0 >= p:
            cost += self.make_fairlet(reds[r0: r0 + q] + blues[b0: b0 + p], dataset)
            r0 += q
            b0 += p
        if R - r0 + B - b0 >= 1 and R - r0 + B - b0 <= p + q:
            cost += self.make_fairlet(reds[r0:] + blues[b0:], dataset)
            r0 = R
            b0 = B
        elif R - r0 != B - b0 and B - b0 >= p:
            cost += self.make_fairlet(
                reds[r0: r0 + (R - r0) - (B - b0) + p] + blues[b0: b0 + p], dataset
            )
            r0 += (R - r0) - (B - b0) + p
            b0 += p
        assert R - r0 == B - b0, "Error in computing fairlet decomposition"
        for i in range(R - r0):
            cost += self.make_fairlet([reds[r0 + i], blues[b0 + i]], dataset)
        return cost

    def node_fairlet_decomposition(self, p, q, node, dataset, donelist, depth):
        # Leaf
        if len(node.children) == 0:
            node.reds = [i for i in node.reds if donelist[i] == 0]
            node.blues = [i for i in node.blues if donelist[i] == 0]
            assert self.balanced(
                p, q, len(node.reds), len(node.blues)
            ), "Reached unbalanced leaf"
            return self.basic_fairlet_decomposition(
                p, q, node.blues, node.reds, dataset
            )

        # Preprocess children nodes to get rid of points that have already been clustered
        for child in node.children:
            child.reds = [i for i in child.reds if donelist[i] == 0]
            child.blues = [i for i in child.blues if donelist[i] == 0]

        R = [len(child.reds) for child in node.children]
        B = [len(child.blues) for child in node.children]

        if sum(R) == 0 or sum(B) == 0:
            assert (
                    sum(R) == 0 and sum(B) == 0
            ), "One color class became empty for this node while the other did not"
            return 0

        NR = 0
        NB = 0

        # Phase 1: Add must-remove nodes
        for i in range(len(node.children)):
            if R[i] >= B[i]:
                must_remove_red = max(0, R[i] - int(np.floor(B[i] * q * 1.0 / p)))
                R[i] -= must_remove_red
                NR += must_remove_red
            else:
                must_remove_blue = max(0, B[i] - int(np.floor(R[i] * q * 1.0 / p)))
                B[i] -= must_remove_blue
                NB += must_remove_blue

        # Calculate how many points need to be added to smaller class until balance
        if NR >= NB:
            # Number of missing blues in (NR,NB)
            missing = max(0, int(np.ceil(NR * p * 1.0 / q)) - NB)
        else:
            # Number of missing reds in (NR,NB)
            missing = max(0, int(np.ceil(NB * p * 1.0 / q)) - NR)

        # Phase 2: Add may-remove nodes until (NR,NB) is balanced or until no more such nodes
        for i in range(len(node.children)):
            if missing == 0:
                assert self.balanced(p, q, NR, NB), "Something went wrong"
                break
            if NR >= NB:
                may_remove_blue = B[i] - int(np.ceil(R[i] * p * 1.0 / q))
                remove_blue = min(may_remove_blue, missing)
                B[i] -= remove_blue
                NB += remove_blue
                missing -= remove_blue
            else:
                may_remove_red = R[i] - int(np.ceil(B[i] * p * 1.0 / q))
                remove_red = min(may_remove_red, missing)
                R[i] -= remove_red
                NR += remove_red
                missing -= remove_red

        # Phase 3: Add unsatuated fairlets until (NR,NB) is balanced
        for i in range(len(node.children)):
            if self.balanced(p, q, NR, NB):
                break
            if R[i] >= B[i]:
                num_saturated_fairlets = int(R[i] / q)
                excess_red = R[i] - q * num_saturated_fairlets
                excess_blue = B[i] - p * num_saturated_fairlets
            else:
                num_saturated_fairlets = int(B[i] / q)
                excess_red = R[i] - p * num_saturated_fairlets
                excess_blue = B[i] - q * num_saturated_fairlets
            R[i] -= excess_red
            NR += excess_red
            B[i] -= excess_blue
            NB += excess_blue

        assert self.balanced(p, q, NR, NB), "Constructed node sets are unbalanced"

        reds = []
        blues = []
        for i in range(len(node.children)):
            for j in node.children[i].reds[R[i]:]:
                reds.append(j)
                donelist[j] = 1
            for j in node.children[i].blues[B[i]:]:
                blues.append(j)
                donelist[j] = 1

        assert len(reds) == NR and len(blues) == NB, "Something went horribly wrong"

        return self.basic_fairlet_decomposition(p, q, blues, reds, dataset) + sum(
            [
                self.node_fairlet_decomposition(
                    p, q, child, dataset, donelist, depth + 1
                )
                for child in node.children
            ]
        )

    def tree_fairlet_decomposition(self, p, q, root, dataset, colors):
        "Main fairlet clustering function, returns cost wrt original metric (not tree metric)"
        assert p <= q, "Please use balance parameters in the correct order"

        root.populate_colors(colors)
        assert self.balanced(
            p, q, len(root.reds), len(root.blues)
        ), "Dataset is unbalanced"
        root.populate_colors(colors)
        donelist = [0] * dataset.shape[0]
        return self.node_fairlet_decomposition(p, q, root, dataset, donelist, 0)

    ### QUADTREE CODE ###

    def build_quadtree(self, dataset, max_levels=0, random_shift=True):
        "If max_levels=0 there no level limit, quadtree will partition until all clusters are singletons"
        dimension = dataset.shape[1]
        lower = np.amin(dataset, axis=0)
        upper = np.amax(dataset, axis=0)

        shift = np.zeros(dimension)
        if random_shift:
            for d in range(dimension):
                spread = upper[d] - lower[d]
                shift[d] = np.random.uniform(0, spread)
                upper[d] += spread

        return self.build_quadtree_aux(
            dataset, range(dataset.shape[0]), lower, upper, max_levels, shift
        )

    def build_quadtree_aux(self, dataset, cluster, lower, upper, max_levels, shift):
        """
        "lower" is the "bottom-left" (in all dimensions) corner of current hypercube
        "upper" is the "upper-right" (in all dimensions) corner of current hypercube
        """

        dimension = dataset.shape[1]
        cell_too_small = True
        for i in range(dimension):
            if upper[i] - lower[i] > self.epsilon:
                cell_too_small = False

        node = TreeNode()
        if max_levels == 1 or len(cluster) <= 1 or cell_too_small:
            # Leaf
            node.set_cluster(cluster)
            return node

        # Non-leaf
        midpoint = 0.5 * (lower + upper)
        subclusters = defaultdict(list)
        for i in cluster:
            subclusters[
                tuple(
                    [dataset[i, d] + shift[d] <= midpoint[d] for d in range(dimension)]
                )
            ].append(i)
        for edge, subcluster in subclusters.items():
            sub_lower = np.zeros(dimension)
            sub_upper = np.zeros(dimension)
            for d in range(dimension):
                if edge[d]:
                    sub_lower[d] = lower[d]
                    sub_upper[d] = midpoint[d]
                else:
                    sub_lower[d] = midpoint[d]
                    sub_upper[d] = upper[d]
            node.add_child(
                self.build_quadtree_aux(
                    dataset, subcluster, sub_lower, sub_upper, max_levels - 1, shift
                )
            )
        return node


class TreeNode:
    def __init__(self):
        self.children = []

    def set_cluster(self, cluster):
        self.cluster = cluster

    def add_child(self, child):
        self.children.append(child)

    def populate_colors(self, colors):
        "Populate auxiliary lists of red and blue points for each node, bottom-up"
        self.reds = []
        self.blues = []
        if len(self.children) == 0:
            # Leaf
            for i in self.cluster:
                if colors[i] == 0:
                    self.reds.append(i)
                else:
                    self.blues.append(i)
        else:
            # Not a leaf
            for child in self.children:
                child.populate_colors(colors)
                self.reds.extend(child.reds)
                self.blues.extend(child.blues)

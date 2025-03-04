from pathlib import Path

import numpy as np


from spikeinterface.sortingcomponents.clustering.peak_svd import extract_peaks_svd
from spikeinterface.sortingcomponents.clustering.graph_tools import create_graph_from_peak_features


class GraphClustering:
    """
    Simple clustering by constructing a global sparse graph using local slinding bins along the probe.

    The edge of the graph is constructed using local distance bewteen svd on waveforms.

    Then a classic algorithm like louvain or hdbscan is used.
    """

    _default_params = {
        "radius_um": 180.,
        "bin_um": 60.,
        "motion": None,
        "seed": None,
        "n_neighbors": 100,
    }

    @classmethod
    def main_function(cls, recording, peaks, params, job_kwargs=dict()):

        import networkx as nx

        radius_um = params["radius_um"]
        bin_um = params["bin_um"]
        motion = params["motion"]
        seed = params["seed"]
        n_neighbors = params["n_neighbors"]

        motion_aware = motion is not None

        assert radius_um >= bin_um * 3

        

        peaks_svd, sparse_mask, _ = extract_peaks_svd(
            recording, peaks,
            radius_um=radius_um,
            motion_aware=motion_aware,
            motion=None,
        )
        # print(peaks_svd.shape)



        channel_locations = recording.get_channel_locations()
        channel_depth = channel_locations[:, 1]
        peak_depths = channel_depth[peaks["channel_index"]]

        # order peaks by depth
        order = np.argsort(peak_depths)
        ordered_peaks = peaks[order]
        ordered_peaks_svd = peaks_svd[order]

        # TODO : try to use real peak location

        distances = create_graph_from_peak_features(
            recording,
            ordered_peaks,
            ordered_peaks_svd,
            sparse_mask,
            peak_locations=None,
            bin_um=bin_um,
            dim=1,
            # mode="full_connected_bin",
            mode="knn",
            direction="y",
            n_neighbors=n_neighbors,
        )
        # print(distances)
        # print(distances.shape)
        # print("sparsity: ", distances.indices.size / (distances.shape[0]**2))        


        distances_bool = distances.copy()
        distances_bool.data[:] = 1

        G = nx.Graph(distances_bool)
        communities = nx.community.louvain_communities(G, seed=seed)

        peak_labels = np.zeros(ordered_peaks.size, dtype=int)
        peak_labels[:] = -1

        k = 0
        for community in communities:
            if len(community) == 1:
                continue
            peak_labels[list(community)] = k
            k += 1
        # print(peak_labels)

        labels_set = np.unique(peak_labels)
        labels_set = labels_set[labels_set >= 0]

        # we need to reorder labels
        reverse_order = np.argsort(order)
        peak_labels = peak_labels[reverse_order]
        
        return labels_set, peak_labels

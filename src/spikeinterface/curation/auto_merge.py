from __future__ import annotations

import numpy as np

from ..core import create_sorting_analyzer
from ..core.template import Templates
from ..core.template_tools import get_template_extremum_channel
from ..postprocessing import compute_correlograms
from ..qualitymetrics import compute_refrac_period_violations, compute_firing_rates

from .mergeunitssorting import MergeUnitsSorting
from .merge_temporal_splits import compute_presence_distance


def get_potential_auto_merge(
    sorting_analyzer,
    minimum_spikes=100,
    maximum_distance_um=150.0,
    peak_sign="neg",
    bin_ms=0.25,
    window_ms=100.0,
    corr_diff_thresh=0.16,
    template_diff_thresh=0.25,
    censored_period_ms=0.3,
    refractory_period_ms=1.0,
    sigma_smooth_ms=0.6,
    contamination_threshold=0.2,
    adaptative_window_threshold=0.5,
    censor_correlograms_ms: float = 0.15,
    num_channels=5,
    num_shift=5,
    firing_contamination_balance=2.5,
    extra_outputs=False,
    steps=None,
    presence_distance_thresh=100,
    preset=None,
    template_metric="l1",
    p_value=0.2,
    CC_threshold=0.1,
    k_nn=5,
    **presence_distance_kwargs,
):
    """
    Algorithm to find and check potential merges between units.

    This is taken from lussac version 1 done by Aurelien Wyngaard and Victor Llobet.
    https://github.com/BarbourLab/lussac/blob/v1.0.0/postprocessing/merge_units.py


    The merges are proposed when the following criteria are met:

        * STEP 1: enough spikes are found in each units for computing the correlogram (`minimum_spikes`)
        * STEP 2: each unit is not contaminated (by checking auto-correlogram - `contamination_threshold`)
        * STEP 3: estimated unit locations are close enough (`maximum_distance_um`)
        * STEP 4: the cross-correlograms of the two units are similar to each auto-corrleogram (`corr_diff_thresh`)
        * STEP 5: the templates of the two units are similar (`template_diff_thresh`)
        * STEP 6: [optional] the presence distance of two units
        * STEP 7: [optional] the cross-contamination is not significant
        * STEP 8: the unit "quality score" is increased after the merge.

    The "quality score" factors in the increase in firing rate (**f**) due to the merge and a possible increase in
    contamination (**C**), wheighted by a factor **k** (`firing_contamination_balance`).

    .. math::

        Q = f(1 - (k + 1)C)


    Parameters
    ----------
    sorting_analyzer : SortingAnalyzer
        The SortingAnalyzer
    minimum_spikes : int, default: 100
        Minimum number of spikes for each unit to consider a potential merge.
        Enough spikes are needed to estimate the correlogram
    maximum_distance_um : float, default: 150
        Maximum distance between units for considering a merge
    peak_sign : "neg" | "pos" | "both", default: "neg"
        Peak sign used to estimate the maximum channel of a template
    bin_ms : float, default: 0.25
        Bin size in ms used for computing the correlogram
    window_ms : float, default: 100
        Window size in ms used for computing the correlogram
    corr_diff_thresh : float, default: 0.16
        The threshold on the "correlogram distance metric" for considering a merge.
        It needs to be between 0 and 1
    template_diff_thresh : float, default: 0.25
        The threshold on the "template distance metric" for considering a merge.
        It needs to be between 0 and 1
    template_metric : 'l1'
        The metric to be used when comparing templates. Default is l1 norm
    censored_period_ms : float, default: 0.3
        Used to compute the refractory period violations aka "contamination"
    refractory_period_ms : float, default: 1
        Used to compute the refractory period violations aka "contamination"
    sigma_smooth_ms : float, default: 0.6
        Parameters to smooth the correlogram estimation
    contamination_threshold : float, default: 0.2
        Threshold for not taking in account a unit when it is too contaminated
    adaptative_window_threshold : : float, default: 0.5
        Parameter to detect the window size in correlogram estimation
    censor_correlograms_ms : float, default: 0.15
        The period to censor on the auto and cross-correlograms
    num_channels : int, default: 5
        Number of channel to use for template similarity computation
    num_shift : int, default: 5
        Number of shifts in samles to be explored for template similarity computation
    firing_contamination_balance : float, default: 2.5
        Parameter to control the balance between firing rate and contamination in computing unit "quality score"
    presence_distance_thresh: float, default: 100
        Parameter to control how present two units should be simultaneously
    extra_outputs : bool, default: False
        If True, an additional dictionary (`outs`) with processed data is returned
    steps : None or list of str, default: None
        which steps to run (gives flexibility to running just some steps)
        If None all steps are done (except presence_distance).
        Pontential steps : "min_spikes", "remove_contaminated", "unit_positions", "correlogram",
        "template_similarity", "presence_distance", "check_increase_score".
        Please check steps explanations above!
    template_metric : 'l1', 'l2' or 'cosine'
        The metric to consider when measuring the distances between templates. Default is l1

    Returns
    -------
    potential_merges:
        A list of tuples of 2 elements.
        List of pairs that could be merged.
    outs:
        Returned only when extra_outputs=True
        A dictionary that contains data for debugging and plotting.
    """
    import scipy

    sorting = sorting_analyzer.sorting
    recording = sorting_analyzer.recording
    unit_ids = sorting.unit_ids
    sorting.register_recording(recording)

    # to get fast computation we will not analyse pairs when:
    #    * not enough spikes for one of theses
    #    * auto correlogram is contaminated
    #    * to far away one from each other

    all_steps = [
        "min_spikes",
        "remove_contaminated",
        "unit_positions",
        "correlogram",
        "template_similarity",
        "presence_distance",
        "knn",
        "cross_contamination",
        "check_increase_score",
    ]

    if steps is None:
        if preset is None:
            steps = [
                "min_spikes",
                "remove_contaminated",
                "unit_positions",
                "template_similarity",
                "correlogram",
                "check_increase_score",
            ]
        elif preset == "temporal_splits":
            steps = [
                "min_spikes",
                "remove_contaminated",
                "unit_positions",
                "template_similarity",
                "correlogram",
                "presence_distance",
                "check_increase_score",
            ]
        elif preset == "lussac":
            steps = [
                "min_spikes",
                "remove_contaminated",
                "unit_positions",
                "template_similarity",
                "cross_contamination",
                "check_increase_score",
            ]
        elif preset == "knn":
            steps = [
                "min_spikes",
                "remove_contaminated",
                "unit_positions",
                "knn",
                "correlogram",
                "check_increase_score",
            ]

    n = unit_ids.size
    pair_mask = np.triu(np.arange(n)) > 0
    outs = dict()

    for step in steps:

        assert step in all_steps, f"{step} is not a valid step"

        # STEP 1 :
        if step == "min_spikes":
            num_spikes = sorting.count_num_spikes_per_unit(outputs="array")
            to_remove = num_spikes < minimum_spikes
            pair_mask[to_remove, :] = False
            pair_mask[:, to_remove] = False

        # STEP 2 : remove contaminated auto corr
        elif step == "remove_contaminated":
            contaminations, nb_violations = compute_refrac_period_violations(
                sorting_analyzer, refractory_period_ms=refractory_period_ms, censored_period_ms=censored_period_ms
            )
            nb_violations = np.array(list(nb_violations.values()))
            contaminations = np.array(list(contaminations.values()))
            to_remove = contaminations > contamination_threshold
            pair_mask[to_remove, :] = False
            pair_mask[:, to_remove] = False

        # STEP 3 : unit positions are estimated roughly with channel
        elif step == "unit_positions" in steps:
            positions_ext = sorting_analyzer.get_extension("unit_locations")
            if positions_ext is not None:
                unit_locations = positions_ext.get_data()[:, :2]
            else:
                chan_loc = sorting_analyzer.get_channel_locations()
                unit_max_chan = get_template_extremum_channel(
                    sorting_analyzer, peak_sign=peak_sign, mode="extremum", outputs="index"
                )
                unit_max_chan = list(unit_max_chan.values())
                unit_locations = chan_loc[unit_max_chan, :]

            unit_distances = scipy.spatial.distance.cdist(unit_locations, unit_locations, metric="euclidean")
            pair_mask = pair_mask & (unit_distances <= maximum_distance_um)
            outs["unit_distances"] = unit_distances

        # STEP 4 : potential auto merge by correlogram
        elif step == "correlogram" in steps:
            correlograms_ext = sorting_analyzer.get_extension("correlograms")
            if correlograms_ext is not None:
                correlograms, bins = correlograms_ext.get_data()
            else:
                correlograms, bins = compute_correlograms(sorting, window_ms=window_ms, bin_ms=bin_ms, method="numba")
            mask = (bins[:-1] >= -censor_correlograms_ms) & (bins[:-1] < censor_correlograms_ms)
            correlograms[:, :, mask] = 0
            correlograms_smoothed = smooth_correlogram(correlograms, bins, sigma_smooth_ms=sigma_smooth_ms)
            # find correlogram window for each units
            win_sizes = np.zeros(n, dtype=int)
            for unit_ind in range(n):
                auto_corr = correlograms_smoothed[unit_ind, unit_ind, :]
                thresh = np.max(auto_corr) * adaptative_window_threshold
                win_size = get_unit_adaptive_window(auto_corr, thresh)
                win_sizes[unit_ind] = win_size
            correlogram_diff = compute_correlogram_diff(
                sorting,
                correlograms_smoothed,
                win_sizes,
                pair_mask=pair_mask,
            )
            # print(correlogram_diff)
            pair_mask = pair_mask & (correlogram_diff < corr_diff_thresh)
            outs["correlograms"] = correlograms
            outs["bins"] = bins
            outs["correlograms_smoothed"] = correlograms_smoothed
            outs["correlogram_diff"] = correlogram_diff
            outs["win_sizes"] = win_sizes

        # STEP 5 : check if potential merge with CC also have template similarity
        elif step == "template_similarity" in steps:
            template_similarity_ext = sorting_analyzer.get_extension("template_similarity")
            if template_similarity_ext is not None:
                templates_similarity = template_similarity_ext.get_data()
                templates_diff = 1 - templates_similarity

            else:
                templates_ext = sorting_analyzer.get_extension("templates")
                assert (
                    templates_ext is not None
                ), "auto_merge with template_similarity requires a SortingAnalyzer with extension templates"
                templates_array = templates_ext.get_data(outputs="numpy")

                templates_diff = compute_templates_diff(
                    sorting,
                    templates_array,
                    num_channels=num_channels,
                    num_shift=num_shift,
                    pair_mask=pair_mask,
                    template_metric=template_metric,
                    sparsity=sorting_analyzer.sparsity,
                )

            pair_mask = pair_mask & (templates_diff < template_diff_thresh)
            outs["templates_diff"] = templates_diff

        elif step == "knn" in steps:
            pair_mask = get_pairs_via_nntree(sorting_analyzer, k_nn, pair_mask)

        # STEP 6 : [optional] check how the rates overlap in times
        elif step == "presence_distance" in steps:
            presence_distances = compute_presence_distance(sorting, pair_mask, **presence_distance_kwargs)
            pair_mask = pair_mask & (presence_distances > presence_distance_thresh)
            outs["presence_distances"] = presence_distances

        # STEP 7 : [optional] check if the cross contamination is significant
        elif step == "cross_contamination" in steps:
            refractory = (censored_period_ms, refractory_period_ms)
            CC, p_values = compute_cross_contaminations(sorting_analyzer, pair_mask, CC_threshold, refractory)
            pair_mask = pair_mask & (p_values > p_value)
            outs["cross_contaminations"] = CC, p_values

        # STEP 8 : validate the potential merges with CC increase the contamination quality metrics
        elif step == "check_increase_score" in steps:
            pair_mask, pairs_decreased_score = check_improve_contaminations_score(
                sorting_analyzer,
                pair_mask,
                contaminations,
                firing_contamination_balance,
                refractory_period_ms,
                censored_period_ms,
            )
            outs["pairs_decreased_score"] = pairs_decreased_score

    # FINAL STEP : create the final list from pair_mask boolean matrix
    ind1, ind2 = np.nonzero(pair_mask)
    potential_merges = list(zip(unit_ids[ind1], unit_ids[ind2]))

    if extra_outputs:
        return potential_merges, outs
    else:
        return potential_merges


def get_pairs_via_nntree(sorting_analyzer, k_nn=5, pair_mask=None, sparse_distances=False):

    sorting = sorting_analyzer.sorting
    unit_ids = sorting.unit_ids
    n = len(unit_ids)

    if pair_mask is None:
        pair_mask = np.ones((n, n), dtype="bool")

    spike_positions = sorting_analyzer.get_extension("spike_locations").get_data()
    spike_amplitudes = sorting_analyzer.get_extension("spike_amplitudes").get_data()
    spikes = sorting_analyzer.sorting.to_spike_vector()

    ## We need to build a sparse distance matrix
    data = np.vstack((spike_amplitudes, spike_positions["x"], spike_positions["y"])).T
    from sklearn.neighbors import NearestNeighbors

    data = (data - data.mean(0)) / data.std(0)

    if sparse_distances:
        import scipy.sparse
        import sklearn.metrics

        distances = scipy.sparse.lil_matrix((len(data), len(data)), dtype=np.float32)

        for unit_ind1 in range(2):
            valid = pair_mask[unit_ind1, unit_ind1+1:]
            valid_indices = np.arange(unit_ind1+1, n)[valid]
            mask_2 = np.isin(spikes["unit_index"], valid_indices)
            if np.sum(mask_2) > 0:
                mask_1 = spikes["unit_index"] == unit_ind1
                tmp = sklearn.metrics.pairwise_distances(data[mask_1], data[mask_2])
                distances[mask_1][:, mask_2] = tmp

    all_spike_counts = sorting_analyzer.sorting.count_num_spikes_per_unit()
    all_spike_counts = np.array(list(all_spike_counts.keys()))

    if sparse_distances:
        kdtree = NearestNeighbors(n_neighbors=k_nn, n_jobs=-1, metric="precomputed")
        kdtree.fit(distances)
    else:
        kdtree = NearestNeighbors(n_neighbors=k_nn, n_jobs=-1)
        kdtree.fit(data)

    for unit_ind in range(n):
        print(unit_ind)
        mask = spikes["unit_index"] == unit_ind
        ind = kdtree.kneighbors(data[mask], return_distance=False)
        ind = ind.flatten()
        chan_inds, all_counts = np.unique(spikes["unit_index"][ind], return_counts=True)
        all_counts = all_counts.astype(float)
        #all_counts /= all_spike_counts[chan_inds]
        best_indices = np.argsort(all_counts)[::-1][1:]
        pair_mask[unit_ind] &= np.isin(np.arange(n), chan_inds[best_indices])
    return pair_mask


def compute_correlogram_diff(sorting, correlograms_smoothed, win_sizes, pair_mask=None):
    """
    Original author: Aurelien Wyngaard (lussac)

    Parameters
    ----------
    sorting : BaseSorting
        The sorting object.
    correlograms_smoothed : array 3d
        The 3d array containing all cross and auto correlograms
        (smoothed by a convolution with a gaussian curve).
    win_sizes : np.array[int]
        Window size for each unit correlogram.
    pair_mask : None or boolean array
        A bool matrix of size (num_units, num_units) to select
        which pair to compute.

    Returns
    -------
    corr_diff : 2D array
        The difference between the cross-correlogram and the auto-correlogram
        for each pair of units.
    """
    unit_ids = sorting.unit_ids
    n = len(unit_ids)

    if pair_mask is None:
        pair_mask = np.ones((n, n), dtype="bool")

    # Index of the middle of the correlograms.
    m = correlograms_smoothed.shape[2] // 2
    num_spikes = sorting.count_num_spikes_per_unit(outputs="array")

    corr_diff = np.full((n, n), np.nan, dtype="float64")
    for unit_ind1 in range(n):
        for unit_ind2 in range(unit_ind1 + 1, n):
            if not pair_mask[unit_ind1, unit_ind2]:
                continue

            num1, num2 = num_spikes[unit_ind1], num_spikes[unit_ind2]

            # Weighted window (larger unit imposes its window).
            win_size = int(round((num1 * win_sizes[unit_ind1] + num2 * win_sizes[unit_ind2]) / (num1 + num2)))
            # Plage of indices where correlograms are inside the window.
            corr_inds = np.arange(m - win_size, m + win_size, dtype=int)

            # TODO : for Aurelien
            shift = 0
            auto_corr1 = normalize_correlogram(correlograms_smoothed[unit_ind1, unit_ind1, :])
            auto_corr2 = normalize_correlogram(correlograms_smoothed[unit_ind2, unit_ind2, :])
            cross_corr = normalize_correlogram(correlograms_smoothed[unit_ind1, unit_ind2, :])
            diff1 = np.sum(np.abs(cross_corr[corr_inds - shift] - auto_corr1[corr_inds])) / len(corr_inds)
            diff2 = np.sum(np.abs(cross_corr[corr_inds - shift] - auto_corr2[corr_inds])) / len(corr_inds)
            # Weighted difference (larger unit imposes its difference).
            w_diff = (num1 * diff1 + num2 * diff2) / (num1 + num2)
            corr_diff[unit_ind1, unit_ind2] = w_diff

    return corr_diff


def normalize_correlogram(correlogram: np.ndarray):
    """
    Normalizes a correlogram so its mean in time is 1.
    If correlogram is 0 everywhere, stays 0 everywhere.

    Parameters
    ----------
    correlogram (np.ndarray):
        Correlogram to normalize.

    Returns
    -------
    normalized_correlogram (np.ndarray) [time]:
        Normalized correlogram to have a mean of 1.
    """
    mean = np.mean(correlogram)
    return correlogram if mean == 0 else correlogram / mean


def smooth_correlogram(correlograms, bins, sigma_smooth_ms=0.6):
    """
    Smooths cross-correlogram with a Gaussian kernel.
    """
    import scipy.signal

    # OLD implementation : smooth correlogram by low pass filter
    # b, a = scipy.signal.butter(N=2, Wn = correlogram_low_pass / (1e3 / bin_ms /2), btype="low")
    # correlograms_smoothed = scipy.signal.filtfilt(b, a, correlograms, axis=2)

    # new implementation smooth by convolution with a Gaussian kernel
    if len(correlograms) == 0:  # fftconvolve will not return the correct shape.
        return np.empty(correlograms.shape, dtype=np.float64)

    smooth_kernel = np.exp(-(bins**2) / (2 * sigma_smooth_ms**2))
    smooth_kernel /= np.sum(smooth_kernel)
    smooth_kernel = smooth_kernel[None, None, :]
    correlograms_smoothed = scipy.signal.fftconvolve(correlograms, smooth_kernel, mode="same", axes=2)

    return correlograms_smoothed


def get_unit_adaptive_window(auto_corr: np.ndarray, threshold: float):
    """
    Computes an adaptive window to correlogram (basically corresponds to the first peak).
    Based on a minimum threshold and minimum of second derivative.
    If no peak is found over threshold, recomputes with threshold/2.

    Parameters
    ----------
    auto_corr : np.ndarray
        Correlogram used for adaptive window.
    threshold : float
        Minimum threshold of correlogram (all peaks under this threshold are discarded).

    Returns
    -------
    unit_window : int
        Index at which the adaptive window has been calculated.
    """
    import scipy.signal

    if np.sum(np.abs(auto_corr)) == 0:
        return 20.0

    derivative_2 = -np.gradient(np.gradient(auto_corr))
    peaks = scipy.signal.find_peaks(derivative_2)[0]

    keep = auto_corr[peaks] >= threshold
    peaks = peaks[keep]
    keep = peaks < (auto_corr.shape[0] // 2)
    peaks = peaks[keep]

    if peaks.size == 0:
        # If none of the peaks crossed the threshold, redo with threshold/2.
        return get_unit_adaptive_window(auto_corr, threshold / 2)

    # keep the last peak (nearest to center)
    win_size = auto_corr.shape[0] // 2 - peaks[-1]

    return win_size


def compute_cross_contaminations(analyzer, pair_mask, CC_threshold, refractory_period):
    """
    Looks at a sorting analyzer, and returns statistical tests for cross_contaminations

    Parameters
    ----------
    analyzer : SortingAnalyzer
        The analyzer to look at
    CC_treshold : float, default: 0.1
        The threshold on the cross-contamination.
        Any pair above this threshold will not be considered.
    refractory_period : array/list/tuple of 2 floats
        (censored_period_ms, refractory_period_ms)

    """

    sorting = analyzer.sorting
    unit_ids = sorting.unit_ids
    n = len(unit_ids)
    sf = analyzer.recording.sampling_frequency
    n_frames = analyzer.recording.get_num_samples()
    from spikeinterface.sortingcomponents.merging.lussac import estimate_cross_contamination

    if pair_mask is None:
        pair_mask = np.ones((n, n), dtype="bool")

    CC = np.zeros((n, n), dtype=np.float32)
    p_values = np.zeros((n, n), dtype=np.float32)

    for unit_ind1 in range(len(unit_ids)):

        unit_id1 = unit_ids[unit_ind1]
        spike_train1 = np.array(sorting.get_unit_spike_train(unit_id1))

        for unit_ind2 in range(unit_ind1 + 1, len(unit_ids)):
            if not pair_mask[unit_ind1, unit_ind2]:
                continue

            unit_id2 = unit_ids[unit_ind2]
            spike_train2 = np.array(sorting.get_unit_spike_train(unit_id2))
            # Compuyting the cross-contamination difference
            CC[unit_ind1, unit_ind2], p_values[unit_ind1, unit_ind2] = estimate_cross_contamination(
                spike_train1, spike_train2, sf, n_frames, refractory_period, limit=CC_threshold
            )

    return CC, p_values


def compute_templates_diff(
    sorting, templates_array, num_channels=5, num_shift=5, pair_mask=None, template_metric="l1", sparsity=None
):
    """
    Computes normalized template differences.

    Parameters
    ----------
    sorting : BaseSorting
        The sorting object
    templates_array : np.array
        The templates array (num_units, num_samples, num_channels).
    num_channels : int, default: 5
        Number of channel to use for template similarity computation
    num_shift : int, default: 5
        Number of shifts in samles to be explored for template similarity computation
    pair_mask : None or boolean array
        A bool matrix of size (num_units, num_units) to select
        which pair to compute.
    template_metric : 'l1', 'l2' or 'cosine'
        The metric to consider when measuring the distances between templates. Default is l1
    sparsity : None or ChannelSparsity
        Optionaly a ChannelSparsity object.

    Returns
    -------
    templates_diff : np.array
        2D array with template differences
    """
    unit_ids = sorting.unit_ids
    n = len(unit_ids)
    assert template_metric in ["l1", "l2", "cosine"], "Not a valid metric!"

    if pair_mask is None:
        pair_mask = np.ones((n, n), dtype="bool")

    if sparsity is None:
        adaptative_masks = False
        sparsity_mask = None
    else:
        adaptative_masks = num_channels == None
        sparsity_mask = sparsity.mask

    templates_diff = np.full((n, n), np.nan, dtype="float64")
    all_shifts = range(-num_shift, num_shift + 1)
    for unit_ind1 in range(n):
        for unit_ind2 in range(unit_ind1 + 1, n):
            if not pair_mask[unit_ind1, unit_ind2]:
                continue

            template1 = templates_array[unit_ind1]
            template2 = templates_array[unit_ind2]
            # take best channels
            if not adaptative_masks:
                chan_inds = np.argsort(np.max(np.abs(template1 + template2), axis=0))[::-1][:num_channels]
            else:
                chan_inds = np.flatnonzero(sparsity_mask[unit_ind1] * sparsity_mask[unit_ind2])

            if len(chan_inds) > 0:
                template1 = template1[:, chan_inds]
                template2 = template2[:, chan_inds]

                num_samples = template1.shape[0]
                if template_metric == "l1":
                    norm = np.sum(np.abs(template1)) + np.sum(np.abs(template2))
                elif template_metric == "l2":
                    norm = np.sum(template1**2) + np.sum(template2**2)
                elif template_metric == "cosine":
                    norm = np.linalg.norm(template1) * np.linalg.norm(template2)
                all_shift_diff = []
                for shift in all_shifts:
                    temp1 = template1[num_shift : num_samples - num_shift, :]
                    temp2 = template2[num_shift + shift : num_samples - num_shift + shift, :]
                    if template_metric == "l1":
                        d = np.sum(np.abs(temp1 - temp2)) / norm
                    elif template_metric == "l2":
                        d = np.linalg.norm(temp1 - temp2) / norm
                    elif template_metric == "cosine":
                        d = 1 - np.sum(temp1 * temp2) / norm
                    all_shift_diff.append(d)
            else:
                all_shift_diff = [1] * len(all_shifts)

            templates_diff[unit_ind1, unit_ind2] = np.min(all_shift_diff)

    return templates_diff


def check_improve_contaminations_score(
    sorting_analyzer, pair_mask, contaminations, firing_contamination_balance, refractory_period_ms, censored_period_ms
):
    """
    Check that the score is improve after a potential merge

    The score is a balance between:
      * contamination decrease
      * firing increase

    Check that the contamination score is improved (decrease)  after
    a potential merge
    """
    recording = sorting_analyzer.recording
    sorting = sorting_analyzer.sorting
    pair_mask = pair_mask.copy()
    pairs_removed = []

    firing_rates = list(compute_firing_rates(sorting_analyzer).values())

    inds1, inds2 = np.nonzero(pair_mask)
    for i in range(inds1.size):
        ind1, ind2 = inds1[i], inds2[i]

        c_1 = contaminations[ind1]
        c_2 = contaminations[ind2]

        f_1 = firing_rates[ind1]
        f_2 = firing_rates[ind2]

        # make a merged sorting and tale one unit (unit_id1 is used)
        unit_id1, unit_id2 = sorting.unit_ids[ind1], sorting.unit_ids[ind2]
        sorting_merged = MergeUnitsSorting(
            sorting, [[unit_id1, unit_id2]], new_unit_ids=[unit_id1], delta_time_ms=censored_period_ms
        ).select_units([unit_id1])

        sorting_analyzer_new = create_sorting_analyzer(sorting_merged, recording, format="memory", sparse=False)

        new_contaminations, _ = compute_refrac_period_violations(
            sorting_analyzer_new, refractory_period_ms=refractory_period_ms, censored_period_ms=censored_period_ms
        )
        c_new = new_contaminations[unit_id1]
        f_new = compute_firing_rates(sorting_analyzer_new)[unit_id1]

        # old and new scores
        k = firing_contamination_balance
        score_1 = f_1 * (1 - (k + 1) * c_1)
        score_2 = f_2 * (1 - (k + 1) * c_2)
        score_new = f_new * (1 - (k + 1) * c_new)

        if score_new < score_1 or score_new < score_2:
            # the score is not improved
            pair_mask[ind1, ind2] = False
            pairs_removed.append((sorting.unit_ids[ind1], sorting.unit_ids[ind2]))

    return pair_mask, pairs_removed

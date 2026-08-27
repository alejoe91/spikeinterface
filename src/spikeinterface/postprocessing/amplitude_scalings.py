import numpy as np

from spikeinterface.core import ChannelSparsity
from spikeinterface.core.template_tools import get_dense_templates_array, _get_nbefore
from spikeinterface.core.sortinganalyzer import register_result_extension
from spikeinterface.core.analyzer_extension_core import BaseSpikeVectorExtension

from spikeinterface.core.node_pipeline import SpikeRetriever, PipelineNode, find_parent_of_type


class ComputeAmplitudeScalings(BaseSpikeVectorExtension):
    """
    Computes the amplitude scalings from a SortingAnalyzer.

    Amplitude scalings are the scaling factor to multiply the
    unit template to best match the waveform. Each waveform
    has an associated amplitude scaling.

    In the case where there are not spike collisions, the scaling is
    the regression of the waveform onto the template, with intercept:
        scaling * template + intercept = waveform

    When there are spike collisions, a different approach is taken.
    Spike collisions are sets of temporally and spatially overlapping spikes.
    Therefore, signal from other spikes can contribute to the amplitude
    of the spike of interest. To address this, a multivariate linear
    regression is used to regress the waveform (that contains multiple spikes,
    the spike of interest and colliding spikes) onto a set of templates.

    Parameters
    ----------
    sorting_analyzer: SortingAnalyzer
        A SortingAnalyzer object
    sparsity: ChannelSparsity or None, default: None
        If waveforms are not sparse, sparsity is required if the number of channels is greater than
        `max_dense_channels`. If the waveform extractor is sparse, its sparsity is automatically used.
    max_dense_channels: int, default: 16
        Maximum number of channels to allow running without sparsity. To compute amplitude scaling using
        dense waveforms, set this to None, sparsity to None, and pass dense waveforms as input.
    ms_before : float or None, default: None
        The cut out to apply before the spike peak to extract local waveforms.
        If None, the SortingAnalyzer ms_before is used.
    ms_after : float or None, default: None
        The cut out to apply after the spike peak to extract local waveforms.
        If None, the SortingAnalyzer ms_after is used.
    handle_collisions: bool, default: True
        Whether to handle collisions between spikes. If True, the amplitude scaling of colliding spikes
        (defined as spikes within `delta_collision_ms` ms and with overlapping sparsity) is computed by fitting a
        multi-linear regression model (with `sklearn.LinearRegression`). If False, each spike is fitted independently.
    delta_collision_ms: float, default: 2
        The maximum time difference in ms before and after a spike to gather colliding spikes.
    """

    extension_name = "amplitude_scalings"
    depend_on = ["templates"]
    nodepipeline_variables = ["amplitude_scalings", "collision_mask"]

    def _set_params(
        self,
        sparsity=None,
        max_dense_channels=16,
        ms_before=None,
        ms_after=None,
        handle_collisions=True,
        delta_collision_ms=2,
    ):
        return super()._set_params(
            sparsity=sparsity,
            max_dense_channels=max_dense_channels,
            ms_before=ms_before,
            ms_after=ms_after,
            handle_collisions=handle_collisions,
            delta_collision_ms=delta_collision_ms,
        )

    def _get_sparsity(self):
        """
        Resolve the ChannelSparsity used to compute the amplitude scalings, either from the
        `sparsity` parameter, from the SortingAnalyzer sparsity, or dense.

        Returns
        -------
        sparsity : ChannelSparsity
            The sparsity used to extract local waveforms and to define spatial overlap between units.
        """
        recording = self.sorting_analyzer.recording

        if self.sorting_analyzer.is_sparse() and self.params["sparsity"] is None:
            sparsity = self.sorting_analyzer.sparsity
        elif self.sorting_analyzer.is_sparse() and self.params["sparsity"] is not None:
            sparsity = self.params["sparsity"]
            # assert provided sparsity is sparser than the one in the waveform extractor
            waveform_sparsity = self.sorting_analyzer.sparsity
            assert np.all(
                np.sum(waveform_sparsity.mask, 1) - np.sum(sparsity.mask, 1) > 0
            ), "The provided sparsity needs to be sparser than the one in the waveform extractor!"
        elif not self.sorting_analyzer.is_sparse() and self.params["sparsity"] is not None:
            sparsity = self.params["sparsity"]
        else:
            if self.params["max_dense_channels"] is not None:
                assert recording.get_num_channels() <= self.params["max_dense_channels"], (
                    "Sparsity must be provided when the number of channels is "
                    f"greater than {self.params['max_dense_channels']}. Alternatively, set max_dense_channels to None "
                    "to compute amplitude scalings using dense waveforms."
                )
            sparsity = ChannelSparsity.create_dense(self.sorting_analyzer)
        return sparsity

    def _get_pipeline_nodes(self):

        recording = self.sorting_analyzer.recording
        sorting = self.sorting_analyzer.sorting

        return_in_uV = self.sorting_analyzer.return_in_uV

        all_templates = get_dense_templates_array(self.sorting_analyzer, return_in_uV=return_in_uV)
        nbefore = _get_nbefore(self.sorting_analyzer)
        nafter = all_templates.shape[1] - nbefore
        templates_ext = self.sorting_analyzer.get_extension("templates")

        # if ms_before / ms_after are set in params then the original templates are shorten
        if self.params["ms_before"] is not None:
            cut_out_before = ms_to_samples(self.params["ms_before"], self.sorting_analyzer.sampling_frequency)
            assert (
                cut_out_before <= nbefore
            ), f"`ms_before` must be smaller than `ms_before` used in ComputeTemplates: {nbefore}"
        else:
            cut_out_before = nbefore

        if self.params["ms_after"] is not None:
            cut_out_after = ms_to_samples(self.params["ms_after"], self.sorting_analyzer.sampling_frequency)
            assert (
                cut_out_after <= nafter
            ), f"`ms_after` must be smaller than `ms_after` used in templates: {templates_ext.params['ms_after']}"
        else:
            cut_out_after = nafter

        extremum_channels_indices = self.sorting_analyzer.get_main_channels(outputs="index", with_dict=True)

        # collisions
        handle_collisions = self.params["handle_collisions"]
        delta_collision_ms = self.params["delta_collision_ms"]
        delta_collision_samples = int(delta_collision_ms / 1000 * self.sorting_analyzer.sampling_frequency)

        sparsity_mask = self._get_sparsity().mask

        spike_retriever_node = SpikeRetriever(
            sorting,
            recording,
            channel_from_template=True,
            include_spikes_in_margin=True,
        )
        amplitude_scalings_node = AmplitudeScalingNode(
            recording,
            parents=[spike_retriever_node],
            return_output=True,
            all_templates=all_templates,
            sparsity_mask=sparsity_mask,
            nbefore=nbefore,
            nafter=nafter,
            cut_out_before=cut_out_before,
            cut_out_after=cut_out_after,
            return_in_uV=return_in_uV,
            handle_collisions=handle_collisions,
            delta_collision_samples=delta_collision_samples,
        )
        nodes = [spike_retriever_node, amplitude_scalings_node]
        return nodes


register_result_extension(ComputeAmplitudeScalings)
compute_amplitude_scalings = ComputeAmplitudeScalings.function_factory()


class AmplitudeScalingNode(PipelineNode):
    def __init__(
        self,
        recording,
        parents,
        return_output,
        all_templates,
        sparsity_mask,
        nbefore,
        nafter,
        cut_out_before,
        cut_out_after,
        return_in_uV,
        handle_collisions,
        delta_collision_samples,
    ):
        PipelineNode.__init__(self, recording, parents=parents, return_output=return_output)
        self.return_in_uV = return_in_uV
        if return_in_uV and recording.has_scaleable_traces():
            self._dtype = np.float32
            self._gains = recording.get_channel_gains()
            self._offsets = recording.get_channel_gains()
        else:
            self._dtype = recording.get_dtype()
            self._gains = None
            self._offsets = None
        spike_retriever = find_parent_of_type(parents, SpikeRetriever)
        assert isinstance(
            spike_retriever, SpikeRetriever
        ), "SpikeAmplitudeNode needs a single SpikeRetriever as a parent"
        assert spike_retriever.include_spikes_in_margin, "Need SpikeRetriever with include_spikes_in_margin=True"
        if not handle_collisions:
            self._margin = max(nbefore, nafter)
        else:
            # in this case we extend the margin to be able to get with collisions outside the chunk
            margin_waveforms = max(nbefore, nafter)
            max_margin_collisions = delta_collision_samples + margin_waveforms
            self._margin = max_margin_collisions

        # for some edge cases a template can be zero, leading to problems later
        template_is_zero = [np.all(template == 0) for template in all_templates]

        self._all_templates = all_templates
        self._sparsity_mask = sparsity_mask
        self._nbefore = nbefore
        self._nafter = nafter
        self._cut_out_before = cut_out_before
        self._cut_out_after = cut_out_after
        self._handle_collisions = handle_collisions
        self._delta_collision_samples = delta_collision_samples
        self._template_is_zero = template_is_zero

        self._kwargs.update(
            all_templates=all_templates,
            sparsity_mask=sparsity_mask,
            nbefore=nbefore,
            nafter=nafter,
            cut_out_before=cut_out_before,
            cut_out_after=cut_out_after,
            return_in_uV=return_in_uV,
            handle_collisions=handle_collisions,
            delta_collision_samples=delta_collision_samples,
            template_is_zero=template_is_zero,
        )

    def get_dtype(self):
        return self._dtype

    def compute(self, traces, peaks):
        from scipy.stats import linregress

        # scale traces with margin to match scaling of templates
        if self._gains is not None:
            traces = traces.astype("float32") * self._gains + self._offsets

        all_templates = self._all_templates
        sparsity_mask = self._sparsity_mask
        nbefore = self._nbefore
        cut_out_before = self._cut_out_before
        cut_out_after = self._cut_out_after
        handle_collisions = self._handle_collisions
        delta_collision_samples = self._delta_collision_samples
        template_is_zero = self._template_is_zero

        # local_spikes_within_margin = peaks
        # i0 = np.searchsorted(local_spikes_within_margin["sample_index"], left_margin)
        # i1 = np.searchsorted(local_spikes_within_margin["sample_index"], traces.shape[0] - right_margin)
        # local_spikes = local_spikes_within_margin[i0:i1]

        local_spikes_within_margin = peaks
        local_spikes = local_spikes_within_margin[~peaks["in_margin"]]

        # set colliding spikes apart (if needed)
        if handle_collisions:
            # local spikes with margin!
            collisions = find_collisions(
                local_spikes, local_spikes_within_margin, delta_collision_samples, sparsity_mask
            )
        else:
            collisions = {}

        # compute the scaling for each spike
        scalings = np.zeros(len(local_spikes), dtype=float)
        spike_collision_mask = np.zeros(len(local_spikes), dtype=bool)

        for spike_index, spike in enumerate(local_spikes):
            if spike_index in collisions.keys():
                # we deal with overlapping spikes later
                continue

            unit_index = spike["unit_index"]

            if template_is_zero[unit_index]:
                # if template is zero, linregress will fail so we intervene
                scalings[spike_index] = 0
                continue

            sample_centered = spike["sample_index"]
            (sparse_indices,) = np.nonzero(sparsity_mask[unit_index])
            template = all_templates[unit_index][:, sparse_indices]
            template = template[nbefore - cut_out_before : nbefore + cut_out_after]
            cut_out_start = sample_centered - cut_out_before
            cut_out_end = sample_centered + cut_out_after
            if sample_centered - cut_out_before < 0:
                local_waveform = traces[:cut_out_end, sparse_indices]
                template = template[cut_out_before - sample_centered :]
            elif sample_centered + cut_out_after > traces.shape[0]:
                local_waveform = traces[cut_out_start:, sparse_indices]
                template = template[: -(sample_centered + cut_out_after - (traces.shape[0]))]
            else:
                local_waveform = traces[cut_out_start:cut_out_end, sparse_indices]
            assert template.shape == local_waveform.shape

            # here we use linregress, which is equivalent to using sklearn LinearRegression with fit_intercept=True
            # y = local_waveform.flatten()
            # X = template.flatten()[:, np.newaxis]
            # reg = LinearRegression(positive=True, fit_intercept=True).fit(X, y)
            # scalings[spike_index] = reg.coef_[0]

            # closed form: W = (X' * X)^-1 X' y
            # y = local_waveform.flatten()[:, None]
            # X = np.ones((len(y), 2))
            # X[:, 0] = template.flatten()
            # W = np.linalg.inv(X.T @ X) @ X.T @ y
            # scalings[spike_index] = W[0, 0]

            linregress_res = linregress(template.flatten(), local_waveform.flatten())
            scalings[spike_index] = linregress_res[0]

        # deal with collisions
        if len(collisions) > 0:
            for spike_index, collision in collisions.items():
                scaled_amps = fit_collision(
                    collision,
                    traces,
                    nbefore,
                    all_templates,
                    sparsity_mask,
                    cut_out_before,
                    cut_out_after,
                )
                # the scaling for the current spike is at index 0
                scalings[spike_index] = scaled_amps[0]
                spike_collision_mask[spike_index] = True

        # TODO: switch to collision mask and return that (to use concatenation)
        return (scalings, spike_collision_mask)

    def get_margin(self):
        return self._margin


### Collision handling ###
def _are_units_spatially_overlapping(sparsity_mask, i, j):
    """
    Returns True if the unit indices i and j are
    spatially overlapping, False otherwise

    Parameters
    ----------
    sparsity_mask: boolean mask
        A num_units x num_channels boolean array indicating whether
        the unit is represented on the channel.
    i: int
        The first unit index
    j: int
        The second unit index

    Returns
    -------
    bool
        True if the units i and j are spatially overlapping, False otherwise
    """
    if np.any(sparsity_mask[i] & sparsity_mask[j]):
        return True
    else:
        return False


def find_collisions(spikes, spikes_within_margin, delta_collision_samples, sparsity_mask):
    """
    Finds the collisions between spikes.

    Given an array of spikes extracted from all units, find the 'spike collisions'
    - incidents where two spikes from different units overlap temporally and spatially.
    Temporal and spatial overlap are defined as:

    Temporal overlap: another spike peak occurring within a specified time window
                      around the spike peak.
    Spatial overlap: two spikes have signal on any shared channel (i.e.
                     two spikes are not spatially overlapping if their signal is spread
                     across two completely separate sets of channels).

    First for each spike, find all other spikes that temporally overlap the spike.
    Next, only these temporally overlapping spikes that also spatially overlap the
    spike are included in the output `collision_spikes_dict`.

    Parameters
    ----------
    spikes: np.array
        An array of spikes, where spikes are represented by their:
            (sample_index, channel_index, amplitude, segment_index, unit_index, in_margin)
    spikes_within_margin: np.array
        An array of spikes, of the same format as `spikes`, whose peaks are close to
        another spike within a given margin
    delta_collision_samples: int
        The maximum number of samples between two spikes to consider them as overlapping
    sparsity_mask: boolean mask
        A num_units x num_channels boolean array indicating whether
        the unit is represented on the channel.

    Returns
    -------
    collision_spikes_dict: dict
        A dictionary with collisions. The key is the index of the spike with collision, the value is an
        array of overlapping spikes, including the spike itself at position 0.
    """
    # TODO: refactor to speed-up
    collision_spikes_dict = {}
    for spike_index, spike in enumerate(spikes):

        # find the index of the spike within spikes_within_margin
        spike_index_within_margin = np.where(spikes_within_margin == spike)[0][0]

        # find the spikes that fall within a temporal window around the spike peak
        spike_collision_window = [
            spike["sample_index"] - delta_collision_samples,
            spike["sample_index"] + delta_collision_samples,
        ]

        consecutive_window_pre, consecutive_window_post = np.searchsorted(
            spikes_within_margin["sample_index"],
            spike_collision_window,
        )

        # Make an array of indices of all spikes that collide with the spike,
        # making sure to exlude the spike itself (it is included in the collision_spikes by construction)
        # The indices here are indices of the spike position in `spikes_within_margin`.
        pre_possible_consecutive_spike_indices = np.arange(consecutive_window_pre, spike_index_within_margin)
        post_possible_consecutive_spike_indices = np.arange(spike_index_within_margin + 1, consecutive_window_post)
        possible_overlapping_spike_indices = np.concatenate(
            (pre_possible_consecutive_spike_indices, post_possible_consecutive_spike_indices)
        )

        # Build the collusion_spikes_dict including only
        # spikes that overlap spatially
        for possible_overlapping_spike_index in possible_overlapping_spike_indices:

            if _are_units_spatially_overlapping(
                sparsity_mask,
                spike["unit_index"],
                spikes_within_margin[possible_overlapping_spike_index]["unit_index"],
            ):
                if spike_index not in collision_spikes_dict:
                    collision_spikes_dict[spike_index] = np.array([spike])
                collision_spikes_dict[spike_index] = np.concatenate(
                    (collision_spikes_dict[spike_index], [spikes_within_margin[possible_overlapping_spike_index]])
                )
    return collision_spikes_dict


def find_collision_indices(spikes, spike_index, delta_collision_samples, sparsity_mask):
    """
    Find the spikes colliding with the spike at `spike_index`.

    This applies the same temporal and spatial overlap criteria as :py:func:`find_collisions`,
    but for a single spike and on a full (concatenated) spike vector, so that collisions can be
    recovered after the amplitude scalings have been computed (e.g. for plotting).

    Parameters
    ----------
    spikes : np.array
        The concatenated spike vector, with fields (sample_index, unit_index, segment_index).
    spike_index : int
        The index in `spikes` of the spike of interest.
    delta_collision_samples : int
        The maximum number of samples between two spikes to consider them as overlapping.
    sparsity_mask : boolean mask
        A num_units x num_channels boolean array indicating whether
        the unit is represented on the channel.

    Returns
    -------
    collision_indices : np.array
        The indices in `spikes` of the colliding spikes, with `spike_index` at position 0.
        If the spike has no collision, only `spike_index` is returned.
    """
    spike = spikes[spike_index]

    # collisions cannot span segments, so restrict the search to the segment of the spike of interest
    segment_start, segment_end = np.searchsorted(
        spikes["segment_index"], [spike["segment_index"], spike["segment_index"] + 1]
    )
    segment_spikes = spikes[segment_start:segment_end]

    # find the spikes that fall within a temporal window around the spike peak
    i0, i1 = np.searchsorted(
        segment_spikes["sample_index"],
        [spike["sample_index"] - delta_collision_samples, spike["sample_index"] + delta_collision_samples],
    )
    candidate_indices = np.arange(segment_start + i0, segment_start + i1)
    # exclude the spike itself, which is added back at position 0
    candidate_indices = candidate_indices[candidate_indices != spike_index]

    # keep only the spikes that also overlap spatially
    spatially_overlapping = np.array(
        [
            _are_units_spatially_overlapping(sparsity_mask, spike["unit_index"], spikes[i]["unit_index"])
            for i in candidate_indices
        ],
        dtype=bool,
    )
    return np.concatenate(([spike_index], candidate_indices[spatially_overlapping])).astype(int)


def fit_collision(
    collision,
    traces_with_margin,
    nbefore,
    all_templates,
    sparsity_mask,
    cut_out_before,
    cut_out_after,
):
    """
    Compute the best fit for a collision between a spike and its overlapping spikes.

    The problem addressed here is to compute the scaling factor of a waveform
    to its unit template in the presence of temporally and spatially
    colliding spikes (i.e. spikes that occur close to a given spike). When
    the waveform of a spike overlaps with other, colliding spikes, these
    colliding spikes will contribute to the spike amplitude.

    This is addressed by fitting a multivariate regression. `y` is
    the observed waveform (including the spike of interest and colliding spikes).
    `X` is the corresponding set of unit templates, with each template
    temporally shifted to match the position of its associated spike in the
    original waveform. The regression coefficients represent the scaling
    factors applied to each template to best match the waveform.

    Parameters
    ----------
    collision: np.ndarray
        A numpy array of shape (n_colliding_spikes, ) containing a set of colliding spikes.
        The first position is the spike of interest, other entries are spikes which
        collide with the spike in the first position.
        Each spike is an array with entries:
            (sample_index, channel_index, amplitude, segment_index, unit_index, in_margin)
    traces_with_margin: np.ndarray
        A numpy array of shape (n_samples, n_channels) containing the traces with a margin.
    nbefore: int
        The number of samples before the spike to consider for the fit.
    all_templates: np.ndarray
        A numpy array of shape (n_units, n_samples, n_channels) containing the templates.
    sparsity_mask: boolean mask
        A num_units x num_channels boolean array indicating whether
        the unit is represented on the channel.
    cut_out_before: int
        The number of samples to cut out before the spike.
    cut_out_after: int
        The number of samples to cut out after the spike.

    Returns
    -------
    np.ndarray
        The fitted scaling factors for the colliding spikes.
    """
    from sklearn.linear_model import LinearRegression

    # Find the first and last spike peak index
    # from the set of colliding spikes.
    sample_first_centered = np.min(collision["sample_index"])
    sample_last_centered = np.max(collision["sample_index"])

    # Find channels that have signal from any of the set of
    # colliding spikes. This is found as the union between
    # all channels with sparsity mask `True` for any
    # unit represented in the set of colliding spikes.
    common_sparse_mask = np.zeros(sparsity_mask.shape[1], dtype="int")
    for spike in collision:
        mask_i = sparsity_mask[spike["unit_index"]]
        common_sparse_mask = np.logical_or(common_sparse_mask, mask_i)
    (sparse_indices,) = np.nonzero(common_sparse_mask)

    # Index out the temporal window that includes all colliding spikes
    # across all channels which contain signal from a colliding spike.
    local_waveform_start = max(0, sample_first_centered - cut_out_before)
    local_waveform_end = min(traces_with_margin.shape[0], sample_last_centered + cut_out_after)
    local_waveform = traces_with_margin[local_waveform_start:local_waveform_end, sparse_indices]
    num_samples_local_waveform = local_waveform.shape[0]

    y = local_waveform.T.flatten()
    X = np.zeros((len(y), len(collision)))
    for i, spike in enumerate(collision):

        full_template = np.zeros_like(local_waveform)

        # For the collision spike, take its unit template and insert
        # it into `full_template` at the time the collision spike occurred.
        sample_centered = spike["sample_index"] - local_waveform_start
        template = all_templates[spike["unit_index"]][:, sparse_indices]
        template_cut = template[nbefore - cut_out_before : nbefore + cut_out_after]

        # Deal with borders - if the unit template goes off the start / end
        # of the full template, clip it.
        if sample_centered - cut_out_before < 0:
            full_template[: sample_centered + cut_out_after] = template_cut[cut_out_before - sample_centered :]
        elif sample_centered + cut_out_after > num_samples_local_waveform:
            full_template[sample_centered - cut_out_before :] = template_cut[
                : -(cut_out_after + sample_centered - num_samples_local_waveform)
            ]
        else:
            full_template[sample_centered - cut_out_before : sample_centered + cut_out_after] = template_cut

        X[:, i] = full_template.T.flatten()

    reg = LinearRegression(fit_intercept=True, positive=True).fit(X, y)
    scalings = reg.coef_
    return scalings


### Debugging ###
def _plot_collisions(sorting_analyzer, sparsity=None, num_collisions=None):
    """
    Plot the fitting of collision spikes for debugging.
    ----------

    Parameters
    sorting_analyzer : SortingAnalyzer
        The SortingAnalyzer object.
    sparsity : ChannelSparsity, default=None
        The ChannelSparsity. If None, only main channels are plotted.
    num_collisions : int, default=None
        Number of collisions to plot. If None, all collisions are plotted.

    Returns
    -------
    axes : list of matplotlib.axes.Axes
        One axis per plotted collision.
    """
    assert sorting_analyzer.has_extension("amplitude_scalings"), "Could not find amplitude scalings extension!"
    sac = sorting_analyzer.get_extension("amplitude_scalings")
    handle_collisions = sac.params["handle_collisions"]
    assert handle_collisions, "Amplitude scalings was run without handling collisions!"
    scalings = sac.get_data(return_data_name="amplitude_scalings")
    collision_mask = sac.get_data(return_data_name="collision_mask")

    spikes = sorting_analyzer.sorting.to_spike_vector()

    # the collision mask only flags which spikes were fitted jointly, so the colliding
    # spikes themselves are recovered with the same criteria used at compute time
    sparsity_mask = sac._get_sparsity().mask
    delta_collision_samples = int(
        sac.params["delta_collision_ms"] / 1000 * sorting_analyzer.sampling_frequency,
    )

    collision_spike_indices = np.flatnonzero(collision_mask)
    if num_collisions is not None:
        collision_spike_indices = collision_spike_indices[:num_collisions]

    axes = []
    for spike_index in collision_spike_indices:
        collision_indices = find_collision_indices(spikes, spike_index, delta_collision_samples, sparsity_mask)
        ax = _plot_one_collision(
            sorting_analyzer, spike_index, collision_indices, spikes, scalings=scalings, sparsity=sparsity
        )
        axes.append(ax)
    return axes


def _plot_one_collision(
    sorting_analyzer,
    spike_index,
    collision_indices,
    spikes,
    scalings=None,
    sparsity=None,
    cut_out_samples=100,
    ax=None,
):
    """
    Internal method for debugging collisions.

    Parameters
    ----------
    sorting_analyzer : SortingAnalyzer
        The SortingAnalyzer object.
    spike_index : int
        The index in `spikes` of the spike of interest.
    collision_indices : np.array
        The indices in `spikes` of the colliding spikes, with `spike_index` at position 0.
    spikes : np.array
        The concatenated spike vector.
    scalings : np.array or None, default=None
        The amplitude scalings, with the same length as `spikes`. If given, the scaled
        templates and their sum (the fit) are plotted on top of the traces.
    sparsity : ChannelSparsity or None, default=None
        The ChannelSparsity. If None, only main channels are plotted.
    cut_out_samples : int, default=100
        Number of samples to plot before and after the colliding spikes.
    ax : matplotlib.axes.Axes or None, default=None
        The axis to plot on. If None, a new figure is created.
    """
    import matplotlib.pyplot as plt

    if ax is None:
        fig, ax = plt.subplots()

    recording = sorting_analyzer.recording
    all_templates = get_dense_templates_array(sorting_analyzer, return_in_uV=True)
    nbefore = _get_nbefore(sorting_analyzer)
    nafter = all_templates.shape[1] - nbefore
    cut_out_samples = max(cut_out_samples, max(nbefore, nafter))

    overlapping_spikes = spikes[collision_indices]

    sparse_indices = np.array([], dtype="int")
    if sparsity is not None:
        unit_inds_to_channel_indices = sparsity.unit_id_to_channel_indices
        for spike in overlapping_spikes:
            sparse_indices_i = unit_inds_to_channel_indices[sorting_analyzer.unit_ids[spike["unit_index"]]]
            sparse_indices = np.union1d(sparse_indices, sparse_indices_i)
    else:
        # the spike vector has no channel information, so fall back on the main channel of each unit
        main_channel_indices = sorting_analyzer.get_main_channels(outputs="index", with_dict=True)
        for spike in overlapping_spikes:
            main_channel_index = main_channel_indices[sorting_analyzer.unit_ids[spike["unit_index"]]]
            sparse_indices = np.union1d(sparse_indices, [main_channel_index])

    channel_ids = recording.channel_ids[sparse_indices]

    center_spike = overlapping_spikes[0]
    segment_index = center_spike["segment_index"]
    max_delta = np.max(np.abs(overlapping_spikes["sample_index"].astype("int64") - center_spike["sample_index"]))
    sf = max(0, center_spike["sample_index"] - max_delta - cut_out_samples)
    ef = min(
        center_spike["sample_index"] + max_delta + cut_out_samples,
        recording.get_num_samples(segment_index=segment_index),
    )
    tr_overlap = recording.get_traces(
        segment_index=segment_index, start_frame=sf, end_frame=ef, channel_ids=channel_ids, return_in_uV=True
    )
    ts = np.arange(sf, ef) / recording.sampling_frequency * 1000
    max_tr = np.max(np.abs(tr_overlap))

    for ch, tr in enumerate(tr_overlap.T):
        _ = ax.plot(ts, tr + 1.2 * ch * max_tr, color="k")
        ax.text(ts[0], 1.2 * ch * max_tr - 0.3 * max_tr, f"Ch:{channel_ids[ch]}")

    used_labels = []
    for i, spike in enumerate(overlapping_spikes):
        label = f"U{spike['unit_index']}"
        if label in used_labels:
            label = None
        else:
            used_labels.append(label)
        ax.axvline(
            spike["sample_index"] / recording.sampling_frequency * 1000, color=f"C{spike['unit_index']}", label=label
        )

    if scalings is not None:
        fitted_traces = np.zeros_like(tr_overlap)

        for spike, spike_scaling in zip(overlapping_spikes, scalings[collision_indices]):
            template = all_templates[spike["unit_index"]]
            template_scaled = spike_scaling * template
            template_scaled_sparse = template_scaled[:, sparse_indices]
            sample_start = spike["sample_index"] - nbefore
            sample_end = sample_start + template_scaled_sparse.shape[0]

            # the template can stick out of the plotted window, so clip it on both sides
            start_in_window = max(sample_start - sf, 0)
            end_in_window = min(sample_end - sf, fitted_traces.shape[0])
            fitted_traces[start_in_window:end_in_window] += template_scaled_sparse[
                start_in_window - (sample_start - sf) : end_in_window - (sample_start - sf)
            ]

            for ch, temp in enumerate(template_scaled_sparse.T):
                ts_template = np.arange(sample_start, sample_end) / recording.sampling_frequency * 1000
                _ = ax.plot(ts_template, temp + 1.2 * ch * max_tr, color=f"C{spike['unit_index']}", ls="--")

        for ch, tr in enumerate(fitted_traces.T):
            _ = ax.plot(ts, tr + 1.2 * ch * max_tr, color="gray", alpha=0.7)

        fitted_line = ax.get_lines()[-1]
        fitted_line.set_label("Fitted")

    ax.legend()
    ax.set_title(f"Spike {spike_index} - sample {center_spike['sample_index']}")
    return ax

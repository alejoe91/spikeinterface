from __future__ import annotations
import numpy as np

from .main import BaseMergingEngine
from spikeinterface.core.sortinganalyzer import create_sorting_analyzer
from spikeinterface.core.analyzer_extension_core import ComputeTemplates
from spikeinterface.curation.auto_merge import get_potential_auto_merge
from spikeinterface.curation.curation_tools import resolve_merging_graph
from spikeinterface.core.sorting_tools import apply_merges_to_sorting


class CircusMerging(BaseMergingEngine):
    """
    Meta merging inspired from the Lussac metric
    """

    default_params = {
        "templates": None,
        "verbose": True,
        "remove_emtpy": True,
        "recursive": True,
        "censor_ms": 3,
        "similarity_kwargs": {"method": "l2", "support": "union", "max_lag_ms": 0.2},
        "curation_kwargs": {
            "minimum_spikes": 50,
            "corr_diff_thresh": 0.5,
            "maximum_distance_um": 50,
            "presence_distance_thresh": 100,
            "template_diff_thresh": 0.5,
        },
        "temporal_splits_kwargs": {
            "minimum_spikes": 50,
            "maximum_distance_um": 50,
            "presence_distance_thresh": 100,
            "template_diff_thresh": 0.5,
        },
    }

    def __init__(self, recording, sorting, kwargs):
        self.params = self.default_params.copy()
        self.params.update(**kwargs)
        self.sorting = sorting
        self.recording = recording
        self.remove_empty = self.params.get("remove_empty", True)
        self.verbose = self.params.pop("verbose")
        self.templates = self.params.pop("templates", None)
        self.recursive = self.params.pop("recursive", True)

        if self.templates is not None:
            sparsity = self.templates.sparsity
            templates_array = self.templates.get_dense_templates().copy()
            self.analyzer = create_sorting_analyzer(sorting, recording, format="memory", sparsity=sparsity)
            self.analyzer.extensions["templates"] = ComputeTemplates(self.analyzer)
            self.analyzer.extensions["templates"].params = {"ms_before": self.templates.ms_before,
                                                            "ms_after": self.templates.ms_after}
            self.analyzer.extensions["templates"].data["average"] = templates_array
            self.analyzer.compute("unit_locations", method="grid_convolution")
        else:
            self.analyzer = create_sorting_analyzer(sorting, recording, format="memory")
            self.analyzer.compute(["random_spikes", "templates"])
            self.analyzer.compute("unit_locations", method="grid_convolution")

        if self.remove_empty:
            from spikeinterface.curation.curation_tools import remove_empty_units

            self.analyzer = remove_empty_units(self.analyzer)

        self.analyzer.compute("template_similarity", **self.params["similarity_kwargs"])

    def _get_new_sorting(self):
        curation_kwargs = self.params.get("curation_kwargs", None)
        if curation_kwargs is not None:
            merges = get_potential_auto_merge(self.analyzer, **curation_kwargs)
        else:
            merges = []
        if self.verbose:
            print(f"{len(merges)} merges have been detected via auto merges")
        temporal_splits_kwargs = self.params.get("temporal_splits_kwargs", None)
        if temporal_splits_kwargs is not None:
            more_merges = get_potential_auto_merge(self.analyzer, **temporal_splits_kwargs, preset="temporal_splits")
            if self.verbose:
                print(f"{len(more_merges)} merges have been detected via additional temporal splits")
            merges += more_merges
        units_to_merge = resolve_merging_graph(self.analyzer.sorting, merges)
        new_analyzer = self.analyzer.merge_units(
            units_to_merge, mode='soft', sparsity_overlap=0.25, censor_ms=self.params["censor_ms"]
        )
        return new_analyzer, merges

    def run(self, extra_outputs=False):
        self.analyzer, merges = self._get_new_sorting()
        num_merges = len(merges)
        all_merges = [merges]

        if self.recursive:
            while num_merges > 0:
                self.analyzer, merges = self._get_new_sorting()
                num_merges = len(merges)
                all_merges += [merges]

        if extra_outputs:
            return self.analyzer.sorting, all_merges
        else:
            return self.analyzer.sorting

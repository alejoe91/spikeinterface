import numpy as np
from pathlib import Path

from .neobaseextractor import NeoBaseRecordingExtractor, NeoBaseSortingExtractor

import neo

class KiloSortSortingExtractor(NeoBaseSortingExtractor):
    """
    Class for reading the sorting from kilosort folder
    
    Based on neo.rawio.PhyRawIO
    """
    mode = 'folder'
    NeoRawIOClass = 'PhyRawIO'
    handle_spike_frame_directly = True
    
    def __init__(self, folder_path, keep_good_only=False, use_natural_unit_ids=True):
        
        neo_kwargs = {'dirname' : folder_path}
        
        # sample rate is in "params.py"
        d = {}
        with open(Path(folder_path) / 'params.py') as f:
            exec(f.read(), None, d)
        sampling_frequency = float(d['sample_rate'])
        
        if keep_good_only:
            raise NotImplementedError
            # TODO must implement annotations in neo to retreive this.

        NeoBaseSortingExtractor.__init__(self, 
                    sampling_frequency=sampling_frequency, # auto guess is correct here
                    use_natural_unit_ids=use_natural_unit_ids,
                    **neo_kwargs)

        self._kwargs = dict(folder_path=folder_path, keep_good_only=keep_good_only,
            use_natural_unit_ids=use_natural_unit_ids)


def read_kilosort(*args, **kargs):
    sorting = KiloSortSortingExtractor(*args, **kargs)
    return sorting
read_kilosort.__doc__ = KiloSortSortingExtractor.__doc__

import numpy as np

from .neobaseextractor import NeoBaseRecordingExtractor, NeoBaseSortingExtractor

import neo

import probeinterface as pi

class MEArecRecordingExtractor(NeoBaseRecordingExtractor):
    """
    Class for reading data from a MEArec simulated data.
    
    
    Parameters
    ----------
    file_path: str
    
    locs_2d: bool
    
    """    
    mode = 'file'
    NeoRawIOClass = 'MEArecRawIO'
    
    def __init__(self, file_path, locs_2d=True):
        
        neo_kwargs = {'filename' : file_path}
        NeoBaseRecordingExtractor.__init__(self, **neo_kwargs)
        
        probe = pi.read_mearec(file_path)
        self.set_probe(probe, in_place=True)
        self.annotate(is_filtered=True)
        
        self._kwargs = {'file_path' : str(file_path), 'locs_2d': locs_2d}
        
    

class MEArecSortingExtractor(NeoBaseSortingExtractor):
    mode = 'file'
    NeoRawIOClass = 'MEArecRawIO'
    handle_spike_frame_directly = False
    
    def __init__(self, file_path, use_natural_unit_ids=True):
        neo_kwargs = {'filename' : file_path}
        NeoBaseSortingExtractor.__init__(self, 
                    sampling_frequency=None, # auto guess is correct here
                    use_natural_unit_ids=use_natural_unit_ids,
                    **neo_kwargs)
        
        self._kwargs = {'file_path' : str(file_path), 'use_natural_unit_ids': use_natural_unit_ids}


def read_mearec(file_path, locs_2d=True, use_natural_unit_ids=True):
    recording = MEArecRecordingExtractor(file_path, locs_2d=locs_2d)
    sorting = MEArecSortingExtractor(file_path, use_natural_unit_ids=use_natural_unit_ids)
    return recording, sorting

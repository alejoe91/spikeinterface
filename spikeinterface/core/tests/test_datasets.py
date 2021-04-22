import shutil
from pathlib import Path
import pytest
import numpy as np


from spikeinterface.core import download_dataset




def test_download_dataset():
    repo = 'https://gin.g-node.org/NeuralEnsemble/ephy_testing_data'
    remote_path='mearec'
    
    # local_folder automatic
    local_path = download_dataset(repo=repo, remote_path=remote_path, local_folder=None)
    # print(local_path)
    
    
    
    
    
if __name__ == '__main__':
    test_download_dataset()
    

    
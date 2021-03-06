'''
SortingExtractor Widgets Gallery
===================================

Here is a gallery of all the available widgets using SortingExtractor objects.
'''
import matplotlib.pyplot as plt

import spikeinterface.extractors as se
import spikeinterface.widgets as sw

##############################################################################
# First, let's create a toy example with the `extractors` module:

recording, sorting = se.toy_example(duration=10, num_channels=4, seed=0, num_segments=1)

##############################################################################
# plot_rasters()
# ~~~~~~~~~~~~~~~~~

w_rs = sw.plot_rasters(sorting)

##############################################################################
# plot_isi_distribution()
# ~~~~~~~~~~~~~~~~~~~~~~~~

#TODO : @alessio: this is for you
#w_isi = sw.plot_isi_distribution(sorting, bins=10, window=1)

##############################################################################
# plot_autocorrelograms()
# ~~~~~~~~~~~~~~~~~~~~~~~~

#TODO : @alessio: this is for you
# w_ach = sw.plot_autocorrelograms(sorting, bin_size=1, window=10, unit_ids=[1, 2, 4, 5, 8, 10, 7])

##############################################################################
# plot_crosscorrelograms()
# ~~~~~~~~~~~~~~~~~~~~~~~~

#TODO : @alessio: this is for you
# w_cch = sw.plot_crosscorrelograms(sorting, unit_ids=[1, 5, 8], bin_size=0.1, window=5)

plt.show()
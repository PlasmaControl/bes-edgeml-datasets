# `bes_data_tools` package

`bes_data_tools` is a package for fetching BES metadata and signals,
storing the data in HDF5 files, and visualizing results.

`bes_data.py` defines a class that fetches and stores BES metadata and signals.

`package_h5.py` defines functions that packages BES metadata and signals
in HDF5 files.  Also, it can analyze metadata to determine BES data
with an 8x8 spatial configuration.

`plots.py` contains utility plotting functions.

By default, data is read from and written to `data/`, and figures are saved
to `figures/`.  These directories are created locally wherever the package or 
modules are imported.
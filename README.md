# BES edge-ml

Code base for BES edge-ML at DIII-D

- `bes_data_tools/` - package for working with BES metadata and signals
- `elm/` - work area for ELM prediction
- `qhmode/` - work area for QH-mode classification
- `ae/` - work area for AE classification

In all areas, data (CSV, HDF5 files) are stored in `data/`, and plot figures
are stored in `figures/`.


## Tutorial

To learn core functionality in `bes_data_tools` package, do

```bash
cd bes_data_tools
python bes_data.py  # class for BES metadata and signals
python package_h5.py   # package data in HDF5 files
python plots.py  # plotting utilities
```
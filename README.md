# BES edge-ml

Code base for BES edge-ML at DIII-D

- `ae_db.py` - Read B. Heidbrink's AE IDL database file, and convert to python dictionary
- `package_bes_data.py` - Package all BES signals for a single DIII-D shot into a single HDF5 file, and package BES
metadata into a single HDF5 file with groups corresponding to shots
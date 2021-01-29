# BES edge-ml

Code base for BES edge-ML at DIII-D

- `ae_db.py` - Read B. Heidbrink's AE IDL database file, convert to python dictionary, 
and save as python pickle file
- `bes2hdf5.py` - Package all BES signals for a single DIII-D shot into a single 
HDF5 file, and package BES metadata into a single HDF5 file with groups corresponding 
to shots.  The primary function for packaging BES data is `package_bes()`.  The function 
`traverse_h5py()` is a utility function to print the summarized contents of an hdf5 file.
`BES_Data` is a class that represents BES signals and metadata for a single shot.
- `package_bes.sh` - Slurm script to run `bes2hdf5.py` on Iris compute node
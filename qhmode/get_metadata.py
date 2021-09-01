
from bes_data_tools import bes2hdf5

max_shots = None

bes2hdf5.package_bes(
    shotlist_csvfile='lh_shotlist.csv',
    max_shots=max_shots,
    output_h5file='lh_metadata.hdf5',
    verbose=True,
    with_signals=False,
)

bes2hdf5.package_bes(
    shotlist_csvfile='qh_shotlist.csv',
    max_shots=max_shots,
    output_h5file='qh_metadata.hdf5',
    verbose=True,
    with_signals=False,
)
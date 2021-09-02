from bes_data_tools.bes2hdf5 import package_bes

package_bes(
    shotlist_csvfile='lh_shotlist.csv',
    max_shots=None,
    output_h5file='lh_metadata.hdf5',
    verbose=True,
    with_signals=False,
)

package_bes(
    shotlist_csvfile='qh_shotlist.csv',
    max_shots=None,
    output_h5file='qh_metadata.hdf5',
    verbose=True,
    with_signals=False,
)